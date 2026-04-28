import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import rdkit

from model import NormalizedPropertyPredictor, Encoder
from smiles_sampling import sample_smiles
from tokenizer import PAD_IDX, SMILESTokenizer


contrib_path = os.path.join(rdkit.__path__[0], "Contrib")
if contrib_path not in sys.path:
    sys.path.insert(0, contrib_path)
from SA_Score import sascorer


CSV_PATH = "ZINC20-Druglike/zinc-druglike-cano.csv"
VOCAB_PATH = "vocab.json"
CKPT_DIR = os.environ.get("CKPT_DIR", "checkpoints")
PREDICTOR_DIR = os.environ.get("PREDICTOR_DIR", os.path.join(CKPT_DIR, "predictors"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

MAX_ROWS = int(os.environ.get("MAX_ROWS", 50_000))
SAMPLE_POOL_ROWS = int(os.environ.get("SAMPLE_POOL_ROWS", 0)) or None
SEED = int(os.environ.get("SEED", 42))
ENCODE_BATCH_SIZE = int(os.environ.get("ENCODE_BATCH_SIZE", 256))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 256))
EPOCHS = int(os.environ.get("EPOCHS", 100))
LR = float(os.environ.get("LR", 1e-3))
VAL_SPLIT = float(os.environ.get("VAL_SPLIT", 0.2))
HIDDEN_DIM = int(os.environ.get("PRED_HIDDEN_DIM", 256))

PROPERTIES = tuple(
    p.strip().lower()
    for p in os.environ.get("PROPERTIES", "qed,sa,mw").split(",")
    if p.strip()
)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_encoder_config(state_dict):
    vocab_size, embed_dim = state_dict["encoder.embedding.weight"].shape
    layer_dims = []
    i = 0
    while f"encoder.lstm_layers.{i}.weight_hh_l0" in state_dict:
        layer_dims.append(state_dict[f"encoder.lstm_layers.{i}.weight_hh_l0"].shape[1])
        i += 1
    latent_dim = state_dict["encoder.fc_mu.weight"].shape[0]
    return vocab_size, embed_dim, layer_dims, latent_dim


def load_encoder(device):
    ckpt_path = os.path.join(CKPT_DIR, "vae_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    vocab_size, embed_dim, layer_dims, latent_dim = infer_encoder_config(ckpt["vae"])
    encoder = Encoder(vocab_size, embed_dim, layer_dims, latent_dim).to(device)
    encoder_state = {
        key[len("encoder."):]: value
        for key, value in ckpt["vae"].items()
        if key.startswith("encoder.")
    }
    encoder.load_state_dict(encoder_state)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    print(f"[VAE] Frozen encoder from {ckpt_path} | embed={embed_dim} latent={latent_dim} layers={layer_dims}")
    return encoder, latent_dim, ckpt_path


def build_optional_scorers():
    scorers = {}
    if "scscore" in PROPERTIES:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scscore_repo"))
        from scscore.standalone_model_numpy import SCScorer

        scorer = SCScorer()
        scorer.restore(
            weight_path=os.path.join(
                "scscore_repo",
                "models",
                "full_reaxys_model_1024bool",
                "model.ckpt-10654.as_numpy.json.gz",
            )
        )
        scorers["scscore"] = scorer

    if "syba" in PROPERTIES:
        from syba.syba import SybaClassifier

        syba = SybaClassifier()
        syba.fitDefaultScore()
        scorers["syba"] = syba
    return scorers


def compute_targets(smiles):
    scorers = build_optional_scorers()
    valid_smiles, rows = [], []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        row = []
        for prop in PROPERTIES:
            if prop == "qed":
                row.append(float(QED.qed(mol)))
            elif prop == "sa":
                row.append(float(sascorer.calculateScore(mol)))
            elif prop == "mw":
                row.append(float(Descriptors.MolWt(mol)))
            elif prop == "scscore":
                _, score = scorers["scscore"].get_score_from_smi(smi)
                row.append(float(score))
            elif prop == "syba":
                row.append(float(scorers["syba"].predict(mol=mol)))
            else:
                raise ValueError(f"Unknown property '{prop}'")

        valid_smiles.append(smi)
        rows.append(row)
    return valid_smiles, torch.tensor(rows, dtype=torch.float32)


def encode_smiles(encoder, tokenizer, smiles, device):
    mus = []
    with torch.no_grad():
        for start in range(0, len(smiles), ENCODE_BATCH_SIZE):
            batch = smiles[start:start + ENCODE_BATCH_SIZE]
            seqs = [torch.tensor(tokenizer.encode(s), dtype=torch.long) for s in batch]
            lengths = torch.tensor([len(seq) for seq in seqs], dtype=torch.long)
            x = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=PAD_IDX).to(device)
            mu, _ = encoder(x, lengths.to(device))
            mus.append(mu.detach().cpu())
    return torch.cat(mus, dim=0)


def standardize_train_val(train, val):
    # Standardization: x_norm = (x - mean) / std.
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, keepdim=True).clamp_min(1e-8)
    return (train - mean) / std, (val - mean) / std, mean, std


def r2_score(y_true, y_pred):
    ss_res = (y_true - y_pred).pow(2).sum(dim=0)
    ss_tot = (y_true - y_true.mean(dim=0, keepdim=True)).pow(2).sum(dim=0).clamp_min(1e-12)
    return 1.0 - ss_res / ss_tot


def pearson(y_true, y_pred):
    yt = y_true - y_true.mean(dim=0, keepdim=True)
    yp = y_pred - y_pred.mean(dim=0, keepdim=True)
    denom = (yt.pow(2).sum(dim=0).sqrt() * yp.pow(2).sum(dim=0).sqrt()).clamp_min(1e-12)
    return (yt * yp).sum(dim=0) / denom


def save_plot(y_val, pred_val, metrics):
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Plot] skipped: {exc}")
        return

    n_props = len(PROPERTIES)
    fig, axes = plt.subplots(1, n_props, figsize=(5 * n_props, 4), squeeze=False)
    for i, prop in enumerate(PROPERTIES):
        ax = axes[0, i]
        truth = y_val[:, i]
        pred = pred_val[:, i]
        ax.scatter(truth.numpy(), pred.numpy(), s=8, alpha=0.3)
        lo = float(min(truth.min(), pred.min()))
        hi = float(max(truth.max(), pred.max()))
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        ax.set_title(f"{prop.upper()} R2={metrics[prop]['r2']:.3f}")
        ax.set_xlabel(f"True {prop.upper()}")
        ax.set_ylabel(f"Predicted {prop.upper()}")
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "property_head_calibration.png")
    fig.savefig(out_path, dpi=160)
    print(f"[Plot] saved {out_path}")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = get_device()
    os.makedirs(PREDICTOR_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    tokenizer = SMILESTokenizer()
    tokenizer.load(VOCAB_PATH)

    encoder, latent_dim, encoder_checkpoint = load_encoder(device)

    if SAMPLE_POOL_ROWS is None:
        print(f"[Data] Sampling {MAX_ROWS} molecules from full file")
    else:
        print(f"[Data] Sampling {MAX_ROWS} molecules from first {SAMPLE_POOL_ROWS} rows")
    smiles = sample_smiles(CSV_PATH, MAX_ROWS, SEED, max_rows=SAMPLE_POOL_ROWS)
    valid_smiles, y = compute_targets(smiles)
    print(f"[Data] {len(valid_smiles)} valid molecules | properties={', '.join(PROPERTIES)}")

    x = encode_smiles(encoder, tokenizer, valid_smiles, device)
    n = x.size(0)
    perm = torch.randperm(n)
    n_val = max(1, int(n * VAL_SPLIT))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    x_train, x_val = x[train_idx], x[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    # Inputs and property targets are standardized with train-set statistics.
    x_train_z, x_val_z, x_mean, x_std = standardize_train_val(x_train, x_val)
    y_train_z, y_val_z, y_mean, y_std = standardize_train_val(y_train, y_val)

    model = NormalizedPropertyPredictor(
        input_dim=latent_dim,
        output_dim=len(PROPERTIES),
        hidden_dim=HIDDEN_DIM,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    x_train_z = x_train_z.to(device)
    x_val_z = x_val_z.to(device)
    y_train_z = y_train_z.to(device)
    y_val_z = y_val_z.to(device)

    print(f"[Train] Normalized property head for {EPOCHS} epochs on frozen latents")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(x_train_z.size(0), device=device)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, x_train_z.size(0), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            pred = model(x_train_z[idx])
            loss = F.mse_loss(pred, y_train_z[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            model.eval()
            with torch.no_grad():
                val_loss = F.mse_loss(model(x_val_z), y_val_z).item()
            print(f"  Epoch {epoch:3d} | train_z_mse={total_loss / n_batches:.4f} val_z_mse={val_loss:.4f}")

    model.eval()
    with torch.no_grad():
        pred_val_z = model(x_val_z).cpu()
    pred_val = pred_val_z * y_std + y_mean
    r2 = r2_score(y_val, pred_val)
    corr = pearson(y_val, pred_val)

    metrics = {}
    for i, prop in enumerate(PROPERTIES):
        true_std = y_val[:, i].std().clamp_min(1e-12)
        pred_std = pred_val[:, i].std()
        metrics[prop] = {
            "r2": float(r2[i]),
            "pearson": float(corr[i]),
            "true_mean": float(y_val[:, i].mean()),
            "true_std": float(true_std),
            "pred_mean": float(pred_val[:, i].mean()),
            "pred_std": float(pred_std),
            "pred_std_over_true_std": float(pred_std / true_std),
        }

    payload = {
        "properties": list(PROPERTIES),
        "encoder_checkpoint": encoder_checkpoint,
        "n_train": int(train_idx.numel()),
        "n_val": int(val_idx.numel()),
        "latent_dim": latent_dim,
        "hidden_dim": HIDDEN_DIM,
        "model_state": model.state_dict(),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "metrics": metrics,
    }
    out_path = os.path.join(PREDICTOR_DIR, "normalized_property_heads.pt")
    torch.save(payload, out_path)

    metrics_path = os.path.join(RESULTS_DIR, "property_head_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({**{k: v for k, v in payload.items() if k not in {"model_state", "x_mean", "x_std", "y_mean", "y_std"}}, "metrics": metrics}, f, indent=2)

    print("[Result]")
    for prop, values in metrics.items():
        print(
            f"  {prop.upper():<7} R2={values['r2']: .4f} "
            f"pearson={values['pearson']: .4f} "
            f"pred_std/true_std={values['pred_std_over_true_std']:.4f}"
        )
    print(f"[Done] saved {out_path}")
    print(f"[Done] saved {metrics_path}")
    save_plot(y_val, pred_val, metrics)


if __name__ == "__main__":
    main()
