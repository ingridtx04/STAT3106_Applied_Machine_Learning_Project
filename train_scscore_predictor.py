import os
import sys
import torch
import torch.nn.functional as F
from rdkit import Chem

# SCScore is not pip-installable; use the cloned repo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scscore_repo"))
from scscore.standalone_model_numpy import SCScorer

from tokenizer import SMILESTokenizer
from model import Encoder, Decoder, VAE, PropertyPredictor

CSV_PATH   = "ZINC20-Druglike/zinc-druglike-cano.csv"
VOCAB_PATH = "vocab.json"
CKPT_DIR   = "checkpoints"

EMBED_DIM  = 256
HIDDEN_DIM = 512
LATENT_DIM = 256
NUM_LAYERS = 3
LAYER_DIMS = [256, 512, 1024]
DROPOUT    = 0.1

MAX_ROWS    = int(os.environ.get("MAX_ROWS", 2_000))
SEED        = int(os.environ.get("SEED", 42))
ENCODE_BATCH_SIZE = int(os.environ.get("ENCODE_BATCH_SIZE", 256))
EPOCHS      = 100
LR          = 1e-3

# SCScore range is [1, 5] — normalise to [0, 1] for training
SC_MIN = 1.0
SC_MAX = 5.0


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_vae(device, vocab_size):
    encoder = Encoder(vocab_size, EMBED_DIM, LAYER_DIMS, LATENT_DIM, DROPOUT)
    decoder = Decoder(vocab_size, EMBED_DIM, HIDDEN_DIM, LATENT_DIM, NUM_LAYERS, DROPOUT)
    vae     = VAE(encoder, decoder).to(device)
    pred_qed = PropertyPredictor(LATENT_DIM).to(device)
    pred_sa  = PropertyPredictor(LATENT_DIM).to(device)
    ckpt = torch.load(os.path.join(CKPT_DIR, "vae_best.pt"), map_location=device)
    vae.load_state_dict(ckpt["vae"])
    pred_qed.load_state_dict(ckpt["pred_qed"])
    pred_sa.load_state_dict(ckpt["pred_sa"])
    print(f"[VAE] Loaded checkpoint from epoch {ckpt['epoch']}")
    return vae


def load_molecules(n):
    smiles = []
    with open(CSV_PATH) as f:
        next(f)
        for line in f:
            smi = line.strip()
            if smi:
                smiles.append(smi)
            if len(smiles) >= n:
                break
    return smiles


if __name__ == "__main__":
    torch.manual_seed(SEED)
    device = get_device()
    print(f"[Device] {device}")

    tok = SMILESTokenizer()
    tok.load(VOCAB_PATH)

    vae = load_vae(device, tok.vocab_size)

    # load pre-trained SCScore model 
    scorer    = SCScorer()
    sc_model  = os.path.join("scscore_repo", "models",
                             "full_reaxys_model_1024bool",
                             "model.ckpt-10654.as_numpy.json.gz")
    scorer.restore(weight_path=sc_model)

    print(f"[Data] Loading {MAX_ROWS} molecules ...")
    smiles = load_molecules(MAX_ROWS)

    # filter valid molecules and compute SCScore labels
    valid_smiles, sc_values = [], []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            _, score = scorer.get_score_from_smi(smi)
            valid_smiles.append(smi)
            sc_values.append(float(score))

    print(f"[Data] {len(valid_smiles)} valid molecules, "
          f"SCScore range [{min(sc_values):.2f}, {max(sc_values):.2f}]")

    # encode to latent space with VAE encoder
    print(f"[Encode] Encoding molecules in batches of {ENCODE_BATCH_SIZE} ...")
    mu = vae.encode(valid_smiles, tok, device, batch_size=ENCODE_BATCH_SIZE).detach()

    # normalise SCScore to [0, 1] — higher = harder to synthesise
    sc_tensor = torch.tensor(sc_values, dtype=torch.float32, device=device)
    sc_target = ((sc_tensor - SC_MIN) / (SC_MAX - SC_MIN)).clamp(0.0, 1.0)

    # train predictor: sigmoid(forward(z)) ≈ sc_target
    pred_scscore = PropertyPredictor(LATENT_DIM).to(device)
    opt          = torch.optim.Adam(pred_scscore.parameters(), lr=LR)

    print(f"[Train] Training SCScore predictor for {EPOCHS} epochs ...")
    for epoch in range(1, EPOCHS + 1):
        pred = torch.sigmoid(pred_scscore(mu)).squeeze()
        loss = F.mse_loss(pred, sc_target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 1 == 0:
            print(f"  Epoch {epoch:3d}  MSE loss: {loss.item():.6f}")

    out_path = os.path.join(CKPT_DIR, "scscore_predictor.pt")
    torch.save({"pred_scscore": pred_scscore.state_dict(),
                "sc_min": SC_MIN, "sc_max": SC_MAX}, out_path)
    print(f"[Done] Saved SCScore predictor to {out_path}")
