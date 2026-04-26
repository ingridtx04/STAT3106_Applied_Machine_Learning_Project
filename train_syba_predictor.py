import os
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from syba.syba import SybaClassifier

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

    print("[SYBA] Loading classifier ...")
    syba = SybaClassifier()
    syba.fitDefaultScore()

    print(f"[Data] Loading {MAX_ROWS} molecules ...")
    smiles = load_molecules(MAX_ROWS)

    # filter valid molecules and compute SYBA labels
    valid_smiles, syba_values = [], []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            score = float(syba.predict(mol=mol))
            valid_smiles.append(smi)
            syba_values.append(score)

    syba_arr  = np.array(syba_values, dtype=np.float32)
    syba_mean = float(syba_arr.mean())
    syba_std  = float(syba_arr.std()) + 1e-8
    print(f"[Data] {len(valid_smiles)} valid molecules, "
          f"SYBA mean={syba_mean:.2f}  std={syba_std:.2f}  "
          f"range=[{syba_arr.min():.1f}, {syba_arr.max():.1f}]")

    # encode to latent space with frozen VAE encoder
    print(f"[Encode] Encoding molecules in batches of {ENCODE_BATCH_SIZE} ...")
    mu = vae.encode(valid_smiles, tok, device, batch_size=ENCODE_BATCH_SIZE).detach()

    #sigmoid-normalise SYBA labels to [0, 1] for regression target
    # cost = 1 - sigmoid(pred(z))
    syba_tensor = torch.tensor(syba_values, dtype=torch.float32, device=device)
    syba_target = torch.sigmoid((syba_tensor - syba_mean) / syba_std)

    # train predictor: sigmoid(forward(z)) ≈ syba_target
    pred_syba = PropertyPredictor(LATENT_DIM).to(device)
    opt       = torch.optim.Adam(pred_syba.parameters(), lr=LR)

    print(f"[Train] Training SYBA predictor for {EPOCHS} epochs ...")
    for epoch in range(1, EPOCHS + 1):
        pred = torch.sigmoid(pred_syba(mu)).squeeze()
        loss = F.mse_loss(pred, syba_target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 1 == 0:
            print(f"  Epoch {epoch:3d}  MSE loss: {loss.item():.6f}")

    out_path = os.path.join(CKPT_DIR, "syba_predictor.pt")
    torch.save({"pred_syba": pred_syba.state_dict(),
                "syba_mean": syba_mean,
                "syba_std":  syba_std}, out_path)
    print(f"[Done] Saved SYBA predictor to {out_path}")
