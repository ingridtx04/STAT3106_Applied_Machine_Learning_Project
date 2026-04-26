import os
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors

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
MW_MIN      = 100.0
MW_MAX      = 600.0


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
    print(f"[Tokenizer] {tok.vocab_size} tokens")

    vae = load_vae(device, tok.vocab_size)

    print(f"[Data] Loading {MAX_ROWS} molecules ...")
    smiles = load_molecules(MAX_ROWS)

    # filter valid molecules and compute MW
    valid_smiles, mw_values = [], []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_smiles.append(smi)
            mw_values.append(Descriptors.MolWt(mol))
    print(f"[Data] {len(valid_smiles)} valid molecules, MW range [{min(mw_values):.1f}, {max(mw_values):.1f}]")

    # encode to latent space using frozen VAE encoder
    print(f"[Encode] Encoding molecules in batches of {ENCODE_BATCH_SIZE} ...")
    mu = vae.encode(valid_smiles, tok, device, batch_size=ENCODE_BATCH_SIZE).detach()

    # normalise MW to [0, 1]
    mw_tensor = torch.tensor(mw_values, dtype=torch.float32, device=device)
    mw_norm   = ((mw_tensor - MW_MIN) / (MW_MAX - MW_MIN)).clamp(0.0, 1.0)

    # train MW predictor
    pred_mw = PropertyPredictor(LATENT_DIM).to(device)
    opt     = torch.optim.Adam(pred_mw.parameters(), lr=LR)

    print(f"[Train] Training MW predictor for {EPOCHS} epochs ...")
    for epoch in range(1, EPOCHS + 1):
        pred = torch.sigmoid(pred_mw(mu)).squeeze()
        loss = F.mse_loss(pred, mw_norm)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 1 == 0:
            print(f"  Epoch {epoch:3d}  MSE loss: {loss.item():.6f}")

    out_path = os.path.join(CKPT_DIR, "mw_predictor.pt")
    torch.save({"pred_mw": pred_mw.state_dict(), "mw_min": MW_MIN, "mw_max": MW_MAX}, out_path)
    print(f"[Done] Saved MW predictor to {out_path}")
