import os
import torch

from tokenizer import SMILESTokenizer
from model import Encoder, Decoder, VAE, PropertyPredictor
from dataset import build_dataloaders
from train import VAETrainer

# config
CSV_PATH   = "ZINC20-Druglike/zinc-druglike-cano.csv"
VOCAB_PATH = "vocab.json"
CKPT_DIR   = "checkpoints"

EMBED_DIM  = 128
HIDDEN_DIM = 512
LATENT_DIM = 256
NUM_LAYERS = 2
DROPOUT    = 0.1
BATCH_SIZE = 512
LR         = 3e-4
EPOCHS     = 30
KL_CYCLE   = 10
PROP_START = 15  # epoch to start training property predictors


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_tokenizer():
    # Load vocab from disk if it exists, otherwise scan the first 200k rows to build it
    tok = SMILESTokenizer()
    if os.path.exists(VOCAB_PATH):
        tok.load(VOCAB_PATH)
        print(f"[Tokenizer] Loaded vocab ({tok.vocab_size} tokens)")
        return tok

    print("[Tokenizer] Building vocab from corpus (streaming) ...")

    def smiles_generator():
        # generator — reads one line at a time, never holds more than one in RAM
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            next(f)  # skip header
            for i, line in enumerate(f):
                smi = line.strip()
                if smi:
                    yield smi
                if i >= 200_000:
                    break

    tok.build_vocab(smiles_generator())
    tok.save(VOCAB_PATH)
    print(f"[Tokenizer] Built vocab ({tok.vocab_size} tokens), saved to {VOCAB_PATH}")
    return tok


def build_vae(vocab_size):
    encoder = Encoder(vocab_size, EMBED_DIM, HIDDEN_DIM, LATENT_DIM, NUM_LAYERS, DROPOUT)
    decoder = Decoder(vocab_size, EMBED_DIM, HIDDEN_DIM, LATENT_DIM, NUM_LAYERS, DROPOUT)
    return VAE(encoder, decoder)


if __name__ == "__main__":
    device = get_device()
    print(f"[Device] {device}")

    tokenizer = build_tokenizer()

    print("[Data] Indexing dataset ...")
    train_loader, val_loader, test_loader = build_dataloaders(
        CSV_PATH, tokenizer, batch_size=BATCH_SIZE, num_workers=0, max_rows=500000
    )
    print(f"[Data] {len(train_loader.dataset)} train / {len(val_loader.dataset)} val / {len(test_loader.dataset)} test")

    vae      = build_vae(tokenizer.vocab_size)
    pred_qed = PropertyPredictor(LATENT_DIM)
    pred_sa  = PropertyPredictor(LATENT_DIM)

    trainer = VAETrainer(
        vae, pred_qed, pred_sa,
        train_loader, val_loader,
        device=device,
        lr=LR,
        kl_cycle_period=KL_CYCLE,
        prop_start_epoch=PROP_START,
        checkpoint_dir=CKPT_DIR,
    )
    trainer.fit(n_epochs=EPOCHS, save_every=5)
    print("\nTraining done. Open compare.ipynb to compare generation.")
