import os
import torch

from tokenizer import SMILESTokenizer
from model import Encoder, Decoder, VAE, PropertyPredictor
from dataset import build_dataloaders
from train import VAETrainer

# config
CSV_PATH   = "ZINC20-Druglike/zinc-druglike-cano.csv"
VOCAB_PATH = "vocab.json"
CKPT_DIR   = os.environ.get("CKPT_DIR", "checkpoints")

EMBED_DIM   = int(os.environ.get("EMBED_DIM", 128))
HIDDEN_DIM  = int(os.environ.get("HIDDEN_DIM", 480))  # decoder hidden dim
LATENT_DIM  = int(os.environ.get("LATENT_DIM", 128))
NUM_LAYERS  = int(os.environ.get("NUM_LAYERS", 2))
LAYER_DIMS  = [int(x) for x in os.environ.get("LAYER_DIMS", "256,512").split(",")]
DROPOUT     = 0.1
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 512))
LR         = float(os.environ.get("LR", 3e-4))
EPOCHS     = int(os.environ.get("EPOCHS", 30))
KL_CYCLE   = int(os.environ.get("KL_CYCLE", 10))
PROP_START = int(os.environ.get("PROP_START", 15))  # epoch to start training property predictors
MAX_ROWS   = int(os.environ.get("MAX_ROWS", 500_000))
SEED       = int(os.environ.get("SEED", 42))
RESUME_FROM = os.environ.get("RESUME_FROM")
SAVE_EVERY  = int(os.environ.get("SAVE_EVERY", 1))


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
    encoder = Encoder(vocab_size, EMBED_DIM, LAYER_DIMS, LATENT_DIM, DROPOUT)
    decoder = Decoder(vocab_size, EMBED_DIM, HIDDEN_DIM, LATENT_DIM, NUM_LAYERS, DROPOUT)
    return VAE(encoder, decoder)


if __name__ == "__main__":
    device = get_device()
    print(f"[Device] {device}")

    tokenizer = build_tokenizer()

    print("[Data] Indexing dataset ...")
    train_loader, val_loader, test_loader = build_dataloaders(
        CSV_PATH, tokenizer, batch_size=BATCH_SIZE, num_workers=0, max_rows=MAX_ROWS, seed=SEED
    )
    print(f"[Data] {len(train_loader.dataset)} train / {len(val_loader.dataset)} val / {len(test_loader.dataset)} test")

    vae      = build_vae(tokenizer.vocab_size)
    pred_qed = PropertyPredictor(LATENT_DIM)
    pred_sa  = PropertyPredictor(LATENT_DIM)
    n_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(
        f"[VAE_Model] embed={EMBED_DIM} latent={LATENT_DIM} "
        f"encoder_layers={LAYER_DIMS} decoder_hidden={HIDDEN_DIM} "
        f"decoder_layers={NUM_LAYERS} params={n_params:,}"
    )

    trainer = VAETrainer(
        vae, pred_qed, pred_sa,
        train_loader, val_loader,
        device=device,
        lr=LR,
        kl_cycle_period=KL_CYCLE,
        prop_start_epoch=PROP_START,
        checkpoint_dir=CKPT_DIR,
    )

    start_epoch = 1
    if RESUME_FROM:
        last_epoch = trainer.load_checkpoint(RESUME_FROM)
        start_epoch = last_epoch + 1
        print(f"[Checkpoint] Resumed from {RESUME_FROM} at epoch {last_epoch}")

    trainer.fit(n_epochs=EPOCHS, start_epoch=start_epoch, save_every=SAVE_EVERY)
    print("\nTraining done. Open compare.ipynb to compare generation.")
