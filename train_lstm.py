import os
import torch

from tokenizer import SMILESTokenizer
from dataset import build_dataloaders
from lstm_baseline import SMILESLanguageModel, LSTMTrainer

# config 
CSV_PATH   = "ZINC20-Druglike/zinc-druglike-cano.csv"
VOCAB_PATH = "vocab.json"
CKPT_DIR   = os.environ.get("CKPT_DIR", "checkpoints")

EMBED_DIM  = int(os.environ.get("EMBED_DIM", 128))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", 512))
NUM_LAYERS = int(os.environ.get("NUM_LAYERS", 2))
DROPOUT    = 0.1
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 512))
LR         = float(os.environ.get("LR", 3e-4))
EPOCHS     = int(os.environ.get("EPOCHS", 15))
MAX_ROWS   = int(os.environ.get("MAX_ROWS", 500_000))
SEED       = int(os.environ.get("SEED", 42))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", 1))


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    device = get_device()
    print(f"[Device] {device}")

    tokenizer = SMILESTokenizer()
    tokenizer.load(VOCAB_PATH)
    print(f"[Tokenizer] Loaded vocab ({tokenizer.vocab_size} tokens)")

    print("[Data] Indexing dataset ...")
    train_loader, val_loader, test_loader = build_dataloaders(
        CSV_PATH, tokenizer, batch_size=BATCH_SIZE, num_workers=0, max_rows=MAX_ROWS, seed=SEED
    )
    print(f"[Data] {len(train_loader.dataset)} train / {len(val_loader.dataset)} val / {len(test_loader.dataset)} test")

    model = SMILESLanguageModel(tokenizer.vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[LSTM_Model] {n_params:,} trainable parameters")

    trainer = LSTMTrainer(
        model, train_loader, val_loader,
        device=device,
        lr=LR,
        checkpoint_dir=CKPT_DIR,
    )
    trainer.fit(n_epochs=EPOCHS, save_every=SAVE_EVERY)
    print("\nDone. Open compare.ipynb to compare generation.")
