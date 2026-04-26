import os
import torch

from tokenizer import SMILESTokenizer
from dataset import build_dataloaders
from lstm_baseline import SMILESLanguageModel, LSTMTrainer

# config 
CSV_PATH   = "ZINC20-Druglike/zinc-druglike-cano.csv"
VOCAB_PATH = "vocab.json"
CKPT_DIR   = "checkpoints_lstm"

EMBED_DIM  = 256
HIDDEN_DIM = 512
NUM_LAYERS = 3
DROPOUT    = 0.1
BATCH_SIZE = 512
LR         = 3e-4
EPOCHS     = 5
MAX_ROWS   = int(os.environ.get("MAX_ROWS", 500_000))
SEED       = int(os.environ.get("SEED", 42))


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
    trainer.fit(n_epochs=EPOCHS, save_every=5)
    print("\nDone. Open compare.ipynb to compare generation.")
