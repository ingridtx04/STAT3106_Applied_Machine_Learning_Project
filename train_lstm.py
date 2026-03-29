import torch

from tokenizer import SMILESTokenizer
from dataset import build_dataloaders
from lstm_baseline import SMILESLanguageModel, LSTMTrainer

# config 
CSV_PATH   = "ZINC20-Druglike/zinc-druglike-cano.csv"
VOCAB_PATH = "vocab.json"
CKPT_DIR   = "checkpoints_lstm"

EMBED_DIM  = 128
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT    = 0.1
BATCH_SIZE = 512
LR         = 3e-4
EPOCHS     = 30


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    tok = SMILESTokenizer()
    tok.load(VOCAB_PATH)
    print(f"[Tokenizer] Loaded vocab ({tok.vocab_size} tokens)")

    print("[Data] Indexing dataset ...")
    train_loader, val_loader, _ = build_dataloaders(
        CSV_PATH, tok, batch_size=BATCH_SIZE, num_workers=0, max_rows=500000
    )
    print(f"[Data] {len(train_loader.dataset)} train / {len(val_loader.dataset)} val")

    model = SMILESLanguageModel(tok.vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {n_params:,} trainable parameters")

    trainer = LSTMTrainer(
        model, train_loader, val_loader,
        device=device,
        lr=LR,
        checkpoint_dir=CKPT_DIR,
    )
    trainer.fit(n_epochs=EPOCHS, save_every=5)
    print("\nDone. Open compare.ipynb to compare generation.")
