import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from tokenizer import PAD_IDX, SOS_IDX, EOS_IDX


class SMILESLanguageModel(nn.Module):
    # LSTM Baseline, predicts next token. 
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=512, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers
        self.embedding   = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm        = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x: (B, T) token indices — hidden/cell state initialised to zeros
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        return self.output_proj(output)  # (B, T, vocab_size)

    @torch.no_grad()
    def sample(self, n, tokenizer, max_len=120, temperature=1.0, device=None):
        device = device or next(self.parameters()).device
        self.eval()

        token = torch.full((n, 1), SOS_IDX, dtype=torch.long, device=device)
        h = torch.zeros(self.num_layers, n, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, n, self.hidden_dim, device=device)

        sequences = [[] for _ in range(n)]
        finished  = torch.zeros(n, dtype=torch.bool, device=device)

        for _ in range(max_len):
            emb = self.embedding(token)
            out, (h, c) = self.lstm(emb, (h, c))
            logits = self.output_proj(out.squeeze(1))
            probs  = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            for i in range(n):
                if not finished[i]:
                    if next_token[i].item() == EOS_IDX:
                        finished[i] = True
                    else:
                        sequences[i].append(next_token[i].item())

            token = next_token.unsqueeze(1)
            if finished.all():
                break

        return [tokenizer.decode(seq, strip_special=True) for seq in sequences]
class LSTMTrainer:
    def __init__(self, model, train_loader, val_loader, device,
                 lr=3e-4, checkpoint_dir="checkpoints_lstm"):
        self.model          = model.to(device)
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.device         = device
        self.checkpoint_dir = checkpoint_dir
        self.best_val_loss  = math.inf
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def _loss(self, batch):
        x      = batch["input"].to(self.device)
        target = batch["target"].to(self.device)
        logits = self.model(x)
        B, T, V = logits.shape
        mask   = target != PAD_IDX
        losses = F.cross_entropy(
            logits.reshape(B * T, V), target.reshape(B * T), reduction="none"
        )
        return losses.reshape(B, T)[mask].mean()

    def train_epoch(self, epoch):
        self.model.train()
        total, n = 0.0, 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} train", leave=False)
        for batch in pbar:
            loss = self._loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            total += loss.item()
            n     += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return total / n

    def validate(self, epoch):
        self.model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {epoch} val  ", leave=False):
                total += self._loss(batch).item()
                n     += 1
        return total / n
    def save_checkpoint(self, epoch, val_loss):
        path = os.path.join(self.checkpoint_dir, f"lstm_epoch{epoch:03d}.pt")
        torch.save({"epoch": epoch, "val_loss": val_loss, "model": self.model.state_dict()}, path)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(torch.load(path), os.path.join(self.checkpoint_dir, "lstm_best.pt"))

    def fit(self, n_epochs, save_every=5):
        import json
        history = {"train_loss": [], "val_loss": []}

        for epoch in tqdm(range(1, n_epochs + 1), desc="Training", unit="epoch"):
            train_loss = self.train_epoch(epoch)
            val_loss   = self.validate(epoch)
            print(f"Epoch {epoch:3d} | train={train_loss:.4f}  val={val_loss:.4f}")
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, val_loss)

        with open(os.path.join(self.checkpoint_dir, "loss_history.json"), "w") as f:
            json.dump(history, f)

        return history
