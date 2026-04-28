import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

from tokenizer import PAD_IDX, SOS_IDX, EOS_IDX


class Encoder(nn.Module):
    # Progressive bidirectional LSTM encoder.
    # Each layer's bidirectional output (2×hidden) feeds the next layer's input,
    #
    # layer_dims controls each BiLSTM's hidden size:
    #   [256, 512, 1024, 1024]  →  per-step outputs: 512, 1024, 2048, 2048
    #
    # Context fed to fc_mu / fc_log_var:
    #   concat(h_fwd, h_bwd, c_fwd, c_bwd) from the last layer = 4 × 1024 = 4096

    def __init__(self, vocab_size, embed_dim=256, layer_dims=None, latent_dim=512, dropout=0.1):
        super().__init__()
        self.layer_dims = layer_dims or [256, 512, 1024, 1024]
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)

        self.lstm_layers = nn.ModuleList()
        self.dropouts    = nn.ModuleList()
        in_dim = embed_dim
        for i, hidden in enumerate(self.layer_dims):
            self.lstm_layers.append(
                nn.LSTM(in_dim, hidden, num_layers=1, batch_first=True, bidirectional=True)
            )
            if i < len(self.layer_dims) - 1:
                self.dropouts.append(nn.Dropout(dropout))
            in_dim = hidden * 2  # bidirectional doubles output width

        last_hidden = self.layer_dims[-1]
        proj_in = last_hidden * 4  # 2 dirs × (h + c)
        self.fc_mu      = nn.Linear(proj_in, latent_dim)
        self.fc_log_var = nn.Linear(proj_in, latent_dim)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)

        h_n, c_n = None, None
        for i, lstm in enumerate(self.lstm_layers):
            packed, (h_n, c_n) = lstm(packed)
            if i < len(self.lstm_layers) - 1:
                packed = PackedSequence(
                    self.dropouts[i](packed.data),
                    packed.batch_sizes,
                    packed.sorted_indices,
                    packed.unsorted_indices,
                )

        # h_n, c_n: (2, B, last_hidden) — index 0=forward, 1=backward
        h_last  = torch.cat([h_n[0], h_n[1]], dim=-1)   # (B, 2*last_hidden)
        c_last  = torch.cat([c_n[0], c_n[1]], dim=-1)   # (B, 2*last_hidden)
        context = torch.cat([h_last, c_last],  dim=-1)  # (B, 4*last_hidden)
        return self.fc_mu(context), self.fc_log_var(context)


class Decoder(nn.Module):
    # LSTM decoder conditioned on a latent vector z.

    def __init__(self, vocab_size, embed_dim=256, hidden_dim=1024, latent_dim=512,
                 num_layers=4, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim

        self.embedding   = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm        = nn.LSTM(
            input_size=embed_dim + latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        self.z_to_h = nn.Linear(latent_dim, num_layers * hidden_dim)
        self.z_to_c = nn.Linear(latent_dim, num_layers * hidden_dim)

    def _init_hidden(self, z):
        B  = z.size(0)
        h0 = self.z_to_h(z).view(self.num_layers, B, self.hidden_dim)
        c0 = self.z_to_c(z).view(self.num_layers, B, self.hidden_dim)
        return h0.contiguous(), c0.contiguous()

    def _condition_inputs(self, embedded, z):
        z_steps = z.unsqueeze(1).expand(-1, embedded.size(1), -1)
        return torch.cat([embedded, z_steps], dim=-1)

    def forward(self, z, target_input):
        h0, c0   = self._init_hidden(z)
        embedded = self.embedding(target_input)
        conditioned = self._condition_inputs(embedded, z)
        output, _ = self.lstm(conditioned, (h0, c0))
        return self.output_proj(output)

    @torch.no_grad()
    def sample(self, z, tokenizer, max_len=120, temperature=1.0, greedy=False):
        B = z.size(0)
        device = z.device
        h, c = self._init_hidden(z)

        token     = torch.full((B, 1), SOS_IDX, dtype=torch.long, device=device)
        sequences = [[] for _ in range(B)]
        finished  = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            emb = self.embedding(token)
            conditioned = self._condition_inputs(emb, z)
            out, (h, c) = self.lstm(conditioned, (h, c))
            logits = self.output_proj(out.squeeze(1))

            if greedy:
                next_token = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            for i in range(B):
                if not finished[i]:
                    if next_token[i].item() == EOS_IDX:
                        finished[i] = True
                    else:
                        sequences[i].append(next_token[i].item())

            token = next_token.unsqueeze(1)
            if finished.all():
                break

        return [tokenizer.decode(seq, strip_special=True) for seq in sequences]


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, log_var):
        if self.training:
            std = (0.5 * log_var).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x, lengths):
        mu, log_var = self.encoder(x, lengths)
        z = self.reparameterize(mu, log_var)
        logits = self.decoder(z, x)
        return {"logits": logits, "mu": mu, "log_var": log_var, "z": z}

    @torch.no_grad()
    def encode(self, smiles_list, tokenizer, device, batch_size=None):
        from torch.nn.utils.rnn import pad_sequence as _pad

        if batch_size is None:
            seqs    = [torch.tensor(tokenizer.encode(s), dtype=torch.long) for s in smiles_list]
            lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
            x = _pad(seqs, batch_first=True, padding_value=PAD_IDX).to(device)
            mu, _ = self.encoder(x, lengths.to(device))
            return mu

        mus = []
        for start in range(0, len(smiles_list), batch_size):
            batch = smiles_list[start:start + batch_size]
            seqs = [torch.tensor(tokenizer.encode(s), dtype=torch.long) for s in batch]
            lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
            x = _pad(seqs, batch_first=True, padding_value=PAD_IDX).to(device)
            mu, _ = self.encoder(x, lengths.to(device))
            mus.append(mu.cpu())
        return torch.cat(mus, dim=0).to(device)


class PropertyPredictor(nn.Module):
    # MLP that predicts QED, SA, SCScore, SYBA, or MW from a latent vector.
    def __init__(self, latent_dim=512, hidden_dim=256):
        super().__init__()
        mid_dim = max(64, hidden_dim // 4)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, 1),
        )

    def forward(self, z):
        return self.net(z)  # (B, 1) raw scalar

    def predict_qed(self, z):
        return torch.sigmoid(self.forward(z))

    def predict_sa(self, z):
        return 1.0 + 9.0 * torch.sigmoid(self.forward(z))


class NormalizedPropertyPredictor(nn.Module):
    # Standardization: z_norm = (z - x_mean) / x_std,
    # and y_norm = (property - y_mean) / y_std.

    def __init__(self, input_dim, output_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)
