import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from tokenizer import PAD_IDX, SOS_IDX, EOS_IDX


class Encoder(nn.Module):
    # Bidirectional LSTM encoder — maps a token sequence to (mu, log_var) in latent space

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=512, latent_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # project last-layer h and c from both directions into latent space
        proj_in = hidden_dim * 2 * 2  # 2 dirs × (h + c) = 4 * hidden_dim
        self.fc_mu      = nn.Linear(proj_in, latent_dim)
        self.fc_log_var = nn.Linear(proj_in, latent_dim)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x, lengths):
        # x: (B, T) token indices, lengths: (B,) actual sequence lengths
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, c_n) = self.lstm(packed)
        # take only the last layer (index -2 forward, -1 backward)
        h_last  = torch.cat([h_n[-2], h_n[-1]], dim=-1)   # (B, 2*hidden_dim)
        c_last  = torch.cat([c_n[-2], c_n[-1]], dim=-1)   # (B, 2*hidden_dim)
        context = torch.cat([h_last, c_last], dim=-1)     # (B, 4*hidden_dim)
        return self.fc_mu(context), self.fc_log_var(context)


class Decoder(nn.Module):
    # LSTM decoder conditioned on a latent vector z
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=512, latent_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding   = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm        = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        # project z into initial hidden and cell states for all layers
        self.z_to_h = nn.Linear(latent_dim, num_layers * hidden_dim)
        self.z_to_c = nn.Linear(latent_dim, num_layers * hidden_dim)

    def _init_hidden(self, z):
        B = z.size(0)
        h0 = self.z_to_h(z).view(self.num_layers, B, self.hidden_dim)
        c0 = self.z_to_c(z).view(self.num_layers, B, self.hidden_dim)
        return h0.contiguous(), c0.contiguous()

    def forward(self, z, target_input):
        h0, c0   = self._init_hidden(z)
        embedded = self.embedding(target_input)
        output, _ = self.lstm(embedded, (h0, c0))
        return self.output_proj(output)  # (B, T, vocab_size)

    @torch.no_grad()
    def sample(self, z, tokenizer, max_len=120, temperature=1.0, greedy=False):
        # Autoregressive decoding — generates one token at a time until EOS or max_len
        B = z.size(0)
        device = z.device
        h, c = self._init_hidden(z)

        token     = torch.full((B, 1), SOS_IDX, dtype=torch.long, device=device)
        sequences = [[] for _ in range(B)]
        finished  = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            emb = self.embedding(token)
            out, (h, c) = self.lstm(emb, (h, c))
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
    # Full VAE: encoder → reparameterize → decoder

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
        # x: (B, T) padded token indices, lengths: (B,) actual lengths
        mu, log_var = self.encoder(x, lengths)
        z = self.reparameterize(mu, log_var)
        logits = self.decoder(z, x)  # teacher-forcing
        return {"logits": logits, "mu": mu, "log_var": log_var, "z": z}

    @torch.no_grad()
    def encode(self, smiles_list, tokenizer, device):
        # Encode a list of SMILES strings to latent vectors (mu, no noise)
        seqs    = [torch.tensor(tokenizer.encode(s), dtype=torch.long) for s in smiles_list]
        lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
        from torch.nn.utils.rnn import pad_sequence as _pad
        x = _pad(seqs, batch_first=True, padding_value=PAD_IDX).to(device)
        mu, _ = self.encoder(x, lengths.to(device))
        return mu


class PropertyPredictor(nn.Module):
    # MLP that predicts a QED & SA from a latent vector

    def __init__(self, latent_dim=256, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, z):
        return self.net(z)  # (B, 1) raw scalar

    def predict_qed(self, z):
        # QED
        return torch.sigmoid(self.forward(z))

    def predict_sa(self, z):
        # SA score 
        return 1.0 + 9.0 * torch.sigmoid(self.forward(z))
