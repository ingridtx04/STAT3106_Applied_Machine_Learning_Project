import math

import torch
import torch.nn.functional as F

from model import VAE, PropertyPredictor
from tokenizer import SMILESTokenizer


class ConstrainedGenerator:
    # Generate molecules by maximising R(z) = QED_pred(z) - lambda * SA_norm_pred(z)

    def __init__(self, vae, predictor_qed, predictor_sa, tokenizer,
                 lambda_=0.5, beta=0.01, device=None, cost_fn=None,
                 cost_terms=None):
        self.vae         = vae
        self.pred_qed    = predictor_qed
        self.pred_sa     = predictor_sa
        self.tokenizer   = tokenizer
        self.lambda_     = lambda_
        self.beta        = beta
        self.device      = device or next(vae.parameters()).device
        self.cost_fn     = cost_fn    # single cost fn scaled by lambda_
        self.cost_terms  = cost_terms # list of (cost_fn, lambda) pairs; overrides cost_fn

        # freeze all weights — only z will have gradients during generation
        for model in (self.vae, self.pred_qed, self.pred_sa):
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)

    def reward(self, z):
        qed = self.pred_qed.predict_qed(z)
        if self.cost_terms is not None:
            # multi-constraint: R(z) = QED - Σ λ_i · cost_i(z)
            return qed - sum(lam * fn(z) for fn, lam in self.cost_terms)
        if self.cost_fn is not None:
            cost = self.cost_fn(z)
        else:
            sa   = self.pred_sa.predict_sa(z)
            cost = (sa - 1.0) / 9.0
        return qed - self.lambda_ * cost

    def gradient_ascent(self, z_init, n_steps=50, lr=0.01):
        # gradient ascent on z to maximise reward, 
        z = z_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=lr)

        for _ in range(n_steps):
            optimizer.zero_grad()
            r      = self.reward(z)
            anchor = self.beta * z.pow(2).mean()
            loss   = -r.mean() + anchor
            loss.backward()
            optimizer.step()

        return z.detach()
    def generate(self, n_molecules, z_init=None, n_steps=50, ga_lr=0.01, temperature=0.8, n_restarts=1):
        # Generate molecules constrained by the economic reward
        latent_dim = self.vae.encoder.fc_mu.out_features

        if z_init is None:
            z_init = torch.randn(n_molecules, latent_dim, device=self.device)

        best_z = None
        best_r = torch.full((n_molecules,), -1e9, device=self.device)

        for _ in range(n_restarts):
            z_start = z_init + 0.1 * torch.randn_like(z_init) if n_restarts > 1 else z_init
            z_opt   = self.gradient_ascent(z_start, n_steps=n_steps, lr=ga_lr)
            r       = self.reward(z_opt).squeeze(1)
            improved = r > best_r
            if best_z is None:
                best_z = z_opt.clone()
                best_r = r.clone()
            else:
                best_z[improved] = z_opt[improved]
                best_r[improved] = r[improved]

        return self.vae.decoder.sample(best_z, self.tokenizer, temperature=temperature)