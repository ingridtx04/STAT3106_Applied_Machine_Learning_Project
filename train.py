import os
import math
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from tokenizer import PAD_IDX
from model import VAE, PropertyPredictor

from rdkit import Chem
from rdkit.Chem import QED
import rdkit, sys


contrib_path = os.path.join(rdkit.__path__[0], "Contrib")
if contrib_path not in sys.path:
    sys.path.insert(0, contrib_path)
from SA_Score import sascorer


def _to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)
    return obj

def reconstruction_loss(logits, target):
    # Masked cross-entropy: only penalise non-padding positions
    B, T, V = logits.shape
    mask = target != PAD_IDX
    losses = F.cross_entropy(logits.reshape(B * T, V), target.reshape(B * T), reduction="none")
    return losses.reshape(B, T)[mask].mean()


def kl_loss(mu, log_var):
    return -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp()).mean()


def cyclical_kl_weight(global_step, steps_per_epoch, cycle_period_epochs=10, ratio=0.5, kl_weight_max=1.0):
    # Cyclical KL annealing (Fu et al., 2019) — linearly ramps up then holds for each cycle
    T   = cycle_period_epochs * steps_per_epoch
    tau = (global_step % T) / T
    if tau < ratio:
        return (tau / ratio) * kl_weight_max
    return kl_weight_max


class VAETrainer:
    # Three-phase trainer for the SMILES VAE + economic constraint metric predictors
    # Phase 1: VAE only (recon + KL)
    # Phase 2: add property predictor training

    def __init__(self, vae, predictor_qed, predictor_sa, train_loader, val_loader,
                 device, lr=3e-4, kl_weight_max=1.0, kl_cycle_period=10,
                 prop_start_epoch=15, checkpoint_dir="checkpoints"):
        self.vae            = vae.to(device)
        self.pred_qed       = predictor_qed.to(device)
        self.pred_sa        = predictor_sa.to(device)
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.device         = device
        self.prop_start     = prop_start_epoch
        self.kl_weight_max  = kl_weight_max
        self.kl_cycle       = kl_cycle_period
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.opt_vae  = torch.optim.Adam(vae.parameters(), lr=lr)
        self.opt_qed  = torch.optim.Adam(predictor_qed.parameters(), lr=lr)
        self.opt_sa   = torch.optim.Adam(predictor_sa.parameters(), lr=lr)

        self.global_step   = 0
        self.best_val_loss = math.inf

    def _get_rdkit_scores(self, smiles_list, device):
        # Compute QED and SA scores for a batch of SMILES
        qeds, sas = [], []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                qeds.append(0.0)
                sas.append(10.0)  # penalise invalid molecules
            else:
                qeds.append(float(QED.qed(mol)))
                sas.append(float(sascorer.calculateScore(mol)))

        return (
            torch.tensor(qeds, dtype=torch.float32, device=device).unsqueeze(1),
            torch.tensor(sas,  dtype=torch.float32, device=device).unsqueeze(1),
        )

    def train_epoch(self, epoch):
        self.vae.train()
        self.pred_qed.train()
        self.pred_sa.train()
        train_prop      = epoch >= self.prop_start
        steps_per_epoch = len(self.train_loader)
        total_recon = total_kl = total_prop = n_batches = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} train", leave=False)
        for batch in pbar:
            x       = batch["input"].to(self.device)
            target  = batch["target"].to(self.device)
            lengths = batch["lengths"].to(self.device)

            out     = self.vae(x, lengths)
            logits  = out["logits"]
            mu      = out["mu"]
            log_var = out["log_var"]
            z       = out["z"]

            recon = reconstruction_loss(logits, target)
            kl    = kl_loss(mu, log_var)
            w_kl  = cyclical_kl_weight(self.global_step, steps_per_epoch,
                                       self.kl_cycle, kl_weight_max=self.kl_weight_max)
            vae_loss = recon + w_kl * kl

            self.opt_vae.zero_grad()
            vae_loss.backward()
            nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=5.0)
            self.opt_vae.step()

            prop_loss_val = 0.0
            if train_prop:
                # decode mu to SMILES for RDKit scoring 
                with torch.no_grad():
                    tokenizer = (
                        self.train_loader.dataset.dataset.tokenizer
                        if hasattr(self.train_loader.dataset, "dataset")
                        else self.train_loader.dataset.tokenizer
                    )
                    smiles_batch = self.vae.decoder.sample(mu.detach(), tokenizer, max_len=120, greedy=True)
                qed_true, sa_true = self._get_rdkit_scores(smiles_batch, self.device)

                z_det    = mu.detach()
                qed_pred = self.pred_qed.predict_qed(z_det)
                sa_pred  = self.pred_sa.predict_sa(z_det)

                prop_loss = F.mse_loss(qed_pred, qed_true) + F.mse_loss(sa_pred, sa_true)
                self.opt_qed.zero_grad()
                self.opt_sa.zero_grad()
                prop_loss.backward()
                self.opt_qed.step()
                self.opt_sa.step()
                prop_loss_val = prop_loss.item()

            recon_val = recon.item()
            kl_val    = kl.item()
            total_recon += recon_val
            total_kl    += kl_val
            total_prop  += prop_loss_val
            n_batches   += 1
            self.global_step += 1
            pbar.set_postfix(recon=f"{recon_val:.4f}", kl=f"{kl_val:.4f}")

        return {
            "recon": total_recon / n_batches,
            "kl":    total_kl    / n_batches,
            "prop":  total_prop  / n_batches,
            "kl_w":  w_kl,
        }

    def validate(self, epoch):
        self.vae.eval()
        total_recon = total_kl = n_batches = 0.0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} val  ", leave=False)
            for batch in pbar:
                x       = batch["input"].to(self.device)
                target  = batch["target"].to(self.device)
                lengths = batch["lengths"].to(self.device)

                out   = self.vae(x, lengths)
                recon = reconstruction_loss(out["logits"], target)
                kl    = kl_loss(out["mu"], out["log_var"])

                total_recon += recon.item()
                total_kl    += kl.item()
                n_batches   += 1
                pbar.set_postfix(recon=f"{recon.item():.4f}", kl=f"{kl.item():.4f}")

        val_loss = total_recon / n_batches + total_kl / n_batches
        return {"recon": total_recon / n_batches, "kl": total_kl / n_batches, "loss": val_loss}

    def save_checkpoint(self, epoch, val_loss):
        path = os.path.join(self.checkpoint_dir, f"vae_epoch{epoch:03d}.pt")
        torch.save({
            "epoch":       epoch,
            "val_loss":    val_loss,
            "vae":         _to_cpu(self.vae.state_dict()),
            "pred_qed":    _to_cpu(self.pred_qed.state_dict()),
            "pred_sa":     _to_cpu(self.pred_sa.state_dict()),
            "opt_vae":     _to_cpu(self.opt_vae.state_dict()),
            "opt_qed":     _to_cpu(self.opt_qed.state_dict()),
            "opt_sa":      _to_cpu(self.opt_sa.state_dict()),
            "global_step": self.global_step,
        }, path)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = os.path.join(self.checkpoint_dir, "vae_best.pt")
            shutil.copyfile(path, best_path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.vae.load_state_dict(ckpt["vae"])
        self.pred_qed.load_state_dict(ckpt["pred_qed"])
        self.pred_sa.load_state_dict(ckpt["pred_sa"])
        self.opt_vae.load_state_dict(ckpt["opt_vae"])
        if "opt_qed" in ckpt:
            self.opt_qed.load_state_dict(ckpt["opt_qed"])
        if "opt_sa" in ckpt:
            self.opt_sa.load_state_dict(ckpt["opt_sa"])
        self.global_step   = ckpt.get("global_step", 0)
        self.best_val_loss = ckpt.get("val_loss", math.inf)
        return ckpt["epoch"]

    def fit(self, n_epochs, start_epoch=1, save_every=5):
        import json
        history = {"train_recon": [], "train_kl": [], "val_loss": []}

        for epoch in tqdm(range(start_epoch, n_epochs + 1), desc="Training", unit="epoch"):
            train_metrics = self.train_epoch(epoch)
            val_metrics   = self.validate(epoch)
            print(
                f"Epoch {epoch:3d} | "
                f"train recon={train_metrics['recon']:.4f} "
                f"kl={train_metrics['kl']:.4f} "
                f"prop={train_metrics['prop']:.4f} "
                f"kl_w={train_metrics['kl_w']:.3f} | "
                f"val recon={val_metrics['recon']:.4f} "
                f"kl={val_metrics['kl']:.4f} "
                f"loss={val_metrics['loss']:.4f}"
            )
            history["train_recon"].append(train_metrics["recon"])
            history["train_kl"].append(train_metrics["kl"])
            history["val_loss"].append(val_metrics["loss"])

            if epoch % save_every == 0:
                self.save_checkpoint(epoch, val_metrics["loss"])

        # save loss history so compare.ipynb can plot it
        history_path = os.path.join(self.checkpoint_dir, "loss_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f)
        return history
