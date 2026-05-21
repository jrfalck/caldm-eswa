"""
CALDM Benchmark — End-to-End Comparison Against PyOD Baselines.

Runs CALDM-D (diffusion-stage context conditioning) and CALDM-J (joint
VAE+diffusion conditioning) against the 13 PyOD baselines used in Juan's
praxis on:

  - BLS compensation dataset (dual-headed VAE, 80:20 split, stratified)
  - All ADBench datasets (single-headed VAE, transductive evaluation)

Output: CSVs that a Streamlit app can read and visualize.

Quick start (Jupyter)
---------------------
    from caldm_benchmark import run_bls_benchmark, run_adbench_benchmark

    # BLS only — use 20% of data, 3 seeds (matches V3 ablation methodology)
    bls = run_bls_benchmark(
        data_path="data/final_data_subsetFLGAAL8.npz",
        subset_pct=0.2,
        seeds=[42, 7, 123],
        output_dir="benchmark_outputs/bls",
    )

    # ADBench only — transductive, 3 seeds
    adb = run_adbench_benchmark(
        input_dir="data/adbench",
        seeds=[42, 7, 123],
        output_dir="benchmark_outputs/adbench",
    )

    # Quick smoke test (1 seed, fewer epochs, skips slow baselines)
    bls = run_bls_benchmark(data_path="...", quick=True)

CSV outputs
-----------
    bls_per_run.csv          one row per (method, seed)
    bls_summary.csv          one row per method (aggregated across seeds)

    adbench_per_run.csv      one row per (dataset, method, seed)
    adbench_per_dataset.csv  one row per (dataset, method)
    adbench_summary.csv      one row per method (aggregated across datasets)

    methods_metadata.csv     one row per method (family, library, version)
    run_manifest.csv         one row per benchmark run (timestamp, hyperparams)

Author: Juan
"""

import os
import sys
import time
import json
import random
import argparse
import warnings
import traceback
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_SEEDS = [42, 7, 123]


# =============================================================================
# PyOD baselines registry
# =============================================================================
# Each entry: name -> (import_path, class_name, kwargs, family, slow_threshold)
# slow_threshold: skip this method if dataset has more rows than this
# (None = always run)

PYOD_BASELINES = {
    "KNN":          ("pyod.models.knn",        "KNN",       {"n_neighbors": 20},      "proximity",     None),
    "LOF":          ("pyod.models.lof",        "LOF",       {"n_neighbors": 20},      "proximity",     None),
    "COF":          ("pyod.models.cof",        "COF",       {"n_neighbors": 20},      "proximity",     None),
    "IForest":      ("pyod.models.iforest",    "IForest",   {"n_estimators": 100},    "ensemble",      None),
    "PCA":          ("pyod.models.pca",        "PCA",       {},                        "projection",    None),
    "CBLOF":        ("pyod.models.cblof",      "CBLOF",     {},                        "clustering",    None),
    "HBOS":         ("pyod.models.hbos",       "HBOS",      {"n_bins": 10},            "statistical",   None),
    "OCSVM":        ("pyod.models.ocsvm",      "OCSVM",     {},                        "boundary",      None),
    "ECOD":         ("pyod.models.ecod",       "ECOD",      {},                        "statistical",   None),
    "COPOD":        ("pyod.models.copod",      "COPOD",     {},                        "statistical",   None),
    "SOD":          ("pyod.models.sod",        "SOD",       {"n_neighbors": 20, "ref_set": 10}, "subspace", None),
    "LODA":         ("pyod.models.loda",       "LODA",      {"n_bins": 10, "n_random_cuts": 100}, "ensemble", None),
    "DeepSVDD":     ("pyod.models.deep_svdd",  "DeepSVDD",  {"n_features": None},      "deep",          None),
}


# =============================================================================
# Model components
# =============================================================================

class ContextEncoder(nn.Module):
    """Transformer-based context encoder. Produces a context embedding from x."""
    def __init__(self, input_dim, embed_dim, hidden_dim, num_heads=4, dropout=0.1, bound_output=True):
        super().__init__()
        self.bound_output = bound_output
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output_layer = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        h = self.fc(x).unsqueeze(1)
        h = self.transformer(h)
        out = self.output_layer(h.squeeze(1))
        return torch.tanh(out) if self.bound_output else out


class SingleHeadVAE(nn.Module):
    """Single-headed VAE for continuous tabular data (used for ADBench)."""
    def __init__(self, input_dim, latent_dim, context_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.use_context = context_dim > 0
        self.context_dim = context_dim
        self.latent_dim = latent_dim
        enc_in = input_dim + context_dim
        dec_in = latent_dim + context_dim
        self.encoder = self._mlp(enc_in, hidden_dim, latent_dim * 2, dropout)
        self.decoder = self._mlp(dec_in, hidden_dim, input_dim, dropout)

    @staticmethod
    def _mlp(in_dim, hid, out_dim, dropout):
        return nn.Sequential(
            nn.Linear(in_dim, hid), nn.LeakyReLU(), nn.Dropout(dropout),
            nn.Linear(hid, hid),    nn.LeakyReLU(), nn.Dropout(dropout),
            nn.Linear(hid, out_dim),
        )

    @staticmethod
    def reparameterize(mu, log_var):
        log_var = torch.clamp(log_var, -5, 5)
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def _maybe_cat(self, x, ctx):
        return torch.cat([x, ctx], dim=1) if self.use_context else x

    def encode(self, x, ctx=None):
        h = self.encoder(self._maybe_cat(x, ctx))
        mu, lv = torch.chunk(h, 2, dim=1)
        return mu, torch.clamp(lv, -5, 5)

    def decode(self, z, ctx=None):
        return self.decoder(self._maybe_cat(z, ctx))

    def forward(self, x, ctx=None):
        mu, lv = self.encode(x, ctx)
        z = self.reparameterize(mu, lv)
        return self.decode(z, ctx), mu, lv


class DualHeadVAE(nn.Module):
    """Dual-headed VAE for BLS-style data (1 numerical + N one-hot categorical).

    Salary feature is at index 0; categoricals are at indices 1..input_dim-1.
    Has separate latent dims for numerical (latent_num) and categorical (latent_cat).
    """
    def __init__(self, input_dim, latent_num, latent_cat, context_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.use_context = context_dim > 0
        self.context_dim = context_dim
        self.latent_num = latent_num
        self.latent_cat = latent_cat
        self.input_dim = input_dim
        self.cat_dim = input_dim - 1  # everything except salary
        total_latent = latent_num + latent_cat
        enc_in = input_dim + context_dim
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, hidden_dim), nn.LeakyReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, total_latent * 2),
        )
        self.dec_num = nn.Sequential(
            nn.Linear(latent_num + context_dim, hidden_dim), nn.LeakyReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.dec_cat = nn.Sequential(
            nn.Linear(latent_cat + context_dim, hidden_dim), nn.LeakyReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, self.cat_dim),
        )

    @staticmethod
    def reparameterize(mu, log_var):
        log_var = torch.clamp(log_var, -5, 5)
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def _maybe_cat(self, x, ctx):
        return torch.cat([x, ctx], dim=1) if self.use_context else x

    def encode(self, x, ctx=None):
        h = self.encoder(self._maybe_cat(x, ctx))
        mu, lv = torch.chunk(h, 2, dim=1)
        return mu, torch.clamp(lv, -5, 5)

    def decode(self, z, ctx=None):
        z_num = z[:, :self.latent_num]
        z_cat = z[:, self.latent_num:]
        z_num_in = torch.cat([z_num, ctx], dim=1) if self.use_context else z_num
        z_cat_in = torch.cat([z_cat, ctx], dim=1) if self.use_context else z_cat
        sal = self.dec_num(z_num_in).squeeze(-1)
        cats = torch.sigmoid(self.dec_cat(z_cat_in))
        return sal, cats

    def forward(self, x, ctx=None):
        mu, lv = self.encode(x, ctx)
        z = self.reparameterize(mu, lv)
        sal, cats = self.decode(z, ctx)
        return sal, cats, mu, lv


class LatentDDPM(nn.Module):
    """Latent diffusion model, optionally context-conditioned."""
    def __init__(self, latent_dim, context_dim, timesteps, hidden_dim, dropout=0.1):
        super().__init__()
        self.timesteps = timesteps
        self.use_context = context_dim > 0
        self.context_dim = context_dim
        self.latent_dim = latent_dim
        in_dim = latent_dim + context_dim + 1

        beta = torch.linspace(0.0001, 0.02, timesteps)
        alpha = torch.clamp(1.0 - beta, min=1e-5, max=1.0)
        alpha_cum = torch.maximum(torch.cumprod(alpha, dim=0), torch.tensor(1e-5))
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_cum", alpha_cum)

        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward_diffusion(self, z0, t):
        ac = self.alpha_cum[t].unsqueeze(1)
        n = torch.randn_like(z0)
        return torch.sqrt(ac) * z0 + torch.sqrt(1 - ac) * n, n

    def reverse_denoising(self, z_t, ctx, t):
        t_norm = t.view(-1, 1).float() / self.timesteps
        if self.use_context and ctx is not None:
            x_in = torch.cat([z_t, ctx, t_norm], dim=1)
        else:
            x_in = torch.cat([z_t, t_norm], dim=1)
        return 0.5 * self.network(x_in).tanh()

    @torch.no_grad()
    def sample(self, batch_size, ctx, steps, device):
        z = torch.randn(batch_size, self.latent_dim, device=device)
        for t in reversed(range(steps)):
            z = z / (torch.norm(z, p=2, dim=1, keepdim=True) + 1e-5)
            t_long = torch.full((batch_size,), t, device=device, dtype=torch.long)
            pred = self.reverse_denoising(z, ctx, t_long)
            ac = torch.clamp(self.alpha_cum[t].view(-1, 1), 1e-3, 1.0)
            z = (z - torch.sqrt(1 - ac) * pred) / (torch.sqrt(ac) + 1e-5)
            if t > 0:
                z = z + 0.05 * torch.sqrt(self.beta[t]) * torch.randn_like(z)
        return z


# =============================================================================
# Training utilities
# =============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def beta_scheduler(epoch, warmup, max_beta):
    if epoch < warmup:
        return max(0.005, (epoch / warmup) ** 2 * max_beta)
    return max_beta


def train_vae_single(vae, ctx_enc, X, epochs, batch_size, device, lr=1e-3, wd=1e-4, warmup=5, max_beta=1.0):
    params = list(vae.parameters())
    if ctx_enc is not None:
        params += list(ctx_enc.parameters())
    opt = optim.Adam(params, lr=lr, weight_decay=wd)
    X_aug = X + torch.randn_like(X) * 0.01
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_aug),
        batch_size=batch_size, shuffle=True
    )
    for epoch in range(epochs):
        vae.train()
        if ctx_enc is not None:
            ctx_enc.train()
        beta = beta_scheduler(epoch, warmup, max_beta)
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            ctx = ctx_enc(xb) if ctx_enc is not None else None
            x_rec, mu, lv = vae(xb, ctx)
            kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / xb.size(0)
            loss = F.mse_loss(x_rec, xb) + beta * kl
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
    vae.eval()
    if ctx_enc is not None:
        ctx_enc.eval()


def train_vae_dual(vae, ctx_enc, X, epochs, batch_size, device, salary_weight=2.0,
                   lr=1e-3, wd=1e-4, warmup=5, max_beta=1.0):
    params = list(vae.parameters())
    if ctx_enc is not None:
        params += list(ctx_enc.parameters())
    opt = optim.Adam(params, lr=lr, weight_decay=wd)
    X_aug = X + torch.randn_like(X) * 0.01
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_aug),
        batch_size=batch_size, shuffle=True
    )
    for epoch in range(epochs):
        vae.train()
        if ctx_enc is not None:
            ctx_enc.train()
        beta = beta_scheduler(epoch, warmup, max_beta)
        for (xb,) in loader:
            xb = xb.to(device)
            sal_t = xb[:, 0]
            cat_t = xb[:, 1:]
            opt.zero_grad()
            ctx = ctx_enc(xb) if ctx_enc is not None else None
            sal_r, cat_r, mu, lv = vae(xb, ctx)
            kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / xb.size(0)
            sal_loss = F.mse_loss(sal_r, sal_t) * salary_weight
            cat_loss = F.binary_cross_entropy(torch.clamp(cat_r, 1e-6, 1 - 1e-6), torch.clamp(cat_t, 0, 1))
            loss = sal_loss + cat_loss + beta * kl
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
    vae.eval()
    if ctx_enc is not None:
        ctx_enc.eval()


def train_diffusion(diff, vae, ctx_enc, variant, X, epochs, batch_size, device,
                    dataset_type, lr=1e-3):
    """Train the latent diffusion. variant: 'D' (encoder co-trained) or 'J' (encoder pre-trained with VAE)."""
    train_encoder = (variant == "D" and ctx_enc is not None)

    params = list(diff.parameters())
    if train_encoder:
        params += list(ctx_enc.parameters())
    opt = optim.Adam(params, lr=lr)

    X_d = X.to(device)
    n = X_d.shape[0]
    vae.eval()

    # Pre-compute latents (and context if not jointly training encoder)
    z_chunks, ctx_chunks = [], []
    use_ctx_pre = (variant == "J" and ctx_enc is not None)  # J trained encoder during VAE phase
    with torch.no_grad():
        for s in range(0, n, batch_size):
            xb = X_d[s:s + batch_size]
            ctx_vae = ctx_enc(xb) if (variant == "J" and ctx_enc is not None) else None
            if dataset_type == "bls":
                mu, lv = vae.encode(xb, ctx_vae)
            else:
                mu, lv = vae.encode(xb, ctx_vae)
            z_chunks.append(vae.reparameterize(mu, lv))
            if use_ctx_pre:
                ctx_chunks.append(ctx_enc(xb))
    z_train = torch.cat(z_chunks, dim=0)
    ctx_pre = torch.cat(ctx_chunks, dim=0) if ctx_chunks else None

    for epoch in range(epochs):
        diff.train()
        if train_encoder:
            ctx_enc.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            opt.zero_grad()
            zb = z_train[idx]
            if ctx_enc is not None:
                if train_encoder:
                    ctx = ctx_enc(X_d[idx])
                else:
                    ctx = ctx_pre[idx]
            else:
                ctx = None
            t = torch.randint(0, diff.timesteps, (zb.shape[0],), device=device)
            z_t, noise = diff.forward_diffusion(zb, t)
            pred = diff.reverse_denoising(z_t, ctx, t)
            loss = F.mse_loss(pred, noise)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
    diff.eval()
    if ctx_enc is not None:
        ctx_enc.eval()


@torch.no_grad()
def score_caldm(vae, diff, ctx_enc, variant, X, device, dataset_type, chunk_size=4096):
    """Score samples via reconstruction loss after diffusion-decode cycle."""
    X = X.to(device)
    n = X.shape[0]
    parts = []
    for s in range(0, n, chunk_size):
        xb = X[s:s + chunk_size]
        ctx_vae = ctx_enc(xb) if (variant == "J" and ctx_enc is not None) else None
        ctx_diff = ctx_enc(xb) if ctx_enc is not None else None
        z = diff.sample(batch_size=xb.shape[0], ctx=ctx_diff, steps=diff.timesteps, device=device)
        if dataset_type == "bls":
            sal_r, cat_r = vae.decode(z, ctx_vae)
            sal_t = xb[:, 0]
            cat_t = xb[:, 1:]
            sal_err = (sal_r - sal_t) ** 2
            cat_err = ((cat_r - cat_t) ** 2).mean(dim=1)
            loss = sal_err + cat_err
        else:
            x_rec = vae.decode(z, ctx_vae)
            loss = torch.mean((xb - x_rec) ** 2, dim=1)
        parts.append(torch.log1p(loss))
    return torch.cat(parts).cpu().numpy()


# =============================================================================
# CALDM hyperparameter auto-scaling (matches caldm_ablation_adbench.py)
# =============================================================================

def scale_hyperparameters(n_rows, n_features):
    hidden_dim = int(np.clip(64 * np.sqrt(n_features / 10), 64, 256))
    latent_dim = int(np.clip(n_features // 4, 4, 32))
    context_dim = int(np.clip(n_features // 4, 4, 40))
    heads = 4
    while hidden_dim % heads != 0 and heads > 1:
        heads -= 1
    if n_rows < 500:
        batch_size = max(32, n_rows // 8)
    elif n_rows < 5000:
        batch_size = 128
    else:
        batch_size = 256
    return {
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "context_dim": context_dim,
        "attention_heads": heads,
        "batch_size": batch_size,
    }


# =============================================================================
# CALDM runner — works for both BLS (dual-head) and ADBench (single-head)
# =============================================================================

def run_caldm(variant, X_train_t, X_eval_t, y_eval, hp, device, dataset_type,
              vae_epochs=15, diff_epochs=30, diff_timesteps=1000, dropout=0.1,
              salary_weight=2.0, latent_num=16, latent_cat=32, vae_max_beta=1.0):
    """variant: 'D' for CALDM-D, 'J' for CALDM-J. dataset_type: 'bls' or 'adbench'.

    vae_max_beta: Final value of the VAE KL weight after warmup. Set to 0.0 to
    disable KL regularization (matches the praxis configuration); 1.0 gives
    a standard-normal latent prior (matches V003.3 Section 4.2 / V3 ablation).
    """
    t0 = time.time()
    n_features = X_train_t.shape[1]

    # Build context encoder
    ctx_enc = ContextEncoder(
        n_features, embed_dim=hp["context_dim"],
        hidden_dim=hp["hidden_dim"], num_heads=hp["attention_heads"], dropout=dropout
    ).to(device)

    # Build VAE
    if dataset_type == "bls":
        vae = DualHeadVAE(
            n_features, latent_num=latent_num, latent_cat=latent_cat,
            context_dim=(hp["context_dim"] if variant == "J" else 0),
            hidden_dim=hp["hidden_dim"], dropout=dropout,
        ).to(device)
        vae_latent_dim = latent_num + latent_cat
    else:
        vae = SingleHeadVAE(
            n_features, latent_dim=hp["latent_dim"],
            context_dim=(hp["context_dim"] if variant == "J" else 0),
            hidden_dim=hp["hidden_dim"], dropout=dropout,
        ).to(device)
        vae_latent_dim = hp["latent_dim"]

    # Train VAE (ctx_enc joins only if J)
    ctx_for_vae = ctx_enc if variant == "J" else None
    if dataset_type == "bls":
        train_vae_dual(vae, ctx_for_vae, X_train_t, vae_epochs, hp["batch_size"], device,
                       salary_weight=salary_weight, max_beta=vae_max_beta)
    else:
        train_vae_single(vae, ctx_for_vae, X_train_t, vae_epochs, hp["batch_size"], device,
                         max_beta=vae_max_beta)

    # Build & train diffusion
    diff = LatentDDPM(
        latent_dim=vae_latent_dim,
        context_dim=hp["context_dim"],  # both D and J use ctx in diffusion
        timesteps=diff_timesteps, hidden_dim=hp["hidden_dim"], dropout=dropout,
    ).to(device)
    train_diffusion(diff, vae, ctx_enc, variant, X_train_t, diff_epochs, hp["batch_size"],
                    device, dataset_type)

    train_time = time.time() - t0

    # Score eval set
    scores = score_caldm(vae, diff, ctx_enc, variant, X_eval_t, device, dataset_type)
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"Non-finite CALDM scores: {(~np.isfinite(scores)).sum()}")

    score_time = time.time() - t0 - train_time

    return scores, train_time, score_time


# =============================================================================
# PyOD baseline runner — graceful import + fit/score
# =============================================================================

def _import_pyod_class(module_path, class_name):
    try:
        mod = __import__(module_path, fromlist=[class_name])
        return getattr(mod, class_name)
    except Exception as e:
        return None


def run_pyod_baseline(method_name, X_train, X_eval, seed):
    """Returns (scores, train_time, score_time, error)."""
    spec = PYOD_BASELINES.get(method_name)
    if spec is None:
        return None, None, None, f"Unknown method: {method_name}"
    module_path, class_name, kwargs, family, slow_threshold = spec

    # Skip slow methods on large datasets
    if slow_threshold is not None and X_train.shape[0] > slow_threshold:
        return None, None, None, f"skipped (n={X_train.shape[0]} > threshold {slow_threshold})"

    cls = _import_pyod_class(module_path, class_name)
    if cls is None:
        return None, None, None, f"could not import {class_name} from {module_path}"

    # Special handling for DeepSVDD which needs n_features
    kwargs = dict(kwargs)
    if class_name == "DeepSVDD":
        kwargs["n_features"] = X_train.shape[1]
        kwargs["random_state"] = seed
        kwargs["epochs"] = 30
    elif "random_state" in cls.__init__.__code__.co_varnames:
        kwargs["random_state"] = seed

    try:
        t0 = time.time()
        np.random.seed(seed)
        random.seed(seed)
        model = cls(**kwargs)
        model.fit(X_train)
        train_time = time.time() - t0

        t1 = time.time()
        scores = model.decision_function(X_eval)
        score_time = time.time() - t1

        if not np.all(np.isfinite(scores)):
            return None, None, None, "non-finite scores"

        return np.asarray(scores), train_time, score_time, None
    except Exception as e:
        return None, None, None, f"{type(e).__name__}: {e}"


# =============================================================================
# Metric computation
# =============================================================================

def compute_metrics(scores, y, expected_outlier_rate=None):
    """Returns metrics dict. Threshold set at expected_outlier_rate or label rate."""
    auc = roc_auc_score(y, scores)
    ap = average_precision_score(y, scores)
    if expected_outlier_rate is None:
        expected_outlier_rate = float(np.array(y).mean())
    threshold = float(np.percentile(scores, 100 * (1 - expected_outlier_rate)))
    y_pred = (scores >= threshold).astype(int)
    return {
        "auc_roc": float(auc),
        "ap": float(ap),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "threshold": threshold,
    }


# =============================================================================
# Incremental CSV append (resilience for long runs)
# =============================================================================

def append_row_to_csv(csv_path, row_dict, max_retries=6):
    """Append a single dict row to a CSV. Creates header on first call.

    Retries on transient PermissionError — common on Windows when OneDrive,
    antivirus, or Excel briefly hold the file open. Backoff: 0.5, 1, 2, 4, 8, 16s
    (~31s total before giving up). If all retries fail, prints a warning and
    returns without crashing the run.
    """
    df_row = pd.DataFrame([row_dict])
    last_err = None
    for attempt in range(max_retries):
        try:
            if os.path.exists(csv_path):
                df_row.to_csv(csv_path, mode="a", header=False, index=False)
            else:
                df_row.to_csv(csv_path, mode="w", header=True, index=False)
            return  # success
        except PermissionError as e:
            last_err = e
            wait = 0.5 * (2 ** attempt)  # 0.5, 1, 2, 4, 8, 16
            print(f"\n  [csv-write retry {attempt+1}/{max_retries}] file locked "
                  f"(close any program viewing it); waiting {wait:.1f}s...", flush=True)
            time.sleep(wait)
    # All retries failed — log but DO NOT crash the run
    print(f"\n  [csv-write FAILED after {max_retries} retries] {last_err}")
    print(f"  [csv-write] lost row: dataset={row_dict.get('dataset','?')}  "
          f"method={row_dict.get('method','?')}  seed={row_dict.get('seed','?')}", flush=True)


def _load_completed_combos(per_run_path, key_cols):
    """Read the existing per_run CSV and return a set of completed-row keys.

    Used by resume=True. A row counts as completed if its 'error' column is
    empty/NaN and its 'auc_roc' value is finite.
    """
    if not os.path.exists(per_run_path):
        return set()
    try:
        df = pd.read_csv(per_run_path)
        # Treat NaN error as "no error"
        err_clean = df.get("error", pd.Series([""] * len(df))).fillna("")
        # Done = no error AND auc_roc is finite
        if "auc_roc" in df.columns:
            done_mask = (err_clean == "") & df["auc_roc"].notna()
        else:
            done_mask = err_clean == ""
        df_done = df[done_mask]
        avail = [c for c in key_cols if c in df_done.columns]
        if len(avail) != len(key_cols):
            return set()
        return set(zip(*[df_done[c].astype(type(df_done[c].iloc[0]) if len(df_done) else str) for c in avail])) if len(df_done) else set()
    except Exception as e:
        print(f"  [resume] could not read existing CSV ({e}); starting fresh")
        return set()


# =============================================================================
# Method registry helpers
# =============================================================================

def all_method_names(include_caldm=True):
    methods = list(PYOD_BASELINES.keys())
    if include_caldm:
        methods += ["CALDM-D", "CALDM-J"]
    return methods


def method_metadata():
    rows = []
    for name, (mod, cls, kw, fam, slow) in PYOD_BASELINES.items():
        rows.append({
            "method": name, "library": "pyod", "module": mod, "class": cls,
            "family": fam, "slow_threshold_rows": slow if slow else "",
            "kwargs": json.dumps(kw),
        })
    rows.append({"method": "CALDM-D", "library": "caldm", "module": "caldm_benchmark",
                 "class": "CALDM (diffusion-stage context)", "family": "deep_diffusion",
                 "slow_threshold_rows": "", "kwargs": ""})
    rows.append({"method": "CALDM-J", "library": "caldm", "module": "caldm_benchmark",
                 "class": "CALDM (joint VAE+diffusion context)", "family": "deep_diffusion",
                 "slow_threshold_rows": "", "kwargs": ""})
    return pd.DataFrame(rows)


# =============================================================================
# BLS data loading
# =============================================================================

def load_bls(data_path, subset_pct=1.0, seed=42, test_size=0.2, verbose=True):
    """Load BLS .npz, optionally sub-sample, stratified 80:20 split.

    Handles many common BLS label encodings:
      - 1D string array of types ('NONE', 'HRS', ...)
      - 1D numeric binary
      - 2D numeric (n, 2): col 0 = binary, col 1 = type code
      - 2D string (n, k): autodetects which col is binary vs type, or falls back
        to multi-label "any non-NONE = outlier"
      - 2D numeric multi-hot

    Returns: X_train_t, X_test_t, y_train, y_test, y_type_test, meta
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"BLS data not found: {data_path}")

    data = np.load(data_path, allow_pickle=True)
    if "X" not in data or "y" not in data:
        raise ValueError(f"Expected 'X' and 'y' keys in {data_path}")

    X = np.asarray(data["X"], dtype=np.float32)
    y_raw = np.asarray(data["y"])
    n_X = X.shape[0]

    if verbose:
        sample_y = y_raw[0] if y_raw.ndim >= 1 and len(y_raw) > 0 else y_raw
        print(f"  [load_bls] X.shape={X.shape}  y.shape={y_raw.shape}  "
              f"y.dtype={y_raw.dtype}  y[0]={sample_y!r}")

    # Normalize to 1-D y (binary) and y_type (int code) of length n_X
    if y_raw.ndim == 2:
        n_y, k = y_raw.shape
        if n_y != n_X:
            raise ValueError(
                f"X has {n_X} rows but y has {n_y} rows along axis 0 — must match. "
                f"y.shape={y_raw.shape}"
            )

        if y_raw.dtype.kind in ("U", "S", "O"):
            # 2D string: try to identify a binary column and a type column
            y_str = y_raw.astype(str)
            upper = np.char.upper(y_str)
            BINARY_TRUE = {"1", "TRUE", "T", "Y", "YES", "OUTLIER"}
            BINARY_FALSE = {"0", "FALSE", "F", "N", "NO", "NORMAL", "NONE"}
            TYPE_HINTS = {"NONE", "HRS", "LRS", "INA", "PVO", "PER"}

            binary_col = None
            for c in range(k):
                col_vals = set(upper[:, c])
                if col_vals.issubset(BINARY_TRUE | BINARY_FALSE) and \
                   not col_vals.issubset({"NONE"}):
                    binary_col = c
                    break

            type_col = None
            for c in range(k):
                col_vals = set(upper[:, c])
                if col_vals & TYPE_HINTS and len(col_vals) >= 2:
                    type_col = c
                    break

            if binary_col is not None and type_col is not None and binary_col != type_col:
                bin_v = upper[:, binary_col]
                y = np.array([1 if v in BINARY_TRUE else 0 for v in bin_v], dtype=int)
                type_v = y_str[:, type_col]
                unique_types = sorted(np.unique(type_v), key=lambda s: (s.upper() != "NONE", s))
                t2c = {t: i for i, t in enumerate(unique_types)}
                y_type = np.array([t2c[t] for t in type_v], dtype=int)
                bls_label_format = f"2D strings (n,{k}) — col {binary_col}=binary, col {type_col}=type"
            elif type_col is not None:
                type_v = y_str[:, type_col]
                y = (np.char.upper(type_v) != "NONE").astype(int)
                unique_types = sorted(np.unique(type_v), key=lambda s: (s.upper() != "NONE", s))
                t2c = {t: i for i, t in enumerate(unique_types)}
                y_type = np.array([t2c[t] for t in type_v], dtype=int)
                bls_label_format = f"2D strings (n,{k}) — col {type_col} = type, binary derived from != NONE"
            else:
                # Multi-label fallback: any column non-NONE = outlier
                is_out = np.any(upper != "NONE", axis=1)
                y = is_out.astype(int)
                primary = []
                for row in y_str:
                    nn = [t for t in row if t.upper() != "NONE"]
                    primary.append(nn[0] if nn else "NONE")
                primary = np.array(primary)
                unique_types = sorted(np.unique(primary), key=lambda s: (s.upper() != "NONE", s))
                t2c = {t: i for i, t in enumerate(unique_types)}
                y_type = np.array([t2c[t] for t in primary], dtype=int)
                bls_label_format = f"2D strings (n,{k}) — multi-label, any-non-NONE = outlier"

        elif k == 2:
            y = y_raw[:, 0].astype(int)
            y_type = y_raw[:, 1].astype(int)
            bls_label_format = "2-col numeric (binary, type)"
        else:
            y = (y_raw.sum(axis=1) > 0).astype(int)
            y_type = np.argmax(y_raw, axis=1).astype(int)
            bls_label_format = f"2D numeric (n,{k}) multi-hot — argmax used as type"

    elif y_raw.ndim == 1:
        if y_raw.dtype.kind in ("U", "S", "O"):
            y_str = y_raw.astype(str)
            y = (np.char.upper(y_str) != "NONE").astype(int)
            unique_types = sorted(np.unique(y_str), key=lambda s: (s.upper() != "NONE", s))
            t2c = {t: i for i, t in enumerate(unique_types)}
            y_type = np.array([t2c[t] for t in y_str], dtype=int)
            bls_label_format = f"1D strings ({', '.join(unique_types)})"
        else:
            y = y_raw.astype(int)
            if "y_type" in data:
                yt = np.asarray(data["y_type"])
                if yt.dtype.kind in ("U", "S", "O"):
                    _, y_type = np.unique(yt.astype(str), return_inverse=True)
                    y_type = y_type.astype(int)
                else:
                    y_type = yt.astype(int).flatten()
            else:
                y_type = y.copy()
            bls_label_format = "1D numeric binary"
    else:
        raise ValueError(f"Unexpected y.ndim={y_raw.ndim}, shape={y_raw.shape}")

    if len(y) != n_X or len(y_type) != n_X:
        raise ValueError(
            f"After parsing: X={n_X}, y={len(y)}, y_type={len(y_type)} — lengths must match. "
            f"Original y.shape={y_raw.shape}, y.dtype={y_raw.dtype}"
        )

    if verbose:
        print(f"  [load_bls] format detected: {bls_label_format}")
        print(f"  [load_bls] outlier rate: {y.mean():.4f}  type codes: {sorted(np.unique(y_type).tolist())}")

    # Optional stratified sub-sample, then stratified train/test split
    rng = np.random.RandomState(seed)
    if subset_pct < 1.0:
        idx_keep = []
        for t in np.unique(y_type):
            t_idx = np.where(y_type == t)[0]
            n_keep = max(1, int(len(t_idx) * subset_pct))
            idx_keep.append(rng.choice(t_idx, size=n_keep, replace=False))
        idx_keep = np.concatenate(idx_keep)
        rng.shuffle(idx_keep)
        X = X[idx_keep]
        y = y[idx_keep]
        y_type = y_type[idx_keep]

    counts = np.bincount(y_type)
    stratify_on = y_type if counts.min() >= 2 else y

    X_tr, X_te, y_tr, y_te, yt_tr, yt_te = train_test_split(
        X, y, y_type, test_size=test_size, random_state=seed, stratify=stratify_on
    )

    scaler = StandardScaler()
    X_tr_s = X_tr.copy()
    X_te_s = X_te.copy()
    X_tr_s[:, :1] = scaler.fit_transform(X_tr[:, :1])
    X_te_s[:, :1] = scaler.transform(X_te[:, :1])

    X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32)
    X_te_t = torch.tensor(X_te_s, dtype=torch.float32)

    meta = {
        "n_total": int(len(X)), "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "n_features": int(X.shape[1]),
        "outlier_rate_total": float(y.mean()),
        "outlier_rate_train": float(y_tr.mean()),
        "outlier_rate_test": float(y_te.mean()),
        "subset_pct": float(subset_pct),
        "label_format": bls_label_format,
    }
    return X_tr_t, X_te_t, y_tr, y_te, yt_te, meta


# =============================================================================
# ADBench data loading
# =============================================================================

def load_adbench_dataset(filepath):
    """Load .npz, return (X_t, y, meta) for transductive evaluation."""
    try:
        data = np.load(filepath, allow_pickle=True)
        if "X" not in data or "y" not in data:
            return None, "missing X or y"
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"]).astype(int).flatten()
    except Exception as e:
        return None, f"load error: {e}"

    if X.shape[0] < 100:
        return None, f"too few rows ({X.shape[0]} < 100)"

    outlier_rate = float(y.mean())
    if outlier_rate <= 0.001 or outlier_rate >= 0.5:
        return None, f"degenerate outlier rate ({outlier_rate:.4f})"

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    X_t = torch.tensor(X_std, dtype=torch.float32)

    meta = {
        "n_rows": int(X.shape[0]), "n_features": int(X.shape[1]),
        "outlier_rate": outlier_rate,
        "n_outliers": int(y.sum()),
    }
    return (X_t, y, meta), None


# =============================================================================
# Run a method on a single (X_train, X_eval, y_eval) — central dispatch
# =============================================================================

def run_method(method_name, X_train_t, X_eval_t, y_eval, seed, hp, device, dataset_type,
               vae_epochs, diff_epochs, vae_max_beta=1.0):
    """Run any method (PyOD or CALDM) and return metrics + timing.

    PyOD methods: train on X_train (numpy), score on X_eval (numpy).
    CALDM methods: train on X_train_t (torch), score on X_eval_t (torch).
    """
    out = {"method": method_name, "seed": seed,
           "auc_roc": np.nan, "ap": np.nan, "f1": np.nan,
           "precision": np.nan, "recall": np.nan, "accuracy": np.nan,
           "threshold": np.nan, "train_time_sec": np.nan, "score_time_sec": np.nan,
           "error": ""}

    try:
        if method_name in ("CALDM-D", "CALDM-J"):
            variant = method_name.split("-")[1]
            scores, t_train, t_score = run_caldm(
                variant, X_train_t, X_eval_t, y_eval, hp, device, dataset_type,
                vae_epochs=vae_epochs, diff_epochs=diff_epochs,
                vae_max_beta=vae_max_beta,
            )
            out.update(compute_metrics(scores, y_eval))
            out["train_time_sec"] = t_train
            out["score_time_sec"] = t_score
        else:
            X_tr_np = X_train_t.cpu().numpy() if isinstance(X_train_t, torch.Tensor) else X_train_t
            X_ev_np = X_eval_t.cpu().numpy() if isinstance(X_eval_t, torch.Tensor) else X_eval_t
            scores, t_train, t_score, err = run_pyod_baseline(method_name, X_tr_np, X_ev_np, seed)
            if err is not None:
                out["error"] = err
            else:
                out.update(compute_metrics(scores, y_eval))
                out["train_time_sec"] = t_train
                out["score_time_sec"] = t_score
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        if os.environ.get("CALDM_DEBUG"):
            traceback.print_exc()
    return out


# =============================================================================
# BLS Benchmark
# =============================================================================

def run_bls_benchmark(data_path, output_dir="bls_outputs",
                     subset_pct=1.0, seeds=None, methods=None,
                     vae_epochs=50, diff_epochs=200, vae_max_beta=0.0,
                     resume=False,
                     quick=False, verbose=True):
    """Run all (or specified) methods on the BLS dataset.

    Args:
        data_path: path to BLS .npz file
        output_dir: where to save CSVs
        subset_pct: fraction of dataset to use (0.0..1.0); 1.0 = full
        seeds: list of random seeds (default: [42, 7, 123])
        methods: subset of method names; default = all 13 baselines + CALDM-D + CALDM-J
        vae_epochs / diff_epochs: CALDM training epochs
        quick: smoke-test mode (1 seed, 5 epochs, only fast baselines)

    Returns:
        dict with 'per_run' (DataFrame), 'summary' (DataFrame), 'meta' (dict)
    """
    if quick:
        seeds = [42]
        vae_epochs = 5
        diff_epochs = 5
        methods = methods or ["IForest", "PCA", "HBOS", "ECOD", "COPOD", "CALDM-D", "CALDM-J"]
        if verbose:
            print("[setup] QUICK mode: 1 seed, 5 epochs, fast baselines only")

    if seeds is None:
        seeds = list(DEFAULT_SEEDS)
    if methods is None:
        methods = all_method_names(include_caldm=True)

    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"[setup] device={DEVICE}  seeds={seeds}  subset_pct={subset_pct}")
        print(f"[setup] methods ({len(methods)}): {', '.join(methods)}")

    records = []
    overall_t0 = time.time()

    per_run_path = os.path.join(output_dir, "bls_per_run.csv")
    # Resume: keep existing CSV and skip already-completed (method, seed) combos
    # Fresh: wipe existing CSV (default behavior)
    completed = set()
    if resume and os.path.exists(per_run_path):
        completed = _load_completed_combos(per_run_path, ["method", "seed"])
        if verbose:
            print(f"[resume] {len(completed)} previously completed (method, seed) combos found in {per_run_path}")
    elif os.path.exists(per_run_path):
        os.remove(per_run_path)
        if verbose:
            print(f"[setup] removed existing per_run.csv (resume=False)")
    if verbose:
        print(f"[setup] writing rows incrementally to {per_run_path}")

    for seed in seeds:
        set_seed(seed)
        # Load data fresh per seed so the stratified sub-sample re-runs
        X_tr, X_te, y_tr, y_te, yt_te, meta = load_bls(data_path, subset_pct, seed=seed)
        n_features = X_tr.shape[1]
        hp = scale_hyperparameters(X_tr.shape[0], n_features)

        if verbose:
            print(f"\n=== seed={seed} ===")
            print(f"  train={meta['n_train']}  test={meta['n_test']}  "
                  f"features={n_features}  outlier_rate_test={meta['outlier_rate_test']:.4f}")
            print(f"  label format detected: {meta['label_format']}")
            print(f"  CALDM hp: {hp}")

        for m in methods:
            if (m, seed) in completed:
                if verbose:
                    print(f"  [skip] {m} seed={seed} (already in CSV)")
                continue
            if verbose:
                print(f"  [running] {m}", end=" ", flush=True)
            t0 = time.time()
            rec = run_method(m, X_tr, X_te, y_te, seed, hp, DEVICE, dataset_type="bls",
                             vae_epochs=vae_epochs, diff_epochs=diff_epochs,
                             vae_max_beta=vae_max_beta)
            rec["dataset"] = "BLS"
            rec["n_train"] = meta["n_train"]
            rec["n_test"] = meta["n_test"]
            rec["n_features"] = n_features
            rec["outlier_rate"] = meta["outlier_rate_test"]
            rec["subset_pct"] = subset_pct
            records.append(rec)
            # ----- incremental write so an overnight crash doesn't lose progress
            append_row_to_csv(per_run_path, rec)
            elapsed = time.time() - t0
            if verbose:
                if rec["error"]:
                    print(f"  ERROR: {rec['error']} ({elapsed:.1f}s)")
                else:
                    print(f"  AUC={rec['auc_roc']:.4f}  AP={rec['ap']:.4f}  ({elapsed:.1f}s)")

    total_time = time.time() - overall_t0
    if verbose:
        print(f"\n[done] BLS total time: {total_time/60:.1f} min")

    # Read full per_run from disk (covers both fresh and resumed runs)
    if os.path.exists(per_run_path):
        df = pd.read_csv(per_run_path)
    else:
        df = pd.DataFrame(records)

    # Summary: per method, mean ± std across seeds
    summary = (df.groupby("method")
                 .agg(auc_roc_mean=("auc_roc", "mean"), auc_roc_std=("auc_roc", "std"),
                      ap_mean=("ap", "mean"), ap_std=("ap", "std"),
                      f1_mean=("f1", "mean"), f1_std=("f1", "std"),
                      precision_mean=("precision", "mean"), precision_std=("precision", "std"),
                      recall_mean=("recall", "mean"), recall_std=("recall", "std"),
                      train_time_mean=("train_time_sec", "mean"),
                      n_seeds_run=("seed", "count"),
                      n_seeds_with_error=("error", lambda s: int((s != "").sum())))
                 .reset_index()
                 .sort_values("auc_roc_mean", ascending=False, na_position="last"))
    summary.to_csv(os.path.join(output_dir, "bls_summary.csv"), index=False)

    if verbose:
        show = summary[["method", "auc_roc_mean", "auc_roc_std", "ap_mean", "ap_std"]]
        print("\n" + "=" * 72 + "\nBLS Summary (sorted by AUROC)\n" + "=" * 72)
        print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return {"per_run": df, "summary": summary,
            "meta": {"total_time_sec": total_time, "subset_pct": subset_pct,
                     "seeds": seeds, "methods": methods,
                     "vae_epochs": vae_epochs, "diff_epochs": diff_epochs}}


# =============================================================================
# ADBench Benchmark
# =============================================================================

def run_adbench_benchmark(input_dir, output_dir="adbench_outputs",
                          seeds=None, methods=None,
                          vae_epochs=50, diff_epochs=200, vae_max_beta=0.0,
                          dataset_filter=None,
                          resume=False,
                          quick=False, verbose=True):
    """Run all (or specified) methods on ADBench datasets (transductive eval).

    Args:
        input_dir: folder of ADBench .npz files
        output_dir: where to save CSVs
        seeds: list of random seeds
        methods: subset of methods
        dataset_filter: list of substrings; only run datasets matching one
        quick: smoke-test mode

    Returns:
        dict with 'per_run', 'per_dataset', 'summary', 'meta'
    """
    if quick:
        seeds = [42]
        vae_epochs = 5
        diff_epochs = 5
        methods = methods or ["IForest", "PCA", "HBOS", "ECOD", "COPOD", "CALDM-D", "CALDM-J"]
        if verbose:
            print("[setup] QUICK mode: 1 seed, 5 epochs, fast baselines only")

    if seeds is None:
        seeds = list(DEFAULT_SEEDS)
    if methods is None:
        methods = all_method_names(include_caldm=True)

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"ADBench dir not found: {input_dir}")

    npz_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".npz")])
    if dataset_filter:
        npz_files = [f for f in npz_files if any(s.lower() in f.lower() for s in dataset_filter)]

    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"[setup] device={DEVICE}  seeds={seeds}")
        print(f"[setup] datasets: {len(npz_files)}  methods: {len(methods)}")

    records = []
    skipped = []
    overall_t0 = time.time()

    per_run_path = os.path.join(output_dir, "adbench_per_run.csv")
    completed = set()
    if resume and os.path.exists(per_run_path):
        completed = _load_completed_combos(per_run_path, ["dataset", "method", "seed"])
        if verbose:
            print(f"[resume] {len(completed)} previously completed (dataset, method, seed) combos found")
    elif os.path.exists(per_run_path):
        os.remove(per_run_path)
        if verbose:
            print(f"[setup] removed existing per_run.csv (resume=False)")
    if verbose:
        print(f"[setup] writing rows incrementally to {per_run_path}")

    for fi, fname in enumerate(npz_files):
        ds_name = os.path.splitext(fname)[0]
        filepath = os.path.join(input_dir, fname)

        loaded, err = load_adbench_dataset(filepath)
        if loaded is None:
            if verbose:
                print(f"\n[skip] {ds_name}: {err}")
            skipped.append({"dataset": ds_name, "reason": err})
            continue

        X_t, y, meta = loaded
        hp = scale_hyperparameters(meta["n_rows"], meta["n_features"])

        if verbose:
            print(f"\n[{fi+1}/{len(npz_files)}] {ds_name}  "
                  f"({meta['n_rows']} rows, {meta['n_features']} features, "
                  f"{meta['outlier_rate']*100:.2f}% outliers)")

        for m in methods:
            for seed in seeds:
                if (ds_name, m, seed) in completed:
                    if verbose:
                        print(f"  [skip] {m} seed={seed} (already in CSV)")
                    continue
                set_seed(seed)
                if verbose:
                    print(f"  [run] {m} seed={seed}", end=" ", flush=True)
                t0 = time.time()
                # Transductive: train and eval on the same X
                rec = run_method(m, X_t, X_t, y, seed, hp, DEVICE, dataset_type="adbench",
                                 vae_epochs=vae_epochs, diff_epochs=diff_epochs,
                                 vae_max_beta=vae_max_beta)
                rec["dataset"] = ds_name
                rec["n_rows"] = meta["n_rows"]
                rec["n_features"] = meta["n_features"]
                rec["outlier_rate"] = meta["outlier_rate"]
                records.append(rec)
                # ----- incremental write: each (dataset, method, seed) row is durable
                append_row_to_csv(per_run_path, rec)
                elapsed = time.time() - t0
                if verbose:
                    if rec["error"]:
                        print(f"  {rec['error'][:50]} ({elapsed:.1f}s)")
                    else:
                        print(f"  AUC={rec['auc_roc']:.3f}  AP={rec['ap']:.3f}  ({elapsed:.1f}s)")

    total_time = time.time() - overall_t0
    if verbose:
        print(f"\n[done] ADBench total time: {total_time/60:.1f} min")

    # Read full per_run from disk (covers both fresh and resumed runs)
    if os.path.exists(per_run_path):
        df = pd.read_csv(per_run_path)
    else:
        df = pd.DataFrame(records)
    if skipped:
        pd.DataFrame(skipped).to_csv(os.path.join(output_dir, "adbench_skipped.csv"), index=False)

    if df.empty:
        return {"per_run": df, "per_dataset": pd.DataFrame(), "summary": pd.DataFrame(),
                "meta": {"total_time_sec": total_time}}

    # Per-dataset (aggregated across seeds)
    per_ds = (df.groupby(["dataset", "method"])
                .agg(auc_roc_mean=("auc_roc", "mean"), auc_roc_std=("auc_roc", "std"),
                     ap_mean=("ap", "mean"), ap_std=("ap", "std"),
                     f1_mean=("f1", "mean"), f1_std=("f1", "std"),
                     n_seeds=("seed", "count"),
                     n_errors=("error", lambda s: int((s != "").sum())),
                     n_rows=("n_rows", "first"), n_features=("n_features", "first"),
                     outlier_rate=("outlier_rate", "first"))
                .reset_index())
    per_ds.to_csv(os.path.join(output_dir, "adbench_per_dataset.csv"), index=False)

    # Summary across all datasets per method
    summary_rows = []
    for m in methods:
        m_data = per_ds[per_ds["method"] == m]
        if m_data.empty:
            continue
        # Win counts: count datasets where this method has highest AUROC mean
        wins_auc = 0
        wins_ap = 0
        for ds in per_ds["dataset"].unique():
            sub = per_ds[per_ds["dataset"] == ds]
            if not sub.empty and not sub["auc_roc_mean"].isna().all():
                if sub.loc[sub["auc_roc_mean"].idxmax(), "method"] == m:
                    wins_auc += 1
            if not sub.empty and not sub["ap_mean"].isna().all():
                if sub.loc[sub["ap_mean"].idxmax(), "method"] == m:
                    wins_ap += 1

        summary_rows.append({
            "method": m,
            "n_datasets": int(m_data.shape[0]),
            "auc_roc_mean": float(m_data["auc_roc_mean"].mean()),
            "auc_roc_std": float(m_data["auc_roc_mean"].std()),
            "ap_mean": float(m_data["ap_mean"].mean()),
            "ap_std": float(m_data["ap_mean"].std()),
            "wins_auc_roc": wins_auc,
            "wins_ap": wins_ap,
            "n_errors_total": int(m_data["n_errors"].sum()),
        })
    summary = pd.DataFrame(summary_rows).sort_values("auc_roc_mean", ascending=False, na_position="last")
    summary.to_csv(os.path.join(output_dir, "adbench_summary.csv"), index=False)

    if verbose:
        print("\n" + "=" * 80 + "\nADBench Summary (sorted by AUROC)\n" + "=" * 80)
        print(summary[["method", "n_datasets", "auc_roc_mean", "auc_roc_std",
                       "ap_mean", "wins_auc_roc", "wins_ap"]].to_string(
                       index=False, float_format=lambda x: f"{x:.4f}"))

    return {"per_run": df, "per_dataset": per_ds, "summary": summary,
            "meta": {"total_time_sec": total_time, "n_datasets": len(npz_files),
                     "n_skipped": len(skipped), "seeds": seeds, "methods": methods,
                     "vae_epochs": vae_epochs, "diff_epochs": diff_epochs}}


# =============================================================================
# Run-manifest writer & methods-metadata writer
# =============================================================================

def write_run_manifest(output_dir, run_info):
    """Write/append a run record to run_manifest.csv."""
    manifest_path = os.path.join(output_dir, "run_manifest.csv")
    row = {"timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z", **run_info}
    if os.path.exists(manifest_path):
        existing = pd.read_csv(manifest_path)
        df = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(manifest_path, index=False)


def write_methods_metadata(output_dir):
    method_metadata().to_csv(os.path.join(output_dir, "methods_metadata.csv"), index=False)


# =============================================================================
# Convenience: run both
# =============================================================================

def run_full_benchmark(bls_path, adbench_dir, output_dir="benchmark_outputs",
                      bls_subset_pct=1.0, seeds=None, methods=None,
                      vae_epochs=50, diff_epochs=200, vae_max_beta=0.0,
                      quick=False, verbose=True):
    """Run BLS + ADBench end-to-end. Outputs land in <output_dir>/{bls,adbench}/."""
    os.makedirs(output_dir, exist_ok=True)
    write_methods_metadata(output_dir)

    bls_dir = os.path.join(output_dir, "bls")
    adb_dir = os.path.join(output_dir, "adbench")

    bls = run_bls_benchmark(
        bls_path, output_dir=bls_dir, subset_pct=bls_subset_pct,
        seeds=seeds, methods=methods, vae_epochs=vae_epochs, diff_epochs=diff_epochs,
        vae_max_beta=vae_max_beta,
        quick=quick, verbose=verbose,
    )
    adb = run_adbench_benchmark(
        adbench_dir, output_dir=adb_dir,
        seeds=seeds, methods=methods, vae_epochs=vae_epochs, diff_epochs=diff_epochs,
        vae_max_beta=vae_max_beta,
        quick=quick, verbose=verbose,
    )

    write_run_manifest(output_dir, {
        "bls_path": bls_path, "adbench_dir": adbench_dir,
        "bls_subset_pct": bls_subset_pct, "vae_epochs": vae_epochs, "diff_epochs": diff_epochs,
        "vae_max_beta": vae_max_beta,
        "seeds": ";".join(map(str, seeds or DEFAULT_SEEDS)),
        "n_methods": len(methods) if methods else len(all_method_names()),
        "bls_total_time_sec": bls["meta"]["total_time_sec"],
        "adbench_total_time_sec": adb["meta"]["total_time_sec"],
        "device": str(DEVICE),
    })

    return {"bls": bls, "adbench": adb}


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="CALDM benchmark")
    p.add_argument("--target", choices=["bls", "adbench", "both"], default="both",
                   help="What to run.")
    p.add_argument("--bls_path", type=str, default=None, help="Path to BLS .npz")
    p.add_argument("--adbench_dir", type=str, default=None, help="Folder of ADBench .npz files")
    p.add_argument("--output_dir", type=str, default="benchmark_outputs")
    p.add_argument("--bls_subset_pct", type=float, default=1.0)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--methods", type=str, nargs="+", default=None,
                   help="Subset of methods; default = all")
    p.add_argument("--vae_epochs", type=int, default=50)
    p.add_argument("--diff_epochs", type=int, default=200)
    p.add_argument("--vae_max_beta", type=float, default=0.0,
                   help="VAE KL weight after warmup. 0.0 = praxis (autoencoder), 1.0 = standard VAE.")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    verbose = not args.quiet

    if args.target in ("both", "bls"):
        if args.bls_path is None:
            raise SystemExit("--bls_path is required for target=bls/both")
    if args.target in ("both", "adbench"):
        if args.adbench_dir is None:
            raise SystemExit("--adbench_dir is required for target=adbench/both")

    if args.target == "both":
        run_full_benchmark(
            bls_path=args.bls_path, adbench_dir=args.adbench_dir,
            output_dir=args.output_dir, bls_subset_pct=args.bls_subset_pct,
            seeds=args.seeds, methods=args.methods,
            vae_epochs=args.vae_epochs, diff_epochs=args.diff_epochs,
            vae_max_beta=args.vae_max_beta,
            quick=args.quick, verbose=verbose,
        )
    elif args.target == "bls":
        os.makedirs(args.output_dir, exist_ok=True)
        write_methods_metadata(args.output_dir)
        run_bls_benchmark(
            args.bls_path, output_dir=args.output_dir,
            subset_pct=args.bls_subset_pct,
            seeds=args.seeds, methods=args.methods,
            vae_epochs=args.vae_epochs, diff_epochs=args.diff_epochs,
            vae_max_beta=args.vae_max_beta,
            quick=args.quick, verbose=verbose,
        )
    else:  # adbench
        os.makedirs(args.output_dir, exist_ok=True)
        write_methods_metadata(args.output_dir)
        run_adbench_benchmark(
            args.adbench_dir, output_dir=args.output_dir,
            seeds=args.seeds, methods=args.methods,
            vae_epochs=args.vae_epochs, diff_epochs=args.diff_epochs,
            vae_max_beta=args.vae_max_beta,
            quick=args.quick, verbose=verbose,
        )


if __name__ == "__main__":
    main()
