"""
CALDM reviewer checks (ESWA revision).

Runs three sanity checks on the BLS ablation setup, using exactly the same
20% sub-sample, split and test set as caldm_ablation_study_v3.py:

  M1. Simple salary baselines (no learning):
        - salary_abs_z  : |standardised salary|
        - cohort_abs_z  : |salary z-score within the record's exact categorical
                          profile|, computed from train+val; falls back to the
                          global z-score for profiles with < MIN_COHORT records.
  M2. CALDM-D with the context encoder fed categorical features only
      (salary removed from the context input), 5 seeds.
  K.  Batch dependence of the context encoder: the encoder attends across the
      records in a scoring chunk. For one trained CALDM-D model we compare
      context vectors and anomaly scores computed in the original test order
      vs. a shuffled order (different chunk membership).

Usage (from the repo root, GPU recommended):
    python scripts/caldm_reviewer_checks.py --data ../data/final_data_subsetFLGAAL8.npz

Outputs go to results/reviewer_checks/.
Author: Juan (with Claude)
"""
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import torch
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import caldm_ablation_study_v3 as ab  # noqa: E402

MIN_COHORT = 5


# ---------------------------------------------------------------------------
# M1 — simple salary baselines
# ---------------------------------------------------------------------------
def cohort_abs_z(X_fit, X_test):
    sal_fit = X_fit[:, 0].numpy().astype(np.float64)
    sal_te = X_test[:, 0].numpy().astype(np.float64)
    key_fit = [r.tobytes() for r in (X_fit[:, 1:].numpy() > 0.5)]
    key_te = [r.tobytes() for r in (X_test[:, 1:].numpy() > 0.5)]
    df = pd.DataFrame({"k": key_fit, "s": sal_fit})
    g = df.groupby("k")["s"].agg(["mean", "std", "count"])
    mu_g, sd_g = sal_fit.mean(), sal_fit.std()
    z = np.empty(len(sal_te))
    n_fallback = 0
    for i, (k, s) in enumerate(zip(key_te, sal_te)):
        if k in g.index:
            m, sd, c = g.loc[k]
            if c >= MIN_COHORT and sd > 1e-8:
                z[i] = (s - m) / sd
                continue
        z[i] = (s - mu_g) / sd_g
        n_fallback += 1
    return np.abs(z), n_fallback, len(g)


# ---------------------------------------------------------------------------
# M2 — CALDM-D with categorical-only context
# ---------------------------------------------------------------------------
_OrigEncoder = ab.ContextEncoder


class CatOnlyContextEncoder(_OrigEncoder):
    """Same encoder, but it never sees column 0 (salary)."""
    def __init__(self, input_dim, **kw):
        super().__init__(input_dim - 1, **kw)

    def forward(self, x):
        return super().forward(x[:, 1:])


def config_by_name(name):
    return next(c for c in ab.CONFIGS if c.name == name)


# ---------------------------------------------------------------------------
# K — chunk / batch dependence
# ---------------------------------------------------------------------------
@torch.no_grad()
def contexts_in_order(enc, X, order, chunk):
    Xo = X[order].to(ab.DEVICE)
    out = torch.cat([enc(Xo[s:s + chunk]) for s in range(0, len(Xo), chunk)]).cpu()
    res = torch.empty_like(out)
    res[torch.as_tensor(order)] = out
    return res


def train_caldm_d(X_train, X_val, input_dim, seed):
    """Mirror of run_single_experiment for config E, returning the models."""
    cfg = config_by_name("E_cxt_in_diff")
    ab.set_seed(seed)
    enc = ab.ContextEncoder(input_dim, embed_dim=ab.CONTEXT_DIM, hidden_dim=ab.HIDDEN_DIM,
                            num_heads=ab.ATTENTION_HEADS, dropout_rate=ab.DROPOUT_RATE).to(ab.DEVICE)
    vae = ab.FlexibleVAE(input_dim, ab.LATENT_DIM_NUM, ab.LATENT_DIM_CAT, 0,
                         ab.HIDDEN_DIM, ab.DROPOUT_RATE).to(ab.DEVICE)
    ab.train_vae(vae, None, X_train, X_val, ab.VAE_EPOCHS, ab.VAE_BATCH_SIZE, ab.DEVICE)
    diff = ab.FlexibleLatentDiffusion(ab.LATENT_DIM_NUM, ab.LATENT_DIM_CAT, ab.CONTEXT_DIM,
                                      ab.DIFF_TIMESTEPS, ab.HIDDEN_DIM, ab.DROPOUT_RATE).to(ab.DEVICE)
    ab.train_latent_diffusion(diff, vae, enc, cfg, X_train, ab.DIFF_EPOCHS, ab.DEVICE)
    return cfg, enc, vae, diff


def score_with_order(cfg, enc, vae, diff, X_test, order, noise_seed):
    torch.manual_seed(noise_seed)
    s = ab.score_latent_pipeline(vae, diff, enc, cfg, X_test[order], ab.DEVICE)
    out = np.empty_like(s)
    out[order] = s
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=ab.DATA_PATH)
    p.add_argument("--out", default="results/reviewer_checks")
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 888, 1337, 2024])
    p.add_argument("--diff_epochs", type=int, default=None,
                   help="Default: the ablation script default (30). Use 200 to match the paper text.")
    p.add_argument("--vae_max_beta", type=float, default=None,
                   help="Default: the ablation script default (1.0). Use 0.1 to match the paper text.")
    p.add_argument("--skip", nargs="*", default=[], choices=["M1", "M2", "K"])
    p.add_argument("--quick", action="store_true", help="Smoke test: 1 seed, 2 epochs.")
    a = p.parse_args()

    if a.diff_epochs is not None:
        ab.DIFF_EPOCHS = a.diff_epochs
    if a.vae_max_beta is not None:
        ab.VAE_MAX_BETA = a.vae_max_beta
    if a.quick:
        a.seeds = a.seeds[:1]
        ab.VAE_EPOCHS, ab.DIFF_EPOCHS = 2, 2
    os.makedirs(a.out, exist_ok=True)
    print(f"[setup] device={ab.DEVICE} seeds={a.seeds} vae_epochs={ab.VAE_EPOCHS} "
          f"diff_epochs={ab.DIFF_EPOCHS} max_beta={ab.VAE_MAX_BETA}")

    X_tr, X_val, X_te, y_tr, y_val, y_te, yt_te = ab.load_data(a.data)
    input_dim = X_tr.shape[1]
    print(f"[setup] train={tuple(X_tr.shape)} val={tuple(X_val.shape)} test={tuple(X_te.shape)} "
          f"outlier rate (test)={y_te.mean():.4f}")
    rows = []

    # ---------------- M1
    if "M1" not in a.skip:
        X_fit = torch.cat([X_tr, X_val])
        s_glob = np.abs(X_te[:, 0].numpy())
        s_coh, n_fb, n_groups = cohort_abs_z(X_fit, X_te)
        print(f"[M1] cohort profiles in train+val: {n_groups}; test rows using global fallback: {n_fb}")
        for name, s in [("salary_abs_z", s_glob), ("cohort_abs_z", s_coh)]:
            m = ab.compute_metrics(s, y_te, yt_te)
            m.update(check="M1", method=name, seed=-1)
            rows.append(m)
            print(f"[M1] {name:13s} AUROC={m['auc_roc']:.4f} AP={m['ap']:.4f} F1={m['f1']:.4f}")

    # ---------------- M2
    if "M2" not in a.skip:
        cfg = config_by_name("E_cxt_in_diff")
        ab.ContextEncoder = CatOnlyContextEncoder
        try:
            for seed in a.seeds:
                t0 = time.time()
                m = ab.run_single_experiment(cfg, seed, X_tr, X_val, X_te, y_te, yt_te,
                                             input_dim, ab.DEVICE)
                m.pop("scores", None)
                m.update(check="M2", method="CALDM-D_cat_only_context", seed=seed)
                rows.append(m)
                print(f"[M2] seed={seed} AUROC={m['auc_roc']:.4f} AP={m['ap']:.4f} "
                      f"F1={m['f1']:.4f} ({time.time() - t0:.0f}s)")
        finally:
            ab.ContextEncoder = _OrigEncoder

    # ---------------- K
    if "K" not in a.skip:
        seed = a.seeds[0]
        cfg, enc, vae, diff = train_caldm_d(X_tr, X_val, input_dim, seed)
        enc.eval(); vae.eval(); diff.eval()
        n = X_te.shape[0]
        rng = np.random.RandomState(0)
        orig = np.arange(n)
        shuf = rng.permutation(n)
        chunk = ab.ENCODER_CHUNK_SIZE
        c0 = contexts_in_order(enc, X_te, orig, chunk)
        c1 = contexts_in_order(enc, X_te, shuf, chunk)
        cos = torch.nn.functional.cosine_similarity(c0, c1, dim=1).numpy()
        l2 = (c0 - c1).norm(dim=1).numpy() / (c0.norm(dim=1).numpy() + 1e-8)
        s_a = score_with_order(cfg, enc, vae, diff, X_te, orig, noise_seed=1)
        s_b = score_with_order(cfg, enc, vae, diff, X_te, orig, noise_seed=2)   # same chunks, new noise
        s_c = score_with_order(cfg, enc, vae, diff, X_te, shuf, noise_seed=1)   # new chunks
        rho_noise = stats.spearmanr(s_a, s_b).correlation
        rho_chunk = stats.spearmanr(s_a, s_c).correlation
        auc = {k: ab.compute_metrics(v, y_te)["auc_roc"] for k, v in
               [("orig", s_a), ("new_noise", s_b), ("shuffled", s_c)]}
        ap = {k: ab.compute_metrics(v, y_te)["ap"] for k, v in
              [("orig", s_a), ("new_noise", s_b), ("shuffled", s_c)]}
        k_summary = {
            "check": "K", "method": "CALDM-D_chunk_dependence", "seed": seed,
            "ctx_cosine_median": float(np.median(cos)), "ctx_cosine_p05": float(np.percentile(cos, 5)),
            "ctx_rel_l2_median": float(np.median(l2)),
            "score_spearman_new_noise": rho_noise, "score_spearman_shuffled_chunks": rho_chunk,
            "auc_orig": auc["orig"], "auc_new_noise": auc["new_noise"], "auc_shuffled": auc["shuffled"],
            "ap_orig": ap["orig"], "ap_new_noise": ap["new_noise"], "ap_shuffled": ap["shuffled"],
        }
        rows.append(k_summary)
        print("[K] context cosine (orig vs shuffled chunks): median={:.4f}  5th pct={:.4f}".format(
            k_summary["ctx_cosine_median"], k_summary["ctx_cosine_p05"]))
        print("[K] score Spearman: same chunks/new noise={:.4f}   shuffled chunks={:.4f}".format(
            rho_noise, rho_chunk))
        print("[K] AUROC orig/new-noise/shuffled = {:.4f} / {:.4f} / {:.4f}".format(
            auc["orig"], auc["new_noise"], auc["shuffled"]))

    out = pd.DataFrame(rows)
    path = os.path.join(a.out, "reviewer_checks.csv")
    out.to_csv(path, index=False)
    m2 = out[out["check"] == "M2"]
    if len(m2):
        print(f"[M2] mean over seeds: AUROC={m2.auc_roc.mean():.4f} ± {m2.auc_roc.std():.4f}  "
              f"AP={m2.ap.mean():.4f} ± {m2.ap.std():.4f}")
    print(f"[ok] wrote {path}")


if __name__ == "__main__":
    main()
