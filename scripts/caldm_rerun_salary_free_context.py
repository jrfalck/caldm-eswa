"""
Rerun the BLS experiments with a salary-free context encoder (ESWA revision).

The context encoder normally sees the full record, including salary (column 0).
This script feeds it the categorical features only, so the context describes the
record's cohort and the diffusion model judges the salary against that context.
Nothing else changes. ADBench is unaffected (it has no privileged column).

Parts (run all by default, or pick with --parts):
  ablation   : all 6 BLS ablation configurations, 5 seeds, paper settings
               -> results/ablation_v5_salary_free_context/   (Tables 7 and 8)
  benchmark  : CALDM-D and CALDM-J on the Table 5 protocol (seeds 69/72/888,
               stratified 20% sub-sample per seed, 80/20 split, auto-scaled
               hyperparameters), plus two rule-based salary baselines on the
               same splits -> results/bls_benchmark_salary_free_context/  (Table 5)
               The ten PyOD baselines do not use context, so their existing
               results in results/bls_per_run.csv remain valid.

Usage (repo root):
    python scripts/caldm_rerun_salary_free_context.py --data ../data/final_data_subsetFLGAAL8.npz
Smoke test:
    python scripts/caldm_rerun_salary_free_context.py --data ... --quick
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def make_cat_only(base_cls):
    class CatOnlyContextEncoder(base_cls):
        """Same encoder, but it never sees column 0 (salary)."""
        def __init__(self, input_dim, *args, **kw):
            super().__init__(input_dim - 1, *args, **kw)

        def forward(self, x):
            return super().forward(x[:, 1:])
    return CatOnlyContextEncoder


def run_ablation_part(a):
    import caldm_ablation_study_v3 as ab
    ab.ContextEncoder = make_cat_only(ab.ContextEncoder)
    out = os.path.join(a.results, "ablation_v5_salary_free_context")
    kw = dict(data_path=a.data, seeds=[7, 42, 888, 1337, 2024], output_dir=out,
              diff_epochs=200, vae_max_beta=0.1, verbose=False)
    if a.quick:
        kw.update(seeds=[7], vae_epochs=2, diff_epochs=2, data_diff_epochs=2)
    print(f"\n##### ABLATION (salary-free context) -> {out}")
    ab.run_ablation(**kw)


def run_benchmark_part(a):
    import caldm_benchmark as bm
    from caldm_reviewer_checks import cohort_abs_z
    bm.ContextEncoder = make_cat_only(bm.ContextEncoder)
    out = os.path.join(a.results, "bls_benchmark_salary_free_context")
    seeds = [69] if a.quick else [69, 72, 888]
    ve, de = (2, 2) if a.quick else (15, 200)
    print(f"\n##### BLS BENCHMARK (salary-free context) -> {out}")
    bm.run_bls_benchmark(data_path=a.data, output_dir=out, subset_pct=0.2, seeds=seeds,
                         methods=["CALDM-D", "CALDM-J"], vae_epochs=ve, diff_epochs=de,
                         vae_max_beta=0.1, resume=False, verbose=True)

    # Rule-based salary baselines on exactly the same per-seed splits
    rows = []
    for seed in seeds:
        X_tr, X_te, y_tr, y_te, yt_te, meta = bm.load_bls(a.data, 0.2, seed=seed, verbose=False)
        s_glob = np.abs(X_te[:, 0].numpy())
        s_coh, n_fb, n_groups = cohort_abs_z(X_tr, X_te)
        for name, s in [("Salary |z| (global)", s_glob), ("Cohort salary |z|", s_coh)]:
            m = bm.compute_metrics(s, y_te)
            m.update(method=name, seed=seed, n_test=len(y_te), cohort_fallback_rows=n_fb)
            rows.append(m)
            print(f"  [rule] seed={seed} {name:20s} AUROC={m['auc_roc']:.4f} AP={m['ap']:.4f} F1={m['f1']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "rule_baselines_per_run.csv"), index=False)
    summ = df.groupby("method")[["auc_roc", "ap", "f1"]].agg(["mean", "std"])
    summ.columns = [f"{m}_{s}" for m, s in summ.columns]
    summ.reset_index().to_csv(os.path.join(out, "rule_baselines_summary.csv"), index=False)
    print(summ.round(4).to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--results", default="results")
    p.add_argument("--parts", nargs="+", default=["ablation", "benchmark"], choices=["ablation", "benchmark"])
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    if a.quick:
        a.results = os.path.join(a.results, "_smoke_test")
    if "ablation" in a.parts:
        run_ablation_part(a)
    if "benchmark" in a.parts:
        run_benchmark_part(a)
    print("\n[done]")


if __name__ == "__main__":
    main()
