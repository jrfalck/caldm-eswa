"""
Table 5 (BLS benchmark) CALDM rows, re-run with the SAME CALDM implementation used
in the BLS ablation (caldm_ablation_study_v3.py: dual-head VAE, two parallel
denoising networks, per-head sampler) and the salary-free context encoder.

Splits are identical to the Table 5 protocol: for each seed (69, 72, 888) a
stratified 20% sub-sample is drawn and split 80/20 by caldm_benchmark.load_bls.
The ten PyOD baselines and the two rule-based salary baselines were already run
on these exact splits, so all Table 5 rows become directly comparable.

Usage (repo root):
    python scripts/caldm_table5_consistent.py --data ../data/final_data_subsetFLGAAL8.npz
Output: results/bls_benchmark_consistent/
"""
import os
import sys
import argparse

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import caldm_ablation_study_v3 as ab           # noqa: E402
import caldm_benchmark as bm                    # noqa: E402
from caldm_rerun_salary_free_context import make_cat_only  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="results/bls_benchmark_consistent")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()

    ab.ContextEncoder = make_cat_only(ab.ContextEncoder)
    ab.DIFF_EPOCHS, ab.VAE_MAX_BETA = 200, 0.1
    seeds = [69, 72, 888]
    if a.quick:
        seeds, ab.VAE_EPOCHS, ab.DIFF_EPOCHS = [69], 2, 2
    os.makedirs(a.out, exist_ok=True)
    configs = {"E_cxt_in_diff": "CALDM-D", "F_full_caldm": "CALDM-J"}
    rows = []
    for seed in seeds:
        X_tr, X_te, y_tr, y_te, yt_te, meta = bm.load_bls(a.data, 0.2, seed=seed, verbose=False)
        print(f"\n=== seed={seed}  train={tuple(X_tr.shape)} test={tuple(X_te.shape)} "
              f"outlier rate (test)={meta['outlier_rate_test']:.4f}")
        for cname, label in configs.items():
            cfg = next(c for c in ab.CONFIGS if c.name == cname)
            m = ab.run_single_experiment(cfg, seed, X_tr, X_tr[:2000], X_te, y_te, yt_te,
                                         X_tr.shape[1], ab.DEVICE)
            m.pop("scores", None)
            m.update(method=label, n_train=meta["n_train"], n_test=meta["n_test"])
            rows.append(m)
            print(f"  {label}: AUROC={m['auc_roc']:.4f} AP={m['ap']:.4f} F1={m['f1']:.4f} "
                  f"({m['train_time_sec']:.0f}s)")
            pd.DataFrame(rows).to_csv(os.path.join(a.out, "bls_per_run.csv"), index=False)
    df = pd.DataFrame(rows)
    s = df.groupby("method")[["auc_roc", "ap", "f1"]].agg(["mean", "std"])
    s.columns = [f"{m}_{k}" for m, k in s.columns]
    s.reset_index().to_csv(os.path.join(a.out, "bls_summary.csv"), index=False)
    print("\n" + s.round(4).to_string())
    print("[done]")


if __name__ == "__main__":
    main()
