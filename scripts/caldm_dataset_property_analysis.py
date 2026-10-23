"""
Section 7.5 dataset-property analysis (ESWA revision).

Can dataset-level properties (rows, features, outlier rate) predict whether
CALDM-D (E) or CALDM-J (F) has the higher AUROC on an ADBench dataset?
Reads results/ablation_adbench/adbench_per_run.csv (Table 8 runs); no GPU needed.

Usage (repo root):  python scripts/caldm_dataset_property_analysis.py
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pr = pd.read_csv("results/ablation_adbench/adbench_per_run.csv")
g = pr.groupby(["dataset", "config"]).agg(auc=("auc_roc", "mean"), n_rows=("n_rows", "first"),
                                           n_features=("n_features", "first"),
                                           outlier_rate=("outlier_rate", "first")).reset_index()
auc = g.pivot(index="dataset", columns="config", values="auc")
meta = g.groupby("dataset")[["n_rows", "n_features", "outlier_rate"]].first().loc[auc.index]
diff = auc["E_caldm_d"] - auc["F_caldm_j"]
y = (diff > 0).astype(int)

print(f"Datasets: {len(y)}   E wins: {y.sum()}   majority-class baseline: {max(y.mean(), 1 - y.mean()):.3f}")
print(f"|E-F| > 0.05 on {(diff.abs() > 0.05).sum()} datasets; > 0.20 on {(diff.abs() > 0.20).sum()}")
X = np.log(meta)
pred = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression()), X, y, cv=LeaveOneOut())
print(f"Logistic regression, leave-one-out accuracy: {(pred == y).mean():.3f}")
for c in meta.columns:
    rho, p = stats.spearmanr(meta[c], diff)
    print(f"Spearman {c:13s} vs AUROC(E)-AUROC(F): rho = {rho:+.3f}, p = {p:.3f}  (Bonferroni x3: {min(1, 3 * p):.3f})")
