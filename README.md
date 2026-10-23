# CALDM: Context-Aware Latent Diffusion for Pay Anomaly Detection

Reproducibility package for:

> **CALDM: A Context-Aware Latent Diffusion Architecture for Pay Anomaly Detection in Tabular HR Data**
> Juan R. Falck and Muhammad F. Islam, The George Washington University.
> Submitted to *Expert Systems with Applications*.

Every table in the paper can be regenerated from the scripts and data described here.
Pre-computed results for all tables are included in `results/`.

---

## Repository structure

```
scripts/
  caldm_ablation_study_v3.py          BLS ablation: model, training, scoring (Tables 5-6)
  caldm_rerun_salary_free_context.py  Runs the BLS ablation with the salary-free context encoder
                                      (Tables 5-6) and the rule-based references for Table 4
  caldm_table5_consistent.py          CALDM-D and CALDM-J rows of Table 4 (BLS benchmark)
  caldm_benchmark.py                  PyOD baselines for Table 4; ADBench benchmark (Appendix A)
  caldm_ablation_adbench.py           ADBench ablation replication (Table 7)
  caldm_reviewer_checks.py            Rule-based references for Tables 5-6; batch-sensitivity
                                      check (Section 8.4)
  caldm_dataset_property_analysis.py  Dataset-property analysis (Section 7.5)
results/                              Pre-computed results (see table below)
data/                                 Not tracked; place the datasets here (see Data)
requirements.txt
```

## Paper tables and where they come from

| Paper | Results folder or file | Produced by |
|---|---|---|
| Table 4, CALDM-D and CALDM-J rows | `results/bls_benchmark_consistent/` | `caldm_table5_consistent.py` |
| Table 4, PyOD baseline rows | `results/bls_per_run.csv`, `results/bls_summary.csv` | `caldm_benchmark.py --target bls` |
| Table 4, rule-based references | `results/bls_benchmark_salary_free_context/rule_baselines_*.csv` | `caldm_rerun_salary_free_context.py` (benchmark part) |
| Tables 5 and 6 (BLS ablation) | `results/ablation_v5_salary_free_context/` | `caldm_rerun_salary_free_context.py` (ablation part) |
| Tables 5 and 6, rule-based references | `results/reviewer_checks/reviewer_checks.csv` (check M1) | `caldm_reviewer_checks.py` |
| Section 7.1, earlier context encoder with salary | `results/ablation_v4_paper_settings/` | `caldm_ablation_study_v3.py` (command below) |
| Table 7 (ADBench ablation replication) | `results/ablation_adbench/` | `caldm_ablation_adbench.py` |
| Section 7.5, dataset-property analysis | computed from `results/ablation_adbench/` | `caldm_dataset_property_analysis.py` |
| Section 8.4, batch sensitivity | `results/reviewer_checks/reviewer_checks.csv` (check K) | `caldm_reviewer_checks.py` |
| Appendix A, Table A.1 | `results/appendix_a/` | `caldm_benchmark.py --target adbench` |

Notes:
- `results/bls_per_run.csv` also contains CALDM-D and CALDM-J rows from the benchmark script's own
  CALDM implementation. These rows are superseded by `results/bls_benchmark_consistent/`, which uses
  the implementation described in Section 4 of the paper. Only the PyOD rows of that file are used.
  For the same reason, `caldm_rerun_salary_free_context.py` also writes CALDM rows to
  `results/bls_benchmark_salary_free_context/bls_per_run.csv`; only its `rule_baselines_*.csv` files
  are used in the paper, and only those are included here.
- On ADBench (Table 7 and Appendix A), CALDM uses a single-headed VAE and the context encoder
  receives all features, as described in Section 4.2.

---

## Setup

Results in the paper were produced with:

| Component | Version |
|---|---|
| Python | 3.8.17 |
| PyTorch | 2.2.1 (CUDA 11.8) |
| NumPy | 1.24.4 |
| pandas | 2.0.3 |
| scikit-learn | 1.3.0 |
| SciPy | 1.10.1 |
| PyOD | 2.0.2 |
| GPU | NVIDIA GeForce RTX 3070 Laptop GPU |

```bash
pip install -r requirements.txt
```

All scripts fix their random seeds and request deterministic cuDNN kernels. Runs on other GPUs or
library versions can differ slightly (typically in the third decimal place).

---

## Data

### BLS compensation dataset

Synthetic individual-level compensation records (596,856 rows) generated from U.S. Bureau of Labor
Statistics OEWS wage percentiles for Florida, Alabama and Georgia, with five types of labeled pay
outliers.

- **Mendeley Data:** https://doi.org/10.17632/rrr2p47s8g.1 (includes a README and `feature_names.csv`
  describing every column)
- Place the file at `data/final_data_subsetFLGAAL8.npz`

### ADBench datasets

Download the following files from https://github.com/Minqi824/ADBench and place them in
`data/adbench/`:

```
3_backdoor.npz   9_census.npz       11_donors.npz    14_glass.npz        15_Hepatitis.npz
17_InternetAds.npz 18_Ionosphere.npz 19_landsat.npz  20_letter.npz       23_mammography.npz
24_mnist.npz     25_musk.npz        26_optdigits.npz 27_PageBlocks.npz   28_pendigits.npz
30_satellite.npz 31_satimage-2.npz  32_shuttle.npz   34_smtp.npz         35_SpamBase.npz
36_speech.npz    38_thyroid.npz     40_vowels.npz    41_Waveform.npz     5_campaign.npz
6_cardio.npz     7_Cardiotocography.npz 8_celeba.npz 10_cover.npz
```

The scripts skip datasets that do not meet their minimum requirements (Hepatitis: fewer than 100
rows; smtp in Appendix A: degenerate outlier rate), leaving 27 datasets for each experiment
(Section 3.2 of the paper). Table 7 was run before `9_census.npz` was added, so to reproduce it
exactly, run `caldm_ablation_adbench.py` with that file removed from `data/adbench/`.

---

## Reproducing the results

Run from the repository root. Times are for an RTX 3070 laptop GPU.

```bash
# Tables 5 and 6 (BLS ablation, salary-free context) and the rule-based references for Table 4
# (about 4 hours)
python scripts/caldm_rerun_salary_free_context.py --data data/final_data_subsetFLGAAL8.npz

# Table 4, CALDM-D and CALDM-J rows (about 1 hour)
python scripts/caldm_table5_consistent.py --data data/final_data_subsetFLGAAL8.npz

# Table 4, PyOD baseline rows
python scripts/caldm_benchmark.py --target bls --bls_path data/final_data_subsetFLGAAL8.npz \
    --bls_subset_pct 0.2 --seeds 69 72 888 --vae_epochs 15 --diff_epochs 200 --vae_max_beta 0.1 \
    --methods KNN LOF IForest PCA CBLOF HBOS ECOD COPOD LODA DeepSVDD --output_dir results

# Rule-based references for Tables 5 and 6, and the Section 8.4 batch-sensitivity check
python scripts/caldm_reviewer_checks.py --data data/final_data_subsetFLGAAL8.npz \
    --diff_epochs 200 --vae_max_beta 0.1

# Section 7.1: earlier context encoder that also received salary
python -c "import sys; sys.path.insert(0,'scripts'); from caldm_ablation_study_v3 import run_ablation; run_ablation(data_path='data/final_data_subsetFLGAAL8.npz', seeds=[7,42,888,1337,2024], diff_epochs=200, vae_max_beta=0.1, output_dir='results/ablation_v4_paper_settings')"

# Table 7 (ADBench ablation replication, about 6 hours)
python scripts/caldm_ablation_adbench.py --input_dir data/adbench --out results/ablation_adbench \
    --seeds 69 72 888

# Section 7.5 dataset-property analysis (seconds, no GPU)
python scripts/caldm_dataset_property_analysis.py

# Appendix A (ADBench benchmark, about 8 hours)
python scripts/caldm_benchmark.py --target adbench --adbench_dir data/adbench --seeds 69 72 888 \
    --vae_epochs 15 --diff_epochs 200 --vae_max_beta 0.1 \
    --methods KNN LOF IForest PCA CBLOF HBOS ECOD COPOD LODA DeepSVDD CALDM-D CALDM-J \
    --output_dir results/appendix_a
```

Each command writes to its own folder, so existing pre-computed results are only replaced by the
run that produced them.

---

## Citation

```bibtex
@article{falck_caldm,
  title   = {CALDM: A Context-Aware Latent Diffusion Architecture for
             Pay Anomaly Detection in Tabular HR Data},
  author  = {Falck, Juan R. and Islam, Muhammad F.},
  journal = {Expert Systems with Applications},
  note    = {Under review}
}

@misc{falck_caldm_data,
  title  = {Synthetic BLS OEWS Compensation Dataset with Labeled Pay Outliers (FL/AL/GA subset)},
  author = {Falck, Juan R. and Islam, Muhammad F.},
  publisher = {Mendeley Data},
  doi    = {10.17632/rrr2p47s8g.1}
}
```

## License

MIT License for the code (see `LICENSE`). The dataset is released under CC BY 4.0 on Mendeley Data.
