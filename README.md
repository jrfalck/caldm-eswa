# CALDM: Context-Aware Latent Diffusion for Pay Anomaly Detection

Reproducibility package for the paper:

> **CALDM: A Context-Aware Latent Diffusion Architecture for Pay Anomaly Detection in Tabular HR Data**
> Juan R. Falck, Muhammad F. Islam — George Washington University
> *Submitted to Expert Systems with Applications (ESWA)*

---

## Repository Structure

```
caldm-eswa/
├── scripts/
│   ├── caldm_ablation_study_v3.py    # Tables 7 & 8 (BLS ablation)
│   ├── caldm_ablation_adbench.py     # Table 9 (ADBench replication)
│   └── caldm_benchmark.py            # Table 5 + Appendix A
├── notebooks/
│   ├── 02_ablation_study_bls.ipynb
│   ├── 03_ablation_adbench_replication.ipynb
│   └── 04_benchmark_comparison.ipynb
├── results/
│   ├── ablation_results_summary.csv  # Table 7
│   ├── ablation_per_type.csv         # Table 8
│   ├── ablation_results_raw.csv
│   ├── bls_summary.csv               # Table 5
│   ├── bls_per_run.csv
│   ├── ablation_adbench/             # Table 9
│   │   ├── adbench_summary.csv
│   │   ├── adbench_per_run.csv
│   │   └── adbench_per_dataset.csv
│   └── appendix_a/                   # Appendix A
│       ├── adbench_summary.csv
│       ├── adbench_per_run.csv
│       └── adbench_per_dataset.csv
└── data/
    └── README.md                     # Download instructions
```

---

## Reproducing Paper Results

All pre-computed results are included in `results/`. To verify a table without
re-running training, open the corresponding notebook and set `RUN_TRAINING = False`
— the notebook will load the pre-computed CSVs and reproduce all tables and
verification checks instantly, with no GPU required.

To re-run training from scratch, set `RUN_TRAINING = True` and ensure the data
files are in place (see **Data** section below).

| Paper table | Notebook | Script | Est. GPU time |
|---|---|---|---|
| Tables 7 & 8 — BLS ablation | 02_ablation_study_bls | caldm_ablation_study_v3.py | ~4 hours |
| Table 9 — ADBench replication | 03_ablation_adbench_replication | caldm_ablation_adbench.py | ~6 hours |
| Table 5 — BLS benchmark | 04_benchmark_comparison | caldm_benchmark.py | ~2–3 hours |
| Appendix A — ADBench benchmark | 04_benchmark_comparison | caldm_benchmark.py | ~8 hours |

---

## Setup

**Requirements:** Python 3.9+. A CUDA-capable GPU is recommended for training
runs but is not required to verify pre-computed results.

```bash
pip install -r requirements.txt
```

**Tested environment:** Python 3.10, PyTorch 2.0, NVIDIA RTX-class GPU (16 GB VRAM),
Windows 11 and Ubuntu 24.

---

## Data

### BLS Compensation Dataset

A synthetic compensation dataset derived from the U.S. Bureau of Labor Statistics
Occupational Employment and Wage Statistics (OEWS) program. Individual records are
generated from real BLS aggregate statistics using distribution-fitting across
MSA × Occupation × Industry cells, with five rule-based labeled outlier types.

**Download:** https://data.hrmatica.com/bls/final_data_subsetFLGAAL8.npz

**Place at:** `data/final_data_subsetFLGAAL8.npz`

### ADBench Datasets

Download the following 27 files from https://github.com/Minqi824/ADBench
and place them all in `data/adbench/`

```
3_backdoor.npz          10_cover.npz            11_donors.npz
14_glass.npz            17_InternetAds.npz      18_Ionosphere.npz
19_landsat.npz          20_letter.npz           23_mammography.npz
24_mnist.npz            25_musk.npz             26_optdigits.npz
27_PageBlocks.npz       28_pendigits.npz        30_satellite.npz
31_satimage-2.npz       32_shuttle.npz          35_SpamBase.npz
36_speech.npz           38_thyroid.npz          40_vowels.npz
41_Waveform.npz         5_campaign.npz          6_cardio.npz
7_Cardiotocography.npz  8_celeba.npz            9_census.npz
```

> **Note:** These are 27 of the 57 datasets available in the ADBench suite.
> The remaining datasets were excluded due to memory or runtime constraints
> on methods with O(n²) complexity, or insufficient row count for the
> evaluation protocol used in this study.

---

## Hyperparameters

All hyperparameters used to produce the paper results are documented in
`results/hyperparameters/Parameters_Log_ESWApaper.xlsx`.

Key settings used across all CALDM experiments:

| Parameter | Value |
|---|---|
| VAE epochs | 15 |
| Diffusion epochs | 200 |
| KL weight β | 0.1 (warmed up over first 5 epochs) |
| Latent dim — numerical | 16 |
| Latent dim — categorical | 32 |
| Context dim | 40 |
| Diffusion timesteps T | 1,000 |
| Batch size | 256 |
| Salary loss weight | 2.0 |

For the protocol-consistent BLS benchmark (Table 5) and auto-scaled ADBench
benchmark (Appendix A), all CALDM hyperparameters were set deterministically
per dataset using an automated scaling rule (no per-dataset tuning). Full
details are provided in Appendix A.1 of the paper.

---

## Quick Smoke Test

To verify the code runs correctly without a long training job, run the
following from the repo root. This runs CALDM-D and two baselines on a
single ADBench dataset with one seed and takes under 5 minutes:

```bash
python scripts/caldm_benchmark.py \
  --target adbench \
  --adbench_dir data/adbench \
  --methods CALDM-D IForest KNN \
  --seeds 42 \
  --output_dir smoke_test \
  --quick
```

---

## Citation

```bibtex
@article{falck2025caldm,
  title   = {CALDM: A Context-Aware Latent Diffusion Architecture for
             Pay Anomaly Detection in Tabular HR Data},
  author  = {Falck, Juan R. and Islam, Muhammad F.},
  journal = {Expert Systems with Applications},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

MIT License — see `LICENSE` for details.
