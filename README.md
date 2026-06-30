# ML — Jamming Attack Classifier

Classifies **5G UE KPI traces** from `rtue` into **clean**, **barrage**, or **random** jamming using a two-stage model:

1. **LSTM autoencoder** — trained on clean data only; learns normal link behavior.
2. **Hybrid classifier** — BiLSTM + attention head on frozen encoder features; predicts the attack type.

Input is semicolon-separated metrics CSVs (1 row ≈ 1 second). Each file is split into 30 s windows (15 s stride) using 12 KPIs plus 4 engineered features (SNR/BLER stats, etc.).

## Data layout

Place CSVs under the folder named in `config.yaml` (`datasets_dir`):

```
datasets/
  clean/run_001_metrics.csv
  barrage/run_001_metrics.csv
  random/run_001_metrics.csv
```

## Setup

```bash
cd ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Edit `config.yaml` if needed (paths, classes, hyperparameters).

## Train & evaluate

**Full pipeline** (recommended):

```bash
python run_all.py
```

**Step by step:**

```bash
python train_ae.py      # autoencoder on clean only
python train_clf.py     # classifier on all classes
python evaluate.py      # metrics + confusion matrix
python make_paper_figures.py   # PNG/PDF figures for the paper
```

**Predict one CSV:**

```bash
python predict.py --csv ../datasets/barrage/run_001_metrics.csv
```

Use `--run-id <id>` on any script to pick a specific run. Default is the latest.

## Main files

| File | Purpose |
|------|---------|
| `config.yaml` | Paths, classes, window size, model/training settings |
| `load_data.py` / `features.py` | CSV loading and feature engineering |
| `models.py` | Autoencoder and classifier architectures |
| `train_ae.py` / `train_clf.py` | Training scripts |
| `evaluate.py` | Test-set report and per-file votes |
| `make_paper_figures.py` | Confusion matrix, KPI plots, bar charts |
