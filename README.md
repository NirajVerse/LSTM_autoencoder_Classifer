# ML — Jamming Attack Classifier

Classifies **5G UE KPI traces** from `rtue` into **clean**, **barrage**, or **random** jamming using a two-stage model:

1. **LSTM autoencoder** — trained on clean data only; learns normal link behavior.
2. **Hybrid classifier** — BiLSTM + attention head on frozen encoder features; predicts the attack type.

Input is semicolon-separated metrics CSVs (1 row ≈ 1 second). Each file is split into 30 s windows (15 s stride) using 12 KPIs plus 4 engineered features.

## Train vs test data (important)

| Folder | Config key | Used by |
|---|---|---|
| `datasets_02/` | `datasets_dir` | **Training only** (`train_ae.py`, `train_clf.py`) |
| `datasets_03/` | `test_datasets_dir` | **Evaluation & paper figures only** |

```
datasets_02/                    datasets_03/
  clean/run_001_metrics.csv       clean/run_001_metrics.csv
  barrage/...                     barrage/...
  random/...                      random/...
```

## Setup

```bash
cd ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Edit `config.yaml` if needed.

## Workflow

**1. Train** (uses `datasets_dir` only):

```bash
python run_all.py
# or: python train_ae.py && python train_clf.py
```

**2. Collect test CSVs** into `datasets_03/` (separate session from training).

**3. Evaluate on test set**:

```bash
python evaluate.py --run-id <run_id>
```

Saves `results/<run_id>/test_evaluation.json` — use these numbers in the paper.

**4. Paper figures** (test KPIs + test confusion matrix):

```bash
python make_paper_figures.py --run-id <run_id>
```

**Predict one CSV** (any file, e.g. a single test capture):

```bash
python predict.py --csv ../datasets_03/barrage/run_001_metrics.csv --run-id <run_id>
```

## Run history

Each training run gets a timestamp id (`20250630_214530`). Checkpoints and results are stored under that id; nothing is overwritten.

```
ml/checkpoints/<run_id>/
ml/results/<run_id>/test_evaluation.json
ml/figures/<run_id>/
ml/latest_run.txt
```

## Main files

| File | Purpose |
|------|---------|
| `config.yaml` | Train/test paths, classes, hyperparameters |
| `train_ae.py` / `train_clf.py` | Train on `datasets_dir` |
| `evaluate.py` | Predict on `test_datasets_dir` only |
| `make_paper_figures.py` | Figures from test data + `test_evaluation.json` |
| `predict.py` | Single-file prediction |
