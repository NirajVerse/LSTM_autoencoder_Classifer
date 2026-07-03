# ML — Jamming Attack Classifier

Classifies **5G UE KPI traces** into **clean**, **barrage**, or **random** using a two-stage model (LSTM autoencoder + hybrid classifier).

## Data flows

### Offline (training & paper evaluation)

```text
rtue → CSV files → datasets_02/ (train)  → train_ae.py / train_clf.py
                 → datasets_03/ (test)   → evaluate.py / paper figures
```

### Live (paper-style, no mitigation)

```text
rtue ──→ InfluxDB (KPI-moni) ──→ detect_live.py (ID-xApp) ──→ alert
  └──→ CSV file (backup / archive)
```

| Paper component | Our implementation |
|---|---|
| KPI-moni xApp | rtue `metrics_influxdb` → InfluxDB |
| ID-xApp | `ml/detect_live.py` |
| SS-xApp (mitigation) | *not implemented* |

## Setup

```bash
cd ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Configure rtue (lab machine)

Use `configs/rtue_ue_uhd_streaming.conf` as reference — enables **both** Influx and CSV:

- `metrics_influxdb_enable = true`
- `ue_data_identifier = ue1_uhd` (must match `influx.ue_data_id` in `config.yaml`)
- `metrics_csv_enable = true` + `metrics_csv_append = true` (backup)

InfluxDB must be running (e.g. `docker compose` in ran-tester-ue).

## 2. Train (offline, CSV datasets)

```bash
python run_all.py
# train on datasets_02; evaluate later on datasets_03
python evaluate.py --run-id <run_id>
```

## 3. Live detection (Influx stream)

After training and with rtue + Influx running:

```bash
python detect_live.py --run-id <run_id>
```

Output example:

```text
[2026-06-30T...]  prediction=barrage   confidence=0.94  probs={...}
```

Alerts append to `ml/results/live_alerts.jsonl`.

Test once and exit:

```bash
python detect_live.py --run-id <run_id> --once
```

## Config (`config.yaml`)

| Key | Purpose |
|---|---|
| `datasets_dir` | Training CSV folders |
| `test_datasets_dir` | Held-out test CSV folders |
| `influx.*` | Influx connection + `ue_data_id` |
| `live_detection.*` | Poll interval, infer stride, alert threshold |

## Main scripts

| Script | Role |
|---|---|
| `train_ae.py` / `train_clf.py` | Train on CSV (`datasets_dir`) |
| `evaluate.py` | Test-set metrics (`test_datasets_dir`) |
| `detect_live.py` | **Live** classification from Influx |
| `detect_live_gnb.py` | **Live** classification from gNB WebSocket |
| `influx_kpi.py` | Influx → KPI row mapping |
| `predict.py` | Single CSV file (manual) |
| `make_paper_figures.py` | Paper figures from test evaluation |
| `gnb_capture.py` | **gNB** WebSocket → CSV (training captures) |
| `gnb_jsonl_to_csv.py` | Convert JSONL archive → CSV |
| `gnb_metrics_ws_logger.py` | gNB WebSocket → JSONL (optional archive) |
| `compare_gnb_scenarios.py` | Summarize clean / barrage / random gNB CSVs |

## gNB KPI capture → ML (`config_gnb.yaml`)

Use srsRAN scheduler KPIs from WebSocket (`remote_control` port **8001**). Config is in `ran-tester-ue/configs/srsran/gnb_uhd.yaml`.

**Lab runbook (each scenario ~5–8 min, traffic on):**

1. Copy updated `gnb_uhd.yaml` to the lab.
2. Terminal A — **start capture before gNB**:
   ```bash
   cd milcom/ml && source .venv/bin/activate
   python gnb_capture.py --output ../datasets_gnb/clean/run_001.csv
   ```
3. Terminal B — 5GC + gNB + UE; run ping/iperf UL and DL.
4. Scenario: clean (no jammer) → save to `datasets_gnb/clean/`; repeat for `barrage/`, `random/`.
5. Ctrl+C logger between runs; use a new filename per run.
6. Train:
   ```bash
   python run_all.py --config config_gnb.yaml
   python evaluate.py --config config_gnb.yaml --run-id <id>
   ```

Held-out test CSVs go under `datasets_gnb_test/{clean,barrage,random}/`.

Key columns: `dl_bler_pct`, `dl_nof_ok`, `dl_nof_nok`, `dl_brate`, `cqi`, `dl_mcs`, `bsr`, `phr`.

### Live detection (gNB WebSocket → model)

After training, run the closed-loop detector while toggling the jammer:

```bash
cd milcom/ml && source .venv/bin/activate
python detect_live_gnb.py --config config_gnb.yaml --run-id <id>
```

Lab terminals: 5GC + gNB + UE + ping, then `detect_live_gnb.py`. Turn jammer on/off during the run.

Optional stdin markers (type in the detector terminal when you toggle the jammer):

```text
jam_on barrage
jam_off
```

Logs:

| File | Content |
|------|---------|
| `ml/results/live_gnb_predictions.jsonl` | Every inference |
| `ml/results/live_gnb_alerts.jsonl` | High-confidence consecutive alerts |
| `ml/results/live_gnb_events.jsonl` | `jam_on` / `jam_off` markers for TTD |

Expect ~30–60 s delay after jammer toggle (30 s window + 15 s infer stride).
