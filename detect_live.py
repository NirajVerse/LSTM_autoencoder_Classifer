#!/usr/bin/env python3
"""ID-xApp equivalent: live jamming classification from Influx KPI stream.

Paper-style data flow (detection only, no mitigation):

    rtue (KPI source) → InfluxDB → detect_live.py → alert

Keep CSV enabled on rtue in parallel for offline training / archive.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from influx_kpi import InfluxConfig, InfluxKpiReader, influx_record_to_kpi_row, kpi_rows_to_window_matrix
from training import load_ae_checkpoint, load_clf_checkpoint, transform_windows
from utils import get_device, get_run_dirs, load_config, resolve_run_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live jamming detection from Influx KPI stream")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--run-id", type=str, default=None, help="Checkpoint run id (default: latest)")
    p.add_argument(
        "--alert-log",
        type=Path,
        default=None,
        help="Append JSONL alerts (default: ml/results/live_alerts.jsonl)",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Run one inference if buffer is full, then exit (for testing)",
    )
    return p.parse_args()


def load_models(cfg: dict, run_id: str):
    device = get_device()
    ckpt_dir, _ = get_run_dirs(cfg, run_id)
    ae_path = ckpt_dir / "autoencoder.pt"
    clf_path = ckpt_dir / "classifier.pt"
    if not ae_path.exists() or not clf_path.exists():
        raise FileNotFoundError(f"Missing checkpoints for run {run_id}. Train first.")

    ae_model, scaler, _ = load_ae_checkpoint(ae_path, cfg)
    clf_model, meta = load_clf_checkpoint(clf_path, cfg)
    ae_model.to(device)
    clf_model.to(device)
    class_names = meta.get("class_names") or cfg["classes"]
    return ae_model, clf_model, scaler, class_names, device


def predict_window(ae_model, clf_model, scaler, x_window: np.ndarray, device, batch_size: int) -> tuple[int, np.ndarray]:
    import torch

    x = transform_windows(x_window, scaler)
    ae_model.eval()
    clf_model.eval()
    with torch.no_grad():
        batch_x = torch.from_numpy(x).to(device)
        z, seq_out = ae_model.encode_sequence(batch_x)
        logits = clf_model(batch_x, z, seq_out)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())
    return pred, probs


def append_alert(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload) + "\n")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    live = cfg.get("live_detection") or {}
    influx_cfg = InfluxConfig.from_cfg(cfg)

    if not influx_cfg.token:
        print("Error: influx.token is empty in config.yaml", file=sys.stderr)
        sys.exit(1)

    run_id = resolve_run_id(args.run_id, cfg=cfg)
    window_size = int(cfg["window_size"])
    stride_sec = float(live.get("infer_stride_sec", cfg.get("window_stride", 15)))
    poll_sec = float(live.get("poll_interval_sec", 1.0))
    min_conf = float(live.get("alert_confidence", 0.6))
    alert_log = args.alert_log
    if alert_log is None:
        alert_log = Path(__file__).resolve().parent / "results" / "live_alerts.jsonl"

    ae_model, clf_model, scaler, class_names, device = load_models(cfg, run_id)
    reader = InfluxKpiReader(influx_cfg)
    buffer: deque[dict[str, float]] = deque(maxlen=window_size * 4)
    last_infer = 0.0

    print("Live intrusion detection (Influx → ML)")
    print(f"  run id:       {run_id}")
    print(f"  influx:       {influx_cfg.url}  bucket={influx_cfg.bucket}")
    print(f"  ue_data_id:   {influx_cfg.ue_data_id}  carrier={influx_cfg.carrier_type}")
    print(f"  window:       {window_size}s KPI rows, infer every {stride_sec}s")
    print(f"  classes:      {class_names}")
    print(f"  alert log:    {alert_log}")
    print("Waiting for KPI stream from rtue...\n")

    try:
        while True:
            raw_rows = reader.fetch_new_rows(lookback_sec=max(120.0, window_size * poll_sec * 2))
            for raw in raw_rows:
                buffer.append(influx_record_to_kpi_row(raw))

            now = time.monotonic()
            ready = len(buffer) >= window_size
            due = (now - last_infer) >= stride_sec

            if ready and due:
                last_infer = now
                window = kpi_rows_to_window_matrix(list(buffer)[-window_size:], cfg)
                pred_idx, probs = predict_window(
                    ae_model, clf_model, scaler, window, device, cfg["batch_size"]
                )
                label = class_names[pred_idx]
                confidence = float(probs[pred_idx])
                ts = datetime.now(timezone.utc).isoformat()

                prob_map = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
                line = (
                    f"[{ts}]  prediction={label:8s}  confidence={confidence:.3f}  "
                    f"probs={prob_map}"
                )
                print(line)

                if confidence >= min_conf:
                    append_alert(
                        alert_log,
                        {
                            "time": ts,
                            "run_id": run_id,
                            "prediction": label,
                            "confidence": confidence,
                            "probabilities": prob_map,
                            "buffer_rows": window_size,
                        },
                    )

                if args.once:
                    return

            if not ready:
                print(f"\rBuffer: {len(buffer)}/{window_size} KPI rows...", end="", flush=True)

            time.sleep(poll_sec)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        reader.close()


if __name__ == "__main__":
    main()
