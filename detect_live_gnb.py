#!/usr/bin/env python3
"""Live jamming classification from srsRAN gNB WebSocket scheduler KPIs.

Closed-loop demo flow (detection only, no mitigation):

    gNB WebSocket → KPI buffer → AE + classifier → prediction / alert

Toggle the jammer during the run; predictions should move clean ↔ barrage/random
after the sliding window fills (~window_size seconds of KPI history).

Optional stdin events (for time-to-detection notes during demo):

    jam_on barrage
    jam_on random
    jam_off

Type one line and press Enter while this script is running.
"""

from __future__ import annotations

import argparse
import json
import select
import signal
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from detect_live import append_alert, load_models, predict_window
from gnb_metrics_io import flatten_ue_row, gnb_rows_to_window_matrix, rows_from_report, unwrap_payload
from utils import load_config, resolve_run_id

try:
    import websocket
except ImportError:
    print("Install: pip install websocket-client", file=sys.stderr)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live gNB jamming detection from WebSocket KPI stream")
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config_gnb.yaml",
        help="Config (default: config_gnb.yaml)",
    )
    p.add_argument("--run-id", type=str, default=None, help="Checkpoint run id (default: latest)")
    p.add_argument("--url", type=str, default=None, help="gNB WebSocket URL (overrides config)")
    p.add_argument("--rnti", default=None, help="Optional RNTI filter (decimal or hex string)")
    p.add_argument(
        "--prediction-log",
        type=Path,
        default=None,
        help="Append every inference (default: ml/results/live_gnb_predictions.jsonl)",
    )
    p.add_argument(
        "--alert-log",
        type=Path,
        default=None,
        help="Append high-confidence alerts (default: ml/results/live_gnb_alerts.jsonl)",
    )
    p.add_argument(
        "--events-log",
        type=Path,
        default=None,
        help="Append jammer on/off markers (default: ml/results/live_gnb_events.jsonl)",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Run one inference when buffer is full, then exit",
    )
    p.add_argument(
        "--no-stdin-events",
        action="store_true",
        help="Disable jam_on/jam_off stdin markers",
    )
    return p.parse_args()


def connect_websocket(url: str) -> websocket.WebSocket:
    print(f"Connecting to {url}...", flush=True)
    while True:
        try:
            ws = websocket.create_connection(url, timeout=5)
            ws.send(json.dumps({"cmd": "metrics_subscribe"}))
            print(f"Subscribe: {ws.recv()}", flush=True)
            return ws
        except (ConnectionRefusedError, OSError) as exc:
            print(f"  waiting for gNB ({exc})", flush=True)
            time.sleep(2.0)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload) + "\n")


def parse_stdin_event(line: str, class_names: list[str]) -> dict | None:
    parts = line.strip().lower().split()
    if not parts:
        return None
    ts = datetime.now(timezone.utc).isoformat()
    if parts[0] == "jam_off":
        return {"time": ts, "event": "jammer_off"}
    if parts[0] == "jam_on" and len(parts) >= 2:
        attack = parts[1]
        if attack == "clean":
            print("Use jam_off instead of jam_on clean.", file=sys.stderr)
            return None
        if attack not in class_names:
            print(f"Unknown attack '{attack}'. Expected one of: {class_names}", file=sys.stderr)
            return None
        return {"time": ts, "event": "jammer_on", "attack": attack}
    print("Unknown event. Use: jam_on barrage | jam_on random | jam_off", file=sys.stderr)
    return None


def poll_stdin_events(
    events_log: Path,
    class_names: list[str],
    pending_jam_on: dict | None,
) -> dict | None:
    """Read optional jammer markers typed at the terminal."""
    if not select.select([sys.stdin], [], [], 0)[0]:
        return pending_jam_on
    line = sys.stdin.readline()
    event = parse_stdin_event(line, class_names)
    if event is None:
        return pending_jam_on
    append_jsonl(events_log, event)
    if event["event"] == "jammer_on":
        pending_jam_on = event
        print(f"\n>>> MARKED jammer ON ({event['attack']}) at {event['time']}\n", flush=True)
    else:
        pending_jam_on = None
        print(f"\n>>> MARKED jammer OFF at {event['time']}\n", flush=True)
    return pending_jam_on


def maybe_report_ttd(
    label: str,
    confidence: float,
    min_conf: float,
    consecutive_needed: int,
    streak_label: str | None,
    streak_count: int,
    pending_jam_on: dict | None,
    reported_ttd_for: set[str],
) -> tuple[str | None, int]:
    """Track consecutive predictions and print time-to-detection once."""
    if confidence < min_conf:
        return None, 0

    if label == streak_label:
        streak_count += 1
    else:
        streak_label = label
        streak_count = 1

    if (
        pending_jam_on
        and label == pending_jam_on.get("attack")
        and streak_count >= consecutive_needed
    ):
        key = f"{pending_jam_on['time']}:{label}"
        if key not in reported_ttd_for:
            t_jam = datetime.fromisoformat(pending_jam_on["time"])
            t_now = datetime.now(timezone.utc)
            ttd_sec = (t_now - t_jam).total_seconds()
            print(
                f"\n*** TIME TO DETECTION: {ttd_sec:.1f} s "
                f"({label}, {streak_count} consecutive @ conf>={min_conf:.2f}) ***\n",
                flush=True,
            )
            reported_ttd_for.add(key)

    return streak_label, streak_count


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if cfg.get("data_format") != "gnb":
        print("Warning: config data_format is not 'gnb'; use config_gnb.yaml", file=sys.stderr)

    live = cfg.get("live_detection") or {}
    run_id = resolve_run_id(args.run_id, cfg=cfg)
    window_size = int(cfg["window_size"])
    stride_sec = float(live.get("infer_stride_sec", cfg.get("window_stride", 15)))
    ws_timeout = float(live.get("websocket_timeout_sec", 1.0))
    min_conf = float(live.get("alert_confidence", 0.6))
    consecutive_needed = int(live.get("consecutive_alerts", 2))
    url = args.url or live.get("websocket_url", "ws://127.0.0.1:8001")

    ml_root = Path(__file__).resolve().parent
    prediction_log = args.prediction_log or ml_root / "results" / "live_gnb_predictions.jsonl"
    alert_log = args.alert_log or ml_root / "results" / "live_gnb_alerts.jsonl"
    events_log = args.events_log or ml_root / "results" / "live_gnb_events.jsonl"

    ae_model, clf_model, scaler, class_names, device = load_models(cfg, run_id)
    buffer: deque[dict] = deque(maxlen=window_size * 4)
    last_infer = 0.0
    stop = False
    pending_jam_on: dict | None = None
    reported_ttd_for: set[str] = set()
    streak_label: str | None = None
    streak_count = 0

    def _stop(*_args: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    print("Live gNB jamming detection (WebSocket → ML)")
    print(f"  run id:            {run_id}")
    print(f"  websocket:         {url}")
    print(f"  window:            {window_size} KPI rows")
    print(f"  infer every:       {stride_sec}s")
    print(f"  alert confidence:  {min_conf}")
    print(f"  consecutive alerts:{consecutive_needed}")
    print(f"  classes:           {class_names}")
    print(f"  prediction log:    {prediction_log}")
    print(f"  alert log:         {alert_log}")
    print(f"  events log:        {events_log}")
    if not args.no_stdin_events:
        print("\nStdin markers:  jam_on barrage | jam_on random | jam_off")
    print("\nWaiting for gNB KPI stream (attach UE + traffic)...\n")

    ws = connect_websocket(url)

    try:
        while not stop:
            if not args.no_stdin_events:
                pending_jam_on = poll_stdin_events(
                    events_log, class_names, pending_jam_on
                )

            try:
                ws.settimeout(ws_timeout)
                msg = ws.recv()
            except websocket.WebSocketTimeoutException:
                msg = None
            except websocket.WebSocketConnectionClosedException:
                print("WebSocket closed; reconnecting...", flush=True)
                ws = connect_websocket(url)
                continue

            if msg:
                try:
                    obj = json.loads(msg)
                except json.JSONDecodeError:
                    obj = None
                if obj and obj.get("cmd") not in ("metrics_subscribe", "metrics_unsubscribe"):
                    report = unwrap_payload(obj)
                    batch = rows_from_report(report)
                    if args.rnti is not None:
                        batch = [r for r in batch if str(r["rnti"]) == args.rnti]
                    for row in batch:
                        buffer.append(row)

            now = time.monotonic()
            ready = len(buffer) >= window_size
            due = (now - last_infer) >= stride_sec

            if ready and due:
                last_infer = now
                window = gnb_rows_to_window_matrix(list(buffer)[-window_size:], cfg)
                pred_idx, probs = predict_window(
                    ae_model, clf_model, scaler, window, device, cfg["batch_size"]
                )
                label = class_names[pred_idx]
                confidence = float(probs[pred_idx])
                ts = datetime.now(timezone.utc).isoformat()
                prob_map = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
                last_kpi_ts = buffer[-1].get("timestamp_iso", "")

                streak_label, streak_count = maybe_report_ttd(
                    label,
                    confidence,
                    min_conf,
                    consecutive_needed,
                    streak_label,
                    streak_count,
                    pending_jam_on,
                    reported_ttd_for,
                )

                line = (
                    f"[{ts}]  prediction={label:8s}  confidence={confidence:.3f}  "
                    f"probs={prob_map}"
                )
                print(line, flush=True)

                payload = {
                    "time": ts,
                    "run_id": run_id,
                    "prediction": label,
                    "confidence": confidence,
                    "probabilities": prob_map,
                    "buffer_rows": window_size,
                    "kpi_timestamp": last_kpi_ts,
                    "consecutive_streak": streak_count if label == streak_label else 1,
                }
                append_jsonl(prediction_log, payload)

                if confidence >= min_conf and streak_count >= consecutive_needed:
                    append_alert(
                        alert_log,
                        {
                            **payload,
                            "alert": True,
                            "consecutive_required": consecutive_needed,
                        },
                    )

                if args.once:
                    return

            elif not ready:
                print(f"\rBuffer: {len(buffer)}/{window_size} KPI rows...", end="", flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            ws.close()
        except Exception:
            pass
        print("\nStopped.")


if __name__ == "__main__":
    main()
