#!/usr/bin/env python3
"""
Capture srsRAN gNB scheduler KPIs to CSV for ML training.

Start this before or after the gNB (waits for WebSocket). Requires gNB yaml with
remote_control + enable_json
(see ran-tester-ue/configs/srsran/gnb_uhd.yaml).

  cd milcom/ml && source .venv/bin/activate
  python gnb_capture.py --output ../datasets_gnb/clean/run_001.csv

While capturing: attach UE, run UL/DL traffic (ping/iperf), enable jammer for
barrage/random runs. Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

from gnb_metrics_io import rows_from_report, unwrap_payload, write_csv_rows

try:
    import websocket
except ImportError:
    print("Install: pip install websocket-client", file=sys.stderr)
    raise SystemExit(1)


def capture(url: str, output: Path, rnti_filter: str | None, wait_sec: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    stop = False

    def _stop(*_args: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    print(f"Connecting to {url}", flush=True)
    print(f"Writing CSV → {output}", flush=True)
    print("Waiting for gNB WebSocket (start gNB if not running)...", flush=True)
    ws = None
    deadline = time.monotonic() + wait_sec if wait_sec > 0 else None
    while not stop:
        try:
            ws = websocket.create_connection(url, timeout=5)
            break
        except (ConnectionRefusedError, OSError):
            if deadline is not None and time.monotonic() >= deadline:
                print(
                    f"Timed out waiting for {url}. Is gNB running with remote_control enabled?",
                    file=sys.stderr,
                )
                return
            time.sleep(2.0)
    if ws is None:
        return
    ws.send(json.dumps({"cmd": "metrics_subscribe"}))
    print(f"Subscribe: {ws.recv()}", flush=True)

    rows_written = 0
    while not stop:
        try:
            ws.settimeout(1.0)
            msg = ws.recv()
        except websocket.WebSocketTimeoutException:
            continue
        if not msg:
            break
        try:
            obj = json.loads(msg)
        except json.JSONDecodeError:
            continue
        if obj.get("cmd") in ("metrics_subscribe", "metrics_unsubscribe"):
            continue

        report = unwrap_payload(obj)
        batch = rows_from_report(report)
        if rnti_filter is not None:
            batch = [r for r in batch if str(r["rnti"]) == rnti_filter]
        if not batch:
            continue

        n = write_csv_rows(output, batch, append=True)
        rows_written += n
        if rows_written % 30 == 0:
            print(f"  {rows_written} rows", flush=True)

    ws.close()
    print(f"Stopped. {rows_written} rows in {output}", flush=True)
    if rows_written == 0:
        print(
            "No rows captured. Check: gNB running, UE attached, remote_control enabled, "
            "enable_json + enable_sched in gNB yaml.",
            file=sys.stderr,
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Capture gNB WebSocket scheduler metrics to CSV")
    p.add_argument("--url", default="ws://127.0.0.1:8001", help="gNB remote_control WebSocket URL")
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV, e.g. ../datasets_gnb/clean/run_001.csv",
    )
    p.add_argument("--rnti", default=None, help="Optional RNTI filter (decimal or hex string)")
    p.add_argument(
        "--wait-sec",
        type=float,
        default=0.0,
        help="Max seconds to wait for gNB WebSocket (0 = wait forever)",
    )
    args = p.parse_args()
    capture(args.url, args.output, args.rnti, args.wait_sec)


if __name__ == "__main__":
    main()
