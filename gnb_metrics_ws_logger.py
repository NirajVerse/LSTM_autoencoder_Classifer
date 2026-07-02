#!/usr/bin/env python3
"""
Save gNB scheduler KPIs via WebSocket (srsRAN commit ~4bf154 / OCUDU).

Your gNB does NOT send metrics on UDP :55555. It exposes JSON on the remote_control
WebSocket server (default ws://127.0.0.1:8001) after you subscribe.

Requires: pip install websocket-client

Usage (start BEFORE gNB):
  python gnb_metrics_ws_logger.py --output ../datasets_gnb/clean/run_001.jsonl
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import websocket
except ImportError:
    print("Install dependency: pip install websocket-client", file=sys.stderr)
    raise SystemExit(1)


def run(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    stop = False

    def _stop(*_args: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    print(f"Connecting to {url}", flush=True)
    ws = websocket.create_connection(url, timeout=10)
    ws.send(json.dumps({"cmd": "metrics_subscribe"}))
    resp = ws.recv()
    print(f"Subscribe response: {resp}", flush=True)

    rows = 0
    with output.open("a") as fh:
        while not stop:
            try:
                ws.settimeout(1.0)
                msg = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            if not msg:
                break
            # Skip command acks; save metric JSON payloads.
            try:
                obj = json.loads(msg)
            except json.JSONDecodeError:
                fh.write(msg.strip() + "\n")
                fh.flush()
                rows += 1
                continue
            if obj.get("cmd") in ("metrics_subscribe", "metrics_unsubscribe"):
                continue
            record = {"received_at": datetime.now(tz=timezone.utc).isoformat(), "payload": obj}
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            rows += 1
            if rows % 30 == 0:
                print(f"  {rows} messages saved", flush=True)

    ws.close()
    print(f"Stopped. Saved {rows} messages to {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Log gNB JSON metrics via WebSocket")
    parser.add_argument("--url", default="ws://127.0.0.1:8001", help="remote_control WebSocket URL")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file")
    args = parser.parse_args()
    run(args.url, args.output)


if __name__ == "__main__":
    main()
