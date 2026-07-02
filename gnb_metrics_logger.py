#!/usr/bin/env python3
"""
Listen for srsRAN Project gNB JSON scheduler metrics (UDP) and append rows to CSV.

NOTE: Only works on NEWER srsRAN (docker metrics server, UDP :55555).
Lab build commit ~4bf154 uses WebSocket instead — use gnb_metrics_ws_logger.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CSV_COLUMNS = [
    "timestamp_iso",
    "timestamp_unix",
    "pci",
    "rnti",
    "cqi",
    "ri",
    "dl_mcs",
    "dl_brate",
    "dl_nof_ok",
    "dl_nof_nok",
    "dl_bler_pct",
    "dl_bs",
    "pusch_snr_db",
    "pucch_snr_db",
    "ul_mcs",
    "ul_brate",
    "ul_nof_ok",
    "ul_nof_nok",
    "ul_bler_pct",
    "bsr",
    "ta_ns",
]


def _bler_pct(ok: int, nok: int) -> float:
    total = ok + nok
    return 100.0 * nok / total if total else 0.0


def _flatten_ue(timestamp: float, ue_info: dict[str, Any]) -> dict[str, Any]:
    ue = dict(ue_info.get("ue_container", ue_info))
    dl_ok = int(ue.get("dl_nof_ok", 0))
    dl_nok = int(ue.get("dl_nof_nok", 0))
    ul_ok = int(ue.get("ul_nof_ok", 0))
    ul_nok = int(ue.get("ul_nof_nok", 0))
    ts = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return {
        "timestamp_iso": ts.isoformat(),
        "timestamp_unix": timestamp,
        "pci": ue.get("pci", ""),
        "rnti": ue.get("rnti", ""),
        "cqi": ue.get("cqi", ""),
        "ri": ue.get("ri", ""),
        "dl_mcs": ue.get("dl_mcs", ""),
        "dl_brate": ue.get("dl_brate", ""),
        "dl_nof_ok": dl_ok,
        "dl_nof_nok": dl_nok,
        "dl_bler_pct": round(_bler_pct(dl_ok, dl_nok), 3),
        "dl_bs": ue.get("dl_bs", ""),
        "pusch_snr_db": ue.get("pusch_snr_db", ""),
        "pucch_snr_db": ue.get("pucch_snr_db", ""),
        "ul_mcs": ue.get("ul_mcs", ""),
        "ul_brate": ue.get("ul_brate", ""),
        "ul_nof_ok": ul_ok,
        "ul_nof_nok": ul_nok,
        "ul_bler_pct": round(_bler_pct(ul_ok, ul_nok), 3),
        "bsr": ue.get("bsr", ""),
        "ta_ns": ue.get("ta_ns", ""),
    }


def _parse_json_stream(text: str) -> tuple[list[dict[str, Any]], str]:
    """Split concatenated JSON objects (same logic as srsRAN metrics_server)."""
    parsed: list[dict[str, Any]] = []
    header = ""
    *items, remainder = text.split("}{")
    for item in items:
        chunk = header + item + "}"
        try:
            parsed.append(json.loads(chunk))
        except json.JSONDecodeError:
            print(f"warn: bad json chunk: {chunk[:120]}...", file=sys.stderr)
        header = "{"
    if remainder:
        remainder = header + remainder
        try:
            parsed.append(json.loads(remainder))
            remainder = ""
        except json.JSONDecodeError:
            pass
    return parsed, remainder


def run(host: str, port: int, output: Path, rnti_filter: str | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists() or output.stat().st_size == 0

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.settimeout(1.0)

    stop = False

    def _stop(*_args: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    print(f"Listening on {host}:{port} → {output}", flush=True)
    buffer = ""
    rows_written = 0

    with output.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()

        while not stop:
            try:
                chunk = sock.recv(1 << 20).decode()
            except socket.timeout:
                continue
            if not chunk:
                break
            buffer += chunk.strip()
            messages, buffer = _parse_json_stream(buffer)
            for msg in messages:
                if "ue_list" not in msg:
                    continue
                ts = float(msg.get("timestamp", datetime.now(tz=timezone.utc).timestamp()))
                for ue_info in msg["ue_list"]:
                    row = _flatten_ue(ts, ue_info)
                    if rnti_filter is not None and str(row["rnti"]) != rnti_filter:
                        continue
                    writer.writerow(row)
                    rows_written += 1
                    if rows_written % 30 == 0:
                        fh.flush()
                        print(f"  {rows_written} rows", flush=True)

    print(f"Stopped. Wrote {rows_written} rows to {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Log srsRAN gNB JSON metrics to CSV")
    parser.add_argument("--host", default="0.0.0.0", help="UDP bind address")
    parser.add_argument("--port", type=int, default=55555, help="UDP port (match gNB metrics.port)")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--rnti", default=None, help="Optional hex RNTI filter, e.g. 4601")
    args = parser.parse_args()
    run(args.host, args.port, args.output, args.rnti)


if __name__ == "__main__":
    main()
