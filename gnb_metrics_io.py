"""Parse srsRAN gNB scheduler metrics (WebSocket JSON / JSONL) into flat CSV rows."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# Columns consumed by ml/config_gnb.yaml and load_data.py (data_format: gnb).
GNB_CSV_COLUMNS = [
    "timestamp_iso",
    "timestamp_unix",
    "pci",
    "rnti",
    "cqi",
    "dl_ri",
    "dl_mcs",
    "dl_brate",
    "dl_nof_ok",
    "dl_nof_nok",
    "dl_bler_pct",
    "dl_bs",
    "pusch_snr_db",
    "pusch_rsrp_db",
    "ul_mcs",
    "ul_brate",
    "ul_nof_ok",
    "ul_nof_nok",
    "ul_bler_pct",
    "bsr",
    "ta_ns",
    "phr",
]


def bler_pct(ok: int, nok: int) -> float:
    total = ok + nok
    return 100.0 * nok / total if total else 0.0


def _num(value: Any, default: float = 0.0) -> float:
    if value is None or value == "" or value == "n/a":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    return int(_num(value, float(default)))


def unwrap_payload(record: dict[str, Any]) -> dict[str, Any]:
    """JSONL lines from gnb_metrics_ws_logger use {received_at, payload}."""
    if "payload" in record and isinstance(record["payload"], dict):
        return record["payload"]
    return record


def iter_ue_metric_pairs(report: dict[str, Any]) -> Iterator[tuple[float | None, dict[str, Any]]]:
    """Yield (timestamp, ue_dict) from srsRAN scheduler JSON (several shapes)."""
    ts = report.get("timestamp")
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except ValueError:
            ts = None
    elif ts is not None:
        ts = float(ts)

    if "cells" in report and isinstance(report["cells"], list):
        for cell in report["cells"]:
            ue_list = cell.get("ue_list") or []
            for ue in ue_list:
                if isinstance(ue, dict):
                    yield ts, ue
        return

    if "ue_list" in report and isinstance(report["ue_list"], list):
        for ue in report["ue_list"]:
            if isinstance(ue, dict):
                yield ts, ue


def flatten_ue_row(timestamp: float | None, ue: dict[str, Any]) -> dict[str, Any]:
    """One scheduler UE metric object → CSV row."""
    raw = dict(ue.get("ue_container", ue))
    dl_ok = _int(raw.get("dl_nof_ok"))
    dl_nok = _int(raw.get("dl_nof_nok"))
    ul_ok = _int(raw.get("ul_nof_ok"))
    ul_nok = _int(raw.get("ul_nof_nok"))

    if timestamp is None:
        timestamp = datetime.now(tz=timezone.utc).timestamp()
    ts_dt = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)

    return {
        "timestamp_iso": ts_dt.isoformat(),
        "timestamp_unix": float(timestamp),
        "pci": raw.get("pci", ""),
        "rnti": raw.get("rnti", ""),
        "cqi": raw.get("cqi", ""),
        "dl_ri": raw.get("dl_ri", raw.get("ri", "")),
        "dl_mcs": raw.get("dl_mcs", ""),
        "dl_brate": raw.get("dl_brate", ""),
        "dl_nof_ok": dl_ok,
        "dl_nof_nok": dl_nok,
        "dl_bler_pct": round(bler_pct(dl_ok, dl_nok), 3),
        "dl_bs": raw.get("dl_bs", ""),
        "pusch_snr_db": raw.get("pusch_snr_db", ""),
        "pusch_rsrp_db": raw.get("pusch_rsrp_db", ""),
        "ul_mcs": raw.get("ul_mcs", ""),
        "ul_brate": raw.get("ul_brate", ""),
        "ul_nof_ok": ul_ok,
        "ul_nof_nok": ul_nok,
        "ul_bler_pct": round(bler_pct(ul_ok, ul_nok), 3),
        "bsr": raw.get("bsr", ""),
        "ta_ns": raw.get("ta_ns", raw.get("pusch_ta_ns", "")),
        "phr": raw.get("last_phr", raw.get("phr", "")),
    }


def rows_from_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ts, ue in iter_ue_metric_pairs(report):
        rows.append(flatten_ue_row(ts, ue))
    return rows


def write_csv_rows(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    append: bool = False,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not append or not path.exists() or path.stat().st_size == 0
    mode = "a" if append else "w"
    with path.open(mode, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=GNB_CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


def jsonl_to_csv(
    jsonl_path: Path,
    csv_path: Path,
    *,
    rnti_filter: str | None = None,
) -> int:
    """Convert gnb_metrics_ws_logger JSONL → training CSV."""
    batch: list[dict[str, Any]] = []

    with jsonl_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            report = unwrap_payload(record)
            for row in rows_from_report(report):
                if rnti_filter is not None and str(row["rnti"]) != rnti_filter:
                    continue
                batch.append(row)

    if not batch:
        return 0
    return write_csv_rows(csv_path, batch, append=False)
