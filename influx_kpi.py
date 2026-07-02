"""Read rtue KPI streams from InfluxDB (O-RAN KPI-moni path).

rtue posts ``rtue_carrier_metric`` points (see rtue/src/metrics_influxdb.cc).
This module maps Influx fields to the same column names used by CSV training.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

try:
    from influxdb_client import InfluxDBClient
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "influxdb-client is required for live detection. "
        "Install with: pip install influxdb-client"
    ) from exc

from features import build_feature_frame, feature_column_names


@dataclass
class InfluxConfig:
    host: str
    port: int
    org: str
    token: str
    bucket: str
    ue_data_id: str
    carrier_type: str = "nr"

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> "InfluxConfig":
        influx = cfg.get("influx") or {}
        return cls(
            host=str(influx.get("host", "127.0.0.1")),
            port=int(influx.get("port", 8086)),
            org=str(influx.get("org", "rtu")),
            token=str(influx.get("token", "")),
            bucket=str(influx.get("bucket", "rtusystem")),
            ue_data_id=str(influx.get("ue_data_id", "ue1_uhd")),
            carrier_type=str(influx.get("carrier_type", "nr")),
        )

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if pd.isna(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def influx_record_to_kpi_row(record: dict[str, Any]) -> dict[str, float]:
    """Map one pivoted Influx row to ML KPI columns (CSV-compatible names)."""
    rx_pkts = _safe_float(record.get("rx_pkts"))
    rx_errors = _safe_float(record.get("rx_errors"))
    tx_pkts = _safe_float(record.get("tx_pkts"))
    tx_errors = _safe_float(record.get("tx_errors"))

    dl_bler = 100.0 * rx_errors / rx_pkts if rx_pkts > 0 else 0.0
    ul_bler = 100.0 * tx_errors / tx_pkts if tx_pkts > 0 else 0.0

    return {
        "rsrp": _safe_float(record.get("rsrp")),
        "cfo": _safe_float(record.get("cfo")),
        "dl_mcs": _safe_float(record.get("dl_mcs")),
        "dl_snr": _safe_float(record.get("sinr")),
        "dl_turbo": _safe_float(record.get("fec_iters")),
        "dl_brate": _safe_float(record.get("rx_brate")),
        "dl_bler": dl_bler,
        "ul_mcs": _safe_float(record.get("ul_mcs")),
        "ul_brate": _safe_float(record.get("tx_brate")),
        "ul_bler": ul_bler,
        "ul_buff": _safe_float(record.get("ul_buffer")),
        "ul_ta": _safe_float(record.get("ta_us")),
    }


class InfluxKpiReader:
    """Poll rtue carrier metrics written by KPI-moni (rtue → Influx)."""

    def __init__(self, cfg: InfluxConfig) -> None:
        self.cfg = cfg
        self._client = InfluxDBClient(
            url=cfg.url,
            token=cfg.token,
            org=cfg.org,
        )
        self._query_api = self._client.query_api()
        self._last_time: datetime | None = None

    def close(self) -> None:
        self._client.close()

    def _flux_since(self, lookback_sec: float) -> str:
        start = f"-{int(max(lookback_sec, 1))}s"
        if self._last_time is not None:
            # Influx expects RFC3339 timestamps in Flux
            ts = self._last_time.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            start = ts
        return start

    def fetch_new_rows(self, lookback_sec: float = 120.0) -> list[dict[str, Any]]:
        """Return KPI rows newer than the last fetch (oldest → newest)."""
        start = self._flux_since(lookback_sec)
        flux = f'''
from(bucket: "{self.cfg.bucket}")
  |> range(start: {start})
  |> filter(fn: (r) => r._measurement == "rtue_carrier_metric")
  |> filter(fn: (r) => r.rtue_data_id == "{self.cfg.ue_data_id}")
  |> filter(fn: (r) => r.carrier_type == "{self.cfg.carrier_type}")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> sort(columns: ["_time"])
'''
        tables = self._query_api.query(flux, org=self.cfg.org)
        rows: list[dict[str, Any]] = []
        for table in tables:
            for record in table.records:
                ts = record.get_time()
                if ts is None:
                    continue
                if self._last_time is not None and ts <= self._last_time:
                    continue
                row = {"_time": ts}
                for key, value in record.values.items():
                    if key.startswith("_") or key in {"result", "table"}:
                        continue
                    row[key] = value
                rows.append(row)

        if rows:
            self._last_time = rows[-1]["_time"]
        return rows

    def reset_cursor(self) -> None:
        self._last_time = None


def kpi_rows_to_window_matrix(
    kpi_rows: list[dict[str, float]],
    cfg: dict[str, Any],
) -> Any:
    """Build (1, window_size, n_features) float32 array from consecutive KPI rows."""
    import numpy as np

    window_size = int(cfg["window_size"])
    if len(kpi_rows) < window_size:
        raise ValueError(f"Need {window_size} KPI rows, got {len(kpi_rows)}")

    raw_features = cfg["raw_features"]
    engineered = cfg["engineered_features"]
    all_features = feature_column_names(raw_features, engineered)

    tail = kpi_rows[-window_size:]
    df = pd.DataFrame(tail)
    feat_df = build_feature_frame(df, raw_features, cfg["roll_window"])
    values = feat_df[all_features].to_numpy(dtype=np.float32)
    return values.reshape(1, window_size, len(all_features))
