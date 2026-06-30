"""Feature extraction from rtue metrics CSV rows."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def build_feature_frame(df: pd.DataFrame, raw_features: list[str], roll_window: int) -> pd.DataFrame:
    """Return DataFrame with raw + engineered columns."""
    out = pd.DataFrame(index=df.index)
    for col in raw_features:
        out[col] = _numeric_series(df, col)

    out["dl_snr_diff"] = out["dl_snr"].diff().fillna(0.0)
    out["dl_snr_roll_std"] = (
        out["dl_snr"].rolling(roll_window, min_periods=1).std().fillna(0.0)
    )
    out["dl_bler_roll_mean"] = (
        out["dl_bler"].rolling(roll_window, min_periods=1).mean().fillna(0.0)
    )
    out["quality_divergence"] = out["rsrp"] - out["dl_snr"]

    return out


def feature_column_names(raw_features: list[str], engineered_features: list[str]) -> list[str]:
    return list(raw_features) + list(engineered_features)
