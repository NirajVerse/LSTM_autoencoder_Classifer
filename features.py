"""Feature extraction from rtue or gNB metrics CSV rows."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def build_feature_frame(
    df: pd.DataFrame,
    raw_features: list[str],
    roll_window: int,
    engineered_features: list[str] | None = None,
) -> pd.DataFrame:
    """Return DataFrame with raw + engineered columns."""
    out = pd.DataFrame(index=df.index)
    for col in raw_features:
        out[col] = _numeric_series(df, col)

    eng = engineered_features or []
    is_gnb = "dl_bler_pct" in raw_features

    if is_gnb:
        if "dl_bler_roll_mean" in eng:
            out["dl_bler_roll_mean"] = (
                out["dl_bler_pct"].rolling(roll_window, min_periods=1).mean().fillna(0.0)
            )
        if "ul_bler_roll_mean" in eng:
            out["ul_bler_roll_mean"] = (
                out["ul_bler_pct"].rolling(roll_window, min_periods=1).mean().fillna(0.0)
            )
        if "dl_brate_roll_mean" in eng:
            out["dl_brate_roll_mean"] = (
                out["dl_brate"].rolling(roll_window, min_periods=1).mean().fillna(0.0)
            )
        if "harq_stress" in eng:
            out["harq_stress"] = out["dl_nof_nok"] / (out["dl_nof_ok"] + 1.0)
    else:
        if "dl_snr_diff" in eng:
            out["dl_snr_diff"] = out["dl_snr"].diff().fillna(0.0)
        if "dl_snr_roll_std" in eng:
            out["dl_snr_roll_std"] = (
                out["dl_snr"].rolling(roll_window, min_periods=1).std().fillna(0.0)
            )
        if "dl_bler_roll_mean" in eng:
            out["dl_bler_roll_mean"] = (
                out["dl_bler"].rolling(roll_window, min_periods=1).mean().fillna(0.0)
            )
        if "quality_divergence" in eng:
            out["quality_divergence"] = out["rsrp"] - out["dl_snr"]

    return out


def feature_column_names(raw_features: list[str], engineered_features: list[str]) -> list[str]:
    return list(raw_features) + list(engineered_features)
