"""Load rtue CSV files, filter rows, build labeled windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from features import build_feature_frame, feature_column_names


@dataclass
class WindowSample:
    x: np.ndarray  # (window_size, n_features)
    label: int
    label_name: str
    source_file: str
    start_row: int


def read_rtue_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", comment="#")
    return df.reset_index(drop=True)


def filter_rows(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    out["rsrp"] = pd.to_numeric(out["rsrp"], errors="coerce")
    out["dl_snr"] = pd.to_numeric(out["dl_snr"], errors="coerce")
    out["dl_brate"] = pd.to_numeric(out["dl_brate"], errors="coerce")
    out["ul_brate"] = pd.to_numeric(out["ul_brate"], errors="coerce")

    if cfg.get("trim_bad_tail", True):
        bad = (out["rsrp"] < cfg["bad_rsrp_threshold"]) | (
            out["dl_snr"] < cfg["bad_dl_snr_threshold"]
        )
        if bad.any():
            first_bad = int(np.argmax(bad.to_numpy()))
            if first_bad > 0:
                out = out.iloc[:first_bad]

    if cfg.get("require_traffic", False):
        mask = (out["dl_brate"] > 0) | (out["ul_brate"] > 0)
        out = out.loc[mask]

    return out.reset_index(drop=True)


def csv_to_windows(
    path: Path,
    label: int,
    label_name: str,
    cfg: dict,
) -> list[WindowSample]:
    raw_features = cfg["raw_features"]
    engineered = cfg["engineered_features"]
    all_features = feature_column_names(raw_features, engineered)
    window_size = cfg["window_size"]
    stride = cfg["window_stride"]

    df = filter_rows(read_rtue_csv(path), cfg)
    if len(df) < window_size:
        return []

    feat_df = build_feature_frame(df, raw_features, cfg["roll_window"])
    values = feat_df[all_features].to_numpy(dtype=np.float32)

    samples: list[WindowSample] = []
    for start in range(0, len(values) - window_size + 1, stride):
        window = values[start : start + window_size]
        samples.append(
            WindowSample(
                x=window,
                label=label,
                label_name=label_name,
                source_file=str(path),
                start_row=start,
            )
        )
    return samples


def load_dataset_from_folders(datasets_dir: Path, class_names: list[str], cfg: dict) -> list[WindowSample]:
    all_samples: list[WindowSample] = []
    for label, name in enumerate(class_names):
        class_dir = datasets_dir / name
        if not class_dir.is_dir():
            continue
        csv_files = sorted(class_dir.glob("*.csv"))
        for csv_path in csv_files:
            all_samples.extend(csv_to_windows(csv_path, label, name, cfg))
    return all_samples


def samples_to_arrays(samples: list[WindowSample]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not samples:
        raise ValueError("No window samples loaded — check datasets/ paths and CSV files.")
    x = np.stack([s.x for s in samples], axis=0)
    y = np.array([s.label for s in samples], dtype=np.int64)
    labels = [s.label_name for s in samples]
    return x, y, labels
