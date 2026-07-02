#!/usr/bin/env python3
"""
Summarize gNB KPI CSVs for clean vs barrage vs random jamming observation.

Expects:
  datasets_gnb/clean/run_001.csv
  datasets_gnb/barrage/run_001.csv
  datasets_gnb/random/run_001.csv

Or pass --root to a folder with those class subdirs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_CLASSES = ("clean", "barrage", "random")
KEY_METRICS = (
    "dl_bler_pct",
    "ul_bler_pct",
    "dl_brate",
    "ul_brate",
    "cqi",
    "ri",
    "dl_mcs",
    "ul_mcs",
    "pusch_snr_db",
    "pucch_snr_db",
)


def _load_class_csvs(root: Path, label: str) -> pd.DataFrame:
    class_dir = root / label
    if not class_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {class_dir}")
    frames: list[pd.DataFrame] = []
    for path in sorted(class_dir.glob("*.csv")):
        df = pd.read_csv(path)
        df["scenario"] = label
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No CSV files in {class_dir}")
    return pd.concat(frames, ignore_index=True)


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in KEY_METRICS:
        if metric not in df.columns:
            continue
        series = pd.to_numeric(df[metric], errors="coerce").dropna()
        if series.empty:
            continue
        rows.append(
            {
                "metric": metric,
                "mean": round(series.mean(), 3),
                "std": round(series.std(), 3),
                "p50": round(series.quantile(0.5), 3),
                "p95": round(series.quantile(0.95), 3),
                "min": round(series.min(), 3),
                "max": round(series.max(), 3),
                "n": len(series),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare gNB KPI traces across jamming scenarios")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "datasets_gnb",
        help="Root folder with clean/, barrage/, random/ subdirs",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(DEFAULT_CLASSES),
        help="Scenario folder names",
    )
    args = parser.parse_args()

    print(f"gNB scenario comparison — root={args.root}\n")
    all_summaries: dict[str, pd.DataFrame] = {}

    for label in args.classes:
        df = _load_class_csvs(args.root, label)
        summary = _summarize(df)
        all_summaries[label] = summary
        print(f"=== {label.upper()} ({len(df)} rows) ===")
        if summary.empty:
            print("  (no numeric metrics found)\n")
            continue
        for _, row in summary.iterrows():
            print(
                f"  {row['metric']:16s}  mean={row['mean']:>10}  "
                f"std={row['std']:>8}  p50={row['p50']:>8}  p95={row['p95']:>8}"
            )
        print()

    # Side-by-side BLER / throughput (most useful for jamming)
    focus = ["dl_bler_pct", "dl_brate", "cqi", "pusch_snr_db"]
    print("=== Quick comparison (mean) ===")
    header = f"{'metric':16s}" + "".join(f"{c:>12s}" for c in args.classes)
    print(header)
    for metric in focus:
        line = f"{metric:16s}"
        for label in args.classes:
            summary = all_summaries.get(label)
            if summary is None or summary.empty:
                line += f"{'n/a':>12s}"
                continue
            hit = summary.loc[summary["metric"] == metric, "mean"]
            line += f"{(hit.iloc[0] if len(hit) else float('nan')):>12.3f}"
        print(line)


if __name__ == "__main__":
    main()
