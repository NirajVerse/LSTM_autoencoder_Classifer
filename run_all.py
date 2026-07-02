#!/usr/bin/env python3
"""Train models only (AE + classifier). Evaluate separately on test_datasets_dir."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from utils import make_run_id, write_latest_run


def run(script: str, run_id: str, config: Path | None) -> None:
    path = Path(__file__).resolve().parent / script
    cmd = [sys.executable, str(path), "--run-id", run_id]
    if config is not None:
        cmd.extend(["--config", str(config)])
    print(f"\n{'=' * 60}\n>>> {script}  (--run-id {run_id})\n{'=' * 60}")
    subprocess.check_call(cmd)


def main() -> None:
    p = argparse.ArgumentParser(description="Train AE + classifier")
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Config YAML (default: config.yaml; use config_gnb.yaml for gNB KPIs)",
    )
    args = p.parse_args()

    run_id = make_run_id()
    write_latest_run(run_id)
    print(f"Training run id: {run_id}")
    if args.config:
        print(f"Config: {args.config}")
    run("train_ae.py", run_id, args.config)
    run("train_clf.py", run_id, args.config)
    cfg_hint = f" --config {args.config}" if args.config else ""
    print(
        f"\nTraining done. Collect held-out CSVs under test_datasets_dir, then run:\n"
        f"  python evaluate.py --run-id {run_id}{cfg_hint}\n"
        f"  python make_paper_figures.py --run-id {run_id}{cfg_hint}"
    )


if __name__ == "__main__":
    main()
