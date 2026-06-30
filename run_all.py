#!/usr/bin/env python3
"""Run full pipeline: train AE -> train classifier -> evaluate."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from utils import make_run_id, write_latest_run


def run(script: str, run_id: str) -> None:
    path = Path(__file__).resolve().parent / script
    print(f"\n{'=' * 60}\n>>> {script}  (--run-id {run_id})\n{'=' * 60}")
    subprocess.check_call([sys.executable, str(path), "--run-id", run_id])


def main() -> None:
    run_id = make_run_id()
    write_latest_run(run_id)
    print(f"Pipeline run id: {run_id}")
    run("train_ae.py", run_id)
    run("train_clf.py", run_id)
    run("evaluate.py", run_id)
    print(f"\nDone. Checkpoints and results saved under run id: {run_id}")


if __name__ == "__main__":
    main()
