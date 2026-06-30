#!/usr/bin/env python3
"""Predict attack class for a single rtue metrics CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from load_data import csv_to_windows
from training import (
    load_ae_checkpoint,
    load_clf_checkpoint,
    predict_proba_windows,
    transform_windows,
)
from utils import get_device, get_run_dirs, load_config, resolve_run_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict jamming class from rtue CSV")
    p.add_argument("--csv", type=Path, required=True, help="Path to rtue metrics CSV")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run folder (default: latest training run)",
    )
    p.add_argument("--label", type=str, default=None, help="Optional ground-truth for comparison")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device()

    run_id = resolve_run_id(args.run_id, cfg=cfg)
    ckpt_dir, _ = get_run_dirs(cfg, run_id)
    class_names = cfg["classes"]
    print(f"Run id: {run_id}")

    ae_model, scaler, _ = load_ae_checkpoint(ckpt_dir / "autoencoder.pt", cfg)
    clf_model, _ = load_clf_checkpoint(ckpt_dir / "classifier.pt", cfg)
    ae_model.to(device)
    clf_model.to(device)

    samples = csv_to_windows(args.csv.resolve(), label=-1, label_name="unknown", cfg=cfg)
    if not samples:
        raise ValueError(f"No windows extracted from {args.csv}")

    x = np.stack([s.x for s in samples], axis=0)
    x = transform_windows(x, scaler)
    probs = predict_proba_windows(ae_model, clf_model, x, cfg["batch_size"])
    preds = probs.argmax(axis=1)

    counts = np.bincount(preds, minlength=len(class_names))
    majority = int(counts.argmax())
    avg_probs = probs.mean(axis=0)

    print(f"File: {args.csv}")
    print(f"Windows: {len(samples)}")
    print(f"Prediction (majority vote): {class_names[majority]}")
    print("Average class probabilities:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {avg_probs[i]:.3f}  (windows={counts[i]})")

    if args.label:
        ok = args.label.lower() == class_names[majority]
        print(f"Ground truth: {args.label}  -> {'CORRECT' if ok else 'WRONG'}")


if __name__ == "__main__":
    main()
