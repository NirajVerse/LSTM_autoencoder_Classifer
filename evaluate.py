#!/usr/bin/env python3
"""Evaluate autoencoder + classifier on dataset windows."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from load_data import csv_to_windows, load_dataset_from_folders, samples_to_arrays
from training import load_ae_checkpoint, load_clf_checkpoint, predict_windows, transform_windows
from utils import (
    get_device,
    get_run_dirs,
    load_config,
    resolve_path,
    resolve_run_id,
    save_json,
    set_seed,
    update_run_info,
    write_latest_run,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate jamming classifier")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run folder (default: latest training run)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()

    run_id = resolve_run_id(args.run_id, cfg=cfg)
    ckpt_dir, results_dir = get_run_dirs(cfg, run_id)
    if run_id != "_legacy":
        results_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir = resolve_path(cfg["datasets_dir"])
    print(f"Run id: {run_id}")

    ae_path = ckpt_dir / "autoencoder.pt"
    clf_path = ckpt_dir / "classifier.pt"
    if not ae_path.exists() or not clf_path.exists():
        raise FileNotFoundError("Run train_ae.py and train_clf.py first.")

    class_names = cfg["classes"]
    samples = load_dataset_from_folders(datasets_dir, class_names, cfg)
    x, y, _ = samples_to_arrays(samples)

    ae_model, scaler, _ = load_ae_checkpoint(ae_path, cfg)
    clf_model, _ = load_clf_checkpoint(clf_path, cfg)
    ae_model.to(device)
    clf_model.to(device)

    x = transform_windows(x, scaler)
    preds = predict_windows(ae_model, clf_model, x, cfg["batch_size"])

    report = classification_report(y, preds, target_names=class_names, digits=3)
    cm = confusion_matrix(y, preds).tolist()

    print("Classification report:\n")
    print(report)
    print("Confusion matrix (rows=true, cols=pred):")
    print("labels:", class_names)
    for row, name in zip(cm, class_names):
        print(f"  {name}: {row}")

    per_file: list[dict] = []
    for csv_path in sorted(datasets_dir.glob("*/*.csv")):
        label_name = csv_path.parent.name
        if label_name not in class_names:
            continue
        label = class_names.index(label_name)
        file_samples = csv_to_windows(csv_path, label, label_name, cfg)
        if not file_samples:
            continue
        fx = np.stack([s.x for s in file_samples], axis=0)
        fx = transform_windows(fx, scaler)
        fp = predict_windows(ae_model, clf_model, fx, cfg["batch_size"])
        vote = int(np.bincount(fp, minlength=len(class_names)).argmax())
        per_file.append(
            {
                "file": str(csv_path.relative_to(datasets_dir.parent)),
                "true_label": label_name,
                "predicted": class_names[vote],
                "correct": vote == label,
                "window_preds": {class_names[i]: int((fp == i).sum()) for i in range(len(class_names))},
            }
        )

    out = {
        "run_id": run_id,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "classification_report": report,
        "confusion_matrix": cm,
        "class_names": class_names,
        "per_file_majority_vote": per_file,
    }
    save_json(results_dir / "evaluation.json", out)
    update_run_info(
        results_dir,
        evaluation_completed_at=out["evaluated_at"],
        evaluation_json=str(results_dir / "evaluation.json"),
    )
    write_latest_run(run_id)
    print(f"\nSaved results to {results_dir / 'evaluation.json'}")
    print(f"Run id: {run_id}")


if __name__ == "__main__":
    main()
