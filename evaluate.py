#!/usr/bin/env python3
"""Evaluate trained models on the held-out TEST dataset only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from load_data import csv_to_windows, load_dataset_from_folders, samples_to_arrays
from training import load_ae_checkpoint, load_clf_checkpoint, predict_windows, transform_windows
from utils import (
    ensure_train_test_separate,
    get_device,
    get_run_dirs,
    get_test_datasets_dir,
    get_train_datasets_dir,
    load_config,
    resolve_run_id,
    save_json,
    set_seed,
    update_run_info,
    validate_datasets_layout,
    write_latest_run,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate jamming classifier on held-out test CSVs (not training data)"
    )
    p.add_argument("--config", type=Path, default=None)
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Training run whose checkpoints to load (default: latest)",
    )
    p.add_argument(
        "--test-datasets",
        type=Path,
        default=None,
        help="Override test_datasets_dir from config.yaml",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    ensure_train_test_separate(cfg)

    run_id = resolve_run_id(args.run_id, cfg=cfg)
    ckpt_dir, results_dir = get_run_dirs(cfg, run_id)
    if run_id != "_legacy":
        results_dir.mkdir(parents=True, exist_ok=True)

    train_dir = get_train_datasets_dir(cfg)
    test_dir = get_test_datasets_dir(cfg, args.test_datasets)
    class_names = cfg["classes"]
    validate_datasets_layout(test_dir, class_names, purpose="Test")

    print(f"Run id: {run_id}")
    print(f"  train data (not used here): {train_dir}")
    print(f"  test data:                  {test_dir}")

    ae_path = ckpt_dir / "autoencoder.pt"
    clf_path = ckpt_dir / "classifier.pt"
    if not ae_path.exists() or not clf_path.exists():
        raise FileNotFoundError("Run train_ae.py and train_clf.py first.")

    samples = load_dataset_from_folders(test_dir, class_names, cfg)
    x, y, _ = samples_to_arrays(samples)
    print(f"Loaded {len(x)} test windows")
    for i, name in enumerate(class_names):
        print(f"  {name}: {int((y == i).sum())} windows")

    ae_model, scaler, _ = load_ae_checkpoint(ae_path, cfg)
    clf_model, _ = load_clf_checkpoint(clf_path, cfg)
    ae_model.to(device)
    clf_model.to(device)

    x = transform_windows(x, scaler)
    preds = predict_windows(ae_model, clf_model, x, cfg["batch_size"])

    report = classification_report(y, preds, target_names=class_names, digits=3)
    cm = confusion_matrix(y, preds).tolist()
    accuracy = float((preds == y).mean())

    print("\nTest-set classification report:\n")
    print(report)
    print("Confusion matrix (rows=true, cols=pred):")
    print("labels:", class_names)
    for row, name in zip(cm, class_names):
        print(f"  {name}: {row}")

    per_file: list[dict] = []
    for csv_path in sorted(test_dir.glob("*/*.csv")):
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
                "file": str(csv_path.relative_to(test_dir.parent)),
                "true_label": label_name,
                "predicted": class_names[vote],
                "correct": vote == label,
                "window_preds": {class_names[i]: int((fp == i).sum()) for i in range(len(class_names))},
            }
        )

    out = {
        "run_id": run_id,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "split": "test",
        "train_datasets_dir": str(train_dir),
        "test_datasets_dir": str(test_dir),
        "n_windows": int(len(y)),
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm,
        "class_names": class_names,
        "per_file_majority_vote": per_file,
    }
    save_json(results_dir / "test_evaluation.json", out)
    update_run_info(
        results_dir,
        test_evaluation_completed_at=out["evaluated_at"],
        test_evaluation_json=str(results_dir / "test_evaluation.json"),
        test_accuracy=accuracy,
        test_datasets_dir=str(test_dir),
    )
    write_latest_run(run_id)
    print(f"\nTest accuracy: {accuracy * 100:.1f}%  ({int((preds == y).sum())}/{len(y)} windows)")
    print(f"Saved results to {results_dir / 'test_evaluation.json'}")
    print(f"Run id: {run_id}")


if __name__ == "__main__":
    main()
