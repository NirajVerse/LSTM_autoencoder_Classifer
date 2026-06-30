#!/usr/bin/env python3
"""Stage 2: Train hybrid classifier on frozen encoder features."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from load_data import load_dataset_from_folders, samples_to_arrays
from models import AttackClassifier
from training import (
    load_ae_checkpoint,
    make_loader,
    save_clf_checkpoint,
    stratified_split,
    transform_windows,
)
from utils import (
    get_device,
    get_run_dirs,
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
    p = argparse.ArgumentParser(description="Train hybrid attack classifier")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run folder (default: latest from train_ae.py)",
    )
    p.add_argument(
        "--ae-checkpoint",
        type=Path,
        default=None,
        help="Path to autoencoder.pt (default: checkpoints/<run-id>/autoencoder.pt)",
    )
    return p.parse_args()


def run_epoch(
    ae_model,
    clf,
    loader,
    device,
    criterion,
    optimizer=None,
    mixup_alpha: float = 0.0,
) -> tuple[float, float]:
    is_train = optimizer is not None
    clf.train(is_train)
    ae_model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Mixup is only applied during training and only on full-size batches
        # (skip tiny last batches where pair-shuffle would degenerate).
        use_mixup = is_train and mixup_alpha > 0.0 and batch_x.size(0) >= 4
        if use_mixup:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            lam = max(lam, 1.0 - lam)  # bias toward the original sample
            idx = torch.randperm(batch_x.size(0), device=device)
            mixed_x = lam * batch_x + (1.0 - lam) * batch_x[idx]
            y_a, y_b = batch_y, batch_y[idx]
        else:
            mixed_x = batch_x

        with torch.no_grad():
            z, seq_out = ae_model.encode_sequence(mixed_x)

        z = z.detach()
        seq_out = seq_out.detach()
        mixed_x = mixed_x.detach()

        logits = clf(mixed_x, z, seq_out)
        if use_mixup:
            loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
        else:
            loss = criterion(logits, batch_y)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        # For accuracy, always evaluate against the true labels (not mixed).
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

    return total_loss / max(len(loader), 1), correct / max(total, 1)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()

    datasets_dir = get_train_datasets_dir(cfg)
    validate_datasets_layout(datasets_dir, class_names, purpose="Train")
    run_id = resolve_run_id(args.run_id, cfg=cfg)
    ckpt_dir, results_dir = get_run_dirs(cfg, run_id)
    if run_id != "_legacy":
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
    ae_path = args.ae_checkpoint or (ckpt_dir / "autoencoder.pt")
    print(f"Run id: {run_id}")

    if not ae_path.exists():
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {ae_path}. Run train_ae.py first.")

    class_names = cfg["classes"]
    samples = load_dataset_from_folders(datasets_dir, class_names, cfg)
    x, y, _ = samples_to_arrays(samples)
    print(f"Loaded {len(x)} windows across classes: {class_names}")
    for i, name in enumerate(class_names):
        print(f"  {name}: {int((y == i).sum())} windows")

    ae_model, scaler, ae_meta = load_ae_checkpoint(ae_path, cfg)
    ae_model.to(device)
    for p in ae_model.parameters():
        p.requires_grad = False

    x = transform_windows(x, scaler)
    x_train, y_train, x_val, y_val = stratified_split(x, y, cfg["val_fraction"], cfg["seed"])

    n_features = ae_meta["n_features"]
    clf = AttackClassifier(
        n_features=n_features,
        latent_size=cfg["ae_latent_size"],
        encoder_hidden=cfg["ae_hidden_size"],
        n_classes=len(class_names),
        hidden_sizes=cfg.get("clf_hidden_sizes", [128, 64]),
        dropout=cfg["clf_dropout"],
        bilstm_hidden=cfg.get("clf_bilstm_hidden", 32),
        attn_heads=cfg.get("clf_attn_heads", 4),
    ).to(device)

    label_smoothing = cfg.get("clf_label_smoothing", 0.05)

    # Class weights — important when classes are imbalanced (e.g. constant has
    # far fewer windows than the others). Computed on the training split only.
    if cfg.get("clf_use_class_weights", True):
        train_counts = np.bincount(y_train, minlength=len(class_names)).astype(np.float32)
        train_counts = np.clip(train_counts, 1.0, None)  # avoid div-by-zero
        weights = train_counts.sum() / (len(class_names) * train_counts)
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f"Class weights: {dict(zip(class_names, [f'{w:.2f}' for w in weights]))}")
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        clf.parameters(),
        lr=cfg["learning_rate_clf"],
        weight_decay=cfg.get("clf_weight_decay", 1e-4),
    )
    mixup_alpha = float(cfg.get("clf_mixup_alpha", 0.2))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-5
    )

    train_loader = make_loader(x_train, y_train, cfg["batch_size"], shuffle=True)
    val_loader = make_loader(x_val, y_val, cfg["batch_size"], shuffle=False)

    best_val_acc = 0.0
    best_epoch = 0
    patience = cfg.get("clf_early_stop_patience", 20)
    stale = 0
    history: list[dict] = []

    for epoch in range(1, cfg["clf_epochs"] + 1):
        train_loss, train_acc = run_epoch(
            ae_model, clf, train_loader, device, criterion, optimizer,
            mixup_alpha=mixup_alpha,
        )
        with torch.no_grad():
            val_loss, val_acc = run_epoch(ae_model, clf, val_loader, device, criterion)

        scheduler.step(val_acc)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            stale = 0
            meta = {
                "n_classes": len(class_names),
                "class_names": class_names,
                "latent_size": cfg["ae_latent_size"],
                "n_features": n_features,
                "ae_checkpoint": str(ae_path),
                "classifier_type": "AttackClassifier",
                "stage": "classifier",
            }
            save_clf_checkpoint(ckpt_dir / "classifier.pt", clf, meta)
        else:
            stale += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  "
                f"lr={optimizer.param_groups[0]['lr']:.1e}"
            )

        if stale >= patience:
            print(f"Early stop at epoch {epoch} (best val_acc={best_val_acc:.3f} @ epoch {best_epoch})")
            break

    save_json(
        results_dir / "clf_history.json",
        {
            "run_id": run_id,
            "history": history,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "class_names": class_names,
        },
    )
    update_run_info(
        results_dir,
        classifier_completed_at=datetime.now(timezone.utc).isoformat(),
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
        classifier_checkpoint=str(ckpt_dir / "classifier.pt"),
    )
    write_latest_run(run_id)
    print(f"Saved classifier to {ckpt_dir / 'classifier.pt'}  (best val_acc={best_val_acc:.3f})")
    print(f"Run id: {run_id}  (use --run-id {run_id} for evaluate / figures)")


if __name__ == "__main__":
    main()
