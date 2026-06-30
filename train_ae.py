#!/usr/bin/env python3
"""Stage 1: Train LSTM autoencoder on clean windows only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from load_data import load_dataset_from_folders, samples_to_arrays
from models import LSTMAutoencoder
from training import fit_scaler, make_loader, save_ae_checkpoint, transform_windows
from utils import (
    ensure_run_dirs,
    get_device,
    load_config,
    make_run_id,
    resolve_path,
    resolve_run_id,
    save_config_snapshot,
    save_json,
    set_seed,
    update_run_info,
    write_latest_run,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LSTM autoencoder on clean KPI windows")
    p.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run folder name (default: new timestamp YYYYMMDD_HHMMSS)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()

    datasets_dir = resolve_path(cfg["datasets_dir"])
    run_id = args.run_id or make_run_id()
    ckpt_dir, results_dir = ensure_run_dirs(cfg, run_id)
    save_config_snapshot(results_dir, cfg, args.config)
    update_run_info(
        results_dir,
        run_id=run_id,
        started_at=datetime.now(timezone.utc).isoformat(),
        stage="autoencoder",
        classes=cfg["classes"],
    )
    print(f"Run id: {run_id}")
    print(f"  checkpoints -> {ckpt_dir}")
    print(f"  results     -> {results_dir}")

    samples = load_dataset_from_folders(datasets_dir, ["clean"], cfg)
    x, _, _ = samples_to_arrays(samples)
    print(f"Loaded {len(x)} clean windows from {datasets_dir}/clean")

    from sklearn.model_selection import train_test_split

    if len(x) >= 2:
        x_train, x_val = train_test_split(
            x,
            test_size=cfg["val_fraction"],
            random_state=cfg["seed"],
        )
    else:
        x_train, x_val = x, x

    scaler = fit_scaler(x_train)
    x_train = transform_windows(x_train, scaler)
    x_val = transform_windows(x_val, scaler)

    n_features = x_train.shape[2]
    model = LSTMAutoencoder(
        n_features=n_features,
        hidden_size=cfg["ae_hidden_size"],
        latent_size=cfg["ae_latent_size"],
        num_layers=cfg["ae_num_layers"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate_ae"])
    criterion = nn.MSELoss()

    train_loader = make_loader(x_train, None, cfg["batch_size"], shuffle=True)
    val_loader = make_loader(x_val, None, cfg["batch_size"], shuffle=False)

    best_val = float("inf")
    history: list[dict] = []

    for epoch in range(1, cfg["ae_epochs"] + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch_x in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch_x in val_loader:
                batch_x = batch_x.to(device)
                recon, _ = model(batch_x)
                val_loss += criterion(recon, batch_x).item()
                n_val += 1
        val_loss /= max(n_val, 1)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            meta = {
                "n_features": n_features,
                "feature_columns": cfg["raw_features"] + cfg["engineered_features"],
                "window_size": cfg["window_size"],
                "classes": cfg["classes"],
                "stage": "autoencoder",
            }
            save_ae_checkpoint(ckpt_dir / "autoencoder.pt", model, scaler, meta)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}  train_mse={train_loss:.6f}  val_mse={val_loss:.6f}")

    save_json(results_dir / "ae_history.json", {"run_id": run_id, "history": history, "best_val_mse": best_val})
    update_run_info(
        results_dir,
        ae_completed_at=datetime.now(timezone.utc).isoformat(),
        best_val_mse=best_val,
        autoencoder_checkpoint=str(ckpt_dir / "autoencoder.pt"),
    )
    write_latest_run(run_id)
    print(f"Saved autoencoder to {ckpt_dir / 'autoencoder.pt'}")
    print(f"Run id: {run_id}  (use --run-id {run_id} for train_clf / evaluate)")


if __name__ == "__main__":
    main()
