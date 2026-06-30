"""Training helpers: scaling, splits, checkpoint I/O, inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from dataset import WindowDataset
from models import AttackClassifier, LSTMAutoencoder
from utils import get_device


def fit_scaler(x: np.ndarray) -> StandardScaler:
    """Fit per-feature scaler on all timesteps in training windows."""
    n_samples, seq_len, n_features = x.shape
    flat = x.reshape(n_samples * seq_len, n_features)
    scaler = StandardScaler()
    scaler.fit(flat)
    return scaler


def transform_windows(x: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    n_samples, seq_len, n_features = x.shape
    flat = x.reshape(n_samples * seq_len, n_features)
    scaled = scaler.transform(flat)
    return scaled.reshape(n_samples, seq_len, n_features).astype(np.float32)


def stratified_split(
    x: np.ndarray,
    y: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=val_fraction,
        random_state=seed,
        stratify=y,
    )
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def make_loader(
    x: np.ndarray,
    y: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = WindowDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def encode_windows(model: LSTMAutoencoder, x: np.ndarray, batch_size: int) -> np.ndarray:
    device = get_device()
    model.eval()
    loader = make_loader(x, None, batch_size, shuffle=False)
    latents: list[np.ndarray] = []
    for batch_x in loader:
        batch_x = batch_x.to(device)
        z = model.encode(batch_x)
        latents.append(z.cpu().numpy())
    return np.concatenate(latents, axis=0)


@torch.no_grad()
def predict_windows(
    ae_model: LSTMAutoencoder,
    clf_model: AttackClassifier,
    x: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Return class index predictions for all windows."""
    device = get_device()
    ae_model.eval()
    clf_model.eval()
    loader = make_loader(x, None, batch_size, shuffle=False)
    preds: list[np.ndarray] = []
    for batch_x in loader:
        batch_x = batch_x.to(device)
        z, seq_out = ae_model.encode_sequence(batch_x)
        logits = clf_model(batch_x, z, seq_out)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


@torch.no_grad()
def predict_proba_windows(
    ae_model: LSTMAutoencoder,
    clf_model: AttackClassifier,
    x: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    device = get_device()
    ae_model.eval()
    clf_model.eval()
    loader = make_loader(x, None, batch_size, shuffle=False)
    probs: list[np.ndarray] = []
    for batch_x in loader:
        batch_x = batch_x.to(device)
        z, seq_out = ae_model.encode_sequence(batch_x)
        logits = clf_model(batch_x, z, seq_out)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def save_ae_checkpoint(
    path: Path,
    model: LSTMAutoencoder,
    scaler: StandardScaler,
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "meta": meta,
        },
        path,
    )


def load_ae_checkpoint(path: Path, cfg: dict) -> tuple[LSTMAutoencoder, StandardScaler, dict[str, Any]]:
    device = get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    model = LSTMAutoencoder(
        n_features=meta["n_features"],
        hidden_size=cfg["ae_hidden_size"],
        latent_size=cfg["ae_latent_size"],
        num_layers=cfg["ae_num_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    scaler.n_features_in_ = len(scaler.mean_)
    return model, scaler, meta


def save_clf_checkpoint(
    path: Path,
    model: AttackClassifier,
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "meta": meta}, path)


def load_clf_checkpoint(path: Path, cfg: dict) -> tuple[AttackClassifier, dict[str, Any]]:
    device = get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    model = AttackClassifier(
        n_features=meta["n_features"],
        latent_size=meta["latent_size"],
        encoder_hidden=cfg["ae_hidden_size"],
        n_classes=meta["n_classes"],
        hidden_sizes=cfg.get("clf_hidden_sizes", [128, 64]),
        dropout=cfg["clf_dropout"],
        bilstm_hidden=cfg.get("clf_bilstm_hidden", 32),
        attn_heads=cfg.get("clf_attn_heads", 4),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, meta
