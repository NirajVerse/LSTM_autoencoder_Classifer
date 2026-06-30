"""Shared utilities: paths, config, seeds, run directories."""

from __future__ import annotations

import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


MILCOM_ROOT = Path(__file__).resolve().parent.parent
ML_ROOT = Path(__file__).resolve().parent
LATEST_RUN_FILE = ML_ROOT / "latest_run.txt"


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or (ML_ROOT / "config.yaml")
    with path.open() as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(relative: str) -> Path:
    return (MILCOM_ROOT / relative).resolve()


def get_train_datasets_dir(cfg: dict[str, Any], override: Path | str | None = None) -> Path:
    if override is not None:
        return Path(override).resolve()
    if not cfg.get("datasets_dir"):
        raise ValueError("datasets_dir must be set in config.yaml")
    return resolve_path(cfg["datasets_dir"])


def get_test_datasets_dir(cfg: dict[str, Any], override: Path | str | None = None) -> Path:
    if override is not None:
        return Path(override).resolve()
    if not cfg.get("test_datasets_dir"):
        raise ValueError(
            "test_datasets_dir must be set in config.yaml for evaluate.py / paper figures"
        )
    return resolve_path(cfg["test_datasets_dir"])


def validate_datasets_layout(
    datasets_dir: Path,
    class_names: list[str],
    *,
    purpose: str,
) -> None:
    if not datasets_dir.is_dir():
        raise FileNotFoundError(f"{purpose} directory not found: {datasets_dir}")

    missing: list[str] = []
    empty: list[str] = []
    for name in class_names:
        class_dir = datasets_dir / name
        if not class_dir.is_dir():
            missing.append(name)
            continue
        if not list(class_dir.glob("*.csv")):
            empty.append(name)

    if missing or empty:
        lines = [f"{purpose} layout invalid under {datasets_dir}:"]
        if missing:
            lines.append(f"  missing folders: {missing}")
        if empty:
            lines.append(f"  folders with no CSV: {empty}")
        lines.append("Expected: <dir>/clean/*.csv, <dir>/barrage/*.csv, <dir>/random/*.csv")
        raise FileNotFoundError("\n".join(lines))


def ensure_train_test_separate(cfg: dict[str, Any]) -> None:
    train_dir = get_train_datasets_dir(cfg)
    test_dir = get_test_datasets_dir(cfg)
    if train_dir.resolve() == test_dir.resolve():
        raise ValueError(
            "datasets_dir and test_datasets_dir must be different folders "
            f"(both point to {train_dir})"
        )


def make_run_id() -> str:
    """Timestamp id for a training run, e.g. 20250630_214530."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_run_checkpoint_dir(cfg: dict[str, Any], run_id: str) -> Path:
    return resolve_path(cfg["checkpoints_dir"]) / run_id


def get_run_results_dir(cfg: dict[str, Any], run_id: str) -> Path:
    return resolve_path(cfg["results_dir"]) / run_id


def ensure_run_dirs(cfg: dict[str, Any], run_id: str) -> tuple[Path, Path]:
    ckpt_dir = get_run_checkpoint_dir(cfg, run_id)
    results_dir = get_run_results_dir(cfg, run_id)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir, results_dir


def write_latest_run(run_id: str) -> None:
    LATEST_RUN_FILE.write_text(run_id + "\n")


def read_latest_run() -> str | None:
    if not LATEST_RUN_FILE.exists():
        return None
    run_id = LATEST_RUN_FILE.read_text().strip()
    return run_id or None


def resolve_run_id(explicit: str | None, *, cfg: dict[str, Any] | None = None) -> str:
    """Return explicit run id, else latest run, else raise."""
    if explicit:
        return explicit
    latest = read_latest_run()
    if latest:
        return latest
    if cfg is not None and _legacy_checkpoint_exists(cfg):
        return "_legacy"
    raise FileNotFoundError(
        "No run id found. Train a model first (train_ae.py) or pass --run-id."
    )


def _legacy_checkpoint_exists(cfg: dict[str, Any]) -> bool:
    ckpt_base = resolve_path(cfg["checkpoints_dir"])
    return (ckpt_base / "autoencoder.pt").exists()


def get_run_dirs(cfg: dict[str, Any], run_id: str) -> tuple[Path, Path]:
    """Resolve checkpoint/results dirs for a run, with legacy flat-layout fallback."""
    if run_id == "_legacy":
        return resolve_path(cfg["checkpoints_dir"]), resolve_path(cfg["results_dir"])
    return get_run_checkpoint_dir(cfg, run_id), get_run_results_dir(cfg, run_id)


def update_run_info(results_dir: Path, **fields: Any) -> None:
    path = results_dir / "run_info.json"
    info: dict[str, Any] = {}
    if path.exists():
        info = load_json(path)
    info.update(fields)
    save_json(path, info)


def save_config_snapshot(
    results_dir: Path,
    cfg: dict[str, Any],
    config_path: Path | None,
) -> None:
    snapshot = results_dir / "config_snapshot.yaml"
    with snapshot.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    if config_path and config_path.exists():
        shutil.copy2(config_path, results_dir / "config_source.yaml")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)
