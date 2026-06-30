#!/usr/bin/env python3
"""Generate candidate figures for the MILCOM 2-page paper.

Each figure is written to ml/figures/ as both .png (for previewing) and .pdf
(for direct LaTeX import). Run with:

    cd ml && .venv/bin/python make_paper_figures.py

Outputs:
    figures/cm.{png,pdf}                Confusion matrix (heatmap)
    figures/metrics_bar.{png,pdf}       Per-class precision/recall/F1 bars
    figures/bler_timeseries.{png,pdf}   dl_bler over time, all classes
    figures/snr_timeseries.{png,pdf}    dl_snr over time, all classes
    figures/kpi_grid.{png,pdf}          2x2 grid of 4 KPIs
    figures/combined_2panel.{png,pdf}   BLER timeseries + confusion matrix
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from utils import get_run_dirs, load_config, resolve_run_id


ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
ML_ROOT = Path(__file__).resolve().parent
OUT_DIR = ML_ROOT / "figures"

# Folder name -> (display label, line color)
CLASS_STYLE = {
    "clean":    ("clean",    "#2ca02c"),
    "barrage":  ("barrage",  "#d62728"),
    "random":   ("random",   "#1f77b4"),
    "constant": ("constant", "#ff7f0e"),
}


def load_csv_column(path: Path, col: str) -> list[float]:
    vals: list[float] = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            v = row.get(col)
            if v in (None, "", "n/a"):
                continue
            try:
                vals.append(float(v))
            except ValueError:
                continue
    return vals


def get_class_csv(class_name: str) -> Path | None:
    folder = DATASETS / class_name
    if not folder.is_dir():
        return None
    files = sorted(folder.glob("*.csv"))
    return files[0] if files else None


def save_fig(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"{name}.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  wrote {out.relative_to(ROOT)}")
    plt.close(fig)


def smooth(x: list[float] | np.ndarray, w: int = 5) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="same")


# ---------------------------------------------------------------------------
# Figure 1: Confusion matrix (standalone)
# ---------------------------------------------------------------------------
def fig_confusion_matrix(eval_data: dict) -> None:
    cm = np.array(eval_data["confusion_matrix"])
    classes = eval_data["class_names"]
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, np.maximum(row_sums, 1))

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel("Predicted class", fontsize=12)
    ax.set_ylabel("True class", fontsize=12)

    acc = np.trace(cm) / max(cm.sum(), 1) * 100
    ax.set_title(
        f"Classifier confusion matrix\n"
        f"Overall accuracy: {acc:.1f}%  (n={cm.sum()} windows)",
        fontsize=11, fontweight="bold",
    )

    for i in range(len(classes)):
        for j in range(len(classes)):
            txt = f"{cm[i, j]}\n({cm_norm[i, j] * 100:.0f}%)"
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Per-class recall", fontsize=10)

    fig.tight_layout()
    save_fig(fig, "cm")


# ---------------------------------------------------------------------------
# Figure 2: Per-class precision/recall/F1 bar chart (mirrors reference paper)
# ---------------------------------------------------------------------------
def fig_metrics_bar(eval_data: dict) -> None:
    classes = eval_data["class_names"]
    cm = np.array(eval_data["confusion_matrix"])
    # Reconstruct per-class true/pred from confusion matrix
    y_true: list[int] = []
    y_pred: list[int] = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            y_true.extend([i] * int(cm[i, j]))
            y_pred.extend([j] * int(cm[i, j]))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(classes))), zero_division=0
    )

    x = np.arange(len(classes))
    width = 0.27

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    b1 = ax.bar(x - width, p, width, label="Precision",  color="#4c72b0")
    b2 = ax.bar(x,         r, width, label="Recall",     color="#dd8452")
    b3 = ax.bar(x + width, f1, width, label="F1-score",  color="#55a868")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.set_title("Per-class classification performance",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.3)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    save_fig(fig, "metrics_bar")


# ---------------------------------------------------------------------------
# Figure 3: dl_bler time series, all classes overlaid
# ---------------------------------------------------------------------------
def fig_bler_timeseries() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for class_name, (label, color) in CLASS_STYLE.items():
        csv_path = get_class_csv(class_name)
        if csv_path is None:
            continue
        bler = load_csv_column(csv_path, "dl_bler")
        if not bler:
            continue
        sm = smooth(bler, w=5)
        mean = float(np.mean(bler))
        ax.plot(range(len(sm)), sm, color=color, linewidth=1.5, alpha=0.9,
                label=f"{label} (mean={mean:.1f}%)")

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("DL BLER (%)", fontsize=12)
    ax.set_title("Downlink Block Error Rate per class",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(-3, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)

    fig.tight_layout()
    save_fig(fig, "bler_timeseries")


# ---------------------------------------------------------------------------
# Figure 4: dl_snr time series, all classes overlaid
# ---------------------------------------------------------------------------
def fig_snr_timeseries() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for class_name, (label, color) in CLASS_STYLE.items():
        csv_path = get_class_csv(class_name)
        if csv_path is None:
            continue
        snr = load_csv_column(csv_path, "dl_snr")
        if not snr:
            continue
        sm = smooth(snr, w=5)
        mean = float(np.mean(snr))
        ax.plot(range(len(sm)), sm, color=color, linewidth=1.5, alpha=0.9,
                label=f"{label} (mean={mean:.1f} dB)")

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("DL SNR (dB)", fontsize=12)
    ax.set_title("Downlink SNR per class",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)

    fig.tight_layout()
    save_fig(fig, "snr_timeseries")


# ---------------------------------------------------------------------------
# Figure 5: 2x2 KPI grid (dl_snr, dl_bler, ul_bler, dl_snr_roll_std)
# ---------------------------------------------------------------------------
def fig_kpi_grid() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5), sharex=False)
    panels = [
        ("dl_snr",  "DL SNR (dB)",    axes[0, 0], None),
        ("dl_bler", "DL BLER (%)",    axes[0, 1], (-3, 105)),
        ("ul_bler", "UL BLER (%)",    axes[1, 0], (-3, 105)),
        ("dl_snr",  "DL SNR rolling std (dB)", axes[1, 1], None),
    ]

    for col, ylabel, ax, ylim in panels:
        for class_name, (label, color) in CLASS_STYLE.items():
            csv_path = get_class_csv(class_name)
            if csv_path is None:
                continue
            vals = load_csv_column(csv_path, col)
            if not vals:
                continue
            if "rolling" in ylabel:
                arr = np.asarray(vals, dtype=np.float32)
                if len(arr) < 5:
                    continue
                roll_std = np.array(
                    [arr[max(0, i - 4): i + 1].std() for i in range(len(arr))]
                )
                series = roll_std
            else:
                series = smooth(vals, w=5)
            mean = float(np.mean(vals))
            ax.plot(range(len(series)), series, color=color, linewidth=1.3,
                    alpha=0.85,
                    label=f"{label} (μ={mean:.1f})")

        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11, fontweight="bold")
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, framealpha=0.95)

    fig.suptitle("Per-class KPI signatures used by the classifier",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "kpi_grid")


# ---------------------------------------------------------------------------
# Figure 6: Combined two-panel — BLER time series + confusion matrix
# ---------------------------------------------------------------------------
def fig_combined_two_panel(eval_data: dict) -> None:
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(11.0, 4.0),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )

    for class_name, (label, color) in CLASS_STYLE.items():
        csv_path = get_class_csv(class_name)
        if csv_path is None:
            continue
        bler = load_csv_column(csv_path, "dl_bler")
        if not bler:
            continue
        sm = smooth(bler, w=5)
        mean = float(np.mean(bler))
        ax_l.plot(range(len(sm)), sm, color=color, linewidth=1.5, alpha=0.9,
                  label=f"{label} (mean={mean:.1f}%)")

    ax_l.set_xlabel("Time (s)", fontsize=11)
    ax_l.set_ylabel("DL BLER (%)", fontsize=11)
    ax_l.set_title("(a) Downlink BLER signature per class",
                   fontsize=11, fontweight="bold")
    ax_l.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax_l.grid(True, alpha=0.3)
    ax_l.set_ylim(-3, 105)

    cm = np.array(eval_data["confusion_matrix"])
    classes = eval_data["class_names"]
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    im = ax_r.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    ax_r.set_xticks(range(len(classes)))
    ax_r.set_yticks(range(len(classes)))
    ax_r.set_xticklabels(classes, fontsize=10)
    ax_r.set_yticklabels(classes, fontsize=10)
    ax_r.set_xlabel("Predicted class", fontsize=11)
    ax_r.set_ylabel("True class", fontsize=11)
    ax_r.set_title("(b) Classifier confusion matrix\n(per-class recall)",
                   fontsize=11, fontweight="bold")

    for i in range(len(classes)):
        for j in range(len(classes)):
            txt = f"{cm[i, j]}\n({cm_norm[i, j] * 100:.0f}%)"
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax_r.text(j, i, txt, ha="center", va="center",
                      color=color, fontsize=10, fontweight="bold")

    acc = np.trace(cm) / max(cm.sum(), 1) * 100
    ax_r.text(0.5, -0.18, f"Overall accuracy: {acc:.1f}%  (n={cm.sum()} windows)",
              ha="center", va="top", transform=ax_r.transAxes,
              fontsize=10, style="italic")

    cbar = plt.colorbar(im, ax=ax_r, fraction=0.046, pad=0.04)
    cbar.set_label("Recall", fontsize=9)

    fig.tight_layout()
    save_fig(fig, "combined_2panel")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate paper figures for a training run")
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
    run_id = resolve_run_id(args.run_id, cfg=cfg)
    _, results_dir = get_run_dirs(cfg, run_id)
    results_path = results_dir / "evaluation.json"

    if not results_path.exists():
        raise SystemExit(
            f"Missing {results_path}. Run evaluate.py first for run id {run_id}."
        )
    with open(results_path) as f:
        eval_data = json.load(f)

    global OUT_DIR
    OUT_DIR = ML_ROOT / "figures" / run_id

    print(f"Run id: {run_id}")
    print("Generating paper figures...")
    fig_confusion_matrix(eval_data)
    fig_metrics_bar(eval_data)
    fig_bler_timeseries()
    fig_snr_timeseries()
    fig_kpi_grid()
    fig_combined_two_panel(eval_data)
    print(f"\nAll figures saved to {OUT_DIR.relative_to(ROOT)}/")
    print(f"Run id: {run_id}")


if __name__ == "__main__":
    main()
