"""Progress 4 validation-only evaluation & visual generator (BAB IV evidence).

Strictly evaluates the held-out VALIDATION set (713 images) using checkpoints saved
by src.train. Test set access is explicitly blocked until Progress 5.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import DEFAULT_SPLIT_PATH, RealWasteDataset, load_uint8_cache
from .models import CLASS_NAMES, CNNBaseline, MobileNetV3Small, NUM_CLASSES
from .models.config import ROOT
from .train import DEFAULT_OUTPUT_DIR, compute_classification_metrics


def _load_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model_name = ckpt["model_name"]
    if model_name == "cnn_baseline":
        model = CNNBaseline(num_classes=NUM_CLASSES, dropout=float(ckpt.get("dropout", 0.3)))
    elif model_name == "mobilenet_v3_small":
        model = MobileNetV3Small(num_classes=NUM_CLASSES, pretrained=False, freeze_backbone=True)
    else:
        raise ValueError(f"Unknown model_name in checkpoint: {model_name}")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt


def plot_training_curves(
    exp_histories: dict[str, dict[str, Any]],
    figures_dir: Path,
) -> tuple[Path, Path]:
    """Generate Gambar 4.1 (Accuracy & Macro F1) and Gambar 4.2 (Loss) from actual epoch logs."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Gambar 4.1: Validation Macro F1 & Train/Val Accuracy
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=300)
    for exp_id, payload in exp_histories.items():
        hist = payload["history"]
        epochs = [h["epoch"] for h in hist]
        val_acc = [h["val_accuracy"] * 100 for h in hist]
        val_f1 = [h["val_macro_f1"] * 100 for h in hist]
        label = payload["title"]
        axes[0].plot(epochs, val_acc, marker="o", markersize=3, linewidth=1.8, label=label)
        axes[1].plot(epochs, val_f1, marker="s", markersize=3, linewidth=1.8, label=label)

    axes[0].set_title("Validation Accuracy per Epoch (%)", fontweight="bold", fontsize=10.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Validation Macro F1-Score per Epoch (%)", fontweight="bold", fontsize=10.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1-Score (%)")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=8)

    plt.suptitle("Gambar 4.1 Kurva Akurasi dan Macro F1 Validasi Antar-Eksperimen (Progres 4)", fontweight="bold", y=1.02)
    plt.tight_layout()
    acc_fig = figures_dir / "p4_training_accuracy_f1_curves.png"
    plt.savefig(acc_fig, bbox_inches="tight")
    plt.close()

    # Gambar 4.2: Train Loss vs Validation Loss
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=300)
    for exp_id, payload in exp_histories.items():
        hist = payload["history"]
        epochs = [h["epoch"] for h in hist]
        tr_loss = [h["train_loss"] for h in hist]
        va_loss = [h["val_loss"] for h in hist]
        label = payload["title"]
        axes[0].plot(epochs, tr_loss, marker="o", markersize=3, linewidth=1.8, label=label)
        axes[1].plot(epochs, va_loss, marker="s", markersize=3, linewidth=1.8, label=label)

    axes[0].set_title("Training Loss per Epoch", fontweight="bold", fontsize=10.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Validation Loss per Epoch", fontweight="bold", fontsize=10.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-Entropy Loss")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=8)

    plt.suptitle("Gambar 4.2 Kurva Training dan Validation Loss Antar-Eksperimen (Progres 4)", fontweight="bold", y=1.02)
    plt.tight_layout()
    loss_fig = figures_dir / "p4_training_loss_curves.png"
    plt.savefig(loss_fig, bbox_inches="tight")
    plt.close()

    return acc_fig, loss_fig


def plot_validation_diagnostics(
    metrics: dict[str, Any],
    best_title: str,
    figures_dir: Path,
) -> tuple[Path, Path]:
    """Generate Gambar 4.3 (Validation Confusion Matrix) and Gambar 4.4 (Per-class Precision/Recall/F1)."""
    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    short_names = [c.replace("Miscellaneous Trash", "Misc. Trash") for c in CLASS_NAMES]

    # Gambar 4.3: Confusion Matrix
    fig, ax = plt.subplots(figsize=(8.8, 7.2), dpi=300)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(short_names, rotation=35, ha="right", fontsize=8.5)
    ax.set_yticklabels(short_names, fontsize=8.5)
    ax.set_xlabel("Prediksi Model", fontweight="bold")
    ax.set_ylabel("Label Sebenarnya (Ground Truth)", fontweight="bold")
    ax.set_title(
        f"Confusion Matrix Validation Set (713 Sampel) — {best_title}\n"
        f"Acc: {metrics['accuracy']*100:.2f}% | Macro F1: {metrics['macro_f1']*100:.2f}%",
        fontweight="bold",
        fontsize=10.5,
        pad=12,
    )

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=8.5,
                color="white" if cm[i, j] > thresh else "#1E293B",
            )
    plt.tight_layout()
    cm_fig = figures_dir / "p4_val_confusion_matrix.png"
    plt.savefig(cm_fig, bbox_inches="tight")
    plt.close()

    # Gambar 4.4: Precision, Recall, F1 per class
    per_cls = metrics["per_class"]
    prec = [per_cls[c]["precision"] * 100 for c in CLASS_NAMES]
    rec = [per_cls[c]["recall"] * 100 for c in CLASS_NAMES]
    f1 = [per_cls[c]["f1"] * 100 for c in CLASS_NAMES]

    x = np.arange(len(CLASS_NAMES))
    w = 0.25
    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=300)
    ax.bar(x - w, prec, width=w, label="Precision (%)", color="#3B82F6")
    ax.bar(x, rec, width=w, label="Recall (%)", color="#10B981")
    ax.bar(x + w, f1, width=w, label="F1-Score (%)", color="#1E3A8A")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=25, ha="right", fontsize=8.5)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Skor (%)")
    ax.set_title(
        f"Precision, Recall, dan F1-Score per Kelas pada Validation Set — {best_title}",
        fontweight="bold",
        fontsize=10.5,
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    cls_fig = figures_dir / "p4_val_per_class_metrics.png"
    plt.savefig(cls_fig, bbox_inches="tight")
    plt.close()

    return cm_fig, cls_fig


def evaluate_validation(
    *,
    results_dir: Path = DEFAULT_OUTPUT_DIR,
    figures_dir: Path = ROOT / "figures",
    data_dir: str | Path | None = None,
    split_path: Path = DEFAULT_SPLIT_PATH,
    cache_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate all trained Progress 4 experiments strictly on validation split."""
    results_dir = Path(results_dir)
    exp_dirs = sorted(p for p in results_dir.iterdir() if p.is_dir() and (p / "summary.json").is_file())
    if not exp_dirs:
        raise FileNotFoundError(f"No completed experiments found under {results_dir}")

    exp_histories: dict[str, dict[str, Any]] = {}
    best_exp_dir: Path | None = None
    best_f1 = -1.0
    best_summary: dict[str, Any] = {}

    for ed in exp_dirs:
        summary = json.loads((ed / "summary.json").read_text(encoding="utf-8"))
        history = json.loads((ed / "history.json").read_text(encoding="utf-8"))
        exp_histories[ed.name] = {"title": summary["spec"]["title"], "history": history}
        if float(summary["best_val_macro_f1"]) > best_f1:
            best_f1 = float(summary["best_val_macro_f1"])
            best_exp_dir = ed
            best_summary = summary

    assert best_exp_dir is not None
    acc_fig, loss_fig = plot_training_curves(exp_histories, Path(figures_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = _load_model_from_checkpoint(best_exp_dir / "best_model.pt", device)

    preloaded = load_uint8_cache(cache_path) if (cache_path is not None and Path(cache_path).is_file()) else None
    val_ds = RealWasteDataset(
        "validation",
        ckpt["model_name"],
        augment=False,
        data_dir=data_dir,
        split_path=split_path,
        cache_path=cache_path,
        preloaded_cache=preloaded,
        allow_test=False,  # Explicit anti-leakage guard for Progress 4
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    all_targets: list[int] = []
    all_preds: list[int] = []
    all_confs: list[float] = []

    with torch.inference_mode():
        for images, labels in val_loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)
            all_targets.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())
            all_confs.extend(confs.cpu().tolist())

    metrics = compute_classification_metrics(np.asarray(all_targets), np.asarray(all_preds))
    cm_fig, cls_fig = plot_validation_diagnostics(metrics, best_summary["spec"]["title"], Path(figures_dir))

    # Extract 5 correct and 5 misclassified samples for BAB IV section 4.3 table
    sample_rows = []
    for (rel_path, _), t_idx, p_idx, conf in zip(val_ds.records, all_targets, all_preds, all_confs):
        sample_rows.append({
            "file": rel_path,
            "true_label": CLASS_NAMES[t_idx],
            "pred_label": CLASS_NAMES[p_idx],
            "confidence": round(float(conf), 4),
            "correct": bool(t_idx == p_idx),
        })

    correct_samples = sorted([r for r in sample_rows if r["correct"]], key=lambda x: -x["confidence"])[:5]
    wrong_samples = sorted([r for r in sample_rows if not r["correct"]], key=lambda x: -x["confidence"])[:5]

    report = {
        "evaluation_split": "validation_only",
        "test_set_untouched": True,
        "validation_count": len(val_ds),
        "best_experiment": {
            "exp_id": best_summary["spec"]["exp_id"],
            "title": best_summary["spec"]["title"],
            "model_name": best_summary["spec"]["model_name"],
            "best_epoch": best_summary["best_epoch"],
        },
        "metrics": metrics,
        "representative_samples": {
            "correct_examples": correct_samples,
            "misclassified_examples": wrong_samples,
        },
        "figures": {
            "accuracy_f1_curves": str(acc_fig),
            "loss_curves": str(loss_fig),
            "val_confusion_matrix": str(cm_fig),
            "val_per_class_metrics": str(cls_fig),
        },
    }
    out_json = results_dir / "val_best_model_evaluation.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Validation evaluation saved: {out_json}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--cache-path", type=Path, default=ROOT / "tmp/realwaste_128_uint8.npz")
    args = parser.parse_args()

    evaluate_validation(
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        data_dir=args.data_dir,
        split_path=args.split_path,
        cache_path=args.cache_path,
    )


if __name__ == "__main__":
    main()
