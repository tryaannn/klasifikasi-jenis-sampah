"""Progress 4 validation-only evaluation, calibration audit, & error analysis.

Strictly evaluates the held-out VALIDATION set (713 images) across all experiments
saved under results_dir. Test set access is explicitly blocked until Progress 5.
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
from .train import (
    DEFAULT_OUTPUT_DIR,
    _get_git_commit,
    compute_classification_metrics,
    write_comparison_table,
)


def _load_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model_name = ckpt["model_name"]
    if model_name == "cnn_baseline":
        model = CNNBaseline(num_classes=NUM_CLASSES, dropout=float(ckpt.get("dropout", 0.3)))
    elif model_name == "mobilenet_v3_small":
        model = MobileNetV3Small(num_classes=NUM_CLASSES, pretrained=False, freeze_backbone=True)
        unfreeze_n = int(ckpt.get("unfreeze_last_n", 0))
        if unfreeze_n > 0:
            model.set_trainable_feature_blocks(unfreeze_n)
    else:
        raise ValueError(f"Unknown model_name in checkpoint: {model_name}")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt


def compute_calibration_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute confidence distributions (correct vs wrong) and Expected Calibration Error (ECE)."""
    correct_mask = (y_true == y_pred)
    correct_confs = confidences[correct_mask]
    wrong_confs = confidences[~correct_mask]

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins_info = []
    n = len(confidences)
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        in_bin = (confidences >= lo) & (confidences <= hi if b == n_bins - 1 else confidences < hi)
        cnt = int(in_bin.sum())
        if cnt > 0:
            acc_bin = float(correct_mask[in_bin].mean())
            conf_bin = float(confidences[in_bin].mean())
            ece += (cnt / max(1, n)) * abs(acc_bin - conf_bin)
            bins_info.append({
                "bin_range": [round(float(lo), 2), round(float(hi), 2)],
                "count": cnt,
                "accuracy": round(acc_bin, 4),
                "avg_confidence": round(conf_bin, 4),
            })

    overconf_wrong_90 = int((wrong_confs > 0.90).sum()) if len(wrong_confs) > 0 else 0
    return {
        "ece_10_bins": round(float(ece), 6),
        "total_samples": int(n),
        "correct_count": int(correct_mask.sum()),
        "wrong_count": int((~correct_mask).sum()),
        "correct_confidence": {
            "mean": round(float(correct_confs.mean()), 4) if len(correct_confs) else 0.0,
            "median": round(float(np.median(correct_confs)), 4) if len(correct_confs) else 0.0,
        },
        "wrong_confidence": {
            "mean": round(float(wrong_confs.mean()), 4) if len(wrong_confs) else 0.0,
            "median": round(float(np.median(wrong_confs)), 4) if len(wrong_confs) else 0.0,
        },
        "overconfident_wrong_gt_090": overconf_wrong_90,
        "overconfident_wrong_ratio": round(overconf_wrong_90 / max(1, len(wrong_confs)), 4),
        "reliability_bins": bins_info,
    }


def plot_training_curves(
    exp_histories: dict[str, dict[str, Any]],
    figures_dir: Path,
) -> tuple[Path, Path]:
    """Generate Gambar 4.1 (Accuracy & Macro F1) and Gambar 4.2 (Loss) across experiments."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    palette = {
        "exp01_cnn_control": ("#94A3B8", "--"),
        "exp02_cnn_aug": ("#F59E0B", "--"),
        "exp03_cnn_weighted": ("#10B981", "--"),
        "exp04_mobilenet_tl": ("#DC2626", "-"),
        "exp05_mobilenet_aug_reg": ("#60A5FA", "-."),
        "exp06_mobilenet_finetune": ("#A78BFA", "-."),
        "exp07_cnn_mitigation": ("#14B8A6", ":"),
        "exp08_mobilenet_224_head": ("#2563EB", "-"),
        "exp09_mobilenet_224_reg": ("#059669", "-"),
        "exp10_mobilenet_224_2stage_ft": ("#7C3AED", "-"),
        "exp11_mobilenet_224_2stage_cw": ("#DB2777", "-"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5), dpi=300)
    for exp_id, payload in exp_histories.items():
        hist = payload["history"]
        epochs = [h["epoch"] for h in hist]
        val_acc = [h["val_accuracy"] * 100 for h in hist]
        val_f1 = [h["val_macro_f1"] * 100 for h in hist]
        color, ls = palette.get(exp_id, ("#334155", "-"))
        lw = 2.1 if ls == "-" else 1.3
        label = payload["title"]
        axes[0].plot(epochs, val_acc, color=color, linestyle=ls, marker="o", markersize=2.2, linewidth=lw, label=label)
        axes[1].plot(epochs, val_f1, color=color, linestyle=ls, marker="s", markersize=2.2, linewidth=lw, label=label)

    axes[0].set_title("Validation Accuracy per Epoch (%)", fontweight="bold", fontsize=10.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=6.8, loc="lower right", ncol=1)

    axes[1].set_title("Validation Macro F1-Score per Epoch (%)", fontweight="bold", fontsize=10.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1-Score (%)")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=6.8, loc="lower right", ncol=1)

    plt.suptitle("Gambar 4.1 Kurva Akurasi dan Macro F1 Validasi (Baseline vs Ablasi 224x224)", fontweight="bold", y=1.02)
    plt.tight_layout()
    acc_fig = figures_dir / "p4_training_accuracy_f1_curves.png"
    plt.savefig(acc_fig, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5), dpi=300)
    for exp_id, payload in exp_histories.items():
        hist = payload["history"]
        epochs = [h["epoch"] for h in hist]
        tr_loss = [h["train_loss"] for h in hist]
        va_loss = [h["val_loss"] for h in hist]
        color, ls = palette.get(exp_id, ("#334155", "-"))
        lw = 2.1 if ls == "-" else 1.3
        label = payload["title"]
        axes[0].plot(epochs, tr_loss, color=color, linestyle=ls, marker="o", markersize=2.2, linewidth=lw, label=label)
        axes[1].plot(epochs, va_loss, color=color, linestyle=ls, marker="s", markersize=2.2, linewidth=lw, label=label)

    axes[0].set_title("Training Loss per Epoch", fontweight="bold", fontsize=10.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=6.8, loc="upper right")

    axes[1].set_title("Validation Loss per Epoch", fontweight="bold", fontsize=10.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-Entropy Loss")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=6.8, loc="upper right")

    plt.suptitle("Gambar 4.2 Kurva Training dan Validation Loss (Baseline vs Ablasi 224x224)", fontweight="bold", y=1.02)
    plt.tight_layout()
    loss_fig = figures_dir / "p4_training_loss_curves.png"
    plt.savefig(loss_fig, bbox_inches="tight")
    plt.close()

    return acc_fig, loss_fig


def plot_train_val_gap_comparison(
    comparison_rows: list[dict[str, Any]],
    figures_dir: Path,
) -> Path:
    """Generate comparison of Train Acc, Val Acc, Macro F1, and Overfitting Gap."""
    labels = [
        f"{r['exp_id'].split('_')[0].upper()}\n{r.get('image_size', 128)}px"
        for r in comparison_rows
    ]
    tr_acc = [r["best_train_accuracy"] * 100 for r in comparison_rows]
    va_acc = [r["best_val_accuracy"] * 100 for r in comparison_rows]
    va_f1 = [r["best_val_macro_f1"] * 100 for r in comparison_rows]
    gaps = [r["train_val_acc_gap"] * 100 for r in comparison_rows]

    x = np.arange(len(labels))
    w = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.0), dpi=300)

    axes[0].bar(x - w, tr_acc, width=w, label="Train Acc (%)", color="#94A3B8")
    axes[0].bar(x, va_acc, width=w, label="Val Acc (%)", color="#2563EB")
    axes[0].bar(x + w, va_f1, width=w, label="Val Macro F1 (%)", color="#1E3A8A")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=7.5)
    axes[0].set_ylim(0, 110)
    axes[0].set_ylabel("Skor (%)")
    axes[0].set_title("Perbandingan Train vs Validation & Macro F1", fontweight="bold", fontsize=10.5)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=8)

    bar_colors = ["#DC2626" if g > 15 else ("#F59E0B" if g > 10 else "#10B981") for g in gaps]
    bars = axes[1].bar(x, gaps, width=0.52, color=bar_colors)
    axes[1].axhline(10.0, color="#0F172A", linestyle="--", linewidth=1.1, label="Target Gap <= 10%")
    for b, g in zip(bars, gaps):
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{g:.1f}%", ha="center", va="bottom", fontsize=7.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=7.5)
    axes[1].set_ylim(0, max(35.0, max(gaps) + 5.0))
    axes[1].set_ylabel("Gap Train - Val Accuracy (%)")
    axes[1].set_title("Penekanan Overfitting (Train–Validation Gap)", fontweight="bold", fontsize=10.5)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=8)

    plt.suptitle("Evaluasi Dampak Ablasi 224x224 & 2-Stage Fine-Tuning terhadap Overfitting", fontweight="bold", y=1.02)
    plt.tight_layout()
    gap_fig = figures_dir / "p4_train_val_gap_comparison.png"
    plt.savefig(gap_fig, bbox_inches="tight")
    plt.close()
    return gap_fig


def plot_validation_diagnostics(
    metrics: dict[str, Any],
    best_title: str,
    figures_dir: Path,
) -> tuple[Path, Path]:
    """Generate Gambar 4.3 (Confusion Matrix) and Gambar 4.4 (Per-class Precision/Recall/F1)."""
    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    short_names = [c.replace("Miscellaneous Trash", "Misc. Trash") for c in CLASS_NAMES]

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


def plot_confidence_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    calib_stats: dict[str, Any],
    best_title: str,
    figures_dir: Path,
) -> Path:
    """Plot confidence histogram (Correct vs Wrong) and 10-bin reliability diagram."""
    correct_mask = (y_true == y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=300)

    axes[0].hist(
        confidences[correct_mask],
        bins=np.linspace(0, 1, 21),
        alpha=0.7,
        label=f"Benar (n={calib_stats['correct_count']}, mean={calib_stats['correct_confidence']['mean']:.3f})",
        color="#10B981",
    )
    if (~correct_mask).any():
        axes[0].hist(
            confidences[~correct_mask],
            bins=np.linspace(0, 1, 21),
            alpha=0.75,
            label=f"Salah (n={calib_stats['wrong_count']}, mean={calib_stats['wrong_confidence']['mean']:.3f})",
            color="#EF4444",
        )
    axes[0].axvline(0.90, color="#1E293B", linestyle="--", linewidth=1.0, label="Threshold Overconfident (0.90)")
    axes[0].set_xlabel("Confidence (Max Softmax Probability)")
    axes[0].set_ylabel("Jumlah Sampel")
    axes[0].set_title("Distribusi Confidence Prediksi Benar vs Salah", fontweight="bold", fontsize=10.5)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=8)

    bins = calib_stats["reliability_bins"]
    bx = [b["avg_confidence"] for b in bins]
    by = [b["accuracy"] for b in bins]
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Kalibrasi Ideal")
    axes[1].plot(bx, by, marker="o", linewidth=2.0, color="#2563EB", label=f"Model (ECE={calib_stats['ece_10_bins']:.4f})")
    axes[1].set_xlabel("Rata-rata Confidence Bin")
    axes[1].set_ylabel("Akurasi Aktual Bin")
    axes[1].set_title("Diagram Reliabilitas Kalibrasi (10-Bin)", fontweight="bold", fontsize=10.5)
    axes[1].set_xlim(0, 1.02)
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=8.5)

    plt.suptitle(f"Audit Kalibrasi Confidence Validation Set — {best_title}", fontweight="bold", y=1.02)
    plt.tight_layout()
    cal_fig = figures_dir / "p4_confidence_calibration.png"
    plt.savefig(cal_fig, bbox_inches="tight")
    plt.close()
    return cal_fig


def evaluate_validation(
    *,
    results_dir: Path = DEFAULT_OUTPUT_DIR,
    figures_dir: Path = ROOT / "figures",
    data_dir: str | Path | None = None,
    split_path: Path = DEFAULT_SPLIT_PATH,
    cache_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate best model strictly on validation split with calibration and error audit."""
    results_dir = Path(results_dir)
    comparison_rows = write_comparison_table(results_dir)
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
    gap_fig = plot_train_val_gap_comparison(comparison_rows, Path(figures_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = _load_model_from_checkpoint(best_exp_dir / "best_model.pt", device)
    img_sz = int(ckpt.get("image_size", best_summary["spec"].get("image_size", 128)))

    preloaded = load_uint8_cache(cache_path) if (cache_path is not None and Path(cache_path).is_file()) else None
    val_ds = RealWasteDataset(
        "validation",
        ckpt["model_name"],
        augment=False,
        image_size=(img_sz, img_sz),
        data_dir=data_dir,
        split_path=split_path,
        cache_path=cache_path,
        preloaded_cache=preloaded,
        allow_test=False,
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

    y_true = np.asarray(all_targets)
    y_pred = np.asarray(all_preds)
    confs_arr = np.asarray(all_confs, dtype=np.float64)

    metrics = compute_classification_metrics(y_true, y_pred)
    calib_stats = compute_calibration_stats(y_true, y_pred, confs_arr)
    cm_fig, cls_fig = plot_validation_diagnostics(metrics, best_summary["spec"]["title"], Path(figures_dir))
    cal_fig = plot_confidence_calibration(
        y_true, y_pred, confs_arr, calib_stats, best_summary["spec"]["title"], Path(figures_dir)
    )

    # Identify 3 weakest classes and top 3 misclassification pairs
    weakest_3 = sorted(
        [{"class": c, **vals} for c, vals in metrics["per_class"].items()],
        key=lambda x: x["f1"],
    )[:3]

    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    pairs = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                pairs.append({
                    "true_class": CLASS_NAMES[i],
                    "pred_class": CLASS_NAMES[j],
                    "count": int(cm[i, j]),
                })
    top_3_pairs = sorted(pairs, key=lambda x: -x["count"])[:3]
    plastic_to_metal = int(cm[CLASS_NAMES.index("Plastic"), CLASS_NAMES.index("Metal")])

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
        "reproducibility": {
            "seed": best_summary["environment"].get("seed", 42),
            "torch_version": torch.__version__,
            "git_commit": _get_git_commit(),
        },
        "best_experiment": {
            "exp_id": best_summary["spec"]["exp_id"],
            "title": best_summary["spec"]["title"],
            "model_name": best_summary["spec"]["model_name"],
            "image_size": img_sz,
            "best_epoch": best_summary["best_epoch"],
            "train_val_acc_gap": best_summary.get("train_val_acc_gap"),
        },
        "metrics": metrics,
        "error_analysis": {
            "weakest_3_classes": weakest_3,
            "top_3_confusion_pairs": top_3_pairs,
            "plastic_to_metal_count": plastic_to_metal,
        },
        "calibration": calib_stats,
        "representative_samples": {
            "correct_examples": correct_samples,
            "misclassified_examples": wrong_samples,
        },
        "figures": {
            "accuracy_f1_curves": str(acc_fig),
            "loss_curves": str(loss_fig),
            "val_confusion_matrix": str(cm_fig),
            "val_per_class_metrics": str(cls_fig),
            "train_val_gap_comparison": str(gap_fig),
            "confidence_calibration": str(cal_fig),
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
    parser.add_argument("--cache-path", type=Path, default=ROOT / "tmp/realwaste_224_uint8.npz")
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
