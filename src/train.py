"""Controlled Progress 4 training runner for RealWaste CNN & MobileNetV3-Small.

Runs EXP-01..EXP-04 on CPU or CUDA GPU, logs per-epoch metrics, saves best_model.pt
by highest validation Macro F1, and never touches the held-out test set.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import (
    DEFAULT_SPLIT_PATH,
    RealWasteDataset,
    build_uint8_cache,
    load_uint8_cache,
)
from .models import CLASS_NAMES, CNNBaseline, MobileNetV3Small, NUM_CLASSES, load_config
from .models.config import ROOT


@dataclass(frozen=True)
class ExperimentSpec:
    exp_id: str
    title: str
    model_name: str
    optimizer_name: str
    lr: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    augment: bool
    class_weighting: bool
    pretrained: bool
    freeze_backbone: bool
    dropout: float = 0.3


DEFAULT_CLASS_WEIGHTS_PATH = ROOT / "results/model_design/class_weights.json"
DEFAULT_OUTPUT_DIR = ROOT / "results/progress4"


def build_experiment_specs(config: dict[str, Any] | None = None) -> dict[str, ExperimentSpec]:
    """Construct controlled experiment matrix from configs/model_initial.yaml."""
    cfg = config or load_config()
    cnn = cfg["models"]["cnn_baseline"]
    mob = cfg["models"]["mobilenet_v3_small"]

    return {
        "exp01": ExperimentSpec(
            exp_id="exp01_cnn_control",
            title="EXP-01: CNN Baseline Kontrol",
            model_name="cnn_baseline",
            optimizer_name=cnn["optimizer"],
            lr=float(cnn["learning_rate"]),
            weight_decay=float(cnn["weight_decay"]),
            batch_size=int(cnn["batch_size"]),
            max_epochs=int(cnn["max_epochs_planned"]),
            augment=False,
            class_weighting=False,
            pretrained=False,
            freeze_backbone=False,
            dropout=float(cnn["dropout"]),
        ),
        "exp02": ExperimentSpec(
            exp_id="exp02_cnn_aug",
            title="EXP-02: CNN Baseline + Augmentasi",
            model_name="cnn_baseline",
            optimizer_name=cnn["optimizer"],
            lr=float(cnn["learning_rate"]),
            weight_decay=float(cnn["weight_decay"]),
            batch_size=int(cnn["batch_size"]),
            max_epochs=int(cnn["max_epochs_planned"]),
            augment=True,
            class_weighting=False,
            pretrained=False,
            freeze_backbone=False,
            dropout=float(cnn["dropout"]),
        ),
        "exp03": ExperimentSpec(
            exp_id="exp03_cnn_weighted",
            title="EXP-03: CNN Baseline + Class Weighting",
            model_name="cnn_baseline",
            optimizer_name=cnn["optimizer"],
            lr=float(cnn["learning_rate"]),
            weight_decay=float(cnn["weight_decay"]),
            batch_size=int(cnn["batch_size"]),
            max_epochs=int(cnn["max_epochs_planned"]),
            augment=False,
            class_weighting=True,
            pretrained=False,
            freeze_backbone=False,
            dropout=float(cnn["dropout"]),
        ),
        "exp04": ExperimentSpec(
            exp_id="exp04_mobilenet_tl",
            title="EXP-04: MobileNetV3-Small Transfer Learning",
            model_name="mobilenet_v3_small",
            optimizer_name=mob["optimizer"],
            lr=float(mob["learning_rate"]),
            weight_decay=float(mob["weight_decay"]),
            batch_size=int(mob["batch_size"]),
            max_epochs=int(mob["max_epochs_planned"]),
            augment=False,
            class_weighting=False,
            pretrained=True,
            freeze_backbone=bool(mob["freeze_backbone"]),
        ),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_train_class_weights(weights_path: Path = DEFAULT_CLASS_WEIGHTS_PATH) -> torch.Tensor:
    """Load normalized inverse class frequency weights derived strictly from training set."""
    data = json.loads(Path(weights_path).read_text(encoding="utf-8"))
    if data.get("metadata", {}).get("source_split") != "train_only":
        raise ValueError("Class weights must be derived from train_only split.")
    inv = data["inverse_class_frequency_weights"]["normalized_mean_1"]
    return torch.tensor([float(inv[c]) for c in CLASS_NAMES], dtype=torch.float32)


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES
) -> dict[str, Any]:
    """Compute confusion matrix, accuracy, and macro precision/recall/F1 without external dependencies."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    support = cm.sum(axis=1).astype(np.int64)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )

    total = int(cm.sum())
    accuracy = float(tp.sum() / total) if total > 0 else 0.0

    return {
        "accuracy": round(accuracy, 6),
        "macro_precision": round(float(precision.mean()), 6),
        "macro_recall": round(float(recall.mean()), 6),
        "macro_f1": round(float(f1.mean()), 6),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": round(float(precision[i]), 6),
                "recall": round(float(recall[i]), 6),
                "f1": round(float(f1[i]), 6),
                "support": int(support[i]),
            }
            for i in range(num_classes)
        },
    }


def instantiate_model(spec: ExperimentSpec, seed: int) -> nn.Module:
    if spec.model_name == "cnn_baseline":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return CNNBaseline(num_classes=NUM_CLASSES, dropout=spec.dropout)
    if spec.model_name == "mobilenet_v3_small":
        return MobileNetV3Small(
            num_classes=NUM_CLASSES,
            pretrained=spec.pretrained,
            freeze_backbone=spec.freeze_backbone,
            seed=seed,
        )
    raise ValueError(f"Unsupported model_name: {spec.model_name}")


def run_experiment(
    spec: ExperimentSpec,
    *,
    data_dir: str | Path | None = None,
    split_path: Path = DEFAULT_SPLIT_PATH,
    cache_path: Path | None = None,
    preloaded_cache: dict[str, Any] | None = None,
    weights_path: Path = DEFAULT_CLASS_WEIGHTS_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seed: int = 42,
    patience: int = 8,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train one controlled experiment and save checkpoint, epoch logs, and summary."""
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(2)

    train_ds = RealWasteDataset(
        "train",
        spec.model_name,
        augment=spec.augment,
        data_dir=data_dir,
        split_path=split_path,
        cache_path=cache_path,
        preloaded_cache=preloaded_cache,
    )
    val_ds = RealWasteDataset(
        "validation",
        spec.model_name,
        augment=False,
        data_dir=data_dir,
        split_path=split_path,
        cache_path=cache_path,
        preloaded_cache=preloaded_cache,
    )

    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=spec.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        generator=gen,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=spec.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = instantiate_model(spec, seed=seed).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    class_weight_tensor = (
        load_train_class_weights(weights_path).to(device) if spec.class_weighting else None
    )
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    opt_cls = getattr(torch.optim, spec.optimizer_name)
    optimizer = opt_cls(
        (p for p in model.parameters() if p.requires_grad),
        lr=spec.lr,
        weight_decay=spec.weight_decay,
    )

    exp_dir = Path(output_dir) / spec.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = exp_dir / "best_model.pt"

    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_metrics: dict[str, Any] = {}
    epochs_without_improve = 0
    start_total = time.perf_counter()

    print(f"\n=== Starting {spec.title} ({spec.exp_id}) on {device} ===")
    print(f"Params: total={total_params:,}, trainable={trainable_params:,} | "
          f"train={len(train_ds)}, val={len(val_ds)}")

    for epoch in range(1, spec.max_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            train_loss_sum += float(loss.item()) * bs
            preds = logits.argmax(dim=1)
            train_correct += int((preds == labels).sum().item())
            train_total += bs

        train_loss = train_loss_sum / max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        val_total = 0
        all_targets: list[np.ndarray] = []
        all_preds: list[np.ndarray] = []

        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                loss = criterion(logits, labels)
                bs = labels.size(0)
                val_loss_sum += float(loss.item()) * bs
                val_total += bs
                all_targets.append(labels.cpu().numpy())
                all_preds.append(logits.argmax(dim=1).cpu().numpy())

        val_loss = val_loss_sum / max(1, val_total)
        y_true = np.concatenate(all_targets)
        y_pred = np.concatenate(all_preds)
        val_metrics = compute_classification_metrics(y_true, y_pred)
        epoch_sec = time.perf_counter() - t0

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch_seconds": round(epoch_sec, 3),
        }
        history.append(row)

        improved = (val_metrics["macro_f1"] > best_f1) or (
            val_metrics["macro_f1"] == best_f1 and val_loss < best_val_loss
        )
        if improved:
            best_f1 = val_metrics["macro_f1"]
            best_val_loss = val_loss
            best_epoch = epoch
            best_metrics = val_metrics
            epochs_without_improve = 0
            torch.save(
                {
                    "exp_id": spec.exp_id,
                    "title": spec.title,
                    "model_name": spec.model_name,
                    "dropout": spec.dropout,
                    "epoch": epoch,
                    "val_macro_f1": best_f1,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_loss": round(val_loss, 6),
                    "model_state_dict": model.state_dict(),
                },
                best_ckpt_path,
            )
        else:
            epochs_without_improve += 1

        print(
            f"Epoch {epoch:02d}/{spec.max_epochs:02d} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_metrics['accuracy']:.4f} "
            f"macro_f1={val_metrics['macro_f1']:.4f} ({epoch_sec:.1f}s)"
        )

        if patience > 0 and epochs_without_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch} (best epoch: {best_epoch}).")
            break

    total_seconds = time.perf_counter() - start_total

    # Save history CSV & JSON
    csv_path = exp_dir / "history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    (exp_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    summary = {
        "spec": asdict(spec),
        "environment": {
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "torch_version": torch.__version__,
            "seed": seed,
        },
        "model_stats": {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        },
        "dataset_counts": {"train": len(train_ds), "validation": len(val_ds), "test_used": False},
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "best_val_accuracy": best_metrics.get("accuracy", 0.0),
        "best_val_macro_f1": best_f1,
        "best_val_macro_precision": best_metrics.get("macro_precision", 0.0),
        "best_val_macro_recall": best_metrics.get("macro_recall", 0.0),
        "epochs_ran": len(history),
        "total_training_seconds": round(total_seconds, 2),
        "best_checkpoint": str(best_ckpt_path),
    }
    (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_all_experiments(
    exp_keys: list[str],
    *,
    data_dir: str | Path | None = None,
    split_path: Path = DEFAULT_SPLIT_PATH,
    cache_path: Path | None = None,
    weights_path: Path = DEFAULT_CLASS_WEIGHTS_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    epochs_override: int | None = None,
    pretrained_override: bool | None = None,
    patience: int = 8,
    seed: int = 42,
) -> list[dict[str, Any]]:
    specs_map = build_experiment_specs()
    selected = list(specs_map.keys()) if exp_keys == ["all"] else exp_keys

    preloaded_cache = None
    if cache_path is not None:
        cache_file = Path(cache_path)
        if not cache_file.is_file():
            if data_dir is None:
                raise FileNotFoundError(f"Cache file {cache_file} missing and no --data-dir provided.")
            print(f"Building 128x128 uint8 cache at {cache_file} ...")
            build_uint8_cache(data_dir, cache_file, split_path)
        preloaded_cache = load_uint8_cache(cache_file)

    summaries: list[dict[str, Any]] = []
    for key in selected:
        if key not in specs_map:
            raise ValueError(f"Unknown experiment '{key}'. Choose from {list(specs_map.keys())}")
        spec = specs_map[key]
        if epochs_override is not None:
            spec = ExperimentSpec(**{**asdict(spec), "max_epochs": epochs_override})
        if pretrained_override is not None and spec.model_name == "mobilenet_v3_small":
            spec = ExperimentSpec(**{**asdict(spec), "pretrained": pretrained_override})

        summary = run_experiment(
            spec,
            data_dir=data_dir,
            split_path=split_path,
            cache_path=cache_path,
            preloaded_cache=preloaded_cache,
            weights_path=weights_path,
            output_dir=output_dir,
            seed=seed,
            patience=patience,
        )
        summaries.append(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = [
        {
            "exp_id": s["spec"]["exp_id"],
            "title": s["spec"]["title"],
            "model_name": s["spec"]["model_name"],
            "augment": s["spec"]["augment"],
            "class_weighting": s["spec"]["class_weighting"],
            "best_epoch": s["best_epoch"],
            "epochs_ran": s["epochs_ran"],
            "best_val_loss": s["best_val_loss"],
            "best_val_accuracy": s["best_val_accuracy"],
            "best_val_macro_f1": s["best_val_macro_f1"],
            "total_training_seconds": s["total_training_seconds"],
        }
        for s in summaries
    ]
    (output_dir / "experiment_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    with (output_dir / "experiment_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison[0].keys()))
        writer.writeheader()
        writer.writerows(comparison)

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", default=["all"],
                        help="Experiment keys: exp01 exp02 exp03 exp04 or all")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--cache-path", type=Path, default=ROOT / "tmp/realwaste_128_uint8.npz")
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_CLASS_WEIGHTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Disable ImageNet download (for offline synthetic smoke tests)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_all_experiments(
        args.experiments,
        data_dir=args.data_dir,
        split_path=args.split_path,
        cache_path=args.cache_path,
        weights_path=args.weights_path,
        output_dir=args.output_dir,
        epochs_override=args.epochs,
        pretrained_override=False if args.no_pretrained else None,
        patience=args.patience,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
