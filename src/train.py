"""Controlled Progress 4 training runner for RealWaste CNN & MobileNetV3-Small.

Supports Phase 1 (EXP-01..04), Phase 2 (EXP-05..07), and Phase 3 Controlled
Ablation Study at 224x224 with 2-Stage Fine-Tuning & Differential LR (EXP-08..11).
Never touches the held-out test set.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import subprocess
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
    save_sanity_batch_grid,
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
    image_size: int = 128
    unfreeze_last_n: int = 0
    label_smoothing: float = 0.0
    use_cosine_lr: bool = False
    backbone_lr_mult: float = 0.3
    stage1_epochs: int = 0
    stage1_lr: float = 1e-3
    warmup_epochs: int = 0
    val_loss_guard_delta: float = 0.0


DEFAULT_CLASS_WEIGHTS_PATH = ROOT / "results/model_design/class_weights.json"
DEFAULT_OUTPUT_DIR = ROOT / "results/progress4"


def _get_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def build_experiment_specs(config: dict[str, Any] | None = None) -> dict[str, ExperimentSpec]:
    """Construct controlled experiment matrix (baselines + ablation study)."""
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
            dropout=0.2,
        ),
        "exp05": ExperimentSpec(
            exp_id="exp05_mobilenet_aug_reg",
            title="EXP-05: MobileNetV3 + Aug + Regularisasi",
            model_name="mobilenet_v3_small",
            optimizer_name="AdamW",
            lr=float(mob["learning_rate"]),
            weight_decay=0.05,
            batch_size=int(mob["batch_size"]),
            max_epochs=30,
            augment=True,
            class_weighting=False,
            pretrained=True,
            freeze_backbone=True,
            dropout=0.4,
            label_smoothing=0.05,
            use_cosine_lr=True,
        ),
        "exp06": ExperimentSpec(
            exp_id="exp06_mobilenet_finetune",
            title="EXP-06: MobileNetV3 Fine-Tune + Aug + Weight",
            model_name="mobilenet_v3_small",
            optimizer_name="AdamW",
            lr=float(mob["learning_rate"]),
            weight_decay=0.05,
            batch_size=int(mob["batch_size"]),
            max_epochs=30,
            augment=True,
            class_weighting=True,
            pretrained=True,
            freeze_backbone=True,
            dropout=0.4,
            unfreeze_last_n=4,
            label_smoothing=0.05,
            use_cosine_lr=True,
            backbone_lr_mult=0.3,
        ),
        "exp07": ExperimentSpec(
            exp_id="exp07_cnn_mitigation",
            title="EXP-07: CNN Baseline Extended + Scheduler",
            model_name="cnn_baseline",
            optimizer_name=cnn["optimizer"],
            lr=0.0015,
            weight_decay=0.0,
            batch_size=int(cnn["batch_size"]),
            max_epochs=60,
            augment=True,
            class_weighting=False,
            pretrained=False,
            freeze_backbone=False,
            dropout=0.2,
            use_cosine_lr=True,
        ),
        # --- Phase 3: Controlled 224x224 Ablation Study ---
        "exp08": ExperimentSpec(
            exp_id="exp08_mobilenet_224_head",
            title="EXP-08: MobileNetV3 224x224 + Aug Bersih",
            model_name="mobilenet_v3_small",
            optimizer_name="AdamW",
            lr=3e-4,
            weight_decay=0.01,
            batch_size=32,
            max_epochs=25,
            augment=True,
            class_weighting=False,
            pretrained=True,
            freeze_backbone=True,
            dropout=0.2,
            image_size=224,
            label_smoothing=0.0,
            use_cosine_lr=True,
            val_loss_guard_delta=0.10,
        ),
        "exp09": ExperimentSpec(
            exp_id="exp09_mobilenet_224_reg",
            title="EXP-09: MobileNetV3 224x224 + Reg Terukur",
            model_name="mobilenet_v3_small",
            optimizer_name="AdamW",
            lr=3e-4,
            weight_decay=0.02,
            batch_size=32,
            max_epochs=25,
            augment=True,
            class_weighting=False,
            pretrained=True,
            freeze_backbone=True,
            dropout=0.35,
            image_size=224,
            label_smoothing=0.05,
            use_cosine_lr=True,
            val_loss_guard_delta=0.10,
        ),
        "exp10": ExperimentSpec(
            exp_id="exp10_mobilenet_224_2stage_ft",
            title="EXP-10: MobileNetV3 224x224 2-Stage FT",
            model_name="mobilenet_v3_small",
            optimizer_name="AdamW",
            lr=1.5e-4,
            weight_decay=0.02,
            batch_size=32,
            max_epochs=30,
            augment=True,
            class_weighting=False,
            pretrained=True,
            freeze_backbone=True,
            dropout=0.35,
            image_size=224,
            unfreeze_last_n=4,
            label_smoothing=0.05,
            use_cosine_lr=True,
            backbone_lr_mult=1.0 / 3.0,  # 5e-5 backbone vs 1.5e-4 head
            stage1_epochs=8,
            stage1_lr=1e-3,
            warmup_epochs=2,
            val_loss_guard_delta=0.10,
        ),
        "exp11": ExperimentSpec(
            exp_id="exp11_mobilenet_224_2stage_cw",
            title="EXP-11: MobileNetV3 224x224 2-Stage FT + CW",
            model_name="mobilenet_v3_small",
            optimizer_name="AdamW",
            lr=1.5e-4,
            weight_decay=0.02,
            batch_size=32,
            max_epochs=30,
            augment=True,
            class_weighting=True,
            pretrained=True,
            freeze_backbone=True,
            dropout=0.35,
            image_size=224,
            unfreeze_last_n=4,
            label_smoothing=0.05,
            use_cosine_lr=True,
            backbone_lr_mult=1.0 / 3.0,  # 5e-5 backbone vs 1.5e-4 head
            stage1_epochs=8,
            stage1_lr=1e-3,
            warmup_epochs=2,
            val_loss_guard_delta=0.10,
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
        model = MobileNetV3Small(
            num_classes=NUM_CLASSES,
            pretrained=spec.pretrained,
            freeze_backbone=spec.freeze_backbone,
            seed=seed,
        )
        if hasattr(model.network.classifier[2], "p"):
            model.network.classifier[2].p = float(spec.dropout)
        # For 2-stage training, start Stage 1 strictly frozen; Stage 2 unfreezes later.
        if spec.stage1_epochs == 0 and spec.unfreeze_last_n > 0:
            model.set_trainable_feature_blocks(spec.unfreeze_last_n)
        return model
    raise ValueError(f"Unsupported model_name: {spec.model_name}")


def _build_optimizer_and_scheduler(
    model: nn.Module,
    spec: ExperimentSpec,
    *,
    stage: int,
    stage_total_epochs: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
    opt_cls = getattr(torch.optim, spec.optimizer_name)

    if stage == 1 and spec.stage1_epochs > 0:
        optimizer = opt_cls(
            (p for p in model.parameters() if p.requires_grad),
            lr=spec.stage1_lr,
            weight_decay=spec.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, stage_total_epochs), eta_min=1e-5
        )
        return optimizer, scheduler

    # Single-stage or Stage 2 differential LR
    if spec.model_name == "mobilenet_v3_small" and spec.unfreeze_last_n > 0:
        assert isinstance(model, MobileNetV3Small)
        feat_params = [p for p in model.network.features.parameters() if p.requires_grad]
        cls_params = [p for p in model.network.classifier.parameters() if p.requires_grad]
        param_groups = [
            {"params": feat_params, "lr": spec.lr * spec.backbone_lr_mult},
            {"params": cls_params, "lr": spec.lr},
        ]
        optimizer = opt_cls(param_groups, weight_decay=spec.weight_decay)
    else:
        optimizer = opt_cls(
            (p for p in model.parameters() if p.requires_grad),
            lr=spec.lr,
            weight_decay=spec.weight_decay,
        )

    if not spec.use_cosine_lr:
        return optimizer, None

    warmup = spec.warmup_epochs if (stage == 2 or spec.stage1_epochs == 0) else 0
    total = max(1, stage_total_epochs)

    def lr_lambda(ep_idx: int) -> float:
        if warmup > 0 and ep_idx < warmup:
            return float(ep_idx + 1) / float(warmup + 1)
        denom = max(1, total - warmup)
        progress = min(1.0, max(0.0, float(ep_idx - warmup) / float(denom)))
        return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


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
    patience: int = 7,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train one controlled experiment (1-stage or 2-stage) and save artifacts."""
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(2)

    img_sz = (spec.image_size, spec.image_size)
    train_ds = RealWasteDataset(
        "train",
        spec.model_name,
        augment=spec.augment,
        image_size=img_sz,
        data_dir=data_dir,
        split_path=split_path,
        cache_path=cache_path,
        preloaded_cache=preloaded_cache,
    )
    val_ds = RealWasteDataset(
        "validation",
        spec.model_name,
        augment=False,
        image_size=img_sz,
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

    class_weight_tensor = (
        load_train_class_weights(weights_path).to(device) if spec.class_weighting else None
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weight_tensor, label_smoothing=float(spec.label_smoothing)
    )

    two_stage = spec.stage1_epochs > 0 and spec.stage1_epochs < spec.max_epochs
    stage = 1 if two_stage else 0
    stage1_limit = spec.stage1_epochs if two_stage else 0
    optimizer, scheduler = _build_optimizer_and_scheduler(
        model,
        spec,
        stage=stage,
        stage_total_epochs=stage1_limit if two_stage else spec.max_epochs,
    )

    exp_dir = Path(output_dir) / spec.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = exp_dir / "best_model.pt"
    stage1_ckpt_path = exp_dir / "stage1_best.pt"

    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_val_loss = float("inf")
    min_val_loss_seen = float("inf")
    best_epoch = 0
    best_metrics: dict[str, Any] = {}
    best_train_acc = 0.0
    best_train_loss = 0.0

    stage1_best_f1 = -1.0
    stage1_best_val_loss = float("inf")
    epochs_without_improve = 0
    start_total = time.perf_counter()

    trainable_params_s1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== Starting {spec.title} ({spec.exp_id}) on {device} [{spec.image_size}x{spec.image_size}] ===")
    print(
        f"Params: total={total_params:,}, trainable(start)={trainable_params_s1:,} | "
        f"train={len(train_ds)}, val={len(val_ds)} | two_stage={two_stage}"
    )

    for epoch in range(1, spec.max_epochs + 1):
        # Stage 1 -> Stage 2 transition
        if two_stage and epoch == stage1_limit + 1:
            if stage1_ckpt_path.is_file():
                s1_ckpt = torch.load(stage1_ckpt_path, map_location=device, weights_only=True)
                model.load_state_dict(s1_ckpt["model_state_dict"])
                print(
                    f"--- Stage 2 Transition @ epoch {epoch}: loaded Stage 1 best "
                    f"(ep {s1_ckpt['epoch']}, val_macro_f1={s1_ckpt['val_macro_f1']:.4f}) ---"
                )
            if isinstance(model, MobileNetV3Small) and spec.unfreeze_last_n > 0:
                model.set_trainable_feature_blocks(spec.unfreeze_last_n)
            stage = 2
            stage2_epochs = spec.max_epochs - stage1_limit
            optimizer, scheduler = _build_optimizer_and_scheduler(
                model, spec, stage=2, stage_total_epochs=stage2_epochs
            )
            epochs_without_improve = 0
            trainable_s2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(
                f"--- Unfroze last {spec.unfreeze_last_n} blocks | trainable(Stage 2)={trainable_s2:,} | "
                f"backbone_lr={spec.lr * spec.backbone_lr_mult:.2e}, head_lr={spec.lr:.2e} ---"
            )

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

        if scheduler is not None:
            scheduler.step()

        train_loss = train_loss_sum / max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        # Validation phase (strictly eval() + inference_mode())
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
        current_lr = optimizer.param_groups[-1]["lr"]

        min_val_loss_seen = min(min_val_loss_seen, val_loss)
        row = {
            "epoch": epoch,
            "stage": stage if two_stage else 1,
            "lr": round(float(current_lr), 8),
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

        # Save Stage 1 best checkpoint for Stage 2 initialization
        if two_stage and stage == 1:
            s1_improved = (val_metrics["macro_f1"] > stage1_best_f1) or (
                val_metrics["macro_f1"] == stage1_best_f1 and val_loss < stage1_best_val_loss
            )
            if s1_improved:
                stage1_best_f1 = val_metrics["macro_f1"]
                stage1_best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "val_macro_f1": stage1_best_f1,
                        "val_loss": round(val_loss, 6),
                        "model_state_dict": model.state_dict(),
                    },
                    stage1_ckpt_path,
                )

        # Global best selection with optional val_loss guard against severe overfitting
        loss_within_guard = (
            spec.val_loss_guard_delta <= 0.0
            or best_epoch == 0
            or val_loss <= min_val_loss_seen + spec.val_loss_guard_delta
        )
        score_improved = (val_metrics["macro_f1"] > best_f1) or (
            val_metrics["macro_f1"] == best_f1 and val_loss < best_val_loss
        )
        if score_improved and loss_within_guard:
            best_f1 = val_metrics["macro_f1"]
            best_val_loss = val_loss
            best_epoch = epoch
            best_metrics = val_metrics
            best_train_acc = train_acc
            best_train_loss = train_loss
            epochs_without_improve = 0
            torch.save(
                {
                    "exp_id": spec.exp_id,
                    "title": spec.title,
                    "model_name": spec.model_name,
                    "image_size": spec.image_size,
                    "dropout": spec.dropout,
                    "unfreeze_last_n": spec.unfreeze_last_n,
                    "stage1_epochs": spec.stage1_epochs,
                    "epoch": epoch,
                    "val_macro_f1": best_f1,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_loss": round(val_loss, 6),
                    "model_state_dict": model.state_dict(),
                },
                best_ckpt_path,
            )
        else:
            # Apply early stopping counter only in Stage 2 (or single-stage)
            if not (two_stage and stage == 1):
                epochs_without_improve += 1

        print(
            f"[S{row['stage']}] Ep {epoch:02d}/{spec.max_epochs:02d} lr={current_lr:.1e} | "
            f"tr_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_metrics['accuracy']:.4f} "
            f"macro_f1={val_metrics['macro_f1']:.4f} ({epoch_sec:.1f}s)"
        )

        if (not (two_stage and stage == 1)) and patience > 0 and epochs_without_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch} (best epoch: {best_epoch}).")
            break

    total_seconds = time.perf_counter() - start_total
    final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

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
            "git_commit": _get_git_commit(),
            "seed": seed,
        },
        "model_stats": {
            "total_parameters": total_params,
            "trainable_parameters": final_trainable_params,
        },
        "dataset_counts": {"train": len(train_ds), "validation": len(val_ds), "test_used": False},
        "best_epoch": best_epoch,
        "best_train_loss": round(best_train_loss, 6),
        "best_train_accuracy": round(best_train_acc, 6),
        "best_val_loss": round(best_val_loss, 6),
        "best_val_accuracy": best_metrics.get("accuracy", 0.0),
        "best_val_macro_f1": best_f1,
        "best_val_macro_precision": best_metrics.get("macro_precision", 0.0),
        "best_val_macro_recall": best_metrics.get("macro_recall", 0.0),
        "train_val_acc_gap": round(best_train_acc - best_metrics.get("accuracy", 0.0), 6),
        "epochs_ran": len(history),
        "total_training_seconds": round(total_seconds, 2),
        "best_checkpoint": str(best_ckpt_path),
    }
    (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_comparison_table(output_dir: Path) -> list[dict[str, Any]]:
    """Aggregate all completed exp* summaries under output_dir into comparison JSON/CSV."""
    exp_dirs = sorted(p for p in output_dir.iterdir() if p.is_dir() and (p / "summary.json").is_file())
    comparison: list[dict[str, Any]] = []
    for ed in exp_dirs:
        s = json.loads((ed / "summary.json").read_text(encoding="utf-8"))
        hist = json.loads((ed / "history.json").read_text(encoding="utf-8"))
        best_row = next(r for r in hist if r["epoch"] == s["best_epoch"])
        tr_acc = float(s.get("best_train_accuracy", best_row["train_accuracy"]))
        va_acc = float(s["best_val_accuracy"])
        comparison.append({
            "exp_id": s["spec"]["exp_id"],
            "title": s["spec"]["title"],
            "model_name": s["spec"]["model_name"],
            "image_size": s["spec"].get("image_size", 128),
            "augment": s["spec"]["augment"],
            "class_weighting": s["spec"]["class_weighting"],
            "unfreeze_last_n": s["spec"].get("unfreeze_last_n", 0),
            "stage1_epochs": s["spec"].get("stage1_epochs", 0),
            "best_epoch": s["best_epoch"],
            "epochs_ran": s["epochs_ran"],
            "best_train_loss": round(float(s.get("best_train_loss", best_row["train_loss"])), 6),
            "best_train_accuracy": round(tr_acc, 6),
            "best_val_loss": s["best_val_loss"],
            "best_val_accuracy": va_acc,
            "train_val_acc_gap": round(tr_acc - va_acc, 6),
            "best_val_macro_f1": s["best_val_macro_f1"],
            "best_val_macro_precision": s.get("best_val_macro_precision", best_row.get("val_macro_precision", 0.0)),
            "best_val_macro_recall": s.get("best_val_macro_recall", best_row.get("val_macro_recall", 0.0)),
            "total_training_seconds": s["total_training_seconds"],
        })

    if comparison:
        (output_dir / "experiment_comparison.json").write_text(
            json.dumps(comparison, indent=2), encoding="utf-8"
        )
        with (output_dir / "experiment_comparison.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(comparison[0].keys()))
            writer.writeheader()
            writer.writerows(comparison)
    return comparison


def run_all_experiments(
    exp_keys: list[str],
    *,
    data_dir: str | Path | None = None,
    split_path: Path = DEFAULT_SPLIT_PATH,
    cache_path: Path | None = None,
    weights_path: Path = DEFAULT_CLASS_WEIGHTS_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    figures_dir: Path | None = None,
    epochs_override: int | None = None,
    stage1_override: int | None = None,
    pretrained_override: bool | None = None,
    patience: int = 7,
    seed: int = 42,
) -> list[dict[str, Any]]:
    specs_map = build_experiment_specs()
    if exp_keys == ["all"]:
        selected = list(specs_map.keys())
    elif exp_keys == ["mitigation"]:
        selected = ["exp05", "exp06", "exp07"]
    elif exp_keys == ["ablation224"]:
        selected = ["exp08", "exp09", "exp10", "exp11"]
    else:
        selected = exp_keys

    for k in selected:
        if k not in specs_map:
            raise ValueError(f"Unknown experiment '{k}'. Choose from {list(specs_map.keys())}")

    # Load or build uint8 cache matched to target resolution
    first_sz = specs_map[selected[0]].image_size
    preloaded_cache = None
    if cache_path is not None:
        cache_file = Path(cache_path)
        if not cache_file.is_file():
            if data_dir is None:
                raise FileNotFoundError(f"Cache file {cache_file} missing and no --data-dir provided.")
            print(f"Building {first_sz}x{first_sz} uint8 cache at {cache_file} ...")
            build_uint8_cache(data_dir, cache_file, split_path, image_size=(first_sz, first_sz))
        preloaded_cache = load_uint8_cache(cache_file)

    # Save visual sanity batch check grid for preprocessing audit
    if figures_dir is not None:
        s0 = specs_map[selected[0]]
        sz = (s0.image_size, s0.image_size)
        tr_sanity = RealWasteDataset(
            "train", s0.model_name, augment=True, image_size=sz,
            data_dir=data_dir, split_path=split_path, cache_path=cache_path, preloaded_cache=preloaded_cache,
        )
        va_sanity = RealWasteDataset(
            "validation", s0.model_name, augment=False, image_size=sz,
            data_dir=data_dir, split_path=split_path, cache_path=cache_path, preloaded_cache=preloaded_cache,
        )
        sanity_path = save_sanity_batch_grid(tr_sanity, va_sanity, Path(figures_dir) / "p4_sanity_train_val_batch.png", seed=seed)
        print(f"Saved preprocessing sanity grid: {sanity_path}")

    summaries: list[dict[str, Any]] = []
    for key in selected:
        spec = specs_map[key]
        overrides: dict[str, Any] = {}
        if epochs_override is not None:
            overrides["max_epochs"] = epochs_override
        if stage1_override is not None and spec.stage1_epochs > 0:
            overrides["stage1_epochs"] = stage1_override
        if pretrained_override is not None and spec.model_name == "mobilenet_v3_small":
            overrides["pretrained"] = pretrained_override
        if overrides:
            spec = ExperimentSpec(**{**asdict(spec), **overrides})

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
    write_comparison_table(output_dir)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["ablation224"],
        help="Experiment keys: exp01..exp11, all, mitigation, or ablation224",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--cache-path", type=Path, default=ROOT / "tmp/realwaste_224_uint8.npz")
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_CLASS_WEIGHTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--stage1-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet download (for offline synthetic smoke tests)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_all_experiments(
        args.experiments,
        data_dir=args.data_dir,
        split_path=args.split_path,
        cache_path=args.cache_path,
        weights_path=args.weights_path,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        epochs_override=args.epochs,
        stage1_override=args.stage1_epochs,
        pretrained_override=False if args.no_pretrained else None,
        patience=args.patience,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
