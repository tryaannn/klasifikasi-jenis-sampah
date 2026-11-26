"""Offline synthetic smoke tests for Progress 4 dataset, training, and validation evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.dataset import RealWasteDataset, build_uint8_cache, resolve_realwaste_root
from src.evaluate_val import evaluate_validation
from src.models import CLASS_NAMES
from src.train import run_all_experiments


def _create_synthetic_realwaste(root: Path, per_class: int = 6) -> tuple[Path, Path, Path]:
    data_root = root / "raw_data" / "RealWaste"
    split_dict = {"train": [], "validation": [], "test": []}
    rng = np.random.default_rng(42)

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = data_root / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            fname = f"{cls_name}_{i}.jpg"
            arr = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
            arr[:, :, 0] = (arr[:, :, 0] // 2) + cls_idx * 12
            Image.fromarray(arr).save(cls_dir / fname)
            rel = f"{cls_name}/{fname}"
            if i < per_class - 2:
                split_dict["train"].append(rel)
            elif i == per_class - 2:
                split_dict["validation"].append(rel)
            else:
                split_dict["test"].append(rel)

    split_path = root / "split.json"
    split_path.write_text(json.dumps(split_dict), encoding="utf-8")

    weights_payload = {
        "metadata": {"source_split": "train_only"},
        "inverse_class_frequency_weights": {
            "normalized_mean_1": {c: 1.0 for c in CLASS_NAMES}
        },
    }
    weights_path = root / "class_weights.json"
    weights_path.write_text(json.dumps(weights_payload), encoding="utf-8")
    return root / "raw_data", split_path, weights_path


def test_anti_leakage_guards_and_cache(tmp_path: Path) -> None:
    raw_dir, split_path, _ = _create_synthetic_realwaste(tmp_path, per_class=4)
    resolved = resolve_realwaste_root(raw_dir)
    assert resolved.name == "RealWaste"

    cache_path = build_uint8_cache(raw_dir, tmp_path / "cache.npz", split_path)
    assert cache_path.is_file()

    # Test split MUST be blocked by default during Progress 4
    with pytest.raises(PermissionError):
        RealWasteDataset("test", "cnn_baseline", split_path=split_path, cache_path=cache_path)

    # Augmentation MUST be blocked on validation split
    with pytest.raises(ValueError):
        RealWasteDataset(
            "validation", "cnn_baseline", augment=True, split_path=split_path, cache_path=cache_path
        )


def test_end_to_end_progress4_smoke(tmp_path: Path) -> None:
    raw_dir, split_path, weights_path = _create_synthetic_realwaste(tmp_path, per_class=5)
    cache_path = tmp_path / "realwaste_128_uint8.npz"
    out_dir = tmp_path / "results_p4"
    fig_dir = tmp_path / "figures_p4"

    summaries = run_all_experiments(
        ["exp01", "exp02", "exp03", "exp04"],
        data_dir=raw_dir,
        split_path=split_path,
        cache_path=cache_path,
        weights_path=weights_path,
        output_dir=out_dir,
        epochs_override=2,
        pretrained_override=False,
        patience=5,
        seed=42,
    )

    assert len(summaries) == 4
    assert all(not s["dataset_counts"]["test_used"] for s in summaries)
    assert (out_dir / "experiment_comparison.json").is_file()
    assert (out_dir / "experiment_comparison.csv").is_file()
    for s in summaries:
        exp_dir = out_dir / s["spec"]["exp_id"]
        assert (exp_dir / "best_model.pt").is_file()
        assert (exp_dir / "history.csv").is_file()
        assert (exp_dir / "summary.json").is_file()

    report = evaluate_validation(
        results_dir=out_dir,
        figures_dir=fig_dir,
        split_path=split_path,
        cache_path=cache_path,
    )
    assert report["evaluation_split"] == "validation_only"
    assert report["test_set_untouched"] is True
    assert report["validation_count"] == 9  # 1 validation sample per class
    for fig_path in report["figures"].values():
        assert Path(fig_path).is_file()
    assert (out_dir / "val_best_model_evaluation.json").is_file()
