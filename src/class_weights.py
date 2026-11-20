"""Class weighting derivation for RealWaste imbalanced multiclass classification.

Calculates inverse class frequency weights derived STRICTLY from the training set
(3,327 samples) to prevent data snooping and data leakage to validation/test sets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .models.config import CLASS_NAMES, NUM_CLASSES
except (ImportError, ModuleNotFoundError):
    # Fallback to direct import without triggering models.__init__ (which requires torch)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
    from config import CLASS_NAMES, NUM_CLASSES

ROOT = Path(__file__).resolve().parents[1]


def compute_training_class_weights(
    data_split_summary_path: Path = ROOT / "results/data_split_summary.json",
    beta: float = 0.999,
) -> dict[str, Any]:
    """Compute balanced inverse frequency and effective-number class weights."""
    with open(data_split_summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    classes_dict = summary["classes"]
    train_counts = {c: classes_dict[c]["train"] for c in CLASS_NAMES}
    total_train = sum(train_counts.values())

    # 1. Standard Inverse Class Frequency: w_c = N_train / (K * N_c_train)
    raw_inv = {c: total_train / (NUM_CLASSES * train_counts[c]) for c in CLASS_NAMES}
    mean_raw = sum(raw_inv.values()) / NUM_CLASSES
    normalized_inv = {c: round(raw_inv[c] / mean_raw, 4) for c in CLASS_NAMES}

    # 2. Effective Number of Samples (Cui et al., CVPR 2019): E_n = (1 - beta) / (1 - beta^n)
    effective_num = {c: (1.0 - beta) / (1.0 - (beta ** train_counts[c])) for c in CLASS_NAMES}
    mean_eff = sum(effective_num.values()) / NUM_CLASSES
    normalized_effective = {c: round(effective_num[c] / mean_eff, 4) for c in CLASS_NAMES}

    # PyTorch-ready weight tensor list ordered by CLASS_NAMES ID 0..8
    torch_tensor_weights = [normalized_inv[c] for c in CLASS_NAMES]

    result = {
        "metadata": {
            "source_split": "train_only",
            "total_training_samples": total_train,
            "num_classes": NUM_CLASSES,
            "anti_leakage_guarantee": "No validation or test labels used in calculation",
        },
        "class_counts_train": train_counts,
        "inverse_class_frequency_weights": {
            "raw": {c: round(raw_inv[c], 4) for c in CLASS_NAMES},
            "normalized_mean_1": normalized_inv,
        },
        "effective_number_weights_beta_0999": {
            "normalized_mean_1": normalized_effective,
        },
        "pytorch_weight_tensor_code": f"torch.tensor({torch_tensor_weights}, dtype=torch.float32)",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "results/data_split_summary.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results/model_design/class_weights.json")
    args = parser.parse_args()

    weights = compute_training_class_weights(args.summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(weights, indent=2), encoding="utf-8")

    print("=== CLASS WEIGHTS COMPUTATION COMPLETED ===")
    print("Class (Ordered 0..8) | Train Count | Normalized Inverse Weight")
    print("-" * 55)
    for c, w in weights["inverse_class_frequency_weights"]["normalized_mean_1"].items():
        cnt = weights["class_counts_train"][c]
        print(f"{c:20s} | {cnt:11d} | {w:23.4f}")
    print("-" * 55)
    print("PyTorch CrossEntropyLoss ready tensor:")
    print(weights["pytorch_weight_tensor_code"])
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()
