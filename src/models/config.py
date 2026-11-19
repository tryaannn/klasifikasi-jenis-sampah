"""Stable RealWaste class IDs and explicit, architecture-specific input policies."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CLASS_NAMES = (
    "Cardboard", "Food Organics", "Glass", "Metal", "Miscellaneous Trash",
    "Paper", "Plastic", "Textile Trash", "Vegetation",
)
CLASS_TO_IDX = {label: i for i, label in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = dict(enumerate(CLASS_NAMES))
NUM_CLASSES = len(CLASS_NAMES)
ORIGINAL_SIZE = (524, 524, 3)  # HWC
INPUT_SIZE = (128, 128)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = ROOT / "configs/model_initial.yaml") -> dict[str, Any]:
    """Read the experiment proposal; reject accidental class/input policy drift."""
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if tuple(config["classes"]) != CLASS_NAMES:
        raise ValueError("Class IDs must preserve the original RealWaste mapping")
    for name, model in config["models"].items():
        if model["num_classes"] != NUM_CLASSES or tuple(model["input_size"]) != INPUT_SIZE:
            raise ValueError(f"Unexpected input/output specification for {name}")
    return config
