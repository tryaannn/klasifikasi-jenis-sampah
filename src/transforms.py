"""Standardized preprocessing and augmentation pipelines for RealWaste models.

Enforces architecture-specific normalization policies:
- CNN Baseline: Bilinear resize to 128x128, scaling pixels to [0, 1].
- MobileNetV3-Small: Bilinear resize to 128x128, scaling to [0, 1] followed by ImageNet mean and std.

Controlled augmentation variants for training:
- Random horizontal flip (p=0.5)
- Random rotation (+/- 15 degrees)
- Random zoom/scale (0.9 to 1.1)
- Color jitter (brightness +/- 0.1, contrast +/- 0.1)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

try:
    from torchvision import transforms
except ImportError:
    transforms = None  # fallback if torchvision not installed in current env

from .models.config import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE

ROOT = Path(__file__).resolve().parents[1]


def get_transforms(
    model_name: str,
    split: str = "train",
    augment: bool = False,
    image_size: tuple[int, int] = INPUT_SIZE,
) -> Any:
    """Construct composable transform pipeline for training, validation, or testing."""
    if transforms is None:
        raise RuntimeError("torchvision is required for get_transforms")

    t_list = []

    # Augmentation applied ONLY on training split when augment=True
    if split == "train" and augment:
        t_list.extend([
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.1), ratio=(0.95, 1.05), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ])
    else:
        t_list.append(transforms.Resize(image_size, antialias=True))

    # Convert PIL Image or NumPy array to FloatTensor in [0, 1]
    t_list.append(transforms.ToTensor())

    # Architecture-specific normalization
    if model_name.lower() in ("mobilenet_v3_small", "mobilenetv3small"):
        t_list.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    elif model_name.lower() in ("cnn_baseline", "cnnbaseline"):
        pass  # [0, 1] scaling is already completed by ToTensor()
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return transforms.Compose(t_list)


def generate_synthetic_augmentation_preview(
    output_path: Path = ROOT / "figures/data_augmentation_preview.png",
    seed: int = 42,
) -> Path:
    """Generate visual demonstration of the controlled augmentation policy on a geometric pattern."""
    np.random.seed(seed)
    # Create a synthetic 128x128 test patch simulating waste texture & edge
    h, w = 128, 128
    grid = np.zeros((h, w, 3), dtype=np.uint8)
    grid[:, :] = [210, 215, 220]  # background
    # Add colored geometric shapes representing waste item
    grid[30:98, 30:98] = [60, 130, 210]  # blue plastic-like object
    grid[45:83, 45:83] = [230, 175, 40]   # inner cardboard-like label
    # Add a thin diagonal wire simulating metal
    for i in range(20, 108):
        grid[i, i] = [40, 40, 40]
        grid[i, min(127, i + 1)] = [180, 180, 180]

    pil_img = Image.fromarray(grid)

    aug_pipeline = transforms.Compose([
        transforms.RandomResizedCrop((h, w), scale=(0.9, 1.1), ratio=(0.95, 1.05), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
    ])

    fig, axes = plt.subplots(1, 6, figsize=(14, 2.8), dpi=300)

    axes[0].imshow(pil_img)
    axes[0].set_title("Original (Kontrol)\nResize 128x128", fontsize=9, fontweight="bold")
    axes[0].axis("off")

    for i in range(1, 6):
        torch.manual_seed(seed + i)
        aug_img = aug_pipeline(pil_img)
        axes[i].imshow(aug_img)
        axes[i].set_title(f"Augmentasi #{i}\nRotasi, Zoom, Flip", fontsize=9)
        axes[i].axis("off")

    plt.suptitle("Preview Kebijakan Augmentasi Terkontrol Progres 3/4 (RealWaste)", fontsize=11, fontweight="bold", y=1.03)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return output_path
