"""Dataset loading, uint8 caching (128x128 / 224x224), and batch sanity verification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .models.config import CLASS_NAMES, CLASS_TO_IDX, INPUT_SIZE, ROOT
from .transforms import get_transforms, unnormalize_for_display

DEFAULT_SPLIT_PATH = ROOT / "results/data_split_indices.json"


def normalize_split_name(split: str) -> str:
    """Map 'val' alias to canonical 'validation' split key."""
    s = split.strip().lower()
    if s == "val":
        return "validation"
    if s not in ("train", "validation", "test"):
        raise ValueError(f"Unsupported split '{split}'. Expected train, validation, or test.")
    return s


def resolve_realwaste_root(data_dir: str | Path) -> Path:
    """Locate the directory directly containing all 9 RealWaste class folders."""
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"Data directory does not exist: {base}")

    required = set(CLASS_NAMES)
    if required.issubset({p.name for p in base.iterdir() if p.is_dir()}):
        return base

    for candidate in sorted(base.rglob("*")):
        if candidate.is_dir():
            subdirs = {p.name for p in candidate.iterdir() if p.is_dir()}
            if required.issubset(subdirs):
                return candidate

    raise FileNotFoundError(
        f"Could not find all 9 RealWaste class folders {CLASS_NAMES} under '{base}'."
    )


def load_split_records(
    split: str,
    split_path: str | Path = DEFAULT_SPLIT_PATH,
    *,
    allow_test: bool = False,
) -> list[tuple[str, int]]:
    """Return (relative_file_path, label_idx) pairs for a split."""
    canonical = normalize_split_name(split)
    if canonical == "test" and not allow_test:
        raise PermissionError(
            "Test split is locked during Progress 4. Set allow_test=True only for final Progress 5 evaluation."
        )
    data = json.loads(Path(split_path).read_text(encoding="utf-8"))
    records: list[tuple[str, int]] = []
    for rel_path in data[canonical]:
        class_name = rel_path.replace("\\", "/").split("/")[0]
        if class_name not in CLASS_TO_IDX:
            raise ValueError(f"Unknown class '{class_name}' in split path '{rel_path}'")
        records.append((rel_path.replace("\\", "/"), CLASS_TO_IDX[class_name]))
    return records


def build_uint8_cache(
    data_dir: str | Path,
    cache_path: str | Path,
    split_path: str | Path = DEFAULT_SPLIT_PATH,
    image_size: tuple[int, int] = INPUT_SIZE,
) -> Path:
    """Resize all split images once to HxWx3 uint8 and store in an uncompressed .npz archive."""
    rw_root = resolve_realwaste_root(data_dir)
    split_data = json.loads(Path(split_path).read_text(encoding="utf-8"))

    all_files: list[str] = []
    for key in ("train", "validation", "test"):
        all_files.extend(split_data.get(key, []))
    unique_files = list(dict.fromkeys(f.replace("\\", "/") for f in all_files))

    h, w = image_size
    images = np.empty((len(unique_files), h, w, 3), dtype=np.uint8)
    labels = np.empty((len(unique_files),), dtype=np.int64)

    for idx, rel_path in enumerate(unique_files):
        img_path = rw_root / rel_path
        if not img_path.is_file():
            raise FileNotFoundError(f"Missing image referenced in split JSON: {img_path}")
        with Image.open(img_path) as img:
            rgb = img.convert("RGB")
            if rgb.size != (w, h):
                rgb = rgb.resize((w, h), Image.Resampling.BILINEAR)
            images[idx] = np.asarray(rgb, dtype=np.uint8)
        class_name = rel_path.split("/")[0]
        labels[idx] = CLASS_TO_IDX[class_name]

    out = Path(cache_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, images=images, files=np.asarray(unique_files), labels=labels)
    return out


def load_uint8_cache(cache_path: str | Path) -> dict[str, Any]:
    """Load prebuilt .npz uint8 cache into a fast lookup dictionary."""
    with np.load(Path(cache_path), allow_pickle=False) as archive:
        images = archive["images"]
        files = [str(f) for f in archive["files"]]
        labels = archive["labels"]
    file_to_idx = {f: i for i, f in enumerate(files)}
    return {
        "images": images,
        "file_to_idx": file_to_idx,
        "labels": labels,
        "image_size": tuple(int(x) for x in images.shape[1:3]),
    }


class RealWasteDataset(Dataset):
    """PyTorch Dataset backed by canonical split JSON and optional uint8 memory cache."""

    def __init__(
        self,
        split: str,
        model_name: str,
        *,
        augment: bool = False,
        image_size: tuple[int, int] = INPUT_SIZE,
        data_dir: str | Path | None = None,
        split_path: str | Path = DEFAULT_SPLIT_PATH,
        cache_path: str | Path | None = None,
        preloaded_cache: dict[str, Any] | None = None,
        allow_test: bool = False,
    ) -> None:
        self.split = normalize_split_name(split)
        if self.split != "train" and augment:
            raise ValueError("Augmentation is strictly forbidden on validation or test splits.")
        self.records = load_split_records(self.split, split_path, allow_test=allow_test)
        self.model_name = model_name
        self.augment = augment
        self.image_size = tuple(image_size)
        self.transform = get_transforms(
            model_name=model_name,
            split="train" if self.split == "train" else "val",
            augment=augment,
            image_size=self.image_size,
        )

        self.cache = preloaded_cache
        if self.cache is None and cache_path is not None and Path(cache_path).is_file():
            self.cache = load_uint8_cache(cache_path)

        # Only use preloaded/loaded cache if its spatial resolution >= requested image_size (or matches)
        if self.cache is not None and self.cache.get("image_size") != self.image_size:
            if data_dir is not None:
                self.cache = None  # fall back to raw data_dir if cache resolution mismatches

        self.rw_root: Path | None = None
        if self.cache is None:
            if data_dir is None:
                raise ValueError("Either data_dir or a resolution-matched cache must be provided.")
            self.rw_root = resolve_realwaste_root(data_dir)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        rel_path, label = self.records[index]
        if self.cache is not None:
            cache_idx = self.cache["file_to_idx"][rel_path]
            pil_img = Image.fromarray(self.cache["images"][cache_idx])
        else:
            assert self.rw_root is not None
            with Image.open(self.rw_root / rel_path) as img:
                pil_img = img.convert("RGB")
        tensor = self.transform(pil_img)
        return tensor, label


def save_sanity_batch_grid(
    train_ds: RealWasteDataset,
    val_ds: RealWasteDataset,
    output_path: str | Path,
    *,
    n_samples: int = 6,
    seed: int = 42,
) -> Path:
    """Save a visual grid of preprocessed Train (augmented) vs Validation (clean) samples."""
    rng = np.random.default_rng(seed)
    tr_indices = rng.choice(len(train_ds), size=min(n_samples, len(train_ds)), replace=False)
    va_indices = rng.choice(len(val_ds), size=min(n_samples, len(val_ds)), replace=False)

    cols = max(len(tr_indices), len(va_indices))
    fig, axes = plt.subplots(2, cols, figsize=(2.4 * cols, 5.8), dpi=200)
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)

    torch.manual_seed(seed)
    for c, idx in enumerate(tr_indices):
        tensor, label = train_ds[int(idx)]
        img = unnormalize_for_display(tensor, train_ds.model_name)
        axes[0, c].imshow(img)
        axes[0, c].set_title(f"Train (Aug)\n{CLASS_NAMES[label]}", fontsize=8.5, fontweight="bold", pad=6)
        axes[0, c].axis("off")

    for c, idx in enumerate(va_indices):
        tensor, label = val_ds[int(idx)]
        img = unnormalize_for_display(tensor, val_ds.model_name)
        axes[1, c].imshow(img)
        axes[1, c].set_title(f"Val (Clean)\n{CLASS_NAMES[label]}", fontsize=8.5, pad=6)
        axes[1, c].axis("off")

    sz = train_ds.image_size
    fig.suptitle(
        f"Sanity Check Preprocessing ({sz[0]}x{sz[1]} RGB, Unnormalized Display) — Train vs Validation",
        fontweight="bold",
        fontsize=11,
        y=0.98,
    )
    fig.subplots_adjust(top=0.85, bottom=0.03, left=0.02, right=0.98, hspace=0.36, wspace=0.08)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build uint8 image cache for RealWaste.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--cache-out", type=Path, default=ROOT / "tmp/realwaste_224_uint8.npz")
    args = parser.parse_args()

    out = build_uint8_cache(
        args.data_dir,
        args.cache_out,
        args.split_path,
        image_size=(args.image_size, args.image_size),
    )
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"Built {args.image_size}x{args.image_size} uint8 cache at {out} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
