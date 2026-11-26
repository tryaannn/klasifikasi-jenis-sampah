"""Dataset loading and uint8 caching utilities for RealWaste (anti-leakage enforced).

Reads the canonical group-aware split from results/data_split_indices.json and
prevents accidental access to the held-out test set during Progress 4 training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .models.config import CLASS_NAMES, CLASS_TO_IDX, INPUT_SIZE, ROOT
from .transforms import get_transforms

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

    # Search nested directories (common on Kaggle / unzipped archives)
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
    # Preserve order while deduplicating
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
    return {"images": images, "file_to_idx": file_to_idx, "labels": labels}


class RealWasteDataset(Dataset):
    """PyTorch Dataset backed by canonical split JSON and optional uint8 memory cache."""

    def __init__(
        self,
        split: str,
        model_name: str,
        *,
        augment: bool = False,
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
        self.transform = get_transforms(
            model_name=model_name,
            split="train" if self.split == "train" else "val",
            augment=augment,
        )

        self.cache = preloaded_cache
        if self.cache is None and cache_path is not None and Path(cache_path).is_file():
            self.cache = load_uint8_cache(cache_path)

        self.rw_root: Path | None = None
        if self.cache is None:
            if data_dir is None:
                raise ValueError("Either data_dir or a valid cache_path/preloaded_cache must be provided.")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 128x128 uint8 image cache for RealWaste.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--cache-out", type=Path, default=ROOT / "tmp/realwaste_128_uint8.npz")
    args = parser.parse_args()

    out = build_uint8_cache(args.data_dir, args.cache_out, args.split_path)
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"Built uint8 cache at {out} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
