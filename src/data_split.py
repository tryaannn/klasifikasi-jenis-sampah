"""Stratified Group-Aware Data Partitioning for RealWaste.

Guarantees 0% data leakage across Train/Validation/Test by ensuring that all
image pairs sharing near-duplicate conveyor backgrounds (dHash <= 6) reside in
the same subset while maintaining the 70/15/15 class distribution.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def build_connected_components(
    inventory_df: pd.DataFrame, near_duplicates_df: pd.DataFrame
) -> tuple[list[list[str]], dict[str, int]]:
    """Cluster images using Disjoint Set / Connected Components from dHash <= 6 pairs."""
    adj = defaultdict(set)
    for _, row in near_duplicates_df.iterrows():
        adj[row["a"]].add(row["b"])
        adj[row["b"]].add(row["a"])

    visited = set()
    groups: list[list[str]] = []
    file_to_group: dict[str, int] = {}

    for file in inventory_df["file"]:
        if file not in visited:
            comp: list[str] = []
            queue = [file]
            visited.add(file)
            while queue:
                curr = queue.pop()
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            gid = len(groups)
            groups.append(comp)
            for f in comp:
                file_to_group[f] = gid

    return groups, file_to_group


def partition_stratified_group(
    inventory_df: pd.DataFrame,
    groups: list[list[str]],
    ratios: dict[str, float] | None = None,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Greedy stratified group partition minimizing relative class imbalance error."""
    if ratios is None:
        ratios = {"train": 0.70, "validation": 0.15, "test": 0.15}

    file_to_class = dict(zip(inventory_df["file"], inventory_df["kelas"]))
    classes = sorted(inventory_df["kelas"].unique())
    class_totals = inventory_df["kelas"].value_counts().to_dict()

    group_records = []
    for gid, comp in enumerate(groups):
        counts = Counter(file_to_class[f] for f in comp)
        group_records.append({
            "gid": gid,
            "size": len(comp),
            "counts": counts,
            "files": comp,
        })

    targets = {
        split: {c: class_totals[c] * ratio for c in classes}
        for split, ratio in ratios.items()
    }

    # Deterministic sort: multi-image clusters first, descending by size, then gid
    np.random.seed(seed)
    group_records.sort(
        key=lambda g: (-g["size"], sorted(g["counts"].items(), key=lambda x: -x[1])[0][0], g["gid"])
    )

    split_names = list(ratios.keys())
    current_counts = {s: Counter() for s in split_names}
    split_files: dict[str, list[str]] = {s: [] for s in split_names}

    for g in group_records:
        best_split = None
        best_delta = float("inf")
        for s in split_names:
            delta = 0.0
            for c, cnt in g["counts"].items():
                curr = current_counts[s][c]
                tgt = targets[s][c]
                err_before = (curr - tgt) / tgt if tgt > 0 else 0
                err_after = (curr + cnt - tgt) / tgt if tgt > 0 else 0
                delta += (err_after ** 2 - err_before ** 2)
            if delta < best_delta:
                best_delta = delta
                best_split = s

        split_files[best_split].extend(g["files"])
        for c, cnt in g["counts"].items():
            current_counts[best_split][c] += cnt

    return split_files


def verify_split_integrity(
    split_dict: dict[str, list[str]],
    file_to_group: dict[str, int],
    total_expected: int = 4752,
) -> dict[str, Any]:
    """Verify zero overlap between splits and complete coverage."""
    all_files = []
    split_group_sets: dict[str, set[int]] = {}

    for split_name, files in split_dict.items():
        all_files.extend(files)
        split_group_sets[split_name] = {file_to_group[f] for f in files}

    assert len(all_files) == total_expected, f"Expected {total_expected}, got {len(all_files)}"
    assert len(set(all_files)) == total_expected, "Duplicate files detected across splits!"

    splits = list(split_dict.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = split_group_sets[splits[i]] & split_group_sets[splits[j]]
            assert len(overlap) == 0, f"Leakage detected between {splits[i]} and {splits[j]}: {overlap}"

    return {
        "status": "PASSED",
        "zero_leakage_verified": True,
        "total_images": total_expected,
        "group_overlaps": 0,
    }


def generate_split_plot(summary: dict[str, Any], output_path: Path) -> None:
    """Generate professional grouped bar chart of class distributions across splits."""
    classes = list(summary["classes"].keys())
    train_counts = [summary["classes"][c]["train"] for c in classes]
    val_counts = [summary["classes"][c]["validation"] for c in classes]
    test_counts = [summary["classes"][c]["test"] for c in classes]

    x = np.arange(len(classes))
    width = 0.26

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    rects1 = ax.bar(x - width, train_counts, width, label="Train (70%)", color="#2b5c8f")
    rects2 = ax.bar(x, val_counts, width, label="Validation (15%)", color="#d97724")
    rects3 = ax.bar(x + width, test_counts, width, label="Test (15%)", color="#2a9d8f")

    ax.set_ylabel("Jumlah Citra", fontsize=11, fontweight="bold")
    ax.set_title("Distribusi Kelas Hasil Stratified Group-Aware Partitioning (RealWaste)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=25, ha="right", fontsize=9.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate totals above bars
    for rects in (rects1, rects2, rects3):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def run_pipeline(
    inv_csv: Path = ROOT / "results/inventory.csv",
    dups_csv: Path = ROOT / "results/near_duplicates.csv",
    out_dir: Path = ROOT / "results",
    fig_dir: Path = ROOT / "figures",
) -> dict[str, Any]:
    inv = pd.read_csv(inv_csv)
    dups = pd.read_csv(dups_csv)

    groups, file_to_group = build_connected_components(inv, dups)
    splits = partition_stratified_group(inv, groups, seed=42)
    integrity = verify_split_integrity(splits, file_to_group, len(inv))

    # Calculate detailed class distribution per split
    file_to_class = dict(zip(inv["file"], inv["kelas"]))
    classes = sorted(inv["kelas"].unique())
    class_totals = inv["kelas"].value_counts().to_dict()

    class_summary = {}
    for c in classes:
        train_c = sum(1 for f in splits["train"] if file_to_class[f] == c)
        val_c = sum(1 for f in splits["validation"] if file_to_class[f] == c)
        test_c = sum(1 for f in splits["test"] if file_to_class[f] == c)
        tot = class_totals[c]
        class_summary[c] = {
            "total": tot,
            "train": train_c, "train_pct": round(train_c / tot * 100, 2),
            "validation": val_c, "validation_pct": round(val_c / tot * 100, 2),
            "test": test_c, "test_pct": round(test_c / tot * 100, 2),
        }

    summary = {
        "metadata": {
            "dataset": "RealWaste",
            "total_images": len(inv),
            "total_groups": len(groups),
            "multi_image_clusters": sum(1 for g in groups if len(g) > 1),
            "clustered_images_count": sum(len(g) for g in groups if len(g) > 1),
            "seed": 42,
            "ratios_target": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "integrity": integrity,
        },
        "splits": {
            "train": {"count": len(splits["train"]), "percentage": round(len(splits["train"]) / len(inv) * 100, 2)},
            "validation": {"count": len(splits["validation"]), "percentage": round(len(splits["validation"]) / len(inv) * 100, 2)},
            "test": {"count": len(splits["test"]), "percentage": round(len(splits["test"]) / len(inv) * 100, 2)},
        },
        "classes": class_summary,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data_split_indices.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    (out_dir / "data_split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Generate figure
    generate_split_plot(summary, fig_dir / "data_split_distribution.png")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inv", type=Path, default=ROOT / "results/inventory.csv")
    parser.add_argument("--dups", type=Path, default=ROOT / "results/near_duplicates.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--fig", type=Path, default=ROOT / "figures")
    args = parser.parse_args()

    summary = run_pipeline(args.inv, args.dups, args.out, args.fig)
    print("=== STRATIFIED GROUP-AWARE SPLIT COMPLETED ===")
    print(f"Total: {summary['metadata']['total_images']} images in {summary['metadata']['total_groups']} groups.")
    print(f"Clusters > 1 image: {summary['metadata']['multi_image_clusters']} clusters ({summary['metadata']['clustered_images_count']} images).")
    for s, info in summary["splits"].items():
        print(f"  {s.upper()}: {info['count']} images ({info['percentage']}%)")
    print("Integrity:", summary["metadata"]["integrity"]["status"])
    print(f"Artefak tersimpan di {args.out / 'data_split_indices.json'}")
    print(f"Grafik tersimpan di {args.fig / 'data_split_distribution.png'}")


if __name__ == "__main__":
    main()
