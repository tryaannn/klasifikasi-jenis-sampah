"""Unit tests for the data engineering, anti-leakage split, and hardware profiling pipeline.

Supports standard python -m unittest without requiring external pytest installation.
"""
from __future__ import annotations

import json
import unittest
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.class_weights import compute_training_class_weights
from src.data_split import (
    build_connected_components,
    partition_stratified_group,
    verify_split_integrity,
)
from src.profile_models import (
    analyze_hardware_feasibility,
    compute_cnn_baseline_complexity,
    compute_mobilenet_v3_complexity,
)


class TestDataPipelineAndProfiling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inv = pd.read_csv(ROOT / "results/inventory.csv")
        cls.dups = pd.read_csv(ROOT / "results/near_duplicates.csv")

    def test_connected_components_clustering(self):
        groups, file_to_group = build_connected_components(self.inv, self.dups)

        # 4752 images must be partitioned into exactly 4586 groups
        self.assertEqual(len(groups), 4586)
        self.assertEqual(len(file_to_group), 4752)

        # Exactly 50 clusters with size > 1 containing 216 images total
        multi_clusters = [g for g in groups if len(g) > 1]
        self.assertEqual(len(multi_clusters), 50)
        self.assertEqual(sum(len(g) for g in multi_clusters), 216)

        # The largest cluster caused by conveyor background must have exactly 84 images
        largest_cluster = max(len(g) for g in groups)
        self.assertEqual(largest_cluster, 84)

    def test_zero_leakage_group_aware_partition(self):
        groups, file_to_group = build_connected_components(self.inv, self.dups)
        splits = partition_stratified_group(self.inv, groups, seed=42)

        # Total and unique image counts
        self.assertEqual(len(splits["train"]), 3327)
        self.assertEqual(len(splits["validation"]), 713)
        self.assertEqual(len(splits["test"]), 712)
        self.assertEqual(len(splits["train"]) + len(splits["validation"]) + len(splits["test"]), 4752)

        # Verify zero leakage (no cluster shared across splits)
        train_groups = {file_to_group[f] for f in splits["train"]}
        val_groups = {file_to_group[f] for f in splits["validation"]}
        test_groups = {file_to_group[f] for f in splits["test"]}

        self.assertEqual(len(train_groups & val_groups), 0, "Train and Validation share near-duplicate images!")
        self.assertEqual(len(train_groups & test_groups), 0, "Train and Test share near-duplicate images!")
        self.assertEqual(len(val_groups & test_groups), 0, "Validation and Test share near-duplicate images!")

        # Run formal integrity verifier
        integrity = verify_split_integrity(splits, file_to_group, 4752)
        self.assertEqual(integrity["status"], "PASSED")
        self.assertTrue(integrity["zero_leakage_verified"])

    def test_class_weights_calculation(self):
        summary_path = ROOT / "results/data_split_summary.json"
        weights = compute_training_class_weights(summary_path)

        self.assertEqual(weights["metadata"]["source_split"], "train_only")
        self.assertEqual(weights["metadata"]["total_training_samples"], 3327)
        self.assertEqual(weights["metadata"]["num_classes"], 9)

        norm_w = weights["inverse_class_frequency_weights"]["normalized_mean_1"]

        # Mean normalized weight must equal 1.0 (within rounding tolerance)
        mean_val = sum(norm_w.values()) / 9
        self.assertAlmostEqual(mean_val, 1.0, delta=1e-3)

        # Minority class (Textile Trash) must have the highest weight
        self.assertEqual(norm_w["Textile Trash"], max(norm_w.values()))

        # Majority class (Plastic) must have the lowest weight
        self.assertEqual(norm_w["Plastic"], min(norm_w.values()))

    def test_hardware_profiling_feasibility(self):
        cnn_prof = compute_cnn_baseline_complexity()
        mob_prof = compute_mobilenet_v3_complexity()
        feas = analyze_hardware_feasibility(8.0)

        # CNN Baseline parameter and MAC contracts
        self.assertEqual(cnn_prof["total_parameters"], 102089)
        self.assertGreater(cnn_prof["macs_millions"], 150.0)

        # MobileNetV3-Small parameter and MAC contracts
        self.assertEqual(mob_prof["total_parameters"], 1527081)
        self.assertEqual(mob_prof["trainable_parameters"], 600073)
        self.assertLess(mob_prof["macs_millions"], 25.0)  # Depthwise conv efficiency

        # RAM feasibility: training footprint must not exceed 10% of 8 GB RAM
        peak = feas["peak_runtime_memory"]
        self.assertLess(peak["percentage_of_8gb_ram_cnn"], 10.0)
        self.assertLess(peak["percentage_of_8gb_ram_mob"], 10.0)


if __name__ == "__main__":
    unittest.main()
