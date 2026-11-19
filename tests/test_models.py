"""Offline synthetic verification; no training, dataset or pretrained downloads."""
import json

import pytest
import torch
from torch import nn

from src.models import CNNBaseline, MobileNetV3Small, CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS, load_config
from src.models.config import IMAGENET_MEAN, IMAGENET_STD
from src.verify_models import build_initial_components, verify_models


@pytest.fixture(scope="module")
def models():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    return [CNNBaseline().eval(), MobileNetV3Small(pretrained=False).eval()]


@pytest.mark.parametrize("index", [0, 1])
def test_forward_contract(models, index):
    model = models[index]
    images = torch.rand(2, 3, 128, 128)
    with torch.inference_mode():
        logits = model(images)
        again = model(images)
        probabilities = logits.softmax(dim=1)
        loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 8], dtype=torch.long))
    assert logits.shape == (2, 9)
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    assert torch.equal(logits, again)
    assert probabilities.shape == (2, 9)
    assert (probabilities >= 0).all() and (probabilities <= 1).all()
    assert torch.allclose(probabilities.sum(1), torch.ones(2))
    assert not any(isinstance(layer, nn.Softmax) for layer in model.modules())


def test_class_mapping_matches_historical_eda():
    import csv
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    with (root / "results/class_distribution.csv").open(encoding="utf-8") as stream:
        original = {row["kelas"] for row in csv.DictReader(stream)}
    expected = ("Cardboard", "Food Organics", "Glass", "Metal", "Miscellaneous Trash",
                "Paper", "Plastic", "Textile Trash", "Vegetation")
    assert CLASS_NAMES == expected and set(CLASS_NAMES) == original
    assert list(IDX_TO_CLASS) == list(range(9))
    for i, label in enumerate(expected):
        assert CLASS_TO_IDX[label] == i and IDX_TO_CLASS[i] == label


def test_architecture_and_counts(models):
    baseline, mobile = models
    convs = [m for m in baseline.modules() if isinstance(m, nn.Conv2d)]
    assert [(m.in_channels, m.out_channels) for m in convs] == [(3, 32), (32, 64), (64, 128)]
    assert baseline.classifier[-1].out_features == 9
    assert mobile.network.classifier[-1].out_features == 9
    assert sum(p.numel() for p in baseline.parameters()) == 102089
    assert sum(p.numel() for p in mobile.parameters()) == 1527081
    assert sum(p.numel() for p in mobile.parameters() if p.requires_grad) == 600073


def test_freezing_includes_batchnorm_and_selected_finetuning():
    model = MobileNetV3Small(pretrained=False)
    assert not model.pretrained_loaded
    model.train()
    assert all(not p.requires_grad for p in model.network.features.parameters())
    assert all(p.requires_grad for p in model.network.classifier.parameters())
    assert all(not block.training for block in model.network.features)
    before = {k: v.clone() for k, v in model.network.features.named_buffers()}
    with torch.inference_mode():
        model(torch.rand(2, 3, 128, 128))
    assert all(torch.equal(before[k], v) for k, v in model.network.features.named_buffers())
    model.set_trainable_feature_blocks(2)
    assert all(not p.requires_grad for b in model.network.features[:-2] for p in b.parameters())
    assert all(p.requires_grad for b in model.network.features[-2:] for p in b.parameters())
    assert all(b.training for b in model.network.features[-2:])
    model.set_trainable_feature_blocks(len(model.network.features))
    assert all(p.requires_grad for p in model.network.features.parameters())
    model.set_trainable_feature_blocks(0)
    assert all(not p.requires_grad for p in model.network.features.parameters())
    with pytest.raises(ValueError):
        model.set_trainable_feature_blocks(-1)
    with pytest.raises(ValueError):
        model.set_trainable_feature_blocks(100)


def test_seeded_initialization():
    a, b = MobileNetV3Small(seed=42), MobileNetV3Small(seed=42)
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


def test_configuration_and_optimizer_contract(models):
    config = load_config()
    assert config["seed"] == 42
    assert not config["controlled_experiments_progress4"]["augmentation"]
    assert not config["controlled_experiments_progress4"]["class_weighting"]
    mobile = config["models"]["mobilenet_v3_small"]
    assert tuple(mobile["normalization"]["mean"]) == IMAGENET_MEAN
    assert tuple(mobile["normalization"]["std"]) == IMAGENET_STD
    assert mobile["pretrained_weights_verification"] is None and mobile["freeze_backbone"]
    for model, name in zip(models, ("cnn_baseline", "mobilenet_v3_small")):
        proposal = config["models"][name]
        assert proposal["num_classes"] == 9 and proposal["input_size"] == [128, 128]
        assert proposal["max_epochs_planned"] == 30
        loss, optimizer = build_initial_components(model, name)
        assert isinstance(loss, nn.CrossEntropyLoss) and loss.weight is None
        assert type(optimizer).__name__ == proposal["optimizer"]
        assert optimizer.param_groups[0]["lr"] == proposal["learning_rate"]
        actual = {id(p) for group in optimizer.param_groups for p in group["params"]}
        assert actual == {id(p) for p in model.parameters() if p.requires_grad}


def test_verification_artifacts(tmp_path):
    report = verify_models(tmp_path)
    saved = json.loads((tmp_path / "model_verification.json").read_text())
    assert saved == report and not saved["training_performed"]
    for name, summary in report["models"].items():
        assert summary["output_shape"] == [2, 9] and summary["finite_outputs"]
        assert 0 < summary["trainable_parameters"] <= summary["total_parameters"]
    assert (tmp_path / "baseline_architecture_summary.json").is_file()
    assert (tmp_path / "mobilenet_architecture_summary.json").is_file()
