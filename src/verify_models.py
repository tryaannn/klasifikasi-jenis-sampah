"""CPU-only architecture evidence. No dataset, weight download or optimizer steps."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

try:
    from .models import CNNBaseline, MobileNetV3Small, load_config
except ImportError:  # python src/verify_models.py
    from models import CNNBaseline, MobileNetV3Small, load_config

ROOT = Path(__file__).resolve().parents[1]


def build_initial_components(model: nn.Module, name: str) -> tuple[nn.Module, torch.optim.Optimizer]:
    """Instantiate proposed loss/optimizer only; never perform an update."""
    config = load_config()["models"][name]
    optimizer_class = getattr(torch.optim, config["optimizer"])
    optimizer = optimizer_class(
        (p for p in model.parameters() if p.requires_grad),
        lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    return nn.CrossEntropyLoss(), optimizer


def verify_models(output_dir: str | Path | None = ROOT / "results/model_design") -> dict:
    config = load_config()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config["seed"])
        images = torch.rand(2, 3, 128, 128, device="cpu")
        models = {"cnn_baseline": CNNBaseline(dropout=config["models"]["cnn_baseline"]["dropout"]),
                  "mobilenet_v3_small": MobileNetV3Small(pretrained=False, seed=config["seed"])}
        summaries = {}
        for name, model in models.items():
            model.eval()
            with torch.inference_mode():
                logits = model(images)
                repeat = model(images)
            assert list(logits.shape) == [2, 9]
            assert torch.isfinite(logits).all()
            assert torch.equal(logits, repeat)
            frozen = None
            if isinstance(model, MobileNetV3Small):
                frozen = all(not p.requires_grad for p in model.network.features.parameters())
                assert frozen
                assert all(p.requires_grad for p in model.network.classifier.parameters())
            summaries[name] = {
                "input_shape": list(images.shape), "output_shape": list(logits.shape),
                "finite_outputs": bool(torch.isfinite(logits).all()),
                "deterministic_eval": bool(torch.equal(logits, repeat)),
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "backbone_frozen": frozen, "pretrained_weights_loaded": False,
                "architecture": str(model),
            }
    report = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cpu", "python": platform.python_version(), "seed": config["seed"],
        "versions": {n: importlib.metadata.version(n) for n in
                     ("torch", "torchvision", "numpy", "Pillow", "PyYAML")},
        "training_performed": False, "dataset_used": False,
        "models": summaries,
    }
    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        for name, filename in [("cnn_baseline", "baseline_architecture_summary.json"),
                               ("mobilenet_v3_small", "mobilenet_architecture_summary.json")]:
            (target / filename).write_text(json.dumps(summaries[name], indent=2), encoding="utf-8")
        (target / "model_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/model_design")
    report = verify_models(parser.parse_args().out)
    for name, summary in report["models"].items():
        print(f"{name}: {summary['input_shape']} -> {summary['output_shape']}; "
              f"finite={summary['finite_outputs']}; parameters={summary['total_parameters']:,}; "
              f"trainable={summary['trainable_parameters']:,}; frozen={summary['backbone_frozen']}")
    print("CPU verification only; no pretrained weights, training or dataset evaluation.")


if __name__ == "__main__":
    main()
