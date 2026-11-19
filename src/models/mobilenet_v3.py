"""Official torchvision MobileNetV3-Small with an explicit offline default."""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from .config import NUM_CLASSES


class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, *, pretrained: bool = False,
                 freeze_backbone: bool = True, seed: int = 42) -> None:
        super().__init__()
        # Local RNG scope reproduces initialization without resetting the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            self.network = mobilenet_v3_small(weights=weights)
            in_features = self.network.classifier[-1].in_features
            self.network.classifier[-1] = nn.Linear(in_features, num_classes)
        self.pretrained_loaded = pretrained  # construction raises if loading fails
        self.initialization_seed = seed
        self.set_trainable_feature_blocks(0 if freeze_backbone else len(self.network.features))

    def set_trainable_feature_blocks(self, last_n: int = 0) -> None:
        """Unfreeze only the last N feature blocks for future fine-tuning.

        Frozen blocks remain in eval mode, including BatchNorm running statistics.
        Rebuild the optimizer after changing this policy in future experiments.
        """
        blocks = list(self.network.features)
        if not 0 <= last_n <= len(blocks):
            raise ValueError(f"last_n must be between 0 and {len(blocks)}")
        self.trainable_feature_blocks = last_n
        for i, block in enumerate(blocks):
            block.requires_grad_(i >= len(blocks) - last_n)
        self.train(self.training)

    def train(self, mode: bool = True) -> MobileNetV3Small:
        super().train(mode)
        for block in self.network.features:
            if not any(p.requires_grad for p in block.parameters()):
                block.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)
