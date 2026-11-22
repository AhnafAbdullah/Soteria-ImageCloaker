"""Cloak generator G: a U-Net with an ImageNet-pretrained EfficientNet-B0 encoder.

The cloak is delta = eps * tanh(G(x)) * M, so the L-inf bound eps * M holds by
construction and no projection step is needed.
"""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CloakGenerator(nn.Module):
    def __init__(self, eps: float, encoder: str = "efficientnet-b0", pretrained: bool = True):
        super().__init__()
        self.eps = eps
        self.net = smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None,
                            in_channels=3, classes=3, activation=None)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

        # Start near delta = 0 so the first steps are not spent undoing a loud
        # random cloak that the LPIPS hinge would punish.
        head = self.net.segmentation_head[0]
        nn.init.normal_(head.weight, std=1e-3)
        nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x in [0, 1], H and W divisible by 32. Returns (cloaked, delta)."""
        raw = self.net((x - self.mean) / self.std)
        delta = self.eps * torch.tanh(raw.float()) * mask
        cloaked = (x + delta).clamp(0, 1)
        return cloaked, cloaked - x

    def encoder_parameters(self):
        return self.net.encoder.parameters()

    def decoder_parameters(self):
        enc = {id(p) for p in self.net.encoder.parameters()}
        return [p for p in self.parameters() if id(p) not in enc]
