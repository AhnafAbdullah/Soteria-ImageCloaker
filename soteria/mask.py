"""Texture mask M: where the cloak is allowed to be loud.

People notice noise on flat regions (skin, sky, walls) far more than on hair,
fabric or foliage. M is a per-pixel scale in [mask_min, 1] built from local
luminance contrast, so the effective budget is eps * M.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d


@torch.no_grad()
def texture_mask(x: torch.Tensor, m_min: float = 0.3, window: int = 7, q: float = 0.95) -> torch.Tensor:
    """x: (B, 3, H, W) in [0, 1] -> (B, 1, H, W) in [m_min, 1]."""
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    pad = window // 2
    padded = F.pad(gray, (pad, pad, pad, pad), mode="reflect")
    mean = F.avg_pool2d(padded, window, stride=1)
    mean_sq = F.avg_pool2d(padded * padded, window, stride=1)
    std = (mean_sq - mean * mean).clamp_min(0).add(1e-8).sqrt()

    # Normalise per image by a high quantile rather than the max, so one
    # specular highlight cannot flatten the rest of the mask.
    flat = std.flatten(1)
    scale = torch.quantile(flat, q, dim=1).clamp_min(1e-4).view(-1, 1, 1, 1)
    m = (std / scale).clamp(0, 1)
    m = gaussian_blur2d(m, (window, window), (window / 3, window / 3), border_type="reflect")
    return m_min + (1 - m_min) * m
