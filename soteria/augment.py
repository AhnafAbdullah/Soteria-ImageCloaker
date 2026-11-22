"""EOT augmentation T: what happens to a photo between cloaking and attack.

Every op is differentiable or has a straight-through gradient, so the generator
learns cloaks that survive the transform instead of cloaks that only work on
the exact pixels it emitted.

JPEG is the transform that matters most, and it runs two ways:
- real libjpeg through PIL in the forward pass with an identity gradient (BPDA).
  This is the codec platforms actually use, so the generator cannot exploit
  an approximation's quirks.
- kornia's differentiable JPEG, which gives a true (approximate) gradient.
Mixing both gets the faithful forward of one and the informative backward of
the other.

The "platform" branch simulates a messaging-app round trip: downscale, JPEG,
then upscale back so the surrogates see the original geometry. Its ranges are
placeholders until the WhatsApp/Instagram calibration replaces them.
"""
from __future__ import annotations

import io
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from kornia.enhance import jpeg_codec_differentiable
from kornia.filters import gaussian_blur2d
from PIL import Image


@dataclass
class Crop:
    """Crop box as fractions of the image; the crop is resized back to full size."""
    x0: float = 0.0
    y0: float = 0.0
    w: float = 1.0
    h: float = 1.0

    def apply_to_landmarks(self, lm: torch.Tensor, width: int, height: int) -> torch.Tensor:
        """lm: (B, 5, 2) in pixels of the uncropped image."""
        offset = lm.new_tensor([self.x0 * width, self.y0 * height])
        scale = lm.new_tensor([1 / self.w, 1 / self.h])
        return (lm - offset) * scale


def real_jpeg(x: torch.Tensor, quality: list[int]) -> torch.Tensor:
    """libjpeg round trip (4:2:0) with a straight-through gradient."""
    with torch.no_grad():
        out = torch.empty_like(x)
        arr = (x.detach().float().clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        for i, (img, q) in enumerate(zip(arr, quality)):
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=int(q), subsampling=2)
            buf.seek(0)
            dec = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255
            out[i] = torch.from_numpy(dec).permute(2, 0, 1).to(x.device, x.dtype)
    return x + (out - x).detach()


def diff_jpeg(x: torch.Tensor, quality: list[int]) -> torch.Tensor:
    """kornia's differentiable JPEG. H and W must be multiples of 16."""
    q = torch.tensor(quality, device=x.device, dtype=torch.float32)
    return jpeg_codec_differentiable(x.float(), q).to(x.dtype)


def jpeg(x: torch.Tensor, quality: list[int], p_real: float) -> torch.Tensor:
    h, w = x.shape[-2:]
    if random.random() < p_real or h % 16 or w % 16:
        return real_jpeg(x, quality)
    return diff_jpeg(x, quality)


def resize_round_trip(x: torch.Tensor, scale: float, multiple: int = 1) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = x.shape[-2:]
    sh = max(multiple, int(round(h * scale / multiple)) * multiple)
    sw = max(multiple, int(round(w * scale / multiple)) * multiple)
    small = F.interpolate(x, size=(sh, sw), mode="bilinear", align_corners=False, antialias=True)
    return small, (h, w)


def upsample(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class EOT:
    """Random transform T, scaled by a curriculum strength in [0, 1]."""

    def __init__(self, cfg):
        self.cfg = cfg

    def _lerp_range(self, lo: float, hi: float, mild: float, s: float) -> tuple[float, float]:
        # At s = 0 the range collapses to the mild end; at s = 1 it is [lo, hi].
        return mild + s * (lo - mild), mild + s * (hi - mild)

    def __call__(self, x: torch.Tensor, strength: float = 1.0) -> tuple[torch.Tensor, Crop]:
        cfg, s = self.cfg, float(min(max(strength, 0.0), 1.0))
        b, crop = x.shape[0], Crop()
        if s == 0 or random.random() < cfg.p_identity:
            return x, crop

        if random.random() < cfg.p_platform * s:
            lo, hi = self._lerp_range(*cfg.platform_scale, 1.0, s)
            qlo, qhi = self._lerp_range(*cfg.platform_q, 95, s)
            small, size = resize_round_trip(x, random.uniform(lo, hi), multiple=16)
            q = [random.randint(int(qlo), int(qhi)) for _ in range(b)]
            return upsample(jpeg(small, q, cfg.p_real_jpeg), size), crop

        h, w = x.shape[-2:]
        if random.random() < 0.3 * s:
            frac = random.uniform(1 - 0.2 * s, 1.0)
            ch, cw = max(1, int(frac * h)), max(1, int(frac * w))
            top, left = random.randint(0, h - ch), random.randint(0, w - cw)
            crop = Crop(left / w, top / h, cw / w, ch / h)
            x = upsample(x[..., top:top + ch, left:left + cw], (h, w))
        if random.random() < 0.3 * s:
            sigma = random.uniform(0.3, 0.3 + 0.9 * s)
            x = gaussian_blur2d(x, (5, 5), (sigma, sigma), border_type="reflect")
        if random.random() < 0.4 * s:
            small, size = resize_round_trip(x, random.uniform(1 - 0.5 * s, 1.0))
            x = upsample(small, size)
        if random.random() < 0.8 * s:
            qlo, qhi = self._lerp_range(*cfg.jpeg_q, 95, s)
            x = jpeg(x, [random.randint(int(qlo), int(qhi)) for _ in range(b)], cfg.p_real_jpeg)
        return x, crop
