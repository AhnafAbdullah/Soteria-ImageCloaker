"""Cloak a single photo with a trained generator.

    python -m soteria.infer runs/base/generator_ema.pt photo.jpg out.png --max-side 1600

--max-side resizes before cloaking. Set it to the target platform's upload
size so the platform re-encodes the photo but does not downscale it:
downscaling averages neighbouring pixels and with them most of the cloak.
The output is a PNG with no metadata (EXIF, GPS) carried over.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from .generator import CloakGenerator
from .mask import texture_mask


def load_generator(path: str, device="cpu", mask_min: float = 0.3) -> CloakGenerator:
    ck = torch.load(path, map_location=device, weights_only=False)
    G = CloakGenerator(ck["eps"], pretrained=False)
    G.load_state_dict(ck["state_dict"])
    G.mask_min = mask_min
    return G.to(device).eval()


@torch.no_grad()
def cloak_image(G: CloakGenerator, img: Image.Image, max_side: int | None = None) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    device = next(G.parameters()).device
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255).permute(2, 0, 1)[None].to(device)
    h, w = x.shape[-2:]
    ph, pw = (-h) % 32, (-w) % 32
    xp = F.pad(x, (0, pw, 0, ph), mode="reflect") if ph or pw else x
    x_cloak, _ = G(xp, texture_mask(xp, getattr(G, "mask_min", 0.3)))
    x_cloak = x_cloak[..., :h, :w]
    arr = (x_cloak[0].permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--max-side", type=int, default=None)
    args = ap.parse_args(argv)
    G = load_generator(args.checkpoint, "cuda" if torch.cuda.is_available() else "cpu")
    cloak_image(G, Image.open(args.src), args.max_side).save(args.dst)


if __name__ == "__main__":
    main()
