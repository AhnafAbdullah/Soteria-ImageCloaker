"""Per-image protection and quality metrics, shared by validation, PGD and eval.

Conventions (lower is better protection for every *_cos and *_ratio column):
- {vae}_ratio : ||E(x') - E(grey)||^2 / ||E(x) - E(grey)||^2, the training target
- {vae}_shift : ||E(x') - E(x)|| / ||E(x)||, untargeted latent displacement
- {clip}_cos  : cosine(emb(x'), emb(x))
- {face}_cos  : cosine(id(x'), id(x)), NaN for images without a face
Each is computed after a named post-processing condition, e.g. "jpeg75" is a
real libjpeg round trip at quality 75 applied to the cloaked image only.
"""
from __future__ import annotations

import math

import torch

from .augment import real_jpeg, resize_round_trip, upsample


def _jpeg(q: int):
    return lambda x: real_jpeg(x, [q] * x.shape[0])


def _resize(scale: float, q: int | None = None):
    """Downscale (antialiased), optionally JPEG, upscale back to the original size."""
    def f(x):
        small, size = resize_round_trip(x, scale)
        return upsample(_jpeg(q)(small) if q else small, size)
    return f


# Training validation uses only the first two; the generator evaluation all.
CONDITIONS = {
    "none": lambda x: x,
    "jpeg75": _jpeg(75),
    "jpeg90": _jpeg(90),
    "jpeg50": _jpeg(50),
    "resize50": _resize(0.5),                 # half size and back
    "platform": _resize(0.75, 80),            # messaging-app style: 0.75x, JPEG q80, back
}


@torch.no_grad()
def protection_metrics(surrogates: dict, x, x_cloak, has_face, landmarks, lpips_model,
                       conditions=("none", "jpeg75")) -> list[dict]:
    b = x.shape[0]
    rows = [dict() for _ in range(b)]

    mse = (x_cloak - x).pow(2).flatten(1).mean(1)
    psnr = 10 * torch.log10(1 / mse.clamp_min(1e-12))
    lp = lpips_model(x_cloak.float() * 2 - 1, x.float() * 2 - 1).flatten()
    linf = (x_cloak - x).abs().flatten(1).amax(1) * 255
    for i in range(b):
        rows[i].update(psnr=psnr[i].item(), lpips=lp[i].item(), linf_255=linf[i].item())

    for cond in conditions:
        xc = CONDITIONS[cond](x_cloak)
        for vae in surrogates.get("vae", []):
            with vae.clean():
                z, z_adv = vae(x.float()), vae(xc.float())
                grey = vae(torch.full_like(x[:1], 0.5).float())
            ratio = (z_adv - grey).pow(2).flatten(1).mean(1) / (z - grey).pow(2).flatten(1).mean(1).clamp_min(1e-6)
            shift = (z_adv - z).flatten(1).norm(dim=1) / z.flatten(1).norm(dim=1).clamp_min(1e-6)
            for i in range(b):
                rows[i][f"{cond}/{vae.key}_ratio"] = ratio[i].item()
                rows[i][f"{cond}/{vae.key}_shift"] = shift[i].item()
        for clip in surrogates.get("clip", []):
            with clip.clean():
                cos = torch.cosine_similarity(clip(xc), clip(x), dim=-1)
            for i in range(b):
                rows[i][f"{cond}/{clip.key}_cos"] = cos[i].item()
        for face in surrogates.get("face", []):
            idx = has_face.nonzero().squeeze(1)
            cos = torch.full((b,), math.nan, device=x.device)
            if len(idx):
                with face.clean():
                    cos[idx] = torch.cosine_similarity(face(xc[idx], landmarks=landmarks[idx]),
                                                       face(x[idx], landmarks=landmarks[idx]), dim=-1)
            for i in range(b):
                rows[i][f"{cond}/{face.key}_cos"] = cos[i].item()
    return rows
