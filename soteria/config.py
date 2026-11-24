"""Run configuration.

One flat dataclass, loaded from a YAML file and overridable from the command
line as ``key=value`` pairs, so a Colab cell can change a single knob without
editing the repo:

    python -m soteria.train --config configs/base_256.yaml steps=500 eps=0.0627
"""
from __future__ import annotations

import ast
import os
import random
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch
import yaml

SEED = 42


@dataclass
class Config:
    seed: int = SEED
    image_size: int = 256

    # Cloak budget. eps is an L-inf bound in [0, 1] pixel units; the texture
    # mask scales it down to eps * mask_min in flat regions such as skin.
    eps: float = 8 / 255
    mask_min: float = 0.3

    # Surrogates. Each family lists one or more registry keys from
    # soteria/surrogates.py; one per family is sampled each step, so adding a
    # key turns on the stochastic ensemble without changing step cost.
    vae: list[str] = field(default_factory=lambda: ["sd15"])
    clip: list[str] = field(default_factory=lambda: ["vit_b16_laion"])
    face: list[str] = field(default_factory=lambda: ["arcface_ir101"])
    ghost_std: float = 0.03          # multiplicative activation noise in surrogates
    vae_fp32: bool = True            # run the VAE encoder outside autocast

    # Loss weights. LPIPS is a hinge: free below the budget, penalised above it.
    w_vae: float = 1.0
    w_clip: float = 1.0
    w_face: float = 1.0
    # Cosine below which the CLIP / face term stops pulling (-1 = never).
    # Reference levels from scripts/check_surrogates.py on FFHQ eval faces:
    # other face photos 0.425 (CLIP), other people -0.002 (ArcFace).
    clip_floor: float = -1.0
    face_floor: float = -1.0
    w_lpips: float = 10.0
    lpips_budget: float = 0.05

    # EOT augmentation (see soteria/augment.py)
    jpeg_q: list[int] = field(default_factory=lambda: [50, 90])
    p_real_jpeg: float = 0.5         # else kornia's differentiable JPEG
    p_platform: float = 0.3          # downscale + JPEG + upscale, a messaging-app round trip
    platform_scale: list[float] = field(default_factory=lambda: [0.6, 1.0])
    platform_q: list[int] = field(default_factory=lambda: [60, 85])
    p_identity: float = 0.1
    curriculum_frac: float = 0.2     # EOT strength ramps 0 -> 1 over this fraction of steps

    # Optimisation
    batch_size: int = 16
    steps: int = 20_000
    lr: float = 2e-4
    encoder_lr_mult: float = 0.5
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    amp: bool = True

    # Bookkeeping
    log_every: int = 50
    val_every: int = 500
    ckpt_every: int = 500
    budget_min: float = 0.0          # wall-clock cap in minutes, 0 = none
    num_workers: int = 2

    # PGD baseline
    pgd_steps: int = 100
    pgd_alpha_frac: float = 0.05     # step size as a fraction of eps
    pgd_images: int = 200
    pgd_batch: int = 8

    data_root: str = "data/prepared"
    out_dir: str = "runs/default"

    @classmethod
    def load(cls, path: str | os.PathLike | None = None, overrides: list[str] | None = None) -> "Config":
        values = {}
        if path:
            values = yaml.safe_load(Path(path).read_text()) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise KeyError(f"unknown config keys in {path}: {sorted(unknown)}")
        cfg = cls(**values)
        for item in overrides or []:
            key, _, raw = item.partition("=")
            if key not in known:
                raise KeyError(f"unknown override key: {key}")
            setattr(cfg, key, _parse(raw))
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)


def _parse(raw: str):
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
