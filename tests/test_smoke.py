"""CPU smoke tests. No downloads: real surrogates and LPIPS are swapped for tiny stand-ins.

    pytest -q
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from soteria import pgd, train
from soteria.augment import EOT, Crop, real_jpeg
from soteria.config import Config
from soteria.data import LM_COLS
from soteria.generator import CloakGenerator
from soteria.losses import CloakLoss
from soteria.mask import texture_mask
from soteria.surrogates import ARCFACE_TEMPLATE, FaceSurrogate, Surrogate, align_faces


# ----------------------------------------------------------------- stand-ins

class TinyVAE(Surrogate):
    family = "vae"

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 8, stride=8)

    def forward(self, x, **_):
        return self.conv(x * 2 - 1)


class TinyCLIP(Surrogate):
    family = "clip"

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 4, stride=4)
        self.fc = nn.Linear(16, 32)

    def forward(self, x, **_):
        x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
        return F.normalize(self.fc(self.conv(x).mean((2, 3))), dim=-1)


class TinyFaceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 8, stride=8)
        self.fc = nn.Linear(16 * 14 * 14, 64)

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


def tiny_surrogates(cfg, device):
    out = {"vae": [TinyVAE()], "clip": [TinyCLIP()], "face": [FaceSurrogate(TinyFaceNet())]}
    for fam, models in out.items():
        for m in models:
            m.freeze(cfg.ghost_std).to(device)
            m.key = f"tiny_{fam}"
    return out


class TinyLPIPS(nn.Module):
    def forward(self, a, b):
        return (a - b).pow(2).flatten(1).mean(1).view(-1, 1, 1, 1)


@pytest.fixture
def cfg():
    return Config(image_size=64, batch_size=2, num_workers=0, steps=3, warmup_steps=1,
                  log_every=1, val_every=2, ckpt_every=2, pgd_steps=2, pgd_batch=2, pgd_images=2)


@pytest.fixture
def dataset(tmp_path):
    """Four 64 px images, two with plausible face landmarks; one per split slot."""
    root = tmp_path / "data"
    (root / "ffhq").mkdir(parents=True)
    rows, rng = [], np.random.default_rng(0)
    lm = (ARCFACE_TEMPLATE.numpy() * (40 / 112) + 12).reshape(-1)
    for i in range(6):
        Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(root / "ffhq" / f"{i}.png")
        face = i % 2 == 0
        rows.append({"path": f"ffhq/{i}.png", "source": "ffhq", "split": "eval" if i >= 4 else "train",
                     "has_face": face, **dict(zip(LM_COLS, lm if face else [np.nan] * 10))})
    pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)
    return root


# --------------------------------------------------------------------- units

def test_config_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("eps: 0.01\nvae: [sd15]\n")
    cfg = Config.load(p, ["steps=7", "face=['a','b']"])
    assert cfg.eps == 0.01 and cfg.steps == 7 and cfg.face == ["a", "b"]
    with pytest.raises(KeyError):
        Config.load(p, ["nonsense=1"])


def test_mask_range():
    m = texture_mask(torch.rand(2, 3, 64, 64), m_min=0.3)
    assert m.shape == (2, 1, 64, 64)
    assert m.min() >= 0.3 - 1e-6 and m.max() <= 1 + 1e-6


def test_generator_respects_budget():
    G = CloakGenerator(eps=8 / 255, pretrained=False)
    nn.init.normal_(G.net.segmentation_head[0].weight, std=10.0)   # saturate tanh
    x = torch.rand(2, 3, 64, 64)
    mask = texture_mask(x)
    x_cloak, delta = G(x, mask)
    assert x_cloak.shape == x.shape
    assert (delta.abs() <= 8 / 255 * mask + 1e-6).all()
    assert 0 <= x_cloak.min() and x_cloak.max() <= 1


def test_real_jpeg_is_straight_through():
    x = torch.rand(2, 3, 64, 64, requires_grad=True)
    y = real_jpeg(x, [50, 50])
    assert not torch.allclose(y, x)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_eot_keeps_shape_and_gradient(cfg):
    eot = EOT(cfg)
    for seed in range(20):
        random.seed(seed)
        x = torch.rand(2, 3, 64, 64, requires_grad=True)
        y, crop = eot(x, 1.0)
        assert y.shape == x.shape
        (g,) = torch.autograd.grad(y.sum(), x)
        assert g.abs().sum() > 0


def test_align_is_identity_on_template():
    x = torch.rand(1, 3, 112, 112)
    out = align_faces(x, ARCFACE_TEMPLATE[None].clone())
    assert torch.allclose(out, x, atol=1e-4)


def test_align_follows_translation():
    x = torch.rand(1, 3, 256, 256)
    lm = ARCFACE_TEMPLATE[None] + torch.tensor([50.0, 30.0])
    assert torch.allclose(align_faces(x, lm), x[..., 30:142, 50:162], atol=1e-4)


def test_crop_moves_landmarks_with_image():
    lm = torch.tensor([[[100.0, 60.0]] * 5])
    crop = Crop(x0=0.25, y0=0.25, w=0.5, h=0.5)       # 256 px image -> 128 px crop at (64, 64)
    assert torch.allclose(crop.apply_to_landmarks(lm, 256, 256), torch.tensor([[[72.0, -8.0]] * 5]))


def test_ghost_noise_and_clean():
    m = TinyCLIP().freeze(ghost_std=0.1)
    x = torch.rand(1, 3, 32, 32)
    assert not torch.allclose(m(x), m(x))
    with m.clean():
        assert torch.allclose(m(x), m(x))


def test_loss_backprops_to_generator(cfg):
    sur = tiny_surrogates(cfg, "cpu")
    crit = CloakLoss(cfg, sur, TinyLPIPS())
    G = CloakGenerator(cfg.eps, pretrained=False)
    x = torch.rand(2, 3, 64, 64)
    lm = (ARCFACE_TEMPLATE * (40 / 112) + 12).expand(2, -1, -1)
    x_cloak, _ = G(x, texture_mask(x))
    x_aug, crop = EOT(cfg)(x_cloak, 1.0)
    loss, logs, chosen = crit(x, x_cloak, x_aug, crop, torch.tensor([True, False]), lm)
    loss.backward()
    assert {"loss_vae", "loss_clip", "loss_face", "loss_lpips"} <= logs.keys()
    assert sum(p.grad.abs().sum() for p in G.parameters() if p.grad is not None) > 0


# ---------------------------------------------------------------- end to end

def _patch(monkeypatch, module):
    monkeypatch.setattr(module, "load_surrogates", tiny_surrogates)
    monkeypatch.setattr(module, "load_lpips", lambda device: TinyLPIPS().to(device))


def test_train_runs_and_resumes(cfg, dataset, tmp_path, monkeypatch):
    _patch(monkeypatch, train)
    monkeypatch.setattr(train, "CloakGenerator", lambda eps: CloakGenerator(eps, pretrained=False))
    out = tmp_path / "run"
    args = ["--config", "", f"data_root={str(dataset)!r}", f"out_dir={str(out)!r}", "image_size=64",
            "batch_size=2", "num_workers=0", "warmup_steps=1", "log_every=1", "val_every=2", "ckpt_every=2"]
    train.main(args + ["steps=3"])
    assert (out / "last.pt").exists() and (out / "generator_ema.pt").exists()
    assert torch.load(out / "last.pt", weights_only=False)["step"] == 3
    train.main(args + ["steps=5"])
    assert torch.load(out / "last.pt", weights_only=False)["step"] == 5
    assert len(pd.read_csv(out / "log.csv")) >= 5


def test_pgd_runs(cfg, dataset, tmp_path, monkeypatch):
    _patch(monkeypatch, pgd)
    out = tmp_path / "pgd"
    pgd.main(["--config", "", f"data_root={str(dataset)!r}", f"out_dir={str(out)!r}", "image_size=64",
              "pgd_steps=2", "pgd_batch=2", "pgd_images=2", "num_workers=0"])
    df = pd.read_csv(out / "metrics.csv")
    assert set(df.method) == {"clean", "pgd"} and len(df) == 4
    assert (df[df.method == "pgd"].linf_255 <= cfg.eps * 255 + 1e-3).all()
