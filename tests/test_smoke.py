"""CPU smoke tests. No downloads: real surrogates and LPIPS are swapped for tiny stand-ins.

    pytest -q
"""
from __future__ import annotations

import json
import random
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from soteria import pgd, train
from soteria import runtime as rt
from soteria.eval import edit
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


def test_align_works_under_autocast():
    # PGD / training run under AMP; SVD and det have no fp16/bf16 kernels.
    x = torch.rand(2, 3, 256, 256)
    lm = (ARCFACE_TEMPLATE + torch.tensor([50.0, 30.0])).expand(2, -1, -1)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = align_faces(x, lm)
    assert out.dtype == torch.float32
    assert torch.allclose(out, align_faces(x, lm))


def test_similarity_floor_stops_gradient(cfg):
    sur = tiny_surrogates(cfg, "cpu")
    x = torch.rand(2, 3, 64, 64)
    lm = (ARCFACE_TEMPLATE * (40 / 112) + 12).expand(2, -1, -1)
    has_face = torch.tensor([True, False])

    def grad_norm(**overrides):
        c = Config(**{**cfg.to_dict(), "w_vae": 0.0, "ghost_std": 0.0, **overrides})
        delta = (torch.rand_like(x) - 0.5) * 0.01
        delta.requires_grad_(True)
        x_cloak = (x + delta).clamp(0, 1)
        loss, logs, _ = CloakLoss(c, sur, TinyLPIPS())(x, x_cloak, x_cloak, Crop(), has_face, lm)
        assert {"clip_cos", "face_cos"} <= logs.keys()
        return torch.autograd.grad(loss, delta)[0].abs().sum()

    assert grad_norm() > 0                                         # default -1: no floor
    assert grad_norm(clip_floor=1.5, face_floor=1.5) == 0          # every image past its floor
    assert grad_norm(clip_floor=1.5) > 0 and grad_norm(face_floor=1.5) > 0


def test_loss_backprops_under_autocast(cfg):
    # CPU bf16 autocast stands in for CUDA fp16: same op lists, catches
    # dtype-only crashes that the fp32 tests above cannot.
    sur = tiny_surrogates(cfg, "cpu")
    crit = CloakLoss(cfg, sur, TinyLPIPS())
    x = torch.rand(2, 3, 64, 64)
    delta = torch.zeros_like(x, requires_grad=True)
    lm = (ARCFACE_TEMPLATE * (40 / 112) + 12).expand(2, -1, -1)
    for seed in range(10):
        random.seed(seed)
        x_cloak = (x + delta).clamp(0, 1)
        x_aug, crop = EOT(cfg)(x_cloak, 1.0)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss, logs, _ = crit(x, x_cloak, x_aug, crop, torch.tensor([True, False]), lm)
        (grad,) = torch.autograd.grad(loss, delta)
        assert "loss_face" in logs and torch.isfinite(grad).all()


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
    # A row logged after the last checkpoint (lost in a disconnect) must not survive the resume.
    with (out / "log.csv").open("a") as f:
        f.write("99" + "," * (len(pd.read_csv(out / "log.csv").columns) - 1) + "\n")
    train.main(args + ["steps=5"])
    assert torch.load(out / "last.pt", weights_only=False)["step"] == 5
    log = pd.read_csv(out / "log.csv")
    assert list(log.step) == [1, 2, 3, 4, 5]
    assert log.notna().all().drop(["loss_face", "face_cos"], errors="ignore").all()   # no shifted columns
    assert list(pd.read_csv(out / "val.csv").step) == [2, 3, 4, 5]      # every 2 steps + end of each run
    status = json.loads((out / "status.json").read_text())
    assert status["step"] == 5 and status["done"] and status["last_val"]["step"] == 5
    train.main(args + ["steps=5"])                                   # finished: returns early
    with pytest.raises(ValueError, match="w_vae"):
        train.main(args + ["steps=8", "w_vae=2.0"])                  # different loss: refused


def test_csv_append_keeps_columns(tmp_path):
    p = tmp_path / "log.csv"
    train.append_csv(p, {"step": 1, "a": 1.0, "b": 2.0})
    train.append_csv(p, {"step": 2, "b": 4.0})                       # missing "a"
    train.append_csv(p, {"step": 3, "a": 5.0, "b": 6.0, "new": 0.0})  # extra key
    df = pd.read_csv(p)
    assert list(df.columns) == ["step", "a", "b"] and list(df.b) == [2.0, 4.0, 6.0]
    train.trim_csv(p, 2)
    assert list(pd.read_csv(p).step) == [1, 2]


def _make_tar(path: Path, tmp_path: Path) -> None:
    src = tmp_path / "src" / "prepared"
    src.mkdir(parents=True, exist_ok=True)
    (src / "manifest.csv").write_text("path\na.png\nb.png\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as t:
        t.add(src, arcname="prepared")


def test_restore_data_colab_style(tmp_path):
    # Colab: prepared.tar in the results folder (Drive), no Kaggle inputs.
    _make_tar(tmp_path / "drive" / "data" / "prepared.tar", tmp_path)
    dest, no_inputs = tmp_path / "local", tmp_path / "no_inputs"
    assert rt.restore_data(tmp_path / "drive", dest, no_inputs) == dest / "prepared"
    assert (dest / "prepared" / "manifest.csv").exists()
    assert rt.restore_data(tmp_path / "drive", dest, no_inputs) == dest / "prepared"   # second call: no-op
    with pytest.raises(FileNotFoundError, match="01_data"):
        rt.restore_data(tmp_path / "nowhere", tmp_path / "empty", no_inputs)


def test_restore_data_kaggle_style(tmp_path):
    # Kaggle: an attached notebook output holds prepared.tar a few folders down.
    inputs = tmp_path / "input"
    _make_tar(inputs / "notebooks" / "me" / "soteria-01-data" / "soteria" / "data" / "prepared.tar", tmp_path)
    dest = tmp_path / "local"
    assert rt.restore_data(tmp_path / "working", dest, inputs) == dest / "prepared"
    # An already unpacked prepared/ folder in an input is used in place.
    unpacked = inputs / "some-dataset" / "prepared"
    unpacked.mkdir(parents=True)
    (unpacked / "manifest.csv").write_text("path\na.png\n")
    assert rt.restore_data(tmp_path / "working", tmp_path / "fresh", inputs) == unpacked


def test_restore_run_from_input(tmp_path):
    inputs, run = tmp_path / "input", tmp_path / "working" / "runs" / "base_256"
    saved = inputs / "soteria-03-train" / "soteria" / "runs" / "base_256"
    saved.mkdir(parents=True)
    (saved / "last.pt").write_bytes(b"ckpt")
    (saved / "log.csv").write_text("step\n1\n")
    rt.restore_run(run, inputs)
    assert (run / "last.pt").read_bytes() == b"ckpt" and (run / "log.csv").exists()
    (run / "last.pt").write_bytes(b"newer")
    rt.restore_run(run, inputs)                                      # a local checkpoint wins
    assert (run / "last.pt").read_bytes() == b"newer"
    rt.restore_run(tmp_path / "working" / "runs" / "other", inputs)  # nothing saved: no-op
    assert not (tmp_path / "working" / "runs" / "other").exists()


def test_session_budget(monkeypatch):
    import time as _time
    monkeypatch.setattr(rt, "platform", lambda: "kaggle")
    assert rt.session_budget_min(started=_time.time()) in (689, 690)                 # fresh session (rounded down)
    assert rt.session_budget_min(started=_time.time() - 50 * 60) in (639, 640)       # after 50 min of setup
    assert rt.session_budget_min(started=_time.time() - 13 * 3600) == 1               # never 0 (= no limit)
    assert 1 <= rt.session_budget_min() <= 690                                        # real kernel start time
    monkeypatch.setattr(rt, "platform", lambda: "colab")
    assert rt.session_budget_min() == 330


def test_find_input_dir(tmp_path):
    (tmp_path / "datasets" / "xhlulu" / "flickrfaceshq-dataset-nvidia-resized-256px").mkdir(parents=True)
    assert rt.find_input_dir("flickrfaceshq", base=tmp_path).name.startswith("flickrfaceshq")
    assert rt.find_input_dir("nothing-like-this", base=tmp_path) is None
    (tmp_path / "Other" / "FFHQ-256").mkdir(parents=True)
    assert rt.find_input_dir("nothing-like-this", "ffhq", base=tmp_path).name == "FFHQ-256"  # any part, any case
    listing = rt.describe_inputs(tmp_path)
    assert "flickrfaceshq-dataset-nvidia-resized-256px" in listing and "entries" in listing
    assert "nothing attached" in rt.describe_inputs(tmp_path / "missing")


class FakeEditor:
    """Stand-in for SD img2img: blur plus noise drawn from each item's generator."""

    def __call__(self, prompt, image, generator, output_type, **_):
        assert output_type == "pt" and len(prompt) == len(generator) == image.shape[0]
        noise = torch.stack([torch.randn(image.shape[1:], generator=g) for g in generator])
        out = F.avg_pool2d(image, 3, 1, 1) + 0.05 * noise
        return type("Out", (), {"images": out.clamp(0, 1)})()


def test_edit_check_runs(cfg, dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(edit, "load_editor", lambda device: FakeEditor())
    monkeypatch.setattr(edit, "load_surrogates", tiny_surrogates)
    monkeypatch.setattr(edit, "load_lpips", lambda device: TinyLPIPS().to(device))
    pgd_root, out = tmp_path / "pgd", tmp_path / "edit"
    for run in ("eps8", "abl_vae_only"):
        (pgd_root / run / "images").mkdir(parents=True)
        for i in range(2):                                           # the fixture's 2 eval images
            Image.fromarray(np.random.default_rng(i).integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(
                pgd_root / run / "images" / f"{i:05d}.png")
    _patch(monkeypatch, pgd)                                         # for --make-missing
    edit.main(["--config", "", f"data_root={str(dataset)!r}", "image_size=64", "pgd_steps=2", "pgd_batch=2",
               "num_workers=0", "--make-missing",
               "--pgd-root", str(pgd_root), "--out", str(out), "--sizes", "64", "128",
               "--runs", "pgd8", "vae_only", "pgd16", "gone=missing_run"])
    # pgd16 had no folder: computed for exactly the edit-check images, with its own eps.
    assert sorted(p.name for p in (pgd_root / "eps16" / "images").iterdir()) == ["00000.png", "00001.png"]
    assert json.loads((pgd_root / "eps16" / "summary.json").read_text())["eps_255"] == pytest.approx(16, abs=0.01)
    df = pd.read_csv(out / "metrics.csv")
    assert set(df.method) == {"clean", "clean_again", "random8", "pgd8", "vae_only", "pgd16"}  # custom missing: skipped
    assert len(df) == 2 * 2 * 6                                                  # images x sizes x methods
    same = df.method.isin(["clean", "clean_again"])
    assert (df[same].lpips_vs_clean_edit == 0).all()                             # same input + seed -> same edit
    assert (df[~same].lpips_vs_clean_edit > 0).all()
    assert df[df.has_face].id_cos.notna().all() and df[~df.has_face].id_cos.isna().all()
    assert (out / "grid_64.png").exists() and (out / "grid_128.png").exists()
    # Upscaled 256->128 here (no images_128/): not native. random8 and the 64 px size always are.
    assert not df[(df["size"] == 128) & (df.method == "pgd8")].native.any()
    assert df[df.method == "random8"].native.all() and df[df["size"] == 64].native.all()
    assert "pink hair" in json.loads((out / "settings.json").read_text())["prompt_texts"]["ffhq"]


def test_generator_eval_and_native_edit_check(cfg, dataset, tmp_path, monkeypatch):
    from soteria.eval import generator as gen_eval
    run = tmp_path / "runs" / "base_256"
    run.mkdir(parents=True)
    G = CloakGenerator(8 / 255, pretrained=False)
    nn.init.normal_(G.net.segmentation_head[0].weight, std=1.0)     # a visible cloak, not ~0
    torch.save({"state_dict": G.state_dict(), "eps": 8 / 255, "step": 123}, run / "generator_ema.pt")
    (run / "config.json").write_text(json.dumps({"mask_min": 0.3}))
    monkeypatch.setattr(gen_eval, "load_surrogates", tiny_surrogates)
    monkeypatch.setattr(gen_eval, "load_lpips", lambda device: TinyLPIPS().to(device))
    out = tmp_path / "eval"
    gen_eval.main(["--checkpoint", str(rt.find_checkpoint(run)), "--out", str(out), "--config", "",
                   f"data_root={str(dataset)!r}", "image_size=64", "num_workers=0",
                   "--batch", "2", "--edit-n", "2", "--edit-sizes", "128", "--speed", "64x1", "--speed-reps", "1"])
    df = pd.read_csv(out / "metrics.csv")
    assert len(df) == 2 * 2 and set(df.method) == {"clean", "generator"}
    for cond in ("none", "jpeg90", "jpeg75", "jpeg50", "resize50", "platform"):
        assert f"{cond}/tiny_vae_ratio" in df and f"{cond}/tiny_clip_cos" in df
    gen = df[df.method == "generator"]
    assert (gen.linf_255 <= 8 + 1e-3).all() and (gen.linf_255 > 0.5).all()      # within budget, not empty
    summary = pd.read_csv(out / "summary.csv", index_col=[0, 1])
    assert ("generator", "all") in summary.index and ("clean", "ffhq") in summary.index
    assert json.loads((out / "settings.json").read_text())["step"] == 123
    assert json.loads((out / "speed.json").read_text())["results"][0]["ms_per_image"] > 0
    assert (out / "samples.png").exists()
    assert len(list((out / "cloaks" / "images").glob("*.png"))) == 2
    native = sorted((out / "cloaks" / "images_128").glob("*.png"))
    assert len(native) == 2 and Image.open(native[0]).size == (128, 128)

    # The notebook's next cell: the edit check picks up the 128 px cloaks as native.
    monkeypatch.setattr(edit, "load_editor", lambda device: FakeEditor())
    monkeypatch.setattr(edit, "load_surrogates", tiny_surrogates)
    monkeypatch.setattr(edit, "load_lpips", lambda device: TinyLPIPS().to(device))
    edit.main(["--config", "", f"data_root={str(dataset)!r}", "image_size=64", "num_workers=0",
               "--pgd-root", str(tmp_path / "pgd"), "--runs", f"generator={out / 'cloaks'}",
               "--sizes", "64", "128", "--out", str(out / "edit_check")])
    ed = pd.read_csv(out / "edit_check" / "metrics.csv")
    assert ed[ed.method == "generator"].native.all()
    assert (ed[ed.method == "generator"].lpips_vs_clean_edit > 0).all()


def test_find_checkpoint(tmp_path):
    inputs = tmp_path / "input"
    with pytest.raises(FileNotFoundError, match="generator_ema.pt"):
        rt.find_checkpoint(tmp_path / "runs" / "base_256", inputs)
    shared = inputs / "datasets" / "friend" / "soteria-trained" / "soteria" / "runs" / "base_256"
    shared.mkdir(parents=True)
    (shared / "generator_ema.pt").write_bytes(b"g")
    assert rt.find_checkpoint(tmp_path / "runs" / "base_256", inputs) == shared / "generator_ema.pt"
    local = tmp_path / "runs" / "base_256"
    local.mkdir(parents=True)
    (local / "generator_ema.pt").write_bytes(b"mine")
    assert rt.find_checkpoint(local, inputs) == local / "generator_ema.pt"          # own run folder first


def test_notebook_cells_are_valid_python():
    for nb in Path(__file__).resolve().parents[1].glob("colab/*.ipynb"):
        for c in json.loads(nb.read_text(encoding="utf-8"))["cells"]:
            if c["cell_type"] == "code":
                src = "".join(c["source"])
                lines = [ln[:len(ln) - len(ln.lstrip())] + "pass" if ln.lstrip().startswith(("!", "%")) else ln
                         for ln in src.splitlines()]
                compile("\n".join(lines), f"{nb.name}", "exec")
                if "rt.prepare(" in src:                             # setup block must not need Colab
                    assert "from google.colab import drive" not in src.split("if not KAGGLE")[0]


def test_pgd_subset_keeps_source_mix():
    # Manifest order: every FFHQ eval row before every COCO one.
    sources = pd.Series(["ffhq"] * 300 + ["coco"] * 100)
    picked = pgd.stratified_indices(sources, 200)
    assert len(picked) == len(set(picked)) == 200
    assert sources[picked].value_counts().to_dict() == {"ffhq": 150, "coco": 50}
    assert len(pgd.stratified_indices(sources, 7)) == 7          # rounding still sums to n
    assert len(pgd.stratified_indices(sources, 1000)) == 400     # capped at the split size


def test_pgd_runs(cfg, dataset, tmp_path, monkeypatch):
    _patch(monkeypatch, pgd)
    out = tmp_path / "pgd"
    pgd.main(["--config", "", f"data_root={str(dataset)!r}", f"out_dir={str(out)!r}", "image_size=64",
              "pgd_steps=2", "pgd_batch=2", "pgd_images=2", "num_workers=0"])
    df = pd.read_csv(out / "metrics.csv")
    assert set(df.method) == {"clean", "pgd"} and len(df) == 4
    assert (df[df.method == "pgd"].linf_255 <= cfg.eps * 255 + 1e-3).all()
