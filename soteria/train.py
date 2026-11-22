"""Train the cloak generator against the frozen surrogates.

    python -m soteria.train --config configs/base_256.yaml [key=value ...]

Colab-safe by design: out_dir should live on Drive. The run checkpoints every
ckpt_every steps (atomic write), resumes automatically from out_dir/last.pt,
and stops cleanly when budget_min runs out, so a reclaimed session costs only
the steps since the last checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from .augment import EOT
from .config import Config, seed_everything
from .data import CloakDataset
from .generator import CloakGenerator
from .losses import CloakLoss, load_lpips
from .mask import texture_mask
from .metrics import protection_metrics
from .surrogates import load_surrogates


def infinite(loader):
    while True:
        yield from loader


def lr_lambda(cfg):
    def f(step):
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        t = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
    return f


@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.lerp_(pm, 1 - decay)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def validate(ema, batch, surrogates, lpips_model, mask_min: float, out_dir: Path, step: int) -> dict:
    x = batch["image"]
    x_cloak, delta = ema(x, texture_mask(x, mask_min))
    rows = protection_metrics(surrogates, x, x_cloak, batch["has_face"], batch["landmarks"], lpips_model)
    keys = rows[0].keys()
    summary = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    n = min(8, x.shape[0])
    grid = torch.cat([x[:n], x_cloak[:n], (delta[:n] * 10 + 0.5).clamp(0, 1)])
    save_image(grid, out_dir / "samples" / f"step_{step:06d}.png", nrow=n)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base_256.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = Config.load(args.config, args.overrides)

    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.amp and device.type == "cuda"
    out_dir = Path(cfg.out_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    train_ds = CloakDataset(cfg.data_root, "train", cfg.image_size, flip=True)
    val_ds = CloakDataset(cfg.data_root, "eval", cfg.image_size)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                        num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
                        persistent_workers=cfg.num_workers > 0)
    # A fixed validation batch, faces first, so curves are comparable across runs.
    val_idx = sorted(range(len(val_ds)), key=lambda i: not val_ds.df.has_face[i])[:min(16, len(val_ds))]
    val_batch = to_device(torch.utils.data.default_collate([val_ds[i] for i in val_idx]), device)
    print(f"train {len(train_ds)} images, val batch {len(val_idx)}, device {device}, amp {use_amp}")

    surrogates = load_surrogates(cfg, device)
    lpips_model = load_lpips(device)
    criterion = CloakLoss(cfg, surrogates, lpips_model)
    eot = EOT(cfg)

    G = CloakGenerator(cfg.eps).to(device)
    ema = copy.deepcopy(G).eval().requires_grad_(False)
    opt = torch.optim.AdamW([
        {"params": G.encoder_parameters(), "lr": cfg.lr * cfg.encoder_lr_mult},
        {"params": G.decoder_parameters(), "lr": cfg.lr},
    ], weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step, ckpt_path = 0, out_dir / "last.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        G.load_state_dict(ck["G"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        random.setstate(ck["rng"]["python"])
        np.random.set_state(ck["rng"]["numpy"])
        torch.set_rng_state(ck["rng"]["torch"].cpu())   # map_location moved it to the GPU
        step = ck["step"]
        print(f"resumed from step {step}")

    def save():
        atomic_save({
            "step": step, "G": G.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
            "sched": sched.state_dict(), "scaler": scaler.state_dict(), "cfg": cfg.to_dict(),
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()},
        }, ckpt_path)
        # Inference-only export: EMA weights + eps, nothing else.
        atomic_save({"state_dict": ema.state_dict(), "eps": cfg.eps, "step": step}, out_dir / "generator_ema.pt")

    writer = SummaryWriter(out_dir / "tb")
    log_file = out_dir / "log.csv"
    started, t_last, seen = time.time(), time.time(), 0
    batches = infinite(loader)
    G.train()

    while step < cfg.steps:
        batch = to_device(next(batches), device)
        x = batch["image"]
        strength = min(1.0, step / max(1, cfg.curriculum_frac * cfg.steps))

        amp = torch.autocast(device.type, dtype=torch.float16, enabled=use_amp)
        with amp:
            x_cloak, _ = G(x, texture_mask(x, cfg.mask_min))
        # EOT runs in fp32: the DCT in kornia's JPEG loses too much in fp16.
        x_aug, crop = eot(x_cloak, strength)
        with amp:
            loss, logs, chosen = criterion(x, x_cloak, x_aug, crop, batch["has_face"], batch["landmarks"])

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(G.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        sched.step()
        ema_update(ema, G, cfg.ema_decay)
        step += 1
        seen += x.shape[0]

        if step % cfg.log_every == 0 or step == 1:
            now = time.time()
            row = {"step": step, "time_min": (now - started) / 60, "img_per_s": seen / (now - t_last),
                   "lr": sched.get_last_lr()[1], "strength": strength, "grad_norm": float(grad_norm),
                   **{k: float(v) for k, v in logs.items()},
                   "surrogates": "+".join(m.key for m in chosen.values())}
            t_last, seen = now, 0
            for k, v in row.items():
                if isinstance(v, float):
                    writer.add_scalar(f"train/{k}", v, step)
            new = not log_file.exists()
            with log_file.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

        if step % cfg.val_every == 0 or step == cfg.steps:
            summary = validate(ema, val_batch, surrogates, lpips_model, cfg.mask_min, out_dir, step)
            for k, v in summary.items():
                writer.add_scalar(f"val/{k}", v, step)
            print("val", json.dumps({k: round(v, 4) for k, v in summary.items()}), flush=True)

        if step % cfg.ckpt_every == 0:
            save()
        if cfg.budget_min and (time.time() - started) / 60 > cfg.budget_min:
            print(f"time budget of {cfg.budget_min} min reached at step {step}")
            break

    save()
    writer.close()
    print(f"done at step {step}; checkpoint in {out_dir}")


if __name__ == "__main__":
    main()
