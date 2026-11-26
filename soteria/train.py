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


# Keys that may differ between sessions of one run. Anything else changing on
# resume (loss weights, floors, eps, surrogates, ...) would silently splice two
# different experiments into one checkpoint, so it is refused.
RESUMABLE_KEYS = {"steps", "budget_min", "log_every", "val_every", "ckpt_every", "num_workers",
                  "data_root", "out_dir"}


def check_resume_config(saved: dict, cfg: Config) -> None:
    now = cfg.to_dict()
    # Keys a checkpoint does not have were added to Config after it was saved.
    diff = {k: (saved[k], now[k]) for k in now if k in saved and k not in RESUMABLE_KEYS and saved[k] != now[k]}
    if diff:
        lines = "\n".join(f"  {k}: checkpoint {a!r} -> now {b!r}" for k, (a, b) in diff.items())
        raise ValueError(f"{cfg.out_dir} holds a run with a different config:\n{lines}\n"
                         "Restore the original settings to resume, or use a new out_dir (RUN_NAME) "
                         "for the new experiment.")


def trim_csv(path: Path, max_step: int) -> None:
    """Drop rows logged after the checkpoint we resume from: those steps are
    about to be re-run, and keeping both copies would double them in plots."""
    if not path.exists():
        return
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, [r for r in reader if int(r["step"]) <= max_step]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def append_csv(path: Path, row: dict) -> None:
    """Append under the file's existing header, so a row with a missing or
    extra key cannot shift its values into the wrong columns."""
    if path.exists():
        with path.open(newline="") as f:
            fields = next(csv.reader(f), None)
    else:
        fields = None
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(row), restval="", extrasaction="ignore")
        if not fields:
            w.writeheader()
        w.writerow(row)


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

    step, ckpt_path, prior_min, last_val = 0, out_dir / "last.pt", 0.0, None
    log_file, val_file = out_dir / "log.csv", out_dir / "val.csv"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        check_resume_config(ck["cfg"], cfg)
        G.load_state_dict(ck["G"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        random.setstate(ck["rng"]["python"])
        np.random.set_state(ck["rng"]["numpy"])
        torch.set_rng_state(ck["rng"]["torch"].cpu())   # map_location moved it to the GPU
        step = ck["step"]
        prior_min, last_val = ck.get("train_min", 0.0), ck.get("last_val")
        trim_csv(log_file, step)
        trim_csv(val_file, step)
        print(f"resumed from step {step} ({prior_min:.0f} GPU-min so far)")
    if step >= cfg.steps:
        print(f"already finished: step {step} of {cfg.steps}. Results are in {out_dir}")
        return
    started = time.time()
    start_step = step

    def save():
        train_min = prior_min + (time.time() - started) / 60
        atomic_save({
            "step": step, "G": G.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
            "sched": sched.state_dict(), "scaler": scaler.state_dict(), "cfg": cfg.to_dict(),
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()},
            "train_min": train_min, "last_val": last_val,
        }, ckpt_path)
        # Inference-only export: EMA weights + eps, nothing else.
        atomic_save({"state_dict": ema.state_dict(), "eps": cfg.eps, "step": step}, out_dir / "generator_ema.pt")
        # Human-readable progress, small enough to read from Drive or paste anywhere.
        session_steps = step - start_step
        min_per_step = (time.time() - started) / 60 / session_steps if session_steps else None
        status = {"step": step, "steps": cfg.steps, "done": step >= cfg.steps, "train_min": round(train_min, 1),
                  "min_per_step": min_per_step,
                  "eta_hours": round((cfg.steps - step) * min_per_step / 60, 1) if min_per_step else None,
                  "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "last_val": last_val}
        tmp = out_dir / "status.json.tmp"
        tmp.write_text(json.dumps(status, indent=2))
        os.replace(tmp, out_dir / "status.json")

    writer = SummaryWriter(out_dir / "tb")
    t_last, seen = time.time(), 0
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
            append_csv(log_file, row)
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

        if step % cfg.val_every == 0 or step == cfg.steps:
            summary = validate(ema, val_batch, surrogates, lpips_model, cfg.mask_min, out_dir, step)
            for k, v in summary.items():
                writer.add_scalar(f"val/{k}", v, step)
            last_val = {"step": step, **summary}
            append_csv(val_file, last_val)
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
