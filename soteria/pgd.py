"""Per-image PGD + EOT cloaks: the baseline the generator has to approach.

    python -m soteria.pgd --config configs/pgd.yaml [eps=0.0627 ...]

Uses exactly the generator's objective, surrogates, mask and augmentations,
so the only difference between the two is per-image optimisation versus one
forward pass. This run is also the go/no-go gate before training: if PGD at
8/255 cannot move the surrogates after a real JPEG q75 round trip, a
generator will not either, and the loss needs fixing first.

Writes to out_dir: images/<idx>.png, metrics.csv (one row per image and
method, "clean" being the unprotected photo under the same conditions) and
summary.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from .augment import EOT
from .config import Config, seed_everything
from .data import CloakDataset
from .losses import CloakLoss, load_lpips
from .mask import texture_mask
from .metrics import protection_metrics
from .surrogates import load_surrogates


def pgd_cloak(x, has_face, landmarks, criterion, eot, cfg, use_amp: bool) -> torch.Tensor:
    bound = cfg.eps * texture_mask(x, cfg.mask_min)
    alpha = cfg.eps * cfg.pgd_alpha_frac
    delta = torch.zeros_like(x, requires_grad=True)
    for _ in range(cfg.pgd_steps):
        x_cloak = (x + delta).clamp(0, 1)
        x_aug, crop = eot(x_cloak, 1.0)
        with torch.autocast(x.device.type, dtype=torch.float16, enabled=use_amp):
            loss, _, _ = criterion(x, x_cloak, x_aug, crop, has_face, landmarks)
        # The update only uses the gradient's sign, so scaling the loss is free
        # and keeps fp16 gradients from underflowing to zero.
        (grad,) = torch.autograd.grad(loss * 1024, delta)
        with torch.no_grad():
            delta -= alpha * grad.sign()
            delta.copy_(torch.maximum(torch.minimum(delta, bound), -bound))
    return (x + delta.detach()).clamp(0, 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pgd.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = Config.load(args.config, args.overrides)

    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.amp and device.type == "cuda"
    out_dir = Path(cfg.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    ds = CloakDataset(cfg.data_root, "eval", cfg.image_size)
    ds = Subset(ds, range(min(cfg.pgd_images, len(ds))))
    loader = DataLoader(ds, batch_size=cfg.pgd_batch, shuffle=False, num_workers=cfg.num_workers)

    surrogates = load_surrogates(cfg, device)
    lpips_model = load_lpips(device)
    criterion = CloakLoss(cfg, surrogates, lpips_model)
    eot = EOT(cfg)

    rows, total_s = [], 0.0
    for batch in loader:
        x = batch["image"].to(device)
        has_face, lm = batch["has_face"].to(device), batch["landmarks"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        x_cloak = pgd_cloak(x, has_face, lm, criterion, eot, cfg, use_amp)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        total_s += dt

        for method, xc in (("clean", x), ("pgd", x_cloak)):
            for i, r in enumerate(protection_metrics(surrogates, x, xc, has_face, lm, lpips_model)):
                rows.append({"index": int(batch["index"][i]), "method": method, "has_face": bool(has_face[i]),
                             "sec_per_image": dt / x.shape[0] if method == "pgd" else 0.0, **r})
        for i, idx in enumerate(batch["index"].tolist()):
            save_image(x_cloak[i], out_dir / "images" / f"{idx:05d}.png")
        print(f"{len(rows) // 2}/{len(ds)} images, {dt / x.shape[0]:.1f} s/image", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics.csv", index=False)
    numeric = df.drop(columns=["index", "has_face"]).groupby("method").agg(lambda s: float(np.nanmean(s)))
    summary = {"eps_255": cfg.eps * 255, "steps": cfg.pgd_steps, "images": len(ds),
               "sec_per_image": total_s / max(1, len(ds)), "by_method": numeric.to_dict(orient="index")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(numeric.T.to_string())


if __name__ == "__main__":
    main()
