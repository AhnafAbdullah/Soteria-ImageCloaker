"""Evaluate a trained cloak generator: protection, image quality, speed.

    python -m soteria.eval.generator --checkpoint <run>/generator_ema.pt --out <root>/eval/<run> \
        [data_root=...] [--edit-n 16] [--edit-sizes 512]

1. Protection + quality on the whole held-out eval split (300 FFHQ + 100 COCO),
   for the clean photo and its cloak, after each post-processing condition in
   soteria.metrics.CONDITIONS (none, JPEG q90/75/50, half-size resize, a
   messaging-app round trip). Same metrics as the PGD baseline, so the two
   tables line up. -> metrics.csv, summary.csv (by method x source)
2. Speed: ms per image for the generator plus texture mask, fp16 on the GPU,
   at several sizes. PGD needed ~25 s per 256 px image. -> speed.json
3. Cloaks for the SD edit check (soteria.eval.edit):
   cloaks/images/<idx>.png      256 px cloaks of every eval image
   cloaks/images_<S>/<idx>.png  the edit-check images upscaled to S px and
                                cloaked *at S px*. The generator is fully
                                convolutional, so unlike PGD it can cloak at
                                the resolution the editor encodes at; the edit
                                check uses these instead of upscaling the 256
                                px cloak.
4. samples.png: clean / cloaked / |delta| x 10 for 8 eval images.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from ..config import Config
from ..data import CloakDataset
from ..infer import load_generator
from ..losses import load_lpips
from ..mask import texture_mask
from ..metrics import CONDITIONS, protection_metrics
from ..pgd import stratified_indices
from ..surrogates import load_surrogates
from .edit import resize


def cloak(G, x: torch.Tensor, mask_min: float) -> torch.Tensor:
    """Cloak a batch the way deployment would: fp16 on the GPU."""
    with torch.autocast(x.device.type, dtype=torch.float16, enabled=x.device.type == "cuda"):
        x_cloak, _ = G(x, texture_mask(x, mask_min))
    return x_cloak.float().clamp(0, 1)


@torch.no_grad()
def benchmark(G, device, mask_min: float, shapes: list[tuple[int, int]], reps: int) -> list[dict]:
    out = []
    for size, batch in shapes:
        x = torch.rand(batch, 3, size, size, device=device)
        for _ in range(2):                         # warm-up: cuDNN autotuning, allocator
            cloak(G, x, mask_min)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            cloak(G, x, mask_min)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / reps / batch * 1000
        out.append({"size": size, "batch": batch, "ms_per_image": round(ms, 2),
                    "images_per_s": round(1000 / ms, 1)})
        print(f"speed {size}px batch {batch}: {ms:.1f} ms/image", flush=True)
    return out


@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="generator_ema.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/pgd.yaml", help="surrogates + data_root")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--edit-n", type=int, default=16, help="images cloaked at native edit sizes")
    ap.add_argument("--edit-sizes", type=int, nargs="*", default=[512])
    ap.add_argument("--speed", nargs="*", default=["256x1", "256x16", "512x1", "1024x1"], help="SIZExBATCH")
    ap.add_argument("--speed-reps", type=int, default=20)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = Config.load(args.config, args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    (out / "cloaks" / "images").mkdir(parents=True, exist_ok=True)

    # The run's own config.json (written by train.py next to the checkpoint) has
    # the mask_min the generator was trained with.
    run_cfg = Path(args.checkpoint).with_name("config.json")
    trained = json.loads(run_cfg.read_text()) if run_cfg.exists() else {}
    mask_min = trained.get("mask_min", cfg.mask_min)
    G = load_generator(args.checkpoint, device, mask_min)
    ck_step = torch.load(args.checkpoint, map_location="cpu", weights_only=False).get("step")
    print(f"generator: {args.checkpoint} (step {ck_step}, eps {G.eps * 255:.1f}/255, mask_min {mask_min})")

    ds = CloakDataset(cfg.data_root, "eval", cfg.image_size)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=cfg.num_workers)
    surrogates = load_surrogates(cfg, device)
    lpips_model = load_lpips(device)

    rows, shown = [], []
    for batch in loader:
        x = batch["image"].to(device)
        has_face, lm = batch["has_face"].to(device), batch["landmarks"].to(device)
        x_cloak = cloak(G, x, mask_min)
        for method, xc in (("clean", x), ("generator", x_cloak)):
            for i, r in enumerate(protection_metrics(surrogates, x, xc, has_face, lm, lpips_model,
                                                     conditions=tuple(CONDITIONS))):
                idx = int(batch["index"][i])
                rows.append({"index": idx, "source": ds.df.source[idx], "has_face": bool(has_face[i]),
                             "method": method, **r})
        for i, idx in enumerate(batch["index"].tolist()):
            save_image(x_cloak[i], out / "cloaks" / "images" / f"{idx:05d}.png")
        if len(shown) < 8:
            shown += list(zip(x.cpu(), x_cloak.cpu()))[: 8 - len(shown)]
        print(f"{len(rows) // 2}/{len(ds)} images", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)
    numeric = [c for c in df.columns if c not in ("index", "source", "has_face", "method")]
    by_method = df.groupby("method")[numeric].mean()
    by_source = df.groupby(["method", "source"])[numeric].mean()
    all_sources = by_method.assign(source="all").set_index("source", append=True)
    pd.concat([all_sources, by_source]).sort_index().to_csv(out / "summary.csv")   # rows: (method, source)

    clean, cloaked = torch.stack([a for a, _ in shown]), torch.stack([b for _, b in shown])
    save_image(torch.cat([clean, cloaked, ((cloaked - clean) * 10 + 0.5).clamp(0, 1)]),
               out / "samples.png", nrow=len(shown))

    # The edit-check images, cloaked at the editor's resolution.
    picked = stratified_indices(ds.df.source, args.edit_n)
    for size in args.edit_sizes:
        folder = out / "cloaks" / f"images_{size}"
        folder.mkdir(parents=True, exist_ok=True)
        for i in picked:
            x = resize(ds[i]["image"][None], size).to(device)
            save_image(cloak(G, x, mask_min)[0], folder / f"{i:05d}.png")

    shapes = [tuple(int(v) for v in s.split("x")) for s in args.speed]
    speed = benchmark(G, device, mask_min, shapes, args.speed_reps)
    device_name = torch.cuda.get_device_name() if device.type == "cuda" else "cpu"
    (out / "speed.json").write_text(json.dumps({"device": device_name, "results": speed}, indent=2))
    (out / "settings.json").write_text(json.dumps({**vars(args), "step": ck_step, "mask_min": mask_min,
                                                   "eps_255": G.eps * 255, "images": len(ds)}, indent=2))

    show = [c for c in by_method.columns if c.split("/")[-1].endswith(("_ratio", "_cos"))]
    print("\nlower = better protection; clean = the unprotected photo under the same condition")
    print(by_method[show].T.round(4).to_string())
    print(by_method[["psnr", "lpips", "linf_255"]].T.round(4).to_string())


if __name__ == "__main__":
    main()
