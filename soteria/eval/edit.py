"""Edit check: does a cloak actually spoil a Stable Diffusion img2img edit?

    python -m soteria.eval.edit --pgd-root <root>/pgd --out <root>/edit_check [--make-missing] \
        [--runs pgd8 vae_only my_run=some_folder ...] [--n 16] [--sizes 256 512] [data_root=...]

The VAE ratio in the PGD tables is only a proxy. This runs the real thing:
SD1.5 img2img on the clean photo and on each cloaked copy, with the same
prompt and the *same seeded noise*, so any difference between the edits
comes from the cloak alone.

Two controls:
- "clean_again": the clean photo edited a second time in the same batch. Its
  scores are the noise floor (batched fp16 GPU kernels are not bit-exact, and
  20 denoising steps can amplify that). Anything near it is no effect at all.
- "random8": sign noise at the full 8/255 texture-masked budget, drawn at each
  size. A cloak only counts if it disrupts the edit clearly more than random8.

Resolution: a run folder (<folder>/images/) may also hold <folder>/images_<S>/
cloaks made at S px. Those are used as they are at size S (the generator can
cloak at any size); otherwise the 256 px cloak is upscaled, which smooths it.
The "native" column says which. Prompts: --prompts hair (default; keeps the
face visible so id_cos is meaningful) or sunglasses (the first edit check).

SD1.5 img2img encodes with the SD1.5 VAE, a seen surrogate, so this is
"unseen pipeline, seen encoder" (see README). Faces are scored with the
seen ArcFace too: fine for a go/no-go check, not for reported transfer.

Per size it writes grid_<size>.png (rows: images; columns: clean input, then
the edit of clean, random8 and each run) and metrics.csv / summary.csv:
- lpips_vs_clean_edit : LPIPS(edit(x'), edit(x)), higher = more disruption
- psnr_vs_clean_edit  : lower = more disruption
- id_cos              : ArcFace cos(edit(x'), x), faces only; lower = identity
                        lost. Its "clean" row is how much identity the edit
                        itself keeps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image

from .. import pgd
from ..config import Config
from ..data import CloakDataset
from ..losses import load_lpips
from ..mask import texture_mask
from ..pgd import stratified_indices
from ..surrogates import load_surrogates

SD_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
# "sunglasses" was the first edit check's prompt: the glasses cover the eyes, so
# even the clean edit loses identity (ArcFace) and id_cos means little. "hair"
# keeps the face visible, so a drop in id_cos is the cloak's doing.
PROMPT_SETS = {
    "hair": {"ffhq": "a photo of the same person with bright pink hair, high quality",
             "coco": "the same scene at night, cinematic lighting, high quality"},
    "sunglasses": {"ffhq": "a photo of a person with blond hair wearing sunglasses, high quality",
                   "coco": "the same scene at night, cinematic lighting, high quality"},
}
# Named PGD runs: folder under --pgd-root, and the overrides that produce it
# (identical to the 02_pgd notebook cells). With --make-missing, a run whose
# folder lacks the needed images is computed here, for just those images.
RUN_SPECS = {
    "pgd8": ("eps8", ["eps=0.03137"]),
    "vae_only": ("abl_vae_only", ["eps=0.03137", "w_clip=0", "w_face=0"]),
    "rebalanced": ("abl_rebalanced", ["eps=0.03137", "clip_floor=0.3", "face_floor=-0.1", "w_vae=4"]),
    "pgd16": ("eps16", ["eps=0.06275"]),
}


def load_editor(device):
    """SD1.5 img2img in fp16. The safety checker is off: it blacks out random
    outputs, which would corrupt a clean-vs-cloaked comparison."""
    from diffusers import StableDiffusionImg2ImgPipeline

    kw = dict(torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
    try:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(SD_MODEL, variant="fp16", **kw)
    except (OSError, ValueError):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(SD_MODEL, **kw)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def read_png(path: Path) -> torch.Tensor:
    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)


def resize(x: torch.Tensor, size: int) -> torch.Tensor:
    if x.shape[-1] == size:
        return x
    return F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)


@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pgd.yaml")
    ap.add_argument("--pgd-root", required=True, help="folder holding the PGD run folders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", nargs="*", default=list(RUN_SPECS),
                    help="names from RUN_SPECS, or name=folder under --pgd-root")
    ap.add_argument("--make-missing", action="store_true",
                    help="run PGD for named runs whose images are missing (~25 s/image on a T4)")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--sizes", type=int, nargs="*", default=[256, 512])
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompts", choices=list(PROMPT_SETS), default="hair")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = Config.load(args.config, args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = CloakDataset(cfg.data_root, "eval", cfg.image_size)
    picked = stratified_indices(ds.df.source, args.n)
    runs = {}
    for spec in args.runs:
        name, _, folder = spec.partition("=")
        folder = folder or RUN_SPECS[name][0]
        img_dir = Path(args.pgd_root) / folder / "images"
        missing = [i for i in picked if not (img_dir / f"{i:05d}.png").exists()]
        if missing and args.make_missing and name in RUN_SPECS and folder == RUN_SPECS[name][0]:
            print(f"== {name}: {len(missing)} images missing in {img_dir}, running PGD for {len(picked)}", flush=True)
            with torch.enable_grad():          # main() runs under no_grad; PGD needs gradients
                pgd.main(["--config", args.config, *args.overrides, *RUN_SPECS[name][1],
                          f"pgd_images={args.n}", f"out_dir={img_dir.parent}"])
            torch.cuda.empty_cache()
            missing = [i for i in picked if not (img_dir / f"{i:05d}.png").exists()]
        if missing:
            print(f"skipping {name}: {len(missing)} of {len(picked)} images missing in {img_dir}")
        else:
            runs[name] = img_dir
    methods = ["clean", "clean_again", "random8", *runs]
    print(f"{len(picked)} images, methods {methods}, sizes {args.sizes}", flush=True)

    editor = load_editor(device)
    lpips_model = load_lpips(device)
    face = load_surrogates(Config(vae=[], clip=[], face=cfg.face, ghost_std=0.0), device)["face"][0]
    rng = torch.Generator().manual_seed(args.seed)

    prompts = PROMPT_SETS[args.prompts]
    rows, grids = [], {s: [] for s in args.sizes}
    for k, i in enumerate(picked):
        item = ds[i]
        x = item["image"][None]
        cloaked = {name: read_png(d / f"{i:05d}.png")[None] for name, d in runs.items()}
        source = ds.df.source[i]

        for size in args.sizes:
            # Every input at this size. A run folder may hold images_<size>/
            # cloaks made at that size (the generator can); otherwise the 256 px
            # cloak is resized like the clean photo. random8 is drawn at each
            # size, so it is always a native control.
            x_size = resize(x, size)
            noise = torch.randint(0, 2, x_size.shape, generator=rng).float() * 2 - 1
            inputs, native = [], []
            for m in methods:
                own = runs[m].parent / f"images_{size}" / f"{i:05d}.png" if m in runs else None
                if m == "random8":
                    inputs.append((x_size + 8 / 255 * texture_mask(x_size, cfg.mask_min) * noise).clamp(0, 1))
                elif own is not None and own.exists():
                    inputs.append(read_png(own)[None])
                else:
                    inputs.append(x_size if m in ("clean", "clean_again") else resize(cloaked[m], size))
                native.append(m == "random8" or (own is not None and own.exists()) or size == x.shape[-1])
            xs = torch.cat(inputs).to(device)
            gens = [torch.Generator(device).manual_seed(args.seed + i) for _ in methods]
            edits = editor(prompt=[prompts[source]] * len(methods), image=xs, strength=args.strength,
                           num_inference_steps=args.steps, guidance_scale=args.guidance,
                           generator=gens, output_type="pt").images.float().clamp(0, 1)
            ref = edits[:1].expand_as(edits)
            lp = lpips_model(edits * 2 - 1, ref * 2 - 1).flatten()
            mse = (edits - ref).pow(2).flatten(1).mean(1)
            psnr = 10 * torch.log10(1 / mse.clamp_min(1e-10))
            id_cos = torch.full((len(methods),), float("nan"))
            if item["has_face"]:
                lm = (item["landmarks"] * size / x.shape[-1])[None].to(device)
                e_ref = face(xs[:1], landmarks=lm)
                e = face(edits, landmarks=lm.expand(len(methods), -1, -1))
                id_cos = torch.cosine_similarity(e, e_ref.expand_as(e), dim=-1).cpu()
            for j, m in enumerate(methods):
                rows.append({"index": i, "source": source, "has_face": bool(item["has_face"]), "size": size,
                             "method": m, "native": native[j], "lpips_vs_clean_edit": lp[j].item(),
                             "psnr_vs_clean_edit": min(psnr[j].item(), 100.0), "id_cos": id_cos[j].item()})
            grids[size].append(resize(torch.cat([xs[:1], edits]).cpu(), 256))
        print(f"{k + 1}/{len(picked)} images", flush=True)

    for size, tiles in grids.items():
        save_image(torch.cat(tiles), out / f"grid_{size}.png", nrow=len(methods) + 1, padding=2)
    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)
    cols = ["lpips_vs_clean_edit", "psnr_vs_clean_edit", "id_cos"]
    summary = df.groupby(["size", "method"], sort=False).agg(native=("native", "all"), **{c: (c, "mean") for c in cols})
    summary.to_csv(out / "summary.csv")
    (out / "settings.json").write_text(json.dumps({**vars(args), "methods": methods, "images": picked,
                                                   "model": SD_MODEL, "prompt_texts": prompts}, indent=2))
    print("grid columns: clean input | edit of: " + " | ".join(methods))
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
