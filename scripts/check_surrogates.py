"""Sanity-check every surrogate on clean held-out images before attacking it.

    python scripts/check_surrogates.py --config configs/pgd.yaml [key=value ...]

A misconfigured surrogate (wrong normalisation, colour order, alignment) still
gives a loss that falls during training, so problems only show up as poor
transfer weeks later. These checks catch them up front:
- face: an image vs its own mirror should match (cos > 0.5), different people
  should not (mean cos < 0.3); aligned crops are saved for eyeballing.
- clip: an image vs a mildly blurred copy should be near-identical, and
  different images clearly less similar.
- vae: latents should be finite with the expected shape (C, H/8, W/8).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kornia.filters import gaussian_blur2d  # noqa: E402

from soteria.config import Config  # noqa: E402
from soteria.data import FLIP_ORDER, CloakDataset  # noqa: E402
from soteria.surrogates import align_faces, load_surrogates  # noqa: E402


def off_diagonal_mean(sim: torch.Tensor) -> float:
    n = sim.shape[0]
    return float((sim.sum() - sim.diag().sum()) / (n * (n - 1)))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pgd.yaml")
    ap.add_argument("--out", default="surrogate_check")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = Config.load(args.config, args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = CloakDataset(cfg.data_root, "eval", cfg.image_size)
    faces = [i for i in range(len(ds)) if ds.df.has_face[i]][:16]
    batch = torch.utils.data.default_collate([ds[i] for i in faces])
    x, lm = batch["image"].to(device), batch["landmarks"].to(device)
    w = x.shape[-1]
    x_flip = x.flip(-1)
    lm_flip = lm[:, FLIP_ORDER].clone()
    lm_flip[..., 0] = w - 1 - lm_flip[..., 0]

    save_image(align_faces(x, lm), out / "aligned_faces.png", nrow=8)
    ok = True
    surrogates = load_surrogates(cfg, device)

    for face in surrogates["face"]:
        e, e_flip = face(x, landmarks=lm), face(x_flip, landmarks=lm_flip)
        same = float(torch.cosine_similarity(e, e_flip).mean())
        diff = off_diagonal_mean(e @ e.T)
        good = same > 0.5 and diff < 0.3
        ok &= good
        print(f"[{'ok' if good else 'FAIL'}] {face.key}: mirror cos {same:.3f} (>0.5), other people {diff:.3f} (<0.3)")

    for clip in surrogates["clip"]:
        e = clip(x)
        e_blur = clip(gaussian_blur2d(x, (5, 5), (0.8, 0.8)))
        same = float(torch.cosine_similarity(e, e_blur).mean())
        diff = off_diagonal_mean(e @ e.T)
        good = same > 0.9 and same - diff > 0.1
        ok &= good
        print(f"[{'ok' if good else 'FAIL'}] {clip.key}: blurred copy cos {same:.3f} (>0.9), other images {diff:.3f}")

    for vae in surrogates["vae"]:
        z = vae(x.float())
        good = bool(torch.isfinite(z).all()) and z.shape[-1] == w // 8
        ok &= good
        print(f"[{'ok' if good else 'FAIL'}] {vae.key}: latent {tuple(z.shape[1:])}, std {float(z.std()):.3f}")

    print(f"aligned crops saved to {out / 'aligned_faces.png'}: eyes and mouth should sit at the same spots")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
