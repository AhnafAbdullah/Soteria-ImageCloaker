"""Build data_root/manifest.csv from raw FFHQ-256 and COCO folders.

    python scripts/prepare_data.py --ffhq /content/raw/ffhq --coco /content/raw/val2017 \
        --out /content/data/prepared

- FFHQ images are copied as-is (already 256 px, already face-aligned).
- COCO images are resized (shorter side = 256) and centre-cropped, then stored
  as JPEG q95 so the training set is not itself heavily compressed.
- 5-point landmarks come from InsightFace's RetinaFace detector (ONNX). They
  are only used to build the fixed alignment warp for the face surrogate, so
  the detector itself never needs to be differentiable.
- The eval split (300 FFHQ + 100 COCO by default) is drawn once with SEED and
  never trained on.
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soteria.config import SEED  # noqa: E402
from soteria.data import LM_COLS  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MIN_FACE_PX = 48
MIN_SCORE = 0.6


def list_images(root: Path, limit: int | None) -> list[Path]:
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)
    return files[:limit] if limit else files


def centre_crop(img: Image.Image, size: int) -> Image.Image:
    scale = size / min(img.size)
    img = img.resize((max(size, round(img.width * scale)), max(size, round(img.height * scale))), Image.BICUBIC)
    left, top = (img.width - size) // 2, (img.height - size) // 2
    return img.crop((left, top, left + size, top + size))


def make_detector():
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"],
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(256, 256))
    return app


def detect(app, path: Path):
    """Largest confident face -> (5, 2) landmarks, or None."""
    bgr = np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1]
    faces = [f for f in app.get(bgr)
             if f.det_score >= MIN_SCORE and min(f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]) >= MIN_FACE_PX]
    if not faces:
        return None
    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return np.asarray(best.kps, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffhq", type=Path, required=True)
    ap.add_argument("--coco", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--ffhq-limit", type=int, default=None)
    ap.add_argument("--coco-limit", type=int, default=None)
    ap.add_argument("--eval-ffhq", type=int, default=300)
    ap.add_argument("--eval-coco", type=int, default=100)
    args = ap.parse_args()

    (args.out / "ffhq").mkdir(parents=True, exist_ok=True)
    (args.out / "coco").mkdir(parents=True, exist_ok=True)
    records = []

    for src in tqdm(list_images(args.ffhq, args.ffhq_limit), desc="ffhq"):
        dst = args.out / "ffhq" / src.name
        if not dst.exists():
            with Image.open(src) as img:
                if img.size == (args.size, args.size):
                    shutil.copyfile(src, dst)
                else:
                    img.convert("RGB").resize((args.size, args.size), Image.BICUBIC).save(dst, quality=95)
        records.append({"path": f"ffhq/{dst.name}", "source": "ffhq"})

    for src in tqdm(list_images(args.coco, args.coco_limit), desc="coco"):
        dst = args.out / "coco" / (src.stem + ".jpg")
        if not dst.exists():
            with Image.open(src) as img:
                centre_crop(img.convert("RGB"), args.size).save(dst, quality=95)
        records.append({"path": f"coco/{dst.name}", "source": "coco"})

    app = make_detector()
    for r in tqdm(records, desc="landmarks"):
        kps = detect(app, args.out / r["path"])
        r["has_face"] = kps is not None
        flat = kps.reshape(-1) if kps is not None else [np.nan] * 10
        r.update(dict(zip(LM_COLS, map(float, flat))))

    df = pd.DataFrame(records)
    df["split"] = "train"
    rng = random.Random(SEED)
    for source, n in (("ffhq", args.eval_ffhq), ("coco", args.eval_coco)):
        idx = df.index[df.source == source].tolist()
        df.loc[rng.sample(idx, min(n, len(idx))), "split"] = "eval"
    df.to_csv(args.out / "manifest.csv", index=False)

    print(df.groupby(["source", "split"]).agg(n=("path", "size"), faces=("has_face", "sum")).to_string())


if __name__ == "__main__":
    main()
