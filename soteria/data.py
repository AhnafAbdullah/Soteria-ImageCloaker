"""Training / held-out images from the prepared manifest.

scripts/prepare_data.py writes data_root/manifest.csv with one row per image:
    path, source (ffhq | coco), split (train | eval), has_face, lm0x, lm0y, ..., lm4x, lm4y
Landmarks are 5-point (InsightFace order) in pixels of the stored image. The
eval split is fixed once by SEED and is never trained on.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

LM_COLS = [f"lm{i}{a}" for i in range(5) for a in "xy"]
# Mirroring swaps image-left and image-right points: eyes 0<->1, mouth 3<->4.
FLIP_ORDER = [1, 0, 2, 4, 3]


class CloakDataset(Dataset):
    def __init__(self, data_root: str | Path, split: str, image_size: int = 256,
                 flip: bool = False, sources: tuple[str, ...] = ("ffhq", "coco")):
        self.root = Path(data_root)
        df = pd.read_csv(self.root / "manifest.csv")
        self.df = df[(df.split == split) & df.source.isin(sources)].reset_index(drop=True)
        self.size = image_size
        self.flip = flip

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.root / row.path).convert("RGB")
        lm = row[LM_COLS].to_numpy(dtype=np.float32).reshape(5, 2)
        if img.size != (self.size, self.size):
            sx, sy = self.size / img.width, self.size / img.height
            img = img.resize((self.size, self.size), Image.BICUBIC)
            lm = lm * np.array([sx, sy], dtype=np.float32)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255).permute(2, 0, 1)
        lm = torch.from_numpy(lm)
        if self.flip and torch.rand(()) < 0.5:
            x = x.flip(-1)
            lm = lm[FLIP_ORDER]
            lm[:, 0] = self.size - 1 - lm[:, 0]
        return {"image": x, "has_face": bool(row.has_face), "landmarks": lm, "index": i}
