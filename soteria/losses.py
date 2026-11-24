"""Cloak objective.

L(G) = w_vae * L_vae + w_clip * L_clip + w_face * L_face + w_lpips * hinge(LPIPS - budget)

- L_vae: targeted encoder attack (PhotoGuard, Salman et al. 2023). Pull the
  latent of T(x') toward the latent of a flat grey image, normalised by the
  clean distance so the term starts at ~1 for every VAE regardless of scale.
- L_clip, L_face: cosine similarity between the embedding of T(x') and that
  of the clean x. Minimising it pushes the cloaked photo away from its own
  semantics / identity.
- clip_floor / face_floor: per-image cosine below which that term stops
  pulling (clamp_min, zero gradient). Once a photo already looks like an
  unrelated image / a stranger, its share of the budget can go to the other
  terms instead. -1 = no floor.
- The face term only covers images that have a detected face.

Each call samples one surrogate per family (the stochastic ensemble), and the
same surrogates are returned so callers can log which ones were used.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from .augment import Crop


class CloakLoss:
    def __init__(self, cfg, surrogates: dict, lpips_model):
        self.cfg = cfg
        self.surrogates = surrogates
        self.lpips = lpips_model
        self._grey_latents: dict[tuple, torch.Tensor] = {}

    def sample(self) -> dict:
        return {fam: random.choice(models) for fam, models in self.surrogates.items() if models}

    def _vae(self, vae, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.vae_fp32:
            with torch.autocast(x.device.type, enabled=False):
                return vae(x.float())
        return vae(x)

    def _grey_latent(self, vae, x: torch.Tensor) -> torch.Tensor:
        key = (vae.key, *x.shape[-2:])
        if key not in self._grey_latents:
            with torch.no_grad(), vae.clean():
                grey = torch.full((1, 3, *x.shape[-2:]), 0.5, device=x.device)
                self._grey_latents[key] = self._vae(vae, grey).float()
        return self._grey_latents[key]

    def __call__(self, x, x_cloak, x_aug, crop: Crop, has_face, landmarks, chosen=None):
        """All images (B, 3, H, W) in [0, 1]; has_face (B,) bool; landmarks (B, 5, 2) px."""
        cfg = self.cfg
        chosen = chosen or self.sample()
        terms, logs = {}, {}
        h, w = x.shape[-2:]

        if "vae" in chosen and cfg.w_vae > 0:
            vae = chosen["vae"]
            with torch.no_grad(), vae.clean():
                z_clean = self._vae(vae, x).float()
            z_adv = self._vae(vae, x_aug).float()
            target = self._grey_latent(vae, x_aug)
            if target.shape[-2:] != z_adv.shape[-2:]:
                target = F.interpolate(target, size=z_adv.shape[-2:], mode="bilinear", align_corners=False)
            if z_clean.shape[-2:] != z_adv.shape[-2:]:
                z_clean = F.interpolate(z_clean, size=z_adv.shape[-2:], mode="bilinear", align_corners=False)
            d_adv = (z_adv - target).pow(2).flatten(1).mean(1)
            d_clean = (z_clean - target).pow(2).flatten(1).mean(1).clamp_min(1e-6)
            terms["vae"] = cfg.w_vae * (d_adv / d_clean).mean()

        if "clip" in chosen and cfg.w_clip > 0:
            clip = chosen["clip"]
            with torch.no_grad(), clip.clean():
                e_clean = clip(x)
            cos = F.cosine_similarity(clip(x_aug), e_clean, dim=-1)
            terms["clip"] = cfg.w_clip * cos.clamp_min(cfg.clip_floor).mean()
            logs["clip_cos"] = cos.mean().detach()

        if "face" in chosen and cfg.w_face > 0 and bool(has_face.any()):
            face, idx = chosen["face"], has_face.nonzero().squeeze(1)
            lm = landmarks[idx]
            with torch.no_grad(), face.clean():
                f_clean = face(x[idx], landmarks=lm)
            lm_aug = crop.apply_to_landmarks(lm, w, h)
            f_adv = face(x_aug[idx], landmarks=lm_aug)
            cos = F.cosine_similarity(f_adv, f_clean, dim=-1)
            terms["face"] = cfg.w_face * cos.clamp_min(cfg.face_floor).mean()
            logs["face_cos"] = cos.mean().detach()

        lp = self.lpips(x_cloak.float() * 2 - 1, x.float() * 2 - 1).flatten()
        terms["lpips"] = cfg.w_lpips * F.relu(lp - cfg.lpips_budget).mean()
        logs["lpips_raw"] = lp.mean().detach()

        total = sum(terms.values())
        logs.update({f"loss_{k}": v.detach() for k, v in terms.items()})
        logs["loss"] = total.detach()
        return total, logs, chosen


def load_lpips(device):
    import lpips

    model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    model.requires_grad_(False)
    return model
