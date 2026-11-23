"""Frozen surrogate models the cloak is trained against.

Every wrapper takes images in [0, 1] and owns its preprocessing (resize,
normalisation, face alignment), so callers never need to know model-specific
conventions. Preprocessing mismatches are the classic silent failure in
adversarial work: the loss still falls, but the cloak fools a misconfigured
copy of the model instead of the real one. Keeping them here, in one place,
makes them checkable (see scripts/check_surrogates.py).

Seen/unseen split: only keys in REGISTRY are ever used for training. Held-out
evaluation models live in soteria/eval and must never be added here.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------- ghost noise

class Surrogate(nn.Module):
    """Frozen model plus optional ghost-network noise.

    Ghost noise multiplies every conv/linear output by (1 + N(0, std)), a
    cheap way to sample a slightly different model each step so the cloak
    cannot overfit the exact surrogate weights (Li et al., "Learning
    Transferable Adversarial Examples via Ghost Networks", AAAI 2020).
    """

    family: str = ""

    def __init__(self):
        super().__init__()
        self.ghost_std = 0.0
        self._hooks = []

    def freeze(self, ghost_std: float = 0.0) -> "Surrogate":
        self.eval().requires_grad_(False)
        self.ghost_std = ghost_std
        if ghost_std > 0 and not self._hooks:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    self._hooks.append(m.register_forward_hook(self._ghost_hook))
        return self

    def _ghost_hook(self, module, inputs, output):
        if self.ghost_std > 0 and isinstance(output, torch.Tensor):
            return output * (1 + self.ghost_std * torch.randn_like(output))
        return output

    @contextlib.contextmanager
    def clean(self):
        """Disable ghost noise, e.g. when computing reference features of x."""
        std, self.ghost_std = self.ghost_std, 0.0
        try:
            yield self
        finally:
            self.ghost_std = std

    def train(self, mode: bool = True):
        # Surrogates stay in eval mode forever: BatchNorm statistics and
        # dropout must not change while the generator trains.
        return super().train(False)


# ------------------------------------------------------------------------ VAE

class VAESurrogate(Surrogate):
    """Latent-diffusion image encoder; output is the posterior mean."""

    family = "vae"

    def __init__(self, repo: str, subfolder: str | None = None):
        super().__init__()
        from diffusers import AutoencoderKL

        vae = AutoencoderKL.from_pretrained(repo, subfolder=subfolder)
        vae.decoder = None      # encode() never touches it; frees ~half the weights
        self.vae = vae

    def forward(self, x: torch.Tensor, **_) -> torch.Tensor:
        return self.vae.encode(x * 2 - 1).latent_dist.mean


# ----------------------------------------------------------------------- CLIP

OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)


class CLIPSurrogate(Surrogate):
    """open_clip image tower; output is the L2-normalised image embedding."""

    family = "clip"

    def __init__(self, arch: str, pretrained: str):
        super().__init__()
        import open_clip

        model = open_clip.create_model(arch, pretrained=pretrained)
        self.visual = model.visual
        size = self.visual.image_size
        self.size = tuple(size) if isinstance(size, (tuple, list)) else (size, size)
        cfg = getattr(self.visual, "preprocess_cfg", {}) or {}
        self.register_buffer("mean", torch.tensor(cfg.get("mean", OPENAI_MEAN)).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(cfg.get("std", OPENAI_STD)).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor, **_) -> torch.Tensor:
        x = F.interpolate(x, size=self.size, mode="bicubic", align_corners=False, antialias=True)
        return F.normalize(self.visual((x - self.mean) / self.std).float(), dim=-1)


# ----------------------------------------------------------------------- Face

# InsightFace 5-point template for 112x112 crops: left eye, right eye, nose,
# left mouth corner, right mouth corner (image left/right, not anatomical).
ARCFACE_TEMPLATE = torch.tensor([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041],
])


def similarity_transform(src: torch.Tensor, dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched Umeyama fit: dst ~= src @ A.T + t, with A = scale * rotation.

    src, dst: (B, N, 2). Returns A (B, 2, 2), t (B, 2).
    """
    n = src.shape[1]
    mu_s, mu_d = src.mean(1, keepdim=True), dst.mean(1, keepdim=True)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.transpose(1, 2) @ sc / n
    u, s, vh = torch.linalg.svd(cov)
    d = torch.sign(torch.det(u @ vh))
    diag = torch.stack([torch.ones_like(d), d], dim=-1)
    rot = u @ torch.diag_embed(diag) @ vh
    var_s = sc.pow(2).sum((1, 2)) / n
    scale = (s * diag).sum(-1) / var_s
    a = scale[:, None, None] * rot
    t = mu_d.squeeze(1) - (a @ mu_s.transpose(1, 2)).squeeze(-1)
    return a, t


def align_faces(x: torch.Tensor, landmarks: torch.Tensor, size: int = 112) -> torch.Tensor:
    """Differentiable 5-point alignment to the ArcFace template.

    landmarks: (B, 5, 2) in pixel coordinates of x (pixel centres at integers,
    the cv2 / detector convention). The warp is fixed per image, so gradients
    flow to x through grid_sample.

    Runs in fp32 with autocast off: under AMP the matmuls would drop to fp16,
    and SVD / det / inv have no fp16 kernels (and 5-point geometry should not
    be fitted at fp16 precision anyway). The face net after it still uses AMP.
    """
    b, _, h, w = x.shape
    with torch.autocast(x.device.type, enabled=False):
        template = ARCFACE_TEMPLATE.to(x.device) * (size / 112)
        a, t = similarity_transform(landmarks.float(), template.expand(b, -1, -1))
        a_inv = torch.linalg.inv(a)
        to_norm = torch.diag(torch.tensor([2 / w, 2 / h], device=x.device))
        half = size / 2
        lin = to_norm @ a_inv * half
        centre = torch.full_like(t, half - 0.5)
        off = (to_norm @ a_inv @ (centre - t).unsqueeze(-1)).squeeze(-1) \
            + torch.tensor([1 / w - 1, 1 / h - 1], device=x.device)
        theta = torch.cat([lin, off.unsqueeze(-1)], dim=-1)
        grid = F.affine_grid(theta, (b, 3, size, size), align_corners=False)
        return F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="border", align_corners=False)


def conv3x3(cin, cout, stride=1):
    return nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)


class IBasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin, eps=1e-5)
        self.conv1 = conv3x3(cin, cout)
        self.bn2 = nn.BatchNorm2d(cout, eps=1e-5)
        self.prelu = nn.PReLU(cout)
        self.conv2 = conv3x3(cout, cout, stride)
        self.bn3 = nn.BatchNorm2d(cout, eps=1e-5)
        self.downsample = downsample

    def forward(self, x):
        out = self.bn3(self.conv2(self.prelu(self.bn2(self.conv1(self.bn1(x))))))
        return out + (self.downsample(x) if self.downsample is not None else x)


class IResNet(nn.Module):
    """InsightFace iresnet, with state-dict keys matching arcface_torch."""

    def __init__(self, layers, num_features=512):
        super().__init__()
        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu = nn.PReLU(64)
        cin, stages = 64, []
        for cout, n in zip((64, 128, 256, 512), layers):
            down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=2, bias=False), nn.BatchNorm2d(cout, eps=1e-5))
            blocks = [IBasicBlock(cin, cout, 2, down)] + [IBasicBlock(cout, cout) for _ in range(n - 1)]
            stages.append(nn.Sequential(*blocks))
            cin = cout
        self.layer1, self.layer2, self.layer3, self.layer4 = stages
        self.bn2 = nn.BatchNorm2d(512, eps=1e-5)
        self.dropout = nn.Dropout(0.0)
        self.fc = nn.Linear(512 * 7 * 7, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-5)

    def forward(self, x):
        x = self.prelu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.dropout(torch.flatten(self.bn2(x), 1))
        return self.features(self.fc(x))


class FaceSurrogate(Surrogate):
    """Face-ID embedder on 112x112 aligned crops, normalised with mean = std = 0.5."""

    family = "face"

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor, landmarks: torch.Tensor | None = None, **_) -> torch.Tensor:
        if landmarks is None:
            raise ValueError("face surrogate needs (B, 5, 2) landmarks")
        crop = align_faces(x, landmarks)
        return F.normalize(self.net((crop - 0.5) / 0.5).float(), dim=-1)


def cvlface_iresnet100(repo: str) -> nn.Module:
    """CVLface checkpoints store the insightface iresnet under 'model.net.'."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    state = load_file(hf_hub_download(repo, "model.safetensors"))
    prefix = "model.net."
    state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    net = IResNet([3, 13, 30, 3])
    net.load_state_dict(state, strict=True)
    return net


def timm_face(repo: str) -> nn.Module:
    import timm

    return timm.create_model(f"hf-hub:{repo}", pretrained=True)


# ------------------------------------------------------------------- registry

# Keys usable in Config.vae / .clip / .face. FLUX.1-schnell is gated on the Hub
# (accept the licence once, then set HF_TOKEN).
REGISTRY = {
    "sd15": lambda: VAESurrogate("stabilityai/sd-vae-ft-mse"),
    "sdxl": lambda: VAESurrogate("madebyollin/sdxl-vae-fp16-fix"),
    "flux": lambda: VAESurrogate("black-forest-labs/FLUX.1-schnell", subfolder="vae"),
    "vit_b16_laion": lambda: CLIPSurrogate("ViT-B-16", "laion2b_s34b_b88k"),
    "vit_b32_openai": lambda: CLIPSurrogate("ViT-B-32", "openai"),
    "arcface_ir101": lambda: FaceSurrogate(cvlface_iresnet100("minchul/cvlface_arcface_ir101_webface4m")),
    "cosface_convnext_nano": lambda: FaceSurrogate(timm_face("gaunernst/convnext_nano.cosface_ms1mv3")),
}


def load_surrogates(cfg, device) -> dict[str, list[Surrogate]]:
    """Returns {family: [surrogate, ...]} for the keys listed in cfg."""
    out = {}
    for family in ("vae", "clip", "face"):
        models = []
        for key in getattr(cfg, family):
            model = REGISTRY[key]().freeze(cfg.ghost_std).to(device)
            if model.family != family:
                raise ValueError(f"{key} is a {model.family} surrogate, listed under {family}")
            model.key = key
            models.append(model)
        out[family] = models
    return out
