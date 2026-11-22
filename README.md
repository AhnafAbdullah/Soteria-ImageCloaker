# Soteria: real-time cloaking against deepfake abuse

A U-Net generator adds a bounded, texture-masked perturbation to a photo in a
single forward pass. It is trained against frozen surrogates: a Stable
Diffusion VAE encoder (editors), CLIP (semantics / VLMs) and ArcFace (identity).
Training uses EOT augmentation that includes real JPEG and messaging-app
round trips. Proposal: `i233012_i233018_Project_Proposal.pdf`.

**Random seed: `42`**, applied everywhere through `soteria.config.seed_everything`.

## Run order (Colab, T4)

| Notebook | What it does | Time |
|---|---|---|
| `colab/01_data.ipynb` | FFHQ-256 + COCO val2017 → 256 px, landmarks, fixed eval split, tar on Drive | ~30–45 min, once |
| `colab/02_pgd.ipynb` | surrogate sanity checks, then the PGD baseline at 8/255 and 16/255 (the go/no-go gate) | ~2–3 h |
| `colab/03_train.ipynb` | 300-step smoke run, then the main 20k-step run (resumable) | ~20 min + 4–6 h |

Set `REPO_URL` in each notebook's first cell, or leave it empty and copy this
folder to `MyDrive/soteria/code/Project-Soteria`. Runs, checkpoints and data
live under `MyDrive/soteria`, so a reclaimed session loses only the steps
since the last checkpoint. Rerunning `03_train.ipynb` resumes from `last.pt`.

## Layout

```
soteria/
  config.py       flat dataclass config; YAML + key=value overrides
  mask.py         texture mask M (local contrast) -> budget eps * M
  generator.py    smp U-Net (EfficientNet-B0), delta = eps * tanh(G(x)) * M
  augment.py      EOT: real-libjpeg (straight-through) + kornia diff-JPEG, platform sim, crop/blur/resize
  surrogates.py   frozen VAE / CLIP / face wrappers, ghost noise, differentiable 5-point alignment
  losses.py       targeted VAE loss + CLIP/face cosine + LPIPS hinge
  metrics.py      per-image protection + quality metrics (shared by val, PGD, eval)
  data.py         manifest-backed dataset with landmarks
  train.py        AMP, EMA, resume, time budget, TensorBoard
  pgd.py          per-image PGD+EOT baseline with the same objective
  infer.py        cloak one photo at any size (optionally pre-resized to the platform size)
configs/          base_256.yaml, smoke.yaml, pgd.yaml
scripts/          prepare_data.py, check_surrogates.py
tests/            CPU smoke tests with tiny stand-in surrogates (no downloads)
build_notebooks.py -> colab/*.ipynb
```

## Local development

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest -q
```

After editing `build_notebooks.py`, regenerate the notebooks with
`python build_notebooks.py`.

## Seen vs held-out models

Only keys in `soteria/surrogates.py:REGISTRY` are ever trained against.
Evaluation models (SD3 / Kandinsky VAEs, SigLIP, OpenAI ViT-L/14, BLIP-2,
FaceNet, AdaFace) must never be added there. That is what makes the transfer
results honest. InstructPix2Pix and SD1.5 inpainting reuse the SD1.5 VAE, so
they are reported as "unseen pipeline, seen encoder".

## Known gaps (planned next)

- `soteria/eval/` (editing, face-ID, VLM, robustness, speed), the Gradio app
  and API are not written yet.
- `platform_scale` / `platform_q` in the config are placeholders until the
  WhatsApp / Instagram round-trip calibration replaces them.
- Stretch items from the plan (surrogate ensemble beyond one per family,
  mid-layer feature losses, M-Attack-style targeted loss, YCbCr low-frequency
  δ, UNet SDS loss) are not implemented. The ensemble only needs extra keys
  in the config lists.
