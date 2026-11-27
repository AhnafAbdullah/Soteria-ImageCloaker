# Soteria: real-time cloaking against deepfake abuse

A U-Net generator adds a bounded, texture-masked perturbation to a photo in a
single forward pass. It is trained against frozen surrogates: a Stable
Diffusion VAE encoder (editors), CLIP (semantics / VLMs) and ArcFace (identity).
Training uses EOT augmentation that includes real JPEG and messaging-app
round trips. Proposal: `i233012_i233018_Project_Proposal.pdf`.

**Random seed: `42`**, applied everywhere through `soteria.config.seed_everything`.

## Run order (Colab or Kaggle, T4)

| Notebook | What it does | Time |
|---|---|---|
| `colab/01_data.ipynb` | FFHQ-256 + COCO val2017 → 256 px, landmarks, fixed eval split, `prepared.tar` | ~30–45 min, once per platform |
| `colab/02_pgd.ipynb` | surrogate sanity checks, PGD baseline at 8/255 and 16/255 (the go/no-go gate), VAE ablation, SD1.5 img2img edit check | ~2.5 h + 50 min + 30 min (edit check alone: ~1 h if it must make its own cloaks) |
| `colab/02b_edit_check.ipynb` | the edit check alone, for Kaggle "Save & Run All" (runs in the background, output kept) | ~1–1.5 h |
| `colab/03_train.ipynb` | 300-step smoke run (measures speed, projects the main run), progress viewer, main 20k-step run (resumable) | ~20 min + ~20 GPU-h (projected from PGD speed; the smoke run measures it) |

Every code cell is self-contained: it detects Colab or Kaggle, pulls the
code, installs requirements and finds or unpacks the data (seconds when
already done; `soteria/runtime.py`). After a disconnect, rerun only the cell
you need. Give each person and each experiment its own `RUN_NAME`; the
training cell refuses to resume with a changed loss config.

| | Colab | Kaggle |
|---|---|---|
| Results (`ROOT`) | Drive `MyDrive/soteria` | `/kaggle/working/soteria`, kept only by **Save Version → Save & Run All** |
| Data | `ROOT/data/prepared.tar` | attach the 01_data output (Add Input → Your Work) |
| FFHQ for 01 | downloaded with `kaggle.json` | attach `xhlulu/flickrfaceshq-dataset-nvidia-resized-256px` |
| Resume training | `ROOT/runs/<RUN_NAME>/last.pt` on Drive | attach the previous version's output; its run folder is copied back |
| Session budget | 330 min | 600 min |
| Secrets (`HF_TOKEN`, `GITHUB_TOKEN`) | Colab secrets | Add-ons → Secrets |

Kaggle also needs Settings → Accelerator **GPU T4 x2** and Internet **On**.

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
  runtime.py      notebook setup for Colab and Kaggle, shared by every cell
  eval/edit.py    SD1.5 img2img edit check: clean vs cloaked edits, same seed, with controls
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

- `soteria/eval/` has only the SD1.5 img2img edit check (`edit.py`); held-out face-ID, VLM,
  robustness and speed evals, the Gradio app
  and API are not written yet.
- `platform_scale` / `platform_q` in the config are placeholders until the
  WhatsApp / Instagram round-trip calibration replaces them.
- Stretch items from the plan are not implemented yet. In priority order:
  1. BlurGuard-style per-region blur of δ, plus a spectral-naturalness loss
     (NeurIPS 2025).
  2. Surrogate ensemble beyond one per family. This only needs extra keys in
     the config lists.
  3. DCT-domain cloak head (DCT-Shield, ICCV 2025).
  4. ImageProtector-style refusal loss on Qwen2-VL (ACL 2026).
  5. Mid-layer feature losses and an M-Attack-style targeted loss.
  6. UNet SDS loss.
- Purification evaluation must include Noisy Upscaling (Hönig et al., 2025).
