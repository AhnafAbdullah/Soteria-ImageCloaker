"""Generate the thin Colab notebooks in colab/.

The notebooks hold no model code: they mount Drive, fetch this repo, install
requirements and call the package. Edit the package, push, and the next
notebook run picks it up; edit this file and rerun it to change the notebooks:

    python build_notebooks.py

Every GPU cell starts with the same short setup block (see boot()), so any
cell can be run on its own on a fresh runtime: after a disconnect, rerun only
the cell you need. The block is idempotent and takes seconds once a runtime
is set up; the heavy lifting lives in soteria/colab.py.
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "colab"
REPO_URL = "https://github.com/AhnafAbdullah/Soteria-ImageCloaker.git"


def md(s):
    return {"cell_type": "markdown", "metadata": {},
            "source": s.strip("\n").splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": s.strip("\n").splitlines(keepends=True)}


def write(name, cells):
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}}}
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote colab/{name}")


def boot(data=True, gpu=True, requirements=("requirements.txt",)):
    """Setup block that makes a cell self-contained."""
    return f'''
# ---- setup: lets this cell run on its own; safe to rerun --------------------
REPO_URL   = "{REPO_URL}"
BRANCH     = "main"
DRIVE_ROOT = "/content/drive/MyDrive/soteria"   # data tar, runs and results live here
import importlib, os, subprocess, sys
from google.colab import drive, userdata
if not os.path.isdir("/content/drive/MyDrive"):
    drive.mount("/content/drive")
CODE = "/content/Project-Soteria"
def _git(*args):
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    print(r.stdout + r.stderr, end=""); r.check_returncode()
if os.path.isdir(CODE):
    _git("-C", CODE, "pull", "-q", "--ff-only")
else:
    try:
        _url = REPO_URL.replace("https://", f"https://{{userdata.get('GITHUB_TOKEN')}}@")
    except Exception:          # no GITHUB_TOKEN secret: fine for a public repo
        _url = REPO_URL
    _git("clone", "-q", "-b", BRANCH, _url, CODE)
os.chdir(CODE); sys.path.insert(0, CODE)
_git("log", "-1", "--format=code: %h %s")
import soteria.colab; importlib.reload(soteria.colab)
soteria.colab.prepare(DRIVE_ROOT, data={data}, gpu={gpu}, requirements={requirements!r})
# -----------------------------------------------------------------------------
'''.strip("\n")


def cell(body, **boot_kw):
    return code(boot(**boot_kw) + "\n\n" + body.strip("\n"))


# ---------------------------------------------------------------- 01 data
write("01_data.ipynb", [
    md(r"""
# 01 · Prepare data (run once)

Downloads FFHQ-256 (70k aligned faces, Kaggle) and COCO val2017 (5k scenes),
normalises everything to 256 px, detects 5-point face landmarks, fixes the
held-out eval split (300 FFHQ + 100 COCO, seed 42), and stores the result as
one tar on Drive. Later notebooks only unpack that tar.

Needs `kaggle.json` (kaggle.com → Settings → API → Create New Token), uploaded
when the cell asks. About 30–45 minutes on a T4 runtime. Run the cells in order.
"""),
    cell(r"""
from pathlib import Path
from google.colab import files
if not Path("/root/.kaggle/kaggle.json").exists():
    up = files.upload()                      # choose kaggle.json
    Path("/root/.kaggle").mkdir(exist_ok=True)
    Path("/root/.kaggle/kaggle.json").write_bytes(up["kaggle.json"])
    os.chmod("/root/.kaggle/kaggle.json", 0o600)
""", data=False, requirements=("requirements.txt", "requirements-data.txt")),
    code(r"""
!mkdir -p /content/raw/ffhq /content/raw/coco
!kaggle datasets download -d xhlulu/flickrfaceshq-dataset-nvidia-resized-256px -p /content/raw/ffhq --unzip -q
!wget -q http://images.cocodataset.org/zips/val2017.zip -O /content/raw/coco/val2017.zip
!unzip -q /content/raw/coco/val2017.zip -d /content/raw/coco && rm /content/raw/coco/val2017.zip
!find /content/raw/ffhq -type f | wc -l && find /content/raw/coco -type f | wc -l
"""),
    code(r"""
!python scripts/prepare_data.py --ffhq /content/raw/ffhq --coco /content/raw/coco/val2017 --out /content/data/prepared
"""),
    code(r"""
!mkdir -p {DRIVE_ROOT}/data
!tar -cf {DRIVE_ROOT}/data/prepared.tar -C /content/data prepared
!ls -lh {DRIVE_ROOT}/data
"""),
])


# ----------------------------------------------------------------- 02 pgd
SUMMARY = r"""
import json, pandas as pd
for run in ("eps8", "eps16"):
    s = json.load(open(f"{DRIVE_ROOT}/pgd/{run}/summary.json"))
    print(f"== {run}: {s['sec_per_image']:.1f} s/image")
    display(pd.DataFrame(s["by_method"]).round(4))
"""

COMPARE = r"""
import pandas as pd
runs = {"baseline": "eps8", "vae_only": "abl_vae_only", "rebalanced": "abl_rebalanced"}
dfs = {name: pd.read_csv(f"{DRIVE_ROOT}/pgd/{d}/metrics.csv") for name, d in runs.items()}
same = set(dfs["vae_only"]["index"])
cols = [c for c in dfs["baseline"].columns if c.startswith("jpeg75/") and not c.endswith("_shift")]
rows = {"clean": dfs["baseline"].query("method == 'clean'")}
rows.update({name: df.query("method == 'pgd'") for name, df in dfs.items()})
table = pd.DataFrame({name: df[df["index"].isin(same)][cols + ["lpips", "psnr"]].mean() for name, df in rows.items()})
print(f"{len(same)} images, after a real JPEG q75 round trip (lower = better protection)")
display(table.round(4))
"""

write("02_pgd.ipynb", [
    md(r"""
# 02 · Surrogate checks, PGD baseline and edit check (the go/no-go gate)

Every code cell sets itself up, so any one can be run alone on a fresh runtime.

1. Check that every surrogate behaves on clean images (identity matches its
   mirror, CLIP is stable under blur, VAE latents are sane). A FAIL here means
   a preprocessing bug: fix it before anything else.
2. Run per-image PGD+EOT at 8/255 and 16/255 on 200 held-out images.

**Gate:** after `jpeg75`, the PGD rows should show clearly lower `*_cos` and
`*_ratio` than the `clean` rows. If 8/255 barely moves them, fix the loss or
the augmentations before training the generator.
"""),
    cell(r"""
!python scripts/check_surrogates.py --config configs/pgd.yaml --out {DRIVE_ROOT}/surrogate_check
from IPython.display import Image as Show
Show(f"{DRIVE_ROOT}/surrogate_check/aligned_faces.png")
"""),
    cell(r"""
!python -m soteria.pgd --config configs/pgd.yaml eps=0.03137 out_dir={DRIVE_ROOT}/pgd/eps8
"""),
    cell(r"""
!python -m soteria.pgd --config configs/pgd.yaml eps=0.06275 out_dir={DRIVE_ROOT}/pgd/eps16
"""),
    cell(SUMMARY, data=False, gpu=False, requirements=()),
    md(r"""
## Ablation: is the VAE term starved, or just hard? (8/255, 64 images, ~25 min each)

In the baseline, CLIP and face are broken at 8/255 but the VAE ratio only
falls 0.985 → 0.897 after JPEG.

- **vae_only** turns the CLIP and face terms off. It is the most the VAE can
  move at 8/255 under this EOT. If it also stalls near 0.85, the VAE
  objective itself is the problem, not the balance between terms.
- **rebalanced** keeps all three, but stops the CLIP / face terms pulling once
  an image is past an unrelated photo / a stranger (`clip_floor=0.3`,
  `face_floor=-0.1`, from the surrogate check: 0.425 and -0.002), and gives
  the VAE 4x weight.

The 64 images are the first of the 200 above, so the last cell compares
all three on exactly the same images.
"""),
    cell(r"""
!python -m soteria.pgd --config configs/pgd.yaml eps=0.03137 pgd_images=64 w_clip=0 w_face=0 out_dir={DRIVE_ROOT}/pgd/abl_vae_only
"""),
    cell(r"""
!python -m soteria.pgd --config configs/pgd.yaml eps=0.03137 pgd_images=64 clip_floor=0.3 face_floor=-0.1 w_vae=4 out_dir={DRIVE_ROOT}/pgd/abl_rebalanced
"""),
    cell(COMPARE, data=False, gpu=False, requirements=()),
    md(r"""
## Edit check: does a cloak actually spoil a Stable Diffusion edit? (~30 min)

The VAE ratio is only a proxy. This runs SD1.5 img2img (strength 0.5) on 16 of
the images above (12 faces, 4 scenes): the clean photo, two controls and every
PGD run, all with the same prompt and the same seeded noise. It runs at 256 px
(the size the cloaks were made at) and 512 px (SD1.5's native size, the cloak
upscaled).

Controls:
- `clean_again`: the clean photo edited a second time. Its score is the
  noise floor of the GPU itself; anything near it means no effect.
- `random8`: random noise at the full 8/255 budget.

Read the results like this:
- `lpips_vs_clean_edit`: how different the edit of the cloaked photo is from
  the edit of the clean one. A cloak works only if this is **clearly above
  `random8`**, not just above 0.
- `id_cos`: ArcFace similarity of the edit to the original person. The
  `clean` row is what the edit keeps by itself; lower than that = identity lost.
- `grid_*.png`: rows are images; columns are the clean input, then the edit of
  clean, clean_again, random8, pgd8, vae_only, rebalanced, pgd16. Look for edits that are
  visibly wrecked (smears, grey patches, wrong content), not just different.

Writes `{DRIVE_ROOT}/edit_check/`: grid_256.png, grid_512.png, metrics.csv, summary.csv.
"""),
    cell(r"""
!python -m soteria.eval.edit --pgd-root {DRIVE_ROOT}/pgd --out {DRIVE_ROOT}/edit_check
from IPython.display import Image as Show, display
for size in (256, 512):
    print(f"grid {size}px: clean input | edit of clean, clean_again, random8, pgd8, vae_only, rebalanced, pgd16")
    display(Show(f"{DRIVE_ROOT}/edit_check/grid_{size}.png", width=900))
"""),
])


# --------------------------------------------------------------- 03 train
TRAIN_SETTINGS = r'''
# ---- this run ----------------------------------------------------------------
RUN_NAME  = "base_256"     # one folder per experiment AND per person, e.g. "ahnaf_base_256"
CONFIG    = "configs/base_256.yaml"
OVERRIDES = ""             # e.g. "clip_floor=0.3 face_floor=-0.1 w_vae=4". Must not change
                           # while resuming one RUN_NAME: train.py refuses a mismatch.
OUT = f"{DRIVE_ROOT}/runs/{RUN_NAME}"
'''

PROGRESS = TRAIN_SETTINGS + r'''
import glob, json, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Image as Show, display
status = json.load(open(f"{OUT}/status.json"))
print(json.dumps({k: v for k, v in status.items() if k != "last_val"}, indent=1))
log = pd.read_csv(f"{OUT}/log.csv")
fig, axes = plt.subplots(1, 3, figsize=(16, 3.5))
log.plot(x="step", y=[c for c in log if c.startswith("loss_")], ax=axes[0], title="train loss terms")
log.plot(x="step", y=[c for c in ("lpips_raw", "clip_cos", "face_cos") if c in log], ax=axes[1], title="train")
try:
    val = pd.read_csv(f"{OUT}/val.csv")
    val.plot(x="step", y=[c for c in val if c.startswith("jpeg75/") and not c.endswith("_shift")],
             ax=axes[2], title="val after JPEG q75 (lower = better)")
    display(val.tail(3).set_index("step").T.round(4))
except FileNotFoundError:
    axes[2].set_title("no validation yet")
plt.tight_layout(); plt.show()
samples = sorted(glob.glob(f"{OUT}/samples/*.png"))
if samples:
    print(samples[-1], "(rows: clean / cloaked / |delta| x10)")
    display(Show(samples[-1]))
'''

write("03_train.ipynb", [
    md(r"""
# 03 · Train the cloak generator

**Every code cell is self-contained.** On a fresh or reconnected runtime you
can run any one of them directly; you never need to rerun the cells above it.

- **Smoke run** (~20–30 min): 300 steps. Checks the loop works, and measures
  the real speed so it can project how long the main run will take.
- **Progress** (no GPU needed): status, ETA, loss and validation curves and the
  latest samples, read from Drive. Run it whenever, even while training runs
  in another session.
- **Main run**: checkpoints to Drive every 500 steps and stops itself before
  the Colab session limit (`budget_min` in the config). **After a disconnect or
  when it stops on the time budget, rerun only this cell**: it resumes from
  `last.pt` and loses at most the steps since the last checkpoint.

What it saves in `MyDrive/soteria/runs/<RUN_NAME>/`: `status.json` (step,
GPU-minutes, ETA, latest validation), `log.csv` (training curves), `val.csv`
(protection metrics every 500 steps), `samples/`, `generator_ema.pt` (the model
to use) and `last.pt` (full resume state).

Each person and each experiment needs its own `RUN_NAME`. Two runs writing to
one folder would resume from each other's checkpoints.
"""),
    cell(r"""
!python -m soteria.train --config configs/smoke.yaml out_dir={DRIVE_ROOT}/runs/smoke
import json, glob, pandas as pd, yaml
from IPython.display import Image as Show, display
log = pd.read_csv(f"{DRIVE_ROOT}/runs/smoke/log.csv")
img_per_s = log["img_per_s"].iloc[len(log) // 2:].median()     # skip warm-up
main = yaml.safe_load(open("configs/base_256.yaml"))
hours = main["steps"] * main["batch_size"] / img_per_s / 3600
sessions = hours * 60 / main["budget_min"]
print(f"{img_per_s:.1f} img/s -> main run ({main['steps']} steps x {main['batch_size']}) ~ {hours:.1f} GPU-hours "
      f"~ {sessions:.1f} sessions of {main['budget_min']} min")
display(Show(sorted(glob.glob(f"{DRIVE_ROOT}/runs/smoke/samples/*.png"))[-1]))
"""),
    cell(PROGRESS, data=False, gpu=False, requirements=()),
    cell(TRAIN_SETTINGS + r'''
!python -m soteria.train --config {CONFIG} out_dir={OUT} {OVERRIDES}
print(open(f"{OUT}/status.json").read())
'''),
])
