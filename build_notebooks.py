"""Generate the thin Colab notebooks in colab/.

The notebooks hold no model code: they mount Drive, fetch this repo, install
requirements and call the package. Edit the package, push, and the next
notebook run picks it up; edit this file and rerun it to change the notebooks:

    python build_notebooks.py
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "colab"


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


CONFIG = code(r"""
# ---- edit these ----
REPO_URL   = "https://github.com/AhnafAbdullah/Soteria-ImageCloaker.git"   # "" = copy from Drive
BRANCH     = "master"
DRIVE_ROOT = "/content/drive/MyDrive/soteria"   # runs, data tar and (optionally) code live here
# Private repo: add a Colab secret GITHUB_TOKEN. FLUX VAE: add HF_TOKEN.
""")

SETUP = code(r"""
import os, shutil, subprocess
from pathlib import Path
from google.colab import drive, userdata

drive.mount("/content/drive")
Path(DRIVE_ROOT).mkdir(parents=True, exist_ok=True)
CODE = "/content/Project-Soteria"

def secret(name):
    try:
        return userdata.get(name)
    except Exception:
        return None

if REPO_URL:
    url = REPO_URL
    if secret("GITHUB_TOKEN"):
        url = url.replace("https://", f"https://{secret('GITHUB_TOKEN')}@")
    if Path(CODE).exists():
        subprocess.run(["git", "-C", CODE, "pull", "--ff-only"], check=True)
    else:
        subprocess.run(["git", "clone", "-b", BRANCH, url, CODE], check=True)
else:
    shutil.copytree(f"{DRIVE_ROOT}/code/Project-Soteria", CODE, dirs_exist_ok=True)

if secret("HF_TOKEN"):
    os.environ["HF_TOKEN"] = secret("HF_TOKEN")
os.chdir(CODE)
!pip -q install -r requirements.txt
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")

RESTORE_DATA = code(r"""
# Unpack the prepared dataset from Drive onto local disk (much faster to read).
if not Path("/content/data/prepared/manifest.csv").exists():
    Path("/content/data").mkdir(exist_ok=True)
    !tar -xf {DRIVE_ROOT}/data/prepared.tar -C /content/data
!head -3 /content/data/prepared/manifest.csv && wc -l /content/data/prepared/manifest.csv
""")


# ---------------------------------------------------------------- 01 data
write("01_data.ipynb", [
    md(r"""
# 01 · Prepare data (run once)

Downloads FFHQ-256 (70k aligned faces, Kaggle) and COCO val2017 (5k scenes),
normalises everything to 256 px, detects 5-point face landmarks, fixes the
held-out eval split (300 FFHQ + 100 COCO, seed 42), and stores the result as
one tar on Drive. Later notebooks only unpack that tar.

Needs `kaggle.json` (kaggle.com → Settings → API → Create New Token), uploaded
when the cell asks. About 30–45 minutes on a T4 runtime.
"""),
    CONFIG, SETUP,
    code(r"""
!pip -q install -r requirements-data.txt
from google.colab import files
if not Path("/root/.kaggle/kaggle.json").exists():
    up = files.upload()                      # choose kaggle.json
    Path("/root/.kaggle").mkdir(exist_ok=True)
    Path("/root/.kaggle/kaggle.json").write_bytes(up["kaggle.json"])
    os.chmod("/root/.kaggle/kaggle.json", 0o600)
"""),
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
Path(f"{DRIVE_ROOT}/data").mkdir(parents=True, exist_ok=True)
!tar -cf {DRIVE_ROOT}/data/prepared.tar -C /content/data prepared
!ls -lh {DRIVE_ROOT}/data
"""),
])


# ----------------------------------------------------------------- 02 pgd
write("02_pgd.ipynb", [
    md(r"""
# 02 · Surrogate checks and PGD baseline (the go/no-go gate)

1. Check that every surrogate behaves on clean images (identity matches its
   mirror, CLIP is stable under blur, VAE latents are sane). A FAIL here means
   a preprocessing bug: fix it before anything else.
2. Run per-image PGD+EOT at 8/255 and 16/255 on 200 held-out images.

**Gate:** after `jpeg75`, the PGD rows should show clearly lower `*_cos` and
`*_ratio` than the `clean` rows. If 8/255 barely moves them, fix the loss or
the augmentations before training the generator.
"""),
    CONFIG, SETUP, RESTORE_DATA,
    code(r"""
!python scripts/check_surrogates.py --config configs/pgd.yaml --out {DRIVE_ROOT}/surrogate_check
from IPython.display import Image as Show
Show(f"{DRIVE_ROOT}/surrogate_check/aligned_faces.png")
"""),
    code(r"""
!python -m soteria.pgd --config configs/pgd.yaml eps=0.03137 out_dir={DRIVE_ROOT}/pgd/eps8
"""),
    code(r"""
!python -m soteria.pgd --config configs/pgd.yaml eps=0.06275 out_dir={DRIVE_ROOT}/pgd/eps16
"""),
    code(r"""
import json, pandas as pd
for run in ("eps8", "eps16"):
    s = json.load(open(f"{DRIVE_ROOT}/pgd/{run}/summary.json"))
    print(f"== {run}: {s['sec_per_image']:.1f} s/image")
    display(pd.DataFrame(s["by_method"]).round(4))
"""),
])


# --------------------------------------------------------------- 03 train
write("03_train.ipynb", [
    md(r"""
# 03 · Train the cloak generator

Run the smoke test first (300 steps, ~20–30 min). It should show the loss
falling and `lpips_raw` settling near the 0.05 budget, and the sample grids
(clean / cloaked / |δ|×10) should look clean in the middle row.

The main run checkpoints to Drive every 500 steps and stops itself before
the session limit (`budget_min`). **If Colab disconnects, rerun this notebook
top to bottom**: it resumes from `last.pt` automatically.
"""),
    CONFIG, SETUP, RESTORE_DATA,
    code(r"""
%load_ext tensorboard
%tensorboard --logdir {DRIVE_ROOT}/runs
"""),
    code(r"""
!python -m soteria.train --config configs/smoke.yaml out_dir={DRIVE_ROOT}/runs/smoke
"""),
    code(r"""
import glob
from IPython.display import Image as Show
Show(sorted(glob.glob(f"{DRIVE_ROOT}/runs/smoke/samples/*.png"))[-1])
"""),
    code(r"""
# Main run. Safe to re-execute after a disconnect: it resumes from last.pt.
!python -m soteria.train --config configs/base_256.yaml out_dir={DRIVE_ROOT}/runs/base_256
"""),
])
