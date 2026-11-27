"""Generate the thin notebooks in colab/ (they run on Colab and on Kaggle).

The notebooks hold no model code: they fetch this repo, install requirements
and call the package. Edit the package, push, and the next notebook run
picks it up; edit this file and rerun it to change the notebooks:

    python build_notebooks.py

Every code cell starts with the same short setup block (see boot()), so any
cell can be run on its own on a fresh runtime: after a disconnect, rerun only
the cell you need. The block works out whether it is on Colab or Kaggle and
sets ROOT (where results go) and DATA (the prepared images); the heavy
lifting lives in soteria/runtime.py.
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "colab"
REPO_URL = "https://github.com/AhnafAbdullah/Soteria-ImageCloaker.git"

PLATFORM_NOTE = r"""
**Colab:** results go to Google Drive (`MyDrive/soteria`). Runtime → T4 GPU.

**Kaggle:** Settings → Accelerator **GPU T4 x2**, Internet **On** (needs a
phone-verified account). Results go to `/kaggle/working/soteria` and are only
kept when you save a version with **Save Version → Save & Run All**; from an
interactive session, download what you need from the Output panel before it ends.
Secrets (`HF_TOKEN`, `GITHUB_TOKEN` for a private fork) go in Add-ons → Secrets.
"""


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
    """Setup block that makes a cell self-contained, on Colab or Kaggle."""
    return f'''
# ---- setup: lets this cell run on its own, on Colab or Kaggle; safe to rerun ----
REPO_URL = "{REPO_URL}"
BRANCH   = "main"
import importlib, os, subprocess, sys
KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.isdir("/kaggle/input")
if not KAGGLE and not os.path.isdir("/content/drive/MyDrive"):
    from google.colab import drive
    drive.mount("/content/drive")                # Colab: results live on Drive
CODE = "/tmp/Project-Soteria" if KAGGLE else "/content/Project-Soteria"
def _git(*args):
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    print(r.stdout + r.stderr, end=""); r.check_returncode()
def _secret(name):
    try:
        if KAGGLE:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret(name)
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return None
if os.path.isdir(CODE):
    _git("-C", CODE, "pull", "-q", "--ff-only")
else:
    _token = _secret("GITHUB_TOKEN")             # only needed for a private fork
    _git("clone", "-q", "-b", BRANCH, REPO_URL.replace("https://", f"https://{{_token}}@") if _token else REPO_URL, CODE)
os.chdir(CODE); sys.path.insert(0, CODE)
_git("log", "-1", "--format=code: %h %s")
import soteria.runtime as rt; importlib.reload(rt)
PLATFORM = rt.platform()
ROOT, DATA = rt.prepare(data={data}, gpu={gpu}, requirements={requirements!r})
# ---------------------------------------------------------------------------------
'''.strip("\n")


def cell(body, **boot_kw):
    return code(boot(**boot_kw) + "\n\n" + body.strip("\n"))


# ---------------------------------------------------------------- 01 data
write("01_data.ipynb", [
    md(r"""
# 01 · Prepare data (run once per platform)

Gets FFHQ-256 (70k aligned faces) and COCO val2017 (5k scenes), normalises
everything to 256 px, detects 5-point face landmarks, fixes the held-out eval
split (300 FFHQ + 100 COCO, seed 42; identical on every platform), and stores
the result as one `prepared.tar`. Later notebooks only unpack that tar.
About 30–45 minutes on a T4.

**Colab:** the cell asks for `kaggle.json` (kaggle.com → Settings → API →
Create New Token) to download FFHQ. The tar goes to Drive.

**Kaggle:** first **Add Input** → search `flickrfaceshq-dataset-nvidia-resized-256px`
(by xhlulu), so nothing is downloaded. Then **Save Version → Save & Run All**.
When it finishes, that version's output holds `soteria/data/prepared.tar`:
attach it to 02 and 03 with **Add Input → Your Work** (this notebook's output).
"""),
    md(PLATFORM_NOTE),
    cell(r"""
from pathlib import Path
sh = get_ipython().system            # same as "!", but usable inside if-blocks
RAW = Path("/tmp/raw")
if PLATFORM == "kaggle":
    FFHQ = rt.find_input_dir("flickrfaceshq", "ffhq")
    if FFHQ is None:
        raise RuntimeError("No FFHQ input found. Add Input > search 'flickrfaceshq-dataset-nvidia-resized-256px' "
                           "(by xhlulu), then rerun. Attached inputs right now:\n" + rt.describe_inputs())
else:
    FFHQ = RAW / "ffhq"
    if not FFHQ.is_dir():
        from google.colab import files
        if not Path("/root/.kaggle/kaggle.json").exists():
            up = files.upload()                      # choose kaggle.json
            Path("/root/.kaggle").mkdir(exist_ok=True)
            Path("/root/.kaggle/kaggle.json").write_bytes(up["kaggle.json"])
            os.chmod("/root/.kaggle/kaggle.json", 0o600)
        sh(f"kaggle datasets download -d xhlulu/flickrfaceshq-dataset-nvidia-resized-256px -p {FFHQ} --unzip -q")
COCO = RAW / "coco" / "val2017"
if not COCO.is_dir():
    sh(f"mkdir -p {RAW}/coco && wget -q http://images.cocodataset.org/zips/val2017.zip -O {RAW}/coco/val2017.zip"
       f" && unzip -q {RAW}/coco/val2017.zip -d {RAW}/coco && rm {RAW}/coco/val2017.zip")
print("FFHQ:", FFHQ, "| COCO:", COCO)

PREPARED = rt.local_data_dir() / "prepared"
if not (PREPARED / "manifest.csv").exists():
    sh(f"python scripts/prepare_data.py --ffhq {FFHQ} --coco {COCO} --out {PREPARED}")
TAR = ROOT / "data" / "prepared.tar"
TAR.parent.mkdir(parents=True, exist_ok=True)
sh(f"tar -cf {TAR} -C {PREPARED.parent} prepared && ls -lh {TAR}")
""", data=False, requirements=("requirements.txt", "requirements-data.txt")),
])


# ----------------------------------------------------------------- 02 pgd
SUMMARY = r"""
import json, pandas as pd
for run in ("eps8", "eps16"):
    s = json.load(open(f"{ROOT}/pgd/{run}/summary.json"))
    print(f"== {run}: {s['images']} images, {s['sec_per_image']:.1f} s/image")
    display(pd.DataFrame(s["by_method"]).round(4))
"""

COMPARE = r"""
import pandas as pd
runs = {"baseline": "eps8", "vae_only": "abl_vae_only", "rebalanced": "abl_rebalanced"}
dfs = {name: pd.read_csv(f"{ROOT}/pgd/{d}/metrics.csv") for name, d in runs.items()}
same = set(dfs["vae_only"]["index"])
cols = [c for c in dfs["baseline"].columns if c.startswith("jpeg75/") and not c.endswith("_shift")]
rows = {"clean": dfs["baseline"].query("method == 'clean'")}
rows.update({name: df.query("method == 'pgd'") for name, df in dfs.items()})
table = pd.DataFrame({name: df[df["index"].isin(same)][cols + ["lpips", "psnr"]].mean() for name, df in rows.items()})
print(f"{len(same)} images, after a real JPEG q75 round trip (lower = better protection)")
display(table.round(4))
"""

EDIT_MD = r"""
## Edit check: does a cloak actually spoil a Stable Diffusion edit?

The VAE ratio is only a proxy. This runs SD1.5 img2img (strength 0.5) on 16 of
the images above (12 faces, 4 scenes): the clean photo, two controls and every
PGD run (pgd8, vae_only, rebalanced, pgd16), all with the same prompt and the
same seeded noise. It runs at 256 px (the size the cloaks were made at) and
512 px (SD1.5's native size, the cloak upscaled).

**This cell runs on its own.** Any PGD run whose images are not in
`ROOT/pgd` (e.g. on Kaggle, where the Colab runs are not available) is
computed first for just these 16 images: ~25 min on a T4, then ~30 min of
edits. On Colab with the earlier runs on Drive it goes straight to the edits.

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
  clean, clean_again, random8, pgd8, vae_only, rebalanced, pgd16. Look for edits
  that are visibly wrecked (smears, grey patches, wrong content), not just different.

Writes `ROOT/edit_check/`: grid_256.png, grid_512.png, metrics.csv, summary.csv.
"""

EDIT_CELL = r"""
!python -m soteria.eval.edit data_root={DATA} --make-missing --pgd-root {ROOT}/pgd --out {ROOT}/edit_check
import pandas as pd
from IPython.display import Image as Show, display
display(pd.read_csv(f"{ROOT}/edit_check/summary.csv").round(4))
for size in (256, 512):
    print(f"grid {size}px: clean input | edit of clean, clean_again, random8, pgd8, vae_only, rebalanced, pgd16")
    display(Show(f"{ROOT}/edit_check/grid_{size}.png", width=900))
"""

write("02_pgd.ipynb", [
    md(r"""
# 02 · Surrogate checks, PGD baseline and edit check (the go/no-go gate)

Every code cell sets itself up, so any one can be run alone on a fresh runtime.
On Kaggle, attach the 01_data output first (Add Input → Your Work).

1. Check that every surrogate behaves on clean images (identity matches its
   mirror, CLIP is stable under blur, VAE latents are sane). A FAIL here means
   a preprocessing bug: fix it before anything else.
2. Run per-image PGD+EOT at 8/255 and 16/255 on 200 held-out images.

**Gate:** after `jpeg75`, the PGD rows should show clearly lower `*_cos` and
`*_ratio` than the `clean` rows. If 8/255 barely moves them, fix the loss or
the augmentations before training the generator.
"""),
    md(PLATFORM_NOTE),
    cell(r"""
!python scripts/check_surrogates.py --config configs/pgd.yaml --out {ROOT}/surrogate_check data_root={DATA}
from IPython.display import Image as Show
Show(f"{ROOT}/surrogate_check/aligned_faces.png")
"""),
    cell(r"""
!python -m soteria.pgd --config configs/pgd.yaml eps=0.03137 data_root={DATA} out_dir={ROOT}/pgd/eps8
"""),
    cell(r"""
!python -m soteria.pgd --config configs/pgd.yaml eps=0.06275 data_root={DATA} out_dir={ROOT}/pgd/eps16
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
!python -m soteria.pgd --config configs/pgd.yaml eps=0.03137 pgd_images=64 w_clip=0 w_face=0 data_root={DATA} out_dir={ROOT}/pgd/abl_vae_only
"""),
    cell(r"""
!python -m soteria.pgd --config configs/pgd.yaml eps=0.03137 pgd_images=64 clip_floor=0.3 face_floor=-0.1 w_vae=4 data_root={DATA} out_dir={ROOT}/pgd/abl_rebalanced
"""),
    cell(COMPARE, data=False, gpu=False, requirements=()),
    md(EDIT_MD),
    cell(EDIT_CELL),
])

# The edit check alone, so Kaggle's "Save & Run All" (background, output kept)
# runs just this instead of every PGD cell in 02.
write("02b_edit_check.ipynb", [
    md(r"""
# 02b · Edit check only

The last cell of 02_pgd as its own notebook, for platforms where long jobs
must run as a saved version.

**Kaggle:** attach the 01_data output (Add Input → Your Work), set GPU T4 x2
and Internet On, then **Save Version → Save & Run All**. It runs in the
background (you can close the tab; ~1–1.5 h) and the results are kept in that
version's **Output** tab under `soteria/edit_check/`. Interactive sessions get
stopped when idle, which loses everything in `/kaggle/working`.
"""),
    md(PLATFORM_NOTE),
    md(EDIT_MD),
    cell(EDIT_CELL),
])


# --------------------------------------------------------------- 03 train
TRAIN_SETTINGS = r'''
# ---- this run ----------------------------------------------------------------
RUN_NAME  = "base_256"     # one folder per experiment AND per person, e.g. "ahnaf_base_256"
CONFIG    = "configs/base_256.yaml"
OVERRIDES = ""             # e.g. "clip_floor=0.3 face_floor=-0.1 w_vae=4". Must not change
                           # while resuming one RUN_NAME: train.py refuses a mismatch.
OUT = ROOT / "runs" / RUN_NAME
rt.restore_run(OUT)        # Kaggle: continue from an earlier saved version attached as input
'''

PROGRESS = TRAIN_SETTINGS + r'''
import glob, json, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Image as Show, display
if not (OUT / "status.json").exists():
    print(f"no run in {OUT} yet")
else:
    status = json.loads((OUT / "status.json").read_text())
    print(json.dumps({k: v for k, v in status.items() if k != "last_val"}, indent=1))
    log = pd.read_csv(OUT / "log.csv")
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.5))
    log.plot(x="step", y=[c for c in log if c.startswith("loss_")], ax=axes[0], title="train loss terms")
    log.plot(x="step", y=[c for c in ("lpips_raw", "clip_cos", "face_cos") if c in log], ax=axes[1], title="train")
    if (OUT / "val.csv").exists():
        val = pd.read_csv(OUT / "val.csv")
        val.plot(x="step", y=[c for c in val if c.startswith("jpeg75/") and not c.endswith("_shift")],
                 ax=axes[2], title="val after JPEG q75 (lower = better)")
        display(val.tail(3).set_index("step").T.round(4))
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
  latest samples. Run it whenever.
- **Main run**: checkpoints every 500 steps and stops itself before the
  session limit (5.5 h on Colab, 10 h on Kaggle). To continue, rerun only
  this cell: it resumes from `last.pt` and loses at most the steps since the
  last checkpoint.

What it saves in `ROOT/runs/<RUN_NAME>/`: `status.json` (step, GPU-minutes,
ETA, latest validation), `log.csv` (training curves), `val.csv` (protection
metrics every 500 steps), `samples/`, `generator_ema.pt` (the model to use)
and `last.pt` (full resume state).

Each person and each experiment needs its own `RUN_NAME`. Two runs writing to
one folder would resume from each other's checkpoints.

**Kaggle:** attach the 01_data output (Add Input → Your Work). Train with
**Save Version → Save & Run All**, so it runs in the background and its output
(the run folder) is kept; set `RUN_SMOKE = False` once the smoke run looks
right, or every version repeats it. To continue in the next session, attach
the previous version's output as an input too (if Kaggle won't attach a
notebook's own output, use Output → New Dataset and attach that): the cells
find `runs/<RUN_NAME>/last.pt` there and resume from it.
"""),
    md(PLATFORM_NOTE),
    cell(r"""
RUN_SMOKE   = True         # set False once checked (Kaggle "Save & Run All" runs every cell)
TRAIN_HOURS = 12           # GPU-hours you can give the main run in total; prints the STEPS that fit
if RUN_SMOKE:
    import glob, pandas as pd, yaml
    from IPython.display import Image as Show, display
    get_ipython().system(f"python -m soteria.train --config configs/smoke.yaml out_dir={ROOT}/runs/smoke data_root={DATA}")
    log = pd.read_csv(f"{ROOT}/runs/smoke/log.csv")
    img_per_s = log["img_per_s"].iloc[len(log) // 2:].median()     # skip warm-up
    main = yaml.safe_load(open("configs/base_256.yaml"))
    hours = main["steps"] * main["batch_size"] / img_per_s / 3600
    session_h = rt.SESSION_MIN[PLATFORM] / 60
    print(f"{img_per_s:.1f} img/s -> main run ({main['steps']} steps x {main['batch_size']}) ~ {hours:.1f} GPU-hours "
          f"~ {hours / session_h:.1f} sessions of {session_h:.1f} h on {PLATFORM}")
    fit = int(TRAIN_HOURS * 3600 * img_per_s / main["batch_size"] // 500 * 500)
    print(f"{TRAIN_HOURS} GPU-hours of training fit ~{fit} steps -> set STEPS = {fit} in the main-run cell")
    samples = sorted(glob.glob(f"{ROOT}/runs/smoke/samples/*.png"))
    if samples:
        display(Show(samples[-1]))
"""),
    cell(PROGRESS, data=False, gpu=False, requirements=()),
    cell(TRAIN_SETTINGS + r'''
RUN_MAIN   = False         # set True to train (Kaggle "Save & Run All" runs every cell)
STEPS      = 20000         # total steps for the whole run; size it from the smoke run's suggestion.
                           # Set it before the first session: the LR schedule is built from it.
BUDGET_MIN = rt.SESSION_MIN[PLATFORM]   # training minutes in THIS session. On Kaggle keep it
                           # below your remaining weekly GPU quota minus ~45 min: a session cut off
                           # by the quota may lose its output; one that stops itself is saved.
if RUN_MAIN:
    get_ipython().system(f"python -m soteria.train --config {CONFIG} out_dir={OUT} data_root={DATA} "
                         f"steps={STEPS} budget_min={BUDGET_MIN} {OVERRIDES}")
    print((OUT / "status.json").read_text() if (OUT / "status.json").exists() else "no status written")
else:
    print("RUN_MAIN is False: main run skipped")
'''),
])
