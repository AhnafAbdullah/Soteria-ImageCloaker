"""Notebook runtime setup for Colab and Kaggle, shared by every notebook cell.

Each notebook cell clones/pulls the repo and then calls prepare(), so any
cell can be run on its own on a fresh runtime (e.g. after a disconnect, rerun
only the training cell). Every step is idempotent and fast when it has
already been done on this runtime.

Where things live:

              results (ROOT)                    data source                         local data copy
    Colab     Drive: MyDrive/soteria            ROOT/data/prepared.tar              /content/data
    Kaggle    /kaggle/working/soteria           an attached input holding           /tmp/data
              (kept only by "Save Version")     prepared.tar or prepared/
    local     ./local_root                      ROOT/data/prepared.tar              ./data

On Kaggle, results survive a session only when the notebook version is saved
with "Save & Run All". To continue a training run in a later session, attach
the earlier version's output as an input: restore_run() copies the run folder
back into /kaggle/working before training resumes.

Standard library only: this runs before requirements are installed.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
ROOTS = {"colab": "/content/drive/MyDrive/soteria", "kaggle": "/kaggle/working/soteria", "local": "local_root"}
LOCAL_DATA = {"colab": "/content/data", "kaggle": "/tmp/data", "local": "data"}
# Training stops itself this long into a session: Colab free sessions last ~6 h,
# Kaggle sessions 12 h (setup, data unpacking, a smoke run and saving need the rest).
SESSION_MIN = {"colab": 330, "kaggle": 600, "local": 0}


def platform() -> str:
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or KAGGLE_INPUT.is_dir():
        return "kaggle"
    if "COLAB_RELEASE_TAG" in os.environ or os.path.exists("/var/colab/hostname"):
        return "colab"
    return "local"


def root() -> Path:
    return Path(ROOTS[platform()])


def local_data_dir() -> Path:
    return Path(LOCAL_DATA[platform()])


def run_cmd(cmd: list[str]) -> None:
    # Capture and print from Python: notebooks do not reliably show a child
    # process's raw stdout/stderr in the cell, which would hide failures.
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout + r.stderr, end="", flush=True)
    r.check_returncode()


def has_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None and subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True).returncode == 0


def secret(name: str) -> str | None:
    """A Colab secret or a Kaggle secret (Add-ons > Secrets), else None."""
    try:
        if platform() == "kaggle":
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret(name)
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return None


def install_requirements(files: tuple[str, ...], marker_dir: Path = Path("/tmp")) -> None:
    """pip install once per runtime; a marker keyed on the file contents
    re-installs automatically when the requirements change."""
    digest = hashlib.sha1(b"".join(Path(f).read_bytes() for f in files)).hexdigest()[:12]
    marker = marker_dir / f".soteria_pip_{digest}"
    if marker.exists():
        return
    args = [a for f in files for a in ("-r", f)]
    run_cmd([sys.executable, "-m", "pip", "install", "-q", *args])
    marker.touch()


def find_inputs(pattern: str, base: Path = KAGGLE_INPUT, max_depth: int = 7) -> list[Path]:
    """Matches of pattern (e.g. "prepared.tar") up to max_depth folders below
    base. Kaggle's input layout differs between datasets and notebook outputs,
    so the depth is not fixed."""
    if not base.is_dir():
        return []
    hits = []
    for depth in range(max_depth + 1):
        hits += sorted(base.glob("*/" * depth + pattern))
    return hits


def find_input_dir(*name_parts: str, base: Path = KAGGLE_INPUT) -> Path | None:
    """Shallowest folder under base whose name contains any of name_parts,
    ignoring case (an attached dataset; Kaggle's mount layout varies)."""
    if not base.is_dir():
        return None
    parts = [n.lower() for n in name_parts]
    for depth in range(4):
        for p in sorted(base.glob("*/" * depth + "*")):
            if p.is_dir() and any(n in p.name.lower() for n in parts):
                return p
    return None


def describe_inputs(base: Path = KAGGLE_INPUT, max_depth: int = 3, max_lines: int = 40) -> str:
    """Folders under base with their entry counts, for error messages."""
    if not base.is_dir():
        return f"{base} does not exist (nothing attached)"
    lines = []
    for depth in range(max_depth):
        for p in sorted(base.glob("*/" * depth + "*")):
            if p.is_dir() and len(lines) < max_lines:
                lines.append(f"  {p}  ({sum(1 for _ in p.iterdir())} entries)")
    return "\n".join(lines) or f"{base} is empty (nothing attached)"


def unpack(tar: Path, dest: Path) -> Path:
    manifest = dest / "prepared" / "manifest.csv"
    if not manifest.exists():
        dest.mkdir(parents=True, exist_ok=True)
        size_gb = tar.stat().st_size / 1e9
        print(f"unpacking {tar} ({size_gb:.1f} GB) -> {dest} ...", flush=True)
        t0 = time.time()
        with tarfile.open(tar) as t:
            t.extractall(dest, filter="data")     # "data": refuse absolute paths / .. escapes
        minutes = (time.time() - t0) / 60
        print(f"unpacked in {minutes:.1f} min ({size_gb * 1000 / max(minutes * 60, 1e-6):.0f} MB/s)", flush=True)
    return manifest.parent


def restore_data(results_root: Path, dest: Path, inputs: Path = KAGGLE_INPUT) -> Path:
    """Return the prepared data folder, unpacking prepared.tar if needed.

    Order: already on local disk; a prepared/ folder in an attached input
    (used in place, read-only is fine); prepared.tar in the results folder
    (Colab: Drive) or in an attached input.
    """
    if (dest / "prepared" / "manifest.csv").exists():
        data = dest / "prepared"
    elif folders := find_inputs("prepared/manifest.csv", inputs):
        data = folders[0].parent
    elif (tar := results_root / "data" / "prepared.tar").exists():
        data = unpack(tar, dest)
    elif tars := find_inputs("prepared.tar", inputs):
        data = unpack(tars[0], dest)
    else:
        where = ("attach the output of the 01_data notebook (or a dataset holding prepared.tar) "
                 "with Add Input" if platform() == "kaggle" else f"expected {tar}")
        raise FileNotFoundError(f"No prepared data found: run 01_data first, then {where}.")
    with (data / "manifest.csv").open() as f:
        print(f"data: {sum(1 for _ in f) - 1} images in {data}")
    return data


def restore_run(run_dir: str | Path, inputs: Path = KAGGLE_INPUT) -> None:
    """Kaggle: copy a run saved by an earlier notebook version (attached as an
    input) back into the writable results folder, so training resumes from it."""
    run_dir = Path(run_dir)
    if (run_dir / "last.pt").exists():
        return
    saved = find_inputs(f"runs/{run_dir.name}/last.pt", inputs)
    if not saved:
        return
    src = max((p.parent for p in saved), key=lambda d: (d / "last.pt").stat().st_mtime)
    print(f"restoring {src} -> {run_dir}", flush=True)
    shutil.copytree(src, run_dir, dirs_exist_ok=True)


def prepare(*, data: bool = True, gpu: bool = True,
            requirements: tuple[str, ...] = ("requirements.txt",)) -> tuple[Path, Path | None]:
    """Set up this runtime. Returns (results root, prepared data folder or None)."""
    plat = platform()
    if gpu and not has_gpu():
        fix = ("Settings > Accelerator > GPU T4 x2" if plat == "kaggle"
               else "Runtime > Change runtime type > T4 GPU (do not accept a runtime with no accelerator)")
        raise RuntimeError(f"No GPU on this runtime. {fix}, then rerun this cell.")
    if os.environ.get("HF_TOKEN") is None and (token := secret("HF_TOKEN")):
        os.environ["HF_TOKEN"] = token
    if requirements:
        install_requirements(requirements)
    results = root()
    results.mkdir(parents=True, exist_ok=True)
    data_root = restore_data(results, local_data_dir()) if data else None
    if gpu:
        run_cmd(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    print(f"platform: {plat}, results: {results}")
    return results, data_root
