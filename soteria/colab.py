"""Colab runtime setup shared by every notebook cell.

Each heavy notebook cell clones/pulls the repo and then calls prepare(), so
any cell can be run on its own on a fresh runtime (e.g. after a disconnect,
rerun only the training cell). Every step is idempotent and fast when it
has already been done on this runtime.

Standard library only: this runs before requirements are installed.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

LOCAL_DATA = Path("/content/data")


def _run(cmd: list[str]) -> None:
    # Capture and print from Python: Colab does not reliably show a child
    # process's raw stdout/stderr in the cell, which would hide failures.
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout + r.stderr, end="", flush=True)
    r.check_returncode()


def has_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None and subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True).returncode == 0


def install_requirements(files: tuple[str, ...], marker_dir: Path = Path("/content")) -> None:
    """pip install once per runtime; a marker keyed on the file contents
    re-installs automatically when the requirements change."""
    digest = hashlib.sha1(b"".join(Path(f).read_bytes() for f in files)).hexdigest()[:12]
    marker = marker_dir / f".soteria_pip_{digest}"
    if marker.exists():
        return
    args = [a for f in files for a in ("-r", f)]
    _run([sys.executable, "-m", "pip", "install", "-q", *args])
    marker.touch()


def restore_data(drive_root: str | Path, dest: Path = LOCAL_DATA) -> Path:
    """Unpack prepared.tar from Drive to local disk (much faster to read)."""
    manifest = dest / "prepared" / "manifest.csv"
    if not manifest.exists():
        tar = Path(drive_root) / "data" / "prepared.tar"
        if not tar.exists():
            raise FileNotFoundError(f"{tar} not found: run colab/01_data.ipynb first")
        dest.mkdir(parents=True, exist_ok=True)
        print(f"unpacking {tar} ...", flush=True)
        with tarfile.open(tar) as t:
            t.extractall(dest, filter="data")     # "data": refuse absolute paths / .. escapes
    with manifest.open() as f:
        print(f"data: {sum(1 for _ in f) - 1} images in {manifest.parent}")
    return manifest.parent


def colab_secret(name: str) -> str | None:
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return None


def prepare(drive_root: str, *, data: bool = True, gpu: bool = True,
            requirements: tuple[str, ...] = ("requirements.txt",)) -> None:
    if gpu and not has_gpu():
        raise RuntimeError("No GPU on this runtime. Runtime > Change runtime type > T4 GPU, then rerun "
                           "this cell. (Do not accept Colab's offer of a runtime with no accelerator.)")
    if os.environ.get("HF_TOKEN") is None and (token := colab_secret("HF_TOKEN")):
        os.environ["HF_TOKEN"] = token
    if requirements:
        install_requirements(requirements)
    Path(drive_root).mkdir(parents=True, exist_ok=True)
    if data:
        restore_data(drive_root)
    if gpu:
        _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
