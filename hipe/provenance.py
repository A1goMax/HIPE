"""Source and runtime provenance, stored with each execution."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect(config, model_config, output_dir, device):
    import torch
    output_dir = Path(output_dir)
    packages = {}
    for name in ["torch", "numpy", "datasets", "transformers", "omegaconf", "flash-attn"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    tracked = git_output("ls-files", "-z") or ""
    extra = git_output("ls-files", "--others", "--exclude-standard", "-z") or ""
    names = sorted(set((tracked + "\0" + extra).split("\0")) - {""})
    if not names:
        names = [p.relative_to(ROOT).as_posix() for folder in ["hipe", "configs", "scripts"]
                 for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    snapshot = output_dir / "source_snapshot.zip"
    with zipfile.ZipFile(snapshot, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            path = (ROOT / name).resolve()
            if path.is_relative_to(ROOT) and path.is_file() and not path.is_relative_to(output_dir.resolve()):
                archive.write(path, name)
    tokenizer_files = {}
    token_root = Path(config.tokenizer_path)
    if token_root.is_dir():
        tokenizer_files = {p.name: sha256(p) for p in token_root.iterdir() if p.is_file()}
    return {
        "schema_version": 1, "experiment": asdict(config),
        "model_config": json.loads(json.dumps(asdict(model_config), default=str)),
        "source": {"commit": git_output("rev-parse", "HEAD"), "branch": git_output("branch", "--show-current"),
                   "status": git_output("status", "--short"), "snapshot": snapshot.name,
                   "snapshot_sha256": sha256(snapshot)},
        "runtime": {"python": platform.python_version(), "packages": packages,
                    "torch_cuda": torch.version.cuda, "device": str(device),
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "tokenizer_file_sha256": tokenizer_files,
        "sampler_protocol": "independent_seeded_permutation_v1",
    }
