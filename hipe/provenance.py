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
    from .config import resolve_layer_sigmas
    layout = {"method": config.method, "n_layers": model_config.n_layers,
              "legacy_absolute_positions": config.legacy_absolute_positions}
    if config.method == "hipe":
        sigmas = resolve_layer_sigmas(model_config.n_layers, config.sigma, config.schedule,
                                     config.layer_threshold, config.layer_sigmas)
        layout.update({"sigma_by_layer": sigmas,
                       "hipe_layer_indices_zero_based": [i for i, value in enumerate(sigmas) if value is not None],
                       "hipe_layer_count": sum(value is not None for value in sigmas)})
    elif config.method == "xpos":
        layout.update({"implementation": config.xpos_variant, "scale_base": config.xpos_scale_base,
                       "rotary_coordinate_layout": "split_half"})
    publication = None
    campaign_path = os.environ.get("HIPE_CAMPAIGN_MANIFEST")
    if campaign_path:
        publication = json.loads(Path(campaign_path).read_text()).get("publication")
    core = ["hipe/train.py", "hipe/data.py", "hipe/config.py",
            "hipe/olmo/model.py", "hipe/olmo/config.py"]
    return {
        "schema_version": 2, "positional_layout": layout, "experiment": asdict(config),
        "model_config": json.loads(json.dumps(asdict(model_config), default=str)),
        "source": {"commit": git_output("rev-parse", "HEAD"), "branch": git_output("branch", "--show-current"),
                   "status": git_output("status", "--short"), "snapshot": snapshot.name,
                   "snapshot_sha256": sha256(snapshot), "publication": publication,
                   "code_file_sha256": {name: sha256(ROOT / name) for name in core}},
        "runtime": {"python": platform.python_version(), "packages": packages,
                    "torch_cuda": torch.version.cuda, "device": str(device),
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                    "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "tokenizer_file_sha256": tokenizer_files,
        "sampler_protocol": "independent_seeded_permutation_v1",
    }
