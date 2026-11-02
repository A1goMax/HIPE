"""Evaluate a checkpoint using its saved model/config and preprocessing protocol."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .data import prepare_data
from .olmo.config import ModelConfig
from .olmo.model import OLMo
from .provenance import sha256
from .spec import ExperimentConfig
from .train import collate_batch, evaluate, resolve_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="Defaults to run_config.json beside checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    folder = checkpoint.parent
    config = ExperimentConfig.load(args.config or folder / "run_config.json")
    device = resolve_device(args.device)
    if device.type == "cuda" and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU evaluation requires a SLURM allocation")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, local_files_only=True)
    # Retain the exact model config, including legacy baseline behavior and learned scales.
    model_config = ModelConfig(**json.loads((folder / "model_config.json").read_text()))
    model = OLMo(model_config).to(device)
    if checkpoint.name == "checkpoint.pt":
        # Full optimizer/RNG checkpoint authored by this project.
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["experiment"] != config.to_dict():
            raise ValueError("Checkpoint and evaluation experiment configuration differ")
        weights = state["model"]
    else:
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    _, validation, identity = prepare_data(config, tokenizer)
    expected = json.loads((folder / "run_manifest.json").read_text())["data"]
    if identity != expected:
        raise ValueError("Evaluation data identity differs from training manifest")
    loader = DataLoader(validation, batch_size=config.micro_batch_size, collate_fn=collate_batch)
    result = evaluate(model, loader, device, config.precision, config.max_val_batches, config.attention_backend)
    result.update({"checkpoint_sha256": sha256(checkpoint), "experiment_id": config.experiment_id,
                   "method": config.method, "legacy_absolute_positions": config.legacy_absolute_positions,
                   "data_identity": identity})
    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
