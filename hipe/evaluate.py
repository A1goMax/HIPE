"""Evaluate a fixed-sigma bipartite HIPE checkpoint on a processed dataset."""

from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .config import build_model_config
from .data import load_train_validation, tokenize_and_chunk
from .train import collate_batch
from .olmo.model import OLMo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--validation-path", default=None)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--model-size", choices=["20M", "60M", "300M"], default="60M")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--sigma", type=float, default=500.0)
    parser.add_argument("--rope-scaling-threshold", type=int, default=None)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--flash-attention", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    _, validation_raw = load_train_validation(args.data_path, args.validation_path)
    validation_data = tokenize_and_chunk(
        validation_raw, tokenizer, args.seq_len, args.num_proc
    )
    if args.max_samples:
        validation_data = validation_data.select(
            range(min(args.max_samples, len(validation_data)))
        )
    loader = DataLoader(validation_data, batch_size=args.batch_size, collate_fn=collate_batch)

    config = build_model_config(
        model_size=args.model_size,
        vocab_size=tokenizer.vocab_size,
        sequence_length=args.seq_len,
        sigma=args.sigma,
        layer_threshold=args.rope_scaling_threshold,
        flash_attention=args.flash_attention,
    )
    model = OLMo(config).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if args.flash_attention and device.type == "cuda"
                else nullcontext()
            )
            with context:
                logits = model(input_ids=inputs).logits
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    reduction="sum",
                )
            total_loss += float(loss.item())
            total_tokens += labels.numel()

    mean_loss = total_loss / total_tokens
    result = {
        "method": "fixed_sigma_bipartite_hipe",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "sequence_length": args.seq_len,
        "tokens": total_tokens,
        "loss": mean_loss,
        "ppl": math.exp(mean_loss),
    }
    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
