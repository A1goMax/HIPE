"""Train the paper's main fixed-sigma bipartite HIPE method."""

from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .config import build_model_config
from .data import load_train_validation, tokenize_and_chunk
from .olmo.model import OLMo


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def collate_batch(batch):
    data = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
    return data[:, :-1].contiguous(), data[:, 1:].contiguous()


def evaluate(model, loader, device, use_autocast: bool, max_batches: int = 0):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch_index, (inputs, labels) in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast_context(device, use_autocast):
                logits = model(input_ids=inputs).logits
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    reduction="sum",
                )
            total_loss += float(loss.item())
            total_tokens += labels.numel()
    model.train()
    if total_tokens == 0:
        return {"loss": None, "ppl": None, "tokens": 0}
    mean_loss = total_loss / total_tokens
    return {"loss": mean_loss, "ppl": math.exp(mean_loss), "tokens": total_tokens}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--validation-path", default=None)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--model-size", choices=["20M", "60M", "300M"], default="60M")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--sigma", type=float, default=500.0)
    parser.add_argument("--rope-scaling-threshold", type=int, default=None)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=100_000_000)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=6198)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--flash-attention", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.global_batch_size % args.micro_batch_size != 0:
        raise ValueError("global-batch-size must be divisible by micro-batch-size")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    train_raw, validation_raw = load_train_validation(args.data_path, args.validation_path)
    train_data = tokenize_and_chunk(train_raw, tokenizer, args.seq_len, args.num_proc)
    validation_data = tokenize_and_chunk(validation_raw, tokenizer, args.seq_len, args.num_proc)

    train_loader = DataLoader(
        train_data,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.micro_batch_size,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
    )

    config = build_model_config(
        model_size=args.model_size,
        vocab_size=tokenizer.vocab_size,
        sequence_length=args.seq_len,
        sigma=args.sigma,
        layer_threshold=args.rope_scaling_threshold,
        flash_attention=args.flash_attention,
    )
    model = OLMo(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    total_steps = args.max_train_steps or max(
        1, args.max_tokens // (args.global_batch_size * args.seq_len)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    grad_accumulation = args.global_batch_size // args.micro_batch_size

    run_config = vars(args).copy()
    run_config.update(
        {
            "device": str(device),
            "vocab_size": tokenizer.vocab_size,
            "total_steps": total_steps,
            "layer_threshold": config.rope_scaling_threshold,
            "method": "fixed_sigma_bipartite_hipe",
        }
    )
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    metrics_path = output_dir / "metrics.jsonl"

    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_iter = iter(train_loader)
    running_loss = 0.0
    use_autocast = args.flash_attention or device.type == "cuda"

    with metrics_path.open("w") as metrics_file:
        for step in range(1, total_steps + 1):
            step_loss = 0.0
            for _ in range(grad_accumulation):
                try:
                    inputs, labels = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    inputs, labels = next(train_iter)
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast_context(device, use_autocast):
                    logits = model(input_ids=inputs).logits
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
                    ) / grad_accumulation
                loss.backward()
                step_loss += float(loss.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running_loss += step_loss

            if step % args.eval_interval == 0 or step == total_steps:
                train_loss = running_loss / min(args.eval_interval, step)
                validation = evaluate(
                    model,
                    validation_loader,
                    device,
                    use_autocast,
                    args.max_val_batches,
                )
                record = {
                    "step": step,
                    "train_loss": train_loss,
                    "validation": validation,
                    "lr": scheduler.get_last_lr()[0],
                }
                metrics_file.write(json.dumps(record) + "\n")
                metrics_file.flush()
                print(json.dumps(record))
                running_loss = 0.0

    torch.save(model.state_dict(), output_dir / "model.pt")


if __name__ == "__main__":
    main()
