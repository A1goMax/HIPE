"""One LM trainer for the Table 2 main comparison and compatible ablations."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, fields
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .config import build_model_config
from .data import prepare_data
from .olmo.model import OLMo
from .provenance import collect, sha256
from .spec import ExperimentConfig


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value):
    return torch.device(("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else value)


def autocast_context(device, precision):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16) if precision == "bf16" else nullcontext()


def backend_context(backend):
    if backend == "math":
        from torch.nn.attention import sdpa_kernel, SDPBackend
        return sdpa_kernel(SDPBackend.MATH)
    return nullcontext()


def collate_batch(batch):
    data = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
    return data[:, :-1].contiguous(), data[:, 1:].contiguous()


class BatchStream:
    """Paired sample order with an exact resumable cursor, independent of model RNG."""
    def __init__(self, dataset, batch_size, seed):
        self.dataset, self.batch_size = dataset, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(len(dataset), generator=self.generator)
        self.cursor = 0

    def next(self):
        if self.cursor == len(self.order):
            self.order = torch.randperm(len(self.dataset), generator=self.generator)
            self.cursor = 0
        ids = self.order[self.cursor:self.cursor + self.batch_size].tolist()
        self.cursor += len(ids)
        return collate_batch([self.dataset[i] for i in ids])

    def state_dict(self):
        return {"order": self.order, "cursor": self.cursor, "generator": self.generator.get_state()}

    def load_state_dict(self, state):
        if len(state["order"]) != len(self.dataset):
            raise ValueError("Cannot resume against a different dataset size")
        self.order, self.cursor = state["order"], state["cursor"]
        self.generator.set_state(state["generator"])


def evaluate(model, loader, device, precision="fp32", max_batches=0, backend="sdpa"):
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad(), backend_context(backend):
        for index, (inputs, labels) in enumerate(loader):
            if max_batches and index >= max_batches:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast_context(device, precision):
                logits = model(input_ids=inputs).logits
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="sum")
            total_loss += float(loss)
            total_tokens += labels.numel()
    model.train(was_training)
    if not total_tokens:
        raise ValueError("Evaluation population has no prediction tokens")
    mean_loss = total_loss / total_tokens
    return {"loss_sum": total_loss, "loss": mean_loss,
            "ppl": math.exp(mean_loss) if mean_loss < 700 else float("inf"), "tokens": total_tokens}


def make_model(config, vocab_size):
    model_config = build_model_config(
        config.model_size, vocab_size, config.seq_len, config.sigma, config.layer_threshold,
        config.attention_backend == "flash", method=config.method, schedule=config.schedule,
        layer_sigmas=config.layer_sigmas, learnable_sigma=config.learnable_sigma,
        attention_pattern=config.attention_pattern, local_window_size=config.local_window_size,
        local_layers=config.local_layers, legacy_absolute_positions=config.legacy_absolute_positions)
    return OLMo(model_config), model_config


def atomic_save(state, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def train_steps(model, stream, optimizer, scheduler, config, device, start_step=0,
                tokens=0, on_step=None):
    """Core execution also used by CPU numerical/resume tests."""
    model.train()
    for step in range(start_step + 1, config.total_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss, step_tokens = 0.0, 0
        with backend_context(config.attention_backend):
            for _ in range(config.accumulation_steps):
                inputs, labels = stream.next()
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast_context(device, config.precision):
                    logits = model(input_ids=inputs).logits
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                    scaled_loss = loss / config.accumulation_steps
                if not torch.isfinite(scaled_loss):
                    raise FloatingPointError("Nonfinite training loss")
                scaled_loss.backward()
                step_loss += float(scaled_loss.detach())
                step_tokens += labels.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        tokens += step_tokens
        if on_step:
            on_step(step, step_loss, tokens)
    return tokens


def run(config, resume=None, *, cpu_check=False):
    config.validate()
    device = resolve_device(config.device)
    if cpu_check:
        if device.type != "cpu" or not config.max_train_steps or config.max_train_steps > 3:
            raise ValueError("CPU code checks are limited to three explicit unit-test steps")
    else:
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("Experiment execution requires SLURM; use hipe.experiments submit")
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Production experiments require an allocated CUDA GPU")
        if "5090" not in torch.cuda.get_device_name(device):
            raise RuntimeError("This campaign requires an allocated RTX 5090")
    if config.attention_backend == "flash" and config.method != "alibi" and config.attention_pattern == "global":
        import flash_attn  # explicit failure rather than an undocumented backend fallback
    out = Path(config.output_dir)
    if resume:
        if not out.is_dir():
            raise FileNotFoundError("Resume output directory is missing")
    else:
        out.mkdir(parents=True, exist_ok=False)
    from .experiments import source_fingerprint
    fingerprint = source_fingerprint()
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, local_files_only=True)
    # Legacy trainers padded the output vocabulary to a multiple of 64.
    raw_vocab = max(tokenizer.vocab_size, len(tokenizer))
    vocab = (raw_vocab + 63) // 64 * 64
    train_data, val_data, data_identity = prepare_data(config, tokenizer)
    model, model_config = make_model(config, vocab)
    model.to(device)
    stream = BatchStream(train_data, config.micro_batch_size, config.seed)
    val_loader = DataLoader(val_data, batch_size=config.micro_batch_size, collate_fn=collate_batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, betas=tuple(config.betas),
                                 weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.total_steps)
    start_step, tokens = 0, 0
    if resume:
        # These are full checkpoints produced by this trainer, including NumPy RNG.
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state.get("source_fingerprint") != fingerprint:
            raise ValueError("Resume source fingerprint differs from checkpoint")
        if state["experiment"] != config.to_dict():
            raise ValueError("Resume configuration differs from checkpoint")
        if state["data_identity"] != data_identity:
            raise ValueError("Resume data fingerprints differ")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        stream.load_state_dict(state["stream"])
        start_step, tokens = state["step"], state["actual_prediction_tokens"]
        restore_rng(state["rng"])
    else:
        manifest = collect(config, model_config, out, device)
        manifest["source"]["fingerprint"] = fingerprint
        manifest["data"] = data_identity
        manifest["vocabulary"] = {"raw": raw_vocab, "padded": vocab}
        manifest["nominal_training_tokens"] = config.total_steps * config.global_batch_size * config.seq_len
        (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (out / "run_config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
        (out / "model_config.json").write_text(json.dumps(manifest["model_config"], indent=2) + "\n")
    metrics_path = out / "metrics.jsonl"
    if resume and metrics_path.exists():
        rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        metrics_path.write_text("".join(json.dumps(r) + "\n" for r in rows if r["step"] <= start_step))

    def checkpoint(step, actual_tokens):
        return {"schema_version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "stream": stream.state_dict(),
                "rng": capture_rng(), "step": step, "actual_prediction_tokens": actual_tokens,
                "experiment": config.to_dict(), "data_identity": data_identity, "source_fingerprint": fingerprint}

    with metrics_path.open("a") as log:
        def on_step(step, train_loss, actual_tokens):
            regular = step % config.eval_interval == 0
            final = step == config.total_steps
            if regular or final:
                result = evaluate(model, val_loader, device, config.precision,
                                  config.max_val_batches, config.attention_backend)
                record = {"step": step, "train_loss": train_loss, "validation": result,
                          "actual_prediction_tokens": actual_tokens, "regular_validation": regular,
                          "final": final, "lr": scheduler.get_last_lr()[0]}
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
                if regular:
                    atomic_save(model.state_dict(), out / "last_validation.pt")
            if step % config.checkpoint_interval == 0 or final:
                atomic_save(checkpoint(step, actual_tokens), out / "checkpoint.pt")
        tokens = train_steps(model, stream, optimizer, scheduler, config, device, start_step, tokens, on_step)
    atomic_save(model.state_dict(), out / "model.pt")
    summary = {"step": config.total_steps, "actual_prediction_tokens": tokens,
               "model_sha256": sha256(out / "model.pt"), "checkpoint_sha256": sha256(out / "checkpoint.pt")}
    (out / "completion.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_config():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--resume")
    # Retain the original single-method shell interface.
    for name in ["output-dir", "data-path", "validation-path", "tokenizer-path", "model-size", "device"]:
        parser.add_argument("--" + name)
    for name in ["seq-len", "global-batch-size", "micro-batch-size", "max-tokens", "max-train-steps",
                 "eval-interval", "max-val-batches", "num-proc", "seed", "rope-scaling-threshold"]:
        parser.add_argument("--" + name, type=int)
    for name in ["sigma", "lr"]:
        parser.add_argument("--" + name, type=float)
    parser.add_argument("--flash-attention", action="store_true")
    args = parser.parse_args()
    if args.config:
        config = ExperimentConfig.load(args.config)
    else:
        config = ExperimentConfig(data_pipeline="release", attention_backend="sdpa")
    for key, value in vars(args).items():
        if key not in {"config", "resume", "flash_attention", "rope_scaling_threshold"} and value is not None:
            setattr(config, key, value)
    if args.rope_scaling_threshold is not None:
        config.layer_threshold = args.rope_scaling_threshold
    if args.flash_attention:
        config.attention_backend = "flash"
    return config.validate(), args.resume


if __name__ == "__main__":
    spec, resume_path = parse_config()
    run(spec, resume_path)
