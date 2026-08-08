"""Stream C4 and materialize the local splits expected by HIPE training."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

from datasets import load_dataset

from .prepare_tokenizer import prepare_tokenizer


def _write_jsonl_shards(
    stream: Iterable[dict],
    split: str,
    count: int,
    shard_size: int,
    staging_dir: Path,
) -> tuple[list[Path], int]:
    files: list[Path] = []
    written = 0
    shard_index = 0
    handle = None

    try:
        for sample in stream:
            text = sample.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            if handle is None or written % shard_size == 0:
                if handle is not None:
                    handle.close()
                shard_path = staging_dir / f"{split}-{shard_index:05d}.jsonl"
                handle = shard_path.open("w", encoding="utf-8")
                files.append(shard_path)
                shard_index += 1
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            written += 1
            if written >= count:
                break
    finally:
        if handle is not None:
            handle.close()

    if written != count:
        raise RuntimeError(
            f"C4 stream ended after {written} usable documents; expected {count}"
        )
    return files, written


def _materialize(files: list[Path], output_dir: Path) -> int:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_dir}")
    dataset = load_dataset(
        "json",
        data_files=[str(path) for path in files],
        split="train",
    )
    dataset.save_to_disk(str(output_dir))
    return len(dataset)


def _resolve_counts(args: argparse.Namespace) -> tuple[int, int]:
    if args.train_docs is not None or args.validation_docs is not None:
        if args.train_docs is None or args.validation_docs is None:
            raise ValueError("train-docs and validation-docs must be provided together")
        return args.train_docs, args.validation_docs

    if args.total_docs <= 0:
        raise ValueError("total-docs must be positive")
    if not 0 < args.train_fraction < 1:
        raise ValueError("train-fraction must be between zero and one")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation-fraction must be between zero and one")
    if args.train_fraction + args.validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a remainder")

    return (
        int(args.total_docs * args.train_fraction),
        int(args.total_docs * args.validation_fraction),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--tokenizer-name", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--dataset-name", default="allenai/c4")
    parser.add_argument("--dataset-config", default="en")
    parser.add_argument("--total-docs", type=int, default=30_000_000)
    parser.add_argument("--train-fraction", type=float, default=0.90)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--train-docs", type=int, default=None)
    parser.add_argument("--validation-docs", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=6198)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    if args.shard_size <= 0:
        raise ValueError("shard-size must be positive")
    train_count, validation_count = _resolve_counts(args)
    if train_count <= 0 or validation_count <= 0:
        raise ValueError("train and validation counts must be positive")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "c4_30M_train"
    validation_path = output / "c4_30M_validation"
    if train_path.exists() or validation_path.exists():
        raise FileExistsError(
            "The output already contains one or more target C4 split directories"
        )

    tokenizer_dir = (
        Path(args.tokenizer_dir)
        if args.tokenizer_dir
        else output.parent / "wikitext" / "tokenizer"
    )

    if args.cache_dir:
        cache_dir = str(Path(args.cache_dir).resolve())
        os.environ["HF_HOME"] = cache_dir
        os.environ["HF_DATASETS_CACHE"] = str(Path(cache_dir) / "datasets")

    stream = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split="train",
        streaming=True,
    ).shuffle(buffer_size=10_000, seed=args.seed)

    with tempfile.TemporaryDirectory(
        dir=str(output.parent), prefix=f".{output.name}.c4-staging-"
    ) as temporary:
        staging_dir = Path(temporary)
        train_files, train_written = _write_jsonl_shards(
            stream, "train", train_count, args.shard_size, staging_dir
        )
        validation_files, validation_written = _write_jsonl_shards(
            stream, "validation", validation_count, args.shard_size, staging_dir
        )
        materialized_train = _materialize(train_files, train_path)
        materialized_validation = _materialize(validation_files, validation_path)

    prepare_tokenizer(args.tokenizer_name, tokenizer_dir)
    manifest = {
        "dataset": args.dataset_name,
        "config": args.dataset_config,
        "source_split": "train",
        "shuffle_seed": args.seed,
        "train_documents": train_written,
        "validation_documents": validation_written,
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "tokenizer_path": str(tokenizer_dir),
        "materialized_rows": {
            "train": materialized_train,
            "validation": materialized_validation,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
