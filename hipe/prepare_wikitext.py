"""Download and materialize WikiText-103 and its local tokenizer."""

import argparse
from pathlib import Path

from datasets import load_dataset

from .prepare_tokenizer import prepare_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-name", default="EleutherAI/gpt-neox-20b")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    dataset.save_to_disk(str(output / "raw"))
    prepare_tokenizer(args.tokenizer_name, output / "tokenizer")


if __name__ == "__main__":
    main()
