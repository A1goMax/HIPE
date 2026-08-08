"""Download a Hugging Face tokenizer into a local reusable directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer


def prepare_tokenizer(name: str, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.save_pretrained(str(output))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = prepare_tokenizer(args.name, args.output_dir)
    print(f"Tokenizer saved to {output}")


if __name__ == "__main__":
    main()
