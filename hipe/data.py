"""Dataset loading and deterministic language-model chunking."""

from pathlib import Path
from typing import Optional, Tuple

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import PreTrainedTokenizerBase


def _load_dataset(path: Path) -> Dataset | DatasetDict:
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    return load_from_disk(str(path))


def load_train_validation(
    data_path: str,
    validation_path: Optional[str] = None,
) -> Tuple[Dataset, Dataset]:
    """Load a DatasetDict or the two processed C4 split directories."""
    root = Path(data_path)

    if validation_path is not None:
        train = _load_dataset(root)
        validation = _load_dataset(Path(validation_path))
        if not isinstance(train, Dataset) or not isinstance(validation, Dataset):
            raise TypeError("Explicit train and validation paths must contain Dataset objects")
        return train, validation

    try:
        dataset = _load_dataset(root)
    except FileNotFoundError:
        raise

    if isinstance(dataset, DatasetDict):
        if "train" not in dataset or "validation" not in dataset:
            raise KeyError("DatasetDict must contain train and validation splits")
        return dataset["train"], dataset["validation"]

    train_path = root / "c4_30M_train"
    validation_path = root / "c4_30M_validation"
    if train_path.exists() and validation_path.exists():
        train = _load_dataset(train_path)
        validation = _load_dataset(validation_path)
        if isinstance(train, Dataset) and isinstance(validation, Dataset):
            return train, validation

    raise TypeError(
        "data_path must be a DatasetDict with train/validation or a directory "
        "containing c4_30M_train and c4_30M_validation"
    )


def _tokenize_batch(examples, tokenizer: PreTrainedTokenizerBase):
    return tokenizer(examples["text"], truncation=False, add_special_tokens=False)


def _group_batch(examples, block_size: int):
    concatenated = {key: sum(examples[key], []) for key in examples.keys()}
    total_length = (len(concatenated["input_ids"]) // block_size) * block_size
    return {
        key: [values[index : index + block_size] for index in range(0, total_length, block_size)]
        for key, values in concatenated.items()
    }


def tokenize_and_chunk(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    sequence_length: int,
    num_proc: int = 1,
) -> Dataset:
    if "text" not in dataset.column_names:
        raise KeyError("The dataset must contain a text column")

    tokenized = dataset.map(
        lambda batch: _tokenize_batch(batch, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="Tokenizing",
    )
    return tokenized.map(
        lambda batch: _group_batch(batch, sequence_length + 1),
        batched=True,
        num_proc=num_proc,
        desc="Grouping",
    )
