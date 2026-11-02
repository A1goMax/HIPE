"""Versioned preprocessing shared by main experiments and ablations."""
from itertools import chain
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk


def _load_dataset(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return load_from_disk(str(path))


def load_train_validation(data_path, validation_path=None):
    root = Path(data_path)
    if validation_path:
        train, validation = _load_dataset(root), _load_dataset(validation_path)
    elif (root / "c4_30M_train").exists():
        train = _load_dataset(root / "c4_30M_train")
        validation = _load_dataset(root / "c4_30M_validation")
    else:
        dataset = _load_dataset(root)
        if not isinstance(dataset, DatasetDict) or not {"train", "validation"} <= set(dataset):
            raise TypeError("Expected a train/validation DatasetDict or explicit split paths")
        train, validation = dataset["train"], dataset["validation"]
    if not isinstance(train, Dataset) or not isinstance(validation, Dataset):
        raise TypeError("Both splits must contain Dataset objects")
    return train, validation


def _group_batch(examples, block_size):
    ids = list(chain.from_iterable(examples["input_ids"]))
    total = len(ids) // block_size * block_size
    return {"input_ids": [ids[i:i + block_size] for i in range(0, total, block_size)]}


def tokenize_and_chunk(dataset, tokenizer, sequence_length, num_proc=1, protocol="release"):
    if "text" not in dataset.column_names:
        raise KeyError("Dataset requires a text column")
    if protocol not in {"release", "legacy_wiki", "legacy_c4"}:
        raise ValueError("Unknown preprocessing protocol")
    block_size = sequence_length + 1
    if protocol == "legacy_c4":
        def c4_chunk(batch):
            ids = list(chain.from_iterable(
                tokenizer.encode(text, add_special_tokens=False)
                for text in batch["text"] if text
            ))
            end = len(ids) // block_size * block_size
            return {"input_ids": [ids[i:i + block_size] for i in range(0, end, block_size)]}
        return dataset.map(c4_chunk, batched=True, batch_size=1000,
                           remove_columns=dataset.column_names, num_proc=num_proc)
    def tokenize(batch):
        # The March WikiText trainer used tokenizer defaults (special tokens enabled).
        result = tokenizer(batch["text"], truncation=False, add_special_tokens=protocol == "legacy_wiki")
        return {"input_ids": result["input_ids"]}
    tokenized = dataset.map(tokenize, batched=True, batch_size=1000,
                            remove_columns=dataset.column_names, num_proc=num_proc)
    return tokenized.map(lambda batch: _group_batch(batch, block_size), batched=True,
                         batch_size=1000, num_proc=num_proc)


def prepare_data(config, tokenizer):
    train, validation = load_train_validation(config.data_path, config.validation_path)
    raw = {"train": train._fingerprint, "validation": validation._fingerprint,
           "train_rows": len(train), "validation_rows": len(validation)}
    if config.max_train_samples:
        train = train.select(range(min(config.max_train_samples, len(train))))
    if config.max_val_samples:
        validation = validation.select(range(min(config.max_val_samples, len(validation))))
    train = tokenize_and_chunk(train, tokenizer, config.seq_len, config.num_proc, config.data_pipeline)
    validation = tokenize_and_chunk(validation, tokenizer, config.seq_len, config.validation_num_proc, config.data_pipeline)
    if not len(train) or not len(validation):
        raise ValueError("Preprocessing produced an empty train/validation split")
    identity = {"raw": raw, "processed_train": train._fingerprint,
                "processed_validation": validation._fingerprint,
                "train_chunks": len(train), "validation_chunks": len(validation),
                "protocol": config.data_pipeline, "map_batch_size": 1000,
                "num_proc": config.num_proc, "validation_num_proc": config.validation_num_proc,
                "train_first_n": config.max_train_samples, "validation_first_n": config.max_val_samples}
    return train, validation, identity
