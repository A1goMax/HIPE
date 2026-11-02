import json
from pathlib import Path
import sys

from datasets import Dataset, DatasetDict
import pytest
import torch

import hipe.train as training
from hipe.config import build_model_config
from hipe.olmo.model import OLMo
from hipe.spec import ExperimentConfig


class LocalTokenizer:
    vocab_size = 64
    def __len__(self):
        return 64
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 64 for c in text]
    def __call__(self, texts, truncation=False, add_special_tokens=False):
        return {"input_ids": [self.encode(text) for text in texts]}


def test_full_cpu_code_check_records_manifest_and_resumes(tmp_path, monkeypatch):
    data_path = tmp_path / "data"
    data = Dataset.from_dict({"text": ["abcdefghijkl"] * 6})
    DatasetDict(train=data, validation=data).save_to_disk(str(data_path))
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    (tokenizer_path / "identity.json").write_text('{"fixture": true}')
    monkeypatch.setattr(training.AutoTokenizer, "from_pretrained", lambda *a, **k: LocalTokenizer())
    def small_model(config, vocab):
        cfg = build_model_config("20M", vocab, config.seq_len, 2.0)
        cfg.d_model, cfg.n_heads, cfg.mlp_ratio = 32, 4, 2
        cfg.init_device = "cpu"
        return OLMo(cfg), cfg
    monkeypatch.setattr(training, "make_model", small_model)
    original_save = training.atomic_save
    def save_with_first_step(state, path):
        original_save(state, path)
        if Path(path).name == "checkpoint.pt" and state["step"] == 1:
            original_save(state, tmp_path / "run" / "resume_one.pt")
    monkeypatch.setattr(training, "atomic_save", save_with_first_step)
    config = ExperimentConfig(output_dir=str(tmp_path / "run"), data_path=str(data_path),
                              tokenizer_path=str(tokenizer_path), device="cpu", precision="fp32",
                              attention_backend="math", max_train_steps=2, seq_len=3,
                              global_batch_size=2, micro_batch_size=1, num_proc=1,
                              validation_num_proc=1, checkpoint_interval=1, eval_interval=1)
    first = training.run(config, cpu_check=True)
    folder = Path(config.output_dir)
    weights = torch.load(folder / "model.pt", weights_only=True)
    manifest = json.loads((folder / "run_manifest.json").read_text())
    assert manifest["experiment"]["global_batch_size"] == 2
    assert manifest["data"]["validation_chunks"] > 0
    assert manifest["source"]["snapshot_sha256"]
    assert manifest["source"]["fingerprint"]
    assert manifest["tokenizer_file_sha256"]["identity.json"]
    assert first["actual_prediction_tokens"] == 12
    import hipe.evaluate as evaluation
    monkeypatch.setattr(sys, "argv", ["evaluate", "--checkpoint", str(folder / "model.pt"),
                                     "--device", "cpu", "--output", str(tmp_path / "evaluation.json")])
    evaluation.main()
    measured = json.loads((tmp_path / "evaluation.json").read_text())
    assert measured["tokens"] > 0 and measured["data_identity"] == manifest["data"]
    with pytest.raises(FileExistsError):
        training.run(config, cpu_check=True)
    training.run(config, resume=folder / "resume_one.pt", cpu_check=True)
    resumed = torch.load(folder / "model.pt", weights_only=True)
    assert all(torch.equal(value, resumed[key]) for key, value in weights.items())
    records = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records] == [1, 2]


def test_production_training_requires_slurm(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="SLURM"):
        training.run(ExperimentConfig(output_dir=str(tmp_path / "blocked")))


def test_cpu_check_cannot_be_used_for_a_full_experiment(tmp_path):
    with pytest.raises(ValueError, match="limited"):
        training.run(ExperimentConfig(device="cpu", max_train_steps=4), cpu_check=True)
