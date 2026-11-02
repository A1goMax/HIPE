import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from hipe.spec import ExperimentConfig
from hipe.train import BatchStream, capture_rng, restore_rng, train_steps, evaluate, resolve_device


class ToyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 6)
        self.output = torch.nn.Linear(6, 8)

    def forward(self, input_ids):
        return SimpleNamespace(logits=self.output(self.embedding(input_ids)))


DATA = [{"input_ids": [i % 8, (i+1) % 8, (i+2) % 8]} for i in range(8)]


def setup(micro=2, steps=2):
    torch.manual_seed(16)
    model = ToyLM()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2)
    cfg = ExperimentConfig(device="cpu", precision="fp32", attention_backend="math",
                           global_batch_size=4, micro_batch_size=micro, seq_len=2, max_train_steps=steps)
    stream = BatchStream(DATA, micro, 42)
    return model, opt, scheduler, cfg, stream


def test_accumulation_matches_original_global_batch_update():
    a, oa, sa, ca, da = setup(micro=1, steps=1)
    b, ob, sb, cb, db = setup(micro=4, steps=1)
    train_steps(a, da, oa, sa, ca, torch.device("cpu"))
    train_steps(b, db, ob, sb, cb, torch.device("cpu"))
    for x, y in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(x, y, atol=1e-7, rtol=1e-6)


def test_resumed_optimizer_scheduler_and_sample_cursor_match(tmp_path):
    full, opt, sched, cfg, stream = setup()
    train_steps(full, stream, opt, sched, cfg, torch.device("cpu"))
    partial, op, sp, cp, dp = setup(steps=1)
    count = train_steps(partial, dp, op, sp, cp, torch.device("cpu"))
    state = {"model": copy.deepcopy(partial.state_dict()), "optimizer": op.state_dict(),
             "scheduler": sp.state_dict(), "stream": dp.state_dict(), "rng": capture_rng()}
    torch.save(state, tmp_path / "resume.pt")
    saved = torch.load(tmp_path / "resume.pt", weights_only=False)
    resumed, ro, rs, rc, rd = setup()
    resumed.load_state_dict(saved["model"])
    ro.load_state_dict(saved["optimizer"])
    rs.load_state_dict(saved["scheduler"])
    rd.load_state_dict(saved["stream"])
    restore_rng(saved["rng"])
    final_count = train_steps(resumed, rd, ro, rs, rc, torch.device("cpu"), start_step=1, tokens=count)
    assert final_count == 16
    for x, y in zip(full.parameters(), resumed.parameters()):
        assert torch.equal(x, y)


def test_evaluation_is_token_weighted_and_preserves_mode():
    model, _, _, _, _ = setup()
    a, b = torch.tensor([[1, 2], [3, 4]]), torch.tensor([[2, 3], [4, 5]])
    c, d = torch.tensor([[6, 7]]), torch.tensor([[7, 0]])
    model.train()
    metrics = evaluate(model, [(a, b), (c, d)], torch.device("cpu"))
    expected = (F.cross_entropy(model(a).logits.flatten(0, 1), b.flatten(), reduction="sum")
                + F.cross_entropy(model(c).logits.flatten(0, 1), d.flatten(), reduction="sum")) / 6
    assert metrics["tokens"] == 6
    assert abs(metrics["loss"] - float(expected.detach())) < 1e-7
    assert model.training


def test_cpu_device_auto_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")
