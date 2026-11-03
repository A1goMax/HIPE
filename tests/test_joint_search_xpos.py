"""Scientific checks for the joint layer/sigma study and reference XPos."""
import json
from pathlib import Path
from collections import Counter
import pytest
import torch

from hipe.config import build_model_config
from hipe.experiments import resolve_suite, verify_publication
from hipe.olmo.model import OLMo, BufferCache, ScaledRotaryEmbedding, XPosEmbedding, LegacyXPosEmbedding


def small(method="hipe", **kwargs):
    config = build_model_config("20M", 64, 2048, method=method, **kwargs)
    config.d_model, config.n_heads, config.mlp_ratio = 32, 4, 2
    config.init_device = "cpu"
    return config


@pytest.mark.parametrize("count,threshold", [(2, 5), (4, 3), (6, 1)])
def test_layer_search_changes_effective_hipe_layers_without_changing_model_depth(count, threshold):
    model = OLMo(small(layer_threshold=threshold, sigma=50))
    active = [i for i, block in enumerate(model.transformer.blocks)
              if isinstance(block.rotary_emb, ScaledRotaryEmbedding) and block.rotary_emb.use_scaling]
    assert len(model.transformer.blocks) == 8
    assert active == list(range(8-count, 8))


def test_bounded_campaign_has_three_seeds_and_reuses_nine_completed_controls():
    cases = resolve_suite("table2_sigma_layers_xpos", data_root="/data", artifacts_root="/out")
    assert len(cases) == 81 and len({c.experiment_id for c in cases}) == 81
    assert Counter(c.method for c in cases) == {"hipe": 45, "xpos": 36}
    assert {c.seed for c in cases} == {6198, 1024, 7}
    assert {c.max_tokens for c in cases} == {100000000}
    assert {c.global_batch_size for c in cases} == {64}
    assert {c.layer_threshold for c in cases if c.method=="hipe"} == {1, 3, 5}
    assert all(not c.legacy_absolute_positions and c.xpos_variant=="torchscale"
               and c.xpos_scale_base==512 for c in cases if c.method=="xpos")
    # These complete original three-seed controls must not be resubmitted.
    controls={("wikitext103",2048,800,3),("c4",512,500,3),("c4",2048,50,3)}
    assert not any((c.dataset,c.seq_len,c.sigma,c.layer_threshold) in controls
                   for c in cases if c.method=="hipe")


def test_reference_and_legacy_xpos_are_distinct_and_absolute_positions_are_explicit():
    reference = OLMo(small("xpos"))
    legacy = OLMo(small("xpos", xpos_variant="legacy", legacy_absolute_positions=True))
    assert isinstance(reference.transformer.blocks[0].rotary_emb, XPosEmbedding)
    assert isinstance(legacy.transformer.blocks[0].rotary_emb, LegacyXPosEmbedding)
    assert "wpe" not in reference.transformer and "wpe" in legacy.transformer


@pytest.mark.parametrize("length", [511, 512, 2048])
def test_reference_xpos_long_bfloat16_forward_and_gradient_are_finite(length):
    embedding = XPosEmbedding(small("xpos"), BufferCache())
    q = torch.randn(1,4,length,8,dtype=torch.bfloat16,requires_grad=True)
    k = torch.randn_like(q,requires_grad=True)
    a,b = embedding(q,k)
    assert a.dtype == b.dtype == torch.bfloat16
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    (a.float().square().mean()+b.float().square().mean()).backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all()
    scale = embedding.get_scale(length,torch.device("cpu"))
    assert scale.dtype == torch.float32 and (scale > 0).all()


def test_reference_xpos_cached_decoding_matches_full_causal_model():
    model = OLMo(small("xpos")).eval()
    tokens = torch.tensor([[1,2,3,4,5,6,7]])
    with torch.no_grad():
        full = model(tokens).logits
        prefix = model(tokens[:,:-1],use_cache=True)
        cached = model(tokens[:,-1:],past_key_values=prefix.attn_key_values,use_cache=True).logits
    torch.testing.assert_close(full[:,-1:],cached,rtol=1e-4,atol=2e-6)


def test_xpos_rejects_invalid_scale_base():
    with pytest.raises(ValueError,match="positive"):
        build_model_config("20M",64,512,method="xpos",xpos_scale_base=0)


def test_publication_gate_requires_the_exact_source_commit(monkeypatch):
    responses=iter(["journal/research-foundation\n","git@github-a1:A1goMax/HIPE.git\n",
                    "oldcommit\trefs/heads/journal/research-foundation\n"])
    monkeypatch.setattr("hipe.experiments.subprocess.check_output",lambda *a,**k:next(responses))
    with pytest.raises(RuntimeError,match="Push the exact source"):
        verify_publication("newcommit")


def test_full_array_waits_for_all_gpu_checks_and_cancels_on_failed_dependency(tmp_path):
    from hipe.experiments import submission_command
    machine={"partition":"debug","cpus_per_gpu":8,"memory_gb_per_gpu":64,
             "time_limit":"96:00:00","python":"/env/python"}
    command=submission_command(machine,2,tmp_path/"manifest.json",tmp_path,array_size=81,afterok=41769)
    assert "--dependency=afterok:41769" in command
    assert "--kill-on-invalid-dep=yes" in command
    assert "--array=0-80%2" in command and "--gres=gpu:1" in command
    with pytest.raises(ValueError,match="positive"):
        submission_command(machine,2,tmp_path/"manifest.json",tmp_path,array_size=81,afterok=0)
