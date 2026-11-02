import copy
import pytest
import torch

from hipe.config import build_model_config, resolve_layer_sigmas
from hipe.olmo.model import OLMo, BufferCache, RotaryEmbedding, ScaledRotaryEmbedding


def tiny(method="hipe", **kwargs):
    cfg = build_model_config("20M", 64, 16, sigma=2.0, method=method, **kwargs)
    cfg.d_model, cfg.n_heads, cfg.mlp_ratio = 32, 4, 2
    cfg.init_device = "cpu"
    return cfg


@pytest.mark.parametrize("method", ["rope", "hipe", "nope", "alibi", "xpos"])
def test_each_method_has_finite_forward_and_gradient(method):
    model = OLMo(tiny(method))
    output = model(torch.tensor([[1, 2, 3, 4, 5]])).logits
    assert output.shape == (1, 5, 64)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_nope_compatibility_is_explicit():
    plain = OLMo(tiny("nope"))
    legacy = OLMo(tiny("nope", legacy_absolute_positions=True))
    assert "wpe" not in plain.transformer
    assert "wpe" in legacy.transformer
    absolute_cfg = tiny("nope")
    absolute_cfg.nope = False
    absolute = OLMo(absolute_cfg)
    absolute.load_state_dict(legacy.state_dict(), strict=True)
    absolute.eval()
    legacy.eval()
    inputs = torch.tensor([[1, 2, 3, 4]])
    assert torch.equal(absolute(inputs).logits, legacy(inputs).logits)


@pytest.mark.parametrize("schedule,scaled", [
    ("bipartite", [4, 5, 6, 7]), ("reverse", [0, 1, 2, 3]), ("uniform", list(range(8)))])
def test_effective_layer_schedule(schedule, scaled):
    model = OLMo(tiny(schedule=schedule))
    actual = [i for i, b in enumerate(model.transformer.blocks)
              if isinstance(b.rotary_emb, ScaledRotaryEmbedding) and b.rotary_emb.use_scaling]
    assert actual == scaled


def test_continuous_schedule_translates_paper_indices():
    values = [None, None, None, 50, 200, 500, 700, 1000]
    model = OLMo(tiny(schedule="explicit", layer_sigmas=values))
    assert not isinstance(model.transformer.blocks[2].rotary_emb, ScaledRotaryEmbedding)
    assert isinstance(model.transformer.blocks[3].rotary_emb, ScaledRotaryEmbedding)
    with pytest.raises(ValueError):
        resolve_layer_sigmas(8, 200, layer_threshold=7)


def test_fixed_spectral_formula_matches_legacy_reference():
    from tests.fixtures.legacy_scaled_rope import LegacyScaledRotaryEmbedding
    cfg = tiny(schedule="uniform")
    current = ScaledRotaryEmbedding(cfg, BufferCache(), sigma=2.0)
    old = LegacyScaledRotaryEmbedding(cfg, BufferCache(), sigma=2.0)
    assert torch.equal(current.scale_factor, old.scale_factor)
    q, k = torch.randn(2, 4, 6, 8), torch.randn(2, 4, 6, 8)
    cq, ck = current(q, k)
    oq, ok = old(q, k)
    assert torch.equal(cq, oq) and torch.equal(ck, ok)


def test_learnable_sigmas_have_finite_gradients_and_reload():
    cfg = tiny(learnable_sigma=True)
    model = OLMo(cfg)
    model(torch.tensor([[1, 2, 3, 4]])).logits.square().mean().backward()
    params = [p for name, p in model.named_parameters() if name.endswith("sigma_param")]
    assert len(params) == 4
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    copied = OLMo(cfg)
    copied.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    copied.eval()
    inputs = torch.tensor([[1, 2, 3]])
    assert torch.equal(model(inputs).logits, copied(inputs).logits)


@pytest.mark.parametrize("pattern,local_ids", [
    ("global", []), ("local-global", [0, 1, 2, 3]), ("global-local", [4, 5, 6, 7])])
def test_local_mask_placement_and_cached_causality(pattern, local_ids):
    cfg = tiny(attention_pattern=pattern, local_window_size=3)
    model = OLMo(cfg).eval()
    seen = {}
    for index, block in enumerate(model.transformer.blocks):
        original = block._scaled_dot_product_attention
        def wrapped(*args, _index=index, _original=original, **kwargs):
            mask = kwargs.get("attn_mask")
            if mask is not None:
                seen[_index] = mask.detach().clone()
            return _original(*args, **kwargs)
        block._scaled_dot_product_attention = wrapped
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.no_grad():
        full = model(tokens).logits
        assert sorted(seen) == local_ids
        for index in local_ids:
            mask = seen[index][0, 0]
            assert torch.isneginf(mask[4, 0]) and mask[4, 2] == 0
            assert torch.isneginf(mask[2, 3])
        prefix = model(tokens[:, :-1], use_cache=True)
        cached = model(tokens[:, -1:], past_key_values=prefix.attn_key_values, use_cache=True).logits
    torch.testing.assert_close(full[:, -1:], cached, rtol=1e-4, atol=2e-6)
