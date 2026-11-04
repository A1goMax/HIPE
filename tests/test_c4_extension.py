import json
from pathlib import Path
import pytest
import torch
from hipe.experiments import resolve_suite
from hipe.config import build_model_config, resolve_layer_sigmas
from hipe.olmo.model import OLMo, BufferCache, ScaledRotaryEmbedding

def test_extension_is_bounded_and_reuses_completed_controls():
    cases=resolve_suite("table2_c4_extension",data_root="/data",artifacts_root="/artifacts")
    assert len(cases)==60 and len({c.experiment_id for c in cases})==60
    assert {c.seed for c in cases}=={6198,1024,7}
    assert {c.micro_batch_size for c in cases}=={4}
    assert {c.global_batch_size for c in cases}=={64}
    assert {c.max_tokens for c in cases}=={100000000}
    assert all(c.dataset=="c4" and c.model_size=="60M" and c.seq_len==2048 and c.total_steps==762 for c in cases)
    resolved=set()
    for c in cases:
        sigmas=resolve_layer_sigmas(8,c.sigma,c.schedule,c.layer_threshold)
        resolved.add((c.sigma,sum(s is not None for s in sigmas)))
    existing={(25,2),(50,2),(50,4),(50,6)}
    expected={(s,h) for s in [1,5,10,25,50,100,200,500,2000] for h in [1,2]}|{(50,h) for h in range(1,9)}
    assert resolved==expected-existing
    assert len(resolved)==20

@pytest.mark.parametrize("sigma",[1,2000])
def test_extension_sigma_extremes_have_finite_spectral_scales_and_gradients(sigma):
    cfg=build_model_config("60M",64,2048,sigma=sigma,method="hipe",schedule="uniform")
    cfg.init_device="cpu"
    emb=ScaledRotaryEmbedding(cfg,BufferCache(),sigma=sigma)
    assert torch.isfinite(emb.scale_factor).all()
    torch.testing.assert_close(emb.scale_factor.square().mean(),torch.tensor(1.0),rtol=1e-5,atol=1e-6)
    q=torch.randn(1,8,5,64,requires_grad=True);k=torch.randn_like(q,requires_grad=True)
    a,b=emb(q,k)
    (a.square().mean()+b.square().mean()).backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all()

@pytest.mark.parametrize("count",[1,3,5,7,8])
def test_extension_really_changes_number_of_effective_hipe_layers(count):
    cfg=build_model_config("60M",64,16,sigma=50,method="hipe",
                           schedule="uniform" if count==8 else "bipartite",
                           layer_threshold=None if count==8 else 7-count)
    cfg.d_model,cfg.n_heads,cfg.mlp_ratio,cfg.init_device=32,4,2,"cpu"
    model=OLMo(cfg)
    active=[i for i,b in enumerate(model.transformer.blocks)
            if isinstance(b.rotary_emb,ScaledRotaryEmbedding) and b.rotary_emb.use_scaling]
    assert active==list(range(8-count,8))

def test_extension_gpu_checks_cover_extremes_and_all_hipe_case():
    cases=resolve_suite("table2_c4_extension_check",data_root="/data",artifacts_root="/out")
    assert len(cases)==4 and {c.seed for c in cases}=={6198}
    assert {(c.sigma,sum(s is not None for s in resolve_layer_sigmas(8,c.sigma,c.schedule,c.layer_threshold))) for c in cases}=={(1,1),(2000,1),(2000,2),(50,8)}
