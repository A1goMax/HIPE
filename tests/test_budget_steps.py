"""Protocol checks for the separate matched-step token-budget exploration."""
from collections import Counter
import pytest
from hipe.experiments import resolve_suite
from hipe.config import resolve_layer_sigmas
from hipe.spec import ExperimentConfig

def planned():
    return resolve_suite("table2_budget_steps",data_root="/data/qijunrong/03-proj/PE",
                         artifacts_root="/data/qijunrong/03-proj/HIPE-journal")

def test_budget_study_is_broad_bounded_and_never_reruns_L512():
    cases=planned()
    assert len(cases)==264 and len({c.experiment_id for c in cases})==264
    assert {c.seq_len for c in cases}=={1024,2048}
    assert {c.seed for c in cases}=={6198,1024,7}
    assert {c.global_batch_size for c in cases}=={64}
    assert all(c.max_train_steps is None and c.total_steps==3051 for c in cases)
    assert Counter(c.method for c in cases)=={"hipe":204,"rope":24,"nope":24,"alibi":12}
    assert all(c.max_tokens=={1024:200000000,2048:400000000}[c.seq_len] for c in cases)
    assert all(c.legacy_absolute_positions for c in cases if c.method=="nope")
    assert not any(c.method=="xpos" for c in cases)

def test_new_L1024_layout_grid_and_observed_L2048_variants_are_retained():
    cases=[c for c in planned() if c.seed==6198 and c.method=="hipe"]
    assert Counter(c.seq_len for c in cases)=={1024:20,2048:48}
    for dataset in ["wikitext103","c4"]:
        for model in ["20M","60M"]:
            rows=[c for c in cases if (c.dataset,c.model_size,c.seq_len)==(dataset,model,1024)]
            assert len(rows)==5
            assert {sum(v is not None for v in resolve_layer_sigmas(8,c.sigma,c.schedule,c.layer_threshold)) for c in rows}=={1,2,4,6,8}
            assert {c.sigma for c in rows}==({500} if (dataset,model)==("wikitext103","20M") else {300})

def test_archived_successful_ALiBi_backends_and_unavailable_shapes_are_explicit():
    rows=[c for c in planned() if c.method=="alibi" and c.seed==6198]
    actual={(c.dataset,c.model_size,c.seq_len):(c.micro_batch_size,c.attention_backend) for c in rows}
    assert actual=={("wikitext103","20M",1024):(16,"sdpa"),
                    ("wikitext103","60M",1024):(16,"math"),
                    ("wikitext103","60M",2048):(4,"math"),
                    ("c4","60M",1024):(16,"math")}

@pytest.mark.parametrize("length,budget,tokens",[(512,100000000,99975168),(1024,200000000,199950336),(2048,400000000,399900672)])
def test_nominal_optimizer_steps_match_at_the_user_budgets(length,budget,tokens):
    c=ExperimentConfig(seq_len=length,max_tokens=budget,global_batch_size=64)
    assert c.total_steps==3051
    assert c.total_steps*length*64==tokens
    assert tokens<=budget<tokens+length*64

def test_budget_study_gpu_checks_are_small_and_representative():
    checks=resolve_suite("table2_budget_steps_check",data_root="/data",artifacts_root="/out")
    assert len(checks)==8 and {c.seed for c in checks}=={6198}
    assert {c.method for c in checks}=={"hipe","nope","alibi"}
    assert any(c.method=="hipe" and c.schedule=="uniform" and c.seq_len==2048 for c in checks)
    assert {c.attention_backend for c in checks if c.method=="alibi"}=={"math","sdpa"}
