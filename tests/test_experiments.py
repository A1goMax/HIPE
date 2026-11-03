import json
import pytest
from hipe.spec import ExperimentConfig, legacy_micro_batch
from hipe.experiments import resolve_suite, submission_command


def test_table2_grid_preserves_original_global_batch_and_sigma_search():
    cases = resolve_suite("table2", data_root="/data", artifacts_root="/artifacts")
    assert len(cases) == 144
    assert {c.global_batch_size for c in cases} == {64}
    assert {c.seed for c in cases} == {6198, 1024, 7}
    assert {c.sigma for c in cases if c.method == "hipe"} == {50, 200, 300, 500, 700}
    assert all(c.accumulation_steps * c.micro_batch_size == 64 for c in cases)
    nope = [c for c in cases if c.method == "nope"]
    assert all(c.legacy_absolute_positions and "legacyAPE" in c.experiment_id for c in nope)


def test_historical_profiles_are_not_silently_interchanged():
    assert legacy_micro_batch("wikitext103", "hipe", "20M", 512, "march2026") == 32
    assert legacy_micro_batch("wikitext103", "hipe", "20M", 512, "january2026") == 16
    assert legacy_micro_batch("c4", "hipe", "60M", 2048, "march2026") == 4
    assert legacy_micro_batch("wikitext103", "alibi", "20M", 2048) == 8


def test_filters_and_correct_nope_have_distinct_identity():
    params = dict(data_root="/data", artifacts_root="/out", datasets=["wikitext103"],
                  models=["20M"], lengths=[512], methods=["rope", "nope", "hipe"],
                  seeds=[6198], sigmas=[200])
    legacy = resolve_suite("table2", **params)
    corrected = resolve_suite("table2", correct_nope=True, **params)
    assert len(legacy) == len(corrected) == 3
    assert {c.experiment_id for c in legacy} != {c.experiment_id for c in corrected}
    assert not any(c.legacy_absolute_positions for c in corrected)


def test_local_global_suite_uses_same_protocol_for_six_conditions():
    cases = resolve_suite("table5", data_root="/data", artifacts_root="/out")
    assert len(cases) == 6
    assert {c.micro_batch_size for c in cases} == {16}
    assert {c.global_batch_size for c in cases} == {64}
    assert {c.max_tokens for c in cases} == {300_000_000}
    assert {c.seq_len for c in cases} == {1024}
    assert {c.sigma for c in cases if c.method == "hipe"} == {500}
    assert {c.attention_pattern for c in cases} == {"global", "local-global", "global-local"}


def test_two_gpu_allocation_keeps_independent_original_batches(tmp_path):
    machine = {"partition": "debug", "cpus_per_gpu": 8, "memory_gb_per_gpu": 64, "time_limit": "96:00:00", "python": "/env/python"}
    command = submission_command(machine, 2, tmp_path / "plan.json", tmp_path)
    assert "--gres=gpu:2" in command and "--ntasks=1" in command
    assert "--cpus-per-task=16" in command


def test_invalid_batch_and_unknown_settings_fail(tmp_path):
    with pytest.raises(ValueError):
        ExperimentConfig(micro_batch_size=3).validate()
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"unknown_switch": True}))
    with pytest.raises(ValueError):
        ExperimentConfig.load(p)


def test_array_requests_one_gpu_per_task_with_two_task_limit(tmp_path):
    machine = {"partition": "debug", "cpus_per_gpu": 8, "memory_gb_per_gpu": 64, "time_limit": "96:00:00", "python": "/env/python"}
    cmd = submission_command(machine, 2, tmp_path / "manifest.json", tmp_path, array_size=324)
    assert "--array=0-323%2" in cmd
    assert "--gres=gpu:1" in cmd and "--cpus-per-task=8" in cmd


def test_fixed_hipe_main_has_36_runs_and_current_paper_layer_split():
    cases = resolve_suite("table2_hipe", data_root="/data", artifacts_root="/out")
    assert len(cases) == 36 and all(c.method == "hipe" for c in cases)
    assert all(c.layer_threshold == 3 and c.max_tokens == 100000000 and c.max_train_steps is None for c in cases)
    assert {c.seed for c in cases} == {6198, 1024, 7}
    sigma = {(c.dataset,c.model_size,c.seq_len):c.sigma for c in cases}
    assert sigma[("wikitext103","60M",512)] == 300
    assert sigma[("c4","60M",2048)] == 50


def test_alibi_restores_automatic_sdpa_and_paper_matched_batch():
    cases = resolve_suite("table2", data_root="/data", artifacts_root="/out", datasets=["wikitext103"], models=["20M"], lengths=[1024], methods=["alibi"], seeds=[6198])
    assert cases[0].micro_batch_size == 16 and cases[0].global_batch_size == 64
    assert cases[0].attention_backend == "sdpa"
