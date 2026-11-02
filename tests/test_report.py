import json
import math
import pytest
from hipe.report import summarize, selected_metrics


def test_paper_interval_endpoint_is_distinct_from_extra_final_evaluation():
    rows = [{"step": 3000, "regular_validation": True, "final": False},
            {"step": 3051, "regular_validation": False, "final": True}]
    assert selected_metrics(rows, "last_interval")["step"] == 3000
    assert selected_metrics(rows, "final")["step"] == 3051


def write_run(root, name, seed, loss):
    folder = root / name
    folder.mkdir()
    manifest = {"experiment": {"method": "rope", "seed": seed, "output_dir": name, "experiment_id": name,
                               "max_tokens": 1000, "source_notes": []},
                "source": {"fingerprint": "fixed-code"}, "data": {"processed_validation": "same-data"}}
    (folder / "run_manifest.json").write_text(json.dumps(manifest))
    (folder / "completion.json").write_text("{}")
    metric = {"step": 100, "regular_validation": True, "final": True,
              "validation": {"loss_sum": loss * 10, "tokens": 10}}
    (folder / "metrics.jsonl").write_text(json.dumps(metric) + "\n")


def test_seed_aggregation_keeps_individual_runs_and_rejects_duplicates(tmp_path):
    write_run(tmp_path, "a", 7, 1)
    write_run(tmp_path, "b", 8, 2)
    groups = summarize(tmp_path)
    assert len(groups) == 1 and groups[0]["n"] == 2
    assert groups[0]["mean_ppl"] == pytest.approx((math.exp(1) + math.exp(2)) / 2)
    write_run(tmp_path, "duplicate", 7, 1)
    with pytest.raises(ValueError, match="Repeated seed"):
        summarize(tmp_path)
