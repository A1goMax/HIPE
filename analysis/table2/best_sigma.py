"""Compare best observed fixed-sigma HIPE means with baseline results.

Selection is by the mean across the complete requested seed set, never by the
best individual seed. Distinct sigma values are separate candidates. Inputs may
contain overlapping campaign exports; identical run paths are deduplicated.
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "artifacts/analysis/table2_20261103/raw_runs.json"
DEFAULT_OUTPUT = ROOT / "outputs/best_sigma_20261103"
CORE_FILES = (
    "hipe/train.py", "hipe/data.py", "hipe/config.py",
    "hipe/olmo/model.py", "hipe/olmo/config.py",
)
IGNORED_CONFIG = {"seed", "output_dir", "experiment_id", "source_notes"}
METHOD_CONTROLS = {
    "method", "sigma", "schedule", "layer_sigmas", "layer_threshold",
    "learnable_sigma", "legacy_absolute_positions", "attention_backend",
    "micro_batch_size",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()[:16]


def load_inputs(paths):
    """Prefer a completed snapshot over an older incomplete copy of one run."""
    campaigns, runs, sources = {}, {}, []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        sources.append({"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()})
        for campaign in data["campaigns"]:
            job = str(campaign["job"])
            hashes = campaign.get("core_file_sha256", {})
            if not set(CORE_FILES) <= set(hashes):
                raise ValueError(f"Missing core-file hashes for campaign {job}")
            core = {name: hashes[name] for name in CORE_FILES}
            old = campaigns.get(job)
            if old and old["core"] != core:
                raise ValueError(f"Conflicting core-file hashes for campaign {job}")
            campaigns[job] = {"core": core, "core_id": digest(core)}
        for run in data["runs"]:
            path_key = run["output_dir"]
            old = runs.get(path_key)
            if old:
                if old["config"] != run["config"]:
                    raise ValueError(f"Conflicting configuration for duplicated run path {path_key}")
                if "completion.json" in old and "completion.json" in run:
                    if old["completion.json"] != run["completion.json"] or old["metrics"] != run["metrics"]:
                        raise ValueError(f"Conflicting completed data for duplicated run path {path_key}")
                if "completion.json" in old:
                    continue
                if "completion.json" not in run and len(old.get("metrics", [])) > len(run.get("metrics", [])):
                    continue
            runs[path_key] = run
    return list(runs.values()), campaigns, sources


def endpoint_value(run, endpoint):
    records = [r for r in run["metrics"] if r["regular_validation"]] if endpoint == "last_interval" else [r for r in run["metrics"] if r["final"]]
    if not records:
        raise ValueError(f"No {endpoint} record for {run['output_dir']}")
    metric = max(records, key=lambda row: row["step"])
    value = metric["validation"]["ppl"]
    reconstructed = math.exp(metric["validation"]["loss_sum"] / metric["validation"]["tokens"])
    if not math.isfinite(value) or not math.isclose(value, reconstructed, rel_tol=1e-12):
        raise ValueError(f"Invalid PPL in {run['output_dir']}")
    return {"ppl": value, "step": metric["step"],
            "training_tokens": metric["actual_prediction_tokens"],
            "validation_tokens": metric["validation"]["tokens"]}


def protocol_identity(run, campaign):
    config = {k: v for k, v in run["config"].items() if k not in IGNORED_CONFIG}
    if config["method"] != "hipe":
        config.update(sigma=None, schedule=None, layer_sigmas=None,
                      layer_threshold=None, learnable_sigma=False)
    else:
        config["sigma"] = float(config["sigma"])
    manifest = run["run_manifest.json"]
    # Normalize different commits only through exactly matching collected files.
    # Job IDs, source checkout location and source ZIP timestamps are not physics.
    return {
        "config": config,
        "core_id": campaign["core_id"],
        "core_file_sha256": campaign["core"],
        "data": manifest["data"],
        "tokenizer_file_sha256": manifest["tokenizer_file_sha256"],
        "vocabulary": manifest["vocabulary"],
        "runtime_packages": manifest["runtime"]["packages"],
        "gpu": manifest["runtime"]["gpu"],
    }


def comparison_identity(identity):
    value = dict(identity)
    value["config"] = {k: v for k, v in identity["config"].items() if k not in METHOD_CONTROLS}
    return value


def selection_identity(identity):
    value = dict(identity)
    value["config"] = {k: v for k, v in identity["config"].items() if k != "sigma"}
    return value


def build_groups(runs, campaigns, seeds):
    groups, failed = {}, []
    for run in runs:
        config = run["config"]
        if config.get("max_train_steps") is not None:
            continue  # preflight or debug executions are not main results
        if "completion.json" not in run:
            failed.append(run)
            continue
        if (run.get("worker_status") or {}).get("returncode") != 0:
            raise ValueError(f"Completed artifact with nonzero worker status: {run['output_dir']}")
        expected = config["max_tokens"] // (config["global_batch_size"] * config["seq_len"])
        if run["completion.json"]["step"] != expected:
            raise ValueError(f"Incomplete nominal training budget: {run['output_dir']}")
        identity = protocol_identity(run, campaigns[str(run["job"])])
        group = groups.setdefault(canonical(identity), {"identity": identity, "runs": []})
        if any(r["config"]["seed"] == config["seed"] for r in group["runs"]):
            raise ValueError(f"Duplicate completed seed in a configuration: {run['output_dir']}")
        group["runs"].append(run)
    summaries = []
    for group in groups.values():
        available = {r["config"]["seed"] for r in group["runs"]}
        eligible = set(seeds) <= available
        for endpoint in ("last_interval", "final"):
            chosen = [r for seed in seeds for r in group["runs"] if r["config"]["seed"] == seed]
            values = [endpoint_value(r, endpoint) for r in chosen]
            if len({(v["step"], v["training_tokens"], v["validation_tokens"]) for v in values}) > 1:
                raise ValueError("Training/evaluation endpoints differ within a seed group")
            summary = {
                "group_id": digest(group["identity"]), "identity": group["identity"],
                "endpoint": endpoint, "eligible": eligible, "n": len(chosen),
                "seeds": [r["config"]["seed"] for r in chosen],
                "seed_values": {str(r["config"]["seed"]): endpoint_value(r, endpoint)["ppl"] for r in chosen},
                "mean_ppl": statistics.mean(v["ppl"] for v in values) if values else None,
                "sample_std_ppl": statistics.stdev(v["ppl"] for v in values) if len(values) > 1 else None,
                "run_paths": [r["output_dir"] for r in chosen],
                "source_commits": sorted({r["run_manifest.json"]["source"]["commit"] for r in chosen}),
                "step": values[0]["step"] if values else None,
                "training_tokens": values[0]["training_tokens"] if values else None,
            }
            summaries.append(summary)
    return summaries, failed


def choose_candidates(candidates):
    if not candidates:
        raise ValueError("No complete fixed-sigma candidate")
    if len({(g["step"], g["training_tokens"]) for g in candidates}) != 1:
        raise ValueError("Candidate sigma budgets/endpoints differ")
    if len({g["identity"]["config"]["sigma"] for g in candidates}) != len(candidates):
        raise ValueError("Repeated sigma in one selection family")
    return min(candidates, key=lambda g: (g["mean_ppl"], g["identity"]["config"]["sigma"]))


def unavailable_baseline(failed, ds, size, length, method, endpoint):
    candidates = [r for r in failed if (r["config"]["dataset"], r["config"]["model_size"],
                  r["config"]["seq_len"], r["config"]["method"]) == (ds, size, length, method)]
    reason = "OOM" if candidates and all("OutOfMemoryError" in (r.get("failure_tail") or "") for r in candidates) else "not_available"
    return {"status": reason, "n": 0, "mean_ppl": None, "sample_std_ppl": None,
            "run_paths": [r["output_dir"] for r in candidates], "endpoint": endpoint}


def compare(summaries, failed):
    families = collections.defaultdict(list)
    baselines = [g for g in summaries if g["eligible"] and g["identity"]["config"]["method"] != "hipe"]
    for summary in summaries:
        if summary["eligible"] and summary["identity"]["config"]["method"] == "hipe":
            families[(canonical(selection_identity(summary["identity"])), summary["endpoint"])].append(summary)
    result = []
    for (family_key, endpoint), candidates in families.items():
        best = choose_candidates(candidates)
        config = best["identity"]["config"]
        ds, size, length = config["dataset"], config["model_size"], config["seq_len"]
        comparable = canonical(comparison_identity(best["identity"]))
        baseline_result = {}
        for method in ("nope", "alibi", "rope"):
            options = [g for g in baselines if g["endpoint"] == endpoint
                       and g["identity"]["config"]["method"] == method
                       and canonical(comparison_identity(g["identity"])) == comparable]
            if len(options) > 1:
                raise ValueError(f"Ambiguous completed baseline protocols for {ds}/{size}/{length}/{method}")
            if options:
                value = options[0]
                baseline_result[method] = {
                    "status": "completed", "n": value["n"],
                    "mean_ppl": value["mean_ppl"], "sample_std_ppl": value["sample_std_ppl"],
                    "seed_values": value["seed_values"], "run_paths": value["run_paths"],
                    "source_commits": value["source_commits"],
                    "micro_batch": value["identity"]["config"]["micro_batch_size"],
                    "backend": value["identity"]["config"]["attention_backend"],
                    "legacy_absolute_positions": value["identity"]["config"]["legacy_absolute_positions"],
                    "micro_batch_equal_to_hipe": value["identity"]["config"]["micro_batch_size"] == config["micro_batch_size"],
                }
            else:
                baseline_result[method] = unavailable_baseline(failed, ds, size, length, method, endpoint)
        rope = baseline_result["rope"]
        result.append({
            "dataset": ds, "model_size": size, "length": length, "endpoint": endpoint,
            "selection_family": digest(json.loads(family_key)),
            "selected_sigma": config["sigma"],
            "complete_sigma_candidates": len(candidates),
            "observed_sigmas": sorted(c["identity"]["config"]["sigma"] for c in candidates),
            "selection_status": "one_candidate_not_optimized" if len(candidates) == 1 else "best_observed_validation_mean",
            "selection_rule": "lowest arithmetic mean PPL across all three requested seeds; tie uses smaller sigma",
            "hipe_mean_ppl": best["mean_ppl"], "hipe_sample_std_ppl": best["sample_std_ppl"],
            "hipe_n": best["n"], "hipe_seed_values": best["seed_values"],
            "hipe_micro_batch": config["micro_batch_size"], "layer_threshold": config["layer_threshold"],
            "hipe_run_paths": best["run_paths"], "hipe_source_commits": best["source_commits"],
            "baselines": baseline_result,
            "relative_gain_vs_rope_percent": 100 * (1 - best["mean_ppl"] / rope["mean_ppl"]) if rope["status"] == "completed" else None,
            "candidates": [{k: c[k] for k in ("group_id", "n", "mean_ppl", "sample_std_ppl", "seed_values", "run_paths", "source_commits")} |
                           {"sigma": c["identity"]["config"]["sigma"]} for c in sorted(candidates, key=lambda c: c["identity"]["config"]["sigma"])],
            "comparison_controls": comparison_identity(best["identity"]),
        })
    return sorted(result, key=lambda r: (("last_interval", "final").index(r["endpoint"]),
                  ("wikitext103", "c4").index(r["dataset"]), ("20M", "60M").index(r["model_size"]),
                  r["length"], r["selection_family"]))


def csv_rows(results):
    rows = []
    for r in results:
        row = {k: r[k] for k in ("dataset", "model_size", "length", "endpoint", "selection_family",
               "selected_sigma", "complete_sigma_candidates", "selection_status",
               "hipe_mean_ppl", "hipe_sample_std_ppl", "hipe_n", "hipe_micro_batch",
               "layer_threshold", "relative_gain_vs_rope_percent")}
        row["observed_sigmas"] = ",".join(f"{v:g}" for v in r["observed_sigmas"])
        for method, label in (("nope", "legacy_nope_ape"), ("alibi", "alibi"), ("rope", "rope")):
            b = r["baselines"][method]
            for field in ("status", "mean_ppl", "sample_std_ppl", "n", "micro_batch", "backend"):
                row[label + "_" + field] = b.get(field)
            row[label + "_run_paths"] = ";".join(b["run_paths"])
        row["hipe_run_paths"] = ";".join(r["hipe_run_paths"])
        rows.append(row)
    return rows


def markdown(results):
    counts = sorted({r["complete_sigma_candidates"] for r in results})
    if counts == [1]:
        coverage = "**Current coverage is limited:** each completed setting currently has one sigma candidate. These rows are the best observed values available, not a completed sigma optimization. New L2048 sweep results can be merged after all three seeds finish."
    else:
        coverage = (f"**Completed sigma coverage varies by setting:** candidate counts range from {min(counts)} to {max(counts)}. "
                    "Each selected result is the best mean among completed three-seed candidates within its fixed protocol. "
                    "Settings with only one completed candidate remain unoptimized. Pending or failed sigma candidates do not enter selection; "
                    "read pending candidate-grid coverage separately from this completed-result ranking.")
    lines = [
        "# Best observed HIPE against baselines", "",
        "HIPE is selected once per dataset/model/length and fixed scientific protocol by the lowest mean validation PPL across seeds 6198, 1024 and 7. The best individual seed is never used for selection. Different sigmas are not averaged.",
        "",
        coverage,
        "",
        "Selection uses validation data and is therefore a tuning result. It is not an independent held-out test estimate. Mean and sample SD use the same three seeds. The primary endpoint is the last regular validation; final-checkpoint results are shown separately.",
    ]
    for endpoint in ("last_interval", "final"):
        lines += ["", "## " + ("Last regular validation" if endpoint == "last_interval" else "Final checkpoint"), "",
                  "| Dataset | Model | Length | Selected sigma | Completed candidates | Legacy NoPE+APE | ALiBi | RoPE | Best observed HIPE |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for r in results:
            if r["endpoint"] != endpoint:
                continue
            def fmt(b):
                return f"{b['mean_ppl']:.6f} +/- {b['sample_std_ppl']:.6f}" if b["status"] == "completed" else b["status"]
            vals = [fmt(r["baselines"][m]) for m in ("nope", "alibi", "rope")]
            vals += [f"{r['hipe_mean_ppl']:.6f} +/- {r['hipe_sample_std_ppl']:.6f}"]
            lines.append(f"| {'WikiText-103' if r['dataset']=='wikitext103' else 'C4'} | {r['model_size']} | {r['length']} | {r['selected_sigma']:g} | {r['complete_sigma_candidates']} | " + " | ".join(vals) + " |")
    lines += [
        "", "## Comparison and provenance limits", "",
        "- Different source commits are normalized only when every collected core-file hash matches. The checked files are the trainer, preprocessing, model-config builder, decoder and vendored model config. All resolved scientific settings and the full data/tokenizer/vocabulary/runtime identity remain part of grouping.",
        "- Incomplete/OOM attempts are excluded from numerical means. Successful corrected ALiBi retries provide the measured baseline where their original attempts failed. Multiple completed baseline protocols or duplicate completed seeds are rejected rather than choosing a favorable result.",
        "- C4 retains different micro-batches for HIPE and RoPE, while global batch remains 64. The CSV records this difference. It limits clean causal attribution.",
        "- The HIPE runs use threshold 3 (paper four/four hierarchy), while five historical C4 metric matches used threshold 4. This comparison keeps the current protocol; historical optima are not assumed to transfer.",
        "- Legacy NoPE+APE is labeled explicitly. OOM is an unavailable computational outcome and is never assigned PPL zero.",
        "",
        "The JSON includes every candidate sigma summary and all selected/baseline input run paths. This analysis does not launch jobs.",
    ]
    return "\n".join(lines) + "\n"


def self_test():
    """End-to-end synthetic selection check; these are not experiment results."""
    work = ROOT / "work/best-sigma-selftest"
    work.mkdir(parents=True, exist_ok=True)
    hashes = {name: "0" * 64 for name in CORE_FILES}
    core = {"core": hashes, "core_id": digest(hashes)}
    fixture = {"campaigns": [{"job": "synthetic", "core_file_sha256": hashes}], "runs": []}
    seed_set = [6198, 1024, 7]
    for length, sigma, values in [(512, 200, [1.0, 8.0, 9.0]),
                                   (512, 800, [5.0, 5.0, 5.0]),
                                   (1024, 700, [11.0, 12.0, 13.0])]:
        final_step = 100_000_000 // (64 * length)
        interval_step = final_step // 100 * 100
        for seed, value in zip(seed_set, values):
            output = f"synthetic_only/L{length}_sigma{sigma}_seed{seed}"
            config = {"dataset": "wikitext103", "model_size": "20M", "seq_len": length,
                      "method": "hipe", "sigma": sigma, "seed": seed, "output_dir": output,
                      "experiment_id": output, "source_notes": ["Synthetic unit fixture only"],
                      "max_tokens": 100_000_000, "global_batch_size": 64,
                      "micro_batch_size": 16, "max_train_steps": None,
                      "layer_threshold": 3, "attention_backend": "flash"}
            metrics = []
            for step, regular, final in [(interval_step, True, False), (final_step, False, True)]:
                metrics.append({"step": step, "actual_prediction_tokens": step * 64 * length,
                                "regular_validation": regular, "final": final,
                                "validation": {"tokens": 10, "loss_sum": math.log(value) * 10,
                                               "loss": math.log(value), "ppl": value}})
            fixture["runs"].append({
                "job": "synthetic", "output_dir": output, "config": config, "metrics": metrics,
                "completion.json": {"step": final_step}, "worker_status": {"returncode": 0},
                "run_manifest.json": {"data": {"processed_validation": f"synthetic-L{length}"},
                                      "tokenizer_file_sha256": {"fixture": "synthetic"},
                                      "vocabulary": {"raw": 64, "padded": 64},
                                      "runtime": {"packages": {"torch": "synthetic-only"}, "gpu": "synthetic-only"},
                                      "source": {"commit": "synthetic-only"}},
            })
    input_path = work / "synthetic_input.json"
    input_path.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    output = work / "results"
    command = [sys.executable, str(Path(__file__).resolve()), "--input", str(input_path),
               str(input_path), "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = json.loads((output / "best_sigma_comparison.json").read_text())
    assert result["unique_execution_records"] == 9
    assert len(result["results"]) == 4
    for row in result["results"]:
        if row["length"] == 512:
            assert row["selected_sigma"] == 800 and row["hipe_mean_ppl"] == 5
            assert row["complete_sigma_candidates"] == 2
            assert row["hipe_sample_std_ppl"] == 0
        else:
            assert row["selected_sigma"] == 700 and row["hipe_mean_ppl"] == 12
            assert row["complete_sigma_candidates"] == 1
    report = (output / "BEST_OBSERVED.md").read_text()
    assert "candidate counts range from 1 to 2" in report
    assert "each completed setting currently has one sigma candidate" not in report
    runs, campaigns, _ = load_inputs([input_path])
    duplicate = copy.deepcopy(runs[0]); duplicate["output_dir"] += "_duplicate"
    try:
        build_groups(runs + [duplicate], campaigns, seed_set)
        raise AssertionError("Duplicate completed seed was accepted")
    except ValueError as exc:
        assert "Duplicate completed seed" in str(exc)
    other_commit = copy.deepcopy(runs[0])
    other_commit["run_manifest.json"]["source"]["commit"] = "different-synthetic-commit"
    assert protocol_identity(runs[0], core) == protocol_identity(other_commit, core)
    changed = copy.deepcopy(core)
    changed["core"]["hipe/train.py"] = "1" * 64
    changed["core_id"] = digest(changed["core"])
    assert protocol_identity(runs[0], core) != protocol_identity(other_commit, changed)
    checks = {"kind": "synthetic code verification, not experimental PPL", "status": "passed",
              "end_to_end_cli": True, "best_three_seed_mean_not_best_single_seed": True,
              "separate_lengths_and_sigmas": True, "dynamic_coverage_text": True,
              "overlapping_inputs_deduplicated": True, "duplicate_seed_rejected": True,
              "core_hash_normalization_checked": True,
              "selection_example": "sigma200 mean6/min1; sigma800 mean5/min5 => selected800"}
    (work / "verification.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run synthetic end-to-end selection checks; no experiment jobs")
    parser.add_argument("--input", nargs="+", type=Path, default=[DEFAULT_INPUT])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[6198, 1024, 7])
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if len(args.seeds) != 3 or len(set(args.seeds)) != 3:
        raise ValueError("This comparison requires three distinct confirmation seeds")
    runs, campaigns, inputs = load_inputs(args.input)
    summaries, failed = build_groups(runs, campaigns, args.seeds)
    results = compare(summaries, failed)
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "selection": "minimum complete three-seed mean within a fixed non-sigma protocol",
        "primary_endpoint": "last_interval", "secondary_endpoint": "final",
        "input_files": inputs, "unique_execution_records": len(runs),
        "completed_run_records": sum("completion.json" in r for r in runs),
        "results": results,
        "all_fixed_sigma_summaries": summaries,
    }
    (args.output / "best_sigma_comparison.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    rows = csv_rows(results)
    with (args.output / "best_sigma_comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "BEST_OBSERVED.md").write_text(markdown(results), encoding="utf-8")
    print(f"Compared {len(results)} setting/endpoint rows from {len(runs)} unique execution records")
    print("Complete candidates per setting:", sorted({r["complete_sigma_candidates"] for r in results}))
    print("Output:", args.output)


if __name__ == "__main__":
    main()
