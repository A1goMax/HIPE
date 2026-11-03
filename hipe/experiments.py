"""Resolve and submit main/ablation configurations through one SLURM interface."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import uuid

from .spec import ExperimentConfig, legacy_micro_batch

ROOT = Path(__file__).resolve().parents[1]


def source_fingerprint():
    digest = hashlib.sha256()
    names = [*ROOT.glob("hipe/**/*.py"), *ROOT.glob("scripts/*.sh"), *ROOT.glob("configs/**/*.json")]
    names += [ROOT / "pyproject.toml", ROOT / "requirements.txt"]
    for path in sorted(p for p in names if p.is_file()):
        digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def resolve_suite(suite, *, data_root, artifacts_root, datasets=None, models=None,
                  lengths=None, methods=None, seeds=None, sigmas=None, batch_profile=None,
                  correct_nope=False):
    suite_path = ROOT / "configs/suites" / (suite + ".json")
    raw = json.loads(suite_path.read_text(encoding="utf-8"))
    result = []
    data_base = PurePosixPath(data_root) if str(data_root).startswith("/") else Path(data_root)
    artifact_base = PurePosixPath(artifacts_root) if str(artifacts_root).startswith("/") else Path(artifacts_root)
    matrix = raw.get("matrix")
    if matrix:
        cases = []
        for dataset in matrix["dataset"]:
            for model in matrix["model_size"]:
                for length in matrix["seq_len"]:
                    for method in matrix["method"]:
                        for seed in seeds or matrix["seed"]:
                            for sigma in (sigmas or matrix["sigma"]) if method == "hipe" else [raw["defaults"].get("sigma", 500)]:
                                cases.append({"dataset": dataset, "model_size": model, "seq_len": length,
                                              "method": method, "seed": seed, "sigma": sigma})
    else:
        cases = [dict(case, seed=seed) for case in raw["cases"] for seed in seeds or raw.get("seeds", [6198])]
    for case in cases:
        item = dict(raw["defaults"], **case)
        if datasets and item["dataset"] not in datasets:
            continue
        if models and item["model_size"] not in models:
            continue
        if lengths and item["seq_len"] not in lengths:
            continue
        if methods and item["method"] not in methods:
            continue
        if sigmas and item["method"] == "hipe" and item["sigma"] not in sigmas:
            continue
        dataset = item["dataset"]
        item["batch_profile"] = batch_profile or item["batch_profile"]
        batch_method = "hipe_learnable" if item.get("learnable_sigma") else item["method"]
        item.setdefault("micro_batch_size", legacy_micro_batch(dataset, batch_method, item["model_size"], item["seq_len"], item["batch_profile"]))
        if dataset == "c4":
            item.setdefault("data_pipeline", "legacy_c4")
            item.setdefault("num_proc", 8)
            item.setdefault("validation_num_proc", 4)
            item.setdefault("max_train_samples", 2_000_000)
            item.setdefault("max_val_samples", 10_000)
            item["data_path"] = str(data_base)
        else:
            item.setdefault("data_pipeline", "legacy_wiki")
            item.setdefault("num_proc", 4)
            item.setdefault("validation_num_proc", 4)
            item["data_path"] = str(data_base / "wikitext/raw")
        item["tokenizer_path"] = str(data_base / "wikitext/tokenizer")
        if item["method"] in {"nope", "xpos"}:
            # Preserve explicitly configured compatibility. New XPos defaults to
            # reference relative positions without learned absolute embeddings.
            item.setdefault("legacy_absolute_positions", item["method"] == "nope")
            if correct_nope:
                item["legacy_absolute_positions"] = False
        if item["method"] == "alibi":
            item["attention_backend"] = "sdpa"
        if item.get("attention_pattern", "global") != "global":
            item["attention_backend"] = "math"
        tag = case.get("case_name", item["method"])
        item.pop("case_name", None)
        if item.get("legacy_absolute_positions"):
            tag += "_legacyAPE"
        if item["method"] == "hipe":
            tag += f"_s{item['sigma']:g}_{item.get('schedule','bipartite')}"
            if item.get("layer_threshold") is not None:
                tag += f"_t{item['layer_threshold']}"
        elif item["method"] == "xpos":
            tag += f"_{item.get('xpos_variant', 'torchscale')}_b{item.get('xpos_scale_base', 512):g}"
        identity = f"{suite}_{dataset}_{item['model_size']}_L{item['seq_len']}_{tag}_seed{item['seed']}"
        item["experiment_id"] = identity
        item["output_dir"] = str(artifact_base / "runs" / identity)
        config = ExperimentConfig(**item).validate()
        result.append(config)
    if not result:
        raise ValueError("Filters selected no experiment conditions")
    ids = [config.experiment_id for config in result]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate experiment identities")
    return result


def submission_command(machine, gpus, manifest_path, log_dir, array_size=None, afterok=None):
    if gpus not in {1, 2}:
        raise ValueError("Request one or two GPUs")
    concurrency = gpus
    gpus = 1 if array_size is not None else gpus
    execution_root = Path(machine.get("repo_root", str(ROOT))) if os.name != "nt" else PurePosixPath(machine.get("repo_root", str(ROOT)))
    command = ["sbatch", "--parsable", "--partition=" + machine["partition"],
            f"--gres=gpu:{gpus}", "--nodes=1", "--ntasks=1",
            f"--cpus-per-task={machine['cpus_per_gpu'] * gpus}",
            f"--mem={machine['memory_gb_per_gpu'] * gpus}G",
            "--time=" + machine["time_limit"], "--job-name=hipe",
            "--chdir=" + str(execution_root), "--output=" + str(log_dir / "slurm-%j.out"),
            "--error=" + str(log_dir / "slurm-%j.err"),
            str(execution_root / "scripts/slurm_worker.sh"), str(manifest_path), machine["python"]]
    if afterok is not None:
        if not isinstance(afterok, int) or afterok <= 0:
            raise ValueError("afterok requires a positive SLURM job ID")
        command[2:2] = [f"--dependency=afterok:{afterok}", "--kill-on-invalid-dep=yes"]
    if array_size is not None:
        if array_size < 1:
            raise ValueError("Array must contain at least one condition")
        command.insert(2, f"--array=0-{array_size-1}%{concurrency}")
        command = [part.replace("slurm-%j", "slurm-%A_%a") for part in command]
    return command


def verify_publication(commit, remote="origin"):
    """Require the exact clean source head to be visible on the selected GitHub branch."""
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if not branch:
        raise RuntimeError("Publish from a named branch before submitting")
    url = subprocess.check_output(["git", "remote", "get-url", remote], cwd=ROOT, text=True).strip()
    ref = "refs/heads/" + branch
    output = subprocess.check_output(["git", "ls-remote", remote, ref], cwd=ROOT,
                                     text=True, timeout=45).strip().splitlines()
    if len(output) != 1 or output[0].split()[0] != commit:
        raise RuntimeError("Push the exact source commit to GitHub before submitting")
    return {"remote": remote, "url": url, "ref": ref, "commit": commit,
            "verified_at": datetime.now(timezone.utc).isoformat()}


def worker(manifest_path):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("The worker must run in a SLURM allocation")
    manifest = json.loads(Path(manifest_path).read_text())
    if source_fingerprint() != manifest["source_fingerprint"]:
        raise RuntimeError("Source/config files changed after submission")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    visible = [value.strip() for value in visible if value.strip()]
    gpus = manifest["gpus"]
    if len(visible) != gpus:
        raise RuntimeError("Allocated visible GPU count differs from campaign plan")
    log_root = Path(manifest["log_dir"])
    configs = manifest["configs"]
    array_index = None
    if manifest.get("array"):
        array_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        if not 0 <= array_index < len(configs):
            raise ValueError("Array index outside campaign")
        configs = [configs[array_index]]
    def slot(index):
        results = []
        for config_path in configs[index::gpus]:
            if source_fingerprint() != manifest["source_fingerprint"]:
                raise RuntimeError("Source changed while a queued condition was waiting")
            expected_hash = manifest.get("config_sha256", {}).get(config_path)
            if expected_hash and hashlib.sha256(Path(config_path).read_bytes()).hexdigest() != expected_hash:
                raise RuntimeError("Resolved configuration changed after submission")
            config = ExperimentConfig.load(config_path)
            env = os.environ.copy()
            env["HIPE_CAMPAIGN_MANIFEST"] = str(Path(manifest_path).resolve())
            env["CUDA_VISIBLE_DEVICES"] = visible[index]
            env["TOKENIZERS_PARALLELISM"] = "false"
            env["OMP_NUM_THREADS"] = str(manifest["cpus_per_gpu"])
            with (log_root / (config.experiment_id + ".log")).open("x") as log:
                result = subprocess.run([sys.executable, "-m", "hipe.train", "--config", config_path],
                                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            record = {"experiment_id": config.experiment_id, "returncode": result.returncode,
                      "gpu_slot": index, "output_dir": config.output_dir}
            (log_root / (config.experiment_id + ".status.json")).write_text(json.dumps(record, indent=2))
            results.append(record)
        return results
    with ThreadPoolExecutor(max_workers=gpus) as executor:
        results = [row for group in executor.map(slot, range(gpus)) for row in group]
    (log_root / (f"campaign_results-{array_index}.json" if array_index is not None else "campaign_results.json")).write_text(json.dumps(results, indent=2))
    if any(row["returncode"] for row in results):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "submit", "worker"])
    parser.add_argument("--suite", default="table2")
    parser.add_argument("--machine", default="labserver2")
    parser.add_argument("--dataset", nargs="+")
    parser.add_argument("--model-size", nargs="+")
    parser.add_argument("--seq-len", type=int, nargs="+")
    parser.add_argument("--method", nargs="+")
    parser.add_argument("--seed", type=int, nargs="+")
    parser.add_argument("--sigma", type=float, nargs="+")
    parser.add_argument("--batch-profile", choices=["january2026", "march2026", "attention2026"])
    parser.add_argument("--correct-nope", action="store_true",
                        help="Disable the explicitly named legacy learned-position compatibility")
    parser.add_argument("--data-root")
    parser.add_argument("--artifacts-root")
    parser.add_argument("--gpus", type=int, choices=[1, 2], default=1)
    parser.add_argument("--array", action="store_true", help="One GPU per Slurm array task; --gpus caps concurrency")
    parser.add_argument("--preflight", action="store_true", help="Two-step, small-data GPU execution check, labeled separately")
    parser.add_argument("--execute", action="store_true", help="Actually call sbatch; default is a preview")
    parser.add_argument("--require-published", action="store_true",
                        help="Require the exact source commit on the origin branch before sbatch")
    parser.add_argument("--afterok", type=int, help="Wait for every task of this SLURM check job to succeed")
    parser.add_argument("--output")
    parser.add_argument("--manifest")
    args = parser.parse_args()
    if args.action == "worker":
        worker(args.manifest)
        return
    machine = json.loads((ROOT / "configs/machines" / (args.machine + ".json")).read_text())
    data_root = args.data_root or machine["data_root"]
    artifact_value = args.artifacts_root or machine["artifacts_root"]
    artifacts_root = PurePosixPath(artifact_value) if str(artifact_value).startswith("/") else Path(artifact_value)
    cases = resolve_suite(args.suite, data_root=data_root, artifacts_root=artifacts_root,
                          datasets=args.dataset, models=args.model_size, lengths=args.seq_len,
                          methods=args.method, seeds=args.seed, sigmas=args.sigma,
                          batch_profile=args.batch_profile, correct_nope=args.correct_nope)
    if args.preflight:
        for config in cases:
            config.max_train_steps = 2
            config.max_train_samples = 512
            config.max_val_samples = 128
            config.eval_interval = config.checkpoint_interval = 2
            config.experiment_id += "_gpucheck2"
            config.output_dir += "_gpucheck2"
            config.source_notes.append("Two-step GPU preflight; excluded from paper experiment results")
    preview = {"suite": args.suite, "conditions": len(cases), "gpus_per_job": args.gpus,
               "execution": "independent one-GPU conditions; original batch per condition",
               "source_fingerprint": source_fingerprint(),
               "cases": [config.to_dict() for config in cases]}
    if args.action == "plan":
        content = json.dumps(preview, indent=2)
        if args.output:
            Path(args.output).write_text(content + "\n")
        else:
            print(content)
        return
    campaign = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    campaign_dir = artifacts_root / "campaigns" / campaign
    snapshot = campaign_dir / "source"
    execution_machine = dict(machine, repo_root=str(snapshot))
    command = submission_command(execution_machine, args.gpus, campaign_dir / "manifest.json", campaign_dir / "logs", len(cases) if args.array else None, afterok=args.afterok)
    if not args.execute:
        print(json.dumps({"conditions": len(cases), "sbatch": shlex.join(command),
                          "one_gpu_per_array_task": args.array, "max_concurrent_conditions": args.gpus,
                          "global_batch_size": sorted({c.global_batch_size for c in cases}),
                          "micro_batch_sizes": sorted({c.micro_batch_size for c in cases}),
                          "afterok": args.afterok, "submitted": False}, indent=2))
        return
    if os.name == "nt":
        raise RuntimeError("Submit from the deployed labserver2 checkout; Windows supports planning previews")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Commit source changes before submitting the frozen campaign")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    publication = verify_publication(source_commit) if args.require_published else None
    campaign_dir = Path(campaign_dir)
    campaign_dir.mkdir(parents=True, exist_ok=False)
    (campaign_dir / "logs").mkdir()
    subprocess.run(["git", "clone", "--shared", "--no-checkout", "--quiet", str(ROOT), str(snapshot)], check=True)
    subprocess.run(["git", "checkout", "--quiet", "--detach", source_commit], cwd=snapshot, check=True)
    config_paths = []
    config_hashes = {}
    for config in cases:
        config.output_dir += "_" + campaign
        path = campaign_dir / (config.experiment_id + ".json")
        path.write_text(json.dumps(config.to_dict(), indent=2) + "\n")
        config_paths.append(str(path))
        config_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"source_fingerprint": source_fingerprint(), "gpus": 1 if args.array else args.gpus,
                "array": args.array, "max_concurrent_conditions": args.gpus,
                "afterok": args.afterok,
                "source_root": str(snapshot), "source_commit": source_commit,
                "publication": publication, "config_sha256": config_hashes,
                "cpus_per_gpu": machine["cpus_per_gpu"], "configs": config_paths,
                "log_dir": str(campaign_dir / "logs")}
    (campaign_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    result = subprocess.check_output(command, text=True).strip()
    (campaign_dir / "submission.json").write_text(json.dumps({"job": result, "command": command}, indent=2))
    print(result)


if __name__ == "__main__":
    main()
