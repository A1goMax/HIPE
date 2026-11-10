"""Export the labserver2 HIPE run evidence and code without checkpoint tensors."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile

RUN_FILES = ("run_config.json", "model_config.json", "run_manifest.json",
             "metrics.jsonl", "completion.json")
ANALYSIS_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".xlsx", ".png",
                     ".svg", ".pdf", ".md", ".txt", ".log", ".out", ".err", ".npz", ".npy", ".py"}
SECRET_NAME = re.compile(r"(?i)(?:^|[._-])(credentials?|secrets?|passwords?|id_rsa|id_ed25519)(?:$|[._-])")

def sha(data):
    return hashlib.sha256(data).hexdigest()

def json_bytes(data):
    return (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])

def scheduler(job):
    cmd = ["sacct", "-X", "-j", job, "-n", "-P",
           "--format=JobID%64,JobIDRaw,State%32,ExitCode,Elapsed,Start,End"]
    process = subprocess.run(cmd, capture_output=True, check=True)
    by_index = {}
    for line in process.stdout.decode().splitlines():
        job_label, raw_id, state, exit_code, elapsed, start, end = line.split("|")
        match = re.fullmatch(re.escape(job) + r"_(\d+|\[[\d,:\-%]+\])", job_label)
        if not match:
            continue
        row = {"job_id": job_label, "raw_job_id": raw_id, "state": state,
               "exit_code": exit_code, "elapsed": elapsed, "start": start, "end": end}
        spec = match[1]
        if spec.isdigit():
            by_index[int(spec)] = row
            continue
        for part in spec[1:-1].split("%", 1)[0].split(","):
            start_text, sep, end_text = part.partition("-")
            if sep:
                last, step_sep, stride = end_text.partition(":")
                for index in range(int(start_text), int(last) + 1, int(stride) if step_sep else 1):
                    by_index.setdefault(index, row)
            else:
                by_index.setdefault(int(start_text), row)
    return by_index, process.stdout

def git_snapshot_match(repo, commit, data):
    if not commit:
        return False, []
    try:
        tree = git(repo, "ls-tree", "-rz", commit)
    except subprocess.CalledProcessError:
        return False, []
    expected = {}
    for line in tree.split(b"\0"):
        if not line:
            continue
        metadata, name = line.split(b"\t", 1)
        mode, kind, blob = metadata.decode().split()
        if kind == "blob":
            expected[name.decode("utf-8", errors="surrogateescape")] = blob
    from io import BytesIO
    mismatches = []
    with zipfile.ZipFile(BytesIO(data)) as z:
        seen = set()
        for item in z.infolist():
            if item.is_dir():
                continue
            name = item.filename
            body = z.read(item)
            blob = hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()
            if name in seen or expected.get(name) != blob:
                mismatches.append(name)
            seen.add(name)
    return not mismatches, mismatches

def export(journal, repo, destination):
    files = {}
    snapshots = {}
    tasks = []
    campaigns = []
    anomalies = []
    matched_run_paths = set()
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        def add(name, data, source):
            if name in files:
                raise ValueError(f"Duplicate archive path: {name}")
            z.writestr(name, data)
            files[name] = {"sha256": sha(data), "bytes": len(data), "source": str(source)}
            return name
        def copy(name, p, allowed_root):
            if p.is_symlink() or not p.resolve().is_relative_to(allowed_root.resolve()):
                raise ValueError(f"Unexpected path outside archive scope: {p}")
            return add(name, p.read_bytes(), p)
        def snapshot(path, manifest, run_dir):
            if not path.is_file():
                return None
            if path.is_symlink() or not path.resolve().is_relative_to(run_dir.resolve()):
                raise ValueError(f"Unexpected source snapshot path: {path}")
            digest = manifest.get("snapshot_sha256")
            body = path.read_bytes()
            actual = sha(body)
            if actual != digest:
                anomalies.append({"path": str(path), "reason": "source snapshot digest mismatch"})
            if actual not in snapshots:
                matches, differences = git_snapshot_match(repo, manifest.get("commit"), body)
                location = add(f"sources/{actual}.zip", body, path)
                snapshots[actual] = {"path": location, "commit": manifest.get("commit"),
                    "git_member_bytes_match": matches, "git_mismatches": differences}
            return {"archive_path": snapshots[actual]["path"], "sha256": actual,
                    "matches_recorded_sha256": actual == digest}
        def archive_run(row, run_dir, config):
            if not run_dir.is_dir():
                return
            if run_dir.is_symlink() or not run_dir.resolve().is_relative_to((journal/"runs").resolve()):
                raise ValueError(f"Unexpected run directory {run_dir}")
            matched_run_paths.add(run_dir.resolve())
            prefix = row["archive_dir"]
            manifest = None
            actual = None
            for name in RUN_FILES:
                f = run_dir/name
                if f.is_file():
                    row["files"][name] = copy(f"{prefix}/{name}", f, journal/"runs")
                    if name == "run_manifest.json":
                        manifest = json.loads(f.read_bytes())
                    if name == "run_config.json":
                        actual = json.loads(f.read_bytes())
            row["observed"] = True
            row["completed"] = (run_dir/"completion.json").is_file()
            row["binary_inventory"] = {p.name:p.stat().st_size for p in run_dir.iterdir()
                                       if p.is_file() and p.suffix in {".pt",".pth",".safetensors"}}
            if manifest:
                source = manifest["source"]
                row["source_commit"] = source.get("commit")
                row["source_fingerprint"] = source.get("fingerprint")
                row["source_status"] = source.get("status")
                row["method"] = manifest["experiment"]["method"]
                row["config_matches_manifest"] = actual == manifest["experiment"]
                row["config_matches_campaign"] = config == actual if config else None
                row["snapshot"] = snapshot(run_dir/source["snapshot"], source, run_dir)
                if actual != manifest["experiment"] or (config and config != actual):
                    anomalies.append({"path": str(run_dir), "reason": "configuration disagreement"})
            else:
                anomalies.append({"path": str(run_dir), "reason": "run manifest absent"})
        for folder in sorted((journal/"campaigns").iterdir()):
            if not folder.is_dir() or folder.is_symlink():
                continue
            mp, sp = folder/"manifest.json", folder/"submission.json"
            if not (mp.is_file() and sp.is_file()):
                anomalies.append({"path": str(folder), "reason": "campaign metadata absent"})
                continue
            m = json.loads(mp.read_bytes())
            job = str(json.loads(sp.read_bytes())["job"]).split(";")[0]
            source = Path(m["source_root"])
            if not source.resolve().is_relative_to(folder.resolve()):
                raise ValueError(f"Unexpected frozen source root {source}")
            commit = git(source,"rev-parse","HEAD").decode().strip()
            if commit != m["source_commit"]:
                anomalies.append({"path": str(source), "reason": "frozen source HEAD disagreement"})
            state, raw_sacct = scheduler(job)
            cprefix = f"campaigns/{job}"
            campaigns.append({"job": job, "campaign":folder.name, "planned_tasks":len(m["configs"]),
                "source_commit":m["source_commit"], "frozen_head":commit,
                "source_fingerprint":m.get("source_fingerprint"),
                "source_status":git(source,"status","--porcelain").decode(),
                "manifest":copy(f"{cprefix}/manifest.json",mp,folder),
                "submission":copy(f"{cprefix}/submission.json",sp,folder),
                "scheduler":add(f"{cprefix}/scheduler.tsv",raw_sacct,f"sacct {job}")})
            logs = Path(m["log_dir"])
            if not logs.resolve().is_relative_to(folder.resolve()):
                raise ValueError(f"Unexpected log root {logs}")
            linked_logs = set()
            for index, path in enumerate(m["configs"]):
                p = Path(path)
                cfg = json.loads(p.read_bytes())
                root = Path(cfg["output_dir"])
                if not root.resolve().is_relative_to((journal/"runs").resolve()):
                    raise ValueError(f"Unexpected output dir {root}")
                prefix = f"runs/{job}/{index:04d}"
                row = {"id":f"{job}_{index}", "job":job, "task_index":index, "campaign":folder.name,
                       "archive_dir":prefix, "experiment_id":cfg["experiment_id"], "method":cfg["method"],
                       "dataset":cfg["dataset"], "model_size":cfg["model_size"], "sequence_length":cfg["seq_len"],
                       "budget_tokens":cfg["max_tokens"], "seed":cfg["seed"],
                       "sigma":cfg.get("sigma") if cfg["method"]=="hipe" else None,
                       "schedule":cfg.get("schedule"), "layer_threshold":cfg.get("layer_threshold"),
                       "preflight":cfg.get("max_train_steps") is not None, "output_dir":str(root),
                       "source_commit":m["source_commit"], "scheduler":state.get(index),
                       "observed":False,"completed":False,"files":{}}
                row["files"]["planned_config.json"] = copy(f"{prefix}/planned_config.json",p,folder)
                expected = m.get("config_sha256",{}).get(str(p))
                row["planned_config_sha256"] = files[row["files"]["planned_config.json"]]["sha256"]
                row["planned_config_hash_verified"] = expected == row["planned_config_sha256"] if expected else None
                if expected and not row["planned_config_hash_verified"]:
                    anomalies.append({"path":str(p), "reason":"planned config digest mismatch"})
                exp = cfg["experiment_id"]
                log_names = {f"worker.log":f"{exp}.log","worker_status.json":f"{exp}.status.json",
                             "worker_result.json":f"campaign_results-{index}.json",
                             "slurm.out":f"slurm-{job}_{index}.out","slurm.err":f"slurm-{job}_{index}.err"}
                for alias, name in log_names.items():
                    f = logs/name
                    if f.is_file():
                        linked_logs.add(f)
                        row["files"][alias] = copy(f"{prefix}/{alias}",f,folder)
                archive_run(row,root,cfg)
                tasks.append(row)
            for f in sorted(logs.iterdir()):
                if f.is_file() and f not in linked_logs:
                    copy(f"{cprefix}/unassigned_logs/{f.name}",f,folder)
        for run_dir in sorted((journal/"runs").iterdir()):
            if not run_dir.is_dir() or run_dir.resolve() in matched_run_paths:
                continue
            index = len(tasks)
            row = {"id":f"orphan_{index}", "archive_dir":f"runs/orphan/{index}",
                   "output_dir":str(run_dir), "observed":False,"completed":False,"files":{}}
            archive_run(row,run_dir,None)
            tasks.append(row)
        with tempfile.TemporaryDirectory(prefix="hipe-archive-code-") as tmp:
            bundle = Path(tmp)/"HIPE.bundle"
            git(repo,"bundle","create",str(bundle),"--branches","--tags")
            copy("code/HIPE.bundle",bundle,Path(tmp))
        add("code/git_refs.txt",git(repo,"for-each-ref","--format=%(refname) %(objectname)","refs/heads","refs/tags"),repo)
        for sub in ["outputs","artifacts/analysis","artifacts/history","artifacts/sources"]:
            root = repo/sub
            if not root.exists():
                continue
            for p in sorted(root.rglob("*")):
                if (p.is_file() and not p.is_symlink() and not p.name.startswith("~$")
                    and not p.name.endswith(".inspect.ndjson") and p.suffix.lower() in ANALYSIS_SUFFIXES
                    and not SECRET_NAME.search(p.name)):
                    copy(f"analysis/server/{p.relative_to(repo).as_posix()}",p,root)
        for p in sorted(repo.parent.glob("The Hippocampal Place Field Gradient*.pdf")):
            if p.is_file() and not p.is_symlink():
                copy(f"analysis/manuscripts/server-{sha(p.read_bytes())[:12]}.pdf",p,repo.parent)
        index = {"schema_version":1,"source_repository":str(repo),"journal_root":str(journal),
            "repo_head":git(repo,"rev-parse","HEAD").decode().strip(),
            "campaigns":campaigns,"tasks":tasks,"snapshots":snapshots,"anomalies":anomalies,
            "counts":{"planned_or_orphan_tasks":len(tasks),"observed":sum(r["observed"] for r in tasks),
            "completed":sum(r["completed"] for r in tasks),"by_method":dict(Counter(r.get("method") for r in tasks if r["observed"]))}}
        add("index.json",json_bytes(index),"generated")
        z.writestr("file_manifest.json",json_bytes(files))
    print(json.dumps({"path":str(destination),"bytes":destination.stat().st_size,**index["counts"],"snapshots":len(snapshots),"anomalies":len(anomalies),"files":len(files)}))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal",type=Path,default=Path("/data/qijunrong/03-proj/HIPE-journal"))
    parser.add_argument("--repo",type=Path,default=Path("/home/qijunrong/03-proj/HIPE"))
    parser.add_argument("--archive",type=Path,required=True)
    args=parser.parse_args()
    export(args.journal.resolve(),args.repo.resolve(),args.archive)
