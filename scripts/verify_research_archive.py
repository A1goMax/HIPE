"""Verify a local or public HIPE archive without contacting the lab server."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

from export_research_archive import git_snapshot_match

def verify(root, repo):
    root=root.resolve()
    files=json.loads((root/"file_manifest.json").read_text(encoding="utf-8"))
    for name, meta in files.items():
        p=root/name
        if p.is_symlink() or not p.resolve().is_relative_to(root):
            raise ValueError(f"Unsafe archive path {name}")
        data=p.read_bytes()
        if len(data)!=meta["bytes"] or hashlib.sha256(data).hexdigest()!=meta["sha256"]:
            raise ValueError(f"File verification failed: {name}")
    index=json.loads((root/"index.json").read_text(encoding="utf-8"))
    seen=set()
    metrics=0
    observed=0
    completed=0
    for task in index["tasks"]:
        assert task["id"] not in seen
        seen.add(task["id"])
        assert all(p in files for p in task["files"].values())
        if "scope" in index:
            assert task["method"] in {"hipe","rope","alibi","nope"}
        if not task["observed"]:
            continue
        observed+=1
        paths=task["files"]
        actual=json.loads((root/paths["run_config.json"]).read_text(encoding="utf-8"))
        run=json.loads((root/paths["run_manifest.json"]).read_text(encoding="utf-8"))
        assert actual==run["experiment"] and actual["method"]==task["method"]
        assert run["source"]["commit"]==task["source_commit"]
        assert task["snapshot"]["matches_recorded_sha256"]
        assert task["snapshot"]["sha256"]==run["source"]["snapshot_sha256"]
        assert files[task["snapshot"]["archive_path"]]["sha256"]==task["snapshot"]["sha256"]
        if "planned_config.json" in paths:
            planned=json.loads((root/paths["planned_config.json"]).read_text(encoding="utf-8"))
            assert actual==planned
        for line in (root/paths["metrics.jsonl"]).read_text(encoding="utf-8").splitlines():
            metric=json.loads(line)
            v=metric.get("validation")
            if v:
                assert v["tokens"]>0 and math.isclose(v["ppl"],math.exp(v["loss_sum"]/v["tokens"]),rel_tol=1e-12)
                metrics+=1
        if task["completed"]:
            assert "completion.json" in paths
            completed+=1
    assert observed==index["counts"]["observed"] and completed==index["counts"]["completed"]
    if repo:
        for commit in {t["source_commit"] for t in index["tasks"]}:
            subprocess.run(["git","-C",str(repo),"cat-file","-e",commit+"^{commit}"],check=True)
        for entry in index["snapshots"].values():
            matches,differences=git_snapshot_match(repo,entry["commit"],(root/entry["path"]).read_bytes())
            if not matches:
                raise ValueError(f"Snapshot tree mismatch: {differences}")
    print(json.dumps({"verified_files":len(files),"unique_task_ids":len(seen),"observed_runs":observed,
                      "completed_runs":completed,"validation_records":metrics,
                      "snapshot_git_verification":bool(repo)},indent=2))
if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("--repo",type=Path)
    args=parser.parse_args()
    verify(args.archive,args.repo)
