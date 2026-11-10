"""Build the public HIPE evidence projection from a verified local export."""
import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import zipfile

METHODS = {"hipe", "rope", "alibi", "nope"}
SECRET = re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----|(?:gh[pousr]_|github_pat_|hf_)[A-Za-z0-9_]{24,}|https?://[^/\s:@]+:[^/\s@]+@")
SUPPLEMENTARY_METRIC = re.compile(rb"(?i)\bxpos[ _-]+(?:mean|ppl|sample[ _-]*sd|wins|perplexity)\b")
SUPPLEMENTARY_ROW = re.compile(rb'(?i)"method"\s*:\s*"xpos"|(?:^|,)xpos(?:,|$)', re.M)
FIGURE_STUDIES = (
    "table2_20261103", "table2_sigma_final", "table2_joint_20261104",
    "table2_c4_20261105", "budget_exploration_20261108",
)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()

def check_text(data, label, result=False):
    if SECRET.search(data):
        raise ValueError(f"Credential pattern in {label}")
    if result and (SUPPLEMENTARY_METRIC.search(data) or SUPPLEMENTARY_ROW.search(data)):
        raise ValueError(f"Supplementary comparison in {label}")

def check_snapshot(data, label):
    with zipfile.ZipFile(BytesIO(data)) as z:
        for item in z.infolist():
            if item.is_dir():
                continue
            name = item.filename
            body = z.read(item)
            if name.endswith(".xlsx"):
                with zipfile.ZipFile(BytesIO(body)) as book:
                    for path in book.namelist():
                        if path.endswith(".xml") and SUPPLEMENTARY_METRIC.search(book.read(path)):
                            raise ValueError(f"Supplementary workbook column in {label}/{name}")
            elif Path(name).suffix in {".md",".json",".jsonl",".csv",".tsv",".py",".sh",".yaml",".yml",".txt"}:
                check_text(body, f"{label}/{name}", name.startswith("experiments/results/") or name.endswith("REPORT.md"))

def project(source, destination, workspace):
    source = source.resolve()
    if destination.exists():
        raise FileExistsError(f"Use an empty destination: {destination}")
    index = json.loads((source/"index.json").read_text(encoding="utf-8"))
    original = json.loads((source/"file_manifest.json").read_text(encoding="utf-8"))
    tasks = [r for r in index["tasks"] if r["method"] in METHODS]
    selected_snapshots = {r["snapshot"]["sha256"] for r in tasks if r.get("snapshot")}
    file_manifest = {}
    destination.mkdir(parents=True)
    def put(name, data):
        path = destination/name
        if not path.resolve().is_relative_to(destination.resolve()) or path.exists():
            raise ValueError(f"Unsafe or repeated archive path {name}")
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
        file_manifest[name] = {"sha256":sha(data),"bytes":len(data)}
    def load(name):
        path=source/name
        if not path.resolve().is_relative_to(source):
            raise ValueError(f"Invalid archive path {name}")
        body=path.read_bytes()
        if sha(body)!=original[name]["sha256"]:
            raise ValueError(f"Source archive checksum mismatch {name}")
        return body
    snapshots={}
    for digest in sorted(selected_snapshots):
        entry=index["snapshots"][digest]
        if not entry["git_member_bytes_match"]:
            raise ValueError(f"Review source snapshot {digest} before publication")
        body=load(entry["path"])
        check_snapshot(body,entry["path"])
        put(entry["path"],body)
        snapshots[digest]=entry
    for row in tasks:
        for alias,name in row["files"].items():
            body=load(name)
            check_text(body,name,result=alias in {"metrics.jsonl","worker.log","slurm.out","slurm.err","worker_result.json"})
            put(name,body)
    campaigns=[]
    for c in index["campaigns"]:
        rows=[r for r in tasks if r["job"]==c["job"]]
        metadata={k:v for k,v in c.items() if k not in {"manifest","submission","scheduler","planned_tasks"}}
        metadata["public_tasks"]=len(rows)
        metadata["original_manifest_sha256"]=original[c["manifest"]]["sha256"]
        metadata["original_submission_sha256"]=original[c["submission"]]["sha256"]
        metadata["tasks"]=[r["id"] for r in rows]
        metadata["scheduler_records"]=[{"task_index":r["task_index"],**r["scheduler"]} for r in rows if r["scheduler"]]
        path=f"campaigns/{c['job']}/campaign.json"
        put(path,encode(metadata))
        campaigns.append({"job":c["job"],"campaign":c["campaign"],"source_commit":c["source_commit"],
                          "path":path,"public_tasks":len(rows)})
    figure_count=0
    for study in FIGURE_STUDIES:
        folder=workspace/"outputs"/study
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".png",".svg"} or "xpos" in path.name.lower():
                continue
            if study=="table2_joint_20261104" and not path.stem.startswith(("HIPE_vs_RoPE","Targeted_sigma_layers")):
                continue
            body=path.read_bytes()
            check_text(body,str(path),result=True)
            put(f"figures/{study}/{path.name}",body)
            figure_count+=1
    counts={"planned_tasks":len(tasks),"observed":sum(r["observed"] for r in tasks),
            "completed":sum(r["completed"] for r in tasks),
            "completed_research_runs":sum(r["completed"] and not r["preflight"] for r in tasks),
            "completed_preflights":sum(r["completed"] and r["preflight"] for r in tasks),
            "figures":figure_count,
            "observed_by_method":dict(Counter(r["method"] for r in tasks if r["observed"]))}
    put("index.json",encode({"schema_version":1,"scope":"Public Table 2 method set",
        "repo_head":index["repo_head"],"campaigns":campaigns,"tasks":tasks,
        "snapshots":snapshots,"counts":counts}))
    (destination/"file_manifest.json").write_bytes(encode(file_manifest))
    print(json.dumps({"public_projection":str(destination),"files":len(file_manifest),**counts}))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True,type=Path)
    parser.add_argument("--destination",required=True,type=Path)
    parser.add_argument("--workspace",required=True,type=Path)
    args=parser.parse_args()
    project(args.source,args.destination,args.workspace)
