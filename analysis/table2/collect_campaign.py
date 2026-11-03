"""Collect recorded campaign metrics on labserver2; no training or job changes."""
from pathlib import Path
import argparse
import json
import hashlib
import subprocess

CAMPAIGNS=Path("/data/qijunrong/03-proj/HIPE-journal/campaigns")
CORE_FILES=["hipe/train.py","hipe/data.py","hipe/config.py","hipe/olmo/model.py","hipe/olmo/config.py"]


def collect(jobs):
    wanted={str(job) for job in jobs};result={"campaigns":[],"runs":[]};found=set()
    for submission in sorted(CAMPAIGNS.glob("*/submission.json")):
        meta=json.loads(submission.read_text());job=str(meta["job"]).split(";")[0]
        if job not in wanted:continue
        found.add(job);folder=submission.parent;manifest=json.loads((folder/"manifest.json").read_text())
        hashes={name:hashlib.sha256((Path(manifest["source_root"])/name).read_bytes()).hexdigest() for name in CORE_FILES}
        accounting=subprocess.run(["sacct","-j",job,"--format=JobID,State,ExitCode,Elapsed,Start,End","--noheader","--parsable2"],capture_output=True,text=True)
        result["campaigns"].append({"job":job,"path":str(folder),"manifest":manifest,
                                    "core_file_sha256":hashes,"sacct":accounting.stdout or accounting.stderr})
        for index,path in enumerate(manifest["configs"]):
            config=json.loads(Path(path).read_text());out=Path(config["output_dir"])
            status=folder/"logs"/(config["experiment_id"]+".status.json")
            completion=out/"completion.json"
            if not status.exists() and not completion.exists():
                continue  # canceled/unstarted tasks are not measured runs
            row={"job":job,"task_index":index,"config":config,"output_dir":str(out),
                 "worker_status":json.loads(status.read_text()) if status.exists() else None}
            for name in ["run_manifest.json","completion.json"]:
                if (out/name).exists():row[name]=json.loads((out/name).read_text())
            metrics=out/"metrics.jsonl"
            row["metrics"]=[json.loads(line) for line in metrics.read_text().splitlines() if line.strip()] if metrics.exists() else []
            if not completion.exists():
                log=folder/"logs"/(config["experiment_id"]+".log")
                row["failure_tail"]=log.read_bytes()[-7000:].decode(errors="replace") if log.exists() else None
            result["runs"].append(row)
    if found!=wanted:raise ValueError(f"Campaigns not found: {sorted(wanted-found)}")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs",nargs="+",required=True,type=int)
    parser.add_argument("--output",required=True)
    args=parser.parse_args();data=collect(args.jobs)
    text=json.dumps(data,indent=2)+"\n"
    if args.output=="-":print(text,end="")
    else:
        target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix(target.suffix+".tmp")
        temporary.write_text(text,encoding="utf-8");temporary.replace(target)
        print(f"Collected {len(data['runs'])} executions from {len(data['campaigns'])} campaigns")


if __name__=="__main__":
    main()
