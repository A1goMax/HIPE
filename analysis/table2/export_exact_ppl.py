"""Export exact logged PPL and seed summaries, separated by sigma and protocol."""
from pathlib import Path
import json
import math
import statistics
import hashlib
import csv
import collections

ROOT=Path(__file__).resolve().parents[2]
INPUT=ROOT/"artifacts/analysis/table2_20261103/raw_runs.json"
OUT=ROOT/"outputs/exact_ppl_20261103"
SEEDS=[6198,1024,7]
ENDPOINTS=["final","last_interval"]
SIGMAS=[50,100,200,300,500,700,1000]


def export_csv(name,rows):
    if not rows:return
    with (OUT/name).open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def effective_config(config):
    result={k:v for k,v in config.items() if k not in {"seed","output_dir","experiment_id","source_notes"}}
    if result["method"]!="hipe":
        result["sigma"]=None
        result["schedule"]=None
        result["layer_threshold"]=None
        result["layer_sigmas"]=None
        result["learnable_sigma"]=False
    return result


def endpoint(run,name):
    candidates=[r for r in run["metrics"] if r["final"]] if name=="final" else [r for r in run["metrics"] if r["regular_validation"]]
    metric=max(candidates,key=lambda r:r["step"]);value=metric["validation"]
    recomputed=math.exp(value["loss_sum"]/value["tokens"])
    assert math.isclose(value["ppl"],recomputed,rel_tol=1e-12)
    return metric,recomputed


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    raw=json.loads(INPUT.read_text(encoding="utf-8"))
    groups={};attempts=[];details=[];curves=[]
    for run in raw["runs"]:
        config=run["config"];protocol=effective_config(config)
        identity={"config":protocol,"source_commit":run.get("run_manifest.json",{}).get("source",{}).get("commit") or
                  next(c["manifest"]["source_commit"] for c in raw["campaigns"] if c["job"]==run["job"])}
        text=json.dumps(identity,sort_keys=True)
        group_id=hashlib.sha256(text.encode()).hexdigest()[:14]
        group=groups.setdefault(text,{"id":group_id,"identity":identity,"runs":[],"failed":[]})
        completed="completion.json" in run
        failed_reason=None if completed else ("OOM" if "OutOfMemoryError" in (run.get("failure_tail") or "") else "incomplete_or_failed")
        attempt={"group_id":group_id,"dataset":config["dataset"],"model_size":config["model_size"],
                 "length":config["seq_len"],"method":config["method"],"sigma":protocol["sigma"],
                 "seed":config["seed"],"status":"completed" if completed else failed_reason,
                 "job":run["job"],"task_index":run["task_index"],"backend":config["attention_backend"],
                 "micro_batch":config["micro_batch_size"],"global_batch":config["global_batch_size"],
                 "threshold":protocol["layer_threshold"],"source_commit":identity["source_commit"],
                 "output_dir":run["output_dir"]}
        attempts.append(attempt)
        group["runs" if completed else "failed"].append(run)
        for metric in run["metrics"]:
            curves.append({**attempt,"step":metric["step"],"prediction_tokens":metric["actual_prediction_tokens"],
                           "validation_ppl":metric["validation"]["ppl"],
                           "validation_ppl_text":repr(metric["validation"]["ppl"]),
                           "regular_validation":metric["regular_validation"],"final":metric["final"]})
        if not completed:continue
        for name in ENDPOINTS:
            metric,recomputed=endpoint(run,name)
            details.append({**attempt,"endpoint":name,"step":metric["step"],
                            "prediction_tokens":metric["actual_prediction_tokens"],
                            "validation_tokens":metric["validation"]["tokens"],
                            "ppl":metric["validation"]["ppl"],
                            "ppl_exact_text":repr(metric["validation"]["ppl"]),
                            "ppl_from_loss":recomputed,
                            "recomputed_minus_logged":recomputed-metric["validation"]["ppl"]})
    summaries=[]
    for group in groups.values():
        config=group["identity"]["config"];runs=group["runs"]
        assert len({r["config"]["seed"] for r in runs})==len(runs),"Duplicate seed in a fixed-sigma protocol"
        for name in ENDPOINTS:
            by_seed={r["config"]["seed"]:endpoint(r,name)[0]["validation"]["ppl"] for r in runs}
            values=list(by_seed.values());steps={endpoint(r,name)[0]["step"] for r in runs}
            assert len(steps)<=1,"Mismatched endpoints in group"
            row={"group_id":group["id"],"dataset":config["dataset"],"model_size":config["model_size"],
                 "length":config["seq_len"],"method":config["method"],"sigma":config["sigma"],
                 "endpoint":name,"n":len(values),"seed_6198":by_seed.get(6198),
                 "seed_1024":by_seed.get(1024),"seed_7":by_seed.get(7),
                 "mean_ppl":statistics.mean(values) if values else None,
                 "sample_std_ppl":statistics.stdev(values) if len(values)>1 else None,
                 "status":"completed" if values else ("OOM" if all("OutOfMemoryError" in (r.get("failure_tail") or "") for r in group["failed"]) else "failed"),
                 "schedule":config["schedule"],"threshold":config["layer_threshold"],
                 "micro_batch":config["micro_batch_size"],"global_batch":config["global_batch_size"],
                 "backend":config["attention_backend"],"step":next(iter(steps)) if steps else None,
                 "source_commit":group["identity"]["source_commit"]}
            summaries.append(row)
    ordering=lambda r:(["wikitext103","c4"].index(r["dataset"]),["20M","60M"].index(r["model_size"]),r["length"],r["method"],r["sigma"] or -1,ENDPOINTS.index(r["endpoint"]),r["source_commit"])
    summaries.sort(key=ordering)
    hipe=[r for r in summaries if r["method"]=="hipe"]
    assert len(hipe)==24 and len(summaries)==100 and len(details)==264
    coverage=[]
    for dataset in ["wikitext103","c4"]:
        for model in ["20M","60M"]:
            for length in [512,1024,2048]:
                for sigma in SIGMAS:
                    matches=[r for r in hipe if (r["dataset"],r["model_size"],r["length"],r["sigma"],r["endpoint"])==(dataset,model,length,sigma,"final")]
                    assert len(matches)<=1
                    found=matches[0] if matches else None
                    coverage.append({"dataset":dataset,"model_size":model,"length":length,"sigma":sigma,
                                     "status":"completed" if found else "not_run",
                                     "n":found["n"] if found else 0,
                                     "mean_final_ppl":found["mean_ppl"] if found else None,
                                     "sample_std_final_ppl":found["sample_std_ppl"] if found else None})
    export_csv("HIPE_by_sigma.csv",hipe)
    export_csv("All_configurations.csv",summaries)
    export_csv("Exact_per_seed.csv",details)
    export_csv("All_attempts.csv",attempts)
    export_csv("All_validation_records.csv",curves)
    export_csv("Sigma_coverage.csv",coverage)
    notes=[
       "Mean and sample SD (n-1) are calculated across seeds for one fixed dataset/model/length/method/sigma/protocol/endpoint.",
       "Different sigma values are never pooled. Baselines have no effective sigma; their default unused config sigma is excluded.",
       "Current completed HIPE results contain only one sigma per dataset/model/length. Other candidate sigmas are not run; no mean or SD is invented.",
       "Final is the evaluation of the final checkpoint near the 100M-token cap; last_interval is the earlier regular validation used in the previous report.",
       "Exact values come directly from the logged validation PPL. Full-precision decimal strings are included in the raw details CSV.",
       "Original failed ALiBi protocols and corrected retries are separate configurations. OOM is not a numeric PPL.",
    ]
    data={"notes":notes,"source":str(INPUT),"input_sha256":hashlib.sha256(INPUT.read_bytes()).hexdigest(),
          "seeds":SEEDS,"hipe_rows":hipe,"all_rows":summaries,"details":details,
          "attempts":attempts,"coverage":coverage,"curve_rows":curves}
    (OUT/"workbook_data.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
    lines=["# Exact PPL by fixed sigma","",*["- "+note for note in notes],"",
           "## HIPE final checkpoint PPL","",
           "| Corpus | Model | Length | Sigma | Seed 6198 | Seed 1024 | Seed 7 | Mean | Sample SD |",
           "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in hipe:
        if row["endpoint"]!="final":continue
        lines.append("| "+("WikiText" if row["dataset"]=="wikitext103" else "C4")+f" | {row['model_size']} | {row['length']} | {row['sigma']:g} | "+" | ".join(f"{row[key]:.6f}" for key in ["seed_6198","seed_1024","seed_7","mean_ppl","sample_std_ppl"])+" |")
    for name in ENDPOINTS:
        lines += ["",f"## All completed configurations: {name}","",
                  "| Corpus | Model | Length | Method | Sigma | Mean PPL | Sample SD | n | Backend | Micro-batch |",
                  "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: |"]
        for row in summaries:
            if row["endpoint"]!=name:continue
            label="Legacy NoPE+APE" if row["method"]=="nope" else row["method"]
            sig=f"{row['sigma']:g}" if row["sigma"] is not None else "n.a."
            mean=f"{row['mean_ppl']:.6f}" if row["mean_ppl"] is not None else row["status"]
            sd=f"{row['sample_std_ppl']:.6f}" if row["sample_std_ppl"] is not None else "n.a."
            lines.append(f"| {row['dataset']} | {row['model_size']} | {row['length']} | {label} | {sig} | {mean} | {sd} | {row['n']} | {row['backend']} | {row['micro_batch']} |")
    (OUT/"EXACT_PPL.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    manifest={"input_sha256":data["input_sha256"],"grouping":"full scientific configuration including effective sigma and source commit; endpoint separated",
              "successful_seed_records":len(details)//2,"hipe_fixed_sigma_groups":len(hipe)//2,
              "reported_endpoints":ENDPOINTS,"unrun_candidate_sigma_cells":sum(r["status"]=="not_run" for r in coverage),
              "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "csv_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob("*.csv")}}
    (OUT/"export_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest,indent=2))
    for row in hipe:
        if row["endpoint"]=="final":print(row["dataset"],row["model_size"],row["length"],row["sigma"],*[f"{row[k]:.6f}" for k in ["seed_6198","seed_1024","seed_7","mean_ppl","sample_std_ppl"]])


if __name__=="__main__":
    main()
