"""Prepare the completed sigma-sweep analysis and formal workbook inputs."""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import json
import math
import statistics

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"outputs/table2_sigma_final"
OLD=ROOT/"artifacts/analysis/table2_20261103/raw_runs.json"
NEW=ROOT/"artifacts/analysis/L2048_sigma_41695/completed/new_sigma_runs.json"
BEST=ROOT/"artifacts/analysis/L2048_sigma_41695/completed/best_sigma_comparison.json"
SEEDS=[6198,1024,7]
ENDPOINTS=["last_interval","final"]
HIGH={700,800,900,1000}


def write_csv(name,rows):
    with (OUT/name).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def endpoint(run,name):
    options=[r for r in run["metrics"] if r["regular_validation"]] if name=="last_interval" else [r for r in run["metrics"] if r["final"]]
    r=max(options,key=lambda r:r["step"]);v=r["validation"]
    assert math.isclose(v["ppl"],math.exp(v["loss_sum"]/v["tokens"]),rel_tol=1e-12)
    return r


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    old=json.loads(OLD.read_text());new=json.loads(NEW.read_text());comparison=json.loads(BEST.read_text())
    combined=old["runs"]+new["runs"];assert len({r["output_dir"] for r in combined})==len(combined)==192
    assert len(new["runs"])==42 and all("completion.json" in r and r["worker_status"]["returncode"]==0 for r in new["runs"])
    all_stats=comparison["all_fixed_sigma_summaries"]
    group_lookup={}
    stats=[]
    for group in all_stats:
        c=group["identity"]["config"];values=[group["seed_values"][str(s)] for s in SEEDS]
        assert group["n"]==3 and math.isclose(statistics.mean(values),group["mean_ppl"],abs_tol=1e-10)
        assert math.isclose(statistics.stdev(values),group["sample_std_ppl"],abs_tol=1e-10)
        row={"group_id":group["group_id"],"dataset":c["dataset"],"model_size":c["model_size"],"length":c["seq_len"],
             "method":c["method"],"method_label":"Legacy NoPE+APE" if c["method"]=="nope" else c["method"].upper(),
             "sigma":c["sigma"] if c["method"]=="hipe" else None,"endpoint":group["endpoint"],
             "seed_6198":values[0],"seed_1024":values[1],"seed_7":values[2],"mean_ppl":group["mean_ppl"],
             "sample_std_ppl":group["sample_std_ppl"],"n":3,"micro_batch":c["micro_batch_size"],
             "global_batch":c["global_batch_size"],"backend":c["attention_backend"],
             "threshold":c["layer_threshold"],"step":group["step"],"prediction_tokens":group["training_tokens"],
             "source_commits":",".join(group["source_commits"])}
        stats.append(row)
        for path in group["run_paths"]:group_lookup[(path,group["endpoint"])]=group["group_id"]
    assert len(stats)==116
    hi=[r for r in stats if r["method"]=="hipe"];assert len(hi)==52
    details=[];attempts=[]
    for run in combined:
        c=run["config"];completed="completion.json" in run
        row={"job":run["job"],"task_index":run["task_index"],"dataset":c["dataset"],"model_size":c["model_size"],
             "length":c["seq_len"],"method":c["method"],"method_label":"Legacy NoPE+APE" if c["method"]=="nope" else c["method"].upper(),
             "sigma":c["sigma"] if c["method"]=="hipe" else None,"seed":c["seed"],
             "status":"completed" if completed else ("OOM" if "OutOfMemoryError" in (run.get("failure_tail") or "") else "failed"),
             "micro_batch":c["micro_batch_size"],"global_batch":c["global_batch_size"],"backend":c["attention_backend"],
             "threshold":c["layer_threshold"],"schedule":c["schedule"],
             "source_commit":run.get("run_manifest.json",{}).get("source",{}).get("commit","unavailable"),
             "output_dir":run["output_dir"],"exit_code":(run.get("worker_status") or {}).get("returncode")}
        attempts.append(row)
        if not completed:continue
        expected=c["max_tokens"]//(c["global_batch_size"]*c["seq_len"])
        assert run["completion.json"]["step"]==expected and c["max_tokens"]==100000000
        for ep in ENDPOINTS:
            m=endpoint(run,ep);value=m["validation"]
            details.append({**row,"group_id":group_lookup[(run["output_dir"],ep)],"endpoint":ep,
                            "step":m["step"],"prediction_tokens":m["actual_prediction_tokens"],
                            "validation_tokens":value["tokens"],"ppl":value["ppl"],"ppl_exact_text":repr(value["ppl"])})
    assert len(details)==348 and sum(r["status"]=="completed" for r in attempts)==174
    best=[]
    for r in comparison["results"]:
        candidates=r["candidates"];selected=min(candidates,key=lambda x:(x["mean_ppl"],x["sigma"]))
        assert selected["sigma"]==r["selected_sigma"] and selected["mean_ppl"]==r["hipe_mean_ppl"]
        best.append(r)
    coverage=[]
    for ds in ["wikitext103","c4"]:
        for model in ["20M","60M"]:
            for sigma in sorted(HIGH):
                group=next(r for r in hi if (r["dataset"],r["model_size"],r["length"],r["sigma"],r["endpoint"])==(ds,model,2048,sigma,"last_interval"))
                paths=[r["output_dir"] for r in details if r["group_id"]==group["group_id"] and r["endpoint"]=="last_interval"]
                assert len(paths)==len(set(paths))==3
                coverage.append({"dataset":ds,"model_size":model,"length":2048,"sigma":sigma,"n":3,
                                 "status":"completed","origin":"reused" if (ds,model,sigma) in {("wikitext103","60M",700),("c4","20M",700)} else "new",
                                 "mean_ppl":group["mean_ppl"],"sample_std_ppl":group["sample_std_ppl"],"group_id":group["group_id"],
                                 "run_paths":";".join(paths)})
    assert len(coverage)==16
    best_rows=[];paired=[]
    for r in best:
        b=r["baselines"];high=[c for c in r["candidates"] if c["sigma"] in HIGH] if r["length"]==2048 else []
        high_best=min(high,key=lambda c:(c["mean_ppl"],c["sigma"])) if high else None
        row={"dataset":r["dataset"],"model_size":r["model_size"],"length":r["length"],"endpoint":r["endpoint"],
             "selected_sigma":r["selected_sigma"],"candidate_count":r["complete_sigma_candidates"],"n":r["hipe_n"],
             "hipe_mean":r["hipe_mean_ppl"],"hipe_sd":r["hipe_sample_std_ppl"],
             "gain_vs_rope_percent":r["relative_gain_vs_rope_percent"],
             "best_high_range_sigma":high_best["sigma"] if high_best else None,
             "best_high_range_mean":high_best["mean_ppl"] if high_best else None,
             "best_high_range_sd":high_best["sample_std_ppl"] if high_best else None}
        for method in ["nope","alibi","rope"]:
            v=b[method]
            row[method+"_mean"]=v["mean_ppl"];row[method+"_sd"]=v["sample_std_ppl"];row[method+"_status"]=v["status"]
        winner=min([("hipe",r["hipe_mean_ppl"])]+[(m,b[m]["mean_ppl"]) for m in ["nope","alibi","rope"] if b[m]["status"]=="completed"],key=lambda x:x[1])[0]
        row["best_available_method"]=winner
        group=next(g for g in hi if g["dataset"]==r["dataset"] and g["model_size"]==r["model_size"] and g["length"]==r["length"] and g["endpoint"]==r["endpoint"] and g["sigma"]==r["selected_sigma"])
        row["selected_group_id"]=group["group_id"]
        row["baseline_group_ids"]={}
        for method in ["nope","alibi","rope"]:
            if b[method]["status"]=="completed":
                match=[g for g in stats if g["method"]==method and g["dataset"]==r["dataset"] and g["model_size"]==r["model_size"] and g["length"]==r["length"] and g["endpoint"]==r["endpoint"]]
                assert len(match)==1;row["baseline_group_ids"][method]=match[0]["group_id"]
        best_rows.append(row)
        delta=[r["hipe_seed_values"][str(s)]-b["rope"]["seed_values"][str(s)] for s in SEEDS]
        paired.append({"dataset":r["dataset"],"model_size":r["model_size"],"length":r["length"],"endpoint":r["endpoint"],
                       "sigma":r["selected_sigma"],"mean_delta":statistics.mean(delta),"seeds_favor_hipe":sum(x<0 for x in delta),
                       "seed_deltas":delta})
    summary={"new_runs":42,"new_successful":42,"reused_requested_runs":6,"requested_grid_seed_runs":48,
             "execution_attempts":192,"completed_unique_seed_runs":174,"failed_attempts":18,
             "completed_HIPE_seed_runs":78,"completed_HIPE_configurations":26,"requested_sigma_grid_complete":True,
             "selection":"minimum complete three-seed mean validation PPL at each dataset/model/length; never min seed or pooled sigma"}
    for ep in ENDPOINTS:
        rows=[r for r in best_rows if r["endpoint"]==ep];pairs=[r for r in paired if r["endpoint"]==ep]
        summary[ep]={"wins_vs_rope":sum(r["gain_vs_rope_percent"]>0 for r in rows),"losses_vs_rope":sum(r["gain_vs_rope_percent"]<0 for r in rows),
                     "seedwise_wins":sum(r["seeds_favor_hipe"] for r in pairs),
                     "wins_by_length":{str(n):sum(r["gain_vs_rope_percent"]>0 for r in rows if r["length"]==n) for n in [512,1024,2048]},
                     "best_methods":dict(collections.Counter(r["best_available_method"] for r in rows))}
    notes=[
      "Best observed HIPE is selected by the lowest complete three-seed mean at each fixed dataset/model/length/protocol. Sample SD is computed within that selected sigma; sigmas are never pooled.",
      "Primary endpoint: last regular validation, matching the earlier analysis. Final-checkpoint results are a separate view. Selected sigmas agree across the two views in this campaign.",
      "Selection uses validation results from these seeds, so best-observed values are tuning outcomes rather than an independent held-out test estimate.",
      "L2048 includes all requested sigma 700/800/900/1000 values. WikiText 20M also includes sigma 500; C4 60M also includes sigma 50. L512/L1024 each have only one completed sigma.",
      "Baseline NoPE is explicitly Legacy NoPE+APE. OOM is unavailable PPL, never zero.",
      "C4 HIPE and RoPE retain different micro-batches with global batch 64 unchanged. Five historical C4 paper-log matches used a 5/3 layer split; this campaign uses the current paper 4/4 split.",
      "Last regular validation: steps 3000/1500/700 for L512/1024/2048; final checkpoints 3051/1525/762. Complete-batch token rounding stays below the 100M cap.",
    ]
    payload={"summary":summary,"notes":notes,"source_files":[str(OLD),str(NEW),str(BEST)],
             "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OLD,NEW,BEST]},
             "best_rows":best_rows,"all_rows":stats,"hipe_rows":hi,"details":details,"attempts":attempts,"coverage":coverage,
             "paired":paired}
    (OUT/"workbook_data.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    write_csv("Best_vs_Baselines.csv",[{k:v for k,v in r.items() if k!="baseline_group_ids"} for r in best_rows])
    write_csv("HIPE_All_Sigmas.csv",hi);write_csv("Exact_Seed_PPL.csv",details);write_csv("Execution_Attempts.csv",attempts);write_csv("L2048_Coverage.csv",coverage)
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    lines=["# Completed Table 2 sigma-sweep analysis","",
           "All 42 new runs completed. Together with 6 reused runs, the requested L2048 grid contains 48 unique seed-runs: 4 dataset/model settings x 4 sigmas x 3 seeds.","",
           "## Main result","",
           "The best observed L2048 sigma is 800 for WikiText 20M, WikiText 60M and C4 20M. For C4 60M, the previously tested sigma 50 remains best among all tested values. Within the requested 700-1000 range, sigma 1000 is best for C4 60M.","",
           "Best observed HIPE now has lower mean PPL than RoPE in 9/12 settings (previously 8/12). The only direction change is C4 20M/L2048, which becomes a small mean improvement. L2048 HIPE improves in 2/4 settings. WikiText 60M and C4 60M still trail RoPE. Final-checkpoint results preserve these directions.","",
           "## Best observed HIPE against baselines","",
           "The primary view is last regular validation. Values are mean +/- sample SD over seeds 6198, 1024 and 7, at one fixed sigma. Selecting the best individual seed or averaging different sigmas is prohibited.","",
           "| Corpus | Model | Length | Best sigma | Candidates | Legacy NoPE+APE | ALiBi | RoPE | Best observed HIPE |",
           "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in best_rows:
        if r["endpoint"]!="last_interval":continue
        fmt=lambda m: f"{r[m+'_mean']:.4f} +/- {r[m+'_sd']:.4f}" if r[m+"_status"]=="completed" else "OOM"
        lines.append(f"| {r['dataset']} | {r['model_size']} | {r['length']} | {r['selected_sigma']:g} | {r['candidate_count']} | {fmt('nope')} | {fmt('alibi')} | {fmt('rope')} | {r['hipe_mean']:.4f} +/- {r['hipe_sd']:.4f} |")
    lines += ["","## L2048 per-sigma results","",
              "| Corpus | Model | Sigma | Mean PPL | Sample SD | n | Selected overall |",
              "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for g in sorted((g for g in hi if g["length"]==2048 and g["endpoint"]=="last_interval"),key=lambda g:(g["dataset"],g["model_size"],g["sigma"])):
        chosen=next(r for r in best_rows if r["endpoint"]=="last_interval" and r["dataset"]==g["dataset"] and r["model_size"]==g["model_size"] and r["length"]==2048)
        lines.append(f"| {g['dataset']} | {g['model_size']} | {g['sigma']:g} | {g['mean_ppl']:.6f} | {g['sample_std_ppl']:.6f} | 3 | {'yes' if g['sigma']==chosen['selected_sigma'] else ''} |")
    lines += ["","## Interpretation","",
              "1. Sigma 800 has the lowest mean among tested high-range values in three settings, but the differences are small relative to seed variation. Larger sigma is not monotonically better.",
              "2. C4 60M is an exception: all newly tested high sigmas underperform the old sigma 50 mean, and their across-seed SD is larger. Values between 50 and 700 were not explored here. Sigma 50 is best observed among tested values.",
              "3. Compared with the old fixed choice, WikiText 20M improves from 114.5185 to 114.0708, WikiText 60M from 97.2231 to 96.8108, and C4 20M from 256.6892 to 254.4656. C4 60M's selected result is unchanged.",
              "4. The best mean is calculated over all three seeds. A low single-seed value does not make its sigma the winner.",
              "5. Among available methods, HIPE has the lowest mean in 5 settings, ALiBi in 6 and RoPE in 1. Four ALiBi settings remain unavailable due to OOM. The legacy NoPE+APE baseline retains learned absolute position embeddings.","",
              "## Scope and uncertainty","",*["- "+note for note in notes],"",
              "These are validation-selected results. The three-seed SD describes variation conditional on a chosen sigma and does not account for hyperparameter-selection uncertainty. No post-selection significance claim is made.",
              "",
              "C4 comparisons versus RoPE retain a micro-batch difference. The within-HIPE sigma comparison holds this constant. Absolute reproduction of paper values remains qualified by historical validation/population and layer-split differences.",
              "",
              "The formal workbook contains both endpoint views, every sigma separately, exact per-seed PPLs, baseline comparisons, failure records and coverage. No additional training was launched for this analysis.",""]
    report="\n".join(lines)
    import re
    report=report.replace("WikiText20M","WikiText 20M").replace("WikiText60M","WikiText 60M").replace("C420M","C4 20M").replace("C460M","C4 60M")
    report=re.sub(r"\b(All|all|with|of|contains|in|from|than|thanRoPE|sigma|seeds|steps|checkpoints|previously)(?=\d)",r"\1 ",report)
    report=report.replace("thanRoPE","than RoPE").replace("forC4","for C4").replace("andC4","and C4").replace("sigma-sweep","sigma-sweep")
    (OUT/"ANALYSIS_REPORT.md").write_text(report,encoding="utf-8")
    (Path(__file__).parent/"COMPLETED_SIGMA_REPORT.md").write_text(report,encoding="utf-8")
    print(json.dumps(summary,indent=2))
    print("WORKBOOK_INPUT",OUT/"workbook_data.json")


if __name__=="__main__":
    main()
