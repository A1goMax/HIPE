"""Organize completed joint sigma/layer and reference-XPos results without pooling protocols."""
from pathlib import Path
from collections import defaultdict, Counter
import csv, hashlib, json, math, statistics

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"outputs/table2_joint_20261104"
INPUTS=[
 ROOT/"artifacts/analysis/table2_20261103/raw_runs.json",
 ROOT/"artifacts/analysis/L2048_sigma_41695/completed/new_sigma_runs.json",
 ROOT/"artifacts/analysis/joint_sigma_layers_20261104/raw_runs.json"]
PRIOR=ROOT/"outputs/table2_sigma_final/workbook_data.json"
SEEDS=[6198,1024,7]
ENDPOINTS=["last_interval","final"]
CORE=["hipe/train.py","hipe/data.py","hipe/config.py","hipe/olmo/model.py","hipe/olmo/config.py"]
IGNORE={"seed","experiment_id","output_dir","source_notes"}
CONTROLS={"method","sigma","schedule","layer_sigmas","layer_threshold","learnable_sigma",
          "legacy_absolute_positions","attention_backend","micro_batch_size",
          "xpos_variant","xpos_scale_base"}
TARGETS={("wikitext103","60M",2048):[500,800],("c4","60M",512):[200,500],("c4","60M",2048):[25,50]}

def canonical(v):return json.dumps(v,sort_keys=True,separators=(",",":"))
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()[:16]
def config_identity(c):
    result={k:v for k,v in c.items() if k not in IGNORE}
    if c["method"]!="hipe":
        for k in ["sigma","schedule","layer_sigmas","layer_threshold","learnable_sigma"]:result.pop(k,None)
    if c["method"]!="xpos":
        for k in ["xpos_variant","xpos_scale_base"]:result.pop(k,None)
    return result

def evidence(run):
    m=run["run_manifest.json"]
    return {"data":m["data"],"tokenizer":m["tokenizer_file_sha256"],"vocabulary":m["vocabulary"],
            "runtime_packages":m["runtime"]["packages"],"gpu":m["runtime"]["gpu"]}

def metric(run,ep):
    key="regular_validation" if ep=="last_interval" else "final"
    row=max((v for v in run["metrics"] if v[key]),key=lambda v:v["step"])
    val=row["validation"];assert math.isclose(val["ppl"],math.exp(val["loss_sum"]/val["tokens"]),rel_tol=1e-12)
    return row

def layers(run):
    c=run["config"];m=run.get("run_manifest.json",{}).get("model_config",{})
    if c["method"]!="hipe":return 0
    assert c["schedule"]=="bipartite"
    n=m.get("n_layers",8);threshold=c["layer_threshold"]
    assert n==8 and threshold in [1,3,5]
    return n-threshold-1

def csv_file(name,rows):
    assert rows
    keys=list(rows[0])
    with (OUT/name).open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for r in rows:writer.writerow({k:canonical(r[k]) if isinstance(r[k],(dict,list)) else r[k] for k in keys})

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    records=[];campaigns={};raw_data=[]
    for p in INPUTS:
        content=json.loads(p.read_text(encoding="utf-8"));raw_data.append(content)
        records+=content["runs"]
        for c in content["campaigns"]:campaigns[str(c["job"])]=c
    assert len(records)==len({r["output_dir"] for r in records})==273
    new=raw_data[-1]["runs"]
    assert len(new)==81 and all("completion.json" in r and r["worker_status"]["returncode"]==0 for r in new)
    successful=[r for r in records if "completion.json" in r];assert len(successful)==255
    assert all(r["config"]["max_train_steps"] is None for r in records)
    compatibility=json.loads((ROOT/"experiments/validation/table2_sigma_layers_xpos_v1.json").read_text())
    assert compatibility["regression"]["cpu_logits_parameters_and_gradients_identical"]
    assert compatibility["regression"]["exact_tensor_matches"]==1008
    old_core={canonical({n:c["core_file_sha256"][n] for n in CORE}) for job,c in campaigns.items() if job!="41770"}
    assert len(old_core)==1
    new_core={n:campaigns["41770"]["core_file_sha256"][n] for n in CORE}
    assert new_core["hipe/data.py"]==json.loads(next(iter(old_core)))["hipe/data.py"]
    grouped=defaultdict(list);group_id={}
    for r in successful:
        c=r["config"];m=r["run_manifest.json"]
        assert c["max_tokens"]==100000000 and c["global_batch_size"]==64
        assert r["completion.json"]["step"]==c["max_tokens"]//(64*c["seq_len"])
        assert r["worker_status"]["returncode"]==0 and m["experiment"]==c
        core={n:campaigns[str(r["job"])]["core_file_sha256"][n] for n in CORE}
        ident={"config":config_identity(c),"evidence":evidence(r),"core_file_sha256":core}
        gid=digest(ident);grouped[gid].append(r);group_id[r["output_dir"]]=gid
        if str(r["job"])=="41770":
            assert m["source"]["commit"]==m["source"]["publication"]["commit"]=="5df7693e38ede95a9fa1747207b59fd2c701e75a"
            assert m["source"]["status"]==""
            assert m["source"]["code_file_sha256"]==new_core
            if c["method"]=="hipe":
                assert m["positional_layout"]["hipe_layer_count"]==layers(r)
                assert m["positional_layout"]["hipe_layer_indices_zero_based"]==list(range(8-layers(r),8))
                assert m["positional_layout"]["sigma_by_layer"]==[None]*(8-layers(r))+[c["sigma"]]*layers(r)
            else:
                assert c["method"]=="xpos" and not c["legacy_absolute_positions"]
                assert c["xpos_variant"]=="torchscale" and c["xpos_scale_base"]==512
    assert len(grouped)==85
    stats=[];lookup={};run_lookup={}
    for gid,runs in grouped.items():
        assert len(runs)==3 and {r["config"]["seed"] for r in runs}==set(SEEDS)
        by_seed={r["config"]["seed"]:r for r in runs};first=runs[0];c=first["config"]
        for ep in ENDPOINTS:
            ms=[metric(by_seed[s],ep) for s in SEEDS];vals=[v["validation"]["ppl"] for v in ms]
            endpoints={(v["step"],v["actual_prediction_tokens"],v["validation"]["tokens"]) for v in ms}
            assert len(endpoints)==1
            row={"group_id":gid,"dataset":c["dataset"],"model_size":c["model_size"],"length":c["seq_len"],
                 "method":c["method"],"method_label":{"hipe":"HIPE","rope":"RoPE","nope":"Legacy NoPE+APE","alibi":"ALiBi","xpos":"XPos"}[c["method"]],
                 "sigma":c["sigma"] if c["method"]=="hipe" else None,
                 "hipe_layers":layers(first),"threshold":c["layer_threshold"] if c["method"]=="hipe" else None,
                 "endpoint":ep,"seed_6198":vals[0],"seed_1024":vals[1],"seed_7":vals[2],
                 "mean_ppl":statistics.mean(vals),"sample_std_ppl":statistics.stdev(vals),"n":3,"status":"completed",
                 "is_new":all(str(r["job"])=="41770" for r in runs),"micro_batch":c["micro_batch_size"],
                 "global_batch":c["global_batch_size"],"backend":c["attention_backend"],
                 "step":ms[0]["step"],"prediction_tokens":ms[0]["actual_prediction_tokens"],
                 "validation_tokens":ms[0]["validation"]["tokens"],
                 "source_commits":",".join(sorted({r["run_manifest.json"]["source"]["commit"] for r in runs})),
                 "core_id":digest({n:campaigns[str(first["job"])]["core_file_sha256"][n] for n in CORE}),
                 "run_paths":[by_seed[s]["output_dir"] for s in SEEDS]}
            stats.append(row);lookup[(gid,ep)]=row
        run_lookup[gid]=first
    assert len(stats)==170 and sum(r["method"]=="hipe" for r in stats)==82
    def find(ds,size,length,method,ep):
        return [r for r in stats if (r["dataset"],r["model_size"],r["length"],r["method"],r["endpoint"])==(ds,size,length,method,ep)]
    def matched(a,b):
        ra,rb=run_lookup[a["group_id"]],run_lookup[b["group_id"]]
        ca={k:v for k,v in config_identity(ra["config"]).items() if k not in CONTROLS}
        cb={k:v for k,v in config_identity(rb["config"]).items() if k not in CONTROLS}
        assert ca==cb and evidence(ra)==evidence(rb),(a["group_id"],b["group_id"])
        assert (a["step"],a["prediction_tokens"],a["validation_tokens"])==(b["step"],b["prediction_tokens"],b["validation_tokens"])
    # Source groups remain distinct. Cross-version contrasts use the archived parity audit;
    # no group mean combines seeds from different versions.
    comparisons=[]
    for g in stats:
        if g["method"]!="hipe":continue
        ropes=find(g["dataset"],g["model_size"],g["length"],"rope",g["endpoint"]);assert len(ropes)==1
        rope=ropes[0];matched(g,rope)
        deltas=[g["seed_"+str(s)]-rope["seed_"+str(s)] for s in SEEDS]
        comparisons.append({**g,"candidate_group_id":g["group_id"],"rope_group_id":rope["group_id"],
                            "rope_mean":rope["mean_ppl"],"rope_sd":rope["sample_std_ppl"],
                            "rope_seed_6198":rope["seed_6198"],"rope_seed_1024":rope["seed_1024"],"rope_seed_7":rope["seed_7"],
                            "delta_ppl":g["mean_ppl"]-rope["mean_ppl"],
                            "gain_vs_rope_percent":100*(1-g["mean_ppl"]/rope["mean_ppl"]),
                            "seed_deltas":deltas,"seed_wins":sum(v<0 for v in deltas),
                            "mean_win":g["mean_ppl"]<rope["mean_ppl"],
                            "origin":"new" if g["is_new"] else "reused",
                            "micro_batch_note":"HIPE and RoPE differ" if g["micro_batch"]!=rope["micro_batch"] else "matched"})
    target_rows=[g for g in comparisons if (g["dataset"],g["model_size"],g["length"]) in TARGETS
                 and g["sigma"] in TARGETS[(g["dataset"],g["model_size"],g["length"])] and g["hipe_layers"] in [2,4,6]]
    assert len(target_rows)==36 and sum(g["is_new"] for g in target_rows)==30
    prior=json.loads(PRIOR.read_text(encoding="utf-8"));best=[]
    for ep in ENDPOINTS:
        for ds in ["wikitext103","c4"]:
            for size in ["20M","60M"]:
                for length in [512,1024,2048]:
                    candidates=find(ds,size,length,"hipe",ep)
                    chosen=min(candidates,key=lambda g:(g["mean_ppl"],g["sigma"],g["hipe_layers"]))
                    previous=next(r for r in prior["best_rows"] if (r["dataset"],r["model_size"],r["length"],r["endpoint"])==(ds,size,length,ep))
                    row={"dataset":ds,"model_size":size,"length":length,"endpoint":ep,
                         "selected_sigma":chosen["sigma"],"hipe_layers":chosen["hipe_layers"],
                         "candidate_count":len(candidates),"n":3,"selected_group_id":chosen["group_id"],
                         "candidate_group_ids":[r["group_id"] for r in sorted(candidates,key=lambda g:(g["sigma"],g["hipe_layers"]))],
                         "hipe_mean":chosen["mean_ppl"],"hipe_sd":chosen["sample_std_ppl"],
                         "previous_sigma":previous["selected_sigma"],"previous_hipe_layers":4,
                         "previous_hipe_mean":previous["hipe_mean"],"previous_hipe_sd":previous["hipe_sd"],
                         "previous_gain_vs_rope_percent":previous["gain_vs_rope_percent"],
                         "selected_is_new":chosen["is_new"],"baseline_group_ids":{}}
                    methods=[("hipe",chosen["mean_ppl"])]
                    for method in ["rope","nope","alibi","xpos"]:
                        groups=find(ds,size,length,method,ep)
                        if groups:
                            assert len(groups)==1
                            g=groups[0];matched(chosen,g)
                            row[method+"_mean"]=g["mean_ppl"];row[method+"_sd"]=g["sample_std_ppl"]
                            row[method+"_status"]="completed";row["baseline_group_ids"][method]=g["group_id"]
                            methods.append((method,g["mean_ppl"]))
                        else:
                            failed=[r for r in records if "completion.json" not in r and
                                    (r["config"]["dataset"],r["config"]["model_size"],r["config"]["seq_len"],r["config"]["method"])==(ds,size,length,method)]
                            assert method=="alibi" and failed
                            row[method+"_mean"]=row[method+"_sd"]=None
                            row[method+"_status"]="OOM" if all("OutOfMemoryError" in (r.get("failure_tail") or "") for r in failed) else "unavailable"
                    row["best_available_method"]=min(methods,key=lambda p:p[1])[0]
                    row["gain_vs_rope_percent"]=100*(1-row["hipe_mean"]/row["rope_mean"])
                    row["mean_win"]=row["hipe_mean"]<row["rope_mean"]
                    row["converted_to_win"]=row["mean_win"] and previous["gain_vs_rope_percent"]<0
                    row["seed_wins_vs_rope"]=next(g["seed_wins"] for g in comparisons if g["group_id"]==chosen["group_id"] and g["endpoint"]==ep)
                    best.append(row)
    details=[];attempts=[]
    for run in records:
        c=run["config"];m=run.get("run_manifest.json",{})
        complete="completion.json" in run
        entry={"job":str(run["job"]),"task_index":run["task_index"],"dataset":c["dataset"],"model_size":c["model_size"],
               "length":c["seq_len"],"method":c["method"],"sigma":c["sigma"] if c["method"]=="hipe" else None,
               "hipe_layers":layers(run) if c["method"]=="hipe" else 0,
               "threshold":c.get("layer_threshold") if c["method"]=="hipe" else None,
               "seed":c["seed"],"status":"completed" if complete else ("OOM" if "OutOfMemoryError" in (run.get("failure_tail") or "") else "failed"),
               "micro_batch":c["micro_batch_size"],"global_batch":c["global_batch_size"],"backend":c["attention_backend"],
               "source_commit":m.get("source",{}).get("commit"),"source_snapshot_sha256":m.get("source",{}).get("snapshot_sha256"),
               "output_dir":run["output_dir"],"exit_code":(run.get("worker_status") or {}).get("returncode")}
        attempts.append(entry)
        if not complete:continue
        for ep in ENDPOINTS:
            v=metric(run,ep)
            details.append({**entry,"group_id":group_id[run["output_dir"]],"endpoint":ep,"step":v["step"],
                            "prediction_tokens":v["actual_prediction_tokens"],"validation_tokens":v["validation"]["tokens"],
                            "ppl":v["validation"]["ppl"],"ppl_exact_text":repr(v["validation"]["ppl"])})
    assert len(details)==510
    summary={"new_runs":81,"new_successful":81,"new_hipe_runs":45,"new_xpos_runs":36,"reused_grid_seed_runs":9,
             "execution_attempts":273,"completed_unique_seed_runs":255,"failed_attempts":18,
             "fixed_configurations":85,"hipe_configurations":41,"primary_endpoint":"last_interval",
             "seeds":SEEDS,"max_tokens":100000000,"global_batch":64,
             "previous_primary_wins_vs_rope":prior["summary"]["last_interval"]["wins_vs_rope"]}
    for ep in ENDPOINTS:
        rows=[r for r in best if r["endpoint"]==ep];cs=[r for r in comparisons if r["endpoint"]==ep and r["is_new"]]
        xgroups=[r for r in stats if r["method"]=="xpos" and r["endpoint"]==ep]
        xwins=sum(g["mean_ppl"]<find(g["dataset"],g["model_size"],g["length"],"rope",ep)[0]["mean_ppl"] for g in xgroups)
        summary[ep]={"wins_vs_rope":sum(r["mean_win"] for r in rows),"losses_vs_rope":sum(not r["mean_win"] for r in rows),
                     "new_winning_configurations":sum(g["mean_win"] for g in cs),
                     "converted_settings":sum(r["converted_to_win"] for r in rows),
                     "paired_seed_wins_selected":sum(r["seed_wins_vs_rope"] for r in rows),
                     "best_methods":dict(Counter(r["best_available_method"] for r in rows)),
                     "xpos_wins_vs_rope":xwins,
                     "hipe_wins_vs_xpos":sum(r["hipe_mean"]<r["xpos_mean"] for r in rows)}
    assert summary["last_interval"]["wins_vs_rope"]==11
    notes=[
      "Mean and sample SD use three seeds at one fixed sigma and HIPE-layer count; sigma/layer configurations are never pooled.",
      "Primary endpoint is last regular validation. Final-checkpoint results are a separate diagnostic view. The selected configurations agree across both views.",
      "Best observed HIPE is selected using these validation results. It is an exploratory tuning outcome, not an independent held-out estimate or a post-selection significance claim.",
      "Eight total model layers are retained. Two/four/six final layers use HIPE; preceding layers use unscaled RoPE.",
      "C4 HIPE and RoPE use different original micro-batches, with global batch 64 unchanged. Reported C4 contrasts retain this qualification.",
      "Old and new source cohorts are kept as distinct fixed-configuration groups. Cross-version contrasts use exact data/runtime matches and the archived unchanged-method numerical audit, not a claim of identical source-file hashes.",
      "XPos is the reference TorchScale variant with scale base 512 and no learned absolute positions. NoPE is Legacy NoPE+APE.",
      "Four ALiBi settings remain unavailable due to historical OOM. Failures are unavailable PPL, never zero.",
      "All 81 full runs completed. Twelve two-step GPU checks completed separately and are excluded from all research statistics.",
    ]
    payload={"summary":summary,"notes":notes,"source_files":[str(p) for p in INPUTS],
             "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUTS},
             "compatibility_evidence":compatibility,"best_rows":best,"all_rows":stats,
             "hipe_rows":[g for g in stats if g["method"]=="hipe"],
             "comparisons":comparisons,"target_rows":target_rows,
             "new_winning_rows":[g for g in comparisons if g["is_new"] and g["mean_win"]],
             "details":details,"attempts":attempts}
    (OUT/"workbook_data.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    csv_file("Best_vs_Baselines.csv",best);csv_file("HIPE_vs_RoPE_All_Configs.csv",comparisons)
    csv_file("Target_Grid.csv",target_rows);csv_file("All_Statistics.csv",stats)
    csv_file("Exact_Seed_PPL.csv",details);csv_file("Execution_Attempts.csv",attempts)
    def ms(mean,sd):return f"{mean:.6f} +/- {sd:.6f}"
    lines=["# Completed HIPE sigma/layer search and XPos comparison","",
      "All 81 new full runs completed (45 HIPE and 36 XPos). Nine previous HIPE seed-runs complete the targeted 18-configuration grid. The 12 two-step GPU checks are excluded.","",
      "## Main findings","",
      "Best observed HIPE has lower three-seed mean PPL than RoPE in 11/12 settings, up from 9/12. Two previous mean losses become wins. Three new fixed configurations have lower mean PPL than RoPE; two belong to the same C4 L512 setting.","",
      "## New HIPE configurations with lower mean PPL than RoPE","",
      "Primary endpoint: last regular validation. Sample SD is across seeds 6198, 1024 and 7 at exactly the stated sigma and layer count.","",
      "| Corpus | Model | Length | Sigma | HIPE layers | HIPE PPL mean +/- SD | RoPE PPL mean +/- SD | Reduction | Paired seeds favor HIPE |",
      "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for g in comparisons:
        if g["endpoint"]=="last_interval" and g["is_new"] and g["mean_win"]:
            lines.append(f"| {g['dataset']} | {g['model_size']} | {g['length']} | {g['sigma']:g} | {g['hipe_layers']} | {ms(g['mean_ppl'],g['sample_std_ppl'])} | {ms(g['rope_mean'],g['rope_sd'])} | {g['gain_vs_rope_percent']:.6f}% | {g['seed_wins']}/3 |")
    lines+=["","## Previously losing settings: selected results","",
      "| Corpus | Model | Length | Selected sigma | HIPE layers | Previous HIPE PPL | New HIPE PPL | RoPE PPL | Reduction |",
      "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in best:
        if r["endpoint"]=="last_interval" and (r["dataset"],r["model_size"],r["length"]) in TARGETS:
            lines.append(f"| {r['dataset']} | {r['model_size']} | {r['length']} | {r['selected_sigma']:g} | {r['hipe_layers']} | {ms(r['previous_hipe_mean'],r['previous_hipe_sd'])} | {ms(r['hipe_mean'],r['hipe_sd'])} | {ms(r['rope_mean'],r['rope_sd'])} | {r['gain_vs_rope_percent']:.6f}% |")
    lines+=["","C4 60M/L2048 is the remaining mean loss. Sigma 50 with two HIPE layers improves substantially over four HIPE layers, but does not surpass RoPE. Six-HIPE-layer candidates lose to RoPE in every paired seed in all three targeted settings.","",
      "The newly positive mean differences are approximately 0.10-0.25%. WikiText-103 60M/L2048 improves in two of three paired seeds. Each new C4 60M/L512 mean win improves in only one of three paired seeds. A lower selected mean therefore should not be described as consistent seed-wise superiority.","",
      "## Best observed HIPE against all baselines","",
      "| Corpus | Model | L | Sigma | HIPE layers | Legacy NoPE+APE | ALiBi | RoPE | XPos | HIPE |",
      "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in best:
        if r["endpoint"]!="last_interval":continue
        fmt=lambda m:ms(r[m+"_mean"],r[m+"_sd"]) if r[m+"_status"]=="completed" else r[m+"_status"]
        lines.append(f"| {r['dataset']} | {r['model_size']} | {r['length']} | {r['selected_sigma']:g} | {r['hipe_layers']} | {fmt('nope')} | {fmt('alibi')} | {fmt('rope')} | {fmt('xpos')} | {ms(r['hipe_mean'],r['hipe_sd'])} |")
    lines+=["","XPos has lower mean PPL than RoPE in 11/12 settings. With XPos included, the lowest available primary means belong to ALiBi in six settings, HIPE in three and XPos in three. HIPE-versus-RoPE wins do not imply wins over every baseline.","",
      "## Protocol and uncertainty","",*["- "+n for n in notes],"",
      "The same selected sigma/layer configurations and win/loss directions hold at the final checkpoint. The workbook retains both endpoint views and all fixed-configuration results, including losing candidates.","",
      "No additional training was launched during this analysis.",""]
    report="\n".join(lines)
    (OUT/"REPORT.md").write_text(report,encoding="utf-8")
    (ROOT/"analysis/table2/JOINT_SEARCH_REPORT.md").write_text(report,encoding="utf-8")
    print(json.dumps(summary,indent=2))
    print("WORKBOOK_INPUT",OUT/"workbook_data.json")

if __name__=="__main__":main()
