"""Analyze the completed C4 extension and update paper-scope Table 2 outputs."""
from pathlib import Path
from collections import defaultdict,Counter
import copy,csv,hashlib,json,math,statistics
from analyze_joint_search import canonical,digest,config_identity,evidence,metric,CONTROLS

ROOT=Path(__file__).resolve().parents[2]
PRIOR=ROOT/"outputs/table2_public_20261104/workbook_data.json"
NEW=ROOT/"artifacts/analysis/c4_extension_20261105/raw_runs.json"
PROTOCOL=ROOT/"experiments/protocols/table2_c4_extension_v2.json"
OUT=ROOT/"outputs/table2_c4_20261105"
SEEDS=[6198,1024,7];EPS=["last_interval","final"]
METHODS={"hipe","rope","alibi","nope"}
CORE=["hipe/train.py","hipe/data.py","hipe/config.py","hipe/olmo/model.py","hipe/olmo/config.py"]

def layer_count(run):
    c=run["config"];model=run["run_manifest.json"]["model_config"];n=model["n_layers"]
    assert n==8
    if c["method"]!="hipe":return 0
    if c["schedule"]=="uniform":
        assert model["rope_scaling_threshold"]==-1
        vec=model["scaled_rope_sigmas"];assert vec==[c["sigma"]]*n
    else:
        assert c["schedule"]=="bipartite" and 0<=c["layer_threshold"]<7
        vec=[None]*(c["layer_threshold"]+1)+[c["sigma"]]*(n-c["layer_threshold"]-1)
    recorded=run["run_manifest.json"]["positional_layout"]
    assert recorded["sigma_by_layer"]==vec
    assert recorded["hipe_layer_count"]==sum(v is not None for v in vec)
    return sum(v is not None for v in vec)

def source_path(recorded):
    # Source registers may have been written on Windows and replayed on labserver2.
    normalized=str(recorded).replace(chr(92),"/")
    if "/artifacts/" in normalized:
        candidate=ROOT/"artifacts"/normalized.split("/artifacts/",1)[1]
    else:
        candidate=Path(recorded)
    if not candidate.is_file():
        raise FileNotFoundError(f"Missing frozen input: {candidate}")
    return candidate

def write_csv(name,rows):
    assert rows
    with (OUT/name).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        for r in rows:w.writerow({k:canonical(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    previous=json.loads(PRIOR.read_text(encoding="utf-8"))
    new=json.loads(NEW.read_text(encoding="utf-8"))
    plan=json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert len(new["runs"])==60 and all(r["worker_status"]["returncode"]==0 and "completion.json" in r for r in new["runs"])
    assert all(r["method"] in METHODS for r in previous["all_rows"]+previous["attempts"])
    oldpaths={r["output_dir"] for r in previous["attempts"]}
    assert not any(r["output_dir"] in oldpaths for r in new["runs"])
    # Obtain exact archived RoPE control manifests for cross-run protocol checks.
    raw_old={}
    old_campaigns={}
    source_paths=[source_path(v) for v in previous["source_files"]]
    for recorded,p in zip(previous["source_files"],source_paths):
        assert hashlib.sha256(p.read_bytes()).hexdigest()==previous["source_sha256"][recorded]
        d=json.loads(p.read_text(encoding="utf-8"))
        raw_old.update({r["output_dir"]:r for r in d["runs"]})
        old_campaigns.update({str(c["job"]):c for c in d["campaigns"]})
    reference_stat=next(r for r in previous["all_rows"] if (r["dataset"],r["model_size"],r["length"],r["method"],r["endpoint"])==("c4","60M",2048,"rope","last_interval"))
    reference=raw_old[reference_stat["run_paths"][0]]
    common_reference={k:v for k,v in config_identity(reference["config"]).items() if k not in CONTROLS}
    core=new["campaigns"][0]["core_file_sha256"]
    assert core==old_campaigns["41770"]["core_file_sha256"]
    groups=defaultdict(list);by_path={}
    for run in new["runs"]:
        c=run["config"];m=run["run_manifest.json"];h=layer_count(run)
        assert c["method"]=="hipe" and c["dataset"]=="c4" and c["model_size"]=="60M" and c["seq_len"]==2048
        assert c["max_train_steps"] is None and c["micro_batch_size"]==4 and c["global_batch_size"]==64
        assert c["max_tokens"]==100000000 and run["completion.json"]["step"]==762
        assert run["completion.json"]["actual_prediction_tokens"]==99876864
        assert m["source"]["commit"]==m["source"]["publication"]["commit"]=="c44c207b83bf4140ec462bd727b1c5e6fc946613"
        assert m["source"]["status"]=="" and m["source"]["code_file_sha256"]==core and m["experiment"]==c
        common={k:v for k,v in config_identity(c).items() if k not in CONTROLS}
        assert common==common_reference and evidence(run)==evidence(reference)
        ident={"config":config_identity(c),"evidence":evidence(run),"core_file_sha256":core}
        gid=digest(ident);groups[gid].append(run);by_path[run["output_dir"]]=gid
    assert len(groups)==20
    stats=copy.deepcopy(previous["all_rows"]);details=copy.deepcopy(previous["details"]);attempts=copy.deepcopy(previous["attempts"])
    for r in stats:r["is_new"]=False
    for r in stats:
        vals=[r["seed_"+str(s)] for s in SEEDS]
        assert math.isclose(statistics.mean(vals),r["mean_ppl"],abs_tol=1e-10)
        assert math.isclose(statistics.stdev(vals),r["sample_std_ppl"],abs_tol=1e-10)
    oldids={r["group_id"] for r in stats}
    assert not oldids.intersection(groups)
    for gid,runs in groups.items():
        assert len(runs)==3 and {r["config"]["seed"] for r in runs}==set(SEEDS)
        byseed={r["config"]["seed"]:r for r in runs};first=runs[0];c=first["config"]
        for ep in EPS:
            ms=[metric(byseed[s],ep) for s in SEEDS];vals=[v["validation"]["ppl"] for v in ms]
            assert len({(v["step"],v["actual_prediction_tokens"],v["validation"]["tokens"]) for v in ms})==1
            rope=next(r for r in previous["all_rows"] if (r["dataset"],r["model_size"],r["length"],r["method"],r["endpoint"])==("c4","60M",2048,"rope",ep))
            assert (ms[0]["step"],ms[0]["actual_prediction_tokens"],ms[0]["validation"]["tokens"])==(rope["step"],rope["prediction_tokens"],rope["validation_tokens"])
            stats.append({"group_id":gid,"dataset":"c4","model_size":"60M","length":2048,"method":"hipe","method_label":"HIPE",
                          "sigma":c["sigma"],"hipe_layers":layer_count(first),"threshold":c["layer_threshold"],
                          "endpoint":ep,"seed_6198":vals[0],"seed_1024":vals[1],"seed_7":vals[2],
                          "mean_ppl":statistics.mean(vals),"sample_std_ppl":statistics.stdev(vals),"n":3,"status":"completed",
                          "is_new":True,"micro_batch":4,"global_batch":64,"backend":c["attention_backend"],
                          "step":ms[0]["step"],"prediction_tokens":ms[0]["actual_prediction_tokens"],
                          "validation_tokens":ms[0]["validation"]["tokens"],"source_commits":mcommit(first),
                          "core_id":digest(core),"run_paths":[byseed[s]["output_dir"] for s in SEEDS]})
    for run in new["runs"]:
        c=run["config"];m=run["run_manifest.json"]
        entry={"job":str(run["job"]),"task_index":run["task_index"],"dataset":"c4","model_size":"60M","length":2048,
               "method":"hipe","sigma":c["sigma"],"hipe_layers":layer_count(run),"threshold":c["layer_threshold"],"seed":c["seed"],
               "status":"completed","micro_batch":4,"global_batch":64,"backend":c["attention_backend"],
               "source_commit":m["source"]["commit"],"source_snapshot_sha256":m["source"]["snapshot_sha256"],
               "output_dir":run["output_dir"],"exit_code":run["worker_status"]["returncode"]}
        attempts.append(entry)
        for ep in EPS:
            v=metric(run,ep)
            details.append({**entry,"group_id":by_path[run["output_dir"]],"endpoint":ep,"step":v["step"],
                            "prediction_tokens":v["actual_prediction_tokens"],"validation_tokens":v["validation"]["tokens"],
                            "ppl":v["validation"]["ppl"],"ppl_exact_text":repr(v["validation"]["ppl"])})
    assert len(stats)==186 and len(details)==558 and len(attempts)==297
    assert len({r["output_dir"] for r in attempts})==297
    def find(ds,size,length,method,ep):
        return [r for r in stats if (r["dataset"],r["model_size"],r["length"],r["method"],r["endpoint"])==(ds,size,length,method,ep)]
    planned={(s,h) for s in plan["sigma_sweep"] for h in [1,2]}|{(50,h) for h in range(1,9)}
    comparisons=[]
    for g in stats:
        if g["method"]!="hipe":continue
        rope=find(g["dataset"],g["model_size"],g["length"],"rope",g["endpoint"])[0]
        delta=[g["seed_"+str(s)]-rope["seed_"+str(s)] for s in SEEDS]
        is_c4=(g["dataset"],g["model_size"],g["length"])==("c4","60M",2048)
        in_grid=is_c4 and (g["sigma"],g["hipe_layers"]) in planned
        origin="New" if g["is_new"] else "Reused grid control" if in_grid else "Earlier result"
        comparisons.append({**g,"candidate_group_id":g["group_id"],"rope_group_id":rope["group_id"],
            "rope_mean":rope["mean_ppl"],"rope_sd":rope["sample_std_ppl"],
            "rope_seed_6198":rope["seed_6198"],"rope_seed_1024":rope["seed_1024"],"rope_seed_7":rope["seed_7"],
            "delta_ppl":g["mean_ppl"]-rope["mean_ppl"],"gain_vs_rope_percent":100*(1-g["mean_ppl"]/rope["mean_ppl"]),
            "seed_deltas":delta,"seed_wins":sum(v<0 for v in delta),"mean_win":g["mean_ppl"]<rope["mean_ppl"],
            "origin":origin,"in_grid":in_grid,"micro_batch_note":"HIPE and RoPE differ" if g["micro_batch"]!=rope["micro_batch"] else "matched"})
    targets=[r for r in comparisons if (r["dataset"],r["model_size"],r["length"])==("c4","60M",2048)]
    assert len(targets)==60 and sum(r["in_grid"] for r in targets)==48
    assert sum(r["is_new"] for r in targets)==40
    assert not any(r["mean_win"] for r in targets)
    best=[]
    for old in previous["best_rows"]:
        ep=old["endpoint"];ds,size,length=old["dataset"],old["model_size"],old["length"]
        candidates=find(ds,size,length,"hipe",ep)
        chosen=min(candidates,key=lambda r:(r["mean_ppl"],r["sigma"],r["hipe_layers"]))
        row=copy.deepcopy(old)
        row.update(selected_sigma=chosen["sigma"],hipe_layers=chosen["hipe_layers"],selected_group_id=chosen["group_id"],
            candidate_group_ids=[g["group_id"] for g in sorted(candidates,key=lambda g:(g["sigma"],g["hipe_layers"]))],
            candidate_count=len(candidates),hipe_mean=chosen["mean_ppl"],hipe_sd=chosen["sample_std_ppl"],selected_is_new=chosen["is_new"],
            previous_sigma=old["selected_sigma"],previous_hipe_layers=old["hipe_layers"],previous_hipe_mean=old["hipe_mean"],
            previous_hipe_sd=old["hipe_sd"],previous_gain_vs_rope_percent=old["gain_vs_rope_percent"],
            gain_vs_rope_percent=100*(1-chosen["mean_ppl"]/old["rope_mean"]),mean_win=chosen["mean_ppl"]<old["rope_mean"],
            converted_to_win=chosen["mean_ppl"]<old["rope_mean"] and not old["mean_win"])
        row["seed_wins_vs_rope"]=next(g["seed_wins"] for g in comparisons if g["group_id"]==chosen["group_id"] and g["endpoint"]==ep)
        available=[("hipe",chosen["mean_ppl"])]+[(m,row[m+"_mean"]) for m in ["rope","nope","alibi"] if row[m+"_status"]=="completed"]
        row["best_available_method"]=min(available,key=lambda v:v[1])[0];best.append(row)
    summary={"new_runs":60,"new_successful":60,"new_hipe_runs":60,"reused_grid_seed_runs":12,
             "execution_attempts":297,"completed_unique_seed_runs":279,"failed_attempts":18,"fixed_configurations":93,
             "hipe_configurations":61,"c4_60m_l2048_configurations":30,"planned_grid_configurations":24,
             "earlier_out_of_grid_configurations":6,"primary_endpoint":"last_interval","seeds":SEEDS,"max_tokens":100000000,
             "global_batch":64,"previous_primary_wins_vs_rope":previous["summary"]["last_interval"]["wins_vs_rope"]}
    for ep in EPS:
        rows=[r for r in best if r["endpoint"]==ep]
        summary[ep]={"wins_vs_rope":sum(r["mean_win"] for r in rows),"losses_vs_rope":sum(not r["mean_win"] for r in rows),
                     "new_winning_configurations":sum(r["is_new"] and r["mean_win"] for r in targets if r["endpoint"]==ep),
                     "converted_settings":sum(r["converted_to_win"] for r in rows),
                     "paired_seed_wins_selected":sum(r["seed_wins_vs_rope"] for r in rows),
                     "best_methods":dict(Counter(r["best_available_method"] for r in rows))}
    notes=[
      "Mean and sample SD use seeds 6198, 1024 and 7 at one fixed sigma/layer configuration. Configurations are never pooled.",
      "Primary endpoint is last regular validation. Final-checkpoint results are separate. Both select sigma 500 with two HIPE layers for C4 60M/L2048.",
      "None of the 20 new configurations has a lower mean PPL than RoPE. The best observed Table 2 count remains 11/12 settings.",
      "Eight total decoder layers are retained. H1-H7 place HIPE in the final H layers; H8 uses the uniform all-HIPE schedule.",
      "C4 HIPE retains micro-batch 4 versus RoPE micro-batch 8, with global batch 64 unchanged. Cross-method comparisons retain this qualification.",
      "The new training core files exactly match the previous joint-search core. Older cohorts remain distinct; comparisons retain archived unchanged-method evidence.",
      "Public Table 2 methods are HIPE, RoPE, ALiBi and Legacy NoPE+APE. Supplementary comparisons remain local.",
      "Four ALiBi settings remain unavailable due to historical OOM; unavailable PPL is never zero.",
      "All 60 new full runs completed. Twelve earlier seed-runs complete the new grid; four two-step GPU checks are excluded.",
      "These are validation-selected tuning outcomes. Three-seed sample SD does not include selection uncertainty or establish significance."
    ]
    payload={"summary":summary,"notes":notes,"source_files":[str(p) for p in source_paths]+[str(NEW)],
             "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths+[NEW]},
             "additional_input_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [PRIOR,PROTOCOL]},
             "publication_scope":"Paper Table 2 methods only","local_comparisons_excluded":True,
             "best_rows":best,"all_rows":stats,"hipe_rows":[r for r in stats if r["method"]=="hipe"],
             "comparisons":comparisons,"target_rows":targets,"new_winning_rows":[],
             "details":details,"attempts":attempts}
    assert all(r["method"] in METHODS for r in stats+details+attempts)
    (OUT/"workbook_data.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    for name,rows in [("Best_vs_Baselines.csv",best),("All_Statistics.csv",stats),
                      ("C4_All_Configurations.csv",targets),("HIPE_vs_RoPE_All_Configs.csv",comparisons),
                      ("Exact_Seed_PPL.csv",details),("Execution_Attempts.csv",attempts)]:
        write_csv(name,rows)
    report=make_report(payload)
    (OUT/"REPORT.md").write_text(report,encoding="utf-8")
    (ROOT/"analysis/table2/C4_EXTENSION_REPORT.md").write_text(report,encoding="utf-8")
    print(json.dumps(summary,indent=2))
    for ep in EPS:
        selected=next(r for r in best if r["endpoint"]==ep and (r["dataset"],r["model_size"],r["length"])==("c4","60M",2048))
        print("SELECTED",ep,json.dumps(selected))
    print("WORKBOOK_INPUT",OUT/"workbook_data.json")

def mcommit(run):return run["run_manifest.json"]["source"]["commit"]

def make_report(data):
    best=data["best_rows"]
    selected=next(r for r in best if r["endpoint"]=="last_interval" and (r["dataset"],r["model_size"],r["length"])==("c4","60M",2048))
    fmt=lambda mean,sd:f"{mean:.6f} +/- {sd:.6f}"
    lines=["# Completed C4 60M/L2048 extension","",
      "All 60 new full runs completed successfully from published commit c44c207b83bf4140ec462bd727b1c5e6fc946613. Four two-step GPU checks completed separately and are excluded. Twelve earlier seed-runs complete the planned 24-configuration grid; six earlier out-of-grid configurations are also retained.","",
      "## Main result","",
      f"The best observed HIPE result is sigma {selected['selected_sigma']:g} with {selected['hipe_layers']} final HIPE layers: **{fmt(selected['hipe_mean'],selected['hipe_sd'])}** PPL, versus RoPE **{fmt(selected['rope_mean'],selected['rope_sd'])}**. This is {selected['hipe_mean']-selected['rope_mean']:.6f} PPL ({-selected['gain_vs_rope_percent']:.6f}%) higher than RoPE.","",
      f"The previous best HIPE mean was {selected['previous_hipe_mean']:.6f} at sigma {selected['previous_sigma']:g}/H{selected['previous_hipe_layers']}. The extension improves that mean by {selected['previous_hipe_mean']-selected['hipe_mean']:.6f} PPL, but no new configuration surpasses RoPE's mean. The overall best-observed HIPE count remains **11/12 settings**.","",
      "## Ranked C4 configurations","",
      "Primary endpoint: last regular validation. Each mean and sample SD uses all three seeds at the stated fixed configuration.","",
      "| Rank | Sigma | HIPE layers | Origin | HIPE mean +/- SD | Difference from RoPE | PPL reduction vs RoPE | Paired seed wins |",
      "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |"]
    targets=sorted((r for r in data["target_rows"] if r["endpoint"]=="last_interval"),key=lambda r:r["mean_ppl"])
    for i,r in enumerate(targets,1):
        lines.append(f"| {i} | {r['sigma']:g} | {r['hipe_layers']} | {r['origin']} | {fmt(r['mean_ppl'],r['sample_std_ppl'])} | {r['delta_ppl']:+.6f} | {r['gain_vs_rope_percent']:+.6f}% | {r['seed_wins']}/3 |")
    lines+=["","Positive reduction would favor HIPE. All observed primary means are higher than RoPE. The selected sigma 500/H2 result improves in two of three matched seeds, which does not change its higher mean.","",
      "## Endpoint check","",
      "| Endpoint | Sigma | HIPE layers | HIPE mean +/- SD | RoPE mean +/- SD | Difference |",
      "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for ep in EPS:
        r=next(r for r in best if r["endpoint"]==ep and (r["dataset"],r["model_size"],r["length"])==("c4","60M",2048))
        lines.append(f"| {ep} | {r['selected_sigma']:g} | {r['hipe_layers']} | {fmt(r['hipe_mean'],r['hipe_sd'])} | {fmt(r['rope_mean'],r['rope_sd'])} | {r['hipe_mean']-r['rope_mean']:+.6f} |")
    lines+=["","## Interpretation","",
      "At sigma 50, two HIPE layers are best among H1-H8. Performance worsens as the count increases beyond two; the all-HIPE H8 result is 285.486183 +/- 19.961167 PPL. One HIPE layer is not uniformly better than two. Sigma 500/H2 gives the best observed mean over the full C4 search, while both very small sigma values and sigma 2000 underperform it.","",
      "The wider search narrows the earlier mean gap but does not establish a C4 60M/L2048 advantage over RoPE. No further experiments were launched during this analysis.","",
      "## Updated paper-scope Table 2","",
      "| Corpus | Model | L | Sigma | HIPE layers | Legacy NoPE+APE | ALiBi | RoPE | HIPE |",
      "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in best:
        if r["endpoint"]!="last_interval":continue
        f=lambda m:fmt(r[m+"_mean"],r[m+"_sd"]) if r[m+"_status"]=="completed" else r[m+"_status"]
        lines.append(f"| {r['dataset']} | {r['model_size']} | {r['length']} | {r['selected_sigma']:g} | {r['hipe_layers']} | {f('nope')} | {f('alibi')} | {f('rope')} | {fmt(r['hipe_mean'],r['hipe_sd'])} |")
    lines+=["","Among the paper's Table 2 methods, HIPE has the lowest mean in five settings, ALiBi in six and RoPE in one.","",
      "## Protocol and uncertainty","",*["- "+n for n in data["notes"]],""]
    return "\n".join(lines)

if __name__=="__main__":main()
