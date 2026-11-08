"""Formal paper-scope analysis of the matched-step token-budget exploration."""
from pathlib import Path
from collections import defaultdict,Counter
import copy,csv,hashlib,json,math,statistics
from analyze_joint_search import canonical,digest,config_identity,evidence,metric
ROOT=Path(__file__).resolve().parents[2]
PRIOR=ROOT/"outputs/table2_c4_20261105/workbook_data.json"
NEW=ROOT/"artifacts/analysis/budget_steps_20261108/raw_runs.json"
PLAN=ROOT/"experiments/protocols/table2_budget_steps_v1.json"
OUT=ROOT/"outputs/budget_exploration_20261108"
SEEDS=[6198,1024,7];EPS=["last_interval","final"];BUDGETS={512:100000000,1024:200000000,2048:400000000}
METHODS={"hipe","rope","alibi","nope"}

def hcount(run):
 c=run["config"];m=run.get("run_manifest.json",{}).get("model_config",{})
 if c["method"]!="hipe":return 0
 if c["schedule"]=="uniform":return 8
 if c["schedule"]=="explicit":return sum(v is not None for v in c["layer_sigmas"])
 return 7-c["layer_threshold"]
def write_csv(name,rows):
 with (OUT/name).open("w",encoding="utf-8-sig",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
  for r in rows:w.writerow({k:canonical(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 old=json.loads(PRIOR.read_text(encoding="utf-8"));new=json.loads(NEW.read_text(encoding="utf-8"));plan=json.loads(PLAN.read_text(encoding="utf-8"))
 assert len(new["runs"])==264 and sum("completion.json" in r for r in new["runs"])==262
 core=new["campaigns"][0]["core_file_sha256"]
 allrows=copy.deepcopy(old["all_rows"]);details=copy.deepcopy(old["details"]);attempts=copy.deepcopy(old["attempts"])
 oldmap={(r["group_id"],r["endpoint"]):r for r in old["all_rows"]}
 study=set();groups=defaultdict(list);group_for={}
 for r in allrows:
  r.update(budget_tokens=100000000,is_new=False,selection_eligible=r["length"]==512,origin="Reused L512 study" if r["length"]==512 else "100M matched control")
  if r["length"]==512:study.add(r["group_id"])
 for r in details+attempts:r.update(budget_tokens=100000000,origin="Reused L512 study" if r["length"]==512 else "100M matched control")
 for run in new["runs"]:
  c=run["config"];assert c["method"] in METHODS and c["seq_len"] in [1024,2048]
  assert c["max_tokens"]==BUDGETS[c["seq_len"]] and c["global_batch_size"]==64 and c["max_train_steps"] is None
  assert run["run_manifest.json"]["source"]["commit"]==run["run_manifest.json"]["source"]["publication"]["commit"]=="12ebc719d980d8dd4fdf7a7bef6cf2ead40719b6"
  assert run["run_manifest.json"]["experiment"]==c
  ident={"config":config_identity(c),"evidence":evidence(run),"core_file_sha256":core}
  gid=digest(ident);groups[gid].append(run);group_for[run["task_index"]]=gid;study.add(gid)
 assert len(groups)==88 and len(study)==109
 for gid,runs in groups.items():
  assert len(runs)==3 and {r["config"]["seed"] for r in runs}==set(SEEDS)
  c=runs[0]["config"];successful={r["config"]["seed"]:r for r in runs if "completion.json" in r}
  n=len(successful);assert n in [2,3]
  for r in successful.values():
   assert r["worker_status"]["returncode"]==0 and r["completion.json"]["step"]==3051
  for ep in EPS:
   ms={s:metric(r,ep) for s,r in successful.items()};vals={s:m["validation"]["ppl"] for s,m in ms.items()}
   assert len({(m["step"],m["actual_prediction_tokens"],m["validation"]["tokens"]) for m in ms.values()})==1
   first=next(iter(ms.values()))
   allrows.append({"group_id":gid,"dataset":c["dataset"],"model_size":c["model_size"],"length":c["seq_len"],
    "budget_tokens":c["max_tokens"],"method":c["method"],"method_label":{"hipe":"HIPE","rope":"RoPE","alibi":"ALiBi","nope":"Legacy NoPE+APE"}[c["method"]],
    "sigma":c["sigma"] if c["method"]=="hipe" else None,"hipe_layers":hcount(runs[0]),
    "threshold":c["layer_threshold"] if c["method"]=="hipe" else None,"endpoint":ep,
    "seed_6198":vals.get(6198),"seed_1024":vals.get(1024),"seed_7":vals.get(7),
    "mean_ppl":statistics.mean(vals.values()) if n==3 else None,
    "sample_std_ppl":statistics.stdev(vals.values()) if n==3 else None,
    "n":n,"status":"completed" if n==3 else "Incomplete: NODE_FAIL","selection_eligible":n==3,
    "is_new":True,"origin":"Exploration","micro_batch":c["micro_batch_size"],"global_batch":64,"backend":c["attention_backend"],
    "step":first["step"],"prediction_tokens":first["actual_prediction_tokens"],"validation_tokens":first["validation"]["tokens"],
    "source_commits":"12ebc719d980d8dd4fdf7a7bef6cf2ead40719b6","core_id":digest(core),
    "run_paths":[successful[s]["output_dir"] for s in SEEDS if s in successful]})
 for run in new["runs"]:
  c=run["config"];m=run["run_manifest.json"];complete="completion.json" in run
  entry={"job":str(run["job"]),"task_index":run["task_index"],"dataset":c["dataset"],"model_size":c["model_size"],"length":c["seq_len"],
    "budget_tokens":c["max_tokens"],"method":c["method"],"sigma":c["sigma"] if c["method"]=="hipe" else None,
    "hipe_layers":hcount(run),"threshold":c["layer_threshold"] if c["method"]=="hipe" else None,"seed":c["seed"],
    "status":"completed" if complete else run["scheduler_state"],"micro_batch":c["micro_batch_size"],"global_batch":64,"backend":c["attention_backend"],
    "source_commit":m["source"]["commit"],"source_snapshot_sha256":m["source"]["snapshot_sha256"],
    "output_dir":run["output_dir"],"exit_code":0 if complete else None,"origin":"Exploration"}
  attempts.append(entry)
  if not complete:continue
  for ep in EPS:
   val=metric(run,ep)
   details.append({**entry,"group_id":group_for[run["task_index"]],"endpoint":ep,"step":val["step"],
     "prediction_tokens":val["actual_prediction_tokens"],"validation_tokens":val["validation"]["tokens"],
     "ppl":val["validation"]["ppl"],"ppl_exact_text":repr(val["validation"]["ppl"])})
 assert len(allrows)==362 and len(details)==1082 and len(attempts)==561
 lookup={(r["group_id"],r["endpoint"]):r for r in allrows}
 studyrows=[r for r in allrows if r["group_id"] in study]
 def fixed(ds,size,length,method,ep):
  return [r for r in studyrows if (r["dataset"],r["model_size"],r["length"],r["method"],r["endpoint"])==(ds,size,length,method,ep)]
 targets=[]
 for g in studyrows:
  if g["method"]!="hipe":continue
  rope=fixed(g["dataset"],g["model_size"],g["length"],"rope",g["endpoint"])[0]
  assert (g["budget_tokens"],g["step"],g["prediction_tokens"],g["validation_tokens"])==(rope["budget_tokens"],rope["step"],rope["prediction_tokens"],rope["validation_tokens"])
  eligible=g["n"]==3
  seed_deltas=[g["seed_"+str(s)]-rope["seed_"+str(s)] if g["seed_"+str(s)] is not None else None for s in SEEDS]
  targets.append({**g,"candidate_group_id":g["group_id"],"rope_group_id":rope["group_id"],
   "rope_mean":rope["mean_ppl"],"rope_sd":rope["sample_std_ppl"],"rope_seed_6198":rope["seed_6198"],"rope_seed_1024":rope["seed_1024"],"rope_seed_7":rope["seed_7"],
   "delta_ppl":g["mean_ppl"]-rope["mean_ppl"] if eligible else None,
   "gain_vs_rope_percent":100*(1-g["mean_ppl"]/rope["mean_ppl"]) if eligible else None,
   "seed_deltas":seed_deltas,"seed_wins":sum(v<0 for v in seed_deltas if v is not None) if eligible else None,
   "mean_win":g["mean_ppl"]<rope["mean_ppl"] if eligible else None,
   "micro_batch_note":"HIPE and RoPE differ" if g["micro_batch"]!=rope["micro_batch"] else "matched"})
 assert len(targets)==154
 best=[]
 for ep in EPS:
  for length in [512,1024,2048]:
   for ds in ["wikitext103","c4"]:
    for size in ["20M","60M"]:
     candidates=fixed(ds,size,length,"hipe",ep);eligible=[r for r in candidates if r["n"]==3]
     chosen=min(eligible,key=lambda r:(r["mean_ppl"],r["sigma"],r["hipe_layers"]))
     oldbest=next(r for r in old["best_rows"] if (r["dataset"],r["model_size"],r["length"],r["endpoint"])==(ds,size,length,ep))
     row={"dataset":ds,"model_size":size,"length":length,"budget_tokens":BUDGETS[length],"endpoint":ep,
      "selected_sigma":chosen["sigma"],"hipe_layers":chosen["hipe_layers"],"candidate_count":len(candidates),
      "complete_candidate_count":len(eligible),"incomplete_candidate_count":len(candidates)-len(eligible),
      "candidate_group_ids":[r["group_id"] for r in candidates],"selected_group_id":chosen["group_id"],"n":3,
      "hipe_mean":chosen["mean_ppl"],"hipe_sd":chosen["sample_std_ppl"],"baseline_group_ids":{},
      "previous_hipe_mean":oldbest["hipe_mean"],"previous_hipe_sd":oldbest["hipe_sd"],"previous_sigma":oldbest["selected_sigma"],
      "previous_hipe_layers":oldbest["hipe_layers"],"previous_gain_vs_rope_percent":oldbest["gain_vs_rope_percent"]}
     for method in ["rope","alibi","nope"]:
      matches=fixed(ds,size,length,method,ep)
      if matches:
       assert len(matches)==1 and matches[0]["n"]==3
       m=matches[0];row["baseline_group_ids"][method]=m["group_id"]
       row[method+"_mean"]=m["mean_ppl"];row[method+"_sd"]=m["sample_std_ppl"];row[method+"_status"]="completed"
      else:
       assert method=="alibi";row[method+"_mean"]=row[method+"_sd"]=None;row[method+"_status"]="Original-shape OOM"
     row["gain_vs_rope_percent"]=100*(1-row["hipe_mean"]/row["rope_mean"])
     row["mean_win"]=row["hipe_mean"]<row["rope_mean"]
     row["converted_to_win"]=row["mean_win"] and not oldbest["mean_win"]
     row["seed_wins_vs_rope"]=next(t["seed_wins"] for t in targets if t["group_id"]==chosen["group_id"] and t["endpoint"]==ep)
     row["best_available_method"]=min([("hipe",row["hipe_mean"])]+[(m,row[m+"_mean"]) for m in ["rope","alibi","nope"] if row[m+"_status"]=="completed"],key=lambda v:v[1])[0]
     best.append(row)
 changes=[]
 for index,m in enumerate(plan["new_configuration_mapping"]):
  gid=group_for[index*3]
  for ep in EPS:
   g=lookup[gid,ep];cid=m["reference_group_id"];control=lookup[cid,ep] if cid else None
   ok=control is not None and g["n"]==3
   changes.append({**g,"control_group_id":cid,"control_status":"Exact 100M control" if control else "No exact 100M control",
    "control_budget_tokens":100000000 if control else None,"control_mean":control["mean_ppl"] if control else None,
    "control_sd":control["sample_std_ppl"] if control else None,
    "control_step":control["step"] if control else None,"control_prediction_tokens":control["prediction_tokens"] if control else None,
    "delta_ppl":g["mean_ppl"]-control["mean_ppl"] if ok else None,
    "gain_percent":100*(1-g["mean_ppl"]/control["mean_ppl"]) if ok else None,
    "gain_vs_control_percent":100*(1-g["mean_ppl"]/control["mean_ppl"]) if ok else None,
    "control_seed_6198":control["seed_6198"] if control else None,
    "control_seed_1024":control["seed_1024"] if control else None,"control_seed_7":control["seed_7"] if control else None})
 assert len(changes)==176 and sum(c["control_group_id"] is not None for c in changes)==144
 summary={"planned_new_runs":264,"new_runs":264,"new_successful":262,"new_interrupted":2,"new_hipe_runs":204,"new_baseline_runs":60,
  "reused_L512_runs":63,"study_expected_seed_runs":327,"study_completed_seed_runs":325,
  "all_input_attempts":561,"execution_attempts":561,"completed_unique_seed_runs":541,"failed_attempts":20,
  "all_fixed_protocols":181,"fixed_configurations":181,"study_fixed_configurations":109,
  "study_complete_configurations":107,"study_incomplete_configurations":2,"control_only_configurations":72,
  "exact100M_control_pairs":72,"complete100M_control_pairs":71,"new_layouts_without100M_control":16,
  "primary_endpoint":"last_interval","nominal_steps":3051,"nominal_primary_step":3000,
  "seeds":SEEDS,"previous_paper100M_wins_vs_rope":11}
 for ep in EPS:
  rs=[r for r in best if r["endpoint"]==ep];cs=[c for c in changes if c["endpoint"]==ep and c["gain_percent"] is not None]
  summary[ep]={"wins_vs_rope":sum(r["mean_win"] for r in rs),"losses_vs_rope":sum(not r["mean_win"] for r in rs),
   "wins_by_length":{str(l):sum(r["mean_win"] for r in rs if r["length"]==l) for l in [512,1024,2048]},
   "paired_seed_wins_selected":sum(r["seed_wins_vs_rope"] for r in rs),
   "best_methods":dict(Counter(r["best_available_method"] for r in rs)),
   "complete_HIPE_fixed_configs":sum(t["n"]==3 for t in targets if t["endpoint"]==ep),
   "HIPE_fixed_config_wins":sum(t["mean_win"] is True for t in targets if t["endpoint"]==ep),
   "complete100M_control_improvements":sum(c["gain_percent"]>0 for c in cs)}
 notes=[
  "Study budgets: L512/100M (reused), L1024/200M and L2048/400M. Each has 3051 nominal optimizer steps; actual token exposures are retained.",
  "Mean and sample SD use seeds 6198, 1024 and 7 at one fixed budget/sigma/layer protocol. Incomplete n=2 groups are visible but ineligible for selection.",
  "Primary endpoint is last regular validation; final checkpoints are a separate view. Select only complete three-seed candidate means.",
  "262 new runs completed; two ended NODE_FAIL. The step-1000 partial checkpoint and empty failed run are excluded from endpoint statistics.",
  "This exploration is separate from the paper's 100M-token results. Current study best-complete HIPE wins are 12/12; original 100M results remain 11/12.",
  "Old 100M long-length groups are controls only, never candidates for the matched-step study summary. Sixteen new L1024 layouts have no exact 100M control.",
  "Data, optimizer and original micro-batches/backends were retained. C4 HIPE/RoPE micro-batches differ; global batch remains 64.",
  "Matching optimizer steps is not matching tokens, compute or unique data. WikiText repeats a fixed corpus and has short epoch-tail micro-batches.",
  "Public methods: HIPE, RoPE, ALiBi and Legacy NoPE+APE. Four original-shape ALiBi OOM cells remain unavailable; supplementary comparisons stay local.",
  "Best configurations are validation-selected. Seed SD does not include selection uncertainty and does not establish statistical significance."
 ]
 payload={"summary":summary,"notes":notes,"source_files":old["source_files"]+[str(NEW)],
  "source_sha256":{**old["source_sha256"],str(NEW):hashlib.sha256(NEW.read_bytes()).hexdigest()},
  "additional_input_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [PRIOR,PLAN]},
  "publication_scope":"Paper Table2 methods only","study_group_ids":sorted(study),
  "all_rows":allrows,"details":details,"attempts":attempts,"best_rows":best,"target_rows":targets,"change_rows":changes,
  "group_intents":{gid:[r["task_index"] for r in rs] for gid,rs in groups.items()}}
 assert all(r["method"] in METHODS for r in allrows+details+attempts)
 (OUT/"workbook_data.json").write_bytes((json.dumps(payload,indent=2)+"\n").encode())
 (OUT/"summary.json").write_bytes((json.dumps(summary,indent=2)+"\n").encode())
 for name,rows in [("Best_vs_Baselines.csv",best),("HIPE_vs_RoPE.csv",targets),("Budget_Changes.csv",changes),
                   ("All_Statistics.csv",allrows),("Exact_Seed_PPL.csv",details),("Attempts.csv",attempts)]:write_csv(name,rows)
 print(json.dumps(summary,indent=2))
 print("PAYLOAD_READY",OUT/"workbook_data.json")
if __name__=="__main__":main()
