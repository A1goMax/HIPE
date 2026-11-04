"""Publication projection for the paper's Table 2 method set.

Additional local comparisons are excluded by construction. This module never
modifies the full local analysis inputs or rewrites Git history.
"""
from pathlib import Path
from collections import Counter
import argparse, copy, csv, hashlib, json

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_SOURCE=ROOT/"outputs/table2_joint_20261104/workbook_data.json"
DEFAULT_OUT=ROOT/"outputs/table2_public_20261104"
METHODS={"hipe","rope","alibi","nope"}

def public_projection(source):
    full=json.loads(Path(source).read_text(encoding="utf-8"))
    data=copy.deepcopy(full)
    for key in ["all_rows","hipe_rows","comparisons","target_rows","new_winning_rows","details","attempts"]:
        data[key]=[r for r in data[key] if r.get("method") in METHODS]
    for row in data["best_rows"]:
        for key in [k for k in row if k.startswith("xpos_")]:row.pop(key)
        row["baseline_group_ids"].pop("xpos",None)
        available=[("hipe",row["hipe_mean"])]
        available += [(m,row[m+"_mean"]) for m in ["rope","nope","alibi"] if row[m+"_status"]=="completed"]
        row["best_available_method"]=min(available,key=lambda item:item[1])[0]
    summary=data["summary"]
    summary.pop("new_xpos_runs",None)
    summary.update(new_runs=sum(r["job"]=="41770" for r in data["attempts"]),
                   new_successful=sum(r["job"]=="41770" and r["status"]=="completed" for r in data["attempts"]),
                   execution_attempts=len(data["attempts"]),
                   completed_unique_seed_runs=sum(r["status"]=="completed" for r in data["attempts"]),
                   failed_attempts=sum(r["status"]!="completed" for r in data["attempts"]),
                   fixed_configurations=len(data["all_rows"])//2)
    for ep in ["last_interval","final"]:
        rows=[r for r in data["best_rows"] if r["endpoint"]==ep]
        summary[ep].pop("xpos_wins_vs_rope",None);summary[ep].pop("hipe_wins_vs_xpos",None)
        summary[ep]["best_methods"]=dict(Counter(r["best_available_method"] for r in rows))
    data["notes"][6]="Public Table 2 methods are HIPE, RoPE, ALiBi and Legacy NoPE+APE. Additional comparison results remain local."
    data["notes"][8]="All 45 new HIPE runs completed. Nine previous HIPE runs complete the target grid. Twelve two-step GPU checks are excluded."
    data["publication_scope"]="Paper Table 2 methods only"
    data["local_comparisons_excluded"]=True
    data["projection_source_sha256"]=hashlib.sha256(Path(source).read_bytes()).hexdigest()
    # The reference-method audit is implementation evidence, not a Table 2 result.
    data["compatibility_evidence"]={"regression":data["compatibility_evidence"]["regression"]}
    assert len(data["all_rows"])==146 and len(data["details"])==438 and len(data["attempts"])==237
    assert summary["completed_unique_seed_runs"]==219
    assert summary["last_interval"]["best_methods"]=={"hipe":5,"alibi":6,"rope":1}
    assert all(row["method"] in METHODS for row in data["all_rows"])
    return data

def report(data):
    fmt=lambda mean,sd:f"{mean:.6f} +/- {sd:.6f}"
    lines=["# Table 2 public results","",
      "This report follows the paper's Table 2 method set: HIPE, RoPE, ALiBi and Legacy NoPE+APE. Additional comparisons are retained in local analysis outputs and are not included in this public table.","",
      "The 45 new HIPE runs completed. Nine earlier HIPE runs complete the targeted 18-configuration grid. Means and sample SDs use seeds 6198, 1024 and 7 at one fixed sigma/layer setting. GPU checks are excluded.","",
      "## Main finding","",
      "Best observed HIPE has lower mean PPL than RoPE in 11/12 settings, up from 9/12 before the layer search. This uses per-setting sigma/layer selection. Two previously losing settings become mean wins; C4 60M/L2048 remains a mean loss.","",
      "## Newly mean-winning configurations","",
      "| Corpus | Model | L | Sigma | HIPE layers | HIPE mean +/- SD | RoPE mean +/- SD | Reduction | Paired seed wins |",
      "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in data["new_winning_rows"]:
        if r["endpoint"]=="last_interval":
            lines.append(f"| {r['dataset']} | {r['model_size']} | {r['length']} | {r['sigma']:g} | {r['hipe_layers']} | {fmt(r['mean_ppl'],r['sample_std_ppl'])} | {fmt(r['rope_mean'],r['rope_sd'])} | {r['gain_vs_rope_percent']:.6f}% | {r['seed_wins']}/3 |")
    lines+=["","## Best observed HIPE and Table 2 baselines","",
      "| Corpus | Model | L | Sigma | HIPE layers | Legacy NoPE+APE | ALiBi | RoPE | HIPE |",
      "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in data["best_rows"]:
        if r["endpoint"]!="last_interval":continue
        f=lambda m:fmt(r[m+"_mean"],r[m+"_sd"]) if r[m+"_status"]=="completed" else r[m+"_status"]
        lines.append(f"| {r['dataset']} | {r['model_size']} | {r['length']} | {r['selected_sigma']:g} | {r['hipe_layers']} | {f('nope')} | {f('alibi')} | {f('rope')} | {fmt(r['hipe_mean'],r['hipe_sd'])} |")
    lines+=["","Among the Table 2 methods, the lowest available primary means belong to HIPE in five settings, ALiBi in six and RoPE in one.","",
      "C4 60M/L2048 remains a mean loss: sigma 50 with two HIPE layers gives 242.677564 +/- 6.525953 versus RoPE 241.550843 +/- 12.911822. Both endpoint views preserve the selected configurations and win/loss directions.","",
      "The new positive mean differences are small and validation-selected. The C4 L512 mean wins favor HIPE in only one of three paired seeds. These are exploratory tuning outcomes, not independent held-out superiority or significance claims.","",
      "## Protocol qualifications","",*["- "+note for note in data["notes"]],"",
      "The full local comparison archive remains unchanged. Historical Git snapshots are preserved; this is a forward publication-scope update.",""]
    return "\n".join(lines)

def write_csv(out,name,rows):
    with (out/name).open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,separators=(",",":")) if isinstance(v,(dict,list)) else v for k,v in row.items()})

def publish(source=DEFAULT_SOURCE,out=DEFAULT_OUT):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    data=public_projection(source)
    (out/"workbook_data.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
    (out/"summary.json").write_text(json.dumps(data["summary"],indent=2)+"\n",encoding="utf-8")
    for name,key in [("Best_vs_Baselines.csv","best_rows"),("All_Statistics.csv","all_rows"),
                     ("HIPE_vs_RoPE_All_Configs.csv","comparisons"),("Target_Grid.csv","target_rows"),
                     ("Exact_Seed_PPL.csv","details"),("Execution_Attempts.csv","attempts")]:
        write_csv(out,name,data[key])
    text=report(data)
    (out/"REPORT.md").write_text(text,encoding="utf-8")
    (ROOT/"analysis/table2/JOINT_SEARCH_REPORT.md").write_text(text,encoding="utf-8")
    print("PUBLIC_TABLE2",out,"fixed groups",len(data["all_rows"])//2,"observations",len(data["details"]))
    return data

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=DEFAULT_SOURCE)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUT)
    args=parser.parse_args();publish(args.source,args.output)
