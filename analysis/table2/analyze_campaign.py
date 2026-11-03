"""Analyze the two completed Table 2 campaigns; no GPU work or job mutations."""
from pathlib import Path
import os
import argparse
import json
import math
import hashlib
import statistics
import collections
import csv
import re
import platform
import importlib.metadata
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = Path(__file__).parent if (Path(__file__).parent / "raw_runs.json").exists() else ROOT
os.environ["MPLCONFIGDIR"] = str(CACHE_ROOT / "work/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import t as student_t

SEEDS = [6198, 1024, 7]
DATASETS = ["wikitext103", "c4"]
SIZES = ["20M", "60M"]
LENGTHS = [512, 1024, 2048]
METHODS = ["nope", "alibi", "rope", "hipe"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def selected(run, endpoint):
    candidates = [r for r in run["metrics"] if r["regular_validation"]] if endpoint == "last_interval" else [r for r in run["metrics"] if r["final"]]
    result = max(candidates, key=lambda r:r["step"])
    value = result["validation"]
    assert value["tokens"] > 0
    loss = value["loss_sum"] / value["tokens"]
    assert math.isclose(math.exp(loss), value["ppl"], rel_tol=1e-12)
    return {"ppl": math.exp(loss), "loss": loss, "step": result["step"],
            "validation_tokens": value["tokens"], "training_tokens": result["actual_prediction_tokens"]}


def summarize(raw, registry):
    successful = {}
    failed = []
    attempts = []
    for run in raw["runs"]:
        c = run["config"]
        assert c["seed"] in SEEDS and c["max_tokens"] == 100_000_000 and c["max_train_steps"] is None
        key = (c["dataset"], c["model_size"], c["seq_len"], c["method"], c["seed"])
        done = "completion.json" in run
        record = {"job": run["job"], "array_index": run["task_index"], "dataset": key[0],
                  "model_size": key[1], "seq_len": key[2], "method": key[3], "seed": key[4],
                  "completed": done, "exit_code": (run["worker_status"] or {}).get("returncode"),
                  "reason": None if done else ("CUDA OOM" if "OutOfMemoryError" in (run.get("failure_tail") or "") else "other_failure"),
                  "micro_batch": c["micro_batch_size"], "backend": c["attention_backend"], "output_dir": run["output_dir"]}
        attempts.append(record)
        if not done:
            failed.append(record)
            continue
        assert run["worker_status"]["returncode"] == 0
        expected = c["max_tokens"] // (c["global_batch_size"] * c["seq_len"])
        assert run["completion.json"]["step"] == expected
        assert run["metrics"][-1]["step"] == expected
        assert run["completion.json"]["actual_prediction_tokens"] == expected * c["global_batch_size"] * c["seq_len"]
        assert key not in successful, ("Duplicate completed condition", key)
        successful[key] = run
    assert len(successful) == 132 and len(attempts) == 150
    paper = {}
    for r in registry["records"]:
        if r["paper_locator"] == "Table 2":
            paper[(r["dataset"],r["model_size"],r["sequence_length"],r["method"])] = None if r["reported_outcome"] == "oom" else r["reported_metrics"]["validation_ppl"]
    rows, per_seed = [], []
    groups = {}
    for ds in DATASETS:
        for size in SIZES:
            for length in LENGTHS:
                for method in METHODS:
                    key = (ds,size,length,method)
                    runs = [successful[(ds,size,length,method,seed)] for seed in SEEDS if (ds,size,length,method,seed) in successful]
                    assert len(runs) in {0,3}, key
                    groups[key] = runs
                    sigma_values = {r["config"]["sigma"] for r in runs} if method == "hipe" else set()
                    assert len(sigma_values) <= 1, "Fixed-main report requires one sigma; use export_exact_ppl.py for a sweep"
                    sigma_value = next(iter(sigma_values)) if sigma_values else None
                    item = {"dataset":ds,"model_size":size,"seq_len":length,"method":method,
                            "label":"Legacy NoPE+APE" if method=="nope" else method.upper(), "sigma": sigma_value,
                            "n":len(runs),"mean_ppl_last_interval":None,"sd_ppl_last_interval":None,
                            "mean_ppl_final":None,"sd_ppl_final":None,
                            "paper_ppl":paper[key],"paper_status":"OOM" if paper[key] is None else "numeric",
                            "rerun_status":"completed" if runs else "OOM",
                            "mean_minus_paper":None,"percent_vs_paper":None,"micro_batch":None,"backend":None}
                    if runs:
                        endpoints = {name:[selected(r,name) for r in runs] for name in ["last_interval","final"]}
                        for endpoint, values in endpoints.items():
                            item["mean_ppl_"+endpoint] = statistics.mean(v["ppl"] for v in values)
                            item["sd_ppl_"+endpoint] = statistics.stdev(v["ppl"] for v in values)
                        item["micro_batch"] = runs[0]["config"]["micro_batch_size"]
                        item["backend"] = runs[0]["config"]["attention_backend"]
                        if paper[key] is not None:
                            item["mean_minus_paper"] = item["mean_ppl_last_interval"]-paper[key]
                            item["percent_vs_paper"] = 100*(item["mean_ppl_last_interval"]/paper[key]-1)
                        for r in runs:
                            a,b=selected(r,"last_interval"),selected(r,"final")
                            per_seed.append({"dataset":ds,"model_size":size,"seq_len":length,"method":method,
                                             "seed":r["config"]["seed"],"sigma":sigma_value,"ppl_last_interval":a["ppl"],"ppl_final":b["ppl"],
                                             "last_interval_step":a["step"],"last_interval_training_tokens":a["training_tokens"],
                                             "final_step":b["step"],"final_training_tokens":b["training_tokens"],
                                             "validation_tokens":a["validation_tokens"],
                                             "output_dir":r["output_dir"]})
                    rows.append(item)
    paired, audits = [], []
    row_map={(r["dataset"],r["model_size"],r["seq_len"],r["method"]):r for r in rows}
    for ds in DATASETS:
        for size in SIZES:
            for length in LENGTHS:
                rope=groups[(ds,size,length,"rope")]
                hipe=groups[(ds,size,length,"hipe")]
                deltas=[]; final_deltas=[]; seed_records=[]
                for a,b in zip(rope,hipe):
                    assert a["config"]["seed"]==b["config"]["seed"]
                    ma,mb=a["run_manifest.json"],b["run_manifest.json"]
                    assert ma["data"]==mb["data"]
                    assert ma["vocabulary"]==mb["vocabulary"]
                    assert a["completion.json"]["actual_prediction_tokens"]==b["completion.json"]["actual_prediction_tokens"]
                    va,vb=selected(a,"last_interval"),selected(b,"last_interval")
                    fa,fb=selected(a,"final"),selected(b,"final")
                    delta=vb["ppl"]-va["ppl"];deltas.append(delta);final_deltas.append(fb["ppl"]-fa["ppl"])
                    seed_records.append({"seed":a["config"]["seed"],"rope_ppl":va["ppl"],"hipe_ppl":vb["ppl"],"delta_ppl":delta})
                mean=statistics.mean(deltas);sd=statistics.stdev(deltas);se=sd/math.sqrt(3)
                critical=float(student_t.ppf(.975,2));p=float(2*student_t.sf(abs(mean/se),2)) if se else (1.0 if mean==0 else 0.0)
                r=row_map[(ds,size,length,"rope")];h=row_map[(ds,size,length,"hipe")]
                paired.append({"dataset":ds,"model_size":size,"seq_len":length,"sigma":hipe[0]["config"]["sigma"],
                               "rope_mean":r["mean_ppl_last_interval"],"rope_sd":r["sd_ppl_last_interval"],
                               "hipe_mean":h["mean_ppl_last_interval"],"hipe_sd":h["sd_ppl_last_interval"],
                               "delta_ppl":mean,"relative_gain_percent":100*(-mean)/r["mean_ppl_last_interval"],
                               "paired_sd":sd,"ci95_low":mean-critical*se,"ci95_high":mean+critical*se,
                               "paired_t_p_exploratory":p,"holm_p_exploratory":None,
                               "seeds_favor_hipe":sum(d<0 for d in deltas),
                               "final_delta_ppl":statistics.mean(final_deltas),
                               "direction_stable_at_final":(mean<0)==(statistics.mean(final_deltas)<0),
                               "paper_rope":r["paper_ppl"],"paper_hipe":h["paper_ppl"],
                               "seed_deltas":seed_records})
                audits.append({"dataset":ds,"model_size":size,"seq_len":length,
                               "data_identity_equal":True,"vocabulary_equal":True,"token_exposure_equal":True,
                               "rope_micro_batch":rope[0]["config"]["micro_batch_size"],
                               "hipe_micro_batch":hipe[0]["config"]["micro_batch_size"],
                               "global_batch_size":rope[0]["config"]["global_batch_size"],
                               "hipe_sigma":hipe[0]["config"]["sigma"],
                               "hipe_layer_threshold":hipe[0]["config"]["layer_threshold"],
                               "validation_fingerprint":rope[0]["run_manifest.json"]["data"]["processed_validation"]})
    running=0
    for rank,index in enumerate(sorted(range(len(paired)),key=lambda i:paired[i]["paired_t_p_exploratory"])):
        running=max(running,(len(paired)-rank)*paired[index]["paired_t_p_exploratory"])
        paired[index]["holm_p_exploratory"]=min(1.0,running)
    return successful,groups,rows,per_seed,paired,audits,attempts


def plots(groups,paired,out):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":130})
    fig,axes=plt.subplots(1,2,figsize=(12,5.5),sharex=False,sharey=True)
    for ax,ds,title in zip(axes,DATASETS,["WikiText-103","C4"]):
        data=[r for r in paired if r["dataset"]==ds]
        for i,r in enumerate(data):
            color="#007F73" if r["delta_ppl"]<0 else "#BD593A"
            ax.hlines(i,r["ci95_low"],r["ci95_high"],color=color,lw=2)
            ax.scatter([v["delta_ppl"] for v in r["seed_deltas"]],[i-.12,i,i+.12],s=21,color="#707070",alpha=.6,zorder=2)
            ax.scatter([r["delta_ppl"]],[i],s=55,color=color,zorder=3)
        ax.axvline(0,color="#555555",lw=1,linestyle="--")
        ax.set_title(title,fontweight="bold");ax.set_xlabel("HIPE PPL minus RoPE PPL")
        ax.set_yticks(range(6),[f"{r['model_size']}, L={r['seq_len']}" for r in data])
        ax.grid(axis="x",alpha=.15)
    axes[0].invert_yaxis()
    fig.suptitle("HIPE gains are most consistent at L=1024; long-context results are mixed",fontweight="bold")
    fig.text(.5,.015,"Lower is better for HIPE. Colored dots: paired mean; gray dots: three seeds; bars: per-cell 95% paired t intervals (not multiplicity-adjusted).",ha="center",fontsize=9)
    fig.tight_layout(rect=[0,.045,1,.93]);fig.savefig(out/"hipe_vs_rope.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    colors={512:"#147D92",1024:"#AF622C",2048:"#7756A6"}
    for row,ds in enumerate(DATASETS):
        for col,size in enumerate(SIZES):
            ax=axes[row,col]
            for length in LENGTHS:
                for method in ["rope","hipe"]:
                    runs=groups[(ds,size,length,method)]
                    xs=[r["actual_prediction_tokens"]/1e6 for r in runs[0]["metrics"]]
                    values=np.array([[r["validation"]["ppl"] for r in run["metrics"]] for run in runs])
                    assert values.shape[1]==len(xs)
                    mean=values.mean(axis=0);sd=values.std(axis=0,ddof=1)
                    ax.plot(xs,mean,color=colors[length],ls="-" if method=="hipe" else "--",lw=1.7)
                    ax.fill_between(xs,np.maximum(mean-sd,1e-6),mean+sd,color=colors[length],alpha=.09)
            ax.set_title(("WikiText-103" if ds=="wikitext103" else "C4")+" / "+size)
            ax.set_yscale("log");ax.grid(alpha=.15);ax.set_xlim(0,101)
            if row==1:ax.set_xlabel("Training prediction tokens (millions)")
            if col==0:ax.set_ylabel("Validation perplexity (log scale)")
    handles=[Line2D([0],[0],color=colors[n],label=f"L={n}") for n in LENGTHS]
    handles += [Line2D([0],[0],color="#333",ls="--",label="RoPE"),Line2D([0],[0],color="#333",ls="-",label="HIPE")]
    fig.legend(handles=handles,loc="lower center",ncol=5,frameon=False)
    fig.suptitle("Learning curves from all recorded validations",fontweight="bold")
    fig.tight_layout(rect=[0,.055,1,.95]);fig.savefig(out/"learning_curves.png",dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    standalone = (Path(__file__).parent / "raw_runs.json").exists()
    parser.add_argument("--input",default=str(Path(__file__).parent / "raw_runs.json") if standalone else str(ROOT/"artifacts/analysis/table2_20261103/raw_runs.json"))
    parser.add_argument("--registry",default=str(Path(__file__).parent / "reference_registry.json") if standalone else str(ROOT/"experiments/registry.json"))
    parser.add_argument("--output",default=str(Path(__file__).parent / "regenerated") if standalone else str(ROOT/"outputs/table2_20261103"))
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    raw=json.loads(Path(args.input).read_text())
    registry=json.loads(Path(args.registry).read_text())
    complete,groups,rows,seeds,paired,audits,attempts=summarize(raw,registry)
    assert raw["campaigns"][0]["core_file_sha256"]==raw["campaigns"][1]["core_file_sha256"]
    best=[]
    for ds in DATASETS:
        for size in SIZES:
            for length in LENGTHS:
                available=[r for r in rows if r["dataset"]==ds and r["model_size"]==size and r["seq_len"]==length and r["n"]]
                winner=min(available,key=lambda r:r["mean_ppl_last_interval"])
                best.append({"dataset":ds,"model_size":size,"seq_len":length,"best_available_method":winner["method"]})
    coverage_changes=[r for r in rows if (r["paper_status"]=="OOM")!=(r["rerun_status"]=="OOM")]
    summary={"attempts":len(attempts),"completed_unique_runs":len(complete),"failed_attempts":sum(not r["completed"] for r in attempts),
             "unique_planned_runs":144,"unavailable_unique_runs":144-len(complete),"numeric_cells":sum(r["n"]==3 for r in rows),
             "hipe_mean_wins":sum(r["delta_ppl"]<0 for r in paired),"hipe_mean_losses":sum(r["delta_ppl"]>0 for r in paired),
             "paired_seed_wins":sum(r["seeds_favor_hipe"] for r in paired),
             "wins_by_length":{str(n):sum(r["delta_ppl"]<0 for r in paired if r["seq_len"]==n) for n in LENGTHS},
             "same_directions_at_final":all(r["direction_stable_at_final"] for r in paired),
             "equal_cell_average_relative_gain_percent":statistics.mean(r["relative_gain_percent"] for r in paired),
             "holm_significant_cells":sum(r["holm_p_exploratory"]<.05 for r in paired),
             "hipe_best_available_method_cells":sum(r["best_available_method"]=="hipe" for r in best),
             "coverage_changes":coverage_changes,"best_methods":best}
    write_csv(out/"table2_summary.csv",rows);write_csv(out/"per_seed_results.csv",seeds)
    write_csv(out/"paired_comparisons.csv",[{k:v for k,v in r.items() if k!="seed_deltas"} for r in paired])
    write_csv(out/"attempts.csv",attempts);write_csv(out/"protocol_audit.csv",audits)
    (out/"analysis.json").write_text(json.dumps({"summary":summary,"paired":paired,"cells":rows,"protocol_audit":audits},indent=2)+"\n",encoding="utf-8")
    plots(groups,paired,out)
    lines=["# Table 2 three-seed analysis","",
           "The campaigns are finished. This is a new three-seed replication under the recorded protocol, with qualified historical comparability. No new experiment or checkpoint evaluation was launched for this analysis.","",
           "## Main finding","",
           "Each HIPE mean and sample SD is computed over three seeds at one fixed sigma. No different sigma values are averaged together. Sigma is explicit in the tables and CSVs.","",
           f"HIPE has lower mean perplexity than RoPE in **{summary['hipe_mean_wins']}/12 settings**, and in **{summary['paired_seed_wins']}/36 matched-seed comparisons**. It wins 3/4 settings at length512, 4/4 at length1024, and 1/4 at length2048. The directions are unchanged when final-checkpoint evaluations replace the last regular validation.",
           "",
           "These results support a conditional advantage, especially at length1024. They do not reproduce the paper's uniform HIPE improvement over RoPE. A fixed-budget PPL comparison alone does not quantify tokens saved; a sample-efficiency claim additionally needs learning-curve/quality-threshold analysis.",
           "",
           "## Coverage and adjudication","",
           "The input contains150 execution attempts:108 originals and42 recovery attempts. There are132 completed unique main runs and12 unavailable unique runs across144 intended runs. Every numeric cell has all three seeds6198,1024,7. Failed attempts remain in attempts.csv.",
           "",
           "The three successful WikiText20M/L1024 ALiBi retries replace failed originals based on completion, not performance. The three C4 retries failed again. The216 canceled tuning tasks and the GPU preflight are excluded.",
           "",
           "Four cells remain OOM: WikiText20M/L2048, C420M/L1024, C420M/L2048, and C460M/L2048. This differs from the paper: WikiText60M/L2048 ALiBi now runs, while C420M/L1024 does not. Equal numeric-cell counts therefore do not establish identical coverage.",
           "",
           "## Endpoint and aggregation","",
           "The prespecified historical endpoint is the **last regular validation**, not the best validation and not a silently selected seed. For lengths512/1024/2048 it is step3000/1500/700, after98.304M/98.304M/91.7504M prediction tokens. Final checkpoints reach3051/1525/762 steps and99.975168M/99.942400M/99.876864M tokens under the100M cap.",
           "",
           "Mean and sample standard deviation are over three independently trained runs. Each run's PPL is exp(total validation token loss / prediction-token count). We average per-run PPL for the paper-style summary; we never average batch PPLs.",
           "",
           "## Full main table","",
           "| Corpus | Model | Method | L512 PPL | L1024 PPL | L2048 PPL |",
           "| --- | --- | --- | ---: | ---: | ---: |"]
    lookup={(r["dataset"],r["model_size"],r["seq_len"],r["method"]):r for r in rows}
    for ds in DATASETS:
        for size in SIZES:
            for method in METHODS:
                cells=[lookup[(ds,size,n,method)] for n in LENGTHS]
                vals=[f"{r['mean_ppl_last_interval']:.2f} +/- {r['sd_ppl_last_interval']:.2f}" if r["n"] else "OOM (0/3)" for r in cells]
                if method == "hipe":
                    vals = [value + f" (sigma {r['sigma']:g})" if r["n"] else value for value,r in zip(vals,cells)]
                label="Legacy NoPE+APE" if method=="nope" else method.upper()
                lines.append("| "+("WikiText-103" if ds=="wikitext103" else "C4")+" | "+size+" | "+label+" | "+" | ".join(vals)+" |")
    lines += ["","## HIPE versus RoPE","",
              "| Corpus | Model | Length | Sigma | RoPE mean +/- SD | HIPE mean +/- SD | HIPE minus RoPE | Relative PPL reduction | Seeds favoring HIPE |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in paired:
        lines.append(f"| {r['dataset']} | {r['model_size']} | {r['seq_len']} | {r['sigma']:g} | {r['rope_mean']:.2f} +/- {r['rope_sd']:.2f} | {r['hipe_mean']:.2f} +/- {r['hipe_sd']:.2f} | {r['delta_ppl']:+.2f} | {r['relative_gain_percent']:+.2f}% | {r['seeds_favor_hipe']}/3 |")
    lines += ["","![Paired HIPE effects](hipe_vs_rope.png)","",
              "The mean losing settings are WikiText60M/L2048, C420M/L2048, C460M/L512 and C460M/L2048. C460M/L512 is worse for HIPE in each of the three seeds. Long-context effects are small or unfavorable relative to their seed variation.",
              "",
              f"Among methods with successful measurements, HIPE has the lowest mean in {summary['hipe_best_available_method_cells']}/12 settings. This ranking is conditional on the OOM coverage and the legacy NoPE label.",
              "",
              "## Uncertainty","",
              "The effect plot shows paired seed differences and per-cell95% t intervals with2 degrees of freedom. These are exploratory parametric intervals, not a robust small-sample guarantee. The CSV includes exploratory paired t p-values and Holm adjustment across12 comparisons.",
              f"No per-cell comparison survives a5% Holm threshold ({summary['holm_significant_cells']}/12). This limits confidence; it does not establish that every underlying effect is zero. The8/12 and25/36 win counts are descriptive and are not treated as independent-binomial significance tests.",
              "",
              "## Learning curves","",
              "![Learning curves](learning_curves.png)","",
              "The curves show all validation measurements, including the final evaluation, as mean +/- one sample SD. Their x-axis uses actual prediction-token exposure. End-of-training gains should not be read as a demonstrated universal gain across training budgets.",
              "",
              "## Protocol audit and historical comparison","",
              "- All12 paired cells have equal raw/processed train and validation identities, tokenizer/vocabulary, global batch64, and token exposure. The padded vocabulary is50304. The two frozen campaigns have byte-identical trainer, preprocessing, model-config builder and decoder files checked in the input manifest.",
              "- WikiText pairs also share micro-batch. C4 HIPE uses half the RoPE micro-batch, preserving the selected historical profiles. Accumulation/dropout RNG consumption and floating-point rounding can differ, so C4 effects are not a perfectly isolated positional intervention.",
              "- The new HIPE runs use the paper's4/4 hierarchy (threshold3). Five C4 log matches used threshold4, implying5/3. Their sigmas were recovered from those old outcomes; they have not been re-tuned for4/4.",
              "- Our NoPE runs explicitly include legacy learned absolute-position embeddings. Their row should not be presented as a position-free baseline.",
              "- ALiBi settings are cell-specific: original successful cells use the first campaign's policy; the successful Wiki20M/L1024 retry restores automatic PyTorch SDPA and micro-batch16. Remaining OOMs are computational outcomes, not poor numerical PPL values.",
              "- Both WikiText RoPE and HIPE absolute PPLs are substantially below the paper values. This is not exact numerical reproduction. Historical source contains a first20-validation-batch average, while the current trainer evaluates the entire validation population with token weighting. Exact executed historical code/data/validation records remain incomplete; this is a confirmed protocol difference, not a proven single cause of the gap.",
              "",
              "The paper is a single reported value per Table2 cell without the current three-seed uncertainty. table2_summary.csv preserves every paper target, rerun mean and difference. No lower rerun number alone proves a better reproduction.",
              "",
              "## Recommended next checks","",
              "1. Freeze the full validation population, preprocessing and endpoint as the journal comparison protocol; explain the historical endpoint/population differences explicitly.",
              "2. Isolate C4 effects with matched micro-batches/accumulation and a small4/4-versus5/3 schedule control, prioritizing the losing C460M settings before a larger campaign.",
              "3. Audit the original validation protocol/checkpoints where possible. Evaluate existing checkpoints under recovered historical subsets as a separate analysis, without relabeling full-validation scores.",
              "4. Use the existing learning curves to define prospective quality thresholds and targeted budget comparisons. Additional seeds or model scale should follow the protocol checks.",
              "",
              "The legacy NoPE+APE baseline should retain its explicit label; a true position-free NoPE comparison remains a separate validation task.",
              "",
              "No further runs were submitted in producing this report. Raw execution records and generated CSVs support every summary; input hashes and source commits are recorded in analysis_manifest.json.",
              ""]
    report = "\n".join(lines)
    for a,b in [("WikiText20M", "WikiText 20M"),("WikiText60M", "WikiText 60M"),("C420M", "C4 20M"),("C460M", "C4 60M"),("Table2", "Table 2"),("per-cell95%", "per-cell 95%"),("a5%", "a 5%"),("The8/12", "The 8/12"),("and25/36", "and 25/36")]:
        report = report.replace(a,b)
    report = re.sub(r"\b(the|The|All|all|contains|are|and|across|after|at|with|has|step|seeds|first|batch|vocabulary|threshold|length)(?=\d)", r"\1 ", report)
    report = report.replace("current three-seed uncertainty", "current three-seed uncertainty")
    (out/"REPORT.md").write_text(report,encoding="utf-8")
    # A versioned report can be reviewed beside the analysis code.
    if not standalone:
        (Path(__file__).parent/"REPORT.md").write_text(report.replace("](hipe_vs_rope.png)","](../../outputs/table2_20261103/hipe_vs_rope.png)").replace("](learning_curves.png)","](../../outputs/table2_20261103/learning_curves.png)"),encoding="utf-8")
    manifest={"analysis_kind":"Table2_three_seed_replication","input":str(Path(args.input)),
              "input_sha256":sha(args.input),"analysis_script_sha256":sha(__file__),
              "source_commits":[c["manifest"]["source_commit"] for c in raw["campaigns"]],
              "primary_endpoint":"last_interval","sensitivity_endpoint":"final",
              "seeds":SEEDS,"cell_uncertainty":"sample standard deviation",
              "paired_intervals":"95% t intervals,df2; exploratory",
              "multiple_comparisons":"Holm across12 paired t tests",
              "adjudication":"completed recovery replaces failed original; no performance-based selection",
              "summary":summary}
    manifest["environment"] = {"python": platform.python_version(), "packages": {name: importlib.metadata.version(name) for name in ["numpy", "matplotlib", "scipy"]}}
    manifest["reference_registry_sha256"] = sha(args.registry)
    manifest["output_sha256"] = {p.name: sha(p) for p in out.iterdir() if p.is_file() and p.suffix in {".png", ".csv", ".md"}}
    (out/"analysis_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    if Path(args.input).resolve() != (out/"raw_runs.json").resolve():
        shutil.copyfile(args.input,out/"raw_runs.json")
    if Path(args.registry).resolve() != (out/"reference_registry.json").resolve():
        shutil.copyfile(args.registry,out/"reference_registry.json")
    (out/"requirements-analysis.txt").write_text("\n".join(name+"=="+importlib.metadata.version(name) for name in ["numpy", "matplotlib", "scipy"])+"\n",encoding="utf-8")
    shutil.copyfile(__file__,out/"analyze_campaign.py")
    archive=out.parent/"Table2_results_20261103.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file():z.write(p,p.name)
    print(json.dumps(summary,indent=2))
    print("REPORT",out/"REPORT.md")
    print("ARCHIVE",archive)


if __name__=="__main__":
    main()
