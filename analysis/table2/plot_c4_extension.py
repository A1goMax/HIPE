"""Scientific figure for the completed wider C4 sigma/layer search."""
from pathlib import Path
import os,json,hashlib
ROOT=Path(__file__).resolve().parents[2]
os.environ["MPLCONFIGDIR"]=str(ROOT/"work/matplotlib-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def main():
    out=ROOT/"outputs/table2_c4_20261105"
    d=json.loads((out/"workbook_data.json").read_text(encoding="utf-8"))
    rows=[r for r in d["target_rows"] if r["endpoint"]=="last_interval"]
    selected=next(r for r in d["best_rows"] if (r["dataset"],r["model_size"],r["length"],r["endpoint"])==("c4","60M",2048,"last_interval"))
    rope,sd=selected["rope_mean"],selected["rope_sd"]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(12,5.7))
    colors=["#326C9B","#BD6E2C"]
    sigmas=[1,5,10,25,50,100,200,500,2000]
    for ax in axes:
        ax.axhspan(rope-sd,rope+sd,color="#777777",alpha=.12,zorder=0)
        ax.axhline(rope,color="#555555",ls="--",lw=1.3)
        ax.set_ylabel("Validation PPL");ax.grid(axis="y",alpha=.15)
    ax=axes[0]
    for j,h in enumerate([1,2]):
        vals=sorted([r for r in rows if r["hipe_layers"]==h and r["sigma"] in sigmas],key=lambda r:r["sigma"])
        assert len(vals)==9
        ax.errorbar([r["sigma"] for r in vals],[r["mean_ppl"] for r in vals],yerr=[r["sample_std_ppl"] for r in vals],
                    label=f"{h} HIPE layer"+("s" if h>1 else ""),color=colors[j],marker="o" if h==1 else "s",capsize=3,lw=1.2,ms=5)
        for r in vals:
            x=np.array([r["sigma"]*.96,r["sigma"],r["sigma"]*1.04])
            ax.scatter(x,[r["seed_6198"],r["seed_1024"],r["seed_7"]],color="#777777",alpha=.5,s=12,zorder=2)
    ax.scatter([500],[selected["hipe_mean"]],marker="*",s=180,color="#006F67",edgecolors="white",linewidths=.6,zorder=5)
    ax.set_xscale("log");ax.set_xticks(sigmas,[str(s) for s in sigmas]);ax.tick_params(axis="x",labelsize=8)
    ax.set_xlabel("Sigma (log scale)");ax.set_title("Sigma sweep at one/two HIPE layers",fontweight="bold")
    ax.legend(frameon=False,loc="upper left")
    ax=axes[1]
    vals=sorted([r for r in rows if r["sigma"]==50],key=lambda r:r["hipe_layers"]);assert len(vals)==8
    ax.errorbar([r["hipe_layers"] for r in vals],[r["mean_ppl"] for r in vals],yerr=[r["sample_std_ppl"] for r in vals],
                color=colors[0],marker="o",capsize=4,lw=1.3)
    for r in vals:
        ax.scatter([r["hipe_layers"]-.08,r["hipe_layers"],r["hipe_layers"]+.08],
                   [r["seed_6198"],r["seed_1024"],r["seed_7"]],color="#777777",alpha=.6,s=15,zorder=2)
    ax.set_xticks(range(1,9));ax.set_xlabel("Number of final HIPE layers (8 total)")
    ax.set_title("Layer sweep at sigma 50",fontweight="bold")
    fig.suptitle("C4 60M/L2048: wider search still trails RoPE's mean",fontweight="bold",fontsize=14)
    fig.legend(handles=[Line2D([0],[0],color="#555555",ls="--",label="RoPE mean (band: sample SD)"),
                       Line2D([0],[0],color="#777777",marker="o",linestyle="",label="Individual HIPE seeds"),
                       Line2D([0],[0],color="#006F67",marker="*",markersize=10,linestyle="",label="Best observed HIPE: sigma 500/H2")],
               loc="lower center",bbox_to_anchor=(.5,.095),ncol=3,frameon=False,fontsize=9)
    fig.text(.5,.06,"Means and sample SD use three seeds at one fixed configuration; primary endpoint is last regular validation.",ha="center",fontsize=9)
    fig.text(.5,.025,"Validation-selected results. Original micro-batches remain HIPE 4 versus RoPE 8; global batch is 64.",ha="center",fontsize=9)
    fig.tight_layout(rect=[0,.16,1,.94])
    paths=[out/"C4_sigma_layer_effects.png",out/"C4_sigma_layer_effects.svg"]
    fig.savefig(paths[0],dpi=190);fig.savefig(paths[1]);plt.close(fig)
    result={"unique_fixed_groups_in_figure":24,"plotted_series_points":26,"input_sha256":hashlib.sha256((out/"workbook_data.json").read_bytes()).hexdigest(),
            "files":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (out/"FIGURE_VERIFICATION.json").write_text(json.dumps(result,indent=2)+"\n")
    print("Rendered scientific figure",paths[0])
if __name__=="__main__":main()
