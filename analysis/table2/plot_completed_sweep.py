"""Standalone scientific figures for the completed L2048 sweep."""
from pathlib import Path
import os,json
ROOT=Path(__file__).resolve().parents[2]
os.environ["MPLCONFIGDIR"]=str(ROOT/"work/matplotlib-cache")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
OUT=ROOT/"outputs/table2_sigma_final"

def main():
    data=json.loads((OUT/"workbook_data.json").read_text())
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,(ds,size) in zip(axes.flat,[("wikitext103","20M"),("wikitext103","60M"),("c4","20M"),("c4","60M")]):
        groups=sorted([r for r in data["hipe_rows"] if r["endpoint"]=="last_interval" and r["length"]==2048 and r["dataset"]==ds and r["model_size"]==size],key=lambda r:r["sigma"])
        best=next(r for r in data["best_rows"] if r["endpoint"]=="last_interval" and r["length"]==2048 and r["dataset"]==ds and r["model_size"]==size)
        rope=best["rope_mean"];sd=best["rope_sd"]
        ax.axhspan(rope-sd,rope+sd,color="#888888",alpha=.12)
        ax.axhline(rope,color="#555555",ls="--",lw=1.3)
        for g in groups:
            is_best=g["sigma"]==best["selected_sigma"]
            ax.errorbar(g["sigma"],g["mean_ppl"],yerr=g["sample_std_ppl"],fmt="*" if is_best else "o",
                        color="#007F73" if is_best else "#2C6D9E",ms=13 if is_best else 6,
                        capsize=4,elinewidth=1.5,zorder=3)
            ax.scatter([g["sigma"]-7,g["sigma"],g["sigma"]+7],[g["seed_6198"],g["seed_1024"],g["seed_7"]],
                       color="#888888",s=15,alpha=.65,zorder=2)
        ax.set_title(("WikiText-103" if ds=="wikitext103" else "C4")+" / "+size,fontweight="bold")
        ax.set_xticks([g["sigma"] for g in groups]);ax.set_xlabel("Sigma (tested values)")
        ax.set_ylabel("Validation PPL");ax.grid(axis="y",alpha=.15)
    handles=[Line2D([0],[0],color="#2C6D9E",marker="o",linestyle="",label="HIPE mean +/- sample SD"),
             Line2D([0],[0],color="#007F73",marker="*",markersize=12,linestyle="",label="Best observed sigma"),
             Line2D([0],[0],color="#555555",linestyle="--",label="RoPE mean (band: +/- sample SD)")]
    fig.legend(handles=handles,loc="lower center",ncol=3,frameon=False)
    fig.suptitle("L2048 sigma sweep: larger sigma is not uniformly better",fontweight="bold")
    fig.text(.5,.94,"Three seeds per sigma; last regular validation. Gray points are individual seeds.",ha="center",fontsize=10)
    fig.tight_layout(rect=[0,.055,1,.91])
    fig.savefig(OUT/"L2048_sigma_sweep.png",dpi=180);plt.close(fig)
    rows=[r for r in data["best_rows"] if r["endpoint"]=="last_interval"]
    matrix=np.array([[next(r["gain_vs_rope_percent"] for r in rows if r["dataset"]==ds and r["model_size"]==size and r["length"]==n) for n in [512,1024,2048]] for ds,size in [("wikitext103","20M"),("wikitext103","60M"),("c4","20M"),("c4","60M")]])
    fig,ax=plt.subplots(figsize=(8.5,4.8))
    image=ax.imshow(matrix,cmap="RdBu",vmin=-4.2,vmax=4.2,aspect="auto")
    ax.set_xticks(range(3),["L512","L1024","L2048"])
    ax.set_yticks(range(4),["WikiText 20M","WikiText 60M","C4 20M","C4 60M"])
    for i,(ds,size) in enumerate([("wikitext103","20M"),("wikitext103","60M"),("c4","20M"),("c4","60M")]):
        for j,n in enumerate([512,1024,2048]):
            r=next(r for r in rows if r["dataset"]==ds and r["model_size"]==size and r["length"]==n)
            ax.text(j,i,f"{matrix[i,j]:+.2f}%\nsigma {r['selected_sigma']:g}",ha="center",va="center",color="white" if abs(matrix[i,j])>2.5 else "#222")
    fig.colorbar(image,ax=ax,label="PPL reduction versus RoPE (%)")
    ax.set_title("Best-observed HIPE versus RoPE",fontweight="bold")
    fig.text(.5,.015,"Positive is better for HIPE. L512/L1024 have one candidate; L2048 has four or five. Validation-selected results.",ha="center",fontsize=8)
    fig.tight_layout(rect=[0,.05,1,.95]);fig.savefig(OUT/"Best_HIPE_vs_RoPE.png",dpi=180);plt.close(fig)
    print("Rendered two standalone figures")

if __name__=="__main__":main()
