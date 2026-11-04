"""Standalone scientific figures for the completed joint sigma/layer study."""
from pathlib import Path
import argparse
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[2]
os.environ["MPLCONFIGDIR"] = str(ROOT / "work/matplotlib-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

TARGETS = [
    ("wikitext103", "60M", 2048, [500, 800]),
    ("c4", "60M", 512, [200, 500]),
    ("c4", "60M", 2048, [25, 50]),
]
SETTINGS = [("wikitext103", "20M"), ("wikitext103", "60M"),
            ("c4", "20M"), ("c4", "60M")]
LENGTHS = [512, 1024, 2048]
SEEDS = [6198, 1024, 7]
PRIMARY = "last_interval"


def setting(row):
    return row["dataset"], row["model_size"], row["length"]


def hipe_layers(row):
    for name in ["hipe_layer_count", "hipe_layers", "selected_hipe_layer_count",
                 "selected_hipe_layers", "selected_layer_count"]:
        if row.get(name) is not None:
            return int(row[name])
    for name in ["layer_threshold", "threshold"]:
        if row.get(name) is not None:
            return 7 - int(row[name])
    raise ValueError("Missing effective HIPE layer count")


def current_hipe_rows(data):
    rows = data.get("hipe_rows")
    if rows is None:
        rows = [r for r in data["all_rows"] if r["method"] == "hipe"]
    return [r for r in rows if r["endpoint"] == PRIMARY]


def selected_layers(row, hipe):
    if row.get("selected_group_id"):
        matches = [r for r in hipe if r["group_id"] == row["selected_group_id"]
                   and r["endpoint"] == row["endpoint"]]
        if len(matches) == 1:
            return hipe_layers(matches[0])
    return hipe_layers(row)


def targeted_figure(data, out):
    hipe = current_hipe_rows(data)
    best = [r for r in data["best_rows"] if r["endpoint"] == PRIMARY]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.5))
    colors = ["#2C6D9E", "#B66C2B"]
    markers = ["o", "s"]
    count = 0
    for ax, (dataset, size, length, sigmas) in zip(axes, TARGETS):
        chosen = next(r for r in best if setting(r) == (dataset, size, length))
        rows = [r for r in hipe if setting(r) == (dataset, size, length)
                and r["sigma"] in sigmas and hipe_layers(r) in [2, 4, 6]]
        assert len(rows) == 6
        rope, rope_sd = chosen["rope_mean"], chosen["rope_sd"]
        ax.axhspan(rope - rope_sd, rope + rope_sd, color="#8C949B", alpha=.14, zorder=0)
        ax.axhline(rope, color="#5D646A", ls="--", lw=1.35, zorder=1)
        for j, sigma in enumerate(sigmas):
            subset = sorted((r for r in rows if r["sigma"] == sigma), key=hipe_layers)
            x = np.array([hipe_layers(r) for r in subset], dtype=float) + (-.15 if j == 0 else .15)
            means = [r["mean_ppl"] for r in subset]
            sds = [r["sample_std_ppl"] for r in subset]
            ax.errorbar(x, means, yerr=sds, color=colors[j], marker=markers[j],
                        ms=6, capsize=4, lw=1.1, elinewidth=1.35,
                        label=f"Sigma {sigma}", zorder=3)
            for xx, row in zip(x, subset):
                values = [row[f"seed_{s}"] for s in SEEDS]
                assert row["n"] == 3
                assert np.isclose(np.mean(values), row["mean_ppl"], rtol=1e-12)
                assert np.isclose(np.std(values, ddof=1), row["sample_std_ppl"], rtol=1e-12)
                ax.scatter(xx + np.array([-.055, 0, .055]), values,
                           color="#6D747B", alpha=.54, s=15, zorder=2)
                count += 1
                if (row["sigma"] == chosen["selected_sigma"]
                        and hipe_layers(row) == selected_layers(chosen, hipe)):
                    ax.scatter([xx], [row["mean_ppl"]], marker="*", s=180,
                               color="#067D70", edgecolor="white", linewidth=.5, zorder=5)
        ds_label = "WikiText-103" if dataset == "wikitext103" else "C4"
        gain = chosen["gain_vs_rope_percent"]
        ax.set_title(f"{ds_label} {size}, L={length}\nBest mean PPL reduction: {gain:+.3f}%",
                     fontsize=11, pad=11, fontweight="bold")
        ax.set_xticks([2, 4, 6])
        ax.set_xlim(1.5, 6.5)
        ax.set_xlabel("Number of final HIPE layers (8 total)")
        ax.set_ylabel("Validation PPL")
        ax.grid(axis="y", color="#CED4D9", alpha=.4, lw=.6)
        ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=.85, fontsize=9)
    fig.suptitle("Targeted sigma/layer search", fontsize=15, fontweight="bold", y=.985)
    fig.text(.5, .915, "Three seeds per fixed configuration. Error bars show sample SD; gray points are individual seeds.",
             ha="center", fontsize=10)
    handles = [
        Line2D([0], [0], color="#5D646A", linestyle="--", label="RoPE mean"),
        Patch(facecolor="#8C949B", alpha=.20, label="RoPE mean +/- sample SD"),
        Line2D([0], [0], marker="*", color="#067D70", linestyle="", markersize=11,
               label="Best observed HIPE"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .105),
               ncol=3, frameon=False, fontsize=9)
    fig.text(.5, .071, "Primary endpoint: last regular validation. Positive reduction favors HIPE. These are validation-selected means.",
             ha="center", fontsize=8.5)
    fig.text(.5, .035, "C4 keeps the original different HIPE/RoPE micro-batches with global batch 64. SD is seed variation, not a significance test.",
             ha="center", fontsize=8.5)
    fig.subplots_adjust(left=.055, right=.985, top=.80, bottom=.25, wspace=.29)
    fig.savefig(out / "Targeted_sigma_layers.png", dpi=180)
    fig.savefig(out / "Targeted_sigma_layers.pdf")
    plt.close(fig)
    return count


def overview_figure(data, previous, out):
    current = [r for r in data["best_rows"] if r["endpoint"] == PRIMARY]
    prior = [r for r in previous["best_rows"] if r["endpoint"] == PRIMARY]
    hipe = current_hipe_rows(data)
    selected_targets = {target[:3] for target in TARGETS}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, rows, title, is_new in [
        (axes[0], prior, "Before joint sigma/layer search", False),
        (axes[1], current, "After joint sigma/layer search", True),
    ]:
        assert len(rows) == 12
        matrix = np.array([[next(r["gain_vs_rope_percent"] for r in rows
                                if setting(r) == (ds, size, length))
                            for length in LENGTHS] for ds, size in SETTINGS])
        image = ax.imshow(matrix, cmap="RdBu", vmin=-4.2, vmax=4.2, aspect="auto")
        ax.set_xticks(range(3), ["L=512", "L=1024", "L=2048"])
        ax.set_yticks(range(4), ["WikiText 20M", "WikiText 60M", "C4 20M", "C4 60M"])
        for i, (ds, size) in enumerate(SETTINGS):
            for j, length in enumerate(LENGTHS):
                row = next(r for r in rows if setting(r) == (ds, size, length))
                count = selected_layers(row, hipe) if is_new else 4
                color = "white" if abs(matrix[i, j]) > 2.5 else "#20262B"
                ax.text(j, i, f"{matrix[i,j]:+.2f}%\nSigma {row['selected_sigma']:g}, H={count}",
                        ha="center", va="center", color=color, fontsize=10,
                        fontweight="bold" if is_new and (ds, size, length) in selected_targets else "normal")
                if (ds, size, length) in selected_targets:
                    ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, fill=False,
                                           edgecolor="#26343D", lw=1.5))
        wins = int((matrix > 0).sum())
        ax.set_title(f"{title}\nLower mean than RoPE: {wins}/12 settings",
                     fontsize=12, pad=10, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_axes([.925, .235, .015, .54])
    bar = fig.colorbar(image, cax=cax)
    bar.set_label("PPL reduction versus RoPE (%)", fontsize=10)
    fig.suptitle("Best observed HIPE versus RoPE", fontsize=15, fontweight="bold", y=.98)
    fig.text(.5, .12, "Black outlines mark the three settings included in the new joint search. H is the number of HIPE layers.",
             ha="center", fontsize=9)
    fig.text(.5, .075, "Positive reduction favors HIPE. Means use three seeds at one fixed sigma/layer setting; configurations are never pooled.",
             ha="center", fontsize=9)
    fig.text(.5, .035, "Last regular validation. Best configurations are selected on validation data; small positive means do not establish significance.",
             ha="center", fontsize=8.5)
    fig.subplots_adjust(left=.10, right=.89, top=.80, bottom=.22, wspace=.34)
    fig.savefig(out / "HIPE_vs_RoPE_overview.png", dpi=180)
    fig.savefig(out / "HIPE_vs_RoPE_overview.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/table2_joint_20261104/workbook_data.json")
    parser.add_argument("--previous", type=Path, default=ROOT / "outputs/table2_sigma_final/workbook_data.json")
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    previous = json.loads(args.previous.read_text(encoding="utf-8"))
    out = args.input.parent
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelsize": 10,
    })
    count = targeted_figure(data, out)
    assert count == 18
    overview_figure(data, previous, out)
    files = ["Targeted_sigma_layers.png", "Targeted_sigma_layers.pdf",
             "HIPE_vs_RoPE_overview.png", "HIPE_vs_RoPE_overview.pdf"]
    metadata = {
        "primary_endpoint": PRIMARY,
        "target_configuration_groups_checked": count,
        "seed_mean_and_sample_sd_checked": True,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "previous_sha256": hashlib.sha256(args.previous.read_bytes()).hexdigest(),
        "files": {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in files},
    }
    (out / "FIGURE_VERIFICATION.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
