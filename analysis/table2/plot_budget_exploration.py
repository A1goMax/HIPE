"""Reproduce the two primary-endpoint exploratory token-budget figures.

Only complete three-seed groups are eligible. No experiment/training code is
changed. Input is the immutable workbook_data.json export; all plotted points
are fixed-configuration three-seed means (not pooled across configurations).
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', type=Path, default=Path(__file__).resolve().parents[2] / 'outputs' / 'budget_exploration_20261108' / 'workbook_data.json')
parser.add_argument('--output-dir', type=Path)
parser.add_argument('--only-scatter', action='store_true', help='Regenerate scatter and manifest while retaining the existing heatmap files')
args = parser.parse_args()
input_path = args.input.resolve()
out = (args.output_dir or input_path.parent).resolve()
out.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(out / '.mplconfig'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, ScalarFormatter
import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


input_sha = sha256(input_path)
data = json.loads(input_path.read_text(encoding='utf-8'))
primary = 'last_interval'
best = [r for r in data['best_rows'] if r['endpoint'] == primary]
changes = [r for r in data['change_rows']
           if r['endpoint'] == primary and r.get('control_group_id')
           and r['n'] == 3 and r['mean_ppl'] is not None
           and r['control_mean'] is not None]
complete_hipe = [r for r in data['target_rows']
                 if r['endpoint'] == primary and r['n'] == 3
                 and r['mean_ppl'] is not None]
all_primary = {(r['group_id'], r['endpoint']): r for r in data['all_rows']}
assert len(best) == 12 and all(r['n'] == 3 for r in best)
assert len(changes) == 71 and len({r['group_id'] for r in changes}) == 71
assert len(complete_hipe) == 75
assert sum(r['mean_win'] for r in complete_hipe) == 32
assert Counter(r['length'] for r in changes) == {1024: 15, 2048: 56}
assert all(r['gain_percent'] > 0 for r in changes)
assert sum(r['seed_wins_vs_rope'] for r in best) == 31
assert Counter(r['best_available_method'] for r in best) == {'hipe': 7, 'alibi': 5}
for row in best:
    selected = all_primary[(row['selected_group_id'], primary)]
    assert selected['n'] == 3 and selected['selection_eligible']
    assert abs(row['hipe_mean'] - selected['mean_ppl']) < 1e-10
    expected = 100 * (row['rope_mean'] - row['hipe_mean']) / row['rope_mean']
    assert abs(expected - row['gain_vs_rope_percent']) < 1e-10
for row in changes:
    control = all_primary[(row['control_group_id'], primary)]
    assert control['n'] == 3 and control['budget_tokens'] == 100_000_000
    assert abs(control['mean_ppl'] - row['control_mean']) < 1e-10
    assert row['budget_tokens'] == {1024: 200_000_000, 2048: 400_000_000}[row['length']]

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 13, 'axes.labelsize': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    'svg.fonttype': 'none', 'svg.hashsalt': 'hipe-budget-exploration-20261108',
    'axes.unicode_minus': True,
})
ink = '#172B3A'
muted = '#4F6170'
method_colors = {'hipe': '#0072B2', 'rope': '#D55E00', 'alibi': '#009E73', 'nope': '#A35B9A'}
method_markers = {'hipe': 'o', 'rope': 's', 'alibi': '^', 'nope': 'D'}
method_names = {'hipe': 'HIPE', 'rope': 'RoPE', 'alibi': 'ALiBi', 'nope': 'Legacy NoPE+APE'}
outputs: list[Path] = []


def save_figure(fig, stem):
    for ext in ('png', 'svg'):
        destination = out / f'{stem}.{ext}'
        if args.only_scatter and stem == 'fig_budget_selected_gain':
            assert destination.exists(), f'Existing heatmap required: {destination}'
        else:
            fig.savefig(destination, dpi=240, metadata={'Creator': 'HIPE budget exploration plotting script'})
        outputs.append(destination)
    plt.close(fig)


# Figure 1: all 12 dataset/model/length cells, with validation-selected HIPE.
row_keys = [('wikitext103', '20M'), ('wikitext103', '60M'), ('c4', '20M'), ('c4', '60M')]
lengths = [512, 1024, 2048]
lookup = {(r['dataset'], r['model_size'], r['length']): r for r in best}
values = np.array([[lookup[(ds, model, length)]['gain_vs_rope_percent']
                    for length in lengths] for ds, model in row_keys])
fig = plt.figure(figsize=(12.4, 9.1))
fig.text(.055, .959, 'Selected HIPE improves over RoPE in all 12 study cells',
         fontsize=18, weight='bold', color=ink, va='top')
fig.text(.055, .916, 'Exploratory matched-step study  |  Primary endpoint: last regular validation (step 3000)',
         fontsize=11.5, color=muted, va='top')
ax = fig.add_axes([.215, .303, .68, .52])
im = ax.imshow(values, cmap='Blues', norm=Normalize(vmin=0, vmax=5), aspect='auto')
ax.set_xticks(range(3), ['L = 512\n100M tokens · reused', 'L = 1024\n200M tokens', 'L = 2048\n400M tokens'])
ax.xaxis.tick_top()
ax.tick_params(axis='x', length=0, pad=14, labelsize=12)
ax.set_yticks(range(4), ['WikiText-103\n20M model', 'WikiText-103\n60M model', 'C4\n20M model', 'C4\n60M model'])
ax.tick_params(axis='y', length=0, pad=16, labelsize=12)
ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
ax.grid(which='minor', color='white', linewidth=3)
ax.tick_params(which='minor', bottom=False, left=False)
for spine in ax.spines.values():
    spine.set_visible(False)
for i, (ds, model) in enumerate(row_keys):
    for j, length in enumerate(lengths):
        r = lookup[(ds, model, length)]
        foreground = 'white' if values[i, j] >= 2.5 else ink
        n_candidates = r['complete_candidate_count']
        ax.text(j, i-.27, f"+{values[i, j]:.2f}%", ha='center', va='center',
                fontsize=18, weight='bold', color=foreground)
        ax.text(j, i-.015, f"$\\sigma$ = {r['selected_sigma']}; H = {r['hipe_layers']}",
                ha='center', va='center', fontsize=11, color=foreground)
        ax.text(j, i+.19, f"{n_candidates} complete candidate" + ('s' if n_candidates != 1 else ''),
                ha='center', va='center', fontsize=9.4, color=foreground)
        unavailable = r['alibi_mean'] is None
        lowest = method_names[r['best_available_method']]
        ax.text(j, i+.38, f"Lowest available: {lowest}" + ('*' if unavailable else ''),
                ha='center', va='center', fontsize=9.2, color=foreground)
cax = fig.add_axes([.923, .303, .017, .52])
cbar = fig.colorbar(im, cax=cax, ticks=np.arange(0, 6))
cbar.ax.set_ylabel('PPL reduction relative to RoPE (%)', fontsize=10.5, labelpad=12)
cbar.outline.set_visible(False)
fig.text(.055, .235, '12 / 12 cell means beat RoPE  ·  31 / 36 paired seed comparisons favor selected HIPE',
         color=ink, fontsize=12, weight='bold')
fig.text(.055, .199, 'Lowest available method mean: HIPE in 7 cells; ALiBi in 5. Across all 75 complete HIPE configurations,\n32 beat RoPE; the heatmap reports validation-selected configurations.',
         color=muted, fontsize=10.5, linespacing=1.5, va='top')
fig.text(.055, .132, 'Selection uses the lowest three-seed mean within each study cell. H = number of HIPE layers; candidate counts vary.\n* ALiBi unavailable in four cells. Two incomplete HIPE groups (n = 2) are excluded from selection and summaries.',
         color=muted, fontsize=9.7, linespacing=1.5, va='top')
fig.text(.055, .068, 'This exploration is separate from the paper\'s 100M-token results. Budgets differ across lengths; matching optimizer steps\ndoes not match tokens, compute, or unique data. Validation selection and seed variation do not establish significance.',
         color=muted, fontsize=9.7, linespacing=1.5, va='top')
save_figure(fig, 'fig_budget_selected_gain')

# Figure 2: exact configuration matches with original 100M controls.
fig, axes = plt.subplots(2, 2, figsize=(11.6, 10.8), sharex=True, sharey=True)
fig.subplots_adjust(left=.09, right=.975, top=.822, bottom=.235, wspace=.16, hspace=.25)
fig.text(.055, .962, 'Longer budgets improve every complete matched configuration',
         fontsize=17.5, weight='bold', color=ink, va='top')
fig.text(.055, .925, '71 exact configuration matches  |  Primary endpoint: last regular validation  |  Lower PPL is better',
         fontsize=11, color=muted, va='top')
legend_handles = [Line2D([0], [0], marker=method_markers[m], color='none', markerfacecolor=method_colors[m],
                         markeredgecolor='white', markersize=8, label=method_names[m])
                  for m in ('hipe', 'rope', 'alibi', 'nope')]
fig.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(.055, .897),
           ncol=4, frameon=False, fontsize=10.5, columnspacing=2, handletextpad=.5)
limits = (20, 420)
ticks = [20, 30, 50, 100, 200, 400]
panel_details = []
for i, length in enumerate([1024, 2048]):
    for j, dataset in enumerate(['wikitext103', 'c4']):
        ax = axes[i, j]
        records = [r for r in changes if r['length'] == length and r['dataset'] == dataset]
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.set_aspect('equal', adjustable='box')
        ax.plot(limits, limits, color='#75818A', lw=1.1, ls=(0, (5, 4)), zorder=1)
        ax.grid(which='major', color='#E7EBEF', linewidth=.7, zorder=0)
        ax.grid(which='minor', visible=False)
        for m in ('hipe', 'rope', 'alibi', 'nope'):
            pts = [r for r in records if r['method'] == m]
            if not pts:
                continue
            ax.errorbar([r['control_mean'] for r in pts], [r['mean_ppl'] for r in pts],
                        xerr=[r['control_sd'] for r in pts], yerr=[r['sample_std_ppl'] for r in pts],
                        fmt='none', ecolor=method_colors[m], elinewidth=.9, capsize=1.8,
                        alpha=.6, zorder=2)
            ax.scatter([r['control_mean'] for r in pts], [r['mean_ppl'] for r in pts],
                       c=method_colors[m], marker=method_markers[m], s=40 if m == 'hipe' else 59,
                       edgecolors='white', linewidths=.5, alpha=.8 if m == 'hipe' else .95, zorder=3)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(FixedLocator(ticks))
            axis.set_major_formatter(ScalarFormatter())
            axis.set_minor_locator(matplotlib.ticker.NullLocator())
        label = 'WikiText-103' if dataset == 'wikitext103' else 'C4'
        ax.set_title(f"{label}  |  L = {length}, {length // 512 * 100}M tokens", color=ink, pad=10, fontsize=11.5)
        counts = Counter(r['method'] for r in records)
        count_lines = [f"{len(records)} configurations", ' · '.join(f"{method_names[m]} {counts[m]}" for m in ('hipe', 'rope') if counts[m]),
                       ' · '.join(f"{method_names[m]} {counts[m]}" for m in ('alibi', 'nope') if counts[m])]
        ax.text(.035, .96, '\n'.join(count_lines), transform=ax.transAxes, va='top',
                fontsize=8.8, color=muted, linespacing=1.5)
        if j == 0:
            ax.set_ylabel('New-budget PPL')
        if i == 1:
            ax.set_xlabel('Original 100M-token PPL')
        ax.text(.88, .895, 'y = x', transform=ax.transAxes, fontsize=9, color='#75818A', rotation=43)
        panel_details.append({'dataset': dataset, 'length': length, 'n': len(records), 'method_counts': dict(counts)})
fig.text(.055, .175, 'Each point is one exact fixed-configuration match; means are not pooled across configurations.\nError bars show ±1 sample SD across the three original seeds (some are smaller than the marker). Both axes are logarithmic.',
         color=muted, fontsize=10.1, linespacing=1.5, va='top')
fig.text(.055, .112, '71 of 88 new configurations are included. Sixteen lack exact 100M controls; one additional matched configuration is incomplete.\nALiBi availability is limited by original OOM cells; unavailable results are not imputed.',
         color=muted, fontsize=10.1, linespacing=1.5, va='top')
fig.text(.055, .048, 'Separate exploratory comparison: larger token budgets change optimizer steps and data exposure versus the 100M controls.\nThese observations describe budget effects; they do not establish statistical significance.',
         color=muted, fontsize=9.8, linespacing=1.5, va='top')
save_figure(fig, 'fig_budget_vs_100M')

assert sha256(input_path) == input_sha, 'Input changed while figures were generated'
verification = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'input': {'path': str(input_path), 'sha256': input_sha},
    'source': {'path': str(Path(__file__).resolve()), 'sha256': sha256(Path(__file__).resolve())},
    'runtime': {'python': os.sys.executable, 'matplotlib': matplotlib.__version__},
    'endpoint': primary,
    'selection': {
        'complete_seed_count': 3, 'selected_cells': 12, 'selected_mean_wins_vs_rope': 12,
        'selected_paired_seed_wins_vs_rope': 31, 'selected_paired_seed_comparisons': 36,
        'lowest_available_method_counts': dict(Counter(r['best_available_method'] for r in best)),
        'complete_HIPE_configurations': len(complete_hipe), 'HIPE_configurations_beating_rope': 32,
        'selected_group_ids': [r['selected_group_id'] for r in best],
    },
    'matched_controls': {
        'complete_configuration_pairs': len(changes), 'all_improve': all(r['gain_percent'] > 0 for r in changes),
        'group_ids': [r['group_id'] for r in changes], 'panels': panel_details,
        'exclusions': {'total_new_configurations': 88, 'unique_excluded_configurations': 17,
                       'no_exact_100M_control_layouts': 16, 'incomplete_groups_total': 2,
                       'incomplete_groups_without_exact_control': 1, 'additional_incomplete_matched_configurations': 1},
        'uncertainty': 'Mean and sample SD of three original seeds, separately per fixed configuration; no pooling or significance claim',
    },
    'automated_checks': {'input_unchanged': True, 'complete_groups_only': True, 'all_selected_data_match_all_rows': True,
                         'exact_control_means_verified': True, 'gain_formula_verified': True},
    'visual_review': {'status': 'pending', 'files': [str(p) for p in outputs if p.suffix == '.png']},
    'outputs': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in outputs],
}
verification_path = out / 'FIGURE_VERIFICATION.json'
verification_path.write_text(json.dumps(verification, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'figures': [str(p) for p in outputs], 'verification': str(verification_path)}, indent=2))
