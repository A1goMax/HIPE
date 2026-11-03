# L2048 high-sigma comparison

Requested sigmas are 700, 800, 900 and 1000 for WikiText-103/C4 and 20M/60M, with seeds 6198,1024,7.

There are 48 requested seed-runs. Six matching sigma 700 runs already completed (Wiki60M andC420M); REUSE.json lists their output paths and model hashes. The suite submits only 42 missing runs.

All non-sigma training settings match the corresponding existing HIPE control: length 2048, 100M-token cap, global batch 64, micro-batches 16/8/8/4 for Wiki20M/Wiki60M/C420M/C460M, fixed bipartite 4/4 hierarchy (threshold 3), BF16 and FlashAttention. Execution uses one RTX 5090 per SLURM array task, at most two concurrently.

Best observed HIPE is selected once per dataset/model/length by the lowest mean PPL across all three seeds, with sampleSD for that same sigma. Sigmas and independent protocols are never pooled. The last regular validation is the primary endpoint for continuity; final-checkpoint results are shown separately. This is validation-based hyperparameter selection, not an independent held-out test result.

The current comparison is in outputs/best_sigma_20261103/. Current settings have only one completed sigma each and are labeled notoptimized. Once the new array finishes, a dependent CPU-only SLURM job collects the new metrics and produces an updated comparison against the existing baselines automatically. Incomplete candidates are not eligible for best-sigma selection.
