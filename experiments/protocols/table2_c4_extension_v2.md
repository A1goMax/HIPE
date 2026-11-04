# C4 60M/L2048 wider sigma/layer search

This bounded exploratory study follows the results pinned at
research/table2-joint-results-v1-20261104 (52d794d). The previous best mean,
242.677564 at sigma50/H2, still trails the archived RoPE mean241.550843.

## Frozen search space

Sweep sigma1,5,10,25,50,100,200,500,2000 at one and two final HIPE layers.
At sigma50, complete one through eight final HIPE layers. Earlier layers
use unscaled RoPE. Total decoder depth remains eight.

The union has24 configurations. Four are already complete and are reused:
sigma25/H2 and sigma50/H2,H4,H6. This leaves20 new configurations, each with
seeds6198,1024,7: **60 full runs**. All previous out-of-grid results remain
available as observed candidates rather than being discarded.

H1-H7 use the existing bipartite schedule, threshold7-H. H8 uses the existing
uniform schedule, which applies HIPE in all eight layers. H0 is the archived
actual RoPE baseline. Sigma0 must not be used as a RoPE alias because the
inherited spectral density compensation remains even at very small sigma.

All full runs use the original micro-batch4 and global batch64 (16 accumulation
steps), the100M-token cap, BF16/FlashAttention, original optimizer/data/tokenizer
and evaluation populations. The complete-batch budget is762 steps and
99,876,864 prediction tokens.

## Checks and execution

Before any experiment, commit and push the public result projection and exact
new suite to A1goMax/HIPE. Freeze that published source for every task.
Two-step checks cover sigma1/H1, sigma2000/H1, sigma2000/H2 and sigma50/H8.
They are excluded from research results. Queue the full array with an afterok
dependency on all checks; cancel it if a check dependency fails.

Use the shared launcher:

    python -m hipe.experiments submit --suite table2_c4_extension_check --preflight --array --gpus 2 --require-published --execute
    python -m hipe.experiments submit --suite table2_c4_extension --array --gpus 2 --require-published --afterok CHECK_JOB_ID --execute

One GPU runs one independent process, with at most two tasks at once and no DDP.
Source publication receipts, frozen config checksums, actual layer vectors,
runtime/data fingerprints and checkpoint/metric hashes link each output to
its code revision.

## Analysis and publication

Select by complete three-seed mean at each fixed sigma/layer configuration.
Report individual seeds, mean and sample SD for every candidate, with
last regular validation as primary and final checkpoint separately.
Do not pool sigma/layer configurations or select the best individual seed.

These are additional validation-selected tuning outcomes. A mean win is not
an independent held-out or significance result. Stop after this grid rather
than expanding automatically to seek a desired winner.

Public Table2 summaries retain the paper's method set. XPos is a local
supplementary comparison. Current public reports are cleaned accordingly;
historical Git snapshots remain unchanged by the user's explicit choice.
