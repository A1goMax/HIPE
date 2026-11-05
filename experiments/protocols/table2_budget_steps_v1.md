# Length/token-budget exploration with matched nominal optimizer steps

The user requested L512/100M, L1024/200M and L2048/400M, and asked for many
configurations. The typed L0124 and L2028 are interpreted as L1024 and L2048.
This is a separate exploration. It does not replace the paper's 100M-token
results. The previous completed results are pinned at
research/table2-c4-results-v2-20261105 (ff82fac).

## Frozen grid and reuse

- L512/100M: reuse all 21 completed public-method configurations (63 seed-runs).
- L1024/200M: use 20 HIPE configurations: H1/H2/H4/H6/H8 at the original
  per-setting paper sigma for each of the two corpora and two model sizes.
  Add the 11 feasible archived baseline configurations.
- L2048/400M: use all 48 previously tested HIPE configurations, including the
  poorer variants, and nine feasible archived baseline configurations.

This gives 88 new configurations and 264 full runs across seeds 6198, 1024
and 7. The HIPE portion has 204 runs and the baseline portion 60.
No L512 training is submitted. Four ALiBi shape settings (12 seed-runs) are
recorded as unavailable under the original batch settings, rather than retried
with longer training or silently changed batches.

Seventy-two new configurations have exact completed 100M configuration
counterparts. Sixteen new L1024 layer variants have no such counterpart and
must not be described as isolated token-budget contrasts. Per-configuration
references and unavailable cases are recorded in the JSON protocol.

Eight total decoder layers are preserved. H1-H7 place HIPE in the final H
layers; H8 uses the supported uniform schedule. Per-setting sigma and layer
choices are frozen before training rather than expanded after seeing outcomes.
XPos remains outside public outputs and is not included in this campaign.

## Budget and optimizer contract

With global batch 64, floor(budget / (64 x length)) is 3051 steps at all
three length/budget combinations. The nominal final prediction-token counts
are 99,975,168; 199,950,336; and 399,900,672. Actual token counts are logged.

The inherited sampler can return partial micro-batches at corpus boundaries.
Actual prediction tokens must be checked rather than replaced by nominal counts,
especially when different methods retain different micro-batches.

All full runs retain original micro-batches, initial LR 6e-4, AdamW settings,
data/tokenizer/evaluation profiles, BF16, and the successful archived method
backend. The launcher now honors an explicitly supplied ALiBi backend while
retaining SDPA as its default. The existing C4 HIPE/RoPE micro-batch difference
is retained and qualified, not silently corrected.

Train fresh from the original seeds. The cosine scheduler uses T_max=3051
from the start. Continuing a completed 100M checkpoint would preserve a
different earlier LR trajectory and is not equivalent. The existing resume
source/config guards remain unchanged.

Matching optimizer steps is not matching tokens, FLOPs, wall time, or unique
data. Longer WikiText runs repeat a fixed corpus; they do not introduce new
training text. Evaluation populations stay fixed within each length; cross-
length PPL and tuning results must retain their length/budget labels.

## Execution and reporting

Commit and push source, suites and protocol before submission. Use the shared
launcher with --require-published, --array and --gpus 2. Eight representative
two-step GPU checks precede the full campaign; a successful-check dependency
prevents full training until every check passes. Invalid dependencies cancel
the full array. Checks are excluded from results.

Every task uses one allocated RTX 5090, with at most two tasks concurrently
and no DDP. Source snapshots, published revision, config hashes, data/tokenizer
identities, effective layer vectors, runtime and metric/checkpoint hashes
connect each result to its code.

Primary endpoint: last regular validation (step 3000). Final checkpoint
(step 3051) is separate. Nominal primary token counts are 98,304,000;
196,608,000; and 393,216,000. Never pool different budgets, sigmas or layer
counts in a seed mean/SD. Retain every seed, candidate, failure and tuning cost.
This is exploratory validation selection, not an independent test superiority
claim. Stop after this grid; no automatic outcome-driven expansion.
