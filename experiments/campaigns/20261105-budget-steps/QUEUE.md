# Exploratory length/token-budget campaign

Published before GPU submission:
research/token-budget-steps-v1-20261105,
commit 12ebc719d980d8dd4fdf7a7bef6cf2ead40719b6.
Prior results remain pinned at research/table2-c4-results-v2-20261105.

| Job | Purpose | Tasks | Initial scheduler state |
| --- | --- | ---: | --- |
| 42168 | Two-step representative GPU checks | 8 | Pending priority |
| 42169 | Full longer-budget exploration | 264 | Pending afterok:42168 |

The full array starts only after every GPU check succeeds; an invalid check
dependency cancels it. One GPU is allocated per task and each phase has a
two-task concurrency cap, without overlap or DDP.

L512/100M is reused (63 seed-runs). New runs are L1024/200M and L2048/400M
with 3051 nominal optimizer steps. All observed L2048 HIPE variants and
H1/H2/H4/H6/H8 at paper L1024 sigmas are included, plus feasible Table 2
baselines. Four known ALiBi OOM configurations remain unavailable.
No L512 or supplementary-method job is submitted.

Train from original seeds with the full-budget cosine schedule, rather than
continuing an exhausted 100M schedule. Record actual tokens and keep every
length/budget/sigma/layer configuration separate in analysis.
The five model/trainer core files are byte-identical to the prior source.
Exact published source receipts, frozen config hashes, output paths and
scheduler records are in LAUNCH.json. Pending state is not completion.
