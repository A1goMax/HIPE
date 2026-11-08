# HIPE Research Status

Updated 8 November 2026, Asia/Shanghai.

## Latest matched-step exploration

All 264 submitted tasks have ended: 262 completed and two ended NODE_FAIL.
All eight GPU checks completed. The study combines those 262 completed new
runs with 63 reused L512/100M runs. No recovery or additional training was
submitted during this analysis.

Best complete three-seed HIPE has lower mean PPL than RoPE in all 12 study
settings: 4/4 per length and 31/36 paired seeds. Budgets are L512/100M,
L1024/200M and L2048/400M, each with 3051 nominal optimizer steps.
The original all-100M experiment remains 11/12 and is kept separate.
Only 32/75 complete study HIPE configurations beat RoPE. Lowest available
public-method means belong to HIPE in seven settings and ALiBi in five.

C4 60M/L2048 at 400M selects sigma 200/H2:
66.056620 +/- 1.137437 versus RoPE 66.298782 +/- 1.491717 (0.365259% lower).
The final checkpoint selects the same configuration and also favors HIPE.

Read [the formal report](analysis/table2/BUDGET_EXPLORATION_REPORT.md).
Workbook, exact seeds, all candidates, budget contrasts, figures and audits:
outputs/budget_exploration_20261108/.
Versioned public archive: experiments/results/budget_exploration_20261108/.

## Coverage and missing seeds

42169_110: C4 20M L2048/400M sigma 700/H4, seed 7, has a step 1000 checkpoint
and partial validation. 42169_111: C4 60M L1024/200M sigma 300/H1, seed 6198,
has no checkpoint or metrics. Both ended NODE_FAIL on cccl2; underlying cause
is not established. Their available n=2 seeds stay visible, with null aggregate
mean/SD and exclusion from selection. Original failed outputs remain intact.

107/109 study configurations have all three seeds. Of 72 exact 100M control
pairs, 71 are complete and all 71 improve at both endpoints. Sixteen new
L1024 layouts lack exact 100M controls. All 181 fixed protocols and 1082
endpoint observations are retained; historical controls never enter new-budget
selection.

## Interpretation and provenance

Budget, sigma, layer count and endpoints are never pooled in seed means.
All available methods within each study setting match actual training and
validation exposures. Partial WikiText micro-batches explain small token
shortfalls. Equal optimizer steps do not mean equal tokens, compute or unique
data. Longer WikiText budgets repeat a fixed corpus.

These are validation-selected outcomes. Three-seed SD does not account for
selection uncertainty or establish significance. Original method batches and
Legacy NoPE+APE behavior remain explicit qualifications.

Training source was published before GPU submission:
12ebc719d980d8dd4fdf7a7bef6cf2ead40719b6,
tag research/token-budget-steps-v1-20261105.
All 264 configs, run configs and source archives were verified, including
1320 core source entries. Current result tag:
research/budget-results-v1-20261108.
Current public methods follow Table 2; supplementary comparisons stay local
and historical snapshots remain intact.

## Next priorities

Recover the two interrupted seeds with recorded provenance before calling
the planned grid complete. Use independent evaluation before journal claims.
Keep this exploration separate from the original 100M reproduction results.
Table 1 retrieval, Table 3 larger-model specifications and Table 4 LoRA remain
pending.
