# Table 2 three-seed analysis

The campaigns are finished. This is a new three-seed replication under the recorded protocol, with qualified historical comparability. No new experiment or checkpoint evaluation was launched for this analysis.

## Main finding

HIPE has lower mean perplexity than RoPE in **8/12 settings**, and in **25/36 matched-seed comparisons**. It wins 3/4 settings at length 512, 4/4 at length 1024, and 1/4 at length 2048. The directions are unchanged when final-checkpoint evaluations replace the last regular validation.

These results support a conditional advantage, especially at length 1024. They do not reproduce the paper's uniform HIPE improvement over RoPE. A fixed-budget PPL comparison alone does not quantify tokens saved; a sample-efficiency claim additionally needs learning-curve/quality-threshold analysis.

## Coverage and adjudication

The input contains 150 execution attempts:108 originals and 42 recovery attempts. There are 132 completed unique main runs and 12 unavailable unique runs across 144 intended runs. Every numeric cell has all three seeds 6198,1024,7. Failed attempts remain in attempts.csv.

The three successful WikiText 20M/L1024 ALiBi retries replace failed originals based on completion, not performance. The three C4 retries failed again. The 216 canceled tuning tasks and the GPU preflight are excluded.

Four cells remain OOM: WikiText 20M/L2048, C4 20M/L1024, C4 20M/L2048, and C4 60M/L2048. This differs from the paper: WikiText 60M/L2048 ALiBi now runs, while C4 20M/L1024 does not. Equal numeric-cell counts therefore do not establish identical coverage.

## Endpoint and aggregation

The prespecified historical endpoint is the **last regular validation**, not the best validation and not a silently selected seed. For lengths512/1024/2048 it is step 3000/1500/700, after 98.304M/98.304M/91.7504M prediction tokens. Final checkpoints reach3051/1525/762 steps and 99.975168M/99.942400M/99.876864M tokens under the 100M cap.

Mean and sample standard deviation are over three independently trained runs. Each run's PPL is exp(total validation token loss / prediction-token count). We average per-run PPL for the paper-style summary; we never average batch PPLs.

## Full main table

| Corpus | Model | Method | L512 PPL | L1024 PPL | L2048 PPL |
| --- | --- | --- | ---: | ---: | ---: |
| WikiText-103 | 20M | Legacy NoPE+APE | 60.18 +/- 0.73 | 91.26 +/- 0.37 | 174.21 +/- 0.82 |
| WikiText-103 | 20M | ALIBI | 50.16 +/- 0.53 | 67.41 +/- 0.52 | OOM (0/3) |
| WikiText-103 | 20M | ROPE | 50.68 +/- 0.37 | 70.77 +/- 0.86 | 115.02 +/- 1.75 |
| WikiText-103 | 20M | HIPE | 49.28 +/- 0.17 | 68.55 +/- 0.38 | 114.52 +/- 0.88 |
| WikiText-103 | 60M | Legacy NoPE+APE | 46.55 +/- 1.88 | 71.17 +/- 0.77 | 140.37 +/- 6.56 |
| WikiText-103 | 60M | ALIBI | 35.76 +/- 0.31 | 42.27 +/- 0.45 | 67.32 +/- 1.04 |
| WikiText-103 | 60M | ROPE | 39.35 +/- 0.40 | 54.89 +/- 0.54 | 96.61 +/- 4.27 |
| WikiText-103 | 60M | HIPE | 39.16 +/- 0.41 | 53.09 +/- 0.34 | 97.22 +/- 2.79 |
| C4 | 20M | Legacy NoPE+APE | 153.47 +/- 0.58 | 222.02 +/- 1.62 | 375.33 +/- 5.81 |
| C4 | 20M | ALIBI | 122.94 +/- 0.11 | OOM (0/3) | OOM (0/3) |
| C4 | 20M | ROPE | 124.84 +/- 3.81 | 169.85 +/- 1.29 | 255.98 +/- 3.47 |
| C4 | 20M | HIPE | 119.77 +/- 1.26 | 165.87 +/- 2.59 | 256.69 +/- 7.84 |
| C4 | 60M | Legacy NoPE+APE | 123.35 +/- 7.47 | 203.08 +/- 3.22 | 351.96 +/- 16.03 |
| C4 | 60M | ALIBI | 87.51 +/- 0.55 | 104.23 +/- 0.62 | OOM (0/3) |
| C4 | 60M | ROPE | 97.39 +/- 1.72 | 143.73 +/- 5.23 | 241.55 +/- 12.91 |
| C4 | 60M | HIPE | 98.11 +/- 1.57 | 138.84 +/- 3.87 | 248.15 +/- 4.88 |

## HIPE versus RoPE

| Corpus | Model | Length | RoPE mean +/- SD | HIPE mean +/- SD | HIPE minus RoPE | Relative PPL reduction | Seeds favoring HIPE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wikitext103 | 20M | 512 | 50.68 +/- 0.37 | 49.28 +/- 0.17 | -1.39 | +2.75% | 3/3 |
| wikitext103 | 20M | 1024 | 70.77 +/- 0.86 | 68.55 +/- 0.38 | -2.22 | +3.14% | 3/3 |
| wikitext103 | 20M | 2048 | 115.02 +/- 1.75 | 114.52 +/- 0.88 | -0.50 | +0.43% | 2/3 |
| wikitext103 | 60M | 512 | 39.35 +/- 0.40 | 39.16 +/- 0.41 | -0.19 | +0.49% | 2/3 |
| wikitext103 | 60M | 1024 | 54.89 +/- 0.54 | 53.09 +/- 0.34 | -1.80 | +3.28% | 3/3 |
| wikitext103 | 60M | 2048 | 96.61 +/- 4.27 | 97.22 +/- 2.79 | +0.62 | -0.64% | 1/3 |
| c4 | 20M | 512 | 124.84 +/- 3.81 | 119.77 +/- 1.26 | -5.07 | +4.06% | 3/3 |
| c4 | 20M | 1024 | 169.85 +/- 1.29 | 165.87 +/- 2.59 | -3.98 | +2.34% | 3/3 |
| c4 | 20M | 2048 | 255.98 +/- 3.47 | 256.69 +/- 7.84 | +0.71 | -0.28% | 1/3 |
| c4 | 60M | 512 | 97.39 +/- 1.72 | 98.11 +/- 1.57 | +0.72 | -0.74% | 0/3 |
| c4 | 60M | 1024 | 143.73 +/- 5.23 | 138.84 +/- 3.87 | -4.90 | +3.41% | 3/3 |
| c4 | 60M | 2048 | 241.55 +/- 12.91 | 248.15 +/- 4.88 | +6.60 | -2.73% | 1/3 |

![Paired HIPE effects](../../outputs/table2_20261103/hipe_vs_rope.png)

The mean losing settings are WikiText 60M/L2048, C4 20M/L2048, C4 60M/L512 and C4 60M/L2048. C4 60M/L512 is worse for HIPE in each of the three seeds. Long-context effects are small or unfavorable relative to their seed variation.

Among methods with successful measurements, HIPE has the lowest mean in 4/12 settings. This ranking is conditional on the OOM coverage and the legacy NoPE label.

## Uncertainty

The effect plot shows paired seed differences and per-cell 95% t intervals with 2 degrees of freedom. These are exploratory parametric intervals, not a robust small-sample guarantee. The CSV includes exploratory paired t p-values and Holm adjustment across 12 comparisons.
No per-cell comparison survives a 5% Holm threshold (0/12). This limits confidence; it does not establish that every underlying effect is zero. The 8/12 and 25/36 win counts are descriptive and are not treated as independent-binomial significance tests.

## Learning curves

![Learning curves](../../outputs/table2_20261103/learning_curves.png)

The curves show all validation measurements, including the final evaluation, as mean +/- one sample SD. Their x-axis uses actual prediction-token exposure. End-of-training gains should not be read as a demonstrated universal gain across training budgets.

## Protocol audit and historical comparison

- All 12 paired cells have equal raw/processed train and validation identities, tokenizer/vocabulary, global batch 64, and token exposure. The padded vocabulary is50304. The two frozen campaigns have byte-identical trainer, preprocessing, model-config builder and decoder files checked in the input manifest.
- WikiText pairs also share micro-batch. C4 HIPE uses half the RoPE micro-batch, preserving the selected historical profiles. Accumulation/dropout RNG consumption and floating-point rounding can differ, so C4 effects are not a perfectly isolated positional intervention.
- The new HIPE runs use the paper's4/4 hierarchy (threshold 3). Five C4 log matches used threshold 4, implying5/3. Their sigmas were recovered from those old outcomes; they have not been re-tuned for4/4.
- Our NoPE runs explicitly include legacy learned absolute-position embeddings. Their row should not be presented as a position-free baseline.
- ALiBi settings are cell-specific: original successful cells use the first campaign's policy; the successful Wiki20M/L1024 retry restores automatic PyTorch SDPA and micro-batch 16. Remaining OOMs are computational outcomes, not poor numerical PPL values.
- Both WikiText RoPE and HIPE absolute PPLs are substantially below the paper values. This is not exact numerical reproduction. Historical source contains a first 20-validation-batch average, while the current trainer evaluates the entire validation population with token weighting. Exact executed historical code/data/validation records remain incomplete; this is a confirmed protocol difference, not a proven single cause of the gap.

The paper is a single reported value per Table 2 cell without the current three-seed uncertainty. table2_summary.csv preserves every paper target, rerun mean and difference. No lower rerun number alone proves a better reproduction.

## Recommended next checks

1. Freeze the full validation population, preprocessing and endpoint as the journal comparison protocol; explain the historical endpoint/population differences explicitly.
2. Isolate C4 effects with matched micro-batches/accumulation and a small4/4-versus5/3 schedule control, prioritizing the losing C4 60M settings before a larger campaign.
3. Audit the original validation protocol/checkpoints where possible. Evaluate existing checkpoints under recovered historical subsets as a separate analysis, without relabeling full-validation scores.
4. Use the existing learning curves to define prospective quality thresholds and targeted budget comparisons. Additional seeds or model scale should follow the protocol checks.

The legacy NoPE+APE baseline should retain its explicit label; a true position-free NoPE comparison remains a separate validation task.

No further runs were submitted in producing this report. Raw execution records and generated CSVs support every summary; input hashes and source commits are recorded in analysis_manifest.json.
