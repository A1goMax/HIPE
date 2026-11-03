# Completed Table 2 sigma-sweep analysis

All 42 new runs completed. Together with 6 reused runs, the requested L2048 grid contains 48 unique seed-runs: 4 dataset/model settings x 4 sigmas x 3 seeds.

## Main result

The best observed L2048 sigma is 800 for WikiText 20M, WikiText 60M and C4 20M. For C4 60M, the previously tested sigma 50 remains best among all tested values. Within the requested 700-1000 range, sigma 1000 is best for C4 60M.

Best observed HIPE now has lower mean PPL than RoPE in 9/12 settings (previously 8/12). The only direction change is C4 20M/L2048, which becomes a small mean improvement. L2048 HIPE improves in 2/4 settings. WikiText 60M and C4 60M still trail RoPE. Final-checkpoint results preserve these directions.

## Best observed HIPE against baselines

The primary view is last regular validation. Values are mean +/- sample SD over seeds 6198, 1024 and 7, at one fixed sigma. Selecting the best individual seed or averaging different sigmas is prohibited.

| Corpus | Model | Length | Best sigma | Candidates | Legacy NoPE+APE | ALiBi | RoPE | Best observed HIPE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wikitext103 | 20M | 512 | 200 | 1 | 60.1819 +/- 0.7292 | 50.1598 +/- 0.5340 | 50.6770 +/- 0.3748 | 49.2841 +/- 0.1653 |
| wikitext103 | 20M | 1024 | 500 | 1 | 91.2630 +/- 0.3710 | 67.4127 +/- 0.5229 | 70.7730 +/- 0.8600 | 68.5489 +/- 0.3765 |
| wikitext103 | 20M | 2048 | 800 | 5 | 174.2054 +/- 0.8186 | OOM | 115.0174 +/- 1.7510 | 114.0708 +/- 0.5267 |
| wikitext103 | 60M | 512 | 300 | 1 | 46.5527 +/- 1.8825 | 35.7567 +/- 0.3127 | 39.3460 +/- 0.3988 | 39.1550 +/- 0.4064 |
| wikitext103 | 60M | 1024 | 300 | 1 | 71.1723 +/- 0.7674 | 42.2666 +/- 0.4466 | 54.8864 +/- 0.5377 | 53.0887 +/- 0.3386 |
| wikitext103 | 60M | 2048 | 800 | 4 | 140.3671 +/- 6.5576 | 67.3229 +/- 1.0396 | 96.6063 +/- 4.2745 | 96.8108 +/- 2.5999 |
| c4 | 20M | 512 | 300 | 1 | 153.4708 +/- 0.5786 | 122.9422 +/- 0.1140 | 124.8425 +/- 3.8097 | 119.7731 +/- 1.2632 |
| c4 | 20M | 1024 | 300 | 1 | 222.0233 +/- 1.6181 | OOM | 169.8521 +/- 1.2940 | 165.8738 +/- 2.5897 |
| c4 | 20M | 2048 | 800 | 4 | 375.3316 +/- 5.8079 | OOM | 255.9780 +/- 3.4747 | 254.4656 +/- 5.1805 |
| c4 | 60M | 512 | 500 | 1 | 123.3507 +/- 7.4727 | 87.5090 +/- 0.5471 | 97.3923 +/- 1.7230 | 98.1110 +/- 1.5719 |
| c4 | 60M | 1024 | 300 | 1 | 203.0784 +/- 3.2250 | 104.2263 +/- 0.6169 | 143.7349 +/- 5.2329 | 138.8367 +/- 3.8750 |
| c4 | 60M | 2048 | 50 | 5 | 351.9555 +/- 16.0274 | OOM | 241.5508 +/- 12.9118 | 248.1520 +/- 4.8764 |

## L2048 per-sigma results

| Corpus | Model | Sigma | Mean PPL | Sample SD | n | Selected overall |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| c4 | 20M | 700 | 256.689193 | 7.843872 | 3 |  |
| c4 | 20M | 800 | 254.465588 | 5.180511 | 3 | yes |
| c4 | 20M | 900 | 255.326631 | 6.272043 | 3 |  |
| c4 | 20M | 1000 | 254.975644 | 6.964979 | 3 |  |
| c4 | 60M | 50 | 248.151969 | 4.876379 | 3 | yes |
| c4 | 60M | 700 | 251.069184 | 8.718976 | 3 |  |
| c4 | 60M | 800 | 251.067885 | 9.657971 | 3 |  |
| c4 | 60M | 900 | 252.191067 | 10.458416 | 3 |  |
| c4 | 60M | 1000 | 250.854945 | 9.398412 | 3 |  |
| wikitext103 | 20M | 500 | 114.518520 | 0.884637 | 3 |  |
| wikitext103 | 20M | 700 | 114.462520 | 0.929590 | 3 |  |
| wikitext103 | 20M | 800 | 114.070849 | 0.526692 | 3 | yes |
| wikitext103 | 20M | 900 | 114.362196 | 1.386380 | 3 |  |
| wikitext103 | 20M | 1000 | 114.522339 | 0.500453 | 3 |  |
| wikitext103 | 60M | 700 | 97.223123 | 2.792966 | 3 |  |
| wikitext103 | 60M | 800 | 96.810837 | 2.599915 | 3 | yes |
| wikitext103 | 60M | 900 | 98.755382 | 4.668157 | 3 |  |
| wikitext103 | 60M | 1000 | 98.227021 | 3.739034 | 3 |  |

## Interpretation

1. Sigma 800 has the lowest mean among tested high-range values in three settings, but the differences are small relative to seed variation. Larger sigma is not monotonically better.
2. C4 60M is an exception: all newly tested high sigmas underperform the old sigma 50 mean, and their across-seed SD is larger. Values between 50 and 700 were not explored here. Sigma 50 is best observed among tested values.
3. Compared with the old fixed choice, WikiText 20M improves from 114.5185 to 114.0708, WikiText 60M from 97.2231 to 96.8108, and C4 20M from 256.6892 to 254.4656. C4 60M's selected result is unchanged.
4. The best mean is calculated over all three seeds. A low single-seed value does not make its sigma the winner.
5. Among available methods, HIPE has the lowest mean in 5 settings, ALiBi in 6 and RoPE in 1. Four ALiBi settings remain unavailable due to OOM. The legacy NoPE+APE baseline retains learned absolute position embeddings.

## Scope and uncertainty

- Best observed HIPE is selected by the lowest complete three-seed mean at each fixed dataset/model/length/protocol. Sample SD is computed within that selected sigma; sigmas are never pooled.
- Primary endpoint: last regular validation, matching the earlier analysis. Final-checkpoint results are a separate view. Selected sigmas agree across the two views in this campaign.
- Selection uses validation results from these seeds, so best-observed values are tuning outcomes rather than an independent held-out test estimate.
- L2048 includes all requested sigma 700/800/900/1000 values. WikiText 20M also includes sigma 500; C4 60M also includes sigma 50. L512/L1024 each have only one completed sigma.
- Baseline NoPE is explicitly Legacy NoPE+APE. OOM is unavailable PPL, never zero.
- C4 HIPE and RoPE retain different micro-batches with global batch 64 unchanged. Five historical C4 paper-log matches used a 5/3 layer split; this campaign uses the current paper 4/4 split.
- Last regular validation: steps 3000/1500/700 for L512/1024/2048; final checkpoints 3051/1525/762. Complete-batch token rounding stays below the 100M cap.

These are validation-selected results. The three-seed SD describes variation conditional on a chosen sigma and does not account for hyperparameter-selection uncertainty. No post-selection significance claim is made.

C4 comparisons versus RoPE retain a micro-batch difference. The within-HIPE sigma comparison holds this constant. Absolute reproduction of paper values remains qualified by historical validation/population and layer-split differences.

The formal workbook contains both endpoint views, every sigma separately, exact per-seed PPLs, baseline comparisons, failure records and coverage. No additional training was launched for this analysis.
