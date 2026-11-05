# Completed C4 60M/L2048 extension

All 60 new full runs completed successfully from published commit c44c207b83bf4140ec462bd727b1c5e6fc946613. Four two-step GPU checks completed separately and are excluded. Twelve earlier seed-runs complete the planned 24-configuration grid; six earlier out-of-grid configurations are also retained.

## Main result

The best observed HIPE result is sigma 500 with 2 final HIPE layers: **242.321885 +/- 7.337535** PPL, versus RoPE **241.550843 +/- 12.911822**. This is 0.771042 PPL (0.319205%) higher than RoPE.

The previous best HIPE mean was 242.677564 at sigma 50/H2. The extension improves that mean by 0.355679 PPL, but no new configuration surpasses RoPE's mean. The overall best-observed HIPE count remains **11/12 settings**.

## Ranked C4 configurations

Primary endpoint: last regular validation. Each mean and sample SD uses all three seeds at the stated fixed configuration.

| Rank | Sigma | HIPE layers | Origin | HIPE mean +/- SD | Difference from RoPE | PPL reduction vs RoPE | Paired seed wins |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 500 | 2 | New | 242.321885 +/- 7.337535 | +0.771042 | -0.319205% | 2/3 |
| 2 | 100 | 1 | New | 242.537140 +/- 8.962635 | +0.986297 | -0.408318% | 1/3 |
| 3 | 200 | 1 | New | 242.672941 +/- 7.837069 | +1.122098 | -0.464539% | 1/3 |
| 4 | 50 | 2 | Reused grid control | 242.677564 +/- 6.525953 | +1.126721 | -0.466453% | 2/3 |
| 5 | 200 | 2 | New | 242.794494 +/- 8.890603 | +1.243651 | -0.514861% | 1/3 |
| 6 | 100 | 2 | New | 242.834852 +/- 8.887215 | +1.284009 | -0.531569% | 1/3 |
| 7 | 500 | 1 | New | 242.953827 +/- 7.990987 | +1.402984 | -0.580823% | 1/3 |
| 8 | 2000 | 1 | New | 243.641252 +/- 8.199860 | +2.090408 | -0.865411% | 1/3 |
| 9 | 50 | 1 | New | 243.761666 +/- 7.846323 | +2.210823 | -0.915262% | 1/3 |
| 10 | 10 | 1 | New | 244.751514 +/- 6.527956 | +3.200671 | -1.325051% | 1/3 |
| 11 | 25 | 2 | Reused grid control | 245.178481 +/- 6.522571 | +3.627638 | -1.501811% | 1/3 |
| 12 | 25 | 1 | New | 245.539530 +/- 9.040858 | +3.988687 | -1.651283% | 0/3 |
| 13 | 1 | 1 | New | 245.635966 +/- 6.885842 | +4.085123 | -1.691206% | 1/3 |
| 14 | 50 | 3 | New | 245.716318 +/- 6.264518 | +4.165475 | -1.724471% | 1/3 |
| 15 | 2000 | 2 | New | 245.959751 +/- 9.526642 | +4.408908 | -1.825250% | 0/3 |
| 16 | 5 | 1 | New | 246.073835 +/- 7.195803 | +4.522992 | -1.872480% | 1/3 |
| 17 | 10 | 2 | New | 246.607247 +/- 6.766317 | +5.056403 | -2.093308% | 1/3 |
| 18 | 1 | 2 | New | 246.950706 +/- 8.490749 | +5.399863 | -2.235498% | 0/3 |
| 19 | 5 | 2 | New | 247.754599 +/- 5.679583 | +6.203756 | -2.568302% | 0/3 |
| 20 | 50 | 4 | Reused grid control | 248.151969 +/- 4.876379 | +6.601126 | -2.732810% | 1/3 |
| 21 | 25 | 4 | Earlier result | 248.367963 +/- 8.431906 | +6.817119 | -2.822230% | 0/3 |
| 22 | 1000 | 4 | Earlier result | 250.854945 +/- 9.398412 | +9.304102 | -3.851819% | 0/3 |
| 23 | 800 | 4 | Earlier result | 251.067885 +/- 9.657971 | +9.517042 | -3.939975% | 0/3 |
| 24 | 700 | 4 | Earlier result | 251.069184 +/- 8.718976 | +9.518340 | -3.940512% | 0/3 |
| 25 | 900 | 4 | Earlier result | 252.191067 +/- 10.458416 | +10.640224 | -4.404962% | 0/3 |
| 26 | 50 | 5 | New | 253.914791 +/- 6.705537 | +12.363947 | -5.118569% | 0/3 |
| 27 | 25 | 6 | Earlier result | 257.056300 +/- 7.251073 | +15.505456 | -6.419127% | 0/3 |
| 28 | 50 | 6 | Reused grid control | 260.123313 +/- 10.564671 | +18.572470 | -7.688845% | 0/3 |
| 29 | 50 | 7 | New | 265.713957 +/- 11.519163 | +24.163114 | -10.003324% | 0/3 |
| 30 | 50 | 8 | New | 285.486183 +/- 19.961167 | +43.935340 | -18.188858% | 0/3 |

Positive reduction would favor HIPE. All observed primary means are higher than RoPE. The selected sigma 500/H2 result improves in two of three matched seeds, which does not change its higher mean.

## Endpoint check

| Endpoint | Sigma | HIPE layers | HIPE mean +/- SD | RoPE mean +/- SD | Difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| last_interval | 500 | 2 | 242.321885 +/- 7.337535 | 241.550843 +/- 12.911822 | +0.771042 |
| final | 500 | 2 | 241.728614 +/- 7.378239 | 240.920115 +/- 12.941783 | +0.808499 |

## Interpretation

At sigma 50, two HIPE layers are best among H1-H8. Performance worsens as the count increases beyond two; the all-HIPE H8 result is 285.486183 +/- 19.961167 PPL. One HIPE layer is not uniformly better than two. Sigma 500/H2 gives the best observed mean over the full C4 search, while both very small sigma values and sigma 2000 underperform it.

The wider search narrows the earlier mean gap but does not establish a C4 60M/L2048 advantage over RoPE. No further experiments were launched during this analysis.

## Updated paper-scope Table 2

| Corpus | Model | L | Sigma | HIPE layers | Legacy NoPE+APE | ALiBi | RoPE | HIPE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wikitext103 | 20M | 512 | 200 | 4 | 60.181882 +/- 0.729248 | 50.159828 +/- 0.534017 | 50.677000 +/- 0.374751 | 49.284134 +/- 0.165303 |
| wikitext103 | 20M | 1024 | 500 | 4 | 91.263017 +/- 0.370956 | 67.412724 +/- 0.522868 | 70.773016 +/- 0.859961 | 68.548913 +/- 0.376462 |
| wikitext103 | 20M | 2048 | 800 | 4 | 174.205354 +/- 0.818574 | OOM | 115.017409 +/- 1.750982 | 114.070849 +/- 0.526692 |
| wikitext103 | 60M | 512 | 300 | 4 | 46.552683 +/- 1.882486 | 35.756719 +/- 0.312725 | 39.345951 +/- 0.398817 | 39.155005 +/- 0.406353 |
| wikitext103 | 60M | 1024 | 300 | 4 | 71.172309 +/- 0.767444 | 42.266619 +/- 0.446609 | 54.886376 +/- 0.537711 | 53.088699 +/- 0.338573 |
| wikitext103 | 60M | 2048 | 500 | 2 | 140.367138 +/- 6.557607 | 67.322903 +/- 1.039623 | 96.606311 +/- 4.274491 | 96.383156 +/- 3.383820 |
| c4 | 20M | 512 | 300 | 4 | 153.470813 +/- 0.578573 | 122.942177 +/- 0.113978 | 124.842452 +/- 3.809679 | 119.773068 +/- 1.263198 |
| c4 | 20M | 1024 | 300 | 4 | 222.023301 +/- 1.618054 | OOM | 169.852084 +/- 1.294024 | 165.873837 +/- 2.589700 |
| c4 | 20M | 2048 | 800 | 4 | 375.331626 +/- 5.807935 | OOM | 255.977976 +/- 3.474659 | 254.465588 +/- 5.180511 |
| c4 | 60M | 512 | 200 | 2 | 123.350728 +/- 7.472727 | 87.509025 +/- 0.547090 | 97.392336 +/- 1.723046 | 97.153268 +/- 0.897829 |
| c4 | 60M | 1024 | 300 | 4 | 203.078424 +/- 3.224998 | 104.226327 +/- 0.616924 | 143.734881 +/- 5.232943 | 138.836694 +/- 3.874966 |
| c4 | 60M | 2048 | 500 | 2 | 351.955488 +/- 16.027390 | OOM | 241.550843 +/- 12.911822 | 242.321885 +/- 7.337535 |

Among the paper's Table 2 methods, HIPE has the lowest mean in five settings, ALiBi in six and RoPE in one.

## Protocol and uncertainty

- Mean and sample SD use seeds 6198, 1024 and 7 at one fixed sigma/layer configuration. Configurations are never pooled.
- Primary endpoint is last regular validation. Final-checkpoint results are separate. Both select sigma 500 with two HIPE layers for C4 60M/L2048.
- None of the 20 new configurations has a lower mean PPL than RoPE. The best observed Table 2 count remains 11/12 settings.
- Eight total decoder layers are retained. H1-H7 place HIPE in the final H layers; H8 uses the uniform all-HIPE schedule.
- C4 HIPE retains micro-batch 4 versus RoPE micro-batch 8, with global batch 64 unchanged. Cross-method comparisons retain this qualification.
- The new training core files exactly match the previous joint-search core. Older cohorts remain distinct; comparisons retain archived unchanged-method evidence.
- Public Table 2 methods are HIPE, RoPE, ALiBi and Legacy NoPE+APE. Supplementary comparisons remain local.
- Four ALiBi settings remain unavailable due to historical OOM; unavailable PPL is never zero.
- All 60 new full runs completed. Twelve earlier seed-runs complete the new grid; four two-step GPU checks are excluded.
- These are validation-selected tuning outcomes. Three-seed sample SD does not include selection uncertainty or establish significance.
