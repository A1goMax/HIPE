# Completed HIPE sigma/layer search and XPos comparison

All 81 new full runs completed (45 HIPE and 36 XPos). Nine previous HIPE seed-runs complete the targeted 18-configuration grid. The 12 two-step GPU checks are excluded.

## Main findings

Best observed HIPE has lower three-seed mean PPL than RoPE in 11/12 settings, up from 9/12. Two previous mean losses become wins. Three new fixed configurations have lower mean PPL than RoPE; two belong to the same C4 L512 setting.

## New HIPE configurations with lower mean PPL than RoPE

Primary endpoint: last regular validation. Sample SD is across seeds 6198, 1024 and 7 at exactly the stated sigma and layer count.

| Corpus | Model | Length | Sigma | HIPE layers | HIPE PPL mean +/- SD | RoPE PPL mean +/- SD | Reduction | Paired seeds favor HIPE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wikitext103 | 60M | 2048 | 500 | 2 | 96.383156 +/- 3.383820 | 96.606311 +/- 4.274491 | 0.230994% | 2/3 |
| c4 | 60M | 512 | 200 | 2 | 97.153268 +/- 0.897829 | 97.392336 +/- 1.723046 | 0.245470% | 1/3 |
| c4 | 60M | 512 | 500 | 2 | 97.293800 +/- 1.045408 | 97.392336 +/- 1.723046 | 0.101175% | 1/3 |

## Previously losing settings: selected results

| Corpus | Model | Length | Selected sigma | HIPE layers | Previous HIPE PPL | New HIPE PPL | RoPE PPL | Reduction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wikitext103 | 60M | 2048 | 500 | 2 | 96.810837 +/- 2.599915 | 96.383156 +/- 3.383820 | 96.606311 +/- 4.274491 | 0.230994% |
| c4 | 60M | 512 | 200 | 2 | 98.111005 +/- 1.571917 | 97.153268 +/- 0.897829 | 97.392336 +/- 1.723046 | 0.245470% |
| c4 | 60M | 2048 | 50 | 2 | 248.151969 +/- 4.876379 | 242.677564 +/- 6.525953 | 241.550843 +/- 12.911822 | -0.466453% |

C4 60M/L2048 is the remaining mean loss. Sigma 50 with two HIPE layers improves substantially over four HIPE layers, but does not surpass RoPE. Six-HIPE-layer candidates lose to RoPE in every paired seed in all three targeted settings.

The newly positive mean differences are approximately 0.10-0.25%. WikiText-103 60M/L2048 improves in two of three paired seeds. Each new C4 60M/L512 mean win improves in only one of three paired seeds. A lower selected mean therefore should not be described as consistent seed-wise superiority.

## Best observed HIPE against all baselines

| Corpus | Model | L | Sigma | HIPE layers | Legacy NoPE+APE | ALiBi | RoPE | XPos | HIPE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wikitext103 | 20M | 512 | 200 | 4 | 60.181882 +/- 0.729248 | 50.159828 +/- 0.534017 | 50.677000 +/- 0.374751 | 50.430704 +/- 0.383523 | 49.284134 +/- 0.165303 |
| wikitext103 | 20M | 1024 | 500 | 4 | 91.263017 +/- 0.370956 | 67.412724 +/- 0.522868 | 70.773016 +/- 0.859961 | 69.535283 +/- 1.306802 | 68.548913 +/- 0.376462 |
| wikitext103 | 20M | 2048 | 800 | 4 | 174.205354 +/- 0.818574 | OOM | 115.017409 +/- 1.750982 | 112.928708 +/- 0.087746 | 114.070849 +/- 0.526692 |
| wikitext103 | 60M | 512 | 300 | 4 | 46.552683 +/- 1.882486 | 35.756719 +/- 0.312725 | 39.345951 +/- 0.398817 | 39.297499 +/- 0.385429 | 39.155005 +/- 0.406353 |
| wikitext103 | 60M | 1024 | 300 | 4 | 71.172309 +/- 0.767444 | 42.266619 +/- 0.446609 | 54.886376 +/- 0.537711 | 53.946524 +/- 1.728474 | 53.088699 +/- 0.338573 |
| wikitext103 | 60M | 2048 | 500 | 2 | 140.367138 +/- 6.557607 | 67.322903 +/- 1.039623 | 96.606311 +/- 4.274491 | 93.597496 +/- 5.166416 | 96.383156 +/- 3.383820 |
| c4 | 20M | 512 | 300 | 4 | 153.470813 +/- 0.578573 | 122.942177 +/- 0.113978 | 124.842452 +/- 3.809679 | 125.450225 +/- 2.055925 | 119.773068 +/- 1.263198 |
| c4 | 20M | 1024 | 300 | 4 | 222.023301 +/- 1.618054 | OOM | 169.852084 +/- 1.294024 | 168.851411 +/- 1.555633 | 165.873837 +/- 2.589700 |
| c4 | 20M | 2048 | 800 | 4 | 375.331626 +/- 5.807935 | OOM | 255.977976 +/- 3.474659 | 251.511910 +/- 4.052947 | 254.465588 +/- 5.180511 |
| c4 | 60M | 512 | 200 | 2 | 123.350728 +/- 7.472727 | 87.509025 +/- 0.547090 | 97.392336 +/- 1.723046 | 97.076542 +/- 1.862667 | 97.153268 +/- 0.897829 |
| c4 | 60M | 1024 | 300 | 4 | 203.078424 +/- 3.224998 | 104.226327 +/- 0.616924 | 143.734881 +/- 5.232943 | 140.719299 +/- 4.869028 | 138.836694 +/- 3.874966 |
| c4 | 60M | 2048 | 50 | 2 | 351.955488 +/- 16.027390 | OOM | 241.550843 +/- 12.911822 | 238.545393 +/- 10.594987 | 242.677564 +/- 6.525953 |

XPos has lower mean PPL than RoPE in 11/12 settings. With XPos included, the lowest available primary means belong to ALiBi in six settings, HIPE in three and XPos in three. HIPE-versus-RoPE wins do not imply wins over every baseline.

## Protocol and uncertainty

- Mean and sample SD use three seeds at one fixed sigma and HIPE-layer count; sigma/layer configurations are never pooled.
- Primary endpoint is last regular validation. Final-checkpoint results are a separate diagnostic view. The selected configurations agree across both views.
- Best observed HIPE is selected using these validation results. It is an exploratory tuning outcome, not an independent held-out estimate or a post-selection significance claim.
- Eight total model layers are retained. Two/four/six final layers use HIPE; preceding layers use unscaled RoPE.
- C4 HIPE and RoPE use different original micro-batches, with global batch 64 unchanged. Reported C4 contrasts retain this qualification.
- Old and new source cohorts are kept as distinct fixed-configuration groups. Cross-version contrasts use exact data/runtime matches and the archived unchanged-method numerical audit, not a claim of identical source-file hashes.
- XPos is the reference TorchScale variant with scale base 512 and no learned absolute positions. NoPE is Legacy NoPE+APE.
- Four ALiBi settings remain unavailable due to historical OOM. Failures are unavailable PPL, never zero.
- All 81 full runs completed. Twelve two-step GPU checks completed separately and are excluded from all research statistics.

The same selected sigma/layer configurations and win/loss directions hold at the final checkpoint. The workbook retains both endpoint views and all fixed-configuration results, including losing candidates.

No additional training was launched during this analysis.
