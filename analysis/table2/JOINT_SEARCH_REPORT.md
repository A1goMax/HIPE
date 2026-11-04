# Table 2 public results

This report follows the paper's Table 2 method set: HIPE, RoPE, ALiBi and Legacy NoPE+APE. Additional comparisons are retained in local analysis outputs and are not included in this public table.

The 45 new HIPE runs completed. Nine earlier HIPE runs complete the targeted 18-configuration grid. Means and sample SDs use seeds 6198, 1024 and 7 at one fixed sigma/layer setting. GPU checks are excluded.

## Main finding

Best observed HIPE has lower mean PPL than RoPE in 11/12 settings, up from 9/12 before the layer search. This uses per-setting sigma/layer selection. Two previously losing settings become mean wins; C4 60M/L2048 remains a mean loss.

## Newly mean-winning configurations

| Corpus | Model | L | Sigma | HIPE layers | HIPE mean +/- SD | RoPE mean +/- SD | Reduction | Paired seed wins |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wikitext103 | 60M | 2048 | 500 | 2 | 96.383156 +/- 3.383820 | 96.606311 +/- 4.274491 | 0.230994% | 2/3 |
| c4 | 60M | 512 | 200 | 2 | 97.153268 +/- 0.897829 | 97.392336 +/- 1.723046 | 0.245470% | 1/3 |
| c4 | 60M | 512 | 500 | 2 | 97.293800 +/- 1.045408 | 97.392336 +/- 1.723046 | 0.101175% | 1/3 |

## Best observed HIPE and Table 2 baselines

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
| c4 | 60M | 2048 | 50 | 2 | 351.955488 +/- 16.027390 | OOM | 241.550843 +/- 12.911822 | 242.677564 +/- 6.525953 |

Among the Table 2 methods, the lowest available primary means belong to HIPE in five settings, ALiBi in six and RoPE in one.

C4 60M/L2048 remains a mean loss: sigma 50 with two HIPE layers gives 242.677564 +/- 6.525953 versus RoPE 241.550843 +/- 12.911822. Both endpoint views preserve the selected configurations and win/loss directions.

The new positive mean differences are small and validation-selected. The C4 L512 mean wins favor HIPE in only one of three paired seeds. These are exploratory tuning outcomes, not independent held-out superiority or significance claims.

## Protocol qualifications

- Mean and sample SD use three seeds at one fixed sigma and HIPE-layer count; sigma/layer configurations are never pooled.
- Primary endpoint is last regular validation. Final-checkpoint results are a separate diagnostic view. The selected configurations agree across both views.
- Best observed HIPE is selected using these validation results. It is an exploratory tuning outcome, not an independent held-out estimate or a post-selection significance claim.
- Eight total model layers are retained. Two/four/six final layers use HIPE; preceding layers use unscaled RoPE.
- C4 HIPE and RoPE use different original micro-batches, with global batch 64 unchanged. Reported C4 contrasts retain this qualification.
- Old and new source cohorts are kept as distinct fixed-configuration groups. Cross-version contrasts use exact data/runtime matches and the archived unchanged-method numerical audit, not a claim of identical source-file hashes.
- Public Table 2 methods are HIPE, RoPE, ALiBi and Legacy NoPE+APE. Additional comparison results remain local.
- Four ALiBi settings remain unavailable due to historical OOM. Failures are unavailable PPL, never zero.
- All 45 new HIPE runs completed. Nine previous HIPE runs complete the target grid. Twelve two-step GPU checks are excluded.

The full local comparison archive remains unchanged. Historical Git snapshots are preserved; this is a forward publication-scope update.
