# HIPE Research Status

Updated 4 November 2026, Asia/Shanghai.

## Completed Table 2 campaigns

The main arrays 41315 and 41639, L2048 sigma array 41695, GPU-check array
41769 and joint sigma/layer + XPos array 41770 have ended.
All 81 new full runs completed successfully: 45 HIPE and 36 XPos.
Nine previously completed HIPE seed-runs complete the 18-configuration target
grid. The 12 two-step GPU checks are excluded from research statistics.

Across the full evidence set there are 273 execution attempts, 255 completed
unique seed-runs and 18 previous failed attempts. There are 85 complete fixed
configurations with three seeds, including 41 HIPE configurations. Four ALiBi
settings remain unavailable due to OOM. Superseded, unstarted tuning tasks are
excluded.

Read [the joint-search report](analysis/table2/JOINT_SEARCH_REPORT.md).
The formal workbook, exact seed PPLs, all fixed sigma/layer statistics, baseline
comparisons, figures and verification records are in outputs/table2_joint_20261104/.
Earlier reports remain historical records of the preceding campaigns.

## Latest findings

Best observed HIPE has lower three-seed mean PPL than RoPE in 11/12 settings,
up from 9/12 before the layer search. This includes per-setting selection of
sigma and the number of HIPE layers, rather than the original four-HIPE-layer
configuration throughout.

The newly mean-winning configurations are:
- WikiText-103 60M/L2048: sigma 500, two HIPE layers, 96.383156 +/- 3.383820
  versus RoPE 96.606311 +/- 4.274491 (two of three paired seeds favor HIPE).
- C4 60M/L512: sigma 200, two HIPE layers, 97.153268 +/- 0.897829 versus
  RoPE 97.392336 +/- 1.723046 (one of three paired seeds).
- C4 60M/L512: sigma 500, two HIPE layers, 97.293800 +/- 1.045408 versus
  the same RoPE control (one of three paired seeds).

C4 60M/L2048 is the remaining mean loss. Sigma 50 with two HIPE layers gives
242.677564 +/- 6.525953, versus RoPE 241.550843 +/- 12.911822.
Every tested fixed-sigma sequence has mean PPL ordered H2 < H4 < H6.
The selected configurations and win/loss directions agree at the final checkpoint.

Reference XPos has lower mean PPL than RoPE in 11/12 settings.
With all available methods included, lowest primary means belong to ALiBi
in six settings, HIPE in three and XPos in three.

## Evidence and qualifications

Mean and sample SD are computed across seeds 6198, 1024 and 7 at one fixed
sigma/layer configuration. Different configurations are never pooled.
The primary endpoint is last regular validation; final-checkpoint evaluations
are separate.

New mean gains over RoPE are small (about 0.10-0.25%) and selected using the
same validation results. They do not establish significance or independent
held-out superiority. C4 mean wins in this search improve in only one of three
matched seeds. SD measures conditional seed variation, not selection uncertainty.

Data, tokenizer/vocabulary identities, runtime and endpoint token exposure
match the archived RoPE controls. Source cohorts remain separate statistical
groups. Cross-version comparisons use the archived numerical parity evidence,
an independent AST audit of unchanged HIPE/training functions and exact
data/runtime matches. Source-file hashes are not asserted identical across the
XPos implementation change.

C4 HIPE retains the original smaller micro-batches than RoPE (8 versus 16 at
L512 and 4 versus 8 at L2048 for 60M), with global batch 64 unchanged.
The legacy NoPE baseline includes learned absolute position embeddings.
Historical paper-log validation/population and layer-split differences remain
qualifications for absolute reproduction of paper values.

## Reproducibility

The completed pre-search campaign is preserved at tag
research/table2-sigma-v1-20261103 (eb35933). The full joint campaign used
published source commit 5df7693e38ede95a9fa1747207b59fd2c701e75a.
Every new resolved configuration, run configuration and source archive was
verified against its recorded checksum; all 405 core source entries inside
the 81 archives matched their manifests.

The archived result index connects source commits, resolved configurations,
data/tokenizer identities, runtime, metrics and checkpoint hashes.
No new training was launched during this analysis.

## Next priorities

Confirm small tuned two-layer gains using independent evaluation before
drawing a journal claim of superiority. Resolve remaining micro-batch,
historical evaluation and layer-split comparisons before scaling.
Keep a true position-free NoPE baseline distinct from Legacy NoPE+APE.
Table 1 retrieval, Table 3 larger-model specifications and Table 4 LoRA
integration remain pending.
