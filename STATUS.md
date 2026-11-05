# HIPE Research Status

Updated 5 November 2026, Asia/Shanghai.

## Latest completed C4 extension

GPU checks 42038 (four two-step tasks) and full array 42039 (60 runs) completed
successfully. The full campaign used the published source
c44c207b83bf4140ec462bd727b1c5e6fc946613, pinned before execution under
research/table2-before-c4-extension-v2-20261104.

The wider search does not produce a lower mean PPL than RoPE for C4 60M/L2048.
Best observed HIPE is sigma 500 with two final HIPE layers:
242.321885 +/- 7.337535, versus RoPE 241.550843 +/- 12.911822.
The difference is +0.771042 PPL (+0.319205%).
The earlier best was sigma 50/H2 at 242.677564 +/- 6.525953.
The selected result improves in two of three paired seeds but retains a
higher mean. The final-checkpoint view selects the same configuration and
also trails RoPE.

Read [the completed extension report](analysis/table2/C4_EXTENSION_REPORT.md).
The verified workbook, every fixed-configuration result, exact seed PPLs,
figures and audits are in outputs/table2_c4_20261105/.
The versioned result archive is experiments/results/table2_c4_20261105/.

## Current Table 2 evidence

Best observed HIPE has lower three-seed mean PPL than RoPE in 11/12 settings.
This count is unchanged after the extension. Among the paper's Table 2
methods, lowest available primary means belong to HIPE in five settings,
ALiBi in six and RoPE in one.

The public evidence contains 297 full execution attempts, 279 completed unique
seed-runs and 18 earlier failures. All 93 completed configurations have three
seeds; 61 configurations use HIPE. Four ALiBi settings remain unavailable due
to OOM. The latest C4 comparison retains 30 observed sigma/layer configurations:
24 in the new planned grid and six earlier out-of-grid configurations.

All means and sample SDs use seeds 6198, 1024 and 7 at one fixed configuration.
Sigma/layer settings are never pooled. Last regular validation is primary;
final-checkpoint values are kept separate. Twelve earlier seed-runs complete
the new grid, and all four GPU checks are excluded from research statistics.

## Interpretation and limitations

At sigma 50, two final HIPE layers give the lowest observed mean over H1-H8.
Increasing the count beyond two worsens the observed mean. One HIPE layer
does not uniformly outperform two. Both very small sigmas and sigma 2000
underperform the selected sigma 500/H2 configuration.

These are validation-selected tuning outcomes. Sample SD describes conditional
seed variation and does not include hyperparameter-selection uncertainty.
The wider search narrows the mean gap but does not demonstrate a C4
60M/L2048 advantage over RoPE or a statistical significance result.

C4 retains its original HIPE/RoPE micro-batches (4 versus 8 at L2048 for 60M),
with global batch 64 unchanged. Data, tokenizer/vocabulary identities, Python,
CUDA/package versions, GPU type and both endpoint exposures match archived
RoPE controls. The legacy NoPE baseline includes learned absolute embeddings.
Historical paper-log population and layer-split differences remain separate
qualifications for absolute reproduction of paper values.

## Provenance and publication

All 60 frozen configurations, run configurations and source archives were
verified; all 300 core source entries inside the archives matched their
recorded checksums. The new training core is byte-identical to the previous
joint-study core. Result records retain commit, resolved configuration,
effective layer vector, runtime/data identities, metrics and checkpoint hashes.

Current public outputs follow the paper's Table 2 method set. XPos comparison
results remain in ignored local outputs. Historical Git snapshots are preserved.
The new result archive is tagged research/table2-c4-results-v2-20261105.
No training was launched during this analysis.

## Next priorities

Resolve the remaining evaluation, micro-batch and layer-split qualifications,
and use independent evaluation before journal superiority claims.
Keep true position-free NoPE distinct from the legacy NoPE+APE baseline.
Table 1 retrieval, Table 3 larger-model specifications and Table 4 LoRA
integration remain pending.
