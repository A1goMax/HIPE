# HIPE Implementation Status

Updated 3 November 2026, Asia/Shanghai.

## Current scope

MS04 is the active 21-page manuscript. The requested scope is HIPE Tables 1-5, with Table 2 first. The earlier biological-simulation campaign is superseded for this task. Table 6 schedules are supported as configurations, outside the current run scope.

## Implemented

- One shared model configuration for RoPE, HIPE, NoPE, ALiBi and XPos.
- Fixed and learnable HIPE, explicit per-layer schedules, and shallow/deep local attention.
- Table 2 suite with named original batch profiles, three seeds and six HIPE sigma values; 324 search runs before filtering.
- Six Table 5 conditions through the same model/trainer with global batch 64 and micro-batch 16.
- One SLURM launcher for one or two RTX 5090 GPUs. Two-GPU jobs run independent conditions with unchanged per-condition batches.
- Named legacy preprocessing, padded vocabulary, recovered optimizer settings, actual token counts, source/data manifests, unique run directories and full resumable checkpoints.
- Saved-config evaluation and explicit last-interval/final-endpoint aggregation with per-seed values and protocol separation.
- Latest-paper registry: 78 reported targets across Tables 1-5.

## Validation

36 CPU tests pass in an isolated source copy on labserver2 with CUDA hidden. These include legacy fixed-formula equality, all baseline gradients, learnable state reload, layer schedules, local causal masks/cached decoding, original global-batch accumulation, preprocessing, full two-step manifest/checkpoint execution, saved-config evaluation and exact CPU resume.

Submission previews are checked without calling sbatch. The source also retains a vendored PyTorch deprecation warning during CPU attention-mask checks.

GPU preflight array 41313 completed for RoPE and HIPE. Main Table 2 array 41315 is running/queued with three seeds and a 100M-token cap. See experiments/campaigns/20261103-table2/RUNNING.md. Full paper metrics are not complete.

The healthy original environment is /home/qijunrong/anaconda3/bin/python, with Torch 2.8.0+cu128 and NumPy 1.26.4. Earlier import failures affected other environments. Observed versions are recorded under environment/.

## Remaining work

1. Review the historical NoPE/XPos learned-position behavior and the C4 200M-versus-paper-100M discrepancy before treating a run as exact historical replay.
2. Recover final per-cell sigma/checkpoint selection and run associations.
3. Run a short paired Table 2 SLURM pilot with the preserved batch settings, then measure cost before scaling the campaign.
4. Finish the separate synthetic retrieval and SST-2 LoRA adapter migrations and the larger-model campaign specifications for Tables 1/3/4.

All source changes are authored in E:/research/03/HIPE and synchronized to journal/research-foundation on labserver2. Existing release history and historical PE checkouts remain available.
