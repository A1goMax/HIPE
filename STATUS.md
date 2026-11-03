# HIPE Research Status

Updated 3 November 2026, Asia/Shanghai.

## Completed Table 2 and L2048 sigma sweep

The main arrays 41315 and 41639 and high-sigma array 41695 have ended. All 42 new high-sigma jobs completed with exit code 0. Six matching previous runs complete the requested L2048 grid: two corpora x two model sizes x four sigmas x three seeds = 48 runs. The dependent CPU analysis job 41698 also completed. No training was launched for this analysis.

The combined evidence contains 174 completed unique seed-runs and 18 failed attempts, across 192 execution attempts. Four ALiBi settings remain unavailable (12 seed-runs). The 216 superseded tuning tasks are excluded.

Best observed HIPE has lower mean PPL than RoPE in 9/12 settings and 26/36 matched-seed comparisons. Wins by length are 3/4 at 512, 4/4 at 1024 and 2/4 at 2048. Among available methods, HIPE has the lowest mean in five settings, ALiBi in six and RoPE in one. Final-checkpoint results preserve these directions.

At L2048, sigma 800 has the lowest three-seed mean among tested candidates for WikiText-103 20M, WikiText-103 60M and C4 20M. C4 60M retains sigma 50; its best candidate within 700-1000 is 1000. The sweep changes only C4 20M/L2048 from a mean loss to a small mean improvement versus RoPE.

Read [the completed sweep report](analysis/table2/COMPLETED_SIGMA_REPORT.md). The formal workbook, separate endpoint views, all fixed-sigma statistics, exact logged seed PPLs, coverage, failure records, figures and verification are in outputs/table2_sigma_final/. The earlier analysis/table2/REPORT.md describes the pre-sweep fixed-setting campaign and remains a historical record.

## Analysis protocol and evidence quality

Choose the sigma with the lowest complete three-seed mean within one fixed non-sigma protocol. Compute sample standard deviation across seeds 6198, 1024 and 7 at that sigma, never across different sigmas. The primary endpoint is last regular validation; final-checkpoint evaluations are separate. L512 and L1024 each have only one completed sigma.

Best observed values are selected using validation results from the same three seeds. They are tuning outcomes, not an independent held-out estimate. Seed SD does not include sigma-selection uncertainty. No post-selection significance claim is made.

All 12 HIPE/RoPE settings share data identity, evaluation population, vocabulary, seeds, global batch 64 and token exposure. The frozen trainer, preprocessing and model files match byte for byte across the relevant campaigns. Every run uses a 100M-token cap, with complete-batch rounding below the cap.

C4 HIPE retains smaller micro-batches than RoPE. Dropout and accumulation effects remain a potential confound. Five historical C4 paper-log matches used a five/three layer split; these runs use the current paper four/four split. NoPE is explicitly labeled Legacy NoPE+APE because it includes learned absolute position embeddings. Historical validation source has a 20-batch limit; current evaluation uses the full validation population with token weighting. Exact historical runtime provenance remains incomplete.

## Next priorities

Resolve the evaluation, micro-batch and layer-split comparisons before scaling the experiments. Keep a true position-free NoPE baseline distinct from Legacy NoPE+APE. Table 1 retrieval, Table 3 larger-model campaign specifications and Table 4 LoRA adapter integration remain pending. The shared implementation and SLURM-only single-GPU execution are preserved.

## Next bounded campaign

The completed version and exact per-run mappings have been pushed to
A1goMax/HIPE under research/table2-sigma-v1-20261103 (eb35933).
The prospective sigma/layer + XPos protocol is in
experiments/protocols/table2_sigma_layers_xpos_v1.md. It contains 81 new full
runs (45 HIPE and 36 XPos), reusing nine prior HIPE seed-runs.
The code passed 50 CPU tests, numerical comparisons with the pinned authors'
XPos source in 12 shape/offset cases, and exact checks of 1,008 existing-model
parameter/gradient tensors plus logits. GPU execution awaits the SLURM checks.

SLURM GPU-check array 41769 (12 two-step checks) is queued with two-task concurrency.
The full array will use an afterok dependency, with cancellation if any check fails.

Full SLURM array 41770 is submitted from published commit 5df7693, with
afterok:41769 and automatic cancellation on an invalid check dependency.
Initial states are pending priority (41769) and pending dependency (41770).
Each phase uses one GPU per task and at most two concurrent tasks.
Exact launch receipts are in
experiments/campaigns/20261103-joint-sigma-layers-xpos/LAUNCH.json.
