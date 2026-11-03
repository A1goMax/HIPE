# HIPE Research Status

Updated 3 November 2026, Asia/Shanghai.

## Completed Table 2 campaign

Arrays 41315 and 41639 have ended. The adjudicated main results contain 132 successful unique runs (44 cells with three seeds), 12 unavailable runs (four OOM cells), and 150 execution attempts including six retries. The 216 superseded tuning tasks are excluded. No new GPU work was submitted for the analysis.

HIPE improves mean perplexity over RoPE in 8/12 settings and 25/36 matched-seed comparisons. Wins by length are 3/4 at 512, 4/4 at 1024, and 1/4 at 2048. Final-checkpoint evaluations leave all directions unchanged. HIPE has the lowest mean among available methods in 4/12 settings. Three-seed uncertainty and protocol differences limit the conclusions.

Read analysis/table2/REPORT.md and outputs/table2_20261103/REPORT.md for the full table, plots, uncertainty, paper comparison and audit. Numerical CSVs and a standalone reproduction package are under outputs/.

## Evidence quality

All 12 paired HIPE/RoPE cells share data identity, evaluation population, vocabulary, seeds, global batch 64 and token exposure. The frozen trainer, preprocessing and model files match byte for byte.

C4 HIPE retains smaller micro-batches than RoPE; dropout and accumulation effects remain a potential confound. Five historical C4 log matches used a 5/3 hierarchy, while these runs use the paper's 4/4 hierarchy. The NoPE runs are explicitly labeled legacy NoPE+APE. Historical validation source has a 20-batch limit; current evaluation uses the full validation population with token weighting. Exact historical runtime provenance remains incomplete.

## Next priorities

Resolve the evaluation, micro-batch and hierarchy comparisons before scaling the experiments. Keep a true position-free NoPE baseline distinct from legacy NoPE+APE. Table 1 retrieval, Table 3 larger-model campaign specifications and Table 4 LoRA adapter integration remain pending. The shared implementation and SLURM-only single-GPU execution are preserved.
