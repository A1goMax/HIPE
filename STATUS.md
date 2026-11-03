# HIPE Research Status

Updated3November2026, Asia/Shanghai.

## Table2 campaign

Arrays41315 and41639 have ended. Adjudicated results contain132successful unique main runs (44cells x3seeds),12unavailable runs (4OOMcells),150execution attempts including6retries. The216superseded tuning tasks were canceled and are excluded. No new GPU work was submitted for analysis.

HIPE improves meanPPL versusRoPE in8/12settings and25/36seed pairs. Wins by length:512=3/4,1024=4/4,2048=1/4. Final-checkpoint sensitivity leaves all directions unchanged. HIPE is best among available methods in4/12settings. Three-seed uncertainty and protocol differences limit publication claims.

Read analysis/table2/REPORT.md and outputs/table2_20261103/REPORT.md for the complete table, plots, uncertainty, papercomparison and audit. Numerical CSVs and a standalone reproduction package are under outputs/.

## Evidence quality

All12paired cells share processed/raw data identity, evaluation population, vocabulary, seeds, globalbatch64 and tokenexposure. Frozen trainer/data/model sources match byte-for-byte. C4HIPE retains smaller micro-batches thanRoPE, so dropout/accumulation numerical effects are a remaining confound. Five historical C4matches used5/3layers while these runs use the paper4/4. Our NoPE is the explicitly labeled legacyNoPE+APE. Historical validation source uses a20batchlimit, while current evaluation is fulltokenweighted; exact historical runtime provenance remains incomplete.

## Next priorities

Resolve the evaluation/micro-batch/hierarchy comparisons before scaling the experiment. Keep a real position-freeNoPE baseline separate from legacyNoPE+APE. Additional Table1retrieval, Table3large-model specifications and Table4LoRAadapter work remain pending. The shared implementation and SLURM-only single-GPU execution are preserved.
