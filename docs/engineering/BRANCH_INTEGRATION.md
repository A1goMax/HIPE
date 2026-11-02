# HIPE Branch Integration and Protocol Findings

The active source is MS04 and Table 2 is the immediate target. The method campaign covers Tables 1-5.

## Code sources

| Feature | Source | Integrated behavior |
| --- | --- | --- |
| Package/model | HIPE bd1d53a, journal 90fd246 | Existing package and history retained |
| Fixed HIPE | PE 060b6a6 | Original spectral weights, square root, global RMS and symmetric query/key scaling; numerical equality fixture |
| Learnable sigma | PE ee75363 and main 6432fc4 | Per-head parameter, clamp/floor and per-head RMS in the shared embedding |
| Local/global attention | PE c0fc0c4 | Explicit layer ID lists with causal masks and cached-key offsets |
| Layer schedules | PE layerwise launchers | Explicit vectors translate paper one-based labels to zero-based layers |
| WikiText main | March wiki20-60 launchers | Global 64; recovered 32/16 micro-batch rule, halved at L2048 |
| January/C4 | PE historical snapshot 8883bd7 | Earlier HIPE micro-batches and C4 preprocessing profiles |
| SST-2 | PE finetune branch/checkouts | Source for a separate future task-adapter integration |

Dirty working-tree source snapshots are local historical evidence, not runtime dependencies or proven run-to-table associations.

## Batch contract

| Profile and method | 20M L512/L1024/L2048 | 60M L512/L1024/L2048 |
| --- | --- | --- |
| March WikiText RoPE/NoPE/HIPE | 32 / 32 / 16 | 16 / 16 / 8 |
| March WikiText ALiBi | 32 / 32 / 8 | 16 / 16 / 4 |
| January WikiText HIPE | 16 / 16 / 8 | 8 / 8 / 4 |
| Historical C4 HIPE | 16 / 16 / 8 | 8 / 8 / 4 |
| Historical C4 baselines | 32 / 32 / 16 | 16 / 16 / 8 |
| Table 5 attention profile | Not used | L1024: 16 for all six conditions |

January WikiText ALiBi contains a stale/unassigned micro-batch variable. It cannot independently establish a reliable rule; the explicit profile uses the later valid ALiBi rule.

Global batch remains 64. Accumulation is global divided by micro-batch. Two GPUs execute independent conditions. Partial epoch-boundary batches retain the inherited mean-loss accumulation behavior; actual prediction tokens are recorded separately from the nominal budget.

## Material findings

1. Legacy NoPE/XPos: disabling RoPE also created and added learned absolute-position embeddings. The compatibility switch preserves this explicitly, while corrected variants omit it. Their identities remain distinct.
2. C4 budget: January launchers specify 200M tokens, while comments/current paper say 100M. The new Table 2 suite uses the paper's 100M.
3. Optimizer/vocabulary: restricted-budget trainers use AdamW, default LR 6e-4 and vocabulary alignment to 64. The prior packaged release had different defaults. Shared settings expose the recovered choices.
4. Preprocessing: January C4 encodes/chunks within each 1,000-document map batch; WikiText/March uses two maps. They produce different boundaries and remain named protocols with fingerprints/process counts.
5. Table 5 launchers: inspected scripts contain L512 and/or sigma 200/300; the paper reports L1024 and sigma 500. The suite follows the paper and records the discrepancy.
6. Validation endpoint: old scripts evaluate every 100 steps and save final weights. An extra final-step evaluation differs from the last interval. Reporting exposes both.
7. Sample ordering: the new independent seeded sampler has a resumable cursor and pairs data across methods. It does not assert bitwise equivalence with old global-RNG consumption.
8. Sigma search: the six-sigma/five-seed grid expands to 540 runs. Aggregation keeps every configuration without implicit favorable-run selection.

## Validation scope

CPU checks use an isolated source copy and the healthy original base environment on labserver2, with CUDA hidden. They cover fixed-formula equality, baseline gradients, schedules, local causal masks and cached decoding, learnable state reload, preprocessing, effective batch accumulation, manifest/checkpoint execution, exact CPU resume and protocol-aware aggregation.

These are code checks, not new paper results or GPU throughput measurements. Production execution is SLURM-only, verifies an allocated RTX 5090 and records source/config/data identity.
