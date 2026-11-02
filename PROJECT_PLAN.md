# HIPE Current Implementation Plan

The active source is MS04, the latest user-supplied 21-page PDF, SHA256 `2d89e566caef56d39362c5e9fcbdd779e3a97dd721850f057f126e4f6ceb20d4`. The current scope is the HIPE experiments in Tables 1-5, with Table 2 first. Biological population-code simulations from the old plan are outside this task.

## Shared implementation milestone

Use one model and trainer with explicit positional method, per-layer scales, fixed/learnable sigma and local/global attention. Recover details from PE history rather than merging whole divergent model files. Author code in the E: workspace and synchronize the same journal branch on labserver2.

| Manuscript item | Current preparation |
| --- | --- |
| Table 2 restricted-budget LM | Baseline/HIPE configuration, batch profiles, sigma grid, preprocessing, trainer, evaluator, SLURM launcher and aggregation implemented |
| Table 5 attention ablations | Six combinations use the shared model/trainer; paper settings and source differences are explicit |
| Table 3 300M scaling | Model supports 300M, XPos and learnable sigma; exact per-cell run specifications still require recovery |
| Table 1 retrieval | Sources retained; task adapter migration remains |
| Table 4 SST-2 LoRA | Historical fine-tuning branch identified; classifier/LoRA adapter validation remains |
| Table 6 layer schedules | Configuration support exists; runs are outside the requested Tables 1-5 scope |

## Execution sequence

1. Complete CPU code checks and the branch/protocol comparison.
2. Adjudicate recorded historical settings and inspect the resolved paired configuration.
3. Run a short paired 20M WikiText check through SLURM on one or two RTX 5090 GPUs, retaining original batch sizes.
4. Measure memory and throughput, freeze an execution revision, then submit filtered Table 2 slices.
5. Re-evaluate checkpoints, aggregate the specified endpoint/sigma search, and compare against reported values.
6. Integrate the remaining Table 1/3/4 adapters and campaign specifications as their settings are verified.

This step prepares code and submission previews. It does not launch GPU experiments. New results must identify source, data and configuration and distinguish protocol replication from exact historical replay.
