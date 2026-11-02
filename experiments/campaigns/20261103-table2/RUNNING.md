# Table 2 campaign started

SLURM array **41315** was submitted on 3 November 2026 (Asia/Shanghai). The first condition is running and initially preparing the full WikiText data; remaining tasks wait for resources.

- Three seeds: **6198, 1024, 7**.
- **100M-token cap**; original complete-global-batch rounding, never 200M. Actual exposures are logged.
- Global batch **64**, recovered micro-batch profiles unchanged.
- **One RTX 5090 per array task**, no DDP, at most **two** tasks concurrently.
- **324 runs**, including the six-sigma HIPE search across both datasets, model sizes and three sequence lengths.
- Frozen execution source: **224bc28**. Queued jobs do not import subsequent edits to the working branch.

The RoPE and HIPE GPU preflight tasks in array **41313** both completed. Initial preflight 41311 failed before training because of a Slurm spool-path issue; the wrapper was corrected and its regression test added.

Remote campaign: `/data/qijunrong/03-proj/HIPE-journal/campaigns/20261102T163208Z-502ddcd1`.

Local copies of every resolved configuration and the campaign/submission manifests are in `artifacts/campaigns/41315/`. Launch facts are recorded in `LAUNCH.json`.

Check live state on labserver2 with `squeue -j 41315`; finished task states with `sacct -j 41315 --format=JobID,State,ExitCode,Elapsed`. This record captures launch state, not completed paper results. Legacy NoPE+APE remains explicitly labeled in its run identities.
