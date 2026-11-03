# Current campaign state

Recovery array **41639** is active: **36 fixed-setting HIPE runs + six corrected ALiBi retries**, one RTX5090 per task, maximum two concurrent tasks, seeds6198/1024/7, 100M-token cap and global batch64. Tasks41639_0 and41639_1 were confirmed training past steps200 and100.

The old baseline portion of41315 ended with93successes and15ALiBiCUDAOOM failures. The216held tuning tasks were canceled after the replacement was accepted. The idle GPUs were caused by the campaign hold, not a server outage.

The new frozen source is ca0e516. HIPE uses the paper4/4split (threshold3), with recovered per-cell sigmas; historical C4threshold4 differences are documented in RECOVERED_SETTINGS.md. The six ALiBi retries restore automatic PyTorch SDPA, with the paper-matched Wiki20M/L1024micro-batch16 and global64.

Check current jobs with `squeue -j 41639`. Details are in RECOVERY_LAUNCH.json. Prior launch history follows.

---

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

## Scope correction

The initial 324-run submission included a six-sigma tuning grid repeated across all three seeds. The main table comparison should target **144 runs**: 108 baselines and 36 HIPE runs with one fixed sigma for each dataset/model/length condition.

All **216 queued HIPE sweep tasks are now held** while the original selected sigmas are recovered. The **108 baseline runs remain eligible**, with one currently running at the scope check. Once the fixed-sigma mapping is established, retain 36 HIPE runs and exclude the remaining 180 sweep tasks. No HIPE sweep task had started when the hold was applied.
