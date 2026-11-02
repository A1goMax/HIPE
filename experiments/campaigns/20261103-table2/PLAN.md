# Table 2 main campaign

Authorized 3 November 2026, Asia/Shanghai.

- Seeds: 6198, 1024, 7.
- Token cap: 100,000,000; complete global-batch steps as in the original scripts.
- Global batch: 64; recovered per-condition micro-batches unchanged.
- SLURM: one RTX 5090 per task; no DDP; at most two concurrent tasks.
- Scope: both corpora, both model sizes, all three sequence lengths, NoPE/ALiBi/RoPE and six HIPE sigma values. This is 324 search runs.
- NoPE retains the explicitly labeled historical learned-absolute-position compatibility mode.
- The user confirmed 200M tokens was a post-paper change; 100M is authoritative.
- A two-condition RoPE/HIPE GPU preflight precedes the full array. Each campaign uses a frozen source checkout.

Job IDs and artifact paths will be recorded locally after submission.
