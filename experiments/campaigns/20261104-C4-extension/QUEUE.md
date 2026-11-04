# Wider C4 sigma/layer campaign

Submitted 4 November 2026 (Asia/Shanghai).

Pinned before GPU submission:
research/table2-before-c4-extension-v2-20261104,
source commit c44c207b83bf4140ec462bd727b1c5e6fc946613.
The tag pairs the cleaned current public Table 2 results with the exact new
suite and validation record. Historical snapshots remain unchanged.

| Job | Purpose | Tasks | Initial state |
| --- | --- | ---: | --- |
| 42038 | Two-step GPU checks | 4 | Pending scheduler priority |
| 42039 | Full C4 extension | 60 | Pending afterok:42038 |

The full array starts only after all four checks succeed. An invalid check
dependency cancels it before training. Both arrays use one GPU per task and
at most two concurrent tasks, with no overlap between phases.

The 20 new configurations use three seeds, 100M tokens, micro-batch 4 and
global batch 64. Twelve archived seed-runs complete the 24-configuration grid.
No XPos experiment is submitted. Current public reports exclude its comparison
results; the full local comparison and historical Git snapshots are retained.

LAUNCH.json records the source publication receipts, config paths and hashes,
output paths, sigma/layer/seed identities and scheduler dependency. The five
model/trainer core files are byte-identical to the completed reference source.
Pending state is not a completion claim.
