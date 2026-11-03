# Sigma/layer search and reference XPos campaign

Submitted 3 November 2026 (Asia/Shanghai).

| Job | Purpose | Source commit | Tasks | Initial state |
| --- | --- | --- | ---: | --- |
| 41769 | Two-step GPU checks | f304bce6f2aadb98ab2344ac548bd5d9e361a6f2 | 12 | Pending scheduler priority |
| 41770 | Full sigma/layer search and XPos comparison | 5df7693e38ede95a9fa1747207b59fd2c701e75a | 81 | Pending afterok:41769 |

Both source commits were pushed to A1goMax/HIPE before submission.
Model/trainer/config-builder files are byte-identical between these two
frozen source trees. The second commit adds scheduler dependency support.

The full array waits for every task in the GPU-check array to succeed.
SLURM cancels it if the dependency becomes invalid. Both arrays use one GPU
per task and a concurrency limit of two; the dependency prevents overlap.
Short checks have a 30-minute limit and are excluded from research results.

Full runs comprise 45 new HIPE runs and 36 XPos runs, each at the 100M-token
cap and the original batch profile. Nine existing HIPE seed-runs are reused.
See the versioned prospective protocol for the grid and selection rule.

LAUNCH.json contains the exact source publication receipts, frozen source
locations, commands, configuration paths/checksums, output directories,
seed/sigma/layer settings and initial scheduler records. Pending state is
not a completion claim. No full run had started at the time of this record.
