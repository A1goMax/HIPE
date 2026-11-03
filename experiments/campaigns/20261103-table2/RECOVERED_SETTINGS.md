# Fixed main-experiment settings

The 108 baseline attempts are finished: 93 succeeded and 15 ALiBi conditions ran out of GPU memory. The old 216-condition HIPE sweep was held during scope correction.

The replacement uses 12 fixed HIPE settings and three seeds (36 runs), the 100M-token cap and the current paper four-shallow/four-deep hierarchy (threshold3). Historical matching logs used threshold4 for five C4 conditions; this difference is disclosed rather than silently replayed as the paper hierarchy.

| Corpus | Model | sigma L512 | sigma L1024 | sigma L2048 |
| --- | --- | --- | --- | --- |
| WikiText | 20M | 200 | 500 | 500 |
| WikiText | 60M | 300 | 300 | 700 |
| C4 | 20M | 300 | 300 | 700 |
| C4 | 60M | 500 | 300 | 50 |

Matching historical validation records and their source lines are in work/sigma-recovery/. Wiki20M/L512 has explicit sigma200, threshold3 and21.3M parameters; length512 is inferred from100M tokens/3051steps/global64.

ALiBi audit: the paper-matching Wiki20M/L1024 log explicitly used micro-batch16, while the submitted job used32. The old code allowed automatic PyTorch SDPA dispatch; the new launcher had forced math. The corrected main specification restores automatic SDPA and the supported Wiki micro-batch16 while retaining global64. Failed attempts remain recorded. No OOM result is silently replaced or relabeled.
