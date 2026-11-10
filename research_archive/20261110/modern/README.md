# Public modern run evidence

Snapshot date 2026-11-10. The index covers 803 tasks across 11 campaigns.
585 tasks produced run directories, including 541 research completions and 24 completed GPU checks.
The scheduler recorded 216 cancelled tasks, 20 failed tasks and 2 node failures.

## Source versions

Each task links its planned configuration and scheduler outcome. Observed runs link their exact saved source ZIP by SHA-256. The saved source members were checked against the recorded Git commit. Completed, incomplete, and GPU-check results remain separate.

| Training commit | Campaign jobs | Indexed tasks | Run directories | Research completions | GPU checks completed |
| --- | --- | ---: | ---: | ---: | ---: |
| [12ebc719](https://github.com/A1goMax/HIPE/commit/12ebc719d980d8dd4fdf7a7bef6cf2ead40719b6) | 42168, 42169 | 272 | 272 | 262 | 8 |
| [1db028e0](https://github.com/A1goMax/HIPE/commit/1db028e05761e9089c4c36a3ad94c22829561535) | 41695 | 42 | 42 | 42 | 0 |
| [224bc280](https://github.com/A1goMax/HIPE/commit/224bc280b72a595d4e1aa5273ac35f47a0ba24d0) | 41313, 41315 | 326 | 110 | 93 | 2 |
| [5df7693e](https://github.com/A1goMax/HIPE/commit/5df7693e38ede95a9fa1747207b59fd2c701e75a) | 41770 | 45 | 45 | 45 | 0 |
| [b7d78492](https://github.com/A1goMax/HIPE/commit/b7d784929e7b9120a96ec9655d05b48156504d79) | 41311 | 2 | 0 | 0 | 0 |
| [c44c207b](https://github.com/A1goMax/HIPE/commit/c44c207b83bf4140ec462bd727b1c5e6fc946613) | 42038, 42039 | 64 | 64 | 60 | 4 |
| [ca0e5163](https://github.com/A1goMax/HIPE/commit/ca0e5163a319ae04aa7930e1c5b8de03ae650c84) | 41639 | 42 | 42 | 39 | 0 |
| [f304bce6](https://github.com/A1goMax/HIPE/commit/f304bce6f2aadb98ab2344ac548bd5d9e361a6f2) | 41769 | 10 | 10 | 0 | 10 |

## Files

- [Task index](index.json) maps every task to its version, seed, setting, and files.
- The runs directory holds the config, runtime/source manifest, exact metrics, completion record and available logs.
- The sources directory contains ten distinct, checksum-verified source ZIPs.
- The campaigns directory contains submission and scheduler evidence.
- The [file manifest](file_manifest.json) records byte counts and SHA-256 values.
- This projection covers HIPE, RoPE, ALiBi, and Legacy NoPE+APE.
- [Figures](figures/) accompany the public-method results.

Model and optimizer filenames and sizes are indexed for reference. The archive scope is code and analysis evidence.
