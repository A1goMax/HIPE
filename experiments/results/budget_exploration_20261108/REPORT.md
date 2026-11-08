# Matched-step token-budget exploration

Snapshot: 8 November 2026, Asia/Shanghai. This report is separate from the paper's 100M-token study.

## Completion and coverage

All 264 submitted tasks have ended: **262 completed successfully and two ended NODE_FAIL**. All eight two-step GPU checks completed and are excluded. The study reuses 63 completed L512/100M runs, giving 325 complete seed-runs out of 327 planned study seed-runs.

There are 109 study configurations: 107 have all three seeds and two have only two completed seeds. Incomplete configurations retain their available seed PPLs, but their formal aggregate mean/SD is unavailable and they cannot enter best-configuration selection.

| Length | Budget | Source | Study configurations | Complete configurations | Successful seed-runs |
| ---: | ---: | --- | ---: | ---: | ---: |
| 512 | 100M | Reused | 21 | 21 | 63 |
| 1024 | 200M | New training | 31 | 30 | 92 |
| 2048 | 400M | New training | 57 | 56 | 170 |

The 72 additional 100M long-length protocols are reference controls only, not candidates for the matched-step summary. The complete evidence package contains 362 fixed-protocol endpoint rows and 1,082 exact PPL observations, including these historical controls.

## Main result

**The best complete three-seed HIPE configuration has lower mean PPL than RoPE in all 12 study settings**, including 4/4 at each length. It improves in 31/36 matched seed comparisons. Both primary regular validation and final-checkpoint views retain the 12/12 direction.

This is best-of-tested sigma/layer selection, not a claim that every HIPE configuration wins. Only 32/75 complete study HIPE configurations have lower mean PPL than RoPE. Among all available paper-scope methods, the lowest selected mean belongs to HIPE in seven settings and ALiBi in five.

**The original all-100M result remains 11/12**. The exploratory 12/12 result combines L512/100M, L1024/200M and L2048/400M; it must not replace or relabel the main paper-budget table.

## Selected complete configurations: primary endpoint

Means and sample SDs are across seeds 6198, 1024 and 7 at one fixed budget/sigma/layer setting. Primary endpoint is the last regular validation, step 3000.

| Corpus | Model | L | Budget | Sigma | HIPE layers | Complete / planned HIPE candidates | HIPE mean +/- SD | RoPE mean +/- SD | Reduction | Seed wins |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wikitext103 | 20M | 512 | 100M | 200 | 4 | 1/1 | 49.284134 +/- 0.165303 | 50.677000 +/- 0.374751 | 2.748516% | 3/3 |
| wikitext103 | 60M | 512 | 100M | 300 | 4 | 1/1 | 39.155005 +/- 0.406353 | 39.345951 +/- 0.398817 | 0.485298% | 2/3 |
| c4 | 20M | 512 | 100M | 300 | 4 | 1/1 | 119.773068 +/- 1.263198 | 124.842452 +/- 3.809679 | 4.060626% | 3/3 |
| c4 | 60M | 512 | 100M | 200 | 2 | 6/6 | 97.153268 +/- 0.897829 | 97.392336 +/- 1.723046 | 0.245470% | 1/3 |
| wikitext103 | 20M | 1024 | 200M | 500 | 4 | 5/5 | 39.744355 +/- 0.240410 | 41.401038 +/- 0.484441 | 4.001549% | 3/3 |
| wikitext103 | 60M | 1024 | 200M | 300 | 1 | 5/5 | 31.501384 +/- 0.263119 | 31.920506 +/- 0.086634 | 1.313017% | 3/3 |
| c4 | 20M | 1024 | 200M | 300 | 4 | 5/5 | 95.359672 +/- 0.513158 | 99.559194 +/- 1.871036 | 4.218115% | 3/3 |
| c4 | 60M | 1024 | 200M | 300 | 2 | 4/5 | 77.937090 +/- 1.994418 | 78.590441 +/- 2.082565 | 0.831336% | 2/3 |
| wikitext103 | 20M | 2048 | 400M | 500 | 4 | 5/5 | 34.444745 +/- 0.100000 | 35.506994 +/- 0.329114 | 2.991663% | 3/3 |
| wikitext103 | 60M | 2048 | 400M | 800 | 2 | 9/9 | 26.465979 +/- 0.472097 | 26.738622 +/- 0.564018 | 1.019658% | 3/3 |
| c4 | 20M | 2048 | 400M | 800 | 4 | 3/4 | 81.427354 +/- 0.762478 | 84.707859 +/- 0.467981 | 3.872728% | 3/3 |
| c4 | 60M | 2048 | 400M | 200 | 2 | 30/30 | 66.056620 +/- 1.137437 | 66.298782 +/- 1.491717 | 0.365259% | 2/3 |

## C4 60M/L2048

The selected 400M configuration is sigma 200 with two final HIPE layers (six preceding RoPE layers). Its primary PPL is **66.056620 +/- 1.137437**, versus RoPE **66.298782 +/- 1.491717**, a **0.365259%** lower selected mean. It improves in two of three paired seeds.

At the final checkpoint, the same configuration gives 66.053765 +/- 1.141383 versus RoPE 66.299857 +/- 1.498941 (0.371181% lower). Four of the 30 C4 60M/L2048 configurations win at both endpoints: sigma 200/500 combined with H1/H2.

The selected 100M result used sigma 500/H2, whereas the 400M selection uses sigma 200/H2. Their best-of-budget values are descriptive outcomes with different selected settings; exact fixed-configuration contrasts are reported separately.

## Exact configuration changes versus 100M

The plan has 72 exact old/new configuration pairs. **71 are complete**, and all 71 have lower mean PPL with the larger budget at both endpoints. At primary validation, reductions range from 32.47-50.37% at L1024 and 64.10-77.57% at L2048. This improvement includes baselines, so it should not be attributed to HIPE alone.

Sixteen new L1024 layouts have no exact 100M counterpart. Their budget-change entries remain unavailable. The matched C4 20M/L2048 sigma700/H4 pair is also unavailable while its missing seed is unresolved.

Primary checkpoint fractions differ for old versus new long budgets (old steps 1500/700, new step 3000). Final checkpoints compare the full intended schedules. Both views and actual token counts are retained; these are not identical-endpoint-token or compute-matched comparisons.

## All public baselines: primary endpoint

| Corpus | Model | L / budget | Legacy NoPE+APE | ALiBi | RoPE | Selected HIPE | Lowest available mean |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| wikitext103 | 20M | 512 / 100M | 60.181882 +/- 0.729248 | 50.159828 +/- 0.534017 | 50.677000 +/- 0.374751 | 49.284134 +/- 0.165303 | hipe |
| wikitext103 | 60M | 512 / 100M | 46.552683 +/- 1.882486 | 35.756719 +/- 0.312725 | 39.345951 +/- 0.398817 | 39.155005 +/- 0.406353 | alibi |
| c4 | 20M | 512 / 100M | 153.470813 +/- 0.578573 | 122.942177 +/- 0.113978 | 124.842452 +/- 3.809679 | 119.773068 +/- 1.263198 | hipe |
| c4 | 60M | 512 / 100M | 123.350728 +/- 7.472727 | 87.509025 +/- 0.547090 | 97.392336 +/- 1.723046 | 97.153268 +/- 0.897829 | alibi |
| wikitext103 | 20M | 1024 / 200M | 49.834819 +/- 0.681375 | 39.888309 +/- 0.239723 | 41.401038 +/- 0.484441 | 39.744355 +/- 0.240410 | hipe |
| wikitext103 | 60M | 1024 / 200M | 36.451339 +/- 0.187594 | 28.541706 +/- 0.129543 | 31.920506 +/- 0.086634 | 31.501384 +/- 0.263119 | alibi |
| c4 | 20M | 1024 / 200M | 124.492851 +/- 2.011382 | Original-shape OOM | 99.559194 +/- 1.871036 | 95.359672 +/- 0.513158 | hipe |
| c4 | 60M | 1024 / 200M | 100.783138 +/- 1.175921 | 67.494321 +/- 0.294546 | 78.590441 +/- 2.082565 | 77.937090 +/- 1.994418 | alibi |
| wikitext103 | 20M | 2048 / 400M | 44.416164 +/- 0.432002 | Original-shape OOM | 35.506994 +/- 0.329114 | 34.444745 +/- 0.100000 | hipe |
| wikitext103 | 60M | 2048 / 400M | 31.485797 +/- 1.056810 | 24.171343 +/- 0.066381 | 26.738622 +/- 0.564018 | 26.465979 +/- 0.472097 | alibi |
| c4 | 20M | 2048 / 400M | 109.896917 +/- 1.652704 | Original-shape OOM | 84.707859 +/- 0.467981 | 81.427354 +/- 0.762478 | hipe |
| c4 | 60M | 2048 / 400M | 84.499641 +/- 2.998060 | Original-shape OOM | 66.298782 +/- 1.491717 | 66.056620 +/- 1.137437 | hipe |

## Interrupted tasks

| Task | Configuration | Missing seed | Retained state | Statistical treatment |
| --- | --- | ---: | --- | --- |
| 42169_110 | C4 20M, L2048/400M, sigma700/H4 | 7 | Full checkpoint at step1000, 131072000 tokens | n=2; aggregate unavailable; excluded from selection |
| 42169_111 | C4 60M, L1024/200M, sigma300/H1 | 6198 | No checkpoint or validation metrics | n=2; aggregate unavailable; excluded from selection |

Both tasks ended NODE_FAIL on cccl2 at 13:41:39 on 6 November 2026 (Asia/Shanghai). The underlying hardware cause is not established. The first has an exact-resume checkpoint; the second requires a fresh same-seed restart. No recovery or additional training was launched during this analysis. Original failure records and partial data remain intact.

The incomplete C4 20M sigma700/H4 group has an n=2 final mean slightly below the selected complete sigma800/H4 group. It remains excluded; available-seed means must not be substituted for the required three-seed comparison.

## Actual exposure and interpretation

Every study setting has 3051 nominal optimizer steps (primary step3000). All available methods within each dataset/model/length setting match actual training and validation exposures. The small WikiText shortfalls from nominal final token counts are explained by partial micro-batches at epoch boundaries.

| Corpus | Model | L | Budget | Primary prediction tokens | Final prediction tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| wikitext103 | 20M | 512 | 100M | 98304000 | 99975168 |
| wikitext103 | 60M | 512 | 100M | 98304000 | 99975168 |
| c4 | 20M | 512 | 100M | 98304000 | 99975168 |
| c4 | 60M | 512 | 100M | 98304000 | 99975168 |
| wikitext103 | 20M | 1024 | 200M | 196606976 | 199949312 |
| wikitext103 | 60M | 1024 | 200M | 196606976 | 199949312 |
| c4 | 20M | 1024 | 200M | 196608000 | 199950336 |
| c4 | 60M | 1024 | 200M | 196608000 | 199950336 |
| wikitext103 | 20M | 2048 | 400M | 393148416 | 399833088 |
| wikitext103 | 60M | 2048 | 400M | 393197568 | 399882240 |
| c4 | 20M | 2048 | 400M | 393216000 | 399900672 |
| c4 | 60M | 2048 | 400M | 393216000 | 399900672 |

Matched optimizer steps do not imply matched tokens, FLOPs, wall time or unique data. WikiText repeats a fixed corpus at the longer budgets. Original method micro-batches/backends remain; the C4 HIPE/RoPE micro-batch difference is still a qualification.

Results are exploratory and validation-selected across unequal numbers of method configurations. Three seeds and conditional sample SD do not establish statistical significance or independent test superiority.

## Provenance and public scope

The full experiment source was committed and pushed before GPU submission: 12ebc719d980d8dd4fdf7a7bef6cf2ead40719b6, tag research/token-budget-steps-v1-20261105. Fresh runs used a full-budget cosine schedule; completed 100M checkpoints were not resumed.

Public outputs contain HIPE, RoPE, ALiBi and the explicitly labeled Legacy NoPE+APE baseline. Supplementary comparisons remain local. Historical results, failures and Git snapshots are preserved.
