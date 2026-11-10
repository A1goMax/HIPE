# HIPE

Shared positional-embedding code for the latest manuscript's main experiments and ablations. **Table 2 is the first target.** The active source is the 21-page PDF registered as MS04; the requested method scope is Tables 1-5.

Code is authored in `E:\research\03\HIPE` and synchronized to `/home/qijunrong/03-proj/HIPE`, branch `journal/research-foundation`. The [portable code and results archive](research_archive/20261110/README.md) is on the research branch.

## One experiment interface

Preview a paired Table 2 condition without submitting a job:

```bash
python -m hipe.experiments submit --suite table2 \
  --dataset wikitext103 --model-size 20M --seq-len 512 \
  --method rope hipe --sigma 200 --seed 6198 --array --gpus 2
```

The preview prints the SLURM command and resolved batch summary. Add `--execute` when actually submitting. One GPU runs one condition at a time. Two GPUs run independent conditions concurrently; each retains its original batch. The launcher does not use DDP.

Inspect resolved settings:

```bash
python -m hipe.experiments plan --suite table2 \
  --dataset wikitext103 --model-size 20M --seq-len 512 \
  --method rope hipe --sigma 200 --seed 6198 --output pair.json
```

Preview all six attention ablations:

```bash
python -m hipe.experiments submit --suite table5 --gpus 2
```

The machine profile uses the `debug` partition, original Python environment and external data paths. Actual experiments require SLURM and an allocated RTX 5090, with one or two GPUs per job.

## Shared implementation

`hipe/config.py` combines positional method (`rope`, `hipe`, `nope`, `alibi`, `xpos`), sigma schedule (bipartite, uniform, reverse, explicit), fixed/learnable sigma and attention placement (global, local-global, global-local). Main experiments and ablations use the same trainer and preprocessing protocol.

The fixed HIPE formula is checked against PE commit 060b6a6. Learnable HIPE preserves the source branch's per-head normalization and weight floor. Explicit schedules use `null` for unscaled RoPE and positive scales for HIPE; zero is not a RoPE alias.

## Historical compatibility

Global batch is 64. Named profiles preserve recovered micro-batch rules; see [branch integration](docs/engineering/BRANCH_INTEGRATION.md). The main Table 2 suite uses one fixed sigma per condition and three seeds: 144 runs. The old grid is retained separately as table2_sweep for historical reference.

Legacy NoPE/XPos also added learned absolute-position embeddings. Compatibility runs retain this explicitly and carry `legacyAPE` in their identities. `--correct-nope` disables the legacy addition for selected NoPE/XPos conditions.

Some C4 launchers specify 200M tokens while the paper states 100M. The Table 2 suite follows the paper budget. Exact historical run-to-table association remains unresolved.

## Evidence and results

Each run stores a recoverable source snapshot, resolved experiment/model configuration, runtime, tokenizer/data identities, evaluation records, resumable checkpoint and final weights. Existing directories are rejected; resume verifies source/config/data identity.

```bash
python -m hipe.evaluate --checkpoint /path/to/run/model.pt
python -m hipe.report --runs-root /path/to/runs --endpoint last_interval --output summary.json
```

The report keeps individual seeds, mean and standard deviation, separates protocols, and rejects duplicate seeds. It distinguishes last-interval validation from an additional final evaluation and does not silently select a best sigma.

## Research records

- [Current status](STATUS.md)
- [Focused project plan](PROJECT_PLAN.md)
- [Branch comparison](docs/engineering/BRANCH_INTEGRATION.md)
- [Experiment registry](experiments/registry.json)
- [Manuscript sources](manuscript/sources/manifest.json)

`planning/` is the earlier local archive; its old manuscript scope is superseded. Legacy release launchers remain for interface history. Use the suite-driven commands for the new campaign.

## Main campaign execution

The main campaign uses seeds 6198, 1024 and 7 and the paper's 100M-token cap. Use --array to request exactly one GPU per SLURM array task; --gpus 2 then caps concurrent tasks at two. Execution uses a frozen Git checkout. Add --preflight for a labeled two-step small-data GPU check.

## Campaign correction

The original baseline portion finished with 93 successes and 15 ALiBi OOM failures. The redundant HIPE sweep was held, then replaced by 36 fixed-setting HIPE runs. The recovery suite table2_remaining additionally retries the six unexpected length1024 ALiBi failures with the restored automatic SDPA policy and the directly supported WikiText micro-batch16. It contains42 tasks but adds only36 missing main conditions; six are retries. All keep global64 and the 100M-token cap.

## Completed Table 2 analysis

The main, joint sigma/layer and wider C4 campaigns have ended. See
[the latest C4 extension report](analysis/table2/C4_EXTENSION_REPORT.md).
The verified workbook and all numerical/figure outputs are in
outputs/table2_c4_20261105/. The code-linked public result archive is
experiments/results/table2_c4_20261105/.

All 60 latest full runs completed. Best C4 60M/L2048 HIPE is sigma 500/H2:
242.321885 +/- 7.337535 versus RoPE 241.550843 +/- 12.911822.
No new configuration beats RoPE's mean; the overall best-observed count
remains 11/12 settings. The final-checkpoint view selects the same configuration
and retains the higher HIPE mean. These are validation-selected results.

Current public outputs follow the paper's Table 2 method set. Supplementary
comparison results remain local; historical snapshots are preserved.

## Joint sigma/layer search

The previous code and results are preserved at Git tag
[research/table2-sigma-v1-20261103](https://github.com/A1goMax/HIPE/tree/research/table2-sigma-v1-20261103).
The new [bounded protocol](experiments/protocols/table2_sigma_layers_xpos_v1.md)
defines 45 new HIPE runs across the three mean-loss settings. Nine completed HIPE control runs are reused. Total decoder
depth remains eight; two, four or six final layers use HIPE.

Use the shared launcher with --suite table2_sigma_layers_xpos --method hipe --array --gpus 2
--require-published. Commit and push before --execute. XPos defaults to the
TorchScale reference formulation without absolute embeddings; the inherited
implementation is explicitly xpos_variant=legacy. New run manifests include
effective layer layouts, code/config hashes and the verified publication receipt.

Public reports follow the paper Table 2 method set. XPos comparison results stay in ignored local outputs. Run the joint analysis normally for the public projection; --include-local-xpos explicitly recomputes the private local comparison. Historical Git snapshots are preserved.

## C4 60M/L2048 extension

The [completed bounded protocol](experiments/protocols/table2_c4_extension_v2.md)
sweeps sigma 1, 5, 10, 25, 50, 100, 200, 500 and 2000 at one/two HIPE layers,
and completes one-through-eight HIPE-layer coverage at sigma 50.
Four complete configurations are reused. The remaining 20 configurations
require 60 full runs across the original three seeds.

Use --suite table2_c4_extension_check --preflight for the four GPU checks,
then --suite table2_c4_extension --afterok CHECK_JOB_ID for the full array.
Both use --array --gpus 2 --require-published --execute after commit/push.
All full runs retain micro-batch 4, global batch 64 and the 100M-token cap.

## Exploratory length/token budgets

The [formal budget-exploration report](analysis/table2/BUDGET_EXPLORATION_REPORT.md)
covers L512/100M (63 reused runs), L1024/200M and L2048/400M.
Of 264 new tasks, 262 completed and two ended NODE_FAIL. Incomplete seed groups
remain visible and are excluded from selection. All means use three seeds at
one fixed budget/sigma/layer setting.

Best complete HIPE has lower mean PPL than RoPE in all 12 study settings;
the original all-100M experiment remains 11/12 and is kept separate.
Among available public methods, HIPE has the lowest selected mean in seven
settings and ALiBi in five. These are validation-selected exploratory outcomes.

The six-tab workbook, exact seeds, budget contrasts, figures and audits are
under outputs/budget_exploration_20261108/. The versioned public archive is
experiments/results/budget_exploration_20261108/.
Training source was published before submission at
research/token-budget-steps-v1-20261105 (12ebc71).
The result snapshot is research/budget-results-v1-20261108.
Supplementary comparisons stay local; historical snapshots are preserved.
