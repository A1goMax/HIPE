# HIPE

Shared positional-embedding code for the latest manuscript's main experiments and ablations. **Table 2 is the first target.** The active source is the 21-page PDF registered as MS04; the requested method scope is Tables 1-5.

Code is authored in `E:\research\03\HIPE` and synchronized to `/home/qijunrong/03-proj/HIPE`, branch `journal/research-foundation`. GitHub publication remains a later decision.

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

## Completed Table2 analysis

The main campaign and requested L2048 high-sigma sweep have ended. See [the completed sweep report](analysis/table2/COMPLETED_SIGMA_REPORT.md). All 42 new runs completed; six matching previous runs complete the 48-run requested grid. The formal workbook and CSV/figure outputs are in outputs/table2_sigma_final/. HIPE has lower mean PPL than RoPE in 9/12 settings after selecting each setting's lowest complete three-seed mean. Sigma 800 is best observed in three L2048 settings; C4 60M retains sigma 50. ALiBi has the lowest available mean in six settings, HIPE in five and RoPE in one. These are validation-selected results. Different sigmas are never pooled. The earlier analysis/table2/REPORT.md is the pre-sweep record.
