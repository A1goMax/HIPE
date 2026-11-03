# Table 2 exploratory sigma/layer search and XPos baseline

The completed campaign is preserved at Git tag
[research/table2-sigma-v1-20261103](https://github.com/A1goMax/HIPE/tree/research/table2-sigma-v1-20261103)
(commit eb35933840e2ca732cc3ad58750a1e79357e52b0).
Its result archive links all previous attempts to their actual training commits.
This study starts after that snapshot.

## Bounded search

Keep the decoder depth at eight layers and vary how many final layers use HIPE.
The preceding layers use unscaled RoPE. A zero-based threshold of 5, 3 or 1
therefore gives 2, 4 or 6 HIPE layers, respectively.

| Previous mean-loss setting | Sigma candidates | HIPE layer counts | Completed control reused |
| --- | --- | --- | --- |
| WikiText-103 60M, L2048 | 500, 800 | 2, 4, 6 | sigma 800, four HIPE layers |
| C4 60M, L512 | 200, 500 | 2, 4, 6 | sigma 500, four HIPE layers |
| C4 60M, L2048 | 25, 50 | 2, 4, 6 | sigma 50, four HIPE layers |

The Cartesian grid has 18 configurations. Three already have complete seed
sets and are reused, leaving 15 configurations x three seeds = 45 new HIPE runs.
This keeps a small factorial design for sigma/layer interaction rather than
a wide search over every already-winning setting.

Add reference XPos in all 12 Table 2 corpus/model/length settings: 36 seed-runs.
The combined suite therefore contains **81 new full runs**, with at most two
running simultaneously. Short labeled GPU checks precede these runs and are
excluded from the research results.

Seeds remain 6198, 1024 and 7. Every full run uses the 100M-token cap, global
batch 64, original per-method micro-batches, optimizer, preprocessing and
validation populations. XPos uses the historical baseline/RoPE micro-batches.
C4 HIPE/RoPE micro-batch differences remain explicitly qualified.

## XPos definition and compatibility

Reference: [Microsoft TorchScale XPos](https://github.com/microsoft/torchscale/blob/4d1e0e82e5adf86dd424f1463192635b73fc8efc/torchscale/component/xpos_relative_position.py).
The reference source SHA256 is
d337ba30c5b837018839d6c442832fe12cf237f9d9b7422d53db7b20fd18b991.

The new baseline uses the authors' dimension-dependent decay, centered
position exponent divided by scale base 512, reciprocal query/key factors and
the decoder's split-half rotary coordinates. This is a fixed coordinate
permutation of the authors' adjacent-pair layout. Rotations and factors are
computed in float32 before casting to the attention dtype. Learned absolute
position embeddings are disabled.

The inherited implementation instead used unnormalized powers of a 0.95-1
decay and a clamped reciprocal. It remains explicitly selectable as
xpos_variant=legacy; it is not labeled as the reference XPos baseline.
Historical model configs without the new variant field retain that legacy
implementation. Existing runs should be evaluated from their frozen source.

## Selection and reporting

Primary endpoint: last regular validation. Final-checkpoint evaluations are
reported separately. Select the lowest complete three-seed mean for each
fixed configuration family; calculate sample SD at that same sigma/layer
setting. Never pool different layer counts or sigmas. Retain exact seed
values, every candidate, failures and tuning cost.

The search targets settings identified after observing previous validation
results. It is exploratory. A successful tuned mean does not establish
independent test superiority or statistical significance. Run this bounded
grid once; do not automatically add candidates after viewing outcomes.

## Version and execution contract

Commit and push the tested source to A1goMax/HIPE before submission. Use:

    python -m hipe.experiments submit --suite table2_sigma_layers_xpos --array --gpus 2 --require-published --execute

The publication gate verifies the exact source head on the remote branch.
Each campaign freezes that commit and hashes every resolved configuration.
Each run records the publication receipt, training revision, source archive
and core-file hashes, effective layer layout, dataset/tokenizer fingerprints,
runtime, SLURM IDs, checkpoints and metrics.

One GPU runs one independent training process. No DDP is used. All GPU work,
including short checks, must be submitted through SLURM.
