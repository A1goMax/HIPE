# HIPE Conference Reference Recovery

Source: MS01, the user-supplied 19-page conference manuscript. The task is to recover what was done and determine which claims a fresh implementation supports. Reported performance is a comparison target, not a training acceptance threshold.

## Recover the evidence

For each registry record, locate its likely launchers, model implementation, data preparation, logs, checkpoints and plotting code. Preserve the full commit and relevant dirty/untracked source snapshot. Record whether each association is verified, inferred or unknown and cite the evidence that supports that label.

The existing log index provides search leads. The explicit March history linking WikiText scripts to logs `exp2_2457` and `exp2_baselines_2458` is useful but does not alone establish the executed working tree. Commit fields can also refer to evaluation rather than pretraining. A filename or modification date cannot resolve the association by itself.

## Recover a complete specification

- Task or corpus, exact splits and ordered evaluation population.
- Tokenizer revision/files, vocabulary, special tokens, concatenation, chunk boundaries and padding/label masking.
- Model shape, RoPE frequency grid, scaling function and normalization, sigma and layer schedule, indexing convention and attention backend.
- Initialization, optimizer, learning-rate/warmup schedule, clipping, batch/accumulation, precision, seed and stopping budget.
- Checkpoint selection, validation frequency, token weighting, PPL computation and any test-set use.

If an item cannot be recovered, record it as unknown rather than replacing it with a convenient default. Proposed replacement settings belong in a separate prospective replication specification.

## Compare implementations

Check the packaged HIPE method against the relevant PE source before treating it as equivalent. Preprocessing and evaluation changes can alter PPL independently of positional embeddings. Compare a fixed small sample and checkpoint where feasible, then a short paired run. Classify differences as intentional revisions, defects or unresolved behavior.

## Numerical evaluation

For language modeling, retain the summed negative log likelihood and the exact number of prediction tokens. Compute aggregate PPL by exponentiating their ratio; do not average batch PPL values. Record whether the score is final-step, best-validation or another specified checkpoint.

For retrieval, specify token-level or sequence-level accuracy, target positions, sample count, loss normalization and task generator. The manuscript reports 20,000 training steps; several inherited launchers have different defaults, so defaults cannot substitute for recovered settings.

For population-code figures, recover task scales, neuron counts, random feature generation, readout, regularization and repeat units. Match the plotted uncertainty definition. Figures 5 and 7 report 1,000 repeats; do not silently equate these repeats with language-model training seeds.

## Report discrepancies

Show the paper value, recovered historical value if available, fresh estimate, independent replicate count and protocol differences. Where the reference provides no variation, exact matching of rounded decimals is not a defensible universal criterion. Interpret discrepancies with the recovered protocol, observed variance and practical effect size.

Record OOM, non-finite loss, early stop, missing checkpoints and incomplete provenance as distinct outcomes. An old ALiBi OOM does not require a new implementation to fail. Report memory/backend/hardware alongside any efficiency comparison.
