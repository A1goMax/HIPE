# HIPE Analysis Organization

Analysis should consume registered runs and write figure/table products with their input run IDs, source revision and aggregation rule. Analysis code for the journal campaign has not yet been integrated here.

Planned outputs include population-code learning curves, discrete-kernel comparisons, retrieval metrics, token-weighted LM perplexity, data-budget curves, paired effect estimates, attention-distance summaries and measured compute/memory profiles.

For a manuscript table or figure, store an analysis manifest containing the contributing runs, exclusions and reasons, checkpoint rule, evaluation population, uncertainty method, plot settings and output hashes. Generated manuscript values should come from these records rather than edits to the reference numbers in the experiment registry.

Keep individual seed outcomes visible. Preserve failed and censored outcomes separately from successful measurements, and do not combine different protocols in an aggregate. A historical reproduction report should show the reported reference alongside recovered and newly executed results.
