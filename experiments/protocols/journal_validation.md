# HIPE Journal Validation Protocol

This is the prospective protocol outline for the journal campaign. Experiment configurations, endpoints and resource estimates will be frozen after evidence recovery and pilots, before confirmatory runs.

## Primary studies

Study JRN01 tests the finite-spectrum construction. Measure the kernel implied by the actual frequencies and scaling, compare it with the target profile, and report approximation error, effective width, normalization and numerical stability. Check what sigma zero means with the frequency-density compensation retained. Keep any revised method distinct from the historical implementation.

Study JRN02 measures sample efficiency using matched baseline/HIPE learning curves. Pilot 10M, 30M and 100M total training-token budgets while separately recording unique data and repeated exposures. Additional budgets depend on the pilot and resources. Primary metrics are held-out loss at matched exposures and the exposures needed to reach a quality threshold fixed before confirmatory analysis. If a method never reaches the threshold within the measured budget, record that outcome without extrapolating a success.

Study JRN03 isolates mechanism. Compare the hierarchy against uniform and reversed schedules, spectral-shape controls and matched normalization. Quantify attention distance or representation width across layers, and use task interventions that can challenge the proposed mechanism. More diffuse attention alone does not establish the source of a performance change.

Study JRN04 tests code-task scale alignment. Specify task scale independently of the field-scale search, choose scale on validation data and evaluate the selected model on held-out data. Keep search spaces and tuning budgets comparable.

Study JRN05 stages transfer to selected model sizes, corpora or downstream conditions. Legacy 300M, learnable-sigma, local/global, XPos, YaRN and SST-2 work supplies candidate evidence and implementation leads. None is automatically a mandatory journal result.

## Experimental controls

Pair conditions on data subsets/order and seed wherever the comparison permits. Match the decoder, tokenizer, splits, optimizer and intended budget. If architectures change the RNG consumption, identify the pairing unit explicitly rather than assuming an identical seed creates identical initialization everywhere.

Compare equal token exposure for the data-efficiency question and report measured runtime/memory separately. Treat equal time, equal compute and equal optimization steps as separate analyses. Include tuning cost when drawing a resource-efficiency conclusion.

Start prospective pilots with seeds `[6198, 1024, 7]`. Additional primary confirmation seeds `[568, 3427]` are a planning choice subject to pilot variance and resources. They are not a statement about the original manuscript. Changing the planned independent replicate count or seed selection requires a recorded reason, especially after seeing outcomes.

## Statistical and selection rules

The training run is the independent unit for between-run LM performance. Thousands of tokens or evaluation examples within one trained model do not create thousands of independent training replicates. Use paired differences when the pairing is valid and report individual run values, aggregate effect size and uncertainty. A small number of seeds limits precision and should be visible in the interpretation.

Use validation for hyperparameter and checkpoint selection. Keep a held-out evaluation population for final generalization claims where the dataset allows it. If a comparison is exploratory, say so. Predefine the primary contrasts; handle multiple comparisons for wider confirmatory claim families rather than selecting whichever test appears favorable.

Store failures and stopping reasons. Resume only with a documented checkpoint, optimizer/scheduler/RNG state and budget accounting; otherwise identify the new execution as a restart. Do not pool incompatible protocols in one mean.

## Campaign gate

Before a large job set, require a validated environment, a complete resolved configuration, a small end-to-end run, data/evaluation fingerprints and a measured cost estimate. The source revision and campaign manifest should identify the entire planned comparison, including controls. Submission readiness depends on the resulting evidence, not on a prespecified winning outcome.
