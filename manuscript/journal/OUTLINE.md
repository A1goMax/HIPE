# HIPE Journal Manuscript Outline

Working scope: a mechanistic account of multiscale representations and sample-efficient learning. This is an argument plan; results and final section emphasis follow the evidence.

## Introduction and contribution

State the precision versus generalization problem and why representational scale is relevant. Distinguish the biological hypothesis, the mathematical construction and the engineering method. Explain the extension beyond the conference reference through finite-spectrum analysis, quantitative budget curves and controlled mechanism tests.

## Grid to place construction

Present the mapping and assumptions, then show numerical agreement with the derivation. Include finite-module/frequency effects and boundary cases. Describe anatomical statements as predictions at the level supported by the evidence.

## Representation scale and learning

Define the contextual task, population code and readout. Show sample-size/task-scale trade-offs, the predicted optimal scale and heterogeneous-code comparisons. State when the analytical approximation applies and where it fails.

## Positional representation in Transformers

Define the actual discrete HIPE construction, its normalization and layer schedule. Resolve the distinction between unweighted RoPE, frequency-density compensation and the sigma-to-zero statement. Give the computational cost and the actual supported attention implementations.

## Controlled mechanism evidence

Use synthetic retrieval and interventions to test the intended mechanism. Combine attention/representation summaries with task outcomes and matched controls. Avoid interpreting a selected attention image as a causal demonstration.

## Sample efficiency across data budgets

Present learning curves with independent run uncertainty, matched evaluation and fair tuning. Report the quality threshold and measured token savings where supported; show conditions in which the target is not reached or the advantage vanishes. Separate total exposures from unique training data and compute cost.

## Transfer and scope

Include only validated extensions selected from additional corpora, model scales and downstream adaptation. Explain which observations support a general principle and which are restricted to a particular architecture or regime. Keep later historical evidence and newly executed results distinguishable.

## Discussion and limitations

Relate the theoretical and empirical findings without treating the Transformer as an empirical model of hippocampal anatomy. Discuss finite spectra, nonlinearities, task assumptions, temporal dynamics, tuning/search costs and failure cases. Limit the conclusions to tested claims.

## Methods and evidence availability

Specify datasets, generators, splits, tokenization, optimizers, normalization, schedules, budgets, independent replicates, uncertainty, checkpoint selection and compute. Connect every numerical table and plot to an experiment, run set, analysis revision and source release.
