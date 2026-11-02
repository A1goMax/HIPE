# HIPE Journal Research Plan

**Working project name:** HIPE — Hippocampus-Inspired Multiscale Representation Learning
**Project code:** HIPE
**Research direction:** From hippocampal population codes to sample-efficient artificial learning
**Planning baseline:** 2 November 2026, Asia/Shanghai
**Status:** Research foundation established; historical validation and journal studies pending

HIPE is a research program connecting a biological principle, a mathematical account of learning, and a computational method. Its journal version should establish when multiscale representations improve learning, explain the mechanism, and define the limits of the effect. Recovering earlier experiments supplies the reference evidence; the journal contribution must go beyond recovering the existing tables.

The original manuscript title remains **The Hippocampal Place Field Gradient: A Bio-inspired Framework Building Multiscale Representation for Better Sample Efficiency**. The working project name organizes the research and software. It does not replace the manuscript title or commit the authors to a journal or a final title.

## Scientific objective

Test the proposition that the distribution of representational scales creates a useful inductive bias whose benefits depend on task structure and the amount of available data. Establish the connection in three settings: grid-to-place population codes, controlled learning tasks, and Transformer positional representations.

The journal paper should answer four questions.

1. How accurately does the spectral construction produce the intended place-field and positional kernels under finite dimensions and discrete frequencies?
2. When does a multiscale code reduce the data needed to reach a given level of performance, and when does precise single-scale coding become preferable?
3. Which part of HIPE causes an observed advantage: spectral shape, normalization, the arrangement of scales across layers, or another implementation or optimization difference?
4. How well do the findings transfer across task scales, data budgets, model sizes, and evaluation conditions?

Positive, null, and adverse results all inform the scope of the theory. A journal result is not defined by matching a particular reported decimal or making HIPE win every comparison.

## Reference evidence and version policy

The user-supplied local PDF is reference **MS01**: a 19-page anonymous conference manuscript with PDF creation metadata dated 29 January 2026. Its SHA256 is `f291aaa4ead54f01d8c25b10c82ebdb1ff62ee23a55ae6542aa95f91cdaa1d3f`. It contains two Transformer experiment groups and three main tables. The PDF metadata refers to ICML 2026; metadata alone does not establish publication status.

The existing server PDF is **MS02**, a distinct 20-page manuscript with named authors, May 2026 PDF metadata, and a different experiment structure. Its additional scaling, adaptation, and attention experiments are candidate evidence for the journal program. They are not requirements stated by MS01. The OpenReview URL supplied earlier is recorded separately as **MS03**; equality with either local file has not been established.

Every experiment and manuscript claim will name its source version. The complete source register is in `manuscript/sources/manifest.json`. References below to tables and figures mean MS01 unless stated otherwise. Reported numbers are reference targets, not newly measured results.

## Research workstreams

| Workstream | Scientific or engineering question | Required outputs | Completion condition |
| --- | --- | --- | --- |
| Theory and biological model | What follows from the spectral construction, and under which assumptions? | Checked derivations; finite-spectrum numerical comparisons; biological assumptions and predictions | Assumptions, boundary effects, approximation error, and counterexamples are explicit |
| Controlled learning | Does task scale determine useful field scale and the precision versus generalization trade-off? | Recovered simulations; sample-size and task-scale sweeps; synthetic retrieval results | Figures can be regenerated from registered runs with uncertainty estimates |
| Unified implementation | Can all comparisons share a transparent and consistent code path? | One model/config interface; explicit method choices; compatible training and evaluation; environment specification | Relevant numerical, gradient, data, checkpoint, and small-run checks pass |
| Historical evidence | Which source, configuration, data, and evaluation produced each earlier result? | Run and artifact index; evidence links; reproduction of conference tables | Every retained numerical claim has traceable evidence or an explicit unresolved status |
| Journal validation | Are the advantages robust, mechanistically supported, and useful beyond one budget? | Budget curves; matched controls; seed analysis; staged transfer studies | Primary hypotheses have a completed analysis, including failures and limitations |
| Manuscript and release | Can a reader inspect the reasoning and regenerate the evidence? | Journal draft; generated figures/tables; archived source and artifact manifests | Every empirical manuscript item links to a frozen experiment and analysis version |

These are workstreams, not separate code forks. Junrong Qi is the project contact. Scientific authorship, collaborator responsibilities, and journal selection remain author decisions; this plan assigns no collaborator obligations.

## Staged roadmap

| Stage | Work | Deliverable | Exit gate |
| --- | --- | --- | --- |
| Foundation | Register manuscript versions, select the research root, define the evidence and configuration structure | This plan, source register, experiment registry, architecture and decision record | Sources and the role of each repository are unambiguous |
| Evidence recovery | Inspect branches, dirty checkouts, data preparation, logs, checkpoints, and environments | Cell-to-run mapping and method-equivalence report | One representative result can be traced end to end; remaining gaps are listed |
| Reference validation | Integrate the required methods and reproduce MS01 figures and tables in priority order | Conference reference report with regenerated outputs and discrepancies | Core claims have repeated evidence; mismatches are explained or retained as unresolved |
| Journal investigation | Run the primary data-budget and mechanism studies, then selected extensions | Evidence for the journal hypotheses and measured compute costs | Primary comparisons and predeclared analyses are complete |
| Robustness and synthesis | Test important failure cases, additional seeds and selected transfer conditions | Robustness report and claim scope | Conclusions reflect the full tested range and the independent units of replication |
| Manuscript and release | Write the journal paper from registered evidence and freeze the implementation | Submission draft, artifact release and reproducibility instructions | Figures/tables regenerate and manuscript claims pass an evidence review |

This is a dependency plan rather than a calendar promise. GPU-hour and elapsed-time estimates will be made from representative pilots and actual queue conditions. Full experiment campaigns should follow those estimates; the number of configurations alone is not a compute estimate.

## Conference reference campaign

The registry enumerates the numerical table cells and the figure families rather than treating the paper as a single training run.

| Reference | Required coverage | Important unresolved details |
| --- | --- | --- |
| Figures 2 to 8 and Appendix derivations | Grid-to-place weights, contextual learning, optimal scale, heterogeneous codes and task spectrum | Locate original simulation sources and recover neuron counts, grids, sampling, readout, regularization and seeds |
| Table 1 | One-to-one and block retrieval; RoPE versus HIPE; 20,000 training steps; HIPE sigma 200 and 50 respectively | Recover exact model, task distributions, sequence settings, batch size, seed and checkpoint selection |
| Table 2 | WikiText-103 and C4; 20M and 60M; lengths 512, 1024 and 2048; NoPE, ALiBi, RoPE and HIPE; 100M training-token budget | Recover per-cell sigma, preprocessing, evaluation population, optimizer and seeds |
| Table 3 | WikiText-103 20M, length 2048; RoPE, uniform, reverse, continuous and bipartite schedules | Resolve one-based layer labels versus zero-based implementation indices and exact scale assignments |
| Figure 1 performance panel and Figure 9 | Scale selection and attention patterns associated with the learning claims | Identify underlying runs and plotting choices; quantify patterns in addition to showing examples |

Table 2 has 48 method/dataset/model/length cells: 44 numerical entries and four reported ALiBi OOM entries. The latter are observations about the reported implementation and hardware, not universal impossibility claims. A new implementation should report its measured memory and execution outcome. Reused baseline cells in Table 3 should reuse the same registered evidence rather than incur duplicate training accidentally.

Historical reproduction and prospective replication are different records. If an original seed or configuration cannot be recovered, a new run may test the scientific claim, but it cannot be described as an exact recreation of that historical run.

## Proposed journal contribution

The primary extension is a **quantitative account of sample efficiency supported by mechanism controls**. The budget and mechanism studies form the central campaign; transfer studies follow only after their pilot results and costs are available.

| Hypothesis | Proposed experiment | Main endpoint |
| --- | --- | --- |
| H1 Finite spectra explain effective representation scale | Compare the discrete kernel with its analytical target across field widths, dimensions and frequency grids | Kernel approximation error, effective width and boundary behavior |
| H2 HIPE changes the amount of data needed for a given quality | Matched RoPE/HIPE learning curves across several token and unique-data budgets | Validation loss and tokens to reach a prespecified quality threshold; resource cost reported separately |
| H3 Hierarchy and spectral shape contribute beyond a change in attention magnitude | Compare sharp, broad, uniform, reversed and bipartite schedules with controlled normalization and tuning effort | Paired performance differences and quantitative representation diagnostics |
| H4 Task scale predicts useful field scale | Synthetic tasks with controlled distance/region structure, followed by carefully selected LM conditions | Relationship between task scale and validation-selected field scale on held-out evaluation |
| H5 The effect has a defined transfer range | Stage selected model-scale, corpus or downstream extensions after primary studies | Effect size, uncertainty and documented failure cases under new conditions |

Suggested pilot token budgets are 10M, 30M and 100M. A 300M or 1B-token extension is conditional on the pilot findings and measured costs. These are proposed journal settings, not settings recovered from the conference paper. Unique training examples, unique tokens and total token exposures must be recorded separately. Equal token budget, equal optimizer steps and equal wall-clock budget answer different questions.

For prospective studies, start with the paired pilot seed list `[6198, 1024, 7]`; consider `[568, 3427]` for the primary confirmatory comparisons after examining pilot variance and costs. This prospective choice does not assert that MS01 used those seeds. Freeze the evaluation population and selection rule before confirmatory runs. Use validation for model and sigma selection, and reserve an independent evaluation set for final claims where the data allow it.

## Implementation and evidence organization

The canonical research root is `/home/qijunrong/03-proj/HIPE` on local branch `journal/research-foundation`. Its existing `hipe/` package supplies the initial method implementation. The branch receives planning and evidence infrastructure first; it does not imply that the earlier release already reproduces every paper experiment.

The `PE` repository, its branches/worktrees, four separate experiment checkouts, and the earlier `PE_reproducibility` staging checkout are historical sources. Their relevant behavior will be integrated into the canonical package after comparison. They will not become permanent runtime dependencies. Exact legacy execution may remain available for historical comparison while the shared implementation is validated.

One experiment registry connects claims to protocols, configurations, runs, checkpoints, metrics and final figures. Each new run must identify its source commit, any dirty snapshot, full resolved configuration, data/tokenizer identities, environment, precision and attention backend, seeds, scheduler job, training budget and evaluation rule. Large artifacts live outside Git; the source tree holds manifests and hashes. See `docs/engineering/ARCHITECTURE.md` and `experiments/manifest_template.json`.

## Immediate priorities and open decisions

1. Recover a representative WikiText 20M RoPE/HIPE pair and verify its data, model, training and evaluation semantics against both existing implementations.
2. Establish an isolated environment and complete a small end-to-end run, checkpoint reload and evaluation comparison. Existing environment import failures are recorded in `STATUS.md`.
3. Resolve the finite-frequency interpretation of the spectral formula before claiming that its sigma-to-zero limit exactly equals standard RoPE. This needs a mathematical decision, not a silent code adjustment.
4. Locate the population-code simulation sources and the task generators behind the synthetic results.
5. Measure pilot costs, then prioritize the primary journal campaign and candidate transfer studies.

The venue, journal manuscript title, final extension scope, compute allocation and eventual GitHub destination remain open decisions. GitHub publication follows preparation of the research repository and selection of owner, name and visibility. There is no automatic push from this planning step.

## Definition of a journal submission candidate

The research is ready for author review when its contribution extends the conference reference, the core claims have reproducible evidence and an honest tested scope, the important controls have been completed, historical and new results are clearly distinguished, and all numerical tables and plots can be regenerated from their manifests. The manuscript must state assumptions, uncertainty, computational costs and failure cases at the points where they affect its conclusions.
