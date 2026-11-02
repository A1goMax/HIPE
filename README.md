# HIPE

**Hippocampus-Inspired Multiscale Representation Learning**

HIPE connects hippocampal population coding, learning theory and Transformer positional representations. The journal research program asks when multiscale representations improve sample efficiency, what mechanism explains the effect, and where the advantage stops.

The existing manuscript title is **The Hippocampal Place Field Gradient: A Bio-inspired Framework Building Multiscale Representation for Better Sample Efficiency**. The research program is organized around its biological, theoretical and computational contributions.

## Start here

- [Project plan](PROJECT_PLAN.md): scientific questions, workstreams, stages and completion criteria.
- [Current status](STATUS.md): completed work, observed blockers and immediate priorities.
- [Claims and evidence](docs/science/CLAIM_EVIDENCE.md): the evidence required for each claim.
- [Experiment registry](experiments/registry.json): conference targets and proposed journal studies.
- [Architecture](docs/engineering/ARCHITECTURE.md): current implementation, target organization and migration rules.
- [Manuscript sources](manuscript/sources/manifest.json): versions and file identities.

## Repository organization

```text
HIPE/
  hipe/                    Existing method package and model implementation
  configs/                 Reference specifications and future resolved run configs
  experiments/             Registry, protocols and run manifest contract
  analysis/                Figure and table generation conventions
  manuscript/              Source register and journal manuscript outline
  docs/                    Scientific reasoning, architecture and decisions
  reproducibility/         Legacy source register and environment requirements
  scripts/                 Existing launchers and research-plan validation
  tests/                   Existing package tests and future equivalence checks
  third_party_licenses/    Upstream license notices
  PROJECT_PLAN.md           Research charter and staged roadmap
  STATUS.md                 Current state and next actions
```

Large datasets, checkpoints, logs, PDFs and generated outputs belong in external artifact storage. Their identities and evidence links belong in the repository. The project does not require a method-specific Git branch for each experiment.

## Current software scope

The initial package exposes fixed-sigma bipartite HIPE through `python -m hipe.train` and checkpoint evaluation through `python -m hipe.evaluate`. Model-size configurations include 20M, 60M and 300M. The journal registry includes additional methods and scientific tasks whose integration and validation are still pending.

The previous method-release README is preserved at [Method release](docs/engineering/METHOD_RELEASE.md), including its setup and invocation examples. Those commands document the inherited interface; they do not certify equivalence to the conference experiments. The current environment and validation status are recorded in `STATUS.md`.

Validate the planning metadata without installing the training dependencies:

```bash
python scripts/check_research_plan.py
```

## Publication scope

The journal campaign combines recovery of the conference evidence with new budget curves, finite-spectrum checks and controlled mechanism studies. Candidate scaling and downstream experiments are staged extensions. Their inclusion depends on the results and resources rather than their presence in a legacy branch.

The current working branch is `journal/research-foundation`. Existing source history and remote configuration are retained. The eventual GitHub destination and release are separate project decisions.
