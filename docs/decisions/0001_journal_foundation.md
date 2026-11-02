# HIPE Journal Foundation Decision

Date: 2 November 2026, Asia/Shanghai.
Status: Working project organization adopted locally.

## Context

The user requested a formal research program and publication structure. The earlier `PE_reproducibility` workspace addressed initial evidence recovery but did not describe the scientific scope. An existing `HIPE` package repository was then found at `/home/qijunrong/03-proj/HIPE`, with baseline commit `bd1d53a6d905e2d8a9e2320852eb7fc2079cf004`, method packaging, tests and release history.

The newly supplied conference PDF differs from the server PDF used in the first audit. Their experimental requirements cannot be treated as a single paper version.

## Decision

Use **HIPE — Hippocampus-Inspired Multiscale Representation Learning** as the working research identity and the existing `HIPE` repository as the canonical project root. Create local branch `journal/research-foundation` for the project plan, source register, experiment registry and journal structure.

Keep the inherited fixed-sigma method interface intact during the planning step. Make the original PE checkouts and the earlier audit workspace historical inputs to a documented migration. Keep MS01 conference targets separate from MS02 additional experiments and prospective journal studies.

## Consequences

The project gains a scientific identity, explicit contribution plan and one destination for maintained code. It still requires evidence recovery, environment validation and method integration; a formal directory structure does not complete those activities.

The existing HIPE repository has configured GitHub remotes. Preserve them as inherited configuration, but do not use their presence as authorization to publish. A future GitHub destination will be chosen with the user. No journal venue, new scientific result, collaborator obligation or publication status is established by this decision.
