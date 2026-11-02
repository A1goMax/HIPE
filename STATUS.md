# HIPE Project Status

Updated 2 November 2026, Asia/Shanghai.

## Completed

- Selected HIPE as the journal research identity and the existing `HIPE/` repository as the canonical research root.
- Established the project plan, workstreams, stage gates, architecture, manuscript outline and evidence protocols.
- Registered the user-supplied 19-page manuscript as MS01 with a file hash; distinguished it from the 20-page server manuscript and the unverified OpenReview file.
- Enumerated the MS01 table targets and figure families in the experiment registry. Reported values are reference data, not new measurements.
- Recorded the earlier audit sources, including the PE branches, worktrees and separate experiment checkouts.
- The earlier audit indexed 769 log files; 37 contain recognized Git-hash fields. Those fields still need interpretation before any exact run-to-code claim.

The existing HIPE package is inherited from commit `bd1d53a6d905e2d8a9e2320852eb7fc2079cf004`. Its supported interface currently exposes the main fixed-sigma bipartite method. This planning step does not change its training or model behavior.

## Validation state

The earlier PE consolidation checkout received a deep-local-attention integration and a release import-path fix. Focused selection and syntax checks passed there. Those changes are staging evidence and have not been promoted into this package.

No paper-scale reproduction campaign has been launched in this work. A journal result has not been established by the planning metadata.

Observed environment issues from the initial audit: the `autoresearch` environment lacks PyTorch; importing the PE model in `recurrent` failed through NumPy/PyArrow imports. These observations do not establish the condition of every available environment. The next validation step is an isolated, recorded environment and a small end-to-end check.

## Next actions

| Priority | Action | Evidence required to finish |
| --- | --- | --- |
| 1 | Recover a representative 20M WikiText baseline/HIPE pair | Exact source/config/data/evaluation association, or a bounded statement of what remains unknown |
| 2 | Establish the journal runtime and numerical checks | Import checks, small forward/backward run, finite scales/gradients, checkpoint reload and evaluation agreement |
| 3 | Resolve method semantics across the packaged and PE implementations | Comparison of spectral scaling, normalization, hierarchy, data chunking, optimization and PPL calculation |
| 4 | Locate biological simulation and synthetic-task sources | Versioned generators, parameters, readout and analysis mapping |
| 5 | Estimate pilot resources and freeze the initial campaign | Measured throughput/memory, run list, seeds, selection rules and expected GPU hours |

## Decisions still open

Journal venue, final paper title, author responsibilities, final transfer-study scope, compute allocation, and GitHub owner/name/visibility. The reference manuscript is pinned for recovery; the relation between the supplied file and the OpenReview version is not yet verified.
