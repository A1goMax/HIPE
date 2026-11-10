# HIPE code and experimental evidence archive

Snapshot date 10 November 2026.

The [modern evidence index](modern/README.md) covers 803 public-method tasks
across 11 campaigns. It includes 585 run directories, 541 completed research
runs, and 24 completed GPU checks. Each observed run links its resolved config,
runtime and data manifest, exact saved source ZIP, metrics, logs and available
completion record. Cancelled tasks and failures retain their recorded status.

| Study | Results and code associations |
| --- | --- |
| Original Table 2 and L2048 sigma sweep | [Archived numerical results](../../experiments/results/table2_sigma_20261103/) |
| Joint sigma and layer search | [Paper-scope workbook and source index](../../experiments/results/table2_joint_20261104/) |
| Wider C4 sigma and layer search | [Workbook, exact seeds and run index](../../experiments/results/table2_c4_20261105/) |
| Matched-step token-budget exploration | [Workbook, budget contrasts, provenance and failures](../../experiments/results/budget_exploration_20261108/) |

[Figures](modern/figures/) accompany the workbooks. The all-100M results and
matched-step budget exploration keep separate protocols and endpoints.

[Legacy source revisions](legacy/source/README.md) preserve reviewed source
files from 18 public relative_PE revisions. The [historical result index](legacy/results/README.md)
contains 593 numerical evidence records in the five manuscript experiment
families. The [recorded-commit crosswalk](legacy/results/recorded-source-links.json)
links 313 rows to their recorded HEAD source. The other 280 have unknown
execution revisions. Dirty-state flags and inferred associations remain
separate from the verified modern run snapshots.

Clone the research branch or download this tag for offline browsing. From
the repository root, run the following command to check the modern archive.

    python scripts/verify_research_archive.py research_archive/20261110/modern --repo .

The modern public projection follows the paper's Table 2 method set.
The workstation archive contains the supplementary XPos comparisons.
Model and optimizer filenames and sizes are indexed for reference.
