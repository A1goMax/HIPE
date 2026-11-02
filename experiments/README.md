# HIPE Experiment Organization

`registry.json` is the research inventory. Each record names a scientific question or a reported table cell, its manuscript source, its current evidence status and the next action. Reported values are preserved as reference data. No registry entry constitutes a newly measured result.

## Experiment families

| Family | Purpose |
| --- | --- |
| `BIO` | Population-code theory and learning simulations |
| `SYN` | Synthetic associative and block retrieval |
| `LM` | Conference restricted-budget language modeling |
| `ABL` | Conference layer-schedule ablations |
| `MECH` | Kernel and trained-attention analyses |
| `JRN` | Proposed journal hypotheses and extensions |

An experiment ID identifies a scientific condition. A run ID identifies one execution of that condition with a specified seed and source snapshot. Failed, incomplete and resumed executions keep their own records. An analysis lists its input run IDs and its selection/aggregation rule.

`manifest_template.json` specifies the intended new-run record. It is a template, not an executed run. Required settings still marked null must be resolved by a future collector/launcher. `configs/reference/conference_20260129.json` is a paper specification; it is not a runnable trainer configuration.

## Protocols

- [Reference recovery](protocols/reference_recovery.md): recover and compare the conference evidence.
- [Journal validation](protocols/journal_validation.md): prospective research, paired comparisons and analysis rules.

The registry contains 48 Table 2 cells, including four reported OOM outcomes, and eight Table 3 rows. The Table 3 RoPE and bipartite HIPE rows explicitly reuse Table 2 conditions. Actual run counts depend on recovered settings, selected seeds and campaign decisions.
