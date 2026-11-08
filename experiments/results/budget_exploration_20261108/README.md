# Matched-step budget exploration: result snapshot

262 of 264 new runs completed; two ended NODE_FAIL. The study also reuses
63 L512/100M runs. The workbook has six verified views: selected complete HIPE
versus baselines, every study HIPE configuration, fixed-configuration changes
versus 100M, all statistics, exact seed observations, and attempts/coverage.

Best complete HIPE means beat RoPE in 12/12 study settings and 31/36 paired
seeds. Lowest available public method: HIPE in seven settings, ALiBi in five.
The original all-100M result stays at 11/12. This exploration combines
L512/100M, L1024/200M and L2048/400M; it does not replace the paper-budget table.

Incomplete n=2 configurations retain visible seed values but null aggregate
mean/SD and no selection eligibility. Partial step-1000 metrics do not count
as completed endpoints. Of 72 exact 100M control pairs, 71 are complete and
all improve. Sixteen new layouts have no exact 100M counterpart.

REPORT.md gives every selected configuration, all baseline results, actual
exposures, limitations and the two interruptions. PNG/SVG figures show selected
gains and exact budget contrasts. Verification covers raw calculations, XLSX
caches, sources/configs and failures. Native Excel recalculation was not exercised.

Training commit: 12ebc719d980d8dd4fdf7a7bef6cf2ead40719b6.
Study tag: research/token-budget-steps-v1-20261105.
Result snapshot: research/budget-results-v1-20261108.
Public methods are HIPE, RoPE, ALiBi and Legacy NoPE+APE.
Supplementary comparisons stay local; historical snapshots remain intact.
No recovery or new training was submitted during this analysis.
