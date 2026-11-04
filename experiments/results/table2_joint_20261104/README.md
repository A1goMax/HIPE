# Completed joint sigma/layer search and XPos comparison

Open HIPE_Table2_Joint_Search_Analysis.xlsx for the formal analysis.
The first view keeps every targeted sigma/layer setting separate and contrasts
its three-seed mean and individual seeds with RoPE. The baseline view includes
all 12 Table 2 settings and XPos. Both endpoints are kept separate.

Exact logged PPL decimal strings are preserved in the raw observations and CSV.
Display precision does not round the underlying data. Means and sample SDs
use exactly seeds 6198, 1024 and 7 at one fixed configuration.

Best observed HIPE has lower mean PPL than RoPE in 11/12 settings. Two settings
switch from a mean loss to a mean win. C4 60M/L2048 remains a mean loss.
The new C4 L512 wins improve in only one of three matched seeds.
These are exploratory validation-selected results, not significance claims.

REPORT.md explains results and material protocol differences.
Targeted_sigma_layers.png/.pdf shows seed variation and layer trends.
HIPE_vs_RoPE_overview.png/.pdf shows the before/after comparison.
Verification files record source/config checks, independent calculations,
figure checks and exported-workbook checks.

The full new experiment array41770 used published commit
5df7693e38ede95a9fa1747207b59fd2c701e75a.
The pre-search code/results are preserved at research/table2-sigma-v1-20261103.
This analysis is archived under research/table2-joint-results-v1-20261104.

No training was launched during this analysis.
