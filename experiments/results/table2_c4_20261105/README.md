# Completed wider C4 search and current Table 2 results

Open HIPE_Table2_C4_Extension_Analysis.xlsx for the verified five-tab analysis.
HIPE vs RoPE ranks all 30 observed C4 configurations within each endpoint.
Best vs baselines updates all 12 Table 2 settings using the paper's method set.
All results calculates mean and sample SD once per fixed configuration.
Run data retains exact logged seed PPLs; Attempts preserves earlier failures.

All 60 new full runs completed. No new configuration beats RoPE's mean.
The best observed result is sigma 500 with two final HIPE layers:
242.321885 +/- 7.337535, compared with RoPE 241.550843 +/- 12.911822.
The overall best-observed Table 2 mean-win count remains 11/12.
The final-checkpoint view selects the same configuration and also trails RoPE.

Every mean and sample SD uses seeds 6198,1024,7 at one fixed sigma/layer setting.
Configurations are never pooled. The four GPU checks are excluded.
Tuned values are validation-selected; SD does not include selection uncertainty.
The original C4 micro-batch difference remains (HIPE 4, RoPE 8, global 64).

REPORT.md and the numerical CSVs contain every candidate and qualification.
C4_sigma_layer_effects.png/.svg shows the planned sigma/layer grid and seeds.
The verification files record raw reconstruction, formula/cache checks and
source/configuration checks. Native Excel recalculation was not exercised.

Training source: c44c207b83bf4140ec462bd727b1c5e6fc946613.
Pinned before execution: research/table2-before-c4-extension-v2-20261104.
Result archive tag: research/table2-c4-results-v2-20261105.
Supplementary comparison outputs stay local. Historical snapshots are preserved.
No training was launched during this analysis.
