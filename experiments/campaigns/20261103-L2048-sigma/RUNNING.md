# L2048 sigma sweep running

SLURM array **41695** contains42newruns, with6matchingcompletedruns reused. The full requestedgrid is sigma700/800/900/1000 forWikiText/C4,20M/60M andseeds6198/1024/7. Eachtask usesoneRTX5090, withatmosttwo concurrent;100Mtoken cap/globalbatch64 andoriginalmicro-batches unchanged.

The firstsix newseedruns completed in the launch snapshot. Current best-observed comparisons (only complete three-seed candidates) are in outputs/best_sigma_20261103/. They are provisional until the full sweep ends. Every candidate has a separate mean and sampleSD; the selected sigma minimizes the mean, never the best individual seed.

Dependent CPU-only SLURM job **41698** will collect the complete sweep and regenerate the comparison against the existing baselines automatically. Final server output: `/home/qijunrong/03-proj/HIPE/outputs/best_sigma_after_41695/`. It uses a frozen copy of the baseline input data.

Source and reused runs are recorded in LAUNCH.json andREUSE.json. Current comparison contains both last-regular-validation andfinal-checkpoint endpoints separately.
