# HIPE Result Analysis

Use python -m hipe.report --runs-root PATH to summarize completed runs. The report keeps individual seeds, mean/std PPL, source identity, full scientific protocol and evaluation fingerprints. It rejects duplicate seeds within a protocol.

The default last_interval endpoint follows the historical validation cadence. The final endpoint is a separately available additional evaluation. The report recomputes PPL from summed token loss and prediction-token count and retains every sigma configuration rather than silently selecting a best result.

Publication plots and final table rendering remain later analysis tasks. Input runs, sigma/checkpoint selection and any exclusions must be recorded. Incomplete runs and SLURM/worker failures retain their own status records.
