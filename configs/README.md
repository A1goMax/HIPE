# HIPE Experiment Configuration

- suites/table2.json: restricted-budget main comparison and explicit sigma search.
- suites/table5.json: local/global attention comparisons through the same model/trainer.
- suites/schedules.json: uniform, reverse and continuous layer schedules, currently outside the requested run scope.
- machines/labserver2.json: server paths, original Python runtime and scheduler resources.
- reference/latest_hipe_tables.json: current paper specification, not directly runnable.
- reference/conference_20260129.json: historical MS01 specification, superseded for the current campaign.

Suite defaults and per-condition choices are resolved into a complete ExperimentConfig. A run stores these resolved settings rather than relying on mutable launcher defaults. Global batch is 64. Named micro-batch profiles record historical differences; GPU count does not change the per-condition batch.

The source comparison documents unresolved settings and intentional protocol choices. Use filtered submission previews before the measured SLURM pilot.
