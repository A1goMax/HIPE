# HIPE Method Experiment Inventory

The current registry is based on MS04, the latest 21-page PDF. It contains 78 reported targets: Table 1 has four method/task conditions; Table 2 has 48 cells; Table 3 has eight larger-model conditions; Table 4 has twelve downstream conditions; Table 5 has six attention conditions.

Reported values are reference evidence, not new results. Table 2 includes four historical OOM entries whose new outcomes must be measured. Biological simulations are outside the current scope. Table 6 schedules have code support but are not part of the requested Tables 1-5 campaign.

Runnable suite specifications live under configs/suites/. Use python -m hipe.experiments for planning and SLURM submission. The current model/trainer supports Table 2 and Table 5 configurations; remaining task adapters and exact larger-model specifications are tracked in STATUS.md.

The earlier protocol documents are retained with their explicit older manuscript source IDs. The current implementation decisions and compatibility findings are in docs/engineering/BRANCH_INTEGRATION.md.

Each execution writes its own run_manifest.json and configuration, data/source identities, checkpoint and evaluation records. experiments/manifest_template.json is an earlier planning template; actual run manifests are authored by hipe/provenance.py and hipe/train.py.
