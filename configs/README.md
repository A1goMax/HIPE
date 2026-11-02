# HIPE Configuration Organization

`reference/conference_20260129.json` captures settings stated in manuscript MS01. Null fields identify settings still requiring recovery. It is a research specification and cannot be passed directly to the current trainer.

Runnable configurations will be added after method/data/training semantics are compared and validated. They should identify model shape, positional method and schedule, corpus/task, tokenizer, optimization, budget, seed and evaluation. Shared defaults must be resolved into each run manifest; a mutable default is not a historical record.

Separate configurations by evidence purpose: recovered conference reference, prospective replication, exploratory journal pilot, and confirmatory journal campaign. Keep their source and status explicit. Machine paths belong in local profiles; large data and outputs remain outside version control.
