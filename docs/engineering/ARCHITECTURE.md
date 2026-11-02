# HIPE Shared Experiment Architecture

Independent configuration controls select the positional method, spectral schedule, fixed/learnable scale and attention connectivity. Main experiments and ablations share training and evaluation.

| Module | Responsibility |
| --- | --- |
| `hipe/config.py` | Shared decoder, method, scale schedule and local-layer list |
| `hipe/spec.py` | Serializable execution settings and batch profiles |
| `hipe/data.py` | Named preprocessing and selected-data fingerprints |
| `hipe/train.py` | Optimization, accumulation, evaluation and resumable checkpoints |
| `hipe/evaluate.py` | Evaluation from saved model and data configuration |
| `hipe/provenance.py` | Recoverable source snapshot, runtime and tokenizer identity |
| `hipe/experiments.py` | Suite expansion and one/two-GPU SLURM execution |
| `hipe/report.py` | Explicit endpoint and protocol-separated seed aggregation |
| `configs/suites/` | Main and ablation specifications |
| `configs/machines/` | Machine paths and scheduler resources |
| `scripts/slurm_worker.sh` | Thin SLURM worker wrapper |

The inherited decoder structure remains under `hipe/olmo`. Integration edits focus on spectral scales, positional compatibility and causal local masks.

A run connects resolved config, recoverable source, data/tokenizer identities, token exposure, checkpoints and metrics. Existing output directories are rejected. Resume verifies source fingerprint, config and processed data.

Two GPUs run independent single-GPU conditions. The per-condition batch remains unchanged. Production training requires SLURM and an RTX 5090. Submission previews resolve settings without calling sbatch.
