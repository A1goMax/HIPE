# HIPE

This directory contains the single main method selected for the formal code release:
fixed-sigma bipartite HIPE.

HIPE is the implementation accompanying the paper:

**The Hippocampal Place Field Gradient: A Bio-inspired Framework Building Multiscale Representation for Better Sample Efficiency**

The public code path is intentionally limited to this method. It does not include
alternative positional encodings, ablations, exploratory schedules, historical
results, or pretrained checkpoints.

## Layout

- `hipe/olmo/`: the required decoder and rotary-position implementation.
- `hipe/config.py`: the three model-size configurations and the hierarchical split.
- `hipe/train.py`: the only training entry point.
- `hipe/evaluate.py`: the checkpoint evaluation entry point.
- `hipe/data.py`: dataset loading and deterministic text chunking.
- `scripts/`: local data preparation and main-method launchers.
- `third_party_licenses/`: license notices for vendored upstream components.

The supported model sizes are 20M, 60M, and 300M. The default hierarchy keeps
standard RoPE in the shallow half of the decoder and applies Gaussian HIPE in the
deep half. The corresponding default layer thresholds are 3 for the 8-layer models
and 7 for the 16-layer model.

## Installation

Use Python 3.10 or newer:

```bash
python -m pip install -r requirements.txt
```

For an editable local installation:

```bash
python -m pip install -e .
```

FlashAttention is optional. The main implementation can use PyTorch attention when
the optional CUDA dependency is not installed; paper-scale runs may enable it with
`FLASH_ATTENTION=1` in the launcher environment.

## Data contract

WikiText data should be materialized as:

```text
DATA_ROOT/wikitext/raw
DATA_ROOT/wikitext/tokenizer
```

The C4 launcher expects:

```text
DATA_ROOT/c4/c4_30M_train
DATA_ROOT/c4/c4_30M_validation
DATA_ROOT/wikitext/tokenizer
```

Raw datasets, tokenizer files, checkpoints, and training artifacts are external to
this source tree.

## Main method entry points

Prepare WikiText:

```bash
bash scripts/prepare_wikitext.sh
```

Run the fixed-sigma bipartite method on WikiText:

```bash
bash scripts/run_hipe_wikitext.sh
```

Run the fixed-sigma bipartite method on the processed C4 subsets:

```bash
bash scripts/run_hipe_c4.sh
```

All paths are controlled by `HIPE_DATA_DIR` and `HIPE_ARTIFACTS_DIR`.

Training writes the model state to `model.pt`, the resolved run configuration to
`run_config.json`, and evaluation records to `metrics.jsonl` under the selected
artifact directory. These generated files are intentionally excluded from the
source tree.

Evaluate a checkpoint with the same fixed-sigma HIPE configuration:

```bash
PYTHONPATH=. python -m hipe.evaluate \
  --checkpoint /path/to/model.pt \
  --data-path /path/to/wikitext/raw \
  --tokenizer-path /path/to/wikitext/tokenizer \
  --model-size 60M \
  --seq-len 1024 \
  --sigma 500 \
  --rope-scaling-threshold 3
```

The repository contains source code only. Dataset files, tokenizer files,
checkpoints, and training artifacts must be supplied externally and should not be
committed to the project.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{zhou2026hippocampal,
  title     = {The Hippocampal Place Field Gradient: A Bio-inspired Framework Building Multiscale Representation for Better Sample Efficiency},
  author    = {Shujun Zhou and Junrong Qi and Guozhang Chen},
  booktitle = {Proceedings of the International Conference on Machine Learning},
  year      = {2026}
}
```

The vendored OLMo components retain their original license notice in
`third_party_licenses/OLMo-LICENSE`.
