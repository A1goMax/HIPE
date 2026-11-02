# HIPE Research Architecture

The journal project has one canonical source repository and explicit experiment configurations. The initial code is the existing HIPE package; migration into it must preserve the meaning of historical experiments and distinguish revised methods from earlier ones.

## Current implementation

| Component | Current path | Current role |
| --- | --- | --- |
| Package | `hipe/` | Fixed-sigma bipartite method through the public interface |
| Model construction | `hipe/config.py` | 20M/60M/300M shapes and the hierarchical split |
| Model implementation | `hipe/olmo/` | Vendored decoder and positional representations |
| Data | `hipe/data.py`, `hipe/prepare_*.py` | Dataset loading, tokenization and chunking |
| Training | `hipe/train.py` | Current single-method training entry point |
| Evaluation | `hipe/evaluate.py` | Current strict checkpoint loading and LM evaluation |
| Launchers | `scripts/` | Environment/path setup and inherited method examples |

The public interface's limited scope does not mean the vendored file has no other legacy code. Reachability and numerical behavior need inspection. The current package has not yet been shown equivalent to the table-producing PE implementations.

## Target module boundaries

Retain the `hipe` package name and a shared train/evaluate interface. Introduce a new module boundary only when moving or implementing the associated behavior; the following are planned responsibilities, not already implemented APIs.

| Responsibility | Planned boundary | Rule |
| --- | --- | --- |
| Positional method | Spectral scale calculation, layer schedule and explicit baseline selection | Store normalization, frequency grid, sigma and layer indices in resolved configuration |
| Decoder model | OLMo integration and attention backend | Baseline and HIPE comparisons share the decoder except the intended intervention |
| Task/data adapters | LM datasets, synthetic retrieval and population-code tasks | Record split/sample identities and version the generator |
| Trainer | Shared budget accounting, optimization, checkpoint and resume | Distinguish tokens, unique data, optimizer steps and elapsed compute |
| Evaluator | Task-specific metrics using a frozen evaluation population | Store token-weighted loss before exponentiating to PPL |
| Experiment execution | Config resolution, manifests, run directories and scheduler integration | Fail on unresolved required settings and avoid overwriting previous runs |
| Analysis | Aggregation, uncertainty, tables and plots | Use run identities rather than hand-entering final values |

Baselines such as RoPE, NoPE and ALiBi must be explicit choices. A method-specific Git branch is historical source, not an experiment configuration. New controls and ablations should use the same interface after semantic validation. YaRN, XPos, learnable scales and local/global attention are candidate integrations whose scientific roles must be stated.

## Migration procedure

1. Freeze source identities: legacy commit, dirty diff, relevant untracked source files, configuration and output evidence.
2. Compare the old implementation, the existing HIPE package and the intended mathematical method. Record differences before adapting behavior.
3. Add each required feature as a named configuration in the canonical implementation. Preserve the exact legacy variant as an identified reference where needed.
4. Verify scales, layer indices, rotations, attention masks, gradients, checkpoint state and evaluation on small controlled cases.
5. Compare a representative paired training run, with the same data population and budget definition. Investigate discrepancies before expanding the campaign.
6. Replace legacy script dependence in the validated experiment specification. Keep source history and provenance links.

Do not copy all old scripts into a new execution path or merge conflicting model files by choosing one branch wholesale. The earlier `PE_reproducibility/unified` fixes are candidates for integration, not certified replacements for this package.

## Evidence and storage model

The evidence chain is:

`claim -> experiment -> resolved config -> run -> checkpoint and metrics -> analysis -> manuscript figure or table`

Track small specifications and manifests in Git. Store datasets, tokenizer files, checkpoints, raw logs, source PDFs and generated products in ignored artifact storage. The machine-specific locations belong in a local profile or source register, not hard-coded into training logic.

For each new run, collect the source repository and full commit, a recoverable source snapshot if dirty, exact command/config, seed and sample ordering, package/environment identity, CUDA/driver/GPU and precision settings, attention backend, dataset and tokenizer identity, split/sample fingerprints, budget and actual exposures, scheduler job identity, checkpoint selection, evaluation rule, output paths and hashes. `experiments/manifest_template.json` defines the planning contract; an automatic collector remains to be implemented.

A checkpoint filename is not a provenance record. A dataset path is not a dataset identity. File manifests should state whether identity is established by content hashing, upstream revision, ordered sample IDs, or a bounded alternative for very large artifacts.

## Validation and release

Keep planning validation independent of the ML runtime. ML validation proceeds from imports to numerical checks, task fixtures, checkpoint reload/resume and short paired runs. Large jobs use the server scheduler and carry the same experiment identity through submission, execution and analysis.

The final repository should have one maintained mainline, short-lived integration branches and versioned release tags. Conference-reference and journal-candidate snapshots can be tagged when their evidence is ready. GitHub owner/name/visibility and release timing remain separate decisions; this local planning branch has not been pushed.
