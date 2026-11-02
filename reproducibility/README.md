# HIPE Evidence Provenance

Reproducibility is a workstream within the HIPE research program. It connects a scientific claim to the source, data, execution and analysis that support it.

`legacy_sources.json` records the source checkouts discovered so far. Its paths describe the current server inventory, not dependencies that the final package should import. Original repositories, branches and result files are retained while evidence is recovered.

The earlier audit at `/home/qijunrong/03-proj/PE_reproducibility` contains branch metadata, tracked diffs and a historical log index. Its filename inventory is not a backup of untracked contents. Relevant untracked scientific source must be captured before relying on a snapshot as complete. Avoid collecting credentials, private keys or unrelated machine configuration into the evidence archive.

## Environment evidence

The inherited `requirements.txt` and `pyproject.toml` specify dependencies but are not an exact environment lock. Establish a separate journal runtime and record versions, platform, CUDA/driver, optional attention backend and relevant determinism settings. Keep installation-time and execution-time records with their respective source revision.

Validate imports and a small forward/backward, checkpoint reload and evaluation before campaign training. Review preprocessing and evaluation semantics before interpreting a legacy result mismatch as a method failure. If a historical environment cannot be reconstructed, distinguish the new-runtime replication from historical replay.

## Source and artifact state

Preserve a recoverable code snapshot for each dirty execution. Include tracked changes and relevant untracked code; a dirty flag or patch hash without the corresponding contents is insufficient. Hash checkpoints and small artifacts directly where practical. For large datasets, record a source revision, preprocessing identity and ordered sample/split fingerprints with the fingerprint method stated.

Classify historical associations as verified, inferred or unknown. Register contradictory evidence and unresolved items instead of filling gaps from memory. Promotion to the canonical implementation requires documented equivalence checks or an explicit account of the scientific revision.
