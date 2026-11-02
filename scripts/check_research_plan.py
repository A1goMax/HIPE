"""Validate HIPE research metadata without importing the training runtime."""

from pathlib import Path
import hashlib
import json
import re


ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    sources = read("manuscript/sources/manifest.json")
    source_map = {s["source_id"]: s for s in sources["sources"]}
    require(len(source_map) == len(sources["sources"]), "Duplicate source IDs")
    ms01 = source_map["MS01"]
    require(ms01["pages"] == 19, "MS01 page count changed")
    for source in source_map.values():
        digest = source.get("sha256")
        if digest is not None:
            require(re.fullmatch(r"[0-9a-f]{64}", digest), f"Invalid SHA256: {source['source_id']}")
        if source.get("artifact_path"):
            artifact = ROOT / source["artifact_path"]
            if artifact.exists():
                actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
                require(actual == digest, f"Artifact hash mismatch: {artifact}")

    registry = read("experiments/registry.json")
    rows = registry["records"]
    ids = {r["experiment_id"] for r in rows}
    require(len(ids) == len(rows), "Duplicate experiment IDs")
    table1 = [r for r in rows if r.get("paper_locator") == "Table 1"]
    table2 = [r for r in rows if r.get("paper_locator") == "Table 2"]
    table3 = [r for r in rows if r.get("paper_locator") == "Table 3"]
    require(len(table1) == 4, "Table 1 must have four method/task conditions")
    require(len(table2) == 48, "Table 2 must have 48 conditions")
    require(sum(r["reported_outcome"] == "oom" for r in table2) == 4, "Table 2 must retain four reported OOM cells")
    require(len(table3) == 8, "Table 3 must have eight schedule rows")
    conditions = {(r["dataset"], r["model_size"], r["sequence_length"], r["method"]) for r in table2}
    expected = {(d, m, length, method) for d in ("wikitext103", "c4") for m in ("20M", "60M") for length in (512, 1024, 2048) for method in ("nope", "alibi", "rope", "hipe")}
    require(conditions == expected, "Incomplete or duplicate Table 2 grid")
    by_id = {r["experiment_id"]: r for r in rows}
    for row in rows:
        sid = row.get("source_id")
        if row["study_type"] == "conference_reference":
            require(sid == "MS01", "Conference target uses a different source version")
        if sid:
            require(sid in source_map, f"Unknown source: {sid}")
            require(all(1 <= p <= source_map[sid]["pages"] for p in row["pdf_pages"]), "Invalid source page")
        if row.get("reuse_experiment_id"):
            parent = by_id.get(row["reuse_experiment_id"])
            require(parent is not None, "Unknown reused experiment")
            require(row["reported_metrics"] == parent["reported_metrics"], "Conflicting values in reused condition")
        if row.get("protocol"):
            require((ROOT / row["protocol"]).is_file(), "Missing study protocol")
    spec = read("configs/reference/conference_20260129.json")
    require(spec["source_sha256"] == ms01["sha256"], "Reference specification source mismatch")
    require(spec["record_type"] == "paper_specification_not_a_runnable_config", "Reference config must state its purpose")
    template = read("experiments/manifest_template.json")
    require(template["record_type"] == "template_not_an_executed_run", "Run template is mislabeled")
    for section in ("source", "configuration", "data", "runtime", "budget", "artifacts", "outcome"):
        require(section in template, f"Missing run manifest section: {section}")
    print(f"PASS: {len(rows)} unique experiments; source identities and available PDF hash; 4/48/8 table conditions; reuse links; protocols and manifest structure")


if __name__ == "__main__":
    main()
