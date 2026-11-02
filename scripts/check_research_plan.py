"""Validate the active paper inventory without the ML runtime."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = json.loads((ROOT / "manuscript/sources/manifest.json").read_text())
    reference = sources["reference_source_id"]
    assert reference == "MS04"
    source = next(s for s in sources["sources"] if s["source_id"] == reference)
    assert source["pages"] == 21
    artifact = ROOT / source["artifact_path"]
    if artifact.exists():
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == source["sha256"]
    registry = json.loads((ROOT / "experiments/registry.json").read_text())
    rows = registry["records"]
    assert len({r["experiment_id"] for r in rows}) == len(rows) == 78
    counts = {f"Table {i}": sum(r["paper_locator"] == f"Table {i}" for r in rows) for i in range(1, 6)}
    assert list(counts.values()) == [4, 48, 8, 12, 6]
    assert all(r["source_id"] == reference for r in rows)
    assert all(1 <= page <= 21 for r in rows for page in r["pdf_pages"])
    table2 = [r for r in rows if r["paper_locator"] == "Table 2"]
    assert sum(r["reported_outcome"] == "oom" for r in table2) == 4
    assert len({(r["dataset"],r["model_size"],r["sequence_length"],r["method"]) for r in table2}) == 48
    print("PASS: MS04 source; 78 targets; table counts 4/48/8/12/6; Table 2 grid and four reported OOM cells")


if __name__ == "__main__":
    main()
