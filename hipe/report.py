"""Aggregate explicitly selected validation endpoints without mixing protocols."""
import argparse
import json
import math
from pathlib import Path
import statistics


def selected_metrics(rows, endpoint):
    if endpoint == "last_interval":
        candidates = [r for r in rows if r["regular_validation"]]
    elif endpoint == "final":
        candidates = [r for r in rows if r["final"]]
    else:
        raise ValueError("Unknown endpoint")
    if not candidates:
        raise ValueError(f"No eligible {endpoint} evaluation")
    return max(candidates, key=lambda r: r["step"])


def summarize(runs_root, endpoint="last_interval"):
    groups = {}
    for manifest_path in sorted(Path(runs_root).rglob("run_manifest.json")):
        folder = manifest_path.parent
        if not (folder / "completion.json").exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        config = manifest["experiment"]
        row = selected_metrics([json.loads(s) for s in (folder / "metrics.jsonl").read_text().splitlines() if s], endpoint)
        # Seeding and storage paths vary; scientific and data/evaluation settings must agree.
        protocol = {k: v for k, v in config.items() if k not in {"seed", "output_dir", "experiment_id", "source_notes"}}
        identity = {"protocol": protocol, "source_fingerprint": manifest["source"]["fingerprint"],
                    "validation_fingerprint": manifest["data"]["processed_validation"], "endpoint": endpoint}
        key = json.dumps(identity, sort_keys=True)
        entry = groups.setdefault(key, {"identity": identity, "runs": []})
        if config["seed"] in {r["seed"] for r in entry["runs"]}:
            raise ValueError("Repeated seed for the same protocol; specify an adjudicated run set")
        loss = row["validation"]["loss_sum"] / row["validation"]["tokens"]
        entry["runs"].append({"path": str(folder), "seed": config["seed"], "step": row["step"],
                              "loss": loss, "ppl": math.exp(loss)})
    result = []
    for entry in groups.values():
        values = [r["ppl"] for r in entry["runs"]]
        entry.update({"n": len(values), "mean_ppl": statistics.mean(values),
                      "std_ppl": statistics.stdev(values) if len(values) > 1 else None})
        result.append(entry)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--endpoint", choices=["last_interval", "final"], default="last_interval")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = {"endpoint": args.endpoint, "sigma_selection": "all configurations; no implicit best-sigma selection",
              "groups": summarize(args.runs_root, args.endpoint)}
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
