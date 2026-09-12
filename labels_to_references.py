#!/usr/bin/env python3
"""
Turn confirmed candidate labels into reference JSONs so predictions can be scored on real scans.

    python labels_to_references.py --cases subject005 subject013 --output-dir labels/pseudo_references

Only candidates labelled "confirmed" become reference daughters; their geometry comes from the rendered
pool the label was made on, and the label's fingerprint must still match that candidate. These are
pseudo-references: they carry the labeller's judgement, not organiser truth, and every file says so.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from autolabel import fingerprint
from evaluate import validate_prediction
from explorer import ROOT


def build_reference(case_id: str, pool: dict, records: list[dict]) -> dict:
    candidates = {c["instance_id"]: c for c in pool["candidates"]}
    daughters: list[dict] = []
    stale: list[str] = []
    for row in sorted(records, key=lambda r: r["instance_id"]):
        if row["label"] != "confirmed":
            continue
        candidate = candidates.get(row["instance_id"])
        if candidate is None or fingerprint(candidate) != row.get("fingerprint"):
            stale.append(row["instance_id"])
            continue
        daughters.append({
            "instance_id": f"branch_{len(daughters) + 1:03d}", "parent_instance_id": "aorta",
            "ostium_xyz_mm": candidate["ostium_xyz_mm"], "seed_xyz_mm": candidate["seed_xyz_mm"],
            "radius_mm": candidate["radius_mm"], "direction_xyz": candidate["direction_xyz"],
            "source_candidate": row["instance_id"],
        })
    reference = {
        "case_id": case_id, "parent": {"instance_id": "aorta"}, "daughters": daughters,
        "provenance": {
            "kind": "pseudo_reference_from_candidate_labels",
            "labellers": sorted({r.get("labeller", "unknown") for r in records}),
            "confirmed": len(daughters), "rejected": sum(r["label"] == "rejected" for r in records),
            "pool_size": len(candidates), "stale_labels": stale, "pool_profile": pool.get("profile"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "limitations": [
                "Daughters are detector candidates a reviewer confirmed; branches the detector never proposed are absent.",
                "Geometry is the detector's own measurement, so ostium/radius/direction errors against it are lower bounds.",
            ],
        },
    }
    validate_prediction(reference)
    return reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reviews", type=Path, default=ROOT / "labels" / "reviews.json")
    parser.add_argument("--pool-dir", type=Path, default=ROOT / "outputs" / "autolabel")
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        records = json.loads(args.reviews.read_text())["records"]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for case_id in args.cases:
            pool_path = args.pool_dir / case_id / "pool.json"
            if not pool_path.is_file():
                raise FileNotFoundError(f"No rendered pool for {case_id}; run autolabel.py render first.")
            rows = [r for r in records if r["case_id"] == case_id]
            if not rows:
                raise ValueError(f"{case_id} has no labels.")
            reference = build_reference(case_id, json.loads(pool_path.read_text()), rows)
            (args.output_dir / f"{case_id}.json").write_text(json.dumps(reference, indent=2, allow_nan=False) + "\n")
            note = f" ({len(reference['provenance']['stale_labels'])} stale labels skipped)" if reference["provenance"]["stale_labels"] else ""
            print(f"{case_id}: {len(reference['daughters'])} reference daughters{note}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"labels_to_references: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
