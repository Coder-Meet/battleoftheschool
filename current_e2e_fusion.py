"""Post-reference experiment: score strict/review separately, then deduplicate.

Consumes an unmodified current_e2e_audit output; does not alter production inference.
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from detector import Branch
from final_evaluation import (
    CASES, ROOT, digest, read_json, score_variant, write_json,
)
from learning import CandidateModel, features


def fuse(strict, review, model, threshold, priority="strict"):
    """Retain current 3 mm merge policy, with scoring before representation selection."""
    candidates = []
    for profile, detection in (("strict", strict), ("review", review)):
        for record in detection["branches"]:
            branch = Branch(**record)
            probability = float(model.scores([features(branch)])[0])
            if probability >= threshold:
                candidates.append((branch, probability, profile))
    if priority == "score":
        candidates.sort(key=lambda row: -row[1])
    kept, provenance = [], []
    for branch, probability, profile in candidates:
        if all(np.linalg.norm(np.subtract(branch.ostium_xyz_mm, old.ostium_xyz_mm)) >= 3 for old in kept):
            branch.instance_id = f"branch_{len(kept) + 1:03d}"
            kept.append(branch)
            provenance.append({"instance_id": branch.instance_id, "profile": profile, "score": probability})
    return {"parent": {"instance_id": "aorta"}, "daughters": [b.prediction() for b in kept]}, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = read_json(args.audit_dir / "provenance.json")
    for name in ("detector.py", "learning.py", "final_evaluation.py"):
        if digest(ROOT / name) != source["source_sha256"][name]:
            raise ValueError("Audit and current source differ")
    model_path = ROOT / "labels/research/current-source-v1/models/logistic.json"
    if digest(model_path) != source["models"]["logistics"]["synthetic"]["sha256"]:
        raise ValueError("Audit and model differ")
    model = CandidateModel.load(model_path)
    thresholds = sorted({0.05, model.threshold, 0.15, 0.3, 0.5, 0.7, 0.85})
    variants, receipts = {}, {}
    for priority in ("strict", "score"):
        for threshold in thresholds:
            key = f"{priority}@{threshold:.17g}"
            predictions = {}
            for case in CASES:
                paths = [args.audit_dir / "cases" / case / f"{profile}-diagnostics.json"
                         for profile in ("strict", "review")]
                strict, review = (read_json(path) for path in paths)
                for path in paths:
                    receipts[str(path.resolve())] = digest(path)
                prediction, origins = fuse(strict, review, model, threshold, priority)
                prediction["case_id"] = case
                predictions[case] = prediction
                write_json(args.output_dir / key / f"{case}.json", prediction)
                write_json(args.output_dir / key / f"{case}-decisions.json", {"origins": origins})
                # No branch coordinate or reference is used to fabricate a proposal.
                pool = [Branch(**row).prediction() for d in (strict, review) for row in d["branches"]]
                for daughter in prediction["daughters"]:
                    geometry = {k: v for k, v in daughter.items() if k != "instance_id"}
                    if not any(geometry == {k: v for k, v in row.items() if k != "instance_id"} for row in pool):
                        raise AssertionError("Fusion changed source branch geometry")
            variants[key] = score_variant(predictions)
    write_json(args.output_dir / "metrics.json", variants)
    write_json(args.output_dir / "provenance.json", {
        "status": "Adaptive post-reference development experiment; no production promotion.",
        "reason": "Current pool merges representations before scoring; test scoring before merging.",
        "model_sha256": digest(model_path), "model": asdict(model),
        "script_sha256": digest(Path(__file__)), "audit_inputs_sha256": receipts,
        "audit_provenance_sha256": digest(args.audit_dir / "provenance.json"),
        "priorities": ["strict", "score"], "thresholds": thresholds,
    })
    for key, value in variants.items():
        s = value["3"]["summary"]
        print(key, s["true_positives"], s["false_positives"], s["false_negatives"], s["f1"])


if __name__ == "__main__":
    main()
