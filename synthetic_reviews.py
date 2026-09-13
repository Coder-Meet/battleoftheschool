"""Export candidate training examples from manifest-backed analytic synthetic references."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from detector import Detection, DetectorConfig, detect, detect_pool
from evaluate import evaluate_case
from learning import FEATURE_NAMES, features, write_json
from nifti_io import read_nifti


def label_candidates(
    result: Detection, reference: dict, tolerance_mm: float = 3, exclusion_mm: float = 5,
) -> tuple[list[dict], dict]:
    if not np.isfinite(exclusion_mm) or exclusion_mm <= tolerance_mm:
        raise ValueError("Exclusion distance must be finite and larger than matching tolerance.")
    case_id = reference["case_id"]
    metrics = evaluate_case(result.prediction(case_id), reference, tolerance_mm)
    matched = {match["prediction_id"]: match["reference_id"] for match in metrics["matches"]}
    expected = np.asarray([branch["ostium_xyz_mm"] for branch in reference["daughters"]])
    rows = []
    ambiguous = []
    for branch in result.branches:
        reference_id = matched.get(branch.instance_id)
        if reference_id is None and len(expected):
            distance = np.linalg.norm(expected - branch.ostium_xyz_mm, axis=1).min()
            if distance <= exclusion_mm:
                ambiguous.append(branch.instance_id)
                continue
        vector = features(branch)
        rows.append({
            "case_id": case_id, "instance_id": branch.instance_id,
            "label": "confirmed" if reference_id is not None else "rejected",
            "labeller": "analytic_synthetic_geometry", "reference_instance_id": reference_id,
            "features": vector,
            "fingerprint": json.dumps([
                branch.ostium_xyz_mm, branch.seed_xyz_mm, branch.direction_xyz, branch.radius_mm, vector,
            ]),
        })
    return rows, {**metrics, "excluded_ambiguous_candidate_ids": ambiguous}


def export_reviews(
    data_root: Path, output: Path, profile: str = "pool",
    tolerance_mm: float = 3, exclusion_mm: float = 5,
) -> dict:
    if output.exists():
        raise FileExistsError("Reviews already exist; choose a new output to preserve provenance.")
    if profile not in ("strict", "pool"):
        raise ValueError("Profile must be strict or pool.")
    if not np.isfinite(tolerance_mm) or tolerance_mm <= 0:
        raise ValueError("Matching tolerance must be positive and finite.")
    if not np.isfinite(exclusion_mm) or exclusion_mm <= tolerance_mm:
        raise ValueError("Exclusion distance must be finite and larger than matching tolerance.")
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("source") != "analytic_synthetic_geometry":
        raise ValueError("Only analytic synthetic references can supply automatic training labels.")
    records, cases = [], []
    identifiers: set[str] = set()
    for entry in manifest["cases"]:
        case_id = entry["case_id"]
        if case_id in identifiers:
            raise ValueError(f"Duplicate synthetic case ID: {case_id}")
        identifiers.add(case_id)
        directory = data_root / case_id
        required = {"orig.nii.gz", "mask.nii.gz", "reference.json"}
        if not required.issubset(entry["files"]):
            raise ValueError(f"Missing input hashes for {case_id}.")
        for filename, digest in entry["files"].items():
            if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Dataset integrity check failed: {case_id}/{filename}")
        reference = json.loads((directory / "reference.json").read_text())
        if reference["case_id"] != case_id:
            raise ValueError("Reference and manifest case IDs must match.")
        image = read_nifti(str(directory / "orig.nii.gz"))
        mask = read_nifti(str(directory / "mask.nii.gz"))
        result = detect_pool(image, mask) if profile == "pool" else detect(image, mask)
        rows, report = label_candidates(result, reference, tolerance_mm, exclusion_mm)
        records.extend(rows)
        cases.append({
            **report, "family": entry.get("family"), "input_sha256": entry["files"],
            "runtime_s": result.timings["total_s"],
        })
        print(f"{case_id}: {len(rows)} labelled, "
              f"{len(report['excluded_ambiguous_candidate_ids'])} ambiguous, "
              f"{report['false_negatives']} unproposed references", flush=True)
    payload = {
        "schema_version": 1, "scope": "candidate_reviews_only", "feature_names": FEATURE_NAMES,
        "label_source": "analytic_synthetic_geometry", "profile": profile,
        "detector_config": asdict(DetectorConfig.review() if profile == "pool" else DetectorConfig()),
        "matching_tolerance_mm": tolerance_mm, "negative_exclusion_mm": exclusion_mm,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("synthetic_reviews.py", "detector.py", "evaluate.py", "learning.py")
        },
        "limitations": [
            "Synthetic candidate labels, not human annotations or real-scan accuracy.",
            "Unmatched candidates within the exclusion distance are ambiguous and excluded from training.",
            "Unproposed references remain misses in cases; candidate classification cannot recover them.",
            "Keep generation seeds disjoint between training, validation and final evaluation.",
        ],
        "cases": cases, "records": records,
    }
    write_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", choices=("strict", "pool"), default="pool")
    parser.add_argument("--tolerance-mm", type=float, default=3)
    parser.add_argument("--exclusion-mm", type=float, default=5)
    args = parser.parse_args()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    try:
        export_reviews(args.data_root, args.output, args.profile, args.tolerance_mm, args.exclusion_mm)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
