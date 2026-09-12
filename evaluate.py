"""Local one-to-one ostium matching; requires actual daughter reference annotations."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def validate_prediction(prediction: dict) -> None:
    if not isinstance(prediction["case_id"], str):
        raise ValueError("case_id must be a string.")
    parent = prediction["parent"]["instance_id"]
    identifiers = set()
    for branch in prediction["daughters"]:
        identifier = branch["instance_id"]
        if not isinstance(identifier, str) or identifier in identifiers or identifier == parent:
            raise ValueError("Branch identifiers must be unique strings, distinct from the parent.")
        identifiers.add(identifier)
        if branch["parent_instance_id"] != parent:
            raise ValueError("Daughters must refer to the declared parent.")
        for key in ("ostium_xyz_mm", "seed_xyz_mm", "direction_xyz"):
            value = np.asarray(branch[key], dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"{key} must contain three finite numbers.")
        if not np.isclose(np.linalg.norm(branch["direction_xyz"]), 1, atol=0.01):
            raise ValueError("Daughter directions must be unit vectors.")
        if not np.isfinite(branch["radius_mm"]) or branch["radius_mm"] <= 0:
            raise ValueError("Daughter radii must be positive and finite.")


def evaluate_case(prediction: dict, reference: dict, tolerance_mm: float = 5) -> dict:
    if not np.isfinite(tolerance_mm) or tolerance_mm <= 0:
        raise ValueError("Matching tolerance must be positive and finite.")
    validate_prediction(prediction)
    validate_prediction(reference)
    if prediction["case_id"] != reference["case_id"]:
        raise ValueError("Prediction and reference case IDs must match.")
    predicted, expected = prediction["daughters"], reference["daughters"]
    matches = []
    if predicted and expected:
        distances = cdist(
            [p["ostium_xyz_mm"] for p in predicted], [r["ostium_xyz_mm"] for r in expected]
        )
        invalid_cost = (min(distances.shape) + 1) * tolerance_mm
        rows, columns = linear_sum_assignment(np.where(distances <= tolerance_mm, distances, invalid_cost))
        for i, j in zip(rows, columns):
            if distances[i, j] > tolerance_mm:
                continue
            p, r = predicted[i], expected[j]
            angle = np.rad2deg(np.arccos(np.clip(
                np.dot(p["direction_xyz"], r["direction_xyz"])
                / (np.linalg.norm(p["direction_xyz"]) * np.linalg.norm(r["direction_xyz"])), -1, 1
            )))
            matches.append({
                "prediction_id": p["instance_id"], "reference_id": r["instance_id"],
                "ostium_error_mm": float(distances[i, j]),
                "seed_error_mm": float(np.linalg.norm(np.asarray(p["seed_xyz_mm"]) - r["seed_xyz_mm"])),
                "radius_error_mm": abs(p["radius_mm"] - r["radius_mm"]),
                "direction_error_degrees": float(angle),
            })
    tp, fp, fn = len(matches), len(predicted) - len(matches), len(expected) - len(matches)
    return {
        "case_id": prediction["case_id"],
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
        "matches": matches,
    }


def summarize_cases(results: list[dict]) -> dict:
    tp, fp, fn = (sum(r[key] for r in results) for key in
                  ("true_positives", "false_positives", "false_negatives"))
    errors = {}
    for key in ("ostium_error_mm", "seed_error_mm", "radius_error_mm", "direction_error_degrees"):
        values = [m[key] for r in results for m in r["matches"]]
        errors[key] = {
            "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None,
        }
    return {
        "cases": len(results), "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
        "matched_errors_only": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--tolerance-mm", type=float, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = evaluate_case(
            json.loads(args.prediction.read_text()), json.loads(args.reference.read_text()), args.tolerance_mm
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    report = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
