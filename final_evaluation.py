"""Shared scoring for the judge-approved five-case development reference release."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from detector import validate_geometry
from evaluate import evaluate_case
from nifti_io import read_nifti
from research_validation import seed_polyline_distance
from score_references import normalize_reference

ROOT = Path(__file__).resolve().parent
REFERENCE_ROOT = ROOT / "labels" / "organizer-v1"
CASES = tuple(f"subject{number:03d}" for number in range(19, 24))
TOLERANCES = (2.0, 3.0, 5.0)
THRESHOLDS = (0.15, 0.3, 0.5, 0.7, 0.85)
ERRORS = (
    "ostium_error_mm", "seed_error_mm", "radius_error_mm", "direction_error_degrees",
    "seed_to_reference_centreline_mm",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def case_paths(case_id: str, data_root: Path = ROOT / "data") -> tuple[Path, Path]:
    paths = [
        sorted((data_root / case_id).glob(pattern)) for pattern in ("orig*.nii*", "mask*.nii*")
    ]
    if any(len(matches) != 1 for matches in paths):
        raise ValueError(f"Need exactly one CT and parent mask for {case_id}.")
    return paths[0][0], paths[1][0]


def load_reference(case_id: str) -> dict:
    return read_json(REFERENCE_ROOT / "references" / f"{case_id}.json")


def score_prediction(prediction: dict, reference: dict, tolerance: float = 3.0) -> dict:
    normalized = normalize_reference(reference, [])
    result = evaluate_case(prediction, normalized, tolerance)
    original = {row["instance_id"]: row for row in reference["daughters"]}
    measured = {row["instance_id"]: row for row in normalized["daughters"]}
    predicted = {row["instance_id"]: row for row in prediction["daughters"]}
    for match in result["matches"]:
        identifier = match["reference_id"]
        if not measured[identifier]["_radius_known"]:
            match["radius_error_mm"] = None
        centerline = original[identifier].get("centerline_xyz_mm")
        match["seed_to_reference_centreline_mm"] = (
            seed_polyline_distance(predicted[match["prediction_id"]]["seed_xyz_mm"], centerline)
            if centerline is not None else None
        )
    return result


def summarize(rows: list[dict]) -> dict:
    tp, fp, fn = (
        sum(row[key] for row in rows)
        for key in ("true_positives", "false_positives", "false_negatives")
    )
    errors = {}
    for key in ERRORS:
        values = [
            match[key] for row in rows for match in row["matches"] if match.get(key) is not None
        ]
        errors[key] = {
            "n": len(values), "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None,
        }
    return {
        "cases": len(rows), "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "count_mae": float(np.mean([r["daughter_counts"]["absolute_error"] for r in rows])) if rows else None,
        "exact_count_cases": sum(r["daughter_counts"]["absolute_error"] == 0 for r in rows),
        "errors": errors,
    }


def score_variant(predictions: dict[str, dict]) -> dict:
    if set(predictions) != set(CASES):
        raise ValueError("A comparison variant must include all five cases, including empty predictions.")
    by_tolerance = {}
    for tolerance in TOLERANCES:
        rows = [score_prediction(predictions[case], load_reference(case), tolerance) for case in CASES]
        by_tolerance[f"{tolerance:g}"] = {"summary": summarize(rows), "cases": rows}
    return by_tolerance


def filtered(prediction: dict, scores: list[float] | np.ndarray, threshold: float) -> dict:
    values = np.asarray(scores, dtype=float)
    if values.shape != (len(prediction["daughters"]),) or not np.isfinite(values).all():
        raise ValueError("Scores must be finite and aligned with every candidate.")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1 or np.any((values < 0) | (values > 1)):
        raise ValueError("Scores and threshold must be probabilities.")
    output = copy.deepcopy(prediction)
    output["daughters"] = [
        row for row, probability in zip(output["daughters"], values) if probability >= threshold
    ]
    return output


def ingest(release: Path) -> dict:
    manifest = read_json(release / "manifest.json")
    expected = {str(i): count for i, count in zip(range(19, 24), (3, 4, 3, 6, 3))}
    if manifest["case_counts"] != expected:
        raise ValueError("Released counts differ from the user's supplied inventory.")
    for row in manifest["files"]:
        path = (release / row["path"]).resolve()
        if not path.is_relative_to(release.resolve()):
            raise ValueError("Manifest path escapes the release.")
        if digest(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError(f"Release hash/size mismatch: {row['path']}")
    checksum_rows = {}
    for line in (release / "SHA256SUMS.txt").read_text().splitlines():
        expected_hash, name = line.split(maxsplit=1)
        checksum_rows[name] = expected_hash
    if checksum_rows != {row["path"]: row["sha256"] for row in manifest["files"]}:
        raise ValueError("Checksum list and manifest disagree.")
    cases = []
    for number, case_id in zip(range(19, 24), CASES):
        image_path, mask_path = case_paths(case_id)
        directory = release / f"case_{number}"
        if digest(image_path) != digest(directory / f"orig{number}.nii.gz"):
            raise ValueError(f"CT identity mismatch: {case_id}")
        if digest(mask_path) != digest(directory / f"aorta{number}.nii.gz"):
            raise ValueError(f"Parent-mask identity mismatch: {case_id}")
        image, mask = read_nifti(str(image_path)), read_nifti(str(mask_path))
        validate_geometry(image, mask)
        raw = read_json(directory / "annotations.json")
        label_image = read_nifti(str(directory / f"daughters{number}_draft.nii.gz"))
        combined_image = read_nifti(str(directory / f"aorta_and_daughters{number}_draft.nii.gz"))
        validate_geometry(image, label_image > 0)
        validate_geometry(image, combined_image > 0)
        label_values = {d["label_value"] for d in raw["daughters"]}
        if set(np.unique(sitk.GetArrayViewFromImage(label_image))) != {0, *label_values}:
            raise ValueError("Daughter mask labels disagree with annotation IDs.")
        if set(np.unique(sitk.GetArrayViewFromImage(combined_image))) != {
            0, 1, *(value + 1 for value in label_values),
        }:
            raise ValueError("Combined mask must use parent=1 and daughter labels offset by 1.")
        if raw["coordinate_system"] != "SimpleITK physical LPS millimetres":
            raise ValueError("Unexpected coordinate convention.")
        if tuple(raw["shape_xyz"]) != image.GetSize() or not np.allclose(
            raw["spacing_xyz_mm"], image.GetSpacing(), atol=1e-6
        ):
            raise ValueError(f"Reference/image geometry mismatch: {case_id}")
        daughters = []
        for daughter in raw["daughters"]:
            landmarks = np.asarray([
                daughter["ostium_xyz_mm"], daughter["seed_xyz_mm"], *daughter["centerline_xyz_mm"],
            ])
            if not np.isfinite(landmarks).all() or landmarks.shape[1] != 3:
                raise ValueError("Invalid reference landmarks.")
            voxels = np.array([
                image.TransformPhysicalPointToContinuousIndex(point.tolist()) for point in landmarks
            ])
            if np.any(voxels < -0.5) or np.any(voxels > np.array(image.GetSize()) - 0.5):
                raise ValueError("Reference landmarks outside source CT.")
            physical = np.array([
                image.TransformContinuousIndexToPhysicalPoint(point)
                for point in daughter["centerline_voxel_xyz"]
            ])
            if not np.allclose(physical, daughter["centerline_xyz_mm"], atol=1e-5):
                raise ValueError("Reference voxel/physical guides disagree.")
            daughters.append({
                key: daughter[key] for key in (
                    "instance_id", "parent_instance_id", "ostium_xyz_mm", "seed_xyz_mm",
                    "radius_mm", "direction_xyz", "centerline_xyz_mm", "review_status",
                )
            })
        if len(daughters) != expected[str(number)]:
            raise ValueError("Reference count differs from manifest.")
        normalized = {
            "case_id": case_id, "coordinate_space": "LPS", "parent": raw["parent"],
            "daughters": daughters,
            "provenance": {
                "kind": "judge_approved_scoring_reference_from_ai_assisted_draft",
                "expert_signoff_in_original_package": False,
                "approval": "User confirmed: The judge has approved these as scoring references",
                "source_sha256": digest(directory / "annotations.json"),
            },
        }
        normalize_reference(normalized, [])
        write_json(REFERENCE_ROOT / "references" / f"{case_id}.json", normalized)
        cases.append({
            "case_id": case_id, "daughters": len(daughters),
            "known_radii": sum(d["radius_mm"] is not None for d in daughters),
            "image_sha256": digest(image_path), "mask_sha256": digest(mask_path),
            "size_xyz": image.GetSize(), "spacing_xyz_mm": image.GetSpacing(),
            "origin_lps_mm": image.GetOrigin(), "direction": image.GetDirection(),
            "nifti_geometry_metadata": {
                key: image.GetMetaData(key) for key in image.GetMetaDataKeys()
                if key.startswith(("qform", "sform", "srow", "qoffset", "quatern"))
            },
        })
    report = {
        "source_url": "https://drive.google.com/drive/folders/1GoIKqCKIMhtqzHNwLgtNkBhmxT_96ZR6",
        "manifest_sha256": digest(release / "manifest.json"),
        "all_manifest_files_verified": len(manifest["files"]),
        "published_checksum_list_matches_manifest": True,
        "judge_approval": "Confirmed by user in this session; original draft provenance retained.",
        "official_evaluator_supplied": False,
        "scoring": "Local maximum-cardinality one-to-one ostium matching at 2/3/5 mm.",
        "cases": cases,
    }
    write_json(REFERENCE_ROOT / "validation.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ingest-release", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(ingest(args.ingest_release), indent=2))


if __name__ == "__main__":
    main()
