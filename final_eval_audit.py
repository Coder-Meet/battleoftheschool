"""Reproducible structural/protocol audit; never adjudicates or tunes the detector."""

import argparse
from collections import Counter
import copy
from dataclasses import asdict
import gzip
import importlib.metadata
from itertools import combinations, permutations
import json
from pathlib import Path
import platform
import struct
import subprocess
import sys
import time

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist
import SimpleITK as sitk

from detector import DetectorConfig, detect, validate_geometry
from final_evaluation import (
    CASES, ROOT, TOLERANCES, case_paths, digest, filtered, load_reference,
    read_json, score_prediction, score_variant, summarize, write_json,
)
from learning import FEATURE_NAMES, features
from nifti_io import read_nifti
from research_corpus import peak_mib, thread_limit
from score_references import mirrored, normalize_reference, score
from tabular_learning import TreeModel, json_sha256

OUTPUT = ROOT / "labels/final-eval/audit"
SOURCE_FILES = (
    "final_eval_audit.py", "tests/test_final_eval_audit.py", "final_evaluation.py",
    "score_references.py", "evaluate.py", "nifti_io.py", "detector.py", "run.py",
    "learning.py", "candidate_patches.py", "tabular_learning.py", "patch_inference.py",
    "patch_learning.py", "research_validation.py", "FINAL_EVALUATION_PROTOCOL.md",
    "requirements.txt", "requirements-dev.txt",
)
BASELINE_SHA256 = "9f96f3eb3c06f5e06d5b7db79b199be70e742130fc3885243d25d066adc1c92e"


def nifti_header(path: Path) -> dict:
    with path.open("rb") as stream:
        magic = stream.read(40)
    if magic.startswith(b"version https://git-lfs"):
        raise ValueError(f"Unresolved LFS pointer: {path}")
    opener = gzip.open if magic.startswith(b"\x1f\x8b") else open
    with opener(path, "rb") as stream:
        header = stream.read(348)
    if len(header) != 348:
        raise ValueError("Truncated NIfTI-1 header.")
    endian = "<" if struct.unpack("<i", header[:4])[0] == 348 else ">"
    if struct.unpack(endian + "i", header[:4])[0] != 348:
        raise ValueError("Expected a NIfTI-1 header.")
    pixdim = struct.unpack_from(endian + "8f", header, 76)
    qcode, scode = struct.unpack_from(endian + "2h", header, 252)
    b, c, d, x, y, z = struct.unpack_from(endian + "6f", header, 256)
    squared = b * b + c * c + d * d
    if squared > 1:
        b, c, d = np.asarray([b, c, d]) / np.sqrt(squared)
    a = np.sqrt(max(0.0, 1 - squared))
    rotation = np.array([
        [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
        [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
        [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-c*c-b*b],
    ])
    qform = np.eye(4)
    qform[:3, :3] = rotation @ np.diag([
        pixdim[1], pixdim[2], pixdim[3] * (-1 if pixdim[0] < 0 else 1),
    ])
    qform[:3, 3] = [x, y, z]
    sform = np.eye(4)
    sform[:3] = np.asarray(struct.unpack_from(endian + "12f", header, 280)).reshape(3, 4)
    return {
        "shape_xyz": list(struct.unpack_from(endian + "8h", header, 40)[1:4]),
        "spacing_xyz_mm": list(pixdim[1:4]), "qform_code": qcode, "sform_code": scode,
        "qform_ras": qform.tolist() if qcode else None,
        "sform_ras": sform.tolist() if scode else None,
        "spatial_units_code": header[123] & 7,
    }


def lps_affine(image: sitk.Image) -> np.ndarray:
    affine = np.eye(4)
    affine[:3, :3] = np.asarray(image.GetDirection()).reshape(3, 3) @ np.diag(image.GetSpacing())
    affine[:3, 3] = image.GetOrigin()
    return affine


def path_at_distance(path: np.ndarray, distance: float) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(lengths)))
    if np.any(lengths <= 0) or not 0 <= distance <= arc[-1]:
        raise ValueError("Path must have positive segments and contain the requested distance.")
    index = min(int(np.searchsorted(arc, distance, side="right") - 1), len(lengths) - 1)
    return path[index] + (distance - arc[index]) / lengths[index] * (path[index + 1] - path[index])


def release_path(name: str, root: Path = ROOT) -> Path:
    parts = Path(name).parts
    if not parts or ".." in parts or Path(name).is_absolute():
        raise ValueError("Invalid release path.")
    directory = root / "eval" / ("data" if parts[0].startswith("case_") else "docs")
    return directory / name


def verify_release(root: Path = ROOT) -> dict:
    manifest = read_json(root / "eval/docs/manifest.json")
    checksum = root / "eval/docs/SHA256SUMS.txt"
    checksum_rows = {}
    for line in checksum.read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        if name in checksum_rows:
            raise ValueError(f"Duplicate checksum path: {name}")
        checksum_rows[name] = expected
    rows = []
    for entry in manifest["files"]:
        path = release_path(entry["path"], root)
        rows.append({
            "path": str(path.relative_to(root)), "sha256": digest(path),
            "expected_sha256": entry["sha256"], "bytes": path.stat().st_size,
            "verified": digest(path) == entry["sha256"] and path.stat().st_size == entry["bytes"],
        })
    archived = []
    for path in sorted((root / "labels/organizer-v1/raw").iterdir()):
        source = (
            root / "eval/data" / path.stem / "annotations.json"
            if path.stem.startswith("case_") else root / "eval/docs" / path.name
        )
        archived.append({
            "path": str(path.relative_to(root)), "sha256": digest(path),
            "release_path": str(source.relative_to(root)),
            "byte_identical": digest(source) == digest(path),
        })
    return {
        "manifest_sha256": digest(root / "eval/docs/manifest.json"),
        "checksum_sha256": digest(checksum), "checksum_contains_crlf": b"\r\n" in checksum.read_bytes(),
        "checksum_matches_manifest": checksum_rows == {
            row["path"]: row["sha256"] for row in manifest["files"]
        },
        "manifest_counts": manifest["case_counts"], "files": rows, "archived_originals": archived,
        "all_verified": all(row["verified"] for row in rows)
        and all(row["byte_identical"] for row in archived)
        and checksum_rows == {row["path"]: row["sha256"] for row in manifest["files"]},
    }


def audit_case(case: str) -> dict:
    start = time.perf_counter()
    number = int(case[-3:])
    directory = ROOT / f"eval/data/case_{number}"
    image_path, mask_path = case_paths(case)
    raw = read_json(directory / "annotations.json")
    reference = load_reference(case)
    image, mask = read_nifti(str(image_path)), read_nifti(str(mask_path))
    daughter_path = directory / f"daughters{number}_draft.nii.gz"
    combined_path = directory / f"aorta_and_daughters{number}_draft.nii.gz"
    daughters, combined = read_nifti(str(daughter_path)), read_nifti(str(combined_path))
    for other in (mask, daughters, combined):
        validate_geometry(image, other > 0)
    array = sitk.GetArrayFromImage(daughters)
    parent = sitk.GetArrayFromImage(mask)
    merged = sitk.GetArrayFromImage(combined)
    face_contact = ndimage.binary_dilation(parent > 0, structure=ndimage.generate_binary_structure(3, 1))
    geometry = []
    ras = np.diag([-1, -1, 1, 1]) @ lps_affine(image)
    for path in (image_path, mask_path, daughter_path, combined_path):
        header = nifti_header(path)
        active = [header[key] for key in ("qform_ras", "sform_ras") if header[key] is not None]
        geometry.append({
            "path": str(path.relative_to(ROOT)), "sha256": digest(path), **header,
            "active_affines_match_sitk": bool(active) and all(
                np.allclose(affine, ras, rtol=0, atol=1e-4) for affine in active
            ),
        })
    branches = []
    canonical = {row["instance_id"]: row for row in reference["daughters"]}
    for branch in raw["daughters"]:
        guide = np.asarray(branch["centerline_xyz_mm"])
        voxels = np.asarray(branch["centerline_voxel_xyz"])
        physical = np.asarray([image.TransformContinuousIndexToPhysicalPoint(p.tolist()) for p in voxels])
        inverse = np.asarray([image.TransformPhysicalPointToContinuousIndex(p.tolist()) for p in guide])
        indices_zyx = np.argwhere(array == branch["label_value"])
        lower, upper = indices_zyx.min(axis=0), indices_zyx.max(axis=0) + 1
        crop = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
        label_crop = array[crop] == branch["label_value"]
        points = indices_zyx[:, ::-1] @ lps_affine(image)[:3, :3].T + image.GetOrigin()
        distances = cdist(guide, points).min(axis=1)
        seed = np.asarray(branch["seed_xyz_mm"])
        ostium = np.asarray(branch["ostium_xyz_mm"])
        seed_voxel = image.TransformPhysicalPointToIndex(seed.tolist())
        seed_label = int(array[tuple(reversed(seed_voxel))])
        direction = (seed - ostium) / np.linalg.norm(seed - ostium)
        branches.append({
            "instance_id": branch["instance_id"], "label_value": branch["label_value"],
            "canonical_fields_unchanged": all(
                value == branch[key] for key, value in canonical[branch["instance_id"]].items()
            ),
            "voxel_count": len(indices_zyx), "recorded_voxel_count": branch["voxel_count"],
            "components_6": int(ndimage.label(label_crop, ndimage.generate_binary_structure(3, 1))[1]),
            "components_26": int(ndimage.label(label_crop, np.ones((3, 3, 3)))[1]),
            "parent_face_contact_voxels": int(np.count_nonzero(label_crop & face_contact[crop])),
            "parent_overlap_voxels": int(np.count_nonzero(label_crop & (parent[crop] > 0))),
            "voxel_to_lps_max_error_mm": float(np.max(np.abs(physical - guide))),
            "lps_to_voxel_max_error": float(np.max(np.abs(inverse - voxels))),
            "all_guide_points_in_bounds": bool(
                np.all(inverse >= -0.5) and np.all(inverse < np.asarray(image.GetSize()) - 0.5)
            ),
            "guide_length_mm": float(np.linalg.norm(np.diff(guide, axis=0), axis=1).sum()),
            "seed_at_5mm_error_mm": float(np.linalg.norm(seed - path_at_distance(guide, 5))),
            "ostium_at_guide_start_error_mm": float(np.linalg.norm(ostium - guide[0])),
            "direction_norm": float(np.linalg.norm(branch["direction_xyz"])),
            "direction_seed_chord_error": float(np.linalg.norm(direction - branch["direction_xyz"])),
            "seed_nearest_native_voxel_label": seed_label,
            "seed_inside_own_label": seed_label == branch["label_value"],
            "seed_to_nearest_label_center_mm": float(cdist([seed], points).min()),
            "guide_to_nearest_label_center_max_mm": float(distances.max()),
            "guide_fraction_within_half_voxel_diagonal": float(np.mean(
                distances <= np.linalg.norm(image.GetSpacing()) / 2
            )),
            "measurement_and_review": {
                key: value for key, value in branch.items()
                if any(word in key for word in (
                    "radius", "diameter", "measurement", "confidence", "status",
                    "search_limit", "threshold", "notes",
                ))
            },
        })
    expected_merged = np.where(array > 0, array + 1, parent)
    return {
        "case_id": case, "daughters": len(branches),
        "known_radii": sum(row["radius_mm"] is not None for row in raw["daughters"]),
        "input_identity": {
            "image_sha256": digest(image_path), "mask_sha256": digest(mask_path),
            "canonical_reference_sha256": digest(ROOT / f"labels/organizer-v1/references/{case}.json"),
            "release_image_identical": digest(image_path) == digest(directory / f"orig{number}.nii.gz"),
            "release_mask_identical": digest(mask_path) == digest(directory / f"aorta{number}.nii.gz"),
            "canonical_source_hash_matches": reference["provenance"]["source_sha256"]
            == digest(directory / "annotations.json"),
        },
        "geometry": geometry, "sitk_lps_affine": lps_affine(image).tolist(),
        "metadata_shape_matches": list(image.GetSize()) == raw["shape_xyz"],
        "metadata_spacing_matches": list(image.GetSpacing()) == raw["spacing_xyz_mm"],
        "raw_coordinate_system": raw["coordinate_system"], "annotation_status": raw["annotation_status"],
        "parent_labels": np.unique(parent).tolist(), "daughter_labels": np.unique(array).tolist(),
        "combined_labels": np.unique(merged).tolist(),
        "combined_exact_parent_and_offset_daughters": bool(np.array_equal(merged, expected_merged)),
        "branches": branches, "review_notes": (directory / "review_notes.md").read_text(),
        "excluded_candidates": read_json(directory / "excluded_candidate.json")
        if (directory / "excluded_candidate.json").exists() else None,
        "runtime_s": time.perf_counter() - start, "process_peak_rss_mb": peak_mib(),
    }


def control_prediction(origins: list[list[float]], case: str = "audit_control") -> dict:
    return {
        "case_id": case, "parent": {"instance_id": "aorta"},
        "daughters": [{
            "instance_id": f"control_{i}", "parent_instance_id": "aorta",
            "ostium_xyz_mm": point, "seed_xyz_mm": (np.asarray(point) + [0, 5, 0]).tolist(),
            "direction_xyz": [0, 1, 0], "radius_mm": 2.0,
        } for i, point in enumerate(origins)],
    }


def coordinate_controls() -> dict:
    image = sitk.Image(32, 32, 32, sitk.sitkUInt8)
    image.SetOrigin((71.0, -103.0, 211.0))
    image.SetSpacing((0.7, 1.5, 2.3))
    image.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    raw = control_prediction([[4.0, 6.0, 8.0]])
    raw["coordinate_space"] = "voxel"
    raw["daughters"][0]["seed_xyz_mm"] = [9.0, 8.0, 8.0]
    del raw["daughters"][0]["direction_xyz"]
    normalized = normalize_reference(raw, [], image)["daughters"][0]
    expected_ostium = np.array([62.0, -100.2, 229.4])
    expected_seed = np.array([59.0, -96.7, 229.4])
    expected_direction = (expected_seed - expected_ostium) / np.linalg.norm(expected_seed - expected_ostium)
    supplied = copy.deepcopy(raw)
    supplied["daughters"][0]["direction_xyz"] = (np.array([5, 2, 0]) / np.sqrt(29)).tolist()
    supplied_direction = normalize_reference(supplied, [], image)["daughters"][0]["direction_xyz"]
    ras = mirrored(control_prediction([[62.0, -100.2, 229.4]]))
    ras["coordinate_space"] = "RAS"
    normalized_ras = normalize_reference(ras, [])["daughters"][0]
    return {
        "origin_lps_mm": list(image.GetOrigin()), "spacing_mm": list(image.GetSpacing()),
        "direction_matrix": list(image.GetDirection()),
        "derived_direction_and_landmarks_pass": bool(
            np.allclose(normalized["ostium_xyz_mm"], expected_ostium)
            and np.allclose(normalized["seed_xyz_mm"], expected_seed)
            and np.allclose(normalized["direction_xyz"], expected_direction)
        ),
        "supplied_voxel_direction_angle_error_degrees": float(np.degrees(np.arccos(np.clip(
            np.dot(expected_direction, supplied_direction), -1, 1
        )))),
        "ras_marker_is_converted_to_lps": bool(np.allclose(
            normalized_ras["ostium_xyz_mm"], [62.0, -100.2, 229.4]
        )),
        "released_references_affected": False,
    }


def matching_controls(output: Path) -> dict:
    rng = np.random.default_rng(88129)
    checked = 0
    for n in range(1, 5):
        for m in range(1, 5):
            for tolerance in TOLERANCES:
                p, r = rng.uniform(-5, 5, (n, 3)), rng.uniform(-5, 5, (m, 3))
                distance = cdist(p, r)
                best = (0, 0.0)
                for size in range(1, min(n, m) + 1):
                    for rows in combinations(range(n), size):
                        for columns in permutations(range(m), size):
                            costs = distance[rows, columns]
                            if np.all(costs <= tolerance):
                                best = min(best, (-size, float(costs.sum())))
                result = score_prediction(control_prediction(p.tolist()), control_prediction(r.tolist()), tolerance)
                observed = (-result["true_positives"], sum(row["ostium_error_mm"] for row in result["matches"]))
                if observed[0] != best[0] or not np.isclose(observed[1], best[1]):
                    raise ValueError("Matching differs from exhaustive cardinality/distance search.")
                checked += 1
    reference = control_prediction([[10.0, 20.0, 0.0], [20.0, 20.0, 0.0], [30.0, 20.0, 0.0]])
    reference["daughters"][0]["radius_mm"] = None
    reference["daughters"][0]["centerline_xyz_mm"] = [[10, 20, 0], [10, 25, 0]]
    prediction = control_prediction([[10.0, 20.0, 0.0], [20.0, 20.0, 0.0], [30.0, 20.0, 0.0]])
    refs, preds = output / "control-reference", output / "control-prediction"
    write_json(refs / "audit_control.json", reference)
    write_json(preds / "audit_control.json", prediction)
    direct = score(refs, preds, None, list(TOLERANCES))
    shared = {f"{t:g}": score_prediction(prediction, reference, t) for t in TOLERANCES}
    for tolerance_key, row in shared.items():
        stripped = copy.deepcopy(row)
        for match in stripped["matches"]:
            match.pop("seed_to_reference_centreline_mm")
        if stripped != direct["cases"][0]["by_tolerance"][tolerance_key]:
            raise ValueError("Direct/shared synthetic scoring differs.")
    mirror_dir = output / "control-mirror"
    write_json(mirror_dir / "audit_control.json", mirrored(prediction))
    mirror = score(refs, mirror_dir, None, list(TOLERANCES))
    empty = {case: control_prediction([], case) for case in CASES}
    empty_scores = score_variant(empty)
    for value in empty_scores.values():
        if value["summary"]["false_negatives"] != 19 or value["summary"]["cases"] != 5:
            raise ValueError("Empty-case denominators lost.")
    missing_rejected = False
    try:
        score_variant({case: value for case, value in empty.items() if case != CASES[-1]})
    except ValueError:
        missing_rejected = True
    write_json(output / "scoring-controls.json", {
        "shared": shared, "direct": direct, "mirrored": mirror, "empty_all_five": empty_scores,
    })
    selected = filtered(prediction, [0.2, 0.9, 0.5], 0.5)
    selection_preserved = [row["instance_id"] for row in selected["daughters"]] == [
        "control_1", "control_2",
    ]
    pooled = summarize(list(shared.values()))
    return {
        "exhaustive_3d_assignments_checked": checked, "direct_shared_equal": True,
        "unknown_radius_error_is_null": shared["3"]["matches"][0]["radius_error_mm"] is None,
        "seed_to_guide_mm": shared["3"]["matches"][0]["seed_to_reference_centreline_mm"],
        "mirror_warning_triggered": mirror["mirror_warning"],
        "all_five_empty_false_negatives": empty_scores["3"]["summary"]["false_negatives"],
        "missing_case_rejected": missing_rejected, "coordinates": coordinate_controls(),
        "filter_preserves_order_and_inclusive_threshold": selection_preserved,
        "pooled_unknown_radius_observations": pooled["errors"]["radius_error_mm"]["n"],
    }


def source_checks(sources: dict) -> dict:
    return {
        name: {
            "expected": sha, "current": digest(ROOT / name) if (ROOT / name).is_file() else None,
            "matches": (ROOT / name).is_file() and digest(ROOT / name) == sha,
        } for name, sha in sources.items()
    }


def model_audit() -> dict:
    paths = sorted(
        set((ROOT / "labels").glob("candidate-model*.json"))
        | set((ROOT / "labels/research").glob("*/models/*.json"))
        | set((ROOT / "labels/research").glob("*/*/model.json"))
    )
    models = []
    for path in paths:
        value = read_json(path)
        if "split" not in value:
            continue
        partitions = {
            part: sorted(set(cases) & set(CASES)) for part, cases in value["split"].items()
        }
        metadata = value.get("metadata", {})
        sources = value.get("contract", metadata.get("inference_contract", {})).get("source_sha256", {})
        if not sources:
            sources = metadata.get("source_sha256", {})
        extra_cases = {
            "metadata_fit_case_ids": metadata.get("fit_case_ids", []),
            "metadata_threshold_case_ids": metadata.get("threshold_case_ids", []),
            "training_records": [row["case_id"] for row in value.get("training", {}).get("train_records", [])],
            "validation_records": [
                row["case_id"] for row in value.get("training", {}).get("validation_records", [])
            ],
        }
        extra_overlap = {key: sorted(set(ids) & set(CASES)) for key, ids in extra_cases.items()}
        overlap = bool(partitions.get("train") or partitions.get("validation") or any(extra_overlap.values()))
        manifest = value.get("provenance", {}).get("manifest")
        manifest_path = path.parent.parent / "manifest.json"
        if manifest is None and manifest_path.exists():
            manifest = read_json(manifest_path)
        if manifest is None:
            manifest = {}
        manifest_cases = manifest.get("cases", [])
        manifest_ids = {row["case_id"] for row in manifest_cases}
        split_ids = set().union(*map(set, value["split"].values()))
        synthetic_only = (
            manifest.get("source") == "analytic_synthetic_geometry" and bool(split_ids)
            and split_ids <= manifest_ids
        )
        integrity: dict = {"kind": "file_sha256_only"}
        if value.get("model_type") == "branchseed_candidate_trees":
            try:
                TreeModel.load(path)
                integrity = {"kind": "TreeModel.load", "passed": True}
            except (ValueError, KeyError, TypeError) as error:
                integrity = {"kind": "TreeModel.load", "passed": False, "error": str(error)}
        if path.with_suffix(".onnx").exists():
            actual = digest(path.with_suffix(".onnx"))
            integrity = {
                "kind": "onnx_file_hash_only_no_inference", "sha256": actual,
                "matches_sidecar": actual == value.get("onnx_sha256"),
                "sidecar_integrity_matches": value.get("sidecar_sha256") == json_sha256({
                    key: item for key, item in value.items() if key != "sidecar_sha256"
                }),
            }
        models.append({
            "path": str(path.relative_to(ROOT)), "sha256": digest(path), "threshold": value["threshold"],
            "released_case_overlap_by_partition": partitions,
            "extra_case_evidence": {
                key: {"count": len(set(ids)), "released_overlap": extra_overlap[key]}
                for key, ids in extra_cases.items()
            },
            "fit_or_validation_overlap": overlap,
            "source_checks": source_checks(sources),
            "training_source_checks": source_checks(metadata.get("source_sha256", {})),
            "sidecar_current_source_checks": source_checks(value.get("current_source_sha256", {})),
            "integrity": integrity,
            "source_contract_available": bool(sources),
            "data_classification": "retrospective_released_image_overlap" if overlap else (
                "analytic_synthetic_only_by_manifest" if synthetic_only else "not_proven_synthetic_only"
            ),
            "manifest_source": manifest.get("source"),
            "manifest_released_case_ids": sorted(manifest_ids & set(CASES)),
            "manifest_split_coverage_complete": split_ids <= manifest_ids if manifest else None,
            "threshold_selected_on": value.get("threshold_selected_on", metadata.get("threshold_selected_on")),
            "normalizer_fit_partition": value.get("normalizer", {}).get("fit_partition"),
            "prior_exposure": metadata.get("prior_exposure", value.get("provenance", {}).get("retrospective_test_reuse")),
            "current_source_compatible": all(row["matches"] for row in source_checks(sources).values())
            if sources else None,
            "clean_selection_data_eligible": not overlap,
            "eligibility_caveat": "Data-only eligibility; source compatibility, synthetic regressions and "
            "runtime gates must pass separately. No model is promoted by this audit.",
            "inference_run": False,
        })
    reviews = read_json(ROOT / "labels/reviews.json")
    by_case = {
        case: {
            "reviewed_candidates": sum(row["case_id"] == case for row in reviews["records"]),
            "labels": dict(Counter(
                row["label"] for row in reviews["records"] if row["case_id"] == case
            )),
            "pseudo_reference_exists": (ROOT / f"labels/pseudo_references/{case}.json").exists(),
            "pseudo_reference_sha256": digest(ROOT / f"labels/pseudo_references/{case}.json"),
            "pseudo_reference_daughters": len(read_json(ROOT / f"labels/pseudo_references/{case}.json")["daughters"]),
        } for case in CASES
    }
    return {
        "models": models, "prior_case_reviews": by_case,
        "reviews_sha256": digest(ROOT / "labels/reviews.json"),
        "identity_scope": "Case IDs link to byte-identical release CT/masks. Legacy reviews do not carry CT hashes.",
        "expert_negatives": False,
        "selection_scope": "Any fitting/calibration overlap excludes the entire variant from clean five-case selection.",
    }


def strict_case(case: str, output: Path) -> None:
    start = time.perf_counter()
    image_path, mask_path = case_paths(case)
    config = DetectorConfig()
    detection = detect(read_nifti(str(image_path)), read_nifti(str(mask_path)), config)
    write_json(output / "strict" / f"{case}.json", detection.prediction(case))
    write_json(output / "strict-candidates" / f"{case}.json", {
        "case_id": case, "configuration": asdict(config), "diagnostics": detection.diagnostics(),
        "feature_names": FEATURE_NAMES,
        "ordered_candidates": [{
            "instance_id": branch.instance_id, "prediction": branch.prediction(),
            "features": features(branch), "model_score": None,
        } for branch in detection.branches],
    })
    write_json(output / "runtime" / f"{case}.json", {
        "runtime_s": time.perf_counter() - start, "peak_rss_mb": peak_mib(),
    })


def strict_comparison(output: Path) -> dict:
    for case in CASES:
        subprocess.run([
            sys.executable, str(Path(__file__).resolve()), "--output", str(output), "--strict-case", case,
        ], check=True, cwd=ROOT)
    predictions = {case: read_json(output / "strict" / f"{case}.json") for case in CASES}
    shared = score_variant(predictions)
    direct = score(ROOT / "labels/organizer-v1/references", output / "strict", None, list(TOLERANCES))
    for tolerance, data in shared.items():
        for expected, observed in zip(data["cases"], direct["cases"]):
            stripped = copy.deepcopy(expected)
            for match in stripped["matches"]:
                match.pop("seed_to_reference_centreline_mm")
            if stripped != observed["by_tolerance"][tolerance]:
                raise ValueError(f"Direct/shared strict scoring differs for {expected['case_id']}.")
    write_json(output / "strict-direct-scoring.json", direct)
    return {
        "name": "audit_strict_scoring_control", "prediction_dir": str((output / "strict").relative_to(ROOT)),
        "configuration": {
            "detector": asdict(DetectorConfig()), "detector_sha256": digest(ROOT / "detector.py"),
            "candidate_filter": None, "threads": 4, "scoring_only_no_selection": True,
            "runtime_scope": "Per-process read+detect+candidate serialization, excluding interpreter startup.",
        },
        "eligible_for_selection": False,
        "ineligible_reason": "Audit scoring control duplicates the baseline family; not a tuned model entry.",
        "scores": shared, "runtime": {case: read_json(output / "runtime" / f"{case}.json") for case in CASES},
        "direct_shared_case_metrics_equal": True, "direct_mirror_warning": direct["mirror_warning"],
        "pooled_signed_count_error": sum(
            row["daughter_counts"]["signed_error"] for row in shared["3"]["cases"]
        ),
    }


def structural_summary(release: dict, cases: list[dict]) -> dict:
    checks = {
        "all_55_manifest_files_and_checksums_verified": release["all_verified"] and len(release["files"]) == 55,
        "inventory_3_4_3_6_3": [case["daughters"] for case in cases] == [3, 4, 3, 6, 3],
        "three_known_sixteen_unknown_radii": sum(case["known_radii"] for case in cases) == 3,
        "exact_input_identity": all(
            all(case["input_identity"][key] for key in (
                "release_image_identical", "release_mask_identical", "canonical_source_hash_matches",
            )) for case in cases
        ),
        "all_active_nifti_affines_match": all(
            header["active_affines_match_sitk"] and header["spatial_units_code"] == 2
            for case in cases for header in case["geometry"]
        ),
        "source_dimensions_spacing_and_coordinate_metadata": all(
            case["metadata_shape_matches"] and case["metadata_spacing_matches"]
            and case["raw_coordinate_system"] == "SimpleITK physical LPS millimetres" for case in cases
        ),
        "exact_integer_label_inventory": all(
            case["daughter_labels"] == list(range(case["daughters"] + 1))
            and case["parent_labels"] == [0, 1] and case["combined_exact_parent_and_offset_daughters"]
            for case in cases
        ),
    }
    branches = [branch for case in cases for branch in case["branches"]]
    checks.update({
        "canonical_reference_fields_unchanged": all(row["canonical_fields_unchanged"] for row in branches),
        "connected_parent_contact_without_overlap": all(
            row["components_6"] == row["components_26"] == 1 and row["parent_face_contact_voxels"] > 0
            and row["parent_overlap_voxels"] == 0 and row["voxel_count"] == row["recorded_voxel_count"]
            for row in branches
        ),
        "lps_voxel_roundtrip_and_domain": all(
            row["voxel_to_lps_max_error_mm"] < 1e-5 and row["lps_to_voxel_max_error"] < 1e-5
            and row["all_guide_points_in_bounds"] for row in branches
        ),
        "seed_at_5mm_and_ostium_at_guide_start": all(
            row["seed_at_5mm_error_mm"] < 1e-5 and row["ostium_at_guide_start_error_mm"] < 1e-5
            and row["seed_inside_own_label"] for row in branches
        ),
        "unit_directions_match_seed_chord": all(
            abs(row["direction_norm"] - 1) < 1e-6 and row["direction_seed_chord_error"] < 1e-6
            for row in branches
        ),
    })
    return {"all_passed": all(checks.values()), "checks": checks}


def findings(cases: list[dict], controls: dict, models: dict) -> list[dict]:
    rows: list[dict] = [
        {
            "id": "audit_reference_authority", "severity": "information", "kind": "provenance",
            "message": "19 targets are judge-approved for this local comparison. The original release remains "
            "AI-assisted, expert-review-pending, and explicitly non-exhaustive; approval does not change authorship.",
        },
        {
            "id": "audit_unknown_radii", "severity": "warning", "kind": "measurement",
            "message": "Only 3 of 19 radii are measured; 16 remain null. Normalization uses an internal validator "
            "placeholder but both scoring paths mask its error. Radius errors describe matched measured radii only.",
        },
        {
            "id": "audit_native_resolution", "severity": "warning", "kind": "measurement",
            "message": "1.5 mm native spacing gives about 1.33 voxels across the 2 mm origin threshold. Visual "
            "origin diameter, seed radius and threshold-derived diameters differ. Guide sampling adds no resolution.",
        },
        {
            "id": "audit_incomplete_references", "severity": "warning", "kind": "methodology",
            "message": "Unmatched predictions are local-reference false positives, not established anatomical "
            "negatives. Excluded/uncertain candidates and an unfinished exhaustive sweep are documented; "
            "accepted targets are unchanged by this audit.",
        },
        {
            "id": "audit_local_metric", "severity": "information", "kind": "methodology",
            "message": "No official evaluator exists in the supplied release. Maximum-cardinality matching at "
            "2/3/5 mm is local; matched-only geometry errors and count agreement do not establish discovery accuracy.",
        },
        {
            "id": "audit_selection_reuse", "severity": "warning", "kind": "selection",
            "message": "LOCO must select on four whole cases, include empty outputs, and exclude all five images "
            "from fitting/calibration of transferable models. Rank by pooled 3 mm F1, count MAE, dependency cost, "
            "measured runtime and deterministic name. Post-reference algorithms are development-only. "
            "Five-case bootstrap cannot remove adaptive algorithm-selection bias or prior inspection exposure.",
        },
        {
            "id": "audit_direct_missing_prediction", "severity": "warning", "kind": "scoring",
            "message": "score_references.score lists missing files but omits them from aggregation. It must only "
            "be used as a control with complete inputs. final_evaluation.score_variant correctly rejects missing cases.",
        },
        {
            "id": "audit_pooled_signed_count", "severity": "information", "kind": "reporting",
            "message": "final_evaluation.summarize includes count MAE but no pooled signed count error. Per-case "
            "signed_error exists; aggregators should sum/average it explicitly if reporting signed count bias.",
        },
    ]
    if controls["coordinates"]["supplied_voxel_direction_angle_error_degrees"] > 0.01:
        rows.append({
            "id": "audit_voxel_direction_normalization", "severity": "warning", "kind": "coordinate_bug",
            "source": "score_references.py:normalize_reference",
            "message": "For a voxel-space reference with a supplied voxel direction, positions are transformed "
            "but the direction is only normalized, not rotated/scaled. Rotated anisotropic control demonstrates "
            "an angular error. Deriving direction from transformed seed/ostium passes. Released LPS references "
            "are unaffected; reject or explicitly convert voxel direction inputs.",
            "evidence": controls["coordinates"],
        })
    if not controls["coordinates"]["ras_marker_is_converted_to_lps"]:
        rows.append({
            "id": "audit_ras_marker_not_converted", "severity": "warning", "kind": "coordinate_limitation",
            "source": "score_references.py:normalize_reference",
            "message": "A coordinate_space=RAS marker is not converted to LPS. The direct scorer's mirror "
            "diagnostic catches the multi-branch control; final_evaluation has no mirror warning. All five "
            "accepted references independently verify as LPS, so this does not invalidate their scores.",
        })
    partial = [
        f"{case['case_id']}/{branch['instance_id']}" for case in cases for branch in case["branches"]
        if branch["guide_fraction_within_half_voxel_diagonal"] < 1
    ]
    rows.append({
        "id": "audit_guide_is_not_mask_extent", "severity": "warning", "kind": "measurement",
        "message": "A 10 mm guide does not measure native mask extent. Some guide samples are farther than "
        "half a voxel diagonal from a label center. This structural measurement does not adjudicate anatomy.",
        "instances": partial,
    })
    rows.append({
        "id": "audit_model_overlap", "severity": "warning", "kind": "training_overlap",
        "models_with_fit_or_validation_overlap": [
            model["path"] for model in models["models"] if model["fit_or_validation_overlap"]
        ],
        "message": "Old real-label logistic models fit subject022/023 and calibrate on subject020. The mixed "
        "CNN also fits subject022/023. Pseudo-labels are not expert negatives. Absence of subject019 in old "
        "fitting lists does not establish a wholly independent five-case evaluation.",
    })
    rows.append({
        "id": "audit_source_contracts", "severity": "warning", "kind": "reproducibility",
        "models_with_mismatch": [
            model["path"] for model in models["models"]
            if any(not source["matches"] for source in model["source_checks"].values())
        ],
        "message": "Use exact frozen extraction source or re-extract/retrain where hashes differ. Current-source "
        "sidecar labels do not supersede the inference contract. Legacy logistic artifacts have no source contract.",
    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--strict-case", choices=CASES)
    parser.add_argument("--skip-strict", action="store_true")
    args = parser.parse_args()
    thread_limit(4)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.strict_case:
        strict_case(args.strict_case, output)
        return
    failures = []
    variants = []
    if not args.skip_strict:
        try:
            variants.append(strict_comparison(output))
        except (subprocess.CalledProcessError, ValueError, OSError) as error:
            failures.append({"name": "audit_strict_scoring_control", "error": str(error)})
    release = verify_release()
    cases = [audit_case(case) for case in CASES]
    controls = matching_controls(output)
    models = model_audit()
    structure = structural_summary(release, cases)
    for name, passed in structure["checks"].items():
        if not passed:
            failures.append({"name": name, "error": "Structural audit check failed."})
    report = {
        "schema_version": 1, "family": "audit", "variants": variants, "failures": failures,
        "scope": "Independent structural/protocol audit; no anatomical adjudication or model tuning.",
        "release": release, "cases": cases, "controls": controls, "model_audit": models,
        "structural_summary": structure,
        "reviewer_checklist": {
            "path": "eval/docs/REVIEWER_CHECKLIST.md",
            "sha256": digest(ROOT / "eval/docs/REVIEWER_CHECKLIST.md"),
            "pending_register_rows": (ROOT / "eval/docs/REVIEWER_CHECKLIST.md").read_text().count("| Pending |"),
            "expert_signoff_in_original_package": False,
        },
        "findings": findings(cases, controls, models),
        "source_sha256": {name: digest(ROOT / name) for name in SOURCE_FILES},
        "baseline_matches_4a43dd4": digest(ROOT / "detector.py") == BASELINE_SHA256,
        "versions": {name: importlib.metadata.version(name) for name in (
            "numpy", "scipy", "SimpleITK", "scikit-image", "pytest", "ruff", "mypy",
        )},
        "python": sys.version, "platform": sys.platform, "threads": 4,
        "machine": {"system": platform.platform(), "processor": platform.processor()},
        "total_instances": sum(case["daughters"] for case in cases),
        "known_radii": sum(case["known_radii"] for case in cases),
        "exclusions": [
            "No reference changes, anatomy decisions, threshold tuning or deployment selection.",
            "Strict baseline is an audit-only duplicate, excluded from model-selection candidates.",
            "Linux resource measurements are not a timed offline Windows deployment validation.",
            "Source hashes are compared, but CNN graphs are not executed in this audit.",
        ],
    }
    write_json(output / "report.json", report)
    print(json.dumps({
        "report": str((output / "report.json").relative_to(ROOT)),
        "instances": report["total_instances"], "known_radii": report["known_radii"],
        "release_verified": release["all_verified"], "failures": failures,
    }))


if __name__ == "__main__":
    main()
