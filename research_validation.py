"""Paired research evaluation; no detector execution or production model changes.

CLI: python research_validation.py --manifest comparison-input.json --output report.json
Paths in an input manifest are relative to that manifest. A report's ``manifest``
can be saved and replayed by the same command; content hashes prevent stale replay.

Required manifest fields: schema_version=1, references, baseline, variants
({name: prediction_directory}), reference_provenance (complete_expert,
analytic_synthetic, candidate_pseudo), references_complete (bool). Optional:
proposals ({run: directory}), histories ({run: {history_complete, split,
inspected_cases, pseudo_trained_cases, training_cases, tuning_cases,
training_groups, tuning_groups}}), groups ({case: seed_group}), expected_cases,
image_paths ({case: NIfTI}), prediction_sources, runtime_reports, compute_budget,
tolerance_mm (default 3), bootstrap_samples (2000), bootstrap_seed (42).

Runtime reports are supplied evidence, never measurements made by this evaluator:
{run: {platform: "Windows", cpu_cores: 4, gpu: false, network: false,
cases: {case: {runtime_s: ..., peak_rss_mb: ...}}}}. A promotion pass requires
an explicit budget {max_runtime_s, max_peak_rss_mb, max_cpu_cores}, complete
expert references, complete exposure histories, no lost branches at 2/3/5 mm,
and a paired confidence interval supporting improvement. The decision is a
conservative research gate, not an organizer scoring rule or clinical approval.

References use score_references layouts/aliases. Optional per-daughter
centreline_xyz_mm (or centerline_xyz_mm) is an ordered physical-mm polyline.
Explicit RAS and ambiguous coordinate units are rejected, never auto-flipped.
Missing physical fields stay unknown even when normalization needs placeholders.
"""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np
import SimpleITK as sitk

from evaluate import evaluate_case, summarize_cases, validate_prediction
from nifti_io import read_nifti
from score_references import (
    CASE_KEYS, DAUGHTER_KEYS, FIELD_ALIASES, VOXEL_HINTS,
    load_references, normalize_case_id, normalize_reference,
)

PROVENANCES = ("complete_expert", "analytic_synthetic", "candidate_pseudo")
ERROR_FIELDS = (
    "ostium_error_mm", "seed_error_mm", "radius_error_mm", "direction_error_degrees",
    "seed_to_reference_centreline_mm", "relative_radius_error",
)
COUNT_FIELDS = ("true_positives", "false_positives", "false_negatives")
BOOTSTRAP_FIELDS = (*COUNT_FIELDS, "precision", "recall", "f1")


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object.")
    return value


def _digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _object_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _vector(value: list, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain three finite physical coordinates.")
    return vector


def seed_polyline_distance(seed: list, polyline: list) -> float:
    """Minimum Euclidean distance to clamped line segments, including repeated points."""
    point = _vector(seed, "seed")
    vertices = np.asarray(polyline, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 2:
        raise ValueError("Reference centreline must have at least two 3D points.")
    if not np.isfinite(vertices).all():
        raise ValueError("Reference centreline must be finite.")
    segments = np.diff(vertices, axis=0)
    lengths_squared = np.sum(segments * segments, axis=1)
    factors = np.divide(
        np.sum((point - vertices[:-1]) * segments, axis=1), lengths_squared,
        out=np.zeros(len(segments)), where=lengths_squared > 0,
    )
    closest = vertices[:-1] + np.clip(factors, 0, 1)[:, None] * segments
    return float(np.min(np.linalg.norm(closest - point, axis=1)))


def _entries(raw: dict) -> list[dict]:
    daughters = next((raw[key] for key in DAUGHTER_KEYS if key in raw), None)
    if isinstance(daughters, dict):
        daughters = [dict(value, instance_id=key) for key, value in daughters.items()]
    if not isinstance(daughters, list) or not all(isinstance(d, dict) for d in daughters):
        raise ValueError("Reference daughter list must contain objects.")
    return daughters


def _reference_sources(path: Path, errors: list[dict]) -> list[dict]:
    sources: list[dict] = []
    files = sorted(path.rglob("*.json")) if path.is_dir() else [path]
    for file in files:
        try:
            loaded = json.loads(file.read_text(encoding="utf-8"))
            rows = load_references(file)
            if not rows:
                raise ValueError("No reference cases found in this file.")
            metadata = (
                {k: v for k, v in loaded.items() if k != "cases"}
                if isinstance(loaded, dict) and "cases" in loaded else {}
            )
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("Reference case must be an object.")
                sources.append(dict(row, _bundle_metadata=metadata))
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append({"case_id": None, "path": str(file), "error": str(error)})
    return sources


def _check_provenance(raw: dict, provenance: str) -> None:
    for source in (raw, raw["_bundle_metadata"]):
        declared = source.get("reference_provenance")
        if declared is not None and declared != provenance:
            raise ValueError("Reference provenance conflicts with the manifest.")
        origin = source.get("provenance", {})
        kind = str(origin.get("kind", "") if isinstance(origin, dict) else origin).lower()
        pseudo = "pseudo" in kind or source.get("scope") == "candidate_reviews_only"
        if pseudo and provenance != "candidate_pseudo":
            raise ValueError("Candidate pseudo-references cannot be relabelled as expert/synthetic references.")
        if "synthetic" in kind and provenance == "complete_expert":
            raise ValueError("Synthetic reference provenance cannot be relabelled as expert.")


def ingest_references(
    path: Path, provenance: str, complete: bool, images: dict[str, sitk.Image] | None = None,
) -> tuple[dict[str, dict], list[dict], list[str]]:
    """Normalize organizer layouts while preserving source provenance and ingestion errors."""
    if provenance not in PROVENANCES or not isinstance(complete, bool):
        raise ValueError("Explicit reference_provenance and boolean references_complete are required.")
    references: dict[str, dict] = {}
    errors: list[dict] = []
    notes: list[str] = []
    seen: set[str] = set()
    for raw in _reference_sources(path, errors):
        case_id = None
        try:
            if not isinstance(raw, dict):
                raise ValueError("Reference case must be an object.")
            case_value = next((raw[key] for key in CASE_KEYS if key in raw), None)
            if case_value is None:
                raise ValueError("Reference has no case identifier.")
            case_id = normalize_case_id(case_value)
            if case_id in seen:
                references.pop(case_id, None)
                raise ValueError(f"Duplicate normalized reference case: {case_id}")
            seen.add(case_id)
            _check_provenance(raw, provenance)
            units = str(raw.get("coordinate_space", raw.get("units", "physical_mm"))).lower()
            if units not in ("physical", "physical_mm", "mm", "lps", "lps_mm", "sitk") and not any(
                hint in units for hint in VOXEL_HINTS
            ):
                raise ValueError(f"Unsupported coordinate convention: {units}; convert explicitly to SimpleITK.")
            entries = _entries(raw)
            parent = raw.get("parent", {"instance_id": "aorta"})
            if not isinstance(parent, dict) or parent.get("instance_id") != "aorta" or any(
                entry.get("parent_instance_id", "aorta") != "aorta" for entry in entries
            ):
                raise ValueError("References must describe direct aortic daughters.")
            if any(hint in units for hint in VOXEL_HINTS):
                for entry in entries:
                    if any(k in entry for k in FIELD_ALIASES["direction_xyz"]) and raw.get(
                        "direction_coordinate_space"
                    ) not in ("physical_mm", "LPS", "lps", "sitk"):
                        raise ValueError("Voxel references with explicit directions must declare direction_coordinate_space.")
                    if any(k in entry for k in ("radius", "r_mm")) and "radius_mm" not in entry:
                        raise ValueError("Voxel references must use explicit radius_mm or diameter_mm.")
            ids = [
                str(next((d[k] for k in FIELD_ALIASES["instance_id"] if k in d), f"reference_{i + 1:03d}"))
                for i, d in enumerate(entries)
            ]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate reference IDs cannot be safely tracked across variants.")
            for entry in entries:
                radius = next((entry[k] for k in FIELD_ALIASES["radius_mm"] if k in entry), None)
                if radius is not None and (not np.isfinite(float(radius)) or float(radius) <= 0):
                    raise ValueError("Supplied reference radius/diameter must be positive and finite.")
            reference = normalize_reference(raw, notes, (images or {}).get(case_id))
            validate_prediction(reference)
            for daughter, entry in zip(reference["daughters"], entries):
                daughter["_seed_known"] = any(k in entry and entry[k] is not None for k in FIELD_ALIASES["seed_xyz_mm"])
                daughter["_direction_known"] = any(
                    k in entry and entry[k] is not None for k in FIELD_ALIASES["direction_xyz"]
                )
                polyline = entry.get("centreline_xyz_mm", entry.get("centerline_xyz_mm"))
                if polyline is not None:
                    seed_polyline_distance(daughter["seed_xyz_mm"], polyline)
                    daughter["_centreline_xyz_mm"] = polyline
            reference["_source"] = raw
            reference["_family"] = str(raw.get("family", "unspecified"))
            reference["_complete"] = (
                complete and provenance != "candidate_pseudo"
                and raw.get("references_complete", True) is True
                and raw["_bundle_metadata"].get("references_complete", True) is True
            )
            reference["_source_sha256"] = _object_digest(raw)
            references[case_id] = reference
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            errors.append({"case_id": case_id, "error": str(error)})
    if not seen and not errors:
        errors.append({"case_id": None, "error": "No reference cases were found."})
    return references, errors, notes


def _predictions(directory: Path, references: dict[str, dict]) -> tuple[dict[str, dict], list[dict]]:
    predictions: dict[str, dict] = {}
    errors: list[dict] = []
    seen: set[str] = set()
    if not directory.is_dir():
        errors.append({"case_id": None, "error": f"Prediction directory is missing: {directory}"})
    for path in sorted(directory.glob("*.json")):
        case_id = None
        try:
            prediction = _read(path)
            validate_prediction(prediction)
            case_id = prediction["case_id"]
            if case_id in seen:
                predictions.pop(case_id, None)
                raise ValueError("Duplicate prediction case.")
            seen.add(case_id)
            if case_id not in references:
                raise ValueError("Unknown prediction case: no valid supplied reference.")
            predictions[case_id] = prediction
        except (ValueError, KeyError, TypeError, OSError, OverflowError) as error:
            errors.append({"case_id": case_id, "path": str(path), "error": str(error)})
    for case_id in sorted(set(references) - set(predictions)):
        errors.append({"case_id": case_id, "error": "Missing valid prediction case."})
    return predictions, errors


def _evaluate(prediction: dict, reference: dict, tolerance: float) -> dict:
    result = evaluate_case(prediction, reference, tolerance)
    refs = {d["instance_id"]: d for d in reference["daughters"]}
    preds = {d["instance_id"]: d for d in prediction["daughters"]}
    for match in result["matches"]:
        ref, pred = refs[match["reference_id"]], preds[match["prediction_id"]]
        if not ref["_seed_known"]:
            match["seed_error_mm"] = None
        if not ref["_direction_known"]:
            match["direction_error_degrees"] = None
        if not ref["_radius_known"]:
            match["radius_error_mm"] = None
        match["relative_radius_error"] = (
            match["radius_error_mm"] / ref["radius_mm"] if ref["_radius_known"] else None
        )
        match["seed_to_reference_centreline_mm"] = (
            seed_polyline_distance(pred["seed_xyz_mm"], ref["_centreline_xyz_mm"])
            if "_centreline_xyz_mm" in ref else None
        )
    result["matched_errors_only"] = {
        field: _statistics([m[field] for m in result["matches"] if m[field] is not None])
        for field in ERROR_FIELDS
    }
    return result


def _statistics(values: list[float]) -> dict:
    return {
        "n": len(values), "mean": float(np.mean(values)) if values else None,
        "median": float(np.median(values)) if values else None,
        "p95": float(np.percentile(values, 95)) if values else None,
        "geometric_mean": (
            0.0 if 0 in values else float(np.exp(np.mean(np.log(values))))
        ) if values else None,
    }


def _summary(rows: list[dict]) -> dict:
    summary = summarize_cases([dict(row, matches=[]) for row in rows])
    summary["matched_errors_only"] = {
        field: _statistics([m[field] for row in rows for m in row["matches"] if m[field] is not None])
        for field in ERROR_FIELDS
    }
    available = [r for r in rows if r.get("proposal_misses") is not None]
    summary["proposal_misses"] = {
        "count": sum(len(r["proposal_misses"]) for r in available) if available else None,
        "cases_with_proposal_data": len(available), "cases_without_proposal_data": len(rows) - len(available),
        "reference_ids_by_case": {r["case_id"]: r["proposal_misses"] for r in available},
    }
    return summary


def _changes(baseline: dict, variant: dict) -> dict:
    before = {m["reference_id"] for m in baseline["matches"]}
    after = {m["reference_id"] for m in variant["matches"]}
    return {
        "newly_lost_reference_ids": sorted(before - after),
        "newly_recovered_reference_ids": sorted(after - before),
        "extra_false_positive_ids": sorted(
            set(variant["unmatched_prediction_ids"]) - set(baseline["unmatched_prediction_ids"])
        ),
        "corrected_false_positive_ids": sorted(
            set(baseline["unmatched_prediction_ids"]) - set(variant["unmatched_prediction_ids"])
        ),
        "delta": {field: variant[field] - baseline[field] for field in COUNT_FIELDS},
        "false_positive_id_comparison": "run-local IDs; comparability requires stable candidate IDs",
    }


def _family_changes(differences: dict[str, dict], cases: list[str], tolerance: str) -> dict:
    fields = (
        "newly_lost_reference_ids", "newly_recovered_reference_ids",
        "extra_false_positive_ids", "corrected_false_positive_ids",
    )
    return {
        "paired_case_ids": cases,
        "delta": {field: sum(differences[c][tolerance]["delta"][field] for c in cases) for field in COUNT_FIELDS},
        **{
            field: [
                {"case_id": case, "instance_id": identifier}
                for case in cases for identifier in differences[case][tolerance][field]
            ] for field in fields
        },
    }


def overlap_report(history: dict | None, cases: list[str], groups: dict[str, str]) -> dict:
    if history is not None and not isinstance(history, dict):
        raise ValueError("Exposure history must be an object.")
    history = history or {}
    split = history.get("split", {})
    if not isinstance(split, dict):
        raise ValueError("History split must be an object.")
    exposures = {key: history.get(key, []) for key in (
        "training_cases", "tuning_cases", "inspected_cases", "pseudo_trained_cases",
    )}
    exposures.update({f"split_{name}": split.get(name, []) for name in ("train", "validation", "test")})
    for values in exposures.values():
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise ValueError("History case lists must contain case-ID strings.")
    overlap = {
        key: sorted(set(cases) & {normalize_case_id(v) for v in values})
        for key, values in exposures.items()
    }
    exposed_groups: set[str] = set()
    for key in ("training_groups", "tuning_groups"):
        values = history.get(key, [])
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise ValueError("History group lists must contain strings.")
        exposed_groups.update(values)
    contaminated = set().union(*(set(v) for k, v in overlap.items() if k != "split_test"))
    exposed_groups.update(groups[c] for c in contaminated)
    group_overlap = sorted(set(groups.values()) & exposed_groups)
    contaminated.update(c for c in cases if groups[c] in exposed_groups)
    complete = history.get("history_complete") is True and (
        all(key in split for key in ("train", "validation", "test"))
        or all(key in history for key in ("training_cases", "tuning_cases"))
    )
    return {
        "status": "contaminated" if contaminated else "no_overlap" if complete else "unknown",
        "history_complete": complete, "overlap_by_exposure": overlap,
        "contaminated_reference_cases": sorted(contaminated), "group_overlap": group_overlap,
        "supplied_history": history,
    }


def paired_bootstrap(
    baseline: dict[str, dict], variant: dict[str, dict], groups: dict[str, str],
    samples: int = 2000, seed: int = 42,
) -> dict:
    """Percentile intervals from paired case/group resampling, never candidate resampling."""
    if type(samples) is not int or type(seed) is not int or samples < 1 or seed < 0:
        raise ValueError("Bootstrap samples must be positive and seed nonnegative.")
    cases = sorted(set(baseline) & set(variant))
    units = sorted({groups[c] for c in cases})
    if not cases:
        return {"units": 0, "cases": [], "intervals": {}, "seed": seed, "samples": samples}
    arrays = []
    for run in (baseline, variant):
        arrays.append(np.asarray([
            [sum(run[c][field] for c in cases if groups[c] == group) for field in COUNT_FIELDS]
            for group in units
        ], dtype=float))
    rng = np.random.default_rng(seed)
    draws = np.empty((2, samples, len(BOOTSTRAP_FIELDS)))
    for index in range(samples):
        selected = rng.integers(0, len(units), len(units))
        for run_index, array in enumerate(arrays):
            tp, fp, fn = array[selected].sum(axis=0)
            draws[run_index, index] = [
                tp, fp, fn, tp / (tp + fp) if tp + fp else np.nan,
                tp / (tp + fn) if tp + fn else np.nan,
                2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
            ]
    intervals: dict[str, dict] = {}
    for name, values in (("baseline", draws[0]), ("variant", draws[1]), ("paired_difference", draws[1] - draws[0])):
        intervals[name] = {}
        for index, field in enumerate(BOOTSTRAP_FIELDS):
            finite = values[:, index][np.isfinite(values[:, index])]
            intervals[name][field] = {
                "low": float(np.percentile(finite, 2.5)) if len(finite) else None,
                "high": float(np.percentile(finite, 97.5)) if len(finite) else None,
                "valid_resamples": len(finite),
            }
    return {
        "units": len(units), "cases": cases, "groups": {c: groups[c] for c in cases},
        "seed": seed, "samples": samples, "confidence": 0.95,
        "method": "paired group percentile; micro-count ratios; undefined ratios omitted and counted",
        "intervals": intervals,
    }


def _compute_guard(report: dict | None, budget: dict | None, cases: list[str]) -> dict:
    reasons = []
    if report is None or budget is None:
        return {"status": "unknown", "reasons": ["runtime_report_or_budget_missing"], "supplied_report": report}
    if not isinstance(report, dict) or not isinstance(budget, dict):
        return {"status": "rejected", "reasons": ["invalid_runtime_report_or_budget"], "supplied_report": report}
    if not all(key in budget for key in ("max_runtime_s", "max_peak_rss_mb", "max_cpu_cores")):
        reasons.append("incomplete_compute_budget")
    for key, value in budget.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            reasons.append(f"invalid_budget:{key}")
    if reasons:
        return {"status": "rejected", "reasons": reasons, "supplied_report": report}
    if report.get("platform") != "Windows" or report.get("gpu") is not False or report.get("network") is not False:
        reasons.append("target_environment_unverified")
    cores = report.get("cpu_cores")
    if not isinstance(cores, int) or isinstance(cores, bool) or not 0 < cores <= min(budget["max_cpu_cores"], 4):
        reasons.append("cpu_budget_exceeded_or_unknown")
    measured = report.get("cases", {})
    if not isinstance(measured, dict):
        return {"status": "rejected", "reasons": ["invalid_runtime_cases"], "supplied_report": report}
    for case in cases:
        row = measured.get(case, {})
        if not isinstance(row, dict):
            reasons.append(f"invalid_runtime_case:{case}")
            continue
        for field, limit in (("runtime_s", budget["max_runtime_s"]), ("peak_rss_mb", min(budget["max_peak_rss_mb"], 8000))):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                reasons.append(f"unknown_measurement:{case}:{field}")
            elif value > limit:
                reasons.append(f"compute_overrun:{case}:{field}")
    return {
        "status": "rejected" if reasons else "pass", "reasons": reasons,
        "supplied_report": report, "measurement_origin": "caller supplied; not measured by research_validation",
    }


def promotion_decision(
    provenance: str, complete: bool, errors: list[dict], overlaps: dict[str, dict],
    compute: dict[str, dict], comparisons: dict[str, dict], intervals: dict,
    baseline: dict, variant: dict,
) -> dict:
    reasons = []
    if provenance != "complete_expert" or not complete:
        reasons.append("insufficient_reference_provenance")
    if errors:
        reasons.append("evaluation_case_errors")
    if any(r["status"] == "contaminated" for r in overlaps.values()):
        reasons.append("contaminated_evaluation")
    if any(not r["history_complete"] for r in overlaps.values()):
        reasons.append("unknown_training_or_tuning_history")
    if any(r["status"] != "pass" for r in compute.values()):
        reasons.append("compute_unverified_or_overrun")
    if baseline["true_positives"] + baseline["false_negatives"] == 0:
        reasons.append("no_positive_references")
    if any(
        delta["newly_lost_reference_ids"]
        for case in comparisons.values() for delta in case.values()
    ):
        reasons.append("true_branches_lost")
    if any(
        delta["delta"]["false_positives"] > 0
        for case in comparisons.values() for delta in case.values()
    ):
        reasons.append("false_positives_increased")
    improved = (
        variant["true_positives"] > baseline["true_positives"]
        or variant["false_positives"] < baseline["false_positives"]
    )
    if not improved:
        reasons.append("no_discovery_improvement")
    if intervals["units"] < 2:
        reasons.append("insufficient_independent_units")
    else:
        paired = intervals["intervals"]["paired_difference"]
        if not (
            paired["true_positives"]["low"] >= 0 and paired["false_positives"]["high"] <= 0
            and (paired["true_positives"]["low"] > 0 or paired["false_positives"]["high"] < 0)
        ):
            reasons.append("paired_improvement_not_supported")
    return {
        "pass": not reasons, "reasons": reasons, "scope": "conservative_complete_expert_research_gate",
        "clinical_accuracy_claim": False, "production_detector_changed": False,
        "policy": "no lost references or per-case FP increases at any reported ostium tolerance",
    }


def _input_hashes(config: dict) -> dict[str, str]:
    paths = [config["references"], config["baseline"], *config["variants"].values()]
    paths.extend(config.get("proposals", {}).values())
    paths.extend(config.get("image_paths", {}).values())
    hashes = {}
    for value in paths:
        path = Path(value)
        for file in sorted(path.rglob("*.json")) if path.is_dir() else [path]:
            if file.is_file():
                hashes[str(file.resolve())] = _digest(file)
    return hashes


def compare(config: dict, root: Path | None = None) -> dict:
    """Compare immutable prediction inputs; return errors without dropping missing cases."""
    config = json.loads(json.dumps(config, allow_nan=False))
    if config.get("schema_version") != 1:
        raise ValueError("Comparison manifest schema_version must be 1.")
    if config.get("reference_provenance") not in PROVENANCES or not isinstance(config.get("references_complete"), bool):
        raise ValueError("Explicit reference_provenance and references_complete are required.")
    if not isinstance(config.get("variants"), dict) or not config["variants"] or "baseline" in config["variants"]:
        raise ValueError("Supply named variants; 'baseline' is reserved.")
    root = root or Path.cwd()
    for key in ("baseline", "references"):
        config[key] = str((root / config[key]).resolve())
    for key in ("variants", "proposals", "image_paths"):
        config[key] = {name: str((root / value).resolve()) for name, value in config.get(key, {}).items()}
    hashes = _input_hashes(config)
    if "input_sha256" in config and config["input_sha256"] != hashes:
        raise ValueError("Comparison input hashes changed; create a new manifest for a new evaluation.")
    source_hashes = {
        name: _digest(Path(__file__).with_name(name))
        for name in ("detector.py", "evaluate.py", "score_references.py", "nifti_io.py", "research_validation.py")
    }
    if "evaluator_source_sha256" in config and config["evaluator_source_sha256"] != source_hashes:
        raise ValueError("Evaluator source hashes changed; create a new manifest for a new evaluation.")
    images = {normalize_case_id(c): read_nifti(p) for c, p in config["image_paths"].items()}
    refs, errors, notes = ingest_references(
        Path(config["references"]), config["reference_provenance"], config["references_complete"], images,
    )
    expected = {normalize_case_id(c) for c in config.get("expected_cases", refs)}
    for case in sorted(expected - set(refs)):
        errors.append({"case_id": case, "error": "Missing valid reference case."})
    for case in sorted(set(refs) - expected):
        errors.append({"case_id": case, "error": "Unexpected reference case."})
    supplied_groups = config.get("groups")
    if supplied_groups is None and any("group_id" in r["_source"] for r in refs.values()):
        supplied_groups = {c: r["_source"].get("group_id") for c, r in refs.items()}
    groups = {c: c for c in refs}
    if supplied_groups is not None:
        normalized_groups = {normalize_case_id(c): g for c, g in supplied_groups.items()}
        if (
            len(normalized_groups) != len(supplied_groups) or set(normalized_groups) != set(refs)
            or not all(isinstance(g, str) and g for g in normalized_groups.values())
        ):
            raise ValueError("Supplied seed groups must cover exactly every valid reference case.")
        source_groups: dict[str, set[str]] = {}
        for case, ref in refs.items():
            if "group_id" in ref["_source"]:
                source_groups.setdefault(str(ref["_source"]["group_id"]), set()).add(normalized_groups[case])
        if any(len(values) > 1 for values in source_groups.values()):
            raise ValueError("Supplied groups split an existing reference seed group.")
        groups = normalized_groups
    primary = float(config.get("tolerance_mm", 3))
    if not np.isfinite(primary) or primary <= 0:
        raise ValueError("Ostium matching tolerance must be positive and finite.")
    tolerances = sorted({2.0, 3.0, 5.0, primary})
    runs = {"baseline": config["baseline"], **config["variants"]}
    for key in ("proposals", "histories", "runtime_reports", "prediction_sources"):
        if set(config.get(key, {})) - set(runs):
            raise ValueError(f"{key} contains an unknown run name.")
    run_reports = {}
    for name, directory in runs.items():
        predictions, run_errors = _predictions(Path(directory), refs)
        proposals: dict[str, dict] = {}
        if name in config["proposals"]:
            proposals, proposal_errors = _predictions(Path(config["proposals"][name]), refs)
            run_errors.extend(dict(e, input="proposals") for e in proposal_errors)
        cases = {}
        for case_id, reference in refs.items():
            if case_id not in predictions:
                cases[case_id] = {"status": "error", "family": reference["_family"], "by_tolerance": {}}
                continue
            by_tolerance = {}
            for tolerance in tolerances:
                result = _evaluate(predictions[case_id], reference, tolerance)
                proposal_result = _evaluate(proposals[case_id], reference, tolerance) if case_id in proposals else None
                result["proposal_misses"] = (
                    proposal_result["unmatched_reference_ids"] if proposal_result is not None else None
                )
                result["unproposed_reference_misses"] = result["proposal_misses"]
                by_tolerance[f"{tolerance:g}"] = result
            cases[case_id] = {
                "status": "scored", "family": reference["_family"], "by_tolerance": by_tolerance,
                "prediction": predictions[case_id],
            }
        scored = [row for row in cases.values() if row["status"] == "scored"]
        run_reports[name] = {
            "cases": cases, "errors": run_errors,
            "summary_by_tolerance_mm": {
                f"{t:g}": _summary([r["by_tolerance"][f"{t:g}"] for r in scored]) for t in tolerances
            },
            "by_family": {
                family: {
                    f"{t:g}": _summary([r["by_tolerance"][f"{t:g}"] for r in scored if r["family"] == family])
                    for t in tolerances
                } for family in sorted({ref["_family"] for ref in refs.values()})
            },
            "proposal_misses_available": name in config["proposals"],
            "missing_case_ids": sorted(set(refs) - set(predictions)),
            "family_case_coverage": {
                family: {
                    "expected_case_ids": sorted(c for c in refs if refs[c]["_family"] == family),
                    "missing_case_ids": sorted(c for c in refs if refs[c]["_family"] == family and c not in predictions),
                } for family in sorted({r["_family"] for r in refs.values()})
            },
            "overlap": overlap_report(config.get("histories", {}).get(name), sorted(refs), groups),
            "compute": _compute_guard(config.get("runtime_reports", {}).get(name), config.get("compute_budget"), sorted(refs)),
            "prediction_source": config.get("prediction_sources", {}).get(name),
            "all_cases_scored": not run_errors and not errors,
        }
    comparisons = {}
    baseline = run_reports["baseline"]
    for name in config["variants"]:
        variant = run_reports[name]
        paired_cases = sorted(
            c for c in refs if baseline["cases"][c]["status"] == variant["cases"][c]["status"] == "scored"
        )
        differences = {
            c: {f"{t:g}": _changes(baseline["cases"][c]["by_tolerance"][f"{t:g}"],
                                 variant["cases"][c]["by_tolerance"][f"{t:g}"]) for t in tolerances}
            for c in paired_cases
        }
        intervals = {
            f"{t:g}": paired_bootstrap(
                {c: baseline["cases"][c]["by_tolerance"][f"{t:g}"] for c in paired_cases},
                {c: variant["cases"][c]["by_tolerance"][f"{t:g}"] for c in paired_cases},
                groups, config.get("bootstrap_samples", 2000), config.get("bootstrap_seed", 42),
            ) for t in tolerances
        }
        comparisons[name] = {
            "cases": differences, "paired_case_ids": paired_cases,
            "by_family": {
                family: {
                    f"{t:g}": _family_changes(
                        differences, [c for c in paired_cases if refs[c]["_family"] == family], f"{t:g}",
                    ) for t in tolerances
                } for family in sorted({r["_family"] for r in refs.values()})
            },
            "excluded_case_ids": sorted(set(refs) - set(paired_cases)),
            "bootstrap_unit": "supplied_seed_group" if supplied_groups is not None else "case",
            "bootstrap_by_tolerance_mm": intervals,
            "promotion": promotion_decision(
                config["reference_provenance"], all(r["_complete"] for r in refs.values()),
                errors + baseline["errors"] + variant["errors"],
                {r: run_reports[r]["overlap"] for r in ("baseline", name)},
                {r: run_reports[r]["compute"] for r in ("baseline", name)},
                differences, intervals[f"{primary:g}"],
                baseline["summary_by_tolerance_mm"][f"{primary:g}"], variant["summary_by_tolerance_mm"][f"{primary:g}"],
            ),
        }
    manifest = dict(config, input_sha256=hashes, evaluator_source_sha256=source_hashes)
    if _input_hashes(config) != hashes:
        raise ValueError("Comparison inputs changed during evaluation.")
    if any(_digest(Path(__file__).with_name(name)) != digest for name, digest in source_hashes.items()):
        raise ValueError("Evaluator sources changed during evaluation.")
    provenance = config["reference_provenance"]
    complete = bool(refs) and not errors and config["references_complete"] and all(r["_complete"] for r in refs.values())
    return {
        "schema_version": 1, "manifest": manifest, "manifest_sha256": _object_digest(manifest),
        "reference_provenance": provenance, "references_complete": complete,
        "metric_scope": (
            "candidate_pseudo_agreement" if provenance == "candidate_pseudo"
            else "incomplete_reference_agreement" if not complete
            else "analytic_synthetic_detection" if provenance == "analytic_synthetic"
            else "complete_expert_reference_detection"
        ),
        "clinical_accuracy_claim": False,
        "evaluator_environment": {
            "python": sys.version, "platform": sys.platform,
            "packages": {name: version(name) for name in ("numpy", "scipy", "SimpleITK")},
        },
        "matching": "evaluate.evaluate_case one-to-one ostium only; no angular or seed gates",
        "references": refs, "reference_errors": errors, "normalization_notes": notes,
        "runs": run_reports, "comparisons": comparisons,
        "detector_source_note": "evaluator_source_sha256 hashes current code, not the origin of supplied predictions",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError("Report already exists; choose a new output.")
        report = compare(_read(args.manifest), args.manifest.resolve().parent)
        inputs = report["manifest"]
        directories = [inputs["baseline"], *inputs["variants"].values(), *inputs["proposals"].values()]
        if Path(inputs["references"]).is_dir():
            directories.append(inputs["references"])
        if any(args.output.resolve().is_relative_to(Path(d)) for d in directories):
            raise ValueError("Output must be outside reference/prediction input directories.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as destination:
            destination.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({name: row["promotion"] for name, row in report["comparisons"].items()}, indent=2))
        return 2 if report["reference_errors"] or any(r["errors"] for r in report["runs"].values()) else 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, OverflowError) as error:
        print(f"research_validation: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
