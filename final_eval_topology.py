"""Auditable topology experiments; references enter diagnosis/scoring only."""

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

import detector
from detector import (
    Branch, Candidate, DetectorConfig, NormalizedScan, VesselEvidence, enhance,
    normalize, on_path, physical_points, propose, same_trunk, validate_geometry,
)
from final_evaluation import (
    CASES, ROOT, case_paths, digest, load_reference, read_json, score_prediction,
    score_variant, summarize, write_json,
)
from nifti_io import read_nifti
from research_resources import measure
from stress import generate_case as stress_case
from synthetic import generate_case as synthetic_case

OUTPUT = ROOT / "labels" / "final-eval" / "topology"
DETECTOR_SHA256 = "9f96f3eb3c06f5e06d5b7db79b199be70e742130fc3885243d25d066adc1c92e"
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS")
STRESS_FAMILIES = (
    "negative_controls_only", "common_trunk_early_split", "nearby_pair_with_short_stub",
    "daughter_of_daughter", "wall_parallel_descending", "tortuous_low_contrast",
    "imperfect_parent_mask", "cropped_short_segment", "touching_vein_and_calcification",
)
STRESS_SEEDS = (31415, 27182)


def shell_labels(scan: NormalizedScan, evidence: VesselEvidence, config: DetectorConfig) -> np.ndarray:
    shell = (
        evidence.support & ~evidence.excluded
        & (scan.outside >= config.shell_inner_mm) & (scan.outside <= config.shell_outer_mm)
        & (evidence.tubular >= config.shell_vesselness_floor)
    )
    return ndi.label(shell, structure=np.ones((3, 3, 3), dtype=int))[0]


def proximal_roots(
    scan: NormalizedScan, evidence: VesselEvidence, config: DetectorConfig, separation_mm: float,
) -> tuple[list[Candidate], list[str]]:
    """Bounded alternate starts in each supported proximal wall contact."""
    if not np.isfinite(separation_mm) or separation_mm < 2:
        raise ValueError("Alternate roots require at least 2 mm separation.")
    labels = shell_labels(scan, evidence, config)
    candidates: list[Candidate] = []
    for label_id, region in enumerate(ndi.find_objects(labels), 1):
        if region is None:
            continue
        points = np.argwhere(labels[region] == label_id) + np.array([s.start for s in region])
        volume = len(points) * config.spacing_mm**3
        if volume < np.pi * config.minimum_radius_mm**2 * config.spacing_mm:
            continue
        points = points[scan.outside[tuple(points.T)] <= config.root_depth_mm]
        quality = evidence.radius[tuple(points.T)] * (0.3 + evidence.tubular[tuple(points.T)])
        selected: list[np.ndarray] = []
        for index in np.argsort(-quality, kind="stable"):
            if len(selected) >= config.roots_per_contact:
                break
            if all(np.linalg.norm(points[index] - old) * config.spacing_mm >= separation_mm for old in selected):
                selected.append(points[index])
                candidates.append((float(quality[index]), points[index], volume))
    candidates.sort(key=lambda c: (-c[0], tuple(c[1])))
    return candidates, (["Candidate limit reached; weaker alternate roots omitted."]
                        if len(candidates) > config.maximum_candidates else [])


def traced_resolve(
    scan: NormalizedScan, evidence: VesselEvidence, candidates: list[Candidate], config: DetectorConfig,
) -> tuple[list[Branch], list[dict], dict[str, int]]:
    """Replay the frozen resolver, retaining rejected geometry and suppressor identity."""
    accepted: list[tuple[Branch, int]] = []
    records: list[dict] = []
    rejections: Counter[str] = Counter()
    for index, (quality, root, volume) in enumerate(candidates):
        record: dict = {
            "rank": index, "root_zyx": root.tolist(),
            "root_lps_mm": physical_points(scan.grid, [root])[0].tolist(),
            "proposal_quality": quality, "contact_volume_mm3": volume,
            "branch": None, "suppressor_rank": None,
        }
        branch = None
        reason = ""
        if index >= config.maximum_candidates:
            reason = "candidate_limit"
        elif volume > config.broad_contact_mm3:
            reason = "broad_wall_contact"
        else:
            branch, reason = detector._trace(
                root, scan.outside, scan.parent, evidence.support, evidence.radius,
                evidence.tubular, evidence.junctions, evidence.signed_distance,
                evidence.context, volume, scan.grid, config,
            )
        if branch is not None:
            record["branch"] = asdict(branch)
            for old, old_index in accepted:
                separation = float(np.linalg.norm(np.subtract(branch.ostium_xyz_mm, old.ostium_xyz_mm)))
                if separation >= max(2.5, branch.radius_mm + old.radius_mm):
                    continue
                if separation < 2.5 and np.linalg.norm(np.subtract(branch.seed_xyz_mm, old.seed_xyz_mm)) < 3:
                    reason = "same_opening_and_path"
                elif same_trunk(branch, old):
                    reason = "common_trunk"
                if reason:
                    record["suppressor_rank"] = old_index
                    break
            if not reason and config.parallel_clearance_mm > 0:
                upstream = [(old, old_index) for old, old_index in accepted
                            if on_path(branch, old, config.spacing_mm)]
                if upstream:
                    reason = "opening_on_another_path"
                    record["suppressor_rank"] = upstream[0][1]
                else:
                    downstream = [(old, old_index) for old, old_index in accepted
                                  if on_path(old, branch, config.spacing_mm)]
                    for old, old_index in downstream:
                        accepted = [(b, i) for b, i in accepted if i != old_index]
                        records[old_index]["reason"] = "opening_on_another_path"
                        records[old_index]["suppressor_rank"] = index
                        rejections["opening_on_another_path"] += 1
            if not reason:
                accepted.append((branch, index))
        record["reason"] = reason or "accepted"
        if reason and reason != "candidate_limit":
            rejections[reason] += 1
        records.append(record)
    accepted.sort(key=lambda pair: (
        pair[0].ostium_xyz_mm[2], pair[0].ostium_xyz_mm[1], pair[0].ostium_xyz_mm[0],
    ))
    for number, (branch, index) in enumerate(accepted, 1):
        branch.instance_id = f"branch_{number:03d}"
        records[index]["branch"] = asdict(branch)
    return [branch for branch, _ in accepted], records, dict(rejections)


def infer(
    image: sitk.Image, mask: sitk.Image, config: DetectorConfig, root_separation_mm: float = 0,
) -> tuple[dict, dict, NormalizedScan, VesselEvidence]:
    start = perf_counter()
    validate_geometry(image, mask)
    scan = normalize(image, mask, config)
    normalized = perf_counter()
    evidence = enhance(scan, config)
    enhanced = perf_counter()
    candidates, warnings = (
        proximal_roots(scan, evidence, config, root_separation_mm)
        if root_separation_mm else propose(scan, evidence, config)
    )
    proposed = perf_counter()
    branches, records, rejections = traced_resolve(scan, evidence, candidates, config)
    elapsed = perf_counter() - start
    prediction = {"parent": {"instance_id": "aorta"}, "daughters": [b.prediction() for b in branches]}
    labels = shell_labels(scan, evidence, config)
    for record, (_, root, _) in zip(records, candidates):
        record["shell_component"] = int(labels[tuple(root)])
    return prediction, {
        "configuration": asdict(config), "root_separation_mm": root_separation_mm, "blood_model": scan.blood,
        "warnings": scan.warnings + warnings, "candidates": records,
        "rejections": rejections,
        "runtime_s": elapsed, "peak_rss_mb": None,
        "timings": {"normalize_s": normalized - start, "enhance_s": enhanced - normalized,
                    "propose_s": proposed - enhanced, "resolve_s": start + elapsed - proposed},
        "working_geometry": {
            "size_xyz": scan.grid.GetSize(), "spacing_xyz_mm": scan.grid.GetSpacing(),
            "origin_lps_mm": scan.grid.GetOrigin(), "direction": scan.grid.GetDirection(),
        },
    }, scan, evidence


def index_points(grid: sitk.Image, points: list | np.ndarray) -> np.ndarray:
    return np.asarray([grid.TransformPhysicalPointToContinuousIndex(list(p)) for p in points])[:, ::-1]


def sample_field(field: np.ndarray, points: np.ndarray, order: int = 1) -> list:
    return ndi.map_coordinates(field.astype(float), points.T, order=order,
                               mode="constant", cval=-1).tolist()


def resample_guide(points: list, step_mm: float = 0.5) -> np.ndarray:
    result: list[np.ndarray] = []
    for start, stop in zip(np.asarray(points)[:-1], np.asarray(points)[1:]):
        steps = max(1, int(np.ceil(np.linalg.norm(stop - start) / step_mm)))
        result.extend(np.linspace(start, stop, steps, endpoint=False))
    result.append(np.asarray(points)[-1])
    return np.asarray(result)


def diagnose(
    reference: dict, prediction: dict, audit: dict, scan: NormalizedScan,
    evidence: VesselEvidence, config: DetectorConfig,
) -> dict:
    labels = shell_labels(scan, evidence, config)
    shell = np.argwhere(labels > 0)
    shell_tree = cKDTree(shell * config.spacing_mm) if len(shell) else None
    support_labels, _ = ndi.label(evidence.support, structure=np.ones((3, 3, 3)))
    parent_components = np.unique(support_labels[scan.parent])
    parent_components = parent_components[parent_components > 0]
    records = audit["candidates"]
    roots = np.array([record["root_lps_mm"] for record in records]).reshape(-1, 3)
    comparisons = score_prediction(prediction, reference, 3)
    matched_references = {row["reference_id"] for row in comparisons["matches"]}
    matched_predictions = {row["prediction_id"] for row in comparisons["matches"]}
    references = []
    for daughter in reference["daughters"]:
        ostium = np.asarray(daughter["ostium_xyz_mm"])
        guide = resample_guide(daughter.get("centerline_xyz_mm") or [
            daughter["ostium_xyz_mm"], daughter["seed_xyz_mm"],
        ])
        points = index_points(scan.grid, guide)
        ostium_index = index_points(scan.grid, [ostium])[0]
        in_bounds = np.all((points >= 0) & (points <= np.array(scan.parent.shape) - 1), axis=1)
        shell_distance, shell_index = shell_tree.query(ostium_index * config.spacing_mm) if shell_tree else (None, None)
        component = int(labels[tuple(shell[int(shell_index)])]) if shell_index is not None else None
        nearest: list[int] = np.argsort(np.linalg.norm(roots - ostium, axis=1))[:8].tolist() if len(roots) else []
        guide_support = np.asarray(sample_field(evidence.support, points, 0))
        guide_outside = np.asarray(sample_field(scan.outside, points))
        component_values = np.asarray(sample_field(support_labels, points, 0), dtype=int)
        near = []
        for rank in nearest:
            candidate = records[int(rank)]
            branch = candidate["branch"]
            near.append({
                "rank": int(rank), "root_to_ostium_mm": float(np.linalg.norm(roots[rank] - ostium)),
                "root_to_guide_mm": float(cKDTree(guide).query(roots[rank])[0]),
                "same_shell_component": candidate["shell_component"] == component,
                "reason": candidate["reason"], "suppressor_rank": candidate["suppressor_rank"],
                "traced_ostium_error_mm": float(np.linalg.norm(
                    np.subtract(branch["ostium_xyz_mm"], ostium),
                )) if branch else None,
            })
        observations = []
        cap = sample_field(evidence.excluded, np.array([ostium_index]), 0)[0]
        if not in_bounds.all():
            observations.append("reference_guide_outside_normalized_crop")
        if cap == 1:
            observations.append("ostium_in_parent_cap_exclusion")
        if shell_distance is None or shell_distance > 5:
            observations.append("no_shell_support_within_5mm_of_reference")
        same = [row for row in records if row["shell_component"] == component]
        if same and all(np.linalg.norm(np.subtract(row["root_lps_mm"], ostium)) > 5 for row in same):
            observations.append("nearest_shell_component_roots_displaced_more_than_5mm")
        if float(np.mean(guide_support > 0)) < 0.5:
            observations.append("less_than_half_reference_guide_has_enhanced_support")
        external = guide_outside[guide_outside > 0]
        if len(external) and external.max() < config.minimum_path_mm + config.spacing_mm / 2:
            observations.append("guide_stays_below_radial_endpoint_clearance")
        if daughter["radius_mm"] is None:
            observations.append("reference_radius_unknown_not_evidence_of_small_origin")
        references.append({
            "reference_id": daughter["instance_id"], "matched_at_3mm": daughter["instance_id"] in matched_references,
            "ostium_lps_mm": ostium.tolist(), "ostium_working_zyx": ostium_index.tolist(),
            "known_reference_radius_mm": daughter["radius_mm"], "observations": observations,
            "nearest_shell_distance_mm": shell_distance, "nearest_shell_component": component,
            "component_root_ranks": [row["rank"] for row in same], "nearby_candidates": near,
            "guide_samples": {
                "lps_mm": guide.tolist(), "inside_working_crop": in_bounds.tolist(),
                "smoothed_hu": sample_field(scan.smooth, points),
                "tubularity": sample_field(evidence.tubular, points),
                "enhanced_support": guide_support.tolist(), "outside_parent_mm": guide_outside.tolist(),
                "cap_excluded": sample_field(evidence.excluded, points, 0),
                "parent_connected_support": np.isin(component_values, parent_components).tolist(),
            },
        })
    extras = []
    ref_points = np.array([d["ostium_xyz_mm"] for d in reference["daughters"]])
    for daughter in prediction["daughters"]:
        if daughter["instance_id"] in matched_predictions:
            continue
        candidate = next(row for row in records if row["reason"] == "accepted"
                         and row["branch"]["instance_id"] == daughter["instance_id"])
        extras.append({
            "prediction_id": daughter["instance_id"], "candidate_rank": candidate["rank"],
            "nearest_reference_mm": float(np.min(np.linalg.norm(ref_points - daughter["ostium_xyz_mm"], axis=1))),
            "features": candidate["branch"]["features"], "warnings": candidate["branch"]["warnings"],
            "interpretation": "Unmatched local-reference origin; incomplete references do not prove an anatomical false positive.",
        })
    return {"references": references, "extra_origins_at_3mm": extras, "comparison_3mm": comparisons}


def source_hashes() -> dict:
    return {name: digest(ROOT / name) for name in (
        "detector.py", "final_eval_topology.py", "final_evaluation.py", "evaluate.py",
        "nifti_io.py", "research_validation.py", "score_references.py", "synthetic.py",
        "requirements.txt", "requirements-dev.txt", "requirements-resources.txt",
        "stress.py", "research_resources.py",
    )}


def baseline_configs() -> dict[str, DetectorConfig]:
    return {
        "topology-strict-audit": DetectorConfig(),
        "topology-review-origin2-audit": DetectorConfig.review(minimum_origin_diameter_mm=2),
    }


def experiment_settings() -> dict[str, dict]:
    combined = DetectorConfig(
        roots_per_contact=6, connector_gap_fraction=0.2,
        parallel_clearance_mm=1.5, support_contrast_fraction=0.3,
    )
    settings = {
        "topology-alternate-roots": (DetectorConfig(roots_per_contact=6), 2.0),
        "topology-connector-gap": (DetectorConfig(connector_gap_fraction=0.2), 0.0),
        "topology-parallel-path": (DetectorConfig(parallel_clearance_mm=1.5), 0.0),
        "topology-contrast-support": (DetectorConfig(support_contrast_fraction=0.3), 0.0),
        "topology-combined": (combined, 2.0),
        "topology-combined-no-roots": (replace(combined, roots_per_contact=1), 0.0),
        "topology-combined-no-gap": (replace(combined, connector_gap_fraction=0), 2.0),
        "topology-combined-no-parallel": (replace(combined, parallel_clearance_mm=0), 2.0),
        "topology-combined-no-contrast": (replace(combined, support_contrast_fraction=0.5), 2.0),
    }
    return {
        name: {"detector": asdict(config), "alternate_root_separation_mm": separation,
               "development_status": "POST-REFERENCE DEVELOPMENT",
               "model_hashes": {}, "minimum_origin_diameter_mm": 2,
               "origin_uncertainty_policy": "Reject only measured upper diameter below 2 mm; unknown stays unknown."}
        for name, (config, separation) in settings.items()
    }


def freeze(output: Path) -> None:
    path = output / "experiment_config.json"
    if path.exists():
        raise ValueError("Config already frozen; do not overwrite scored settings.")
    initial = read_json(output / "baseline_report.json")
    write_json(path, {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_status": "POST-REFERENCE DEVELOPMENT", "eligible_for_selection": False,
        "source_hashes": source_hashes(), "initial_diagnosis_sha256": digest(output / "baseline_report.json"),
        "initial_diagnosis_variants": [v["name"] for v in initial["variants"]],
        "hypotheses": [
            "Disconnected straight wall connectors may reflect limited contrast support: test a bounded 20% gap allowance.",
            "Wall-parallel reference guides and no_supported_5mm_path motivate the existing 1.5 mm geodesic-clearance fallback.",
            "Displaced roots and strict/review disagreement motivate <=6 proximal starts per shell component, spaced >=2 mm.",
            "Low enhanced support along several reference guides motivates a 0.3 contrast-fraction ablation.",
            "Combined and leave-one-component-out variants distinguish recovery sources from increased false origins.",
        ],
        "variants": experiment_settings(),
        "synthetic": {"stress_families": STRESS_FAMILIES, "stress_seeds": STRESS_SEEDS,
                      "simple_scenarios": list(range(6)), "simple_seed": 20260913},
        "selection": "No weights fitted, no thresholds chosen by all-five scores; all variants remain ineligible.",
        "exclusions": [
            "No radius-based reference exclusion; only three real radii are measured.",
            "No reference-guided root generation or fixed output counts.",
            "No new wall connector through unsupported tissue; gap tolerance is an explicit ablation, not proof of connectivity.",
        ],
    })


def load_frozen(output: Path) -> dict:
    frozen = read_json(output / "experiment_config.json")
    if frozen["source_hashes"] != source_hashes():
        raise ValueError("Frozen experiment source hash mismatch; do not silently run modified algorithms.")
    return frozen


def run_worker(output: Path, name: str, case: str) -> None:
    setting = load_frozen(output)["variants"][name]
    image_path, mask_path = case_paths(case)
    image, mask = read_nifti(str(image_path)), read_nifti(str(mask_path))
    config = DetectorConfig(**setting["detector"])
    prediction, audit, scan, evidence = infer(image, mask, config, setting["alternate_root_separation_mm"])
    prediction["case_id"] = case
    audit["diagnosis"] = diagnose(load_reference(case), prediction, audit, scan, evidence, config)
    write_json(output / name / f"{case}.json", prediction)
    write_json(output / "diagnosis" / name / f"{case}.json", audit)
    print(case, name, len(prediction["daughters"]), audit["rejections"], flush=True)


def run_experiments(output: Path) -> None:
    frozen = load_frozen(output)
    report: dict = {
        "family": "topology", "development_status": "POST-REFERENCE DEVELOPMENT",
        "variants": [], "failures": [], "source_hashes": source_hashes(), "model_hashes": {},
        "frozen_config_sha256": digest(output / "experiment_config.json"),
        "input_hashes": read_json(output / "baseline_report.json")["input_hashes"],
        "reference_hashes": read_json(output / "baseline_report.json")["reference_hashes"],
        "metrics": "Local one-to-one 2/3/5 mm comparisons, not an official challenge score.",
        "runtime_definition": "Inference stages only; peak RSS is sampled process-tree upper bound including loading/diagnosis/serialization. Full command time in resources.",
        "exclusions": frozen["exclusions"],
    }
    for name, setting in frozen["variants"].items():
        predictions = {}
        runtimes = {}
        for case in CASES:
            resources_path = output / "resources" / name / f"{case}.json"
            if resources_path.exists():
                resources = read_json(resources_path)
            else:
                resources = measure([
                    sys.executable, str(Path(__file__).resolve()), "worker",
                    "--output", str(output), "--variant", name, "--case", case,
                ], case_id=case, cores=4)
                write_json(resources_path, resources)
            if resources["exit_code"]:
                report["failures"].append({"variant": name, "case_id": case, "resources": str(resources_path),
                                           "exit_code": resources["exit_code"]})
                continue
            audit = read_json(output / "diagnosis" / name / f"{case}.json")
            predictions[case] = read_json(output / name / f"{case}.json")
            runtimes[case] = {"runtime_s": audit["runtime_s"],
                              "peak_rss_mb": resources["cases"][case]["peak_rss_mb"]}
        if set(predictions) == set(CASES):
            report["variants"].append({
                "name": name, "prediction_dir": str((output / name).relative_to(ROOT)),
                "configuration": setting, "eligible_for_selection": False,
                "ineligible_reason": "POST-REFERENCE DEVELOPMENT; algorithms designed after inspecting these five references.",
                "scores": score_variant(predictions), "runtime": runtimes,
            })
        write_json(output / "report.json", report)
    if report["failures"]:
        raise RuntimeError("Experiment failures recorded; no invalid variants scored.")


def run_synthetic(output: Path) -> None:
    frozen = load_frozen(output)
    cohort = frozen["synthetic"]
    specs = [
        ("stress", family, seed) for seed in cohort["stress_seeds"] for family in cohort["stress_families"]
    ] + [("simple", scenario, cohort["simple_seed"]) for scenario in cohort["simple_scenarios"]]
    settings = {
        **{name: {"detector": asdict(config), "alternate_root_separation_mm": 0}
           for name, config in baseline_configs().items()},
        **frozen["variants"],
    }
    results: dict[str, list] = {name: [] for name in settings}
    failures = []
    case_records = []
    for kind, family, seed in specs:
        case = stress_case(family, seed) if kind == "stress" else synthetic_case(family, seed)
        case_id = case.reference["case_id"]
        for daughter, geometry in zip(case.reference["daughters"], case.provenance["geometry"]):
            daughter["centerline_xyz_mm"] = geometry["proximal_centerline_xyz_mm"]
        case_records.append({"reference": case.reference, "provenance": case.provenance})
        for name, setting in settings.items():
            checkpoint = output / "synthetic" / name / f"{case_id}.json"
            try:
                if checkpoint.exists():
                    record = read_json(checkpoint)
                else:
                    prediction, audit, scan, evidence = infer(
                        case.image, case.parent, DetectorConfig(**setting["detector"]),
                        setting["alternate_root_separation_mm"],
                    )
                    prediction["case_id"] = case_id
                    scores = {str(tolerance): score_prediction(prediction, case.reference, tolerance)
                              for tolerance in (2, 3, 5)}
                    record = {"prediction": prediction, "audit": audit, "scores": scores}
                    write_json(checkpoint, record)
                    del scan, evidence
                results[name].append(record)
            except Exception as error:
                failures.append({"variant": name, "case_id": case_id,
                                 "error": f"{type(error).__name__}: {error}"})
        print("synthetic", case_id, {name: len(rows[-1]["prediction"]["daughters"])
                                    for name, rows in results.items() if rows}, flush=True)
        write_json(output / "synthetic_report.json", {
            "cohort": case_records, "source_hashes": source_hashes(),
            "frozen_config_sha256": digest(output / "experiment_config.json"), "failures": failures,
            "variants": [{
                "name": name, "completed_cases": len(rows),
                "scores": {str(tolerance): summarize([row["scores"][str(tolerance)] for row in rows])
                           for tolerance in (2, 3, 5)},
                "runtime_s": {row["prediction"]["case_id"]: row["audit"]["runtime_s"] for row in rows},
            } for name, rows in results.items()],
            "interpretation": "Fixed procedural regression cohort, not independent clinical evaluation. Failures remain visible.",
        })
    if failures:
        raise RuntimeError("Synthetic failures recorded; incomplete variants require review.")


def run_baselines(output: Path) -> None:
    configs = baseline_configs()
    report: dict = {
        "family": "topology", "stage": "initial_diagnosis", "variants": [], "failures": [],
        "source_hashes": source_hashes(), "models": [],
        "method": "Frozen detector stages with instrumented faithful resolver; references used after inference.",
        "input_hashes": {}, "reference_hashes": {}, "diagnosis_files": {},
        "metrics": "Local one-to-one 2/3/5 mm comparison, not an official weighted score.",
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "threads": {key: os.environ.get(key) for key in THREAD_KEYS}},
        "runtime_definition": "In-process normalize/enhance/propose/resolve; excludes NIfTI loading, diagnostics and scoring. RSS not sampled.",
    }
    predictions: dict[str, dict] = {name: {} for name in configs}
    runtimes: dict[str, dict] = {name: {} for name in configs}
    for case in CASES:
        image_path, mask_path = case_paths(case)
        report["input_hashes"][case] = {"image": digest(image_path), "mask": digest(mask_path)}
        report["reference_hashes"][case] = digest(ROOT / "labels/organizer-v1/references" / f"{case}.json")
        image, mask = read_nifti(str(image_path)), read_nifti(str(mask_path))
        for name, config in configs.items():
            try:
                prediction, audit, scan, evidence = infer(image, mask, config)
                prediction["case_id"] = case
                predictions[name][case] = prediction
                runtimes[name][case] = {key: audit[key] for key in ("runtime_s", "peak_rss_mb")}
                audit["diagnosis"] = diagnose(load_reference(case), prediction, audit, scan, evidence, config)
                write_json(output / name / f"{case}.json", prediction)
                write_json(output / "diagnosis" / name / f"{case}.json", audit)
                print(case, name, len(prediction["daughters"]), audit["rejections"], flush=True)
                del scan, evidence, audit
            except Exception as error:
                report["failures"].append({"case_id": case, "variant": name,
                                           "error": f"{type(error).__name__}: {error}"})
                print(case, name, repr(error), flush=True)
    for name, config in configs.items():
        if set(predictions[name]) != set(CASES):
            continue
        report["variants"].append({
            "name": name, "prediction_dir": str((output / name).relative_to(ROOT)),
            "configuration": asdict(config), "eligible_for_selection": False,
            "ineligible_reason": "Topology diagnosis family; not submitted as a clean frozen-family selection candidate.",
            "scores": score_variant(predictions[name]), "runtime": runtimes[name],
        })
        report["diagnosis_files"][name] = [
            str((output / "diagnosis" / name / f"{case}.json").relative_to(ROOT)) for case in CASES
        ]
    write_json(output / "baseline_report.json", report)
    if report["failures"]:
        raise RuntimeError("Baseline failures recorded; incomplete variants not scored.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["baseline", "freeze", "experiments", "synthetic", "worker"])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--variant")
    parser.add_argument("--case", choices=CASES)
    args = parser.parse_args()
    if digest(ROOT / "detector.py") != DETECTOR_SHA256:
        raise ValueError("Frozen detector source changed; review and revalidate before running.")
    for key in THREAD_KEYS:
        if int(os.environ.get(key, "999")) > 4:
            raise ValueError(f"Set {key} <=4 before starting Python.")
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    if args.mode == "baseline":
        run_baselines(args.output)
    elif args.mode == "freeze":
        freeze(args.output)
    elif args.mode == "experiments":
        run_experiments(args.output)
    elif args.mode == "synthetic":
        run_synthetic(args.output)
    else:
        if args.variant is None or args.case is None:
            parser.error("worker needs --variant and --case")
        run_worker(args.output, args.variant, args.case)


if __name__ == "__main__":
    main()
