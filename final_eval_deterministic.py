"""Frozen deterministic development matrix; inference never reads references."""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter, sleep
from types import FrameType
from typing import cast

import numpy as np
import SimpleITK as sitk

from detector import (
    Branch, Detection, DetectorConfig, NormalizedScan, VesselEvidence,
    detect, detect_pool, physical_points, validate_geometry,
)
from evaluate import validate_prediction
from final_evaluation import (
    CASES, ROOT, TOLERANCES, case_paths, digest, filtered, load_reference,
    read_json, score_prediction, score_variant, summarize, write_json,
)
from nifti_io import read_nifti
from paper_methods import METHODS, detect_paper

OUTPUT = ROOT / "labels/final-eval/deterministic"
BASELINE_SHA256 = "9f96f3eb3c06f5e06d5b7db79b199be70e742130fc3885243d25d066adc1c92e"
SOURCE_FILES = (
    "detector.py", "paper_methods.py", "nifti_io.py", "final_evaluation.py",
    "evaluate.py", "score_references.py", "research_validation.py",
    "requirements.txt", "requirements-dev.txt", "FINAL_EVALUATION_PROTOCOL.md",
)
THREAD_ENV = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "NUMEXPR_NUM_THREADS",
)
SCALARS = (
    "gap", "length", "before_junction", "displacement", "seed_radius",
    "measured_radius", "origin_radius", "origin_diameter", "origin_diameter_upper",
    "proximal_hu", "origin_level", "mean_vesselness", "hu_along",
)


@dataclass(frozen=True)
class Specification:
    name: str
    engine: str
    config: DetectorConfig
    method: str = ""
    category: str = "frozen_existing_algorithm"

    def configuration(self) -> dict:
        return {
            "engine": self.engine, "detector_config": asdict(self.config),
            "paper_method": self.method, "model_hashes": {},
            "strict_union_config": asdict(DetectorConfig()) if self.engine == "union" else None,
            "category": self.category, "trained_on_reference_cases": False,
            "post_reference_new_algorithm": False,
            "origin_size_eligibility_disabled": self.config.minimum_origin_diameter_mm == 0,
            "candidate_threshold": None,
        }


def matrix() -> list[Specification]:
    strict, review = DetectorConfig(), DetectorConfig.review()
    specs = [
        Specification("deterministic-strict", "detect", strict),
        Specification("deterministic-review", "detect", review),
        Specification("deterministic-review-union", "union", review),
        *(Specification(f"deterministic-paper-{method}", "paper", strict, method) for method in METHODS),
    ]
    for label, engine, default in (("strict", "detect", strict), ("review-union", "union", review)):
        for contrast in (0.3, 0.5, 0.7):
            for native in (0.0, 1.2):
                config = replace(default, support_contrast_fraction=contrast, native_contrast_scale=native)
                if config != default:
                    specs.append(Specification(
                        f"deterministic-{label}-contrast{contrast:g}-native{native:g}", engine, config,
                    ))
        for spacing in (0.75, 1.5):
            specs.append(Specification(
                f"deterministic-{label}-spacing{spacing:g}", engine, replace(default, spacing_mm=spacing),
            ))
        specs.append(Specification(
            f"deterministic-{label}-wall-parallel", engine, replace(default, parallel_clearance_mm=2.5),
        ))
    for spec in list(specs):
        if spec.name == "deterministic-strict" or spec.engine == "paper":
            specs.append(replace(
                spec, name=f"{spec.name}-origin-disabled",
                config=replace(spec.config, minimum_origin_diameter_mm=0),
            ))
    return specs


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def source_hashes() -> dict[str, str]:
    return {name: digest(ROOT / name) for name in SOURCE_FILES}


def record_key(row: dict) -> str:
    value = {key: row[key] for key in ("ostium_xyz_mm", "seed_xyz_mm", "radius_mm", "direction_xyz")}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def freeze(output: Path) -> dict:
    path = output / "matrix.json"
    if path.exists():
        raise FileExistsError("The expanded matrix is immutable; use its existing run or a new directory.")
    sources = source_hashes()
    if sources["detector.py"] != BASELINE_SHA256:
        raise ValueError("Production detector differs from the declared 4a43dd4 baseline.")
    validation = read_json(ROOT / "labels/organizer-v1/validation.json")
    verified = {row["case_id"]: row for row in validation["cases"]}
    inputs = {}
    for case in CASES:
        image, mask = case_paths(case)
        hashes = {"image_sha256": digest(image), "mask_sha256": digest(mask)}
        if any(hashes[key] != verified[case][key] for key in hashes):
            raise ValueError(f"Input identity mismatch or unresolved LFS pointer: {case}")
        inputs[case] = {"image": relative(image), "mask": relative(mask), **hashes}
    specs = matrix()
    value = {
        "schema_version": 1, "frozen_at_utc": datetime.now(UTC).isoformat(),
        "git_head_at_freeze": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        "runner_at_freeze_sha256": digest(Path(__file__)),
        "source_hashes": sources, "input_hashes": inputs,
        "reference_hashes": {
            case: digest(ROOT / f"labels/organizer-v1/references/{case}.json") for case in CASES
        },
        "cases": list(CASES), "tolerances_mm": list(TOLERANCES),
        "threads": 4, "worker_rss_limit_mb": 7168, "worker_timeout_s": 1800,
        "bootstrap_seed": 20260913, "bootstrap_replicates": 2000,
        "specifications": [{"name": spec.name, **spec.configuration()} for spec in specs],
        "derived_variants": [
            {"name": f"{spec.name}-trace-ceiling", "source": spec.name, "operation": "all_successful_traces",
             "eligible_for_selection": False}
            for spec in specs
        ],
        "selection": {
            "rule": "four-case pooled 3mm F1, count MAE, runtime dependency count, mean runtime, name",
            "dependency_count": 4,
            "eligibility": "Only existing configurations with the 2mm origin gate enabled.",
        },
        "notes": [
            "Exact expanded matrix frozen before any reference score is computed.",
            "No training, reference coordinates, target counts or case-specific thresholds enter inference.",
            "Wall-parallel clearance=2.5mm is the existing tests/test_stress.py ablation.",
            "Spacing and wall-parallel settings are standalone, not a Cartesian expansion of the contrast grid.",
            "detect_pool always adds DEFAULT strict detections, including when its review config is ablated.",
            "Origin-disabled outputs are proposal baselines, not challenge-eligible deployable algorithms.",
            "Trace ceilings include duplicates and below-size outputs if their source permits them.",
            "Roots rejected before a valid branch exists remain diagnostics, never invented prediction points.",
        ],
    }
    write_json(path, value)
    return value


def verify_matrix(output: Path) -> dict:
    frozen = read_json(output / "matrix.json")
    if source_hashes() != frozen["source_hashes"]:
        raise ValueError("Frozen detector/scorer source changed; do not mix extractions.")
    expected = [{"name": spec.name, **spec.configuration()} for spec in matrix()]
    if frozen["specifications"] != expected:
        raise ValueError("Expanded settings differ from the frozen matrix.")
    return frozen


class CandidateAudit:
    """Observe frozen function calls without replacing functions or changing results."""

    def __init__(self) -> None:
        self.groups: list[dict] = []
        self.by_root: dict[tuple, dict] = {}
        self.by_object: dict[int, dict] = {}

    def profile(self, frame: FrameType, event: str, result: object) -> None:
        if frame.f_code.co_filename != str(ROOT / "detector.py"):
            return
        name = frame.f_code.co_name
        local = frame.f_locals
        if name == "resolve" and event == "call":
            scan = cast(NormalizedScan, local["scan"])
            evidence = cast(VesselEvidence, local["evidence"])
            config = cast(DetectorConfig, local["config"])
            candidates = local["candidates"]
            rows = []
            self.by_root, self.by_object = {}, {}
            for index, (quality, root, volume) in enumerate(candidates):
                location = tuple(root)
                reason = (
                    "candidate_limit" if index >= config.maximum_candidates else
                    "broad_wall_contact" if volume > config.broad_contact_mm3 else "not_traced"
                )
                row = {
                    "proposal_index": index, "root_zyx": root.tolist(),
                    "root_xyz_mm": physical_points(scan.grid, [root])[0].tolist(),
                    "proposal_quality": float(quality), "volume_mm3": float(volume),
                    "root_support": bool(evidence.support[location]),
                    "root_radius_mm": float(evidence.radius[location]),
                    "root_vesselness": float(evidence.tubular[location]),
                    "root_hu": float(scan.smooth[location]),
                    "trace_reason": reason, "branch": None, "retained_after_resolve": False,
                }
                rows.append(row)
                self.by_root[location] = row
            self.groups.append({
                "config": asdict(config), "blood_model": scan.blood,
                "grid_size_xyz": scan.grid.GetSize(),
                "support_voxels": int(np.count_nonzero(evidence.support)),
                "candidates": rows,
            })
        elif name == "_trace" and event == "return":
            branch, reason = cast(tuple[Branch | None, str], result)
            row = self.by_root[tuple(local["root"])]
            row["trace_reason"] = reason or "trace_passed"
            row["measurements"] = {
                key: float(local[key]) if local[key] is not None else None
                for key in SCALARS if key in local
            }
            row["reachable_voxels"] = int(np.count_nonzero(np.isfinite(local["cumulative"])))
            row["supported_endpoints"] = int(np.count_nonzero(local["endpoints"]))
            if "path_mm" in local:
                row["partial_path_xyz_mm"] = local["path_mm"].tolist()
                row["estimated_ostium_xyz_mm"] = local["path_mm"][0].tolist()
            if branch is not None:
                row["branch"] = asdict(branch)
                self.by_object[id(branch)] = row
        elif name == "resolve" and event == "return":
            branches, rejections = cast(tuple[list[Branch], dict[str, int]], result)
            for branch in branches:
                row = self.by_object[id(branch)]
                row["retained_after_resolve"] = True
                row["resolved_instance_id"] = branch.instance_id
            self.groups[-1]["rejections"] = rejections.copy()
            self.groups[-1]["resolution_caveat"] = (
                "Trace-pass candidates not retained were removed by duplicate/topology resolution; "
                "the frozen API provides aggregate, not per-candidate, resolution reason counts."
            )


def peak_rss_mb(pid: int | None = None) -> float | None:
    path = Path("/proc") / (str(pid) if pid is not None else "self") / "status"
    if not path.exists():
        return None
    try:
        for line in path.read_text().splitlines():
            if line.startswith("VmHWM:"):
                return float(line.split()[1]) / 1024
    except (FileNotFoundError, ProcessLookupError):
        return None
    return None


def infer(spec: Specification, image: sitk.Image, mask: sitk.Image) -> Detection:
    if spec.engine == "union":
        return detect_pool(image, mask, spec.config)
    if spec.engine == "paper":
        return detect_paper(image, mask, spec.method, spec.config)
    return detect(image, mask, spec.config)


def worker(output: Path, name: str, case: str) -> None:
    frozen = verify_matrix(output)
    spec = next(spec for spec in matrix() if spec.name == name)
    target = output / "runs" / name / f"{case}.json"
    if target.exists():
        raise FileExistsError(target)
    image_path, mask_path = case_paths(case)
    inputs = frozen["input_hashes"][case]
    if digest(image_path) != inputs["image_sha256"] or digest(mask_path) != inputs["mask_sha256"]:
        raise ValueError("Input mismatch, including unresolved LFS pointers.")
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    started = perf_counter()
    image, mask = read_nifti(str(image_path)), read_nifti(str(mask_path))
    validate_geometry(image, mask)
    loaded = perf_counter()
    audit = CandidateAudit()
    previous = sys.getprofile()
    try:
        sys.setprofile(audit.profile)
        result = infer(spec, image, mask)
    finally:
        sys.setprofile(previous)
    finished = perf_counter()
    prediction = filtered(result.prediction(case), [branch.evidence_score for branch in result.branches], 0.0)
    validate_prediction(prediction)
    write_json(target, {
        "name": name, "case_id": case, "matrix_sha256": digest(output / "matrix.json"),
        "runner_sha256": digest(Path(__file__)), "prediction": prediction,
        "ordered_candidate_hashes": [record_key(branch) for branch in prediction["daughters"]],
        "diagnostics": result.diagnostics(), "proposal_audit": audit.groups,
        "runtime": {
            "runtime_s": finished - started, "inference_s": finished - loaded,
            "input_loading_s": loaded - started, "peak_rss_mb": peak_rss_mb(),
            "measurement": "Fresh sequential subprocess; CT loading and observational audit included.",
        },
    })
    print(f"{name} {case}: {len(result.branches)} predictions, {finished - started:.2f}s", flush=True)


def run(output: Path) -> dict:
    frozen = verify_matrix(output)
    environment = {**os.environ, **{key: "4" for key in THREAD_ENV}}
    environment["PYTHONHASHSEED"] = "0"
    failures = []
    for spec in matrix():
        for case in CASES:
            target = output / "runs" / spec.name / f"{case}.json"
            if target.exists():
                record = read_json(target)
                if record["matrix_sha256"] != digest(output / "matrix.json"):
                    raise ValueError(f"Stale result: {target}")
                continue
            command = [
                sys.executable, str(Path(__file__)), "worker", "--output-dir", str(output),
                "--name", spec.name, "--case", case,
            ]
            started = perf_counter()
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True)
            abort = ""
            while child.poll() is None:
                rss = peak_rss_mb(child.pid)
                if rss is not None and rss > frozen["worker_rss_limit_mb"]:
                    abort = "Memory guard exceeded; no empty-prediction substitution."
                if perf_counter() - started > frozen["worker_timeout_s"]:
                    abort = "Worker timeout; no empty-prediction substitution."
                if abort:
                    child.kill()
                    break
                sleep(0.2)
            stdout, stderr = child.communicate()
            print(stdout, end="", flush=True)
            if child.returncode != 0 or abort:
                failure = {
                    "name": spec.name, "case_id": case, "exit_code": child.returncode,
                    "reason": abort, "stderr": stderr[-20000:], "elapsed_s": perf_counter() - started,
                }
                failures.append(failure)
                print(json.dumps(failure), flush=True)
                write_json(output / "failures.json", {"failures": failures})
    result = {"failures": failures, "completed_at_utc": datetime.now(UTC).isoformat()}
    write_json(output / "failures.json", result)
    return result


def ceiling(record: dict) -> dict:
    daughters: list[dict] = []
    for group in record["proposal_audit"]:
        for candidate in group["candidates"]:
            branch = candidate["branch"]
            if branch is None:
                continue
            daughters.append({
                **{key: branch[key] for key in (
                    "parent_instance_id", "ostium_xyz_mm", "seed_xyz_mm", "radius_mm", "direction_xyz",
                )},
                "instance_id": f"trace_{len(daughters) + 1:04d}",
            })
    prediction = {"case_id": record["case_id"], "parent": {"instance_id": "aorta"}, "daughters": daughters}
    validate_prediction(prediction)
    return prediction


def size_status(features: dict) -> str:
    upper = features.get("origin_diameter_upper_mm")
    diameter = features.get("origin_diameter_mm")
    if upper is None:
        return "unknown"
    if upper < 2:
        return "confidently_below_2mm"
    if diameter is not None and diameter < 2:
        return "borderline_with_native_voxel_allowance"
    return "at_least_2mm"


def reference_diagnostics(record: dict, reference: dict) -> dict:
    prediction = record["prediction"]
    scores = score_prediction(prediction, reference, 3)
    branches = record["diagnostics"]["branches"]
    origins = Counter(size_status(branch["features"]) for branch in branches)
    candidates = [
        {"stage": index, **row}
        for index, group in enumerate(record["proposal_audit"]) for row in group["candidates"]
    ]
    details = []
    for target in reference["daughters"]:
        position = np.asarray(target["ostium_xyz_mm"])
        neighbors = []
        for row in candidates:
            point = row.get("estimated_ostium_xyz_mm", row["root_xyz_mm"])
            distance = float(np.linalg.norm(position - point))
            if distance > 10:
                continue
            neighbors.append({
                "distance_mm": distance,
                "position_kind": "estimated_ostium" if "estimated_ostium_xyz_mm" in row else "proposal_root",
                "stage": row["stage"], "proposal_index": row["proposal_index"],
                "trace_reason": row["trace_reason"],
                "retained_after_resolve": row["retained_after_resolve"],
                "measurements": row.get("measurements", {}),
                "root_hu": row["root_hu"], "root_vesselness": row["root_vesselness"],
                "root_radius_mm": row["root_radius_mm"],
                "volume_mm3": row["volume_mm3"],
            })
        neighbors.sort(key=lambda row: row["distance_mm"])
        details.append({
            "reference_id": target["instance_id"],
            "matched_at_3mm": target["instance_id"] not in scores["unmatched_reference_ids"],
            "nearby_candidate_evidence": neighbors,
            "interpretation": (
                "Spatial diagnostics only, not proof that a proposal is the reference vessel. "
                "A root is not an estimated ostium; no nearby proposal means failure before trace or >10mm away."
            ),
        })
    return {"origin_size_counts": dict(origins), "references": details}


def selection(variants: list[dict]) -> dict:
    eligible = [variant for variant in variants if variant["eligible_for_selection"]]

    def rank(variant: dict, cases: tuple[str, ...]) -> tuple:
        rows = [row for row in variant["scores"]["3"]["cases"] if row["case_id"] in cases]
        summary = summarize(rows)
        runtime = np.mean([variant["runtime"][case]["runtime_s"] for case in cases])
        return (-summary["f1"], summary["count_mae"], 4, float(runtime), variant["name"])

    if not eligible:
        return {"failure": "No valid eligible configurations."}
    winner = min(eligible, key=lambda variant: rank(variant, CASES))
    folds = []
    heldout: dict[str, list[dict]] = {f"{t:g}": [] for t in TOLERANCES}
    for case in CASES:
        training = tuple(item for item in CASES if item != case)
        selected = min(eligible, key=lambda variant: rank(variant, training))
        folds.append({"held_out_case": case, "selected_name": selected["name"], "selection_cases": training})
        for tolerance in heldout:
            heldout[tolerance].append(next(
                row for row in selected["scores"][tolerance]["cases"] if row["case_id"] == case
            ))
    return {
        "development_winner": winner["name"], "development_winner_is_not_heldout": True,
        "eligible_variants": [variant["name"] for variant in eligible], "folds": folds,
        "selection_counts": dict(Counter(fold["selected_name"] for fold in folds)),
        "held_out_scores": {
            tolerance: {"summary": summarize(rows), "cases": rows} for tolerance, rows in heldout.items()
        },
        "caveat": "LOCO of this frozen finite family on five reused development images; no independent hidden test.",
    }


def paired_bootstrap(variants: list[dict], selected: dict, frozen: dict) -> dict:
    if "development_winner" not in selected:
        return {}
    baseline = next(variant for variant in variants if variant["name"] == "deterministic-strict")
    winner = next(variant for variant in variants if variant["name"] == selected["development_winner"])
    rng = np.random.default_rng(frozen["bootstrap_seed"])
    samples = rng.integers(0, len(CASES), size=(frozen["bootstrap_replicates"], len(CASES)))
    intervals = {}
    for tolerance in ("2", "3", "5"):
        arrays = []
        for variant in (baseline, winner):
            rows = variant["scores"][tolerance]["cases"]
            counts = np.array([
                [row[key] for key in ("true_positives", "false_positives", "false_negatives")]
                for row in rows
            ])
            pooled = counts[samples].sum(axis=1)
            tp, fp, fn = pooled.T
            arrays.append(2 * tp / (2 * tp + fp + fn))
        intervals[tolerance] = {
            "baseline_f1_95_percentile": np.percentile(arrays[0], [2.5, 97.5]).tolist(),
            "development_winner_f1_95_percentile": np.percentile(arrays[1], [2.5, 97.5]).tolist(),
            "paired_f1_difference_95_percentile": np.percentile(arrays[1] - arrays[0], [2.5, 97.5]).tolist(),
        }
    return {
        "seed": frozen["bootstrap_seed"], "replicates": len(samples), "intervals": intervals,
        "caveat": "Paired case bootstrap of a fixed development winner; does not remove model-selection bias.",
    }


def report(output: Path) -> dict:
    frozen = verify_matrix(output)
    for case, expected in frozen["reference_hashes"].items():
        if digest(ROOT / f"labels/organizer-v1/references/{case}.json") != expected:
            raise ValueError("References changed after freezing.")
    variants: list[dict] = []
    failures: list[dict] = []
    all_diagnostics: dict[str, dict] = {}
    for spec in matrix():
        paths = {case: output / "runs" / spec.name / f"{case}.json" for case in CASES}
        missing = [case for case, path in paths.items() if not path.exists()]
        if missing:
            failures.append({"name": spec.name, "missing_cases": missing, "excluded_from_scores": True})
            continue
        records = {case: read_json(path) for case, path in paths.items()}
        for case, record in records.items():
            if record["matrix_sha256"] != digest(output / "matrix.json"):
                raise ValueError("Result belongs to a different matrix.")
            validate_prediction(record["prediction"])
            if record["prediction"]["case_id"] != case:
                raise ValueError("Result case mismatch.")
        predictions = {case: record["prediction"] for case, record in records.items()}
        runtime = {case: record["runtime"] for case, record in records.items()}
        diagnostics = {
            case: reference_diagnostics(record, load_reference(case)) for case, record in records.items()
        }
        all_diagnostics[spec.name] = diagnostics
        for operation in ("final", "trace-ceiling"):
            name = spec.name if operation == "final" else f"{spec.name}-trace-ceiling"
            values = predictions if operation == "final" else {
                case: ceiling(record) for case, record in records.items()
            }
            directory = output / "predictions" / name
            for case, prediction in values.items():
                write_json(directory / f"{case}.json", prediction)
            eligible = operation == "final" and spec.config.minimum_origin_diameter_mm >= 2
            reason = "" if eligible else (
                "Pre-resolution successful traces include duplicates; diagnostic recall ceiling only."
                if operation != "final" else
                "Origin-size eligibility disabled; may retain confidently below-2mm origins. Proposal baseline only."
            )
            if any((row["peak_rss_mb"] or 0) > 7168 for row in runtime.values()):
                eligible, reason = False, "Exceeds reserved 8GB deployment memory envelope."
            variants.append({
                "name": name, "prediction_dir": relative(directory),
                "configuration": {
                    **spec.configuration(), "operation": operation,
                    "source_hashes": frozen["source_hashes"], "threads": 4,
                },
                "eligible_for_selection": eligible, "ineligible_reason": reason,
                "scores": score_variant(values), "runtime": runtime,
                "runtime_note": "Source inference including audit; trace-ceiling replay time is not added.",
                "prediction_hashes": {case: digest(directory / f"{case}.json") for case in CASES},
            })
    chosen = selection(variants)
    compare: dict[str, dict] = {}
    baseline = next((variant for variant in variants if variant["name"] == "deterministic-strict"), None)
    if baseline is not None:
        for variant in variants:
            if variant["configuration"]["operation"] != "final":
                continue
            compare[variant["name"]] = {}
            for before, after in zip(baseline["scores"]["3"]["cases"], variant["scores"]["3"]["cases"]):
                old = {row["reference_id"] for row in before["matches"]}
                new = {row["reference_id"] for row in after["matches"]}
                compare[variant["name"]][after["case_id"]] = {
                    "lost_reference_ids_vs_strict": sorted(old - new),
                    "gained_reference_ids_vs_strict": sorted(new - old),
                }
    result = {
        "schema_version": 1, "family": "deterministic", "variants": variants,
        "matrix_path": relative(output / "matrix.json"), "matrix_sha256": digest(output / "matrix.json"),
        "source_hashes": frozen["source_hashes"], "input_hashes": frozen["input_hashes"],
        "reference_hashes": frozen["reference_hashes"], "model_hashes": {},
        "runner_sha256": digest(Path(__file__)), "generated_at_utc": datetime.now(UTC).isoformat(),
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(), "cpu": platform.processor(),
            "packages": {name: version(name) for name in (
                "SimpleITK", "numpy", "scipy", "scikit-image", "matplotlib", "ruff", "mypy", "pytest",
            )},
            "numerical_threads": 4, "worker_rss_limit_mb": 7168,
            "runtime_limits": "Linux measurements; Windows/four-core deployment not validated on organizer hardware.",
        },
        "failures": failures, "worker_failures": read_json(output / "failures.json"),
        "reference_diagnostics_path": relative(output / "reference-diagnostics.json"),
        "reference_changes_vs_strict_at_3mm": compare,
        "selection": chosen, "paired_bootstrap": paired_bootstrap(variants, chosen, frozen),
        "limitations": [
            "Judge-approved references retain AI-assisted draft provenance and possible omissions.",
            "Only three of 19 reference radii are measured; the shared scorer masks all unknown radius errors.",
            "Local 2/3/5mm one-to-one comparisons are not an official weighted challenge score.",
            "Unknown origin diameter is uncertain, not known below 2mm; frozen strict retains unknown/borderline estimates.",
            "Seed radius is not origin diameter; no radius is fabricated for reporting.",
            "All candidates, scores, features, ordering and rejected-root evidence are retained in runs/.",
            "Trace ceilings cannot recover branches rejected before a complete geometry exists.",
            "Evidence scores are uncalibrated; no classifier fitting or post-reference detector algorithm added.",
            "No hidden-test or clinical accuracy claim and no production detector change.",
        ],
        "reproduction_commands": [
            "uv venv --python 3.13.3 .venv313",
            "uv pip install --python .venv313/bin/python -r requirements-dev.txt",
            ".venv313/bin/python final_eval_deterministic.py freeze",
            "OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "
            "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4 "
            ".venv313/bin/python final_eval_deterministic.py run",
            ".venv313/bin/python final_eval_deterministic.py report",
        ],
    }
    write_json(output / "reference-diagnostics.json", all_diagnostics)
    write_json(output / "report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "worker", "report"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--name")
    parser.add_argument("--case", choices=CASES)
    args = parser.parse_args()
    if args.command == "freeze":
        print(f"Frozen {len(freeze(args.output_dir)['specifications'])} extraction configurations.")
    elif args.command == "worker":
        worker(args.output_dir, args.name, args.case)
    elif args.command == "run":
        result = run(args.output_dir)
        if result["failures"]:
            raise SystemExit(1)
    else:
        result = report(args.output_dir)
        print(f"Scored {len(result['variants'])} variants; {len(result['failures'])} failures.")


if __name__ == "__main__":
    main()
