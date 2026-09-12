"""Opt-in classical experiments derived from Danilov, Tahoces and Riffaud.

These are adaptations, not reproductions of the papers' complete pipelines.
See ADDITIONAL_PAPERS.md for equations, exclusions and evaluation limitations.
The production detector is never modified or selected through this module.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
from scipy import ndimage as ndi

from detector import (
    Candidate, Detection, DetectorConfig, NormalizedScan, VesselEvidence,
    branch_junctions, detect, enhance, normalize, parent_angle, point_tuple,
    propose, resolve, validate_geometry,
)
from evaluate import evaluate_case, summarize_cases
from nifti_io import read_nifti

METHODS = ("contact-growth", "border-cleaning", "pca-direction")
BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]
NEIGHBORS = np.ones((3, 3, 3), dtype=bool)


def clean_wall_layers(
    parent: BoolArray, support: BoolArray, spacing_mm: float, clearance_mm: float = 5.0,
) -> BoolArray:
    """Retain near-wall support with an outward chain to a Chebyshev layer.

    Layers use 26-neighbor grid distance on an isotropic grid, scaled by spacing;
    clearance_mm is not an Euclidean wall-distance or path-length criterion.
    Support at/beyond the terminal layer is the fixed anchor.
    """
    if (
        parent.ndim != 3 or parent.shape != support.shape
        or parent.dtype != np.bool_ or support.dtype != np.bool_ or not parent.any()
    ):
        raise ValueError("Layer cleaning needs matching 3D boolean masks and a nonempty parent.")
    if not np.isfinite([spacing_mm, clearance_mm]).all() or min(spacing_mm, clearance_mm) <= 0:
        raise ValueError("Layer spacing and clearance must be positive and finite.")
    ratio = clearance_mm / spacing_mm
    if not np.isfinite(ratio):
        raise ValueError("Layer count cannot be represented.")
    layers = ndi.distance_transform_cdt(~parent, metric="chessboard")
    depth = int(np.ceil(ratio))
    result = support.copy() | parent
    for distance in range(min(depth - 1, int(layers.max())), 0, -1):
        onward = ndi.binary_dilation(result & (layers == distance + 1), structure=NEIGHBORS)
        result[(layers == distance) & ~onward] = False
    return result


def proximal_pca_direction(path_xyz_mm: npt.ArrayLike, radius_mm: float) -> FloatArray:
    """Origin-anchored PCA on an arc-uniform proximal prefix of min(3r,10) mm."""
    path = np.asarray(path_xyz_mm, dtype=float)
    if (
        path.ndim != 2 or path.shape[1] != 3 or len(path) < 2
        or not np.isfinite(path).all() or not np.isfinite(radius_mm) or radius_mm <= 0
    ):
        raise ValueError("PCA requires a finite physical path and positive finite radius.")
    with np.errstate(over="ignore", invalid="ignore"):
        steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    if not np.isfinite(steps).all() or not steps.any():
        raise ValueError("PCA path needs positive finite length.")
    path = path[np.r_[True, steps > 0]]
    arc = np.r_[0.0, np.cumsum(steps[steps > 0])]
    extent = min(3 * radius_mm, 10.0, float(arc[-1]))
    samples = np.linspace(0, extent, max(3, int(np.ceil(extent / 0.5)) + 1))
    offsets = np.column_stack([np.interp(samples, arc, path[:, axis]) for axis in range(3)]) - path[0]
    eigenvalues, vectors = np.linalg.eigh(offsets.T @ offsets)
    if eigenvalues[-1] <= 0 or eigenvalues[-1] - eigenvalues[-2] <= eigenvalues[-1] * 1e-8:
        raise ValueError("PCA path has no identifiable principal direction.")
    direction = vectors[:, -1]
    orientation = float(np.sum(offsets @ direction))
    if abs(orientation) <= np.sqrt(eigenvalues[-1]) * 1e-8:
        raise ValueError("PCA path has no identifiable outward orientation.")
    return direction if orientation > 0 else -direction


def _with_support(scan: NormalizedScan, evidence: VesselEvidence, support: BoolArray,
                  config: DetectorConfig) -> VesselEvidence:
    return replace(
        evidence, support=support,
        radius=ndi.distance_transform_edt(support, sampling=config.spacing_mm).astype(np.float32),
        junctions=branch_junctions(support, scan.parent, config.spacing_mm),
    )


def _contact_candidates(
    scan: NormalizedScan, evidence: VesselEvidence, config: DetectorConfig,
) -> tuple[list[Candidate], list[str]]:
    contact = (
        evidence.support & ~scan.parent & ~evidence.excluded
        & ndi.binary_dilation(scan.parent, structure=NEIGHBORS)
    )
    labels, _ = ndi.label(contact, structure=NEIGHBORS)
    candidates: list[Candidate] = []
    for label, region in enumerate(ndi.find_objects(labels), 1):
        if region is None:
            continue
        points = np.argwhere(labels[region] == label)
        points += np.array([axis.start for axis in region])
        volume = len(points) * config.spacing_mm**3
        if volume < np.pi * config.minimum_radius_mm**2 * config.spacing_mm:
            continue
        quality = evidence.radius[tuple(points.T)]
        root = points[int(np.argmax(quality))]
        candidates.append((float(quality.max()), root, volume))
    candidates.sort(key=lambda candidate: (-candidate[0], tuple(candidate[1])))
    warnings = []
    if len(candidates) > config.maximum_candidates:
        warnings.append(f"Candidate limit reached ({config.maximum_candidates}); weaker contacts were omitted.")
    return candidates, warnings


def detect_paper(
    image: sitk.Image, mask: sitk.Image, method: str, config: DetectorConfig | None = None,
) -> Detection:
    if method not in METHODS:
        raise ValueError(f"Unknown paper method: {method}")
    config = replace(config or DetectorConfig(), profile=f"paper:{method}")
    validate_geometry(image, mask)
    if not np.any(sitk.GetArrayViewFromImage(mask) > 0):
        return detect(image, mask, config)
    start = perf_counter()
    scan = normalize(image, mask, config)
    prepared = perf_counter()
    evidence = enhance(scan, config)
    diagnostics: dict[str, int] = {}
    if method == "contact-growth":
        allowed = (
            (scan.smooth >= scan.blood["lower_hu"]) & (scan.smooth <= scan.blood["upper_hu"])
            & (scan.outside <= 16.0) & ~evidence.excluded
        ) | scan.parent
        grown = ndi.binary_propagation(scan.parent, structure=NEIGHBORS, mask=allowed)
        diagnostics["growth_added_support_voxels"] = int(np.count_nonzero(grown & ~evidence.support))
        evidence = _with_support(scan, evidence, grown, config)
        candidates, warnings = _contact_candidates(scan, evidence, config)
    else:
        if method == "border-cleaning":
            support = clean_wall_layers(scan.parent, evidence.support, config.spacing_mm)
            diagnostics["wall_cleaning_removed_voxels"] = int(np.count_nonzero(evidence.support & ~support))
            evidence = _with_support(scan, evidence, support, config)
        candidates, warnings = propose(scan, evidence, config)
    enhanced = perf_counter()
    branches, rejections = resolve(scan, evidence, candidates, config)
    if method == "pca-direction":
        for branch in branches:
            try:
                direction = proximal_pca_direction(branch.path_xyz_mm, branch.radius_mm)
            except ValueError:
                branch.warnings.append("Proximal PCA direction is ambiguous; using ostium-to-seed direction.")
                diagnostics["pca_fallbacks"] = diagnostics.get("pca_fallbacks", 0) + 1
                continue
            local_direction = (np.asarray(scan.grid.GetDirection()).reshape(3, 3).T @ direction)[::-1]
            ostium = np.asarray(scan.grid.TransformPhysicalPointToContinuousIndex(branch.ostium_xyz_mm))[::-1]
            branch.direction_xyz = point_tuple(direction)
            branch.features["parent_angle_degrees"] = round(
                parent_angle(scan.parent, ostium, local_direction, config.spacing_mm), 2,
            )
    return Detection(
        branches,
        {
            "prepare_s": round(prepared - start, 3),
            "enhance_s": round(enhanced - prepared, 3),
            "trace_s": round(perf_counter() - enhanced, 3),
            "total_s": round(perf_counter() - start, 3),
        },
        scan.blood, len(candidates), rejections,
        [
            *scan.warnings, *warnings,
            *(f"{name}: {count}" for name, count in diagnostics.items()),
            f"Experimental paper adaptation: {method}; no clinical accuracy established.",
            "No named-vessel count, orientation, iliac-presence or anatomical-order rule is applied.",
        ], config,
    )


def benchmark_papers(data_root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Choose a new report path to preserve previous experiments.")
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("source") != "analytic_synthetic_geometry":
        raise ValueError("This development benchmark requires complete analytic synthetic references.")
    cases: list[dict] = []
    identifiers: set[str] = set()
    for entry in manifest["cases"]:
        case_id = entry["case_id"]
        if case_id in identifiers or Path(case_id).name != case_id or case_id in (".", ".."):
            raise ValueError("Case identifiers must be unique directory names.")
        identifiers.add(case_id)
        directory = data_root / case_id
        if not {"orig.nii.gz", "mask.nii.gz", "reference.json"}.issubset(entry["files"]):
            raise ValueError(f"Missing input hashes for {case_id}.")
        for name, digest in entry["files"].items():
            if Path(name).name != name or name in (".", ".."):
                raise ValueError("Manifest files must be local filenames.")
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Input integrity failure: {case_id}/{name}")
        reference = json.loads((directory / "reference.json").read_text())
        if reference["case_id"] != case_id:
            raise ValueError("Reference and manifest case identifiers must match.")
        image, mask = (read_nifti(str(directory / name)) for name in ("orig.nii.gz", "mask.nii.gz"))
        variants: dict[str, dict] = {}
        for method in ("baseline", *METHODS):
            result = detect(image, mask) if method == "baseline" else detect_paper(image, mask, method)
            prediction = result.prediction(case_id)
            variants[method] = {
                "prediction": prediction, "runtime_s": result.timings["total_s"],
                "metrics": {str(t): evaluate_case(prediction, reference, t) for t in (2, 3, 5)},
                "rejections": result.rejections,
            }
        cases.append({"case_id": case_id, "family": entry.get("family"), "variants": variants})
        print(f"{case_id}: " + ", ".join(
            f"{method} {record['metrics']['3']['true_positives']}/"
            f"{record['metrics']['3']['false_positives']}/{record['metrics']['3']['false_negatives']}"
            for method, record in variants.items()
        ), flush=True)
    report = {
        "schema_version": 1,
        "usage": "Inspected synthetic development/regression cases; no clinical or independent test claim.",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("paper_methods.py", "detector.py", "stress.py", "evaluate.py")
        },
        "summary": {
            method: {
                "tolerances_mm": {
                    str(t): summarize_cases([case["variants"][method]["metrics"][str(t)] for case in cases])
                    for t in (2, 3, 5)
                },
                "families_3mm": {
                    family: summarize_cases([
                        case["variants"][method]["metrics"]["3"] for case in cases if case["family"] == family
                    ]) for family in sorted({case["family"] for case in cases if case["family"]})
                },
                "mean_runtime_s": float(np.mean([case["variants"][method]["runtime_s"] for case in cases]))
                if cases else None,
            } for method in ("baseline", *METHODS)
        },
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--image", required=True, type=Path)
    run.add_argument("--aorta-mask", required=True, type=Path)
    run.add_argument("--method", required=True, choices=METHODS)
    run.add_argument("--case-id")
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--diagnostics", type=Path)
    run.add_argument("--minimum-radius-mm", type=float, default=0.7)
    run.add_argument("--spacing-mm", type=float, default=1.0)
    batch = commands.add_parser("benchmark")
    batch.add_argument("--data-root", required=True, type=Path)
    batch.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    try:
        if args.command == "benchmark":
            benchmark_papers(args.data_root, args.output)
        else:
            paths = [args.output, *([args.diagnostics] if args.diagnostics else [])]
            if len({path.resolve() for path in paths}) != len(paths) or any(path.exists() for path in paths):
                raise FileExistsError("Prediction and diagnostics need distinct, new output paths.")
            result = detect_paper(
                read_nifti(str(args.image)), read_nifti(str(args.aorta_mask)), args.method,
                DetectorConfig(minimum_radius_mm=args.minimum_radius_mm, spacing_mm=args.spacing_mm),
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as destination:
                destination.write(json.dumps(
                    result.prediction(args.case_id or args.image.parent.name), indent=2, allow_nan=False,
                ) + "\n")
            if args.diagnostics:
                args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
                with args.diagnostics.open("x", encoding="utf-8") as destination:
                    destination.write(json.dumps(result.diagnostics(), indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
