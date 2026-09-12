"""Score existing predictions against organizer reference files of uncertain schema.

    python score_references.py --references refs/ --predictions outputs/batch --data-root data

References may be one JSON per case, one JSON holding a list of cases, or a
directory of either. Field names are normalized with a small alias table;
every normalization applied is recorded in the report so nothing is silent.
Voxel-index coordinates are converted with the case image geometry when a
data root is given. An LPS/RAS mirror diagnostic is reported when flipping
x and y would match many more ostia, which points to a convention mismatch
rather than a detector error.
"""

import argparse
import json
from pathlib import Path
import re

import numpy as np

from evaluate import evaluate_case, summarize_cases
from nifti_io import read_nifti

CASE_KEYS = ("case_id", "case", "subject", "subject_id", "id", "name")
DAUGHTER_KEYS = ("daughters", "branches", "children", "instances", "daughter_instances")
FIELD_ALIASES = {
    "instance_id": ("instance_id", "id", "name", "label", "branch_id"),
    "ostium_xyz_mm": ("ostium_xyz_mm", "ostium_mm", "ostium", "ostium_xyz", "origin_mm", "origin", "ostium_point"),
    "seed_xyz_mm": ("seed_xyz_mm", "seed_mm", "seed", "seed_xyz", "seed_point"),
    "direction_xyz": ("direction_xyz", "direction", "direction_mm", "dir", "tangent"),
    "radius_mm": ("radius_mm", "radius", "r_mm", "diameter_mm"),
}
VOXEL_HINTS = ("voxel", "ijk", "index", "vox")
SEED_OFFSET_MM = 5.0


def normalize_case_id(value) -> str:
    text = str(value)
    match = re.search(r"(\d+)", text)
    if match and re.fullmatch(r"(?i)(subject|subj|case|orig|mask|s|c)?[_-]?0*\d+(\.nii(\.gz)?)?", text):
        return f"subject{int(match.group(1)):03d}"
    return text


def _pick(mapping: dict, aliases: tuple[str, ...], notes: list[str], label: str):
    for alias in aliases:
        if alias in mapping:
            if alias != aliases[0]:
                notes.append(f"{label}: read '{alias}' as '{aliases[0]}'")
            return mapping[alias]
    return None


def _to_vector(value, label: str) -> np.ndarray:
    if isinstance(value, dict):
        value = [value.get(axis) for axis in ("x", "y", "z")]
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain three finite numbers, got {value!r}.")
    return vector


def normalize_reference(raw: dict, notes: list[str], image=None) -> dict:
    case_value = next((raw[key] for key in CASE_KEYS if key in raw), None)
    if case_value is None:
        raise ValueError("Reference has no case identifier.")
    case_id = normalize_case_id(case_value)
    if case_id != case_value:
        notes.append(f"{case_id}: case identifier '{case_value}' normalized")
    daughters_raw = next((raw[key] for key in DAUGHTER_KEYS if key in raw), None)
    if daughters_raw is None:
        raise ValueError(f"{case_id}: reference has no daughter list.")
    if isinstance(daughters_raw, dict):
        daughters_raw = [dict(value, instance_id=key) for key, value in daughters_raw.items()]
    voxel_space = any(hint in str(raw.get("coordinate_space", raw.get("units", ""))).lower() for hint in VOXEL_HINTS)
    daughters = []
    for index, entry in enumerate(daughters_raw):
        label = f"{case_id}#{index}"
        identifier = _pick(entry, FIELD_ALIASES["instance_id"], notes, label)
        identifier = str(identifier) if identifier is not None else f"reference_{index + 1:03d}"
        ostium_raw = _pick(entry, FIELD_ALIASES["ostium_xyz_mm"], notes, label)
        if ostium_raw is None:
            voxel_keys = [key for key in entry if any(hint in key.lower() for hint in VOXEL_HINTS) and "ost" in key.lower()]
            if not voxel_keys:
                raise ValueError(f"{label}: no ostium coordinate found in {sorted(entry)}.")
            ostium_raw, entry_voxel = entry[voxel_keys[0]], True
        else:
            entry_voxel = voxel_space
        ostium = _to_vector(ostium_raw, f"{label} ostium")
        seed_raw = _pick(entry, FIELD_ALIASES["seed_xyz_mm"], notes, label)
        seed = _to_vector(seed_raw, f"{label} seed") if seed_raw is not None else None
        if entry_voxel:
            if image is None:
                raise ValueError(f"{label}: voxel coordinates need --data-root for conversion.")
            ostium = np.asarray(image.TransformContinuousIndexToPhysicalPoint(ostium.tolist()))
            if seed is not None:
                seed = np.asarray(image.TransformContinuousIndexToPhysicalPoint(seed.tolist()))
            notes.append(f"{label}: converted voxel indices to physical mm with the case image geometry")
        direction_raw = _pick(entry, FIELD_ALIASES["direction_xyz"], notes, label)
        direction = _to_vector(direction_raw, f"{label} direction") if direction_raw is not None else None
        if direction is None:
            if seed is None:
                raise ValueError(f"{label}: needs a direction or a seed.")
            direction = seed - ostium
            notes.append(f"{label}: direction derived from seed minus ostium")
        norm = float(np.linalg.norm(direction))
        if norm <= 0:
            raise ValueError(f"{label}: zero-length direction.")
        if not np.isclose(norm, 1, atol=0.01):
            notes.append(f"{label}: direction renormalized from length {norm:.3f}")
        direction = direction / norm
        if seed is None:
            seed = ostium + SEED_OFFSET_MM * direction
            notes.append(f"{label}: seed placed {SEED_OFFSET_MM:g} mm along the direction")
        radius_raw = _pick(entry, FIELD_ALIASES["radius_mm"], notes, label)
        radius_known = radius_raw is not None and np.isfinite(float(radius_raw)) and float(radius_raw) > 0
        if radius_known:
            radius = float(radius_raw)
            if "diameter_mm" in entry and "radius_mm" not in entry:
                radius /= 2
                notes.append(f"{label}: radius taken as half the diameter")
        else:
            radius = 1.0
            notes.append(f"{label}: no radius; radius error is not meaningful for this daughter")
        daughters.append({
            "instance_id": identifier, "parent_instance_id": "aorta",
            "ostium_xyz_mm": ostium.tolist(), "seed_xyz_mm": seed.tolist(),
            "radius_mm": radius, "direction_xyz": direction.tolist(), "_radius_known": radius_known,
        })
    identifiers = [d["instance_id"] for d in daughters]
    if len(set(identifiers)) != len(identifiers):
        for index, daughter in enumerate(daughters):
            daughter["instance_id"] = f"{daughter['instance_id']}_{index + 1:03d}"
        notes.append(f"{case_id}: duplicate reference identifiers were suffixed")
    return {"case_id": case_id, "parent": {"instance_id": "aorta"}, "daughters": daughters}


def load_references(path: Path) -> list[dict]:
    files = sorted(path.rglob("*.json")) if path.is_dir() else [path]
    raws = []
    for file in files:
        loaded = json.loads(file.read_text())
        if isinstance(loaded, list):
            raws.extend(loaded)
        elif isinstance(loaded, dict) and any(key in loaded for key in DAUGHTER_KEYS):
            if not any(key in loaded for key in CASE_KEYS):
                loaded = dict(loaded, case_id=file.stem)
            raws.append(loaded)
        elif isinstance(loaded, dict) and "cases" in loaded:
            raws.extend(loaded["cases"])
        elif isinstance(loaded, dict):
            raws.extend(dict(value, case_id=key) for key, value in loaded.items() if isinstance(value, dict))
    return raws


def mirrored(prediction: dict) -> dict:
    flipped = json.loads(json.dumps(prediction))
    for daughter in flipped["daughters"]:
        for key in ("ostium_xyz_mm", "seed_xyz_mm", "direction_xyz"):
            daughter[key][0] = -daughter[key][0]
            daughter[key][1] = -daughter[key][1]
    return flipped


def strip_private(reference: dict) -> dict:
    return {
        **reference,
        "daughters": [{k: v for k, v in d.items() if not k.startswith("_")} for d in reference["daughters"]],
    }


def score(references_path: Path, predictions_dir: Path, data_root: Path | None, tolerances: list[float]) -> dict:
    notes: list[str] = []
    per_case = []
    missing = []
    for raw in load_references(references_path):
        case_id = normalize_case_id(next(raw[key] for key in CASE_KEYS if key in raw))
        image = None
        if data_root is not None:
            number = int(case_id[-3:]) if case_id[-3:].isdigit() else None
            candidates = sorted((data_root / case_id).glob(f"orig{number}.nii*")) if number is not None else []
            image = read_nifti(str(candidates[0])) if candidates else None
        reference = normalize_reference(raw, notes, image)
        prediction_path = predictions_dir / f"{reference['case_id']}.json"
        if not prediction_path.exists():
            missing.append(reference["case_id"])
            continue
        prediction = json.loads(prediction_path.read_text())
        scored = strip_private(reference)
        radius_known = {d["instance_id"]: d["_radius_known"] for d in reference["daughters"]}
        results = {}
        for tolerance in tolerances:
            result = evaluate_case(prediction, scored, tolerance)
            for match in result["matches"]:
                if not radius_known[match["reference_id"]]:
                    match["radius_error_mm"] = None
            results[f"{tolerance:g}"] = result
        mirror = evaluate_case(mirrored(prediction), scored, max(tolerances))
        primary = results[f"{max(tolerances):g}"]
        per_case.append({
            "case_id": reference["case_id"], "reference_daughters": len(reference["daughters"]),
            "predicted_daughters": len(prediction["daughters"]),
            "by_tolerance": results,
            "mirror_xy_true_positives": mirror["true_positives"],
            "mirror_suspected": mirror["true_positives"] > primary["true_positives"] + 1,
        })
    summaries = {}
    for tolerance in tolerances:
        rows = []
        for case in per_case:
            row = json.loads(json.dumps(case["by_tolerance"][f"{tolerance:g}"]))
            row["matches"] = [
                dict(m, radius_error_mm=m["radius_error_mm"] if m["radius_error_mm"] is not None else np.nan)
                for m in row["matches"]
            ]
            rows.append(row)
        summary = summarize_cases(rows)
        radius = [m["radius_error_mm"] for r in rows for m in r["matches"] if np.isfinite(m["radius_error_mm"])]
        summary["matched_errors_only"]["radius_error_mm"] = {
            "mean": float(np.mean(radius)) if radius else None,
            "median": float(np.median(radius)) if radius else None,
            "p95": float(np.percentile(radius, 95)) if radius else None,
        }
        summaries[f"{tolerance:g}"] = summary
    return {
        "cases": per_case, "summary_by_tolerance_mm": summaries,
        "missing_predictions": missing, "normalization_notes": notes,
        "mirror_warning": any(case["mirror_suspected"] for case in per_case),
    }


def render(report: dict) -> str:
    lines = [f"{'case':<12}{'ref':>5}{'pred':>6}" + "".join(f"{'TP/FP/FN@' + t:>16}" for t in report["summary_by_tolerance_mm"])]
    for case in report["cases"]:
        row = f"{case['case_id']:<12}{case['reference_daughters']:>5}{case['predicted_daughters']:>6}"
        for tolerance, result in case["by_tolerance"].items():
            row += f"{result['true_positives']}/{result['false_positives']}/{result['false_negatives']}".rjust(16)
        lines.append(row)
    for tolerance, summary in report["summary_by_tolerance_mm"].items():
        errors = summary["matched_errors_only"]

        def fmt(value):
            return "n/a" if value is None else f"{value:.2f}"

        lines.append(
            f"@{tolerance} mm: TP={summary['true_positives']} FP={summary['false_positives']} "
            f"FN={summary['false_negatives']} precision={fmt(summary['precision'])} "
            f"recall={fmt(summary['recall'])} F1={fmt(summary['f1'])} | matched mean ostium "
            f"{fmt(errors['ostium_error_mm']['mean'])} mm, seed {fmt(errors['seed_error_mm']['mean'])} mm, "
            f"radius {fmt(errors['radius_error_mm']['mean'])} mm, "
            f"direction {fmt(errors['direction_error_degrees']['mean'])} deg"
        )
    if report["missing_predictions"]:
        lines.append("No prediction file for: " + ", ".join(report["missing_predictions"]))
    if report["mirror_warning"]:
        lines.append(
            "WARNING: mirroring x and y (LPS<->RAS) matches many more ostia; check the coordinate "
            "convention before treating these as detector errors."
        )
    if report["normalization_notes"]:
        lines.append(f"{len(report['normalization_notes'])} normalization note(s); see report JSON.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--tolerances-mm", type=float, nargs="+", default=[2, 3, 5])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(not np.isfinite(t) or t <= 0 for t in args.tolerances_mm):
        parser.error("Tolerances must be positive and finite.")
    try:
        report = score(args.references, args.predictions, args.data_root, sorted(set(args.tolerances_mm)))
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        parser.error(str(error))
    if not report["cases"] and not report["missing_predictions"]:
        parser.error("No reference cases were found.")
    if args.output:
        if args.output.exists():
            parser.error("Report already exists; choose a new output to preserve prior scores.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
