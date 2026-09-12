#!/usr/bin/env python3
"""
AI labelling of detector candidates, no web page involved.

    python autolabel.py render --cases subject001          # or --all: pool + one evidence PNG per candidate
    python autolabel.py apply --verdicts verdicts.json     # write AI verdicts into labels/reviews.json
    python autolabel.py status                             # per-case progress against the rendered pools

`render` runs the loose review-profile detector merged with the strict result, saves every candidate
with its 13 features under outputs/autolabel/<case>/pool.json, and renders the CT evidence each
candidate rests on (whole-aorta locators, ±2 mm slab views, consecutive axial slices, traced path).
An AI reviewer reads those images and answers one question per candidate: does a bright tube leave
the aorta outline at the origin, along the traced path, for at least 5 mm? Its answers go into a
verdicts file {"subject001": {"branch_001": "confirmed", "branch_002": "rejected"}} and `apply`
records them in the training schema that learning.py consumes. Unreviewed candidates are skipped.
"""

import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
import numpy as np
import SimpleITK as sitk

from detector import Branch, DetectorConfig, detect_pool, prepare_roi
from explorer import ROOT
from learning import FEATURE_NAMES, features as candidate_features
from nifti_io import read_nifti

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

HALF_MM = 15.0
STRIP_OFFSETS_MM = (-4, -2, 0, 2, 4)
SLAB_MM = 2.0
WINDOW = (-100.0, 600.0)
MIP_WINDOW = (100.0, 600.0)
LABELS = ("confirmed", "rejected", "unreviewed")
OPPOSITE = {"A": "P", "P": "A", "L": "R", "R": "L", "S": "I", "I": "S"}


# --- candidate pool ---------------------------------------------------------------------------


def candidate_record(branch: Branch) -> dict:
    vector = candidate_features(branch)
    return {
        **branch.prediction(), "path_xyz_mm": branch.path_xyz_mm, "evidence_score": branch.evidence_score,
        "mean_vesselness": branch.mean_vesselness, "warnings": branch.warnings,
        "features": dict(zip(FEATURE_NAMES, vector)), "feature_vector": vector,
    }


def fingerprint(candidate: dict) -> str:
    return json.dumps([
        candidate["ostium_xyz_mm"], candidate["seed_xyz_mm"], candidate["direction_xyz"],
        candidate["radius_mm"], candidate["feature_vector"],
    ])


def case_paths(directory: Path) -> tuple[Path, Path]:
    images, masks = sorted(directory.glob("orig*.nii*")), sorted(directory.glob("mask*.nii*"))
    if len(images) != 1 or len(masks) != 1:
        raise ValueError(f"{directory.name}: expected one orig*.nii[.gz] and one mask*.nii[.gz].")
    return images[0], masks[0]


def is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as source:
        return source.read(40).startswith(b"version https://git-lfs")


# --- evidence rendering -----------------------------------------------------------------------


def _orientation(grid: sitk.Image) -> dict[str, str]:
    letters = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(list(grid.GetDirection()))
    return dict(zip(("x_right", "y_up", "z_up"), letters))


def _panel(ax: Any, image: np.ndarray, mask: np.ndarray, point: tuple[float, float] | None,
           path: np.ndarray | None, title: str, edges: tuple[str, str] | None = None,
           window: tuple[float, float] = WINDOW) -> None:
    ax.imshow(np.clip((image - window[0]) / (window[1] - window[0]), 0, 1), cmap="gray",
              origin="lower", vmin=0, vmax=1, interpolation="nearest")
    if mask.any():
        ax.contour(mask, levels=[0.5], colors="#48e0c0", linewidths=1.0, origin="lower")
    if path is not None and len(path) > 1:
        ax.plot(path[:, 0], path[:, 1], "-", color="#ff5c5c", lw=1.8, solid_capstyle="round")
        ax.plot(path[-1, 0], path[-1, 1], "s", color="#ff5c5c", ms=4)
    if point is not None:
        ax.plot(point[0], point[1], "o", color="#ffd43b", ms=6, mec="black")
    if edges:
        up, right = edges
        style = dict(transform=ax.transAxes, color="#48e0c0", fontsize=9, fontweight="bold")
        ax.text(0.5, 0.98, up, ha="center", va="top", **style)
        ax.text(0.5, 0.02, OPPOSITE[up], ha="center", va="bottom", **style)
        ax.text(0.98, 0.5, right, ha="right", va="center", **style)
        ax.text(0.02, 0.5, OPPOSITE[right], ha="left", va="center", **style)
    ax.set_title(title, color="#c9d1d9", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")


def render_candidate(grid: sitk.Image, ct: np.ndarray, mask: np.ndarray, candidate: dict) -> bytes:
    """One PNG: two whole-aorta locators, three ±2 mm slab views through the origin, five consecutive axial slices."""
    spacing = float(grid.GetSpacing()[0])

    def to_zyx(point: Any) -> np.ndarray:
        return np.asarray(grid.TransformPhysicalPointToContinuousIndex([float(v) for v in point]))[::-1]

    ost = to_zyx(candidate["ostium_xyz_mm"])
    path = np.asarray([to_zyx(p) for p in candidate["path_xyz_mm"]], dtype=float)
    if len(path) < 2:
        path = np.vstack((ost, to_zyx(np.add(candidate["ostium_xyz_mm"], 8.0 * np.asarray(candidate["direction_xyz"])))))
    half = max(4, int(round(HALF_MM / spacing)))
    centre = np.clip(np.round(ost).astype(int), 0, np.asarray(ct.shape) - 1)
    lo = np.maximum(centre - half, 0)
    hi = np.minimum(centre + half + 1, ct.shape)
    iz, iy, ix = (int(v) for v in centre)
    axes_of = _orientation(grid)
    slab = max(1, int(round(SLAB_MM / spacing)))
    z0, z1 = max(0, iz - slab), min(ct.shape[0], iz + slab + 1)
    y0, y1 = max(0, iy - slab), min(ct.shape[1], iy + slab + 1)
    x0, x1 = max(0, ix - slab), min(ct.shape[2], ix + slab + 1)
    local = path - lo
    fig = Figure(figsize=(16, 5.6), dpi=88)
    fig.patch.set_facecolor("#0d1117")
    gs = fig.add_gridspec(2, 20, hspace=0.28, wspace=0.18)
    _panel(fig.add_subplot(gs[0, 0:4]), ct.max(axis=1), mask.max(axis=1), (ost[2], ost[0]),
           path[:, [2, 0]], "locator · coronal MIP", (axes_of["z_up"], axes_of["x_right"]), MIP_WINDOW)
    _panel(fig.add_subplot(gs[1, 0:4]), ct.max(axis=2), mask.max(axis=2), (ost[1], ost[0]),
           path[:, [1, 0]], "locator · sagittal MIP", (axes_of["z_up"], axes_of["y_up"]), MIP_WINDOW)
    title = f"±{SLAB_MM:g} mm slab at the origin"
    views = [
        (f"axial · {title}", ct[z0:z1, lo[1]:hi[1], lo[2]:hi[2]].max(axis=0), mask[iz, lo[1]:hi[1], lo[2]:hi[2]],
         (ost[2] - lo[2], ost[1] - lo[1]), local[:, [2, 1]], (axes_of["y_up"], axes_of["x_right"])),
        (f"coronal · {title}", ct[lo[0]:hi[0], y0:y1, lo[2]:hi[2]].max(axis=1), mask[lo[0]:hi[0], iy, lo[2]:hi[2]],
         (ost[2] - lo[2], ost[0] - lo[0]), local[:, [2, 0]], (axes_of["z_up"], axes_of["x_right"])),
        (f"sagittal · {title}", ct[lo[0]:hi[0], lo[1]:hi[1], x0:x1].max(axis=2), mask[lo[0]:hi[0], lo[1]:hi[1], ix],
         (ost[1] - lo[1], ost[0] - lo[0]), local[:, [1, 0]], (axes_of["z_up"], axes_of["y_up"])),
    ]
    for column, (name, image, outline, point, trace, edges) in enumerate(views):
        start = 5 + column * 5
        _panel(fig.add_subplot(gs[0, start:start + 5]), image, outline, point, trace, name, edges)
    for column, offset_mm in enumerate(STRIP_OFFSETS_MM):
        z = int(np.clip(iz + round(offset_mm / spacing), 0, ct.shape[0] - 1))
        marker: tuple[float, float] | None = (ost[2] - lo[2], ost[1] - lo[1]) if offset_mm == 0 else None
        start = 5 + column * 3
        _panel(fig.add_subplot(gs[1, start:start + 3]), ct[z, lo[1]:hi[1], lo[2]:hi[2]],
               mask[z, lo[1]:hi[1], lo[2]:hi[2]], marker, None, f"axial {offset_mm:+d} mm",
               (axes_of["y_up"], axes_of["x_right"]))
    header = (f"{candidate['instance_id']} · radius {candidate['radius_mm']:.2f} mm · evidence {candidate['evidence_score']:.2f}"
              f" · HU vs blood {candidate['features']['path_hu_relative']:.2f} · bone {candidate['features']['bone_distance_mm']:.1f} mm"
              f" · angle {candidate['features']['parent_angle_degrees']:.0f}° · wall gap {candidate['features']['connector_gap']:.2f}")
    fig.suptitle(header, color="#c9d1d9", fontsize=11, x=0.02, ha="left")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor=fig.get_facecolor())
    return buffer.getvalue()


def render_case(directory: Path, output_dir: Path, config: DetectorConfig) -> dict:
    image_path, mask_path = case_paths(directory)
    image, mask = read_nifti(str(image_path)), read_nifti(str(mask_path))
    result = detect_pool(image, mask, config)
    grid, parent = prepare_roi(image, mask, config)
    ct = sitk.GetArrayFromImage(grid).astype(np.float32)
    case_dir = output_dir / directory.name
    case_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for branch in result.branches:
        record = candidate_record(branch)
        (case_dir / f"{branch.instance_id}.png").write_bytes(render_candidate(grid, ct, parent, record))
        record["image"] = str(case_dir / f"{branch.instance_id}.png")
        candidates.append(record)
    pool = {
        "case_id": directory.name, "profile": config.profile, "feature_names": FEATURE_NAMES,
        "candidates": candidates, "rejections": result.rejections, "blood_model": result.blood_model,
        "timings": result.timings, "rendered_at": datetime.now(timezone.utc).isoformat(),
    }
    (case_dir / "pool.json").write_text(json.dumps(pool, indent=2, allow_nan=False) + "\n")
    return {"case_id": directory.name, "candidates": len(candidates), "seconds": result.timings["total_s"]}


# --- verdicts ---------------------------------------------------------------------------------


def load_reviews_file(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {"schema_version": 1, "feature_names": FEATURE_NAMES, "scope": "candidate_reviews_only", "records": []}


def apply_verdicts(verdicts: dict[str, dict[str, str]], pool_dir: Path, reviews_path: Path, labeller: str) -> dict:
    payload = load_reviews_file(reviews_path)
    if payload.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Existing reviews use a different feature contract; start a new file.")
    records = [r for r in payload["records"] if isinstance(r, dict)]
    written = {"confirmed": 0, "rejected": 0, "cleared": 0}
    for case_id, labels in verdicts.items():
        pool_path = pool_dir / case_id / "pool.json"
        if not pool_path.is_file():
            raise ValueError(f"No rendered pool for {case_id}; run `autolabel.py render --cases {case_id}` first.")
        candidates = {c["instance_id"]: c for c in json.loads(pool_path.read_text())["candidates"]}
        for instance_id, label in labels.items():
            if label not in LABELS:
                raise ValueError(f"{case_id}/{instance_id}: label must be one of {LABELS}.")
            if instance_id not in candidates:
                raise ValueError(f"{case_id}/{instance_id} is not in the rendered pool.")
            records = [r for r in records if not (r["case_id"] == case_id and r["instance_id"] == instance_id)]
            if label == "unreviewed":
                written["cleared"] += 1
                continue
            candidate = candidates[instance_id]
            records.append({
                "case_id": case_id, "instance_id": instance_id, "label": label,
                "features": [float(v) for v in candidate["feature_vector"]],
                "fingerprint": fingerprint(candidate), "labeller": labeller,
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
            })
            written[label] += 1
    payload["records"] = sorted(records, key=lambda r: (r["case_id"], r["instance_id"]))
    reviews_path.parent.mkdir(parents=True, exist_ok=True)
    reviews_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return written


def print_status(pool_dir: Path, reviews_path: Path) -> None:
    payload = load_reviews_file(reviews_path)
    verdicts: dict[str, dict[str, int]] = {}
    for row in payload["records"]:
        counts = verdicts.setdefault(row["case_id"], {"confirmed": 0, "rejected": 0})
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    pools = {p.parent.name: json.loads(p.read_text()) for p in sorted(pool_dir.glob("*/pool.json"))}
    print(f"{'case':<12}{'pool':>6}{'confirmed':>11}{'rejected':>10}{'pending':>9}")
    pending_total = 0
    for case_id in sorted(set(pools) | set(verdicts)):
        pool = len(pools.get(case_id, {}).get("candidates", []))
        counts = verdicts.get(case_id, {"confirmed": 0, "rejected": 0})
        pending = max(0, pool - counts["confirmed"] - counts["rejected"]) if pool else 0
        pending_total += pending
        print(f"{case_id:<12}{pool or '-':>6}{counts['confirmed']:>11}{counts['rejected']:>10}{pending:>9}")
    print(f"\n{len(payload['records'])} verdicts across {len(verdicts)} cases; {pending_total} rendered candidates still pending.")


# --- CLI --------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--pool-dir", type=Path, default=ROOT / "outputs" / "autolabel")
    parser.add_argument("--reviews", type=Path, default=ROOT / "labels" / "reviews.json")
    commands = parser.add_subparsers(dest="command", required=True)
    render = commands.add_parser("render", help="Detect candidates and render their CT evidence.")
    render.add_argument("--cases", nargs="*", default=[])
    render.add_argument("--all", action="store_true")
    apply = commands.add_parser("apply", help="Record verdicts {case: {instance: label}} in the training file.")
    apply.add_argument("--verdicts", required=True, type=Path)
    apply.add_argument("--labeller", default="claude")
    commands.add_parser("status", help="Per-case progress of the reviews file against rendered pools.")
    args = parser.parse_args()
    try:
        if args.command == "status":
            print_status(args.pool_dir, args.reviews)
            return 0
        if args.command == "apply":
            written = apply_verdicts(json.loads(args.verdicts.read_text()), args.pool_dir, args.reviews, args.labeller)
            print(f"Recorded {written['confirmed']} confirmed, {written['rejected']} rejected, "
                  f"{written['cleared']} cleared -> {args.reviews}")
            return 0
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
        directories = sorted(p for p in args.data_root.iterdir() if p.is_dir())
        available = [d for d in directories if not any(is_lfs_pointer(p) for p in case_paths(d))]
        selected = available if args.all else [d for d in available if d.name in set(args.cases)]
        missing = sorted(set(args.cases) - {d.name for d in selected})
        if missing or not selected:
            print(f"Unknown or unresolved cases: {missing or 'none selected'}. Pass --cases or --all.", file=sys.stderr)
            return 2
        for directory in selected:
            print(json.dumps(render_case(directory, args.pool_dir, DetectorConfig.review())), flush=True)
        print(f"Evidence rendered under {args.pool_dir}. Judge each PNG, write verdicts, then `autolabel.py apply`.")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"autolabel: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
