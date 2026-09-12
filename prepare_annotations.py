"""Prepare offline CT review packets from frozen predictions, without inventing ground truth."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import numpy as np
import numpy.typing as npt
import SimpleITK as sitk

from evaluate import validate_prediction
from nifti_io import read_nifti

SCOPE = "unreviewed_proposals_not_ground_truth"
GEOMETRY_KEYS = ("ostium_xyz_mm", "seed_xyz_mm", "radius_mm", "direction_xyz")
PLANES = ((2, 0, 1), (1, 0, 2), (0, 1, 2))


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def point_index(image: sitk.Image, point: list[float]) -> list[float]:
    index = np.asarray(image.TransformPhysicalPointToContinuousIndex(point))
    if not np.isfinite(index).all() or np.any(index < -0.5) or np.any(index >= np.array(image.GetSize()) - 0.5):
        raise ValueError(f"Proposal lies outside the CT: {point}")
    return index.tolist()


def proposal_packet(case_id: str, image: sitk.Image, sources: dict[str, dict]) -> dict:
    proposals: dict[str, dict] = {}
    for source, prediction in sources.items():
        validate_prediction(prediction)
        if prediction["case_id"] != case_id:
            raise ValueError("Prediction case ID does not match requested case.")
        for branch in prediction["daughters"]:
            geometry = {key: branch[key] for key in GEOMETRY_KEYS}
            key = json.dumps(geometry, sort_keys=True)
            if key not in proposals:
                proposals[key] = {
                    "candidate_id": f"candidate_{len(proposals) + 1:03d}",
                    "proposal": geometry,
                    "ostium_index_xyz": point_index(image, branch["ostium_xyz_mm"]),
                    "seed_index_xyz": point_index(image, branch["seed_xyz_mm"]),
                    "sources": [],
                    "review": {
                        "status": "unreviewed", "reviewer": None, "notes": "",
                        "corrected_ostium_xyz_mm": None, "corrected_seed_xyz_mm": None,
                        "measured_radius_mm": None, "measured_direction_xyz": None,
                    },
                }
            proposals[key]["sources"].append({"variant": source, "instance_id": branch["instance_id"]})
    candidates = list(proposals.values())
    for candidate in candidates:
        candidate["nearby_proposals_within_3mm"] = [
            other["candidate_id"] for other in candidates
            if other is not candidate and np.linalg.norm(
                np.array(candidate["proposal"]["ostium_xyz_mm"]) - other["proposal"]["ostium_xyz_mm"]
            ) < 3
        ]
    return {
        "schema_version": 1, "scope": SCOPE, "case_id": case_id,
        "coordinate_system": "SimpleITK physical LPS millimetres; indices are zero-based xyz",
        "geometry": {
            "size_xyz": image.GetSize(), "spacing_xyz_mm": image.GetSpacing(),
            "origin_xyz_mm": image.GetOrigin(), "direction_row_major": image.GetDirection(),
        },
        "case_review": {
            "reviewer": None, "entire_parent_reviewed": False,
            "eligible_daughter_count": None, "eligibility_policy": "", "notes": "",
        },
        "missed_candidates_to_add": [],
        "candidates": candidates,
    }


def crop_bounds(mask: npt.NDArray, spacing: npt.NDArray, margin_mm: float = 20) -> npt.NDArray:
    occupied = np.argwhere(mask)
    if not len(occupied):
        raise ValueError("Aorta mask is empty.")
    lower = occupied.min(axis=0)[::-1]
    upper = occupied.max(axis=0)[::-1] + 1
    pad = np.ceil(margin_mm / spacing).astype(int)
    return np.column_stack((np.maximum(0, lower - pad), np.minimum(mask.shape[::-1], upper + pad)))


def draw_plane(
    ax: Axes, ct: npt.NDArray, mask: npt.NDArray, spacing: npt.NDArray,
    bounds: npt.NDArray, plane: tuple[int, int, int], index: int,
    markers: list[tuple[list[float], str, str]] | None = None,
) -> None:
    normal, horizontal, vertical = plane
    if not 0 <= index < ct.shape[2 - normal]:
        ax.text(0.5, 0.5, "Outside scan", ha="center", va="center")
        ax.set_axis_off()
        return
    selection: list[slice | int] = [slice(int(lo), int(hi)) for lo, hi in bounds[::-1]]
    selection[2 - normal] = index
    extent = (
        (bounds[horizontal, 0] - 0.5) * spacing[horizontal],
        (bounds[horizontal, 1] - 0.5) * spacing[horizontal],
        (bounds[vertical, 0] - 0.5) * spacing[vertical],
        (bounds[vertical, 1] - 0.5) * spacing[vertical],
    )
    ax.imshow(ct[tuple(selection)], cmap="gray", vmin=-100, vmax=600,
              origin="lower", extent=extent, interpolation="nearest", aspect="equal")
    if markers is not None:
        outline = mask[tuple(selection)]
        if outline.any() and not outline.all():
            ax.contour(outline, levels=[0.5], colors="cyan", linewidths=0.6,
                       origin="lower", extent=extent)
        for point, color, symbol in markers:
            if abs(point[normal] - index) <= 0.5:
                ax.plot(point[horizontal] * spacing[horizontal], point[vertical] * spacing[vertical],
                        marker=symbol, color=color, markersize=6, markerfacecolor="none")
    ax.set_title(f"{'ijk'[normal]}={index}", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"{'ijk'[horizontal]} →    {'ijk'[vertical]} ↑  (acquisition axes)", fontsize=6)


def render_packet(image: sitk.Image, parent: sitk.Image, packet: dict, directory: Path) -> dict:
    if image.GetSize() != parent.GetSize() or any(
        not np.allclose(a, b, atol=1e-6, rtol=0)
        for a, b in ((image.GetSpacing(), parent.GetSpacing()), (image.GetOrigin(), parent.GetOrigin()),
                     (image.GetDirection(), parent.GetDirection()))
    ):
        raise ValueError("CT and aorta mask geometry differ.")
    ct, mask = sitk.GetArrayFromImage(image), sitk.GetArrayFromImage(parent) > 0
    spacing = np.asarray(image.GetSpacing())
    bounds = crop_bounds(mask, spacing)
    case_id = packet["case_id"]
    survey_slices = list(range(int(bounds[2, 0]), int(bounds[2, 1])))
    survey_pages = 0
    with PdfPages(directory / "01_blinded_survey.pdf") as pdf:
        for start in range(0, len(survey_slices), 16):
            figure = Figure(figsize=(12, 12), layout="constrained")
            axes = figure.subplots(4, 4).ravel()
            for ax, index in zip(axes, survey_slices[start:start + 16]):
                draw_plane(ax, ct, mask, spacing, bounds, PLANES[0], index)
            for ax in axes[len(survey_slices[start:start + 16]):]:
                ax.set_axis_off()
            figure.suptitle(
                f"{case_id} | BLINDED native CT survey | -100..600 HU\n"
                "No predicted points or mask overlay. Mark possible missed openings before viewing proposals.",
                fontsize=11,
            )
            pdf.savefig(figure, dpi=140)
            survey_pages += 1
    with PdfPages(directory / "02_proposed_openings.pdf") as pdf:
        if not packet["candidates"]:
            figure = Figure(figsize=(12, 4))
            figure.text(0.08, 0.6, f"{case_id}: no model proposals. This is NOT a negative reference.", fontsize=14)
            figure.text(0.08, 0.4, "Review the entire supplied parent for missed eligible branches.", fontsize=12)
            pdf.savefig(figure)
        for candidate in packet["candidates"]:
            point = np.asarray(candidate["ostium_index_xyz"])
            radius = np.ceil(20 / spacing).astype(int)
            center = np.rint(point).astype(int)
            local = np.column_stack((np.maximum(0, center - radius),
                                     np.minimum(image.GetSize(), center + radius + 1)))
            figure = Figure(figsize=(15, 11))
            axes = figure.subplots(3, 5)
            figure.subplots_adjust(left=0.025, right=0.99, bottom=0.045, top=0.89, hspace=0.28, wspace=0.08)
            markers = [(candidate["ostium_index_xyz"], "yellow", "+"),
                       (candidate["seed_index_xyz"], "magenta", "o")]
            for row, plane in enumerate(PLANES):
                for column, offset in enumerate(range(-2, 3)):
                    draw_plane(axes[row, column], ct, mask, spacing, local, plane,
                               int(center[plane[0]] + offset), markers)
            ostium = ", ".join(f"{x:.2f}" for x in candidate["proposal"]["ostium_xyz_mm"])
            figure.suptitle(
                f"{case_id} / {candidate['candidate_id']} | UNREVIEWED model proposal | ostium LPS mm ({ostium})\n"
                "Native consecutive slices; cyan=parent, yellow=ostium, magenta=seed (only in-plane). "
                "Check continuity in original CT.",
                fontsize=11,
            )
            pdf.savefig(figure, dpi=140)
    return {"survey_pages": survey_pages, "survey_k_indices": survey_slices,
            "proposal_pages": max(1, len(packet["candidates"]))}


def write_worksheet(packet: dict, path: Path) -> None:
    measurement_columns = [
        *(f"corrected_{point}_{axis}_mm" for point in ("ostium", "seed") for axis in "xyz"),
        "measured_radius_mm", *(f"measured_direction_{axis}" for axis in "xyz"),
    ]
    fields = ["case_id", "candidate_id", "status", "sources", "proposed_ostium_lps_mm",
              "proposed_seed_lps_mm", "proposed_radius_mm", "proposed_direction",
              *measurement_columns, "reviewer", "notes"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for candidate in packet["candidates"]:
            geometry = candidate["proposal"]
            writer.writerow({
                "case_id": packet["case_id"], "candidate_id": candidate["candidate_id"], "status": "unreviewed",
                "sources": "; ".join(f"{s['variant']}/{s['instance_id']}" for s in candidate["sources"]),
                "proposed_ostium_lps_mm": json.dumps(geometry["ostium_xyz_mm"]),
                "proposed_seed_lps_mm": json.dumps(geometry["seed_xyz_mm"]),
                "proposed_radius_mm": geometry["radius_mm"],
                "proposed_direction": json.dumps(geometry["direction_xyz"]),
            })


def prepare(data_root: Path, prediction_dirs: list[Path], cases: list[str], output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Output already exists; choose a new directory to protect reviews.")
    if not cases or len(cases) != len(set(cases)) or any(Path(c).name != c or c in (".", "..") for c in cases):
        raise ValueError("Cases must be unique directory names.")
    if not prediction_dirs or len({p.name for p in prediction_dirs}) != len(prediction_dirs):
        raise ValueError("Prediction directories must have distinct names.")
    output.mkdir(parents=True)
    shutil.copyfile(Path(__file__).with_name("ANNOTATION_GUIDE.md"), output / "START_HERE.md")
    entries = []
    for case_id in cases:
        directory = data_root / case_id
        image_paths, mask_paths = sorted(directory.glob("orig*.nii*")), sorted(directory.glob("mask*.nii*"))
        if len(image_paths) != 1 or len(mask_paths) != 1:
            raise ValueError(f"{case_id}: expected exactly one original CT and aorta mask.")
        image, parent = read_nifti(str(image_paths[0])), read_nifti(str(mask_paths[0]))
        sources = {p.name: json.loads((p / f"{case_id}.json").read_text()) for p in prediction_dirs}
        packet = proposal_packet(case_id, image, sources)
        packet["source_sha256"] = {
            "image": digest(image_paths[0]), "aorta_mask": digest(mask_paths[0]),
            **{p.name: digest(p / f"{case_id}.json") for p in prediction_dirs},
        }
        target = output / case_id
        target.mkdir()
        pages = render_packet(image, parent, packet, target)
        (target / "proposals.json").write_text(json.dumps(packet, indent=2, allow_nan=False) + "\n")
        write_worksheet(packet, target / "review_sheet.csv")
        entry = {"case_id": case_id, "proposals": len(packet["candidates"]), **pages,
                 "files": {p.name: digest(p) for p in sorted(target.iterdir())}}
        entries.append(entry)
        print(f"{case_id}: {entry['proposals']} unreviewed proposals, {entry['survey_pages']} survey pages",
              flush=True)
    manifest = {
        "schema_version": 1, "scope": SCOPE, "cases": entries,
        "builder_sha256": digest(Path(__file__)),
        "guide_sha256": digest(output / "START_HERE.md"),
        "warning": "Model agreement is not independent evidence. These cases are not ground truth.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        prepare(args.data_root, args.predictions, args.cases, args.output_dir)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
