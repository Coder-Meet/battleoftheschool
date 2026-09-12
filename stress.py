"""Generate adversarial synthetic cases that imitate realistic failure modes.

These are procedural stress cases, not anatomy. They exist to expose detector
failures that the smooth training phantoms cannot: tortuous parents, thick
slices, touching veins, calcification, mural thrombus, imperfect masks and
dense branch fields. Reference labels come from analytic geometry only.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
from scipy import ndimage as ndi

Array = npt.NDArray[np.float64]
FAMILIES = (
    "tortuous_low_contrast",
    "thick_slice_anisotropic",
    "touching_vein_and_calcification",
    "aneurysm_with_mural_thrombus",
    "common_trunk_early_split",
    "daughter_of_daughter",
    "nearby_pair_with_short_stub",
    "cropped_short_segment",
    "imperfect_parent_mask",
    "dense_branch_field",
    "negative_controls_only",
    "high_noise_small_branches",
    "curved_daughters",
    "wall_parallel_descending",
)
STUB_LENGTH_MM = 3.0
ELIGIBLE_LENGTH_MM = 16.0


@dataclass
class StressCase:
    image: sitk.Image
    parent: sitk.Image
    reference: dict
    provenance: dict


def tube(coordinates: Array, centers: Array, radii: Array, chunk: int = 8) -> Array:
    """Signed distance to a union of spheres: negative inside the lumen."""
    signed = np.full(coordinates.shape[:-1], np.inf)
    for start in range(0, len(centers), chunk):
        offsets = coordinates[..., None, :] - centers[start : start + chunk]
        distance = np.linalg.norm(offsets, axis=-1) - radii[start : start + chunk]
        signed = np.minimum(signed, distance.min(axis=-1))
    return signed


def axis_from(azimuth: float, slope: float) -> Array:
    direction = np.array([np.cos(azimuth), np.sin(azimuth), slope])
    return direction / np.linalg.norm(direction)


def wall_crossing(inside: Array, direction: Array, parent: "ParentTube") -> Array | None:
    """March outward along the daughter axis to the parent lumen boundary."""
    steps = np.arange(0, 40, 0.01)
    points = inside + steps[:, None] * direction
    values = parent.signed(points)
    outside = np.flatnonzero(values > 0)
    if not len(outside) or outside[0] == 0:
        return None
    index = int(outside[0])
    span = values[index] - values[index - 1]
    fraction = -values[index - 1] / span if span else 0.0
    return points[index - 1] + fraction * (points[index] - points[index - 1])


@dataclass
class ParentTube:
    centers: Array
    radii: Array

    def signed(self, points: Array) -> Array:
        offsets = points[:, None, :] - self.centers
        return (np.linalg.norm(offsets, axis=-1) - self.radii).min(axis=-1)


def parallel_axis(
    height: float, azimuth: float, parent: ParentTube, radius: float, clearance: float, length: float,
) -> tuple[Array, Array, Array] | None:
    """Daughter that leaves the wall and then descends alongside the parent, like an inferior mesenteric artery."""
    radial = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
    descent = np.arange(-6.0, length + 8.0, 0.3)
    points = []
    for t in descent:
        nearest = int(np.argmin(np.abs(parent.centers[:, 2] - (height - t))))
        settle = 1 - np.exp(-max(t, 0.0) / 3.0)
        offset = parent.radii[nearest] - 0.5 + 0.4 * min(t, 0.0) + (radius + clearance + 0.5) * settle
        points.append(parent.centers[nearest] + offset * radial)
    axis = np.asarray(points)
    signed = parent.signed(axis)
    outside = np.flatnonzero(signed > 0)
    if not len(outside) or outside[0] == 0:
        return None
    index = int(outside[0])
    span = signed[index] - signed[index - 1]
    fraction = -signed[index - 1] / span if span else 0.0
    ostium = axis[index - 1] + fraction * (axis[index] - axis[index - 1])
    axis = np.vstack((axis[:index], ostium, axis[index:]))
    steps = np.linalg.norm(np.diff(axis, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(steps)]
    along = arc - arc[index]
    keep = along <= length
    return axis[keep], along[keep], ostium


def generate_case(family: str, seed: int) -> StressCase:
    if family not in FAMILIES:
        raise ValueError(f"Unknown stress family: {family}")
    if seed < 0:
        raise ValueError("Seed must be nonnegative.")
    rng = np.random.default_rng(np.random.SeedSequence([seed, FAMILIES.index(family)]))
    thick = family == "thick_slice_anisotropic"
    spacing = np.array((0.75, 0.75, 2.5) if thick else (0.7, 0.7, 1.0 + 0.5 * rng.random()))
    extent = np.array((76.0, 76.0, 92.0))
    size = np.ceil(extent / spacing).astype(int) + 1
    coordinates = np.moveaxis(np.indices(tuple(size[::-1]), dtype=float)[::-1], 0, -1) * spacing
    x, y, z = np.moveaxis(coordinates, -1, 0)

    # Curved, tapering parent aorta with an optional aneurysm bulge.
    cropped = family == "cropped_short_segment"
    bottom, top = (29.0, 57.0) if cropped else (4.0, 89.0)
    heights = np.arange(bottom - 10, top + 10, 0.6)
    sway = 4.0 + 5.0 * rng.random() if family == "tortuous_low_contrast" else 1.5 * rng.random()
    centers = np.column_stack([
        38 + sway * np.sin(heights / 17 + rng.random()),
        38 + 0.7 * sway * np.cos(heights / 23 + rng.random()),
        heights,
    ])
    lumen_radii = 8.5 - 0.022 * (heights - heights[0]) + 0.3 * rng.standard_normal(len(heights)).cumsum() * 0.05
    aneurysm = family == "aneurysm_with_mural_thrombus"
    if aneurysm:
        lumen_radii = lumen_radii + 4.5 * np.exp(-((heights - 46) ** 2) / (2 * 9.0**2))
    parent = ParentTube(centers, lumen_radii)
    parent_signed = tube(coordinates, centers, lumen_radii)
    lumen = (parent_signed <= 0) & (z >= bottom) & (z <= top)
    outer = tube(coordinates, centers, lumen_radii + (3.2 if aneurysm else 1.4)) <= 0

    counts = {
        "negative_controls_only": 0,
        "cropped_short_segment": 1,
        "dense_branch_field": 6,
        "nearby_pair_with_short_stub": 2,
        "wall_parallel_descending": 2,
    }
    count = counts.get(family, int(rng.integers(2, 4)))
    small = family in ("high_noise_small_branches", "thick_slice_anisotropic")
    specs = []
    branch_heights = np.linspace(bottom + 10, top - 10, count + 2)[1:-1]
    for height in branch_heights:
        height += float(rng.uniform(-2, 2))
        radius = float(rng.uniform(0.9, 1.5) if small else rng.uniform(1.4, 3.4))
        specs.append((height, float(rng.uniform(0, 2 * np.pi)), float(rng.uniform(-1.1, 1.1)), radius))
    if family == "nearby_pair_with_short_stub":
        base = float(rng.uniform(26, 60))
        azimuth = float(rng.uniform(0, 2 * np.pi))
        specs = [
            (base, azimuth, 0.15, 2.2),
            (base + 5.5, azimuth + 0.55, -0.1, 1.9),
        ]

    branch_lumen = np.zeros_like(lumen)
    daughters: list[dict] = []
    ineligible: list[dict] = []
    geometry: list[dict] = []
    for index, (height, azimuth, slope, radius) in enumerate(specs, 1):
        direction = axis_from(azimuth, slope)
        inside = centers[np.argmin(np.abs(heights - height))]
        trunk = family == "common_trunk_early_split" and index == 1
        length = ELIGIBLE_LENGTH_MM if not trunk else 8.0
        hugging = family == "wall_parallel_descending" and index == 1
        if hugging:
            radius = float(rng.uniform(1.0, 1.6))
            parallel = parallel_axis(height, azimuth, parent, radius, float(rng.uniform(1.5, 3.0)), 22.0)
            if parallel is None:
                continue
            axis_points, along, ostium = parallel
        else:
            crossing = wall_crossing(inside, direction, parent)
            if crossing is None:
                continue
            ostium = crossing
            along = np.unique(np.r_[np.arange(-8.0, length, 0.35), 0.0, 5.0, length])
            axis_points = ostium[None] + along[:, None] * direction
        if family == "curved_daughters":
            side = np.cross(direction, [0.0, 0.0, 1.0])
            side /= np.linalg.norm(side)
            axis_points += (0.025 * np.maximum(along, 0)**2)[:, None] * side
        radii = np.full(len(axis_points), radius)
        if family == "high_noise_small_branches":
            radii *= 1 - 0.25 * np.exp(-((np.arange(len(axis_points)) - len(axis_points) * 0.7) ** 2) / 12)
        branch_lumen |= tube(coordinates, axis_points, radii) <= 0
        children = []
        if trunk:
            for sign in (-1, 1):
                pivot = ostium + length * direction
                child_axis = direction + sign * 0.85 * np.cross(direction, [0, 0, 1.0])
                child_axis /= np.linalg.norm(child_axis)
                child = pivot[None] + np.arange(0, 16.0, 0.35)[:, None] * child_axis
                branch_lumen |= tube(coordinates, child, np.full(len(child), radius * 0.78)) <= 0
                children.append((pivot, child_axis))
        if family == "daughter_of_daughter" and index == 1:
            pivot = ostium + 11.0 * direction
            side = np.cross(direction, [0.0, 0.0, 1.0])
            side /= np.linalg.norm(side)
            grandchild = pivot[None] + np.arange(0, 15.0, 0.35)[:, None] * side
            branch_lumen |= tube(coordinates, grandchild, np.full(len(grandchild), radius * 0.7)) <= 0
            children.append((pivot, side))
        proximal = axis_points[along >= 0]
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(proximal, axis=0), axis=1))]
        seed_point = np.array([np.interp(5.0, arc, proximal[:, i]) for i in range(3)])
        seed_axis = seed_point - ostium
        seed_axis /= np.linalg.norm(seed_axis)
        record = {
            "instance_id": f"branch_{index:03d}",
            "parent_instance_id": "aorta",
            "ostium_xyz_mm": ostium.tolist(),
            "seed_xyz_mm": seed_point.tolist(),
            "radius_mm": float(np.interp(5.0, arc, radii[along >= 0])),
            "direction_xyz": seed_axis.tolist(),
        }
        daughters.append(record)
        geometry.append({
            "instance_id": record["instance_id"],
            "proximal_centerline_xyz_mm": proximal.tolist(),
            "downstream_centerlines_xyz_mm": [[p.tolist(), (p + 12 * d).tolist()] for p, d in children],
            "first_bifurcation_mm_from_ostium": 8.0 if trunk else 11.0 if children else None,
        })
    if family == "nearby_pair_with_short_stub" and daughters:
        direction = axis_from(float(rng.uniform(0, 2 * np.pi)), 0.4)
        inside = centers[np.argmin(np.abs(heights - 70))]
        stub_ostium = wall_crossing(inside, direction, parent)
        if stub_ostium is not None:
            axis_points = stub_ostium[None] + np.arange(-6.0, STUB_LENGTH_MM, 0.3)[:, None] * direction
            branch_lumen |= tube(coordinates, axis_points, np.full(len(axis_points), 1.6)) <= 0
            ineligible.append({
                "instance_id": "stub_001",
                "parent_instance_id": "aorta",
                "ostium_xyz_mm": stub_ostium.tolist(),
                "seed_xyz_mm": (stub_ostium + 5 * direction).tolist(),
                "radius_mm": 1.6,
                "direction_xyz": direction.tolist(),
                "reason": "3 mm centerline plus 1.6 mm end cap is shorter than the 5 mm eligibility length",
            })

    # Negative structures: touching vein, separate vein, calcification, bone.
    vein = np.zeros_like(lumen)
    if family in ("touching_vein_and_calcification", "negative_controls_only", "dense_branch_field"):
        offset = centers.mean(axis=0)[:2] + np.array([lumen_radii.mean() + 5.4, 0.0])
        vein_centers = np.column_stack([
            np.full(len(heights), offset[0]), np.full(len(heights), offset[1]), heights
        ])
        vein |= tube(coordinates, vein_centers, np.full(len(heights), 6.0)) <= 0
    if family in ("negative_controls_only", "tortuous_low_contrast"):
        vein |= (x - 16) ** 2 + (y - 46) ** 2 <= 2.2**2
    calcification = np.zeros_like(lumen)
    if family in ("touching_vein_and_calcification", "aneurysm_with_mural_thrombus", "dense_branch_field"):
        for _ in range(4):
            spot = centers[int(rng.integers(0, len(centers)))]
            angle = rng.uniform(0, 2 * np.pi)
            radius_at = lumen_radii[int(np.argmin(np.abs(heights - spot[2])))]
            place = spot + (radius_at + 0.6) * np.array([np.cos(angle), np.sin(angle), 0])
            calcification |= (x - place[0]) ** 2 + (y - place[1]) ** 2 + (z - place[2]) ** 2 <= 2.0**2

    contrast = float(rng.uniform(150, 210) if family == "tortuous_low_contrast" else rng.uniform(240, 420))
    noise = float(rng.uniform(26, 38) if family == "high_noise_small_branches" else rng.uniform(9, 20))
    intensity = np.full(lumen.shape, 22.0)
    intensity[outer & ~lumen & ~branch_lumen] = 58 if aneurysm else 40
    intensity[vein] = contrast * 0.45
    intensity[(parent_signed <= 0) | branch_lumen] = contrast
    intensity[calcification] = 780
    intensity[(x - 60) ** 2 + (y - 16) ** 2 <= 7.0**2] = 320  # vertebral body
    intensity = ndi.gaussian_filter(intensity, sigma=0.45 / spacing[::-1])
    intensity += rng.normal(0, noise, intensity.shape)

    mask_voxels = lumen.copy()
    perturbation = "none"
    if family == "imperfect_parent_mask":
        perturbation = ("dilated", "eroded", "shifted")[int(rng.integers(0, 3))]
        if perturbation == "dilated":
            mask_voxels = ndi.binary_dilation(mask_voxels)
        elif perturbation == "eroded":
            mask_voxels = ndi.binary_erosion(mask_voxels)
        else:
            mask_voxels = np.roll(mask_voxels, 1, axis=2)

    angle = float(rng.uniform(-np.pi, np.pi))
    tilt = float(rng.uniform(-0.12, 0.12))
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle) * np.cos(tilt), np.cos(angle) * np.cos(tilt), -np.sin(tilt)],
        [np.sin(angle) * np.sin(tilt), np.cos(angle) * np.sin(tilt), np.cos(tilt)],
    ])
    rotation, _ = np.linalg.qr(rotation)
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    origin = np.array((-158.4, 43.9, 512.7)) + rng.uniform(-40, 40, 3)

    image = sitk.GetImageFromArray(intensity.astype(np.float32))
    image.SetSpacing(spacing.tolist())
    image.SetOrigin(origin.tolist())
    image.SetDirection(rotation.ravel(order="C").tolist())
    mask = sitk.GetImageFromArray(mask_voxels.astype(np.uint8))
    mask.CopyInformation(image)

    def to_world(point: list[float]) -> list[float]:
        return (rotation @ np.asarray(point) + origin).tolist()

    for record in (*daughters, *ineligible):
        record["ostium_xyz_mm"] = to_world(record["ostium_xyz_mm"])
        record["seed_xyz_mm"] = to_world(record["seed_xyz_mm"])
        record["direction_xyz"] = (rotation @ np.asarray(record["direction_xyz"])).tolist()
    for entry in geometry:
        entry["proximal_centerline_xyz_mm"] = [to_world(p) for p in entry["proximal_centerline_xyz_mm"]]
        entry["downstream_centerlines_xyz_mm"] = [
            [to_world(p) for p in line] for line in entry["downstream_centerlines_xyz_mm"]
        ]
    case_id = f"stress_{FAMILIES.index(family):02d}_{family}_{seed}"
    return StressCase(image, mask, {
        "case_id": case_id,
        "parent": {"instance_id": "aorta"},
        "daughters": daughters,
    }, {
        "schema_version": 1,
        "case_id": case_id,
        "family": family,
        "seed": seed,
        "source": "analytic_synthetic_geometry",
        "intended_use": "robustness_stress_testing",
        "coordinate_system": "LPS_mm",
        "contrast_hu": contrast,
        "noise_standard_deviation_hu": noise,
        "spacing_mm": spacing.tolist(),
        "parent_mask_perturbation": perturbation,
        "parent_lumen_radius_range_mm": [float(lumen_radii.min()), float(lumen_radii.max())],
        "geometry": geometry,
        "ineligible_structures": ineligible,
        "negative_structures": {
            "touching_or_separate_vein": bool(vein.any()),
            "wall_calcification": bool(calcification.any()),
            "vertebral_body": True,
            "cropped_parent_ends": True,
            "mural_thrombus": bool(aneurysm),
        },
        "limitations": [
            "Procedural geometry, not human anatomy; no clinical accuracy inference.",
            "Predictions at ineligible stubs or negative structures count as false positives.",
            "Mask perturbations retain the anatomical reference; boundary disagreement is intentional.",
            "Analytic references precede voxelization, partial volume and noise.",
        ],
    })


def write_case(case: StressCase, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    sitk.WriteImage(case.image, str(directory / "orig.nii.gz"))
    sitk.WriteImage(case.parent, str(directory / "mask.nii.gz"))
    for name, value in (("reference.json", case.reference), ("provenance.json", case.provenance)):
        (directory / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return {
        "case_id": case.reference["case_id"],
        "family": case.provenance["family"],
        "partition": "stress",
        "eligible_daughters": len(case.reference["daughters"]),
        "ineligible_stubs": len(case.provenance["ineligible_structures"]),
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=4001)
    parser.add_argument("--families", nargs="*", default=list(FAMILIES))
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("Seed must be nonnegative.")
    unknown = sorted(set(args.families) - set(FAMILIES))
    if unknown:
        parser.error(f"Unknown families: {unknown}")
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new directory to protect labels.")
    args.output_dir.mkdir(parents=True)
    cases = []
    for family in args.families:
        case = generate_case(family, args.seed)
        cases.append(write_case(case, args.output_dir / case.reference["case_id"]))
        print(f"{cases[-1]['case_id']}: {cases[-1]['eligible_daughters']} eligible", flush=True)
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "source": "analytic_synthetic_geometry",
        "seed": args.seed,
        "usage": "Robustness stress testing. Not anatomy and not an independent clinical test set.",
        "cases": cases,
    }, indent=2) + "\n")
    print(f"Wrote {len(cases)} stress cases to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
