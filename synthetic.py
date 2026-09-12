"""Generate reproducible synthetic training CTs with analytic vessel ground truth."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
from scipy import ndimage as ndi

Array = npt.NDArray[np.float64]
SCENARIOS = {
    1: "radial_branches",
    2: "oblique_branches",
    3: "small_low_contrast_anisotropic",
    4: "common_trunk",
    5: "nearby_openings_and_distractors",
}


@dataclass
class SyntheticCase:
    image: sitk.Image
    parent: sitk.Image
    instances: sitk.Image
    reference: dict
    provenance: dict


def tube_distance(coordinates: Array, points: Array) -> Array:
    distance = np.full(coordinates.shape[:-1], np.inf)
    for start, end in zip(points[:-1], points[1:]):
        delta = end - start
        fraction = np.clip(np.sum((coordinates - start) * delta, axis=-1) / np.dot(delta, delta), 0, 1)
        distance = np.minimum(distance, np.linalg.norm(coordinates - start - fraction[..., None] * delta, axis=-1))
    return distance


def generate_case(scenario: int, seed: int = 2026) -> SyntheticCase:
    if scenario not in SCENARIOS and scenario != 0:
        raise ValueError("Scenario must be 0 (negative control) or 1 through 5.")
    rng = np.random.default_rng(seed)
    spacing = np.array((0.7, 0.8, 1.3) if scenario == 3 else (0.8, 0.8, 1.0))
    size = np.ceil(np.array((72, 72, 84)) / spacing).astype(int) + 1
    coordinates = np.moveaxis(np.indices(tuple(size[::-1]), dtype=float)[::-1], 0, -1) * spacing
    x, y, z = np.moveaxis(coordinates, -1, 0)
    parent = ((x - 32) ** 2 + (y - 32) ** 2 <= 8**2) & (z >= 5) & (z <= 78)
    instances = parent.astype(np.uint8)
    blood = parent.copy()
    specifications: dict[int, list[tuple[float, float, float, float]]] = {
        0: [],
        1: [(25, 0, 0, 2.5), (55, np.pi, 0, 2.5)],
        2: [(25, 0, 0.9, 2.0), (58, np.pi / 2, -0.7, 2.2)],
        3: [(26, 0, 0.2, 1.35), (54, np.pi, -0.2, 1.65)],
        4: [(30, 0, 0, 2.5), (58, np.pi, 0, 2.1)],
        5: [(32, 0, 0, 1.5), (38, 0, 0, 1.6)],
    }
    branch_specs = specifications[scenario]
    angle = float(rng.uniform(-0.4, 0.4))
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    origin = np.array((-110.3, 21.7, 307.5))
    daughters = []
    geometry = []
    nearest = np.full(parent.shape, np.inf)
    for index, (height, azimuth, slope, nominal_radius) in enumerate(branch_specs, 1):
        height += float(rng.uniform(-0.3, 0.3))
        radius = nominal_radius + float(rng.uniform(-0.08, 0.08))
        radial = np.array([np.cos(azimuth), np.sin(azimuth), 0])
        direction = radial + np.array([0, 0, slope])
        direction /= np.linalg.norm(direction)
        ostium = np.array([32, 32, height]) + 8 * radial
        endpoint = ostium + 21 * direction
        common = scenario == 4 and index == 1
        junction = ostium + 9 * direction if common else endpoint
        path = np.array([ostium - 7 * direction, junction])
        distance = tube_distance(coordinates, path)
        downstream = []
        if common:
            for sign in (-1, 1):
                child = np.array([junction, junction + np.array([13, sign * 9, 0])])
                child_distance = tube_distance(coordinates, child)
                distance = np.minimum(distance / radius, child_distance / 1.9) * radius
                downstream.append((child @ rotation.T + origin).tolist())
        lumen = distance <= radius
        owned = lumen & ~parent & (distance / radius < nearest)
        instances[owned] = index + 1
        nearest[owned] = distance[owned] / radius
        blood |= lumen
        seed_point = ostium + 5 * direction
        identifier = f"branch_{index:03d}"
        daughters.append({
            "instance_id": identifier,
            "parent_instance_id": "aorta",
            "ostium_xyz_mm": (rotation @ ostium + origin).tolist(),
            "seed_xyz_mm": (rotation @ seed_point + origin).tolist(),
            "radius_mm": radius,
            "direction_xyz": (rotation @ direction).tolist(),
        })
        geometry.append({
            "instance_id": identifier, "voxel_label": index + 1,
            "proximal_centerline_xyz_mm": (np.array([ostium, junction]) @ rotation.T + origin).tolist(),
            "downstream_centerlines_xyz_mm": downstream,
        })
    distractors = np.zeros_like(parent)
    if scenario in (0, 5):
        distractors |= ((x - 20) ** 2 + (y - 32) ** 2 <= 1.8**2) & (z >= 14) & (z <= 68)
        distractors |= (x - 47) ** 2 + (y - 46) ** 2 + (z - 50) ** 2 <= 3.5**2
    if scenario == 0:
        blood |= (x - 32) ** 2 + (y - 32) ** 2 <= 8**2
    contrast, noise = (120, 14) if scenario == 3 else (320, 10)
    intensity = np.where(blood | distractors, contrast, 25).astype(float)
    intensity[(x - 24) ** 2 + (y - 32) ** 2 + (z - 45) ** 2 <= 1.4**2] = 900
    intensity = ndi.gaussian_filter(intensity, sigma=0.35 / spacing[::-1])
    intensity += rng.normal(0, noise, parent.shape)
    image = sitk.GetImageFromArray(intensity.astype(np.float32))
    image.SetSpacing(spacing.tolist())
    image.SetOrigin(origin.tolist())
    image.SetDirection(rotation.ravel().tolist())
    mask = sitk.GetImageFromArray(parent.astype(np.uint8))
    labels = sitk.GetImageFromArray(instances)
    mask.CopyInformation(image)
    labels.CopyInformation(image)
    name = SCENARIOS.get(scenario, "negative_control")
    case_id = f"synthetic_{scenario:03d}_{seed}"
    return SyntheticCase(image, mask, labels, {
        "case_id": case_id, "parent": {"instance_id": "aorta"}, "daughters": daughters,
    }, {
        "schema_version": 1, "generator_version": 1, "case_id": case_id, "seed": seed,
        "scenario": name, "source": "analytic_synthetic_geometry", "intended_use": "training_and_development",
        "coordinate_system": "LPS_mm", "parent_radius_mm": 8,
        "labels": {"0": "background_and_distractors", "1": "aorta",
                   **{str(g["voxel_label"]): g["instance_id"] for g in geometry}},
        "geometry": geometry, "contrast_hu": contrast, "noise_standard_deviation_hu": noise,
        "limitations": [
            "Procedural tubes are not real human anatomy; no clinical accuracy inference.",
            "Continuous analytic references precede voxelization, partial volume and CT noise.",
            "Distal children of a common trunk share its direct-daughter instance label.",
            "Do not evaluate on these training cases as an independent test set.",
        ],
    })


def write_case(case: SyntheticCase, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    for name, volume in (("orig.nii.gz", case.image), ("mask.nii.gz", case.parent),
                         ("instances.nii.gz", case.instances)):
        sitk.WriteImage(volume, str(directory / name))
    for name, value in (("reference.json", case.reference), ("provenance.json", case.provenance)):
        (directory / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return {
        "case_id": case.reference["case_id"], "scenario": case.provenance["scenario"], "partition": "train",
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("Seed must be nonnegative.")
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new directory to protect labels.")
    args.output_dir.mkdir(parents=True)
    cases = []
    for scenario in SCENARIOS:
        case = generate_case(scenario, args.seed)
        cases.append(write_case(case, args.output_dir / case.reference["case_id"]))
    manifest = {
        "schema_version": 1, "source": "analytic_synthetic_geometry", "seed": args.seed, "cases": cases,
        "usage": "All five cases are training/development data, not an independent clinical test set.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote five synthetic ground-truth cases to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
