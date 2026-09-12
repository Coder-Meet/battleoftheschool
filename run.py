#!/usr/bin/env python3
"""
Branchseed challenge — entry point.

Detects arteries that branch directly off a supplied parent aorta in a CT
volume and writes one prediction JSON per case.

CLI contract (required by the challenge spec):
    python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json

This is a scaffold: it loads the volume/mask, sets up the physical
coordinate handling correctly, and writes a valid (currently empty)
prediction file. Fill in `find_daughter_branches` with the real detection
logic (vessel enhancement + region growing / graph search / etc).
"""

import argparse
import json
import sys

import numpy as np
import SimpleITK as sitk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect direct daughter arteries branching off a parent aorta."
    )
    parser.add_argument("--image", required=True, help="Path to CT volume (.nii/.nii.gz)")
    parser.add_argument("--aorta-mask", required=True, help="Path to binary parent aorta mask (.nii/.nii.gz)")
    parser.add_argument("--output", required=True, help="Path to write prediction JSON")
    parser.add_argument(
        "--case-id",
        default=None,
        help="Case identifier to embed in the output JSON. Defaults to the image filename stem.",
    )
    return parser.parse_args()


def load_volume(path: str) -> sitk.Image:
    return sitk.ReadImage(path)


def voxel_to_physical(image: sitk.Image, index_xyz) -> tuple:
    """Convert a (i, j, k) voxel index to physical mm using the image's own
    coordinate system, per the spec (SimpleITK.TransformIndexToPhysicalPoint)."""
    return image.TransformIndexToPhysicalPoint([int(v) for v in index_xyz])


def find_daughter_branches(image: sitk.Image, aorta_mask: sitk.Image) -> list:
    """
    Core detection logic — TODO.

    Should return a list of dicts, each with:
        instance_id, parent_instance_id, ostium_xyz_mm, seed_xyz_mm,
        radius_mm, direction_xyz

    Approach ideas (see challenge doc for full definitions):
      1. Find the aortic wall surface from `aorta_mask`.
      2. Look for bright, tubular structures (vesselness filter, e.g.
         Frangi/Sato) adjacent to the wall in `image`.
      3. Cluster candidate ostia, dedupe common trunks vs separate origins.
      4. For each ostium, trace up to 10mm along the daughter centreline
         (or to the first bifurcation) to estimate seed point, radius,
         and initial direction.
    """
    # Placeholder: no daughters detected yet.
    return []


def build_output(case_id: str, daughters: list) -> dict:
    return {
        "case_id": case_id,
        "parent": {"instance_id": "aorta"},
        "daughters": daughters,
    }


def main() -> int:
    args = parse_args()

    image = load_volume(args.image)
    aorta_mask = load_volume(args.aorta_mask)

    case_id = args.case_id
    if case_id is None:
        import os

        case_id = os.path.splitext(os.path.basename(args.image))[0]

    daughters = find_daughter_branches(image, aorta_mask)
    output = build_output(case_id, daughters)

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {len(daughters)} daughter instance(s) for '{case_id}' -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
