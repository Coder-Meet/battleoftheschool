#!/usr/bin/env python3
"""
Branchseed challenge — entry point.

Detects arteries that branch directly off a supplied parent aorta in a CT
volume and writes one prediction JSON per case.

CLI contract (required by the challenge spec):
    python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json

The default pipeline scores strict and review detector proposals with the bundled
logistic model at 0.15, then merges retained origins with strict-first priority.
Use --pipeline strict for the classical detector without the bundled filter.
"""

import argparse
import json
from pathlib import Path
import sys

import SimpleITK as sitk

from detector import DetectorConfig
from nifti_io import read_nifti
from pipeline import add_pipeline_arguments, run_pipeline


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
    parser.add_argument("--diagnostics", help="Optional JSON path for timings, paths and evidence.")
    add_pipeline_arguments(parser)
    parser.add_argument(
        "--minimum-radius-mm", type=float, default=0.7,
        help="Minimum tracing/seed lumen radius; not the organizer's minimum origin size.",
    )
    parser.add_argument(
        "--minimum-origin-diameter-mm", type=float, default=2.0,
        help="Minimum proximal lumen diameter with one native voxel of uncertainty (0 disables).",
    )
    parser.add_argument("--spacing-mm", type=float, default=1.0, help="Isotropic working spacing; finer grids cost more CPU.")
    parser.add_argument("--threads", type=int, default=4, help="SimpleITK CPU threads (default: 4).")
    parser.add_argument(
        "--parallel-clearance-mm", type=float, default=0.0,
        help="Experimental: also accept daughters that run alongside the aorta wall, clearing it by at least"
             " this many mm (0 = off; see ROBUSTNESS_PROTOCOL.md).",
    )
    return parser.parse_args()


def load_volume(path: str) -> sitk.Image:
    return read_nifti(path)


def voxel_to_physical(image: sitk.Image, index_xyz) -> tuple:
    """Convert a (i, j, k) voxel index to physical mm using the image's own
    coordinate system, per the spec (SimpleITK.TransformIndexToPhysicalPoint)."""
    return image.TransformIndexToPhysicalPoint([int(v) for v in index_xyz])


def find_daughter_branches(image: sitk.Image, aorta_mask: sitk.Image) -> list:
    result, _ = run_pipeline(image, aorta_mask)
    return [branch.prediction() for branch in result.branches]


def build_output(case_id: str, daughters: list) -> dict:
    return {
        "case_id": case_id,
        "parent": {"instance_id": "aorta"},
        "daughters": daughters,
    }


def main() -> int:
    args = parse_args()
    if args.threads < 1:
        print("--threads must be positive.", file=sys.stderr)
        return 2
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(args.threads)
    try:
        image = load_volume(args.image)
        aorta_mask = load_volume(args.aorta_mask)
        image_path = Path(args.image)
        case_id = args.case_id or (
            image_path.parent.name if image_path.parent.name.startswith("subject")
            else image_path.name.removesuffix(".gz").removesuffix(".nii")
        )
        result, workflow_diagnostics = run_pipeline(image, aorta_mask, DetectorConfig(
            minimum_radius_mm=args.minimum_radius_mm, spacing_mm=args.spacing_mm,
            minimum_origin_diameter_mm=args.minimum_origin_diameter_mm,
            parallel_clearance_mm=args.parallel_clearance_mm,
        ), workflow=args.pipeline, model_path=args.candidate_model, threshold=args.candidate_threshold)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result.prediction(case_id), indent=2, allow_nan=False) + "\n")
        if args.diagnostics:
            diagnostics = Path(args.diagnostics)
            diagnostics.parent.mkdir(parents=True, exist_ok=True)
            payload = result.diagnostics()
            payload["workflow"] = workflow_diagnostics
            diagnostics.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        print(f"Branchseed: {error}", file=sys.stderr)
        return 1
    print(f"Wrote {len(result.branches)} daughter instance(s) for '{case_id}' -> {args.output}"
          f" ({result.timings['total_s']:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
