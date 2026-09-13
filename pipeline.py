"""Production workflow: score strict and review proposals before merging survivors."""

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import SimpleITK as sitk

from detector import Branch, Detection, DetectorConfig, detect
from learning import CandidateModel, FEATURE_NAMES, features, filter_detection
from origin_recovery import detect_connected

DEFAULT_MODEL = Path(__file__).resolve().parent / "models/production-v1/logistic.json"
DEFAULT_MODEL_SHA256 = "e9f864956aa65c3f05c038b13b0c8ca9e7289e4a44f6f0eb7837ab85814c730e"
DEFAULT_THRESHOLD = 0.15
DEFAULT_PIPELINE = "score-before-merge"


def add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pipeline", choices=(DEFAULT_PIPELINE, "strict"), default=DEFAULT_PIPELINE,
                        help="Default: score strict/review separately, then merge. strict restores classical proposals.")
    parser.add_argument("--candidate-model", type=Path,
                        help="Override the bundled logistic model; uses the custom artifact's threshold unless overridden.")
    parser.add_argument("--candidate-threshold", type=float,
                        help="Override filter threshold (0–1). Bundled workflow uses 0.15.")


def run_pipeline(
    image: sitk.Image, mask: sitk.Image, config: DetectorConfig | None = None, *,
    workflow: str = DEFAULT_PIPELINE, model_path: Path | None = None, threshold: float | None = None,
    recover_connected_origins: bool = False,
) -> tuple[Detection, dict]:
    start = perf_counter()
    config = config or DetectorConfig()
    if workflow == DEFAULT_PIPELINE:
        # The audited fusion operating point; strict retains the released 1.2 baseline.
        config = replace(config, native_contrast_scale=0.9)
    if workflow not in (DEFAULT_PIPELINE, "strict"):
        raise ValueError("Unknown production pipeline.")
    if threshold is not None and (not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("Candidate threshold must be finite and between zero and one.")
    if workflow == "strict" and model_path is None and threshold is not None:
        raise ValueError("Strict threshold override requires --candidate-model.")
    if recover_connected_origins and workflow != "strict":
        raise ValueError("Connected-origin recovery requires --pipeline strict; fusion recovery is not validated.")
    model = None
    metadata: dict = {"name": workflow, "schema_version": 1}
    if workflow == DEFAULT_PIPELINE or model_path is not None:
        selected_path = model_path if model_path is not None else DEFAULT_MODEL
        sha256 = hashlib.sha256(selected_path.read_bytes()).hexdigest()
        if model_path is None and sha256 != DEFAULT_MODEL_SHA256:
            raise ValueError("Bundled production model hash differs; restore the versioned model.")
        model = CandidateModel.load(selected_path)
        saved_threshold = model.threshold
        model.threshold = threshold if threshold is not None else (
            DEFAULT_THRESHOLD if model_path is None else saved_threshold
        )
        metadata["candidate_model"] = {
            "path": str(selected_path), "sha256": sha256, "saved_threshold": saved_threshold,
            "threshold": model.threshold, "feature_names": FEATURE_NAMES,
        }
    metadata["recover_connected_origins"] = recover_connected_origins
    strict = (detect_connected(image, mask, config, require_root=True)
              if recover_connected_origins else detect(image, mask, config))
    if workflow == "strict":
        if model is not None:
            metadata["filter"] = filter_detection(strict, model)
        strict.timings["total_s"] = round(perf_counter() - start, 3)
        return strict, metadata
    assert model is not None
    # Apply the same user-specified physical/eligibility settings to both passes.
    # Origin eligibility was disabled in the audit's review pool; enforcing 2 mm
    # here is separately checked for exact five-case parity with the audit winner.
    review_config = replace(
        DetectorConfig.review(), spacing_mm=config.spacing_mm, margin_mm=config.margin_mm,
        minimum_radius_mm=config.minimum_radius_mm,
        minimum_origin_diameter_mm=config.minimum_origin_diameter_mm,
        parallel_clearance_mm=config.parallel_clearance_mm,
        support_contrast_fraction=config.support_contrast_fraction,
        native_contrast_scale=config.native_contrast_scale,
    )
    review = detect(image, mask, review_config)
    kept: list[Branch] = []
    decisions: list[dict] = []
    below, merged = 0, 0
    # Deterministic strict-first order reproduces the audited winning variant.
    for profile, result in (("strict", strict), ("review", review)):
        matrix = np.asarray([features(b) for b in result.branches]).reshape(-1, len(FEATURE_NAMES))
        for branch, value in zip(result.branches, model.scores(matrix)):
            row = {"profile": profile, "source_instance_id": branch.instance_id,
                   "score": float(value), "candidate": asdict(branch), "output_instance_id": None}
            if value < model.threshold:
                row["decision"] = "below_threshold"
                below += 1
            else:
                suppressor = next((old for old in kept if np.linalg.norm(
                    np.subtract(branch.ostium_xyz_mm, old.ostium_xyz_mm),
                ) < 3.0), None)
                if suppressor is not None:
                    row["decision"] = "merged"
                    row["suppressor_instance_id"] = suppressor.instance_id
                    merged += 1
                else:
                    selected = copy.deepcopy(branch)
                    selected.instance_id = f"branch_{len(kept) + 1:03d}"
                    kept.append(selected)
                    row["decision"] = "retained"
                    row["output_instance_id"] = selected.instance_id
            decisions.append(row)
    result = Detection(
        kept, {"strict_s": strict.timings["total_s"], "review_s": review.timings["total_s"],
               "total_s": round(perf_counter() - start, 3)}, strict.blood_model,
        strict.candidates + review.candidates,
        {"learned_candidate_filter": below, "merged_after_scoring": merged},
        list(dict.fromkeys(strict.warnings + review.warnings)) + [
            "Score-before-merge logistic workflow selected on five reused development cases; scores are uncalibrated.",
        ], config,
    )
    metadata.update({
        "merge_distance_mm": 3.0, "priority": "strict_first", "decisions": decisions,
        "profiles": {name: {"config": asdict(d.config), "blood_model": d.blood_model,
                             "candidate_roots": d.candidates, "proposals": len(d.branches),
                             "rejections": d.rejections} for name, d in (("strict", strict), ("review", review))},
    })
    return result, metadata
