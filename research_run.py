"""Research-only scoring of strict or review-union proposals; run.py stays the submission CLI."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import SimpleITK as sitk

from candidate_patches import EXTRA_FEATURE_NAMES, extract_candidate_data, preprocessing_metadata
from detector import Detection, DetectorConfig, detect, detect_pool
from learning import FEATURE_NAMES, features
from nifti_io import read_nifti
from patch_inference import BlendModel, PatchModel, file_sha256, score_batch
from tabular_learning import TreeModel, feature_matrix, json_sha256, write_json

SOURCE_FILES = (
    "detector.py", "candidate_patches.py", "learning.py", "tabular_learning.py",
    "patch_inference.py", "nifti_io.py", "research_run.py",
)


def source_hashes() -> dict[str, str]:
    return {name: file_sha256(Path(__file__).with_name(name)) for name in SOURCE_FILES}


def validate_tree_source(model: TreeModel, sources: dict[str, str]) -> None:
    contract = model.metadata.get("inference_contract")
    if not isinstance(contract, dict):
        raise ValueError("Tree lacks a verified extraction contract; train from source-hashed candidate records.")
    if contract.get("source_sha256") != {
        name: sources[name] for name in ("detector.py", "candidate_patches.py", "learning.py")
    } or contract.get("preprocessing") != preprocessing_metadata():
        raise ValueError("Tree physical preprocessing/source differs; use its frozen source or re-extract/retrain.")


def select_prediction(result: Detection, case_id: str, scores: np.ndarray, threshold: float, mode: str) -> dict:
    if mode not in ("scores-only", "filter"):
        raise ValueError("Unknown research mode.")
    if (
        scores.shape != (len(result.branches),) or not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1)) or not np.isfinite(threshold) or not 0 <= threshold <= 1
    ):
        raise ValueError("Invalid ordered candidate scores or threshold.")
    prediction = result.prediction(case_id)
    if mode == "filter":
        prediction["daughters"] = [
            row for row, value in zip(prediction["daughters"], scores) if value >= threshold
        ]
    return prediction


def score_detection(
    image: sitk.Image, mask: sitk.Image, result: Detection, case_id: str, group_id: str,
    input_hashes: dict[str, str], *, tree_path: Path | None = None,
    onnx_path: Path | None = None, blend_path: Path | None = None, threads: int = 4,
) -> tuple[np.ndarray, float, dict]:
    if tree_path is None and onnx_path is None:
        raise ValueError("Supply a tree or ONNX model.")
    if (tree_path is not None and onnx_path is not None) != (blend_path is not None):
        raise ValueError("Combined inference requires both models and their frozen --blend-model.")
    sources = source_hashes()
    start = perf_counter()
    tree = TreeModel.load(tree_path) if tree_path is not None else None
    if tree is not None:
        validate_tree_source(tree, sources)
    cnn = PatchModel.load(onnx_path, threads=threads) if onnx_path is not None else None
    blend = BlendModel.load(blend_path) if blend_path is not None else None
    size, spacing = 32, 1.0
    if cnn is not None:
        prep = cnn.contract["preprocessing"]
        size, spacing = prep["size"], prep["spacing_mm"]
        if cnn.contract["source_sha256"] != {
            name: sources[name] for name in ("detector.py", "candidate_patches.py")
        }:
            raise ValueError("ONNX extraction source differs; use its frozen source or re-extract/retrain.")
    if tree is not None and len(tree.feature_names) > len(FEATURE_NAMES) and (size, spacing) != (32, 1.0):
        raise ValueError("Extended tree features require the saved 32-pixel/1-mm extraction grid.")
    if blend is not None and tree is not None and cnn is not None:
        if (
            not set(tree.split["train"]).issubset(cnn.split["train"])
            or any(sorted(tree.split[p]) != sorted(cnn.split[p]) for p in ("validation", "test"))
            or blend.split != cnn.split
        ):
            raise ValueError("Combined models have incompatible case splits.")
    loaded = perf_counter()
    extracted = (
        extract_candidate_data(image, mask, result, size=size, spacing_mm=spacing)
        if cnn is not None or (tree is not None and len(tree.feature_names) > len(FEATURE_NAMES)) else None
    )
    records: list[dict] = []
    for index, branch in enumerate(result.branches):
        vector = features(branch)
        row: dict = {
            "case_id": case_id, "group_id": group_id, "instance_id": branch.instance_id,
            "features": vector, "fingerprint": json.dumps([
                branch.ostium_xyz_mm, branch.seed_xyz_mm, branch.direction_xyz, branch.radius_mm, vector,
            ]),
            "input_sha256": input_hashes, "detector_sha256": sources["detector.py"],
            "extractor_sha256": sources["candidate_patches.py"],
        }
        if extracted is not None:
            row["extra_features"] = dict(zip(EXTRA_FEATURE_NAMES, extracted.extra_features[index].tolist()))
        records.append(row)
    extracted_at = perf_counter()
    contract = {
        "feature_names": FEATURE_NAMES, "extra_feature_names": EXTRA_FEATURE_NAMES,
        "preprocessing": preprocessing_metadata(size, spacing),
        "source_sha256": {name: sources[name] for name in ("detector.py", "candidate_patches.py")},
        "manifest_sha256": json_sha256({"inputs": input_hashes, "case_id": case_id, "group_id": group_id}),
    }
    model_hashes = {
        name: file_sha256(path) for name, path in (
            ("tree", tree_path), ("onnx", onnx_path), ("blend", blend_path),
        ) if path is not None
    }
    if onnx_path is not None:
        model_hashes["onnx_sidecar"] = file_sha256(onnx_path.with_suffix(".json"))
    components: dict[str, list[float]] = {}
    values = np.empty(0, dtype=np.float64)
    threshold = 0.0
    if tree is not None:
        vectors = [
            row["features"] + (
                [row["extra_features"][name] for name in EXTRA_FEATURE_NAMES]
                if len(tree.feature_names) > len(FEATURE_NAMES) else []
            ) for row in records
        ]
        matrix = np.asarray(vectors, dtype=np.float64).reshape(len(records), len(tree.feature_names))
        values = tree.scores(feature_matrix(matrix, tree.feature_names))
        threshold = tree.threshold
        components["tree"] = values.tolist()
    if cnn is not None and extracted is not None:
        cnn_batch = cnn.score_candidates(extracted.patches, records, contract)
        if blend is not None:
            tree_batch = score_batch(values, records, contract, model_hashes["tree"])
            values = blend.scores(tree_batch, cnn_batch)
            threshold = blend.threshold
        else:
            values, threshold = cnn_batch.scores, cnn.threshold
        components["onnx"] = cnn_batch.scores.tolist()
    return values, threshold, {
        "schema_version": 1, "scope": "research_only", "clinical_accuracy_claim": False,
        "promotion": "not_authorized", "source_sha256": sources, "model_sha256": model_hashes,
        "input_sha256": input_hashes, "contract": contract, "records": records,
        "scores": values.tolist(), "component_scores": components, "threshold": threshold,
        "below_threshold_ids": [b.instance_id for b, value in zip(result.branches, values) if value < threshold],
        "model_load_s": loaded - start, "extraction_s": extracted_at - loaded,
        "scoring_s": perf_counter() - extracted_at,
        "history": {
            "independent_case_claim": False,
            "prior_exposed_case_ids": [f"subject{i:03d}" for i in range(1, 26)],
            "note": "All 25 supplied images were inspected/pseudo-trained; renamed cases may also overlap.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--aorta-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--proposals-output", type=Path, required=True)
    parser.add_argument("--case-id")
    parser.add_argument("--group-id")
    parser.add_argument("--tree-model", type=Path)
    parser.add_argument("--onnx-model", type=Path)
    parser.add_argument("--blend-model", type=Path)
    parser.add_argument("--mode", choices=("scores-only", "filter"), default="scores-only")
    parser.add_argument("--proposals", choices=("strict", "review-union"), default="strict")
    parser.add_argument("--threads", type=int, choices=(1, 2, 3, 4), default=4)
    args = parser.parse_args()
    start = perf_counter()
    try:
        destinations = [args.output.resolve(), args.diagnostics.resolve(), args.proposals_output.resolve()]
        if len(set(destinations)) != 3 or any(path.exists() for path in destinations):
            raise ValueError("Prediction, diagnostics and preserved proposals must be distinct new paths.")
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(args.threads)
        image, mask = read_nifti(str(args.image)), read_nifti(str(args.aorta_mask))
        case_id = args.case_id or (
            args.image.parent.name if args.image.parent.name.startswith("subject")
            else args.image.name.removesuffix(".gz").removesuffix(".nii")
        )
        result = detect(image, mask) if args.proposals == "strict" else detect_pool(image, mask)
        values, threshold, diagnostics = score_detection(
            image, mask, result, case_id, args.group_id or f"unknown:{case_id}",
            {"image": file_sha256(args.image), "aorta_mask": file_sha256(args.aorta_mask)},
            tree_path=args.tree_model, onnx_path=args.onnx_model, blend_path=args.blend_model, threads=args.threads,
        )
        prediction = select_prediction(result, case_id, values, threshold, args.mode)
        diagnostics.update({
            "mode": args.mode, "proposal_selection": args.proposals, "detector": result.diagnostics(),
            "strict_config": asdict(DetectorConfig()), "review_config": asdict(DetectorConfig.review()),
            "unfiltered_prediction": result.prediction(case_id), "output_sha256": json_sha256(prediction),
            "wall_before_output_s": perf_counter() - start,
        })
        for path in destinations:
            path.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.proposals_output, result.prediction(case_id))
        write_json(args.output, prediction)
        write_json(args.diagnostics, diagnostics)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        print(f"Branchseed research: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
