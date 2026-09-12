"""Evaluate a labeled case bundle offline, preserving per-case errors and predictions."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
import SimpleITK as sitk

from detector import DetectorConfig, detect, detect_pool
from evaluate import evaluate_case, summarize_cases
from learning import CandidateModel, filter_detection
from nifti_io import read_nifti


def benchmark(
    data_root: Path, output: Path, tolerance_mm: float, config: DetectorConfig | None = None,
    candidate_model: Path | None = None, candidate_pool: bool = False,
) -> dict:
    if output.exists():
        raise FileExistsError("Report already exists; choose a new output to preserve prior evaluations.")
    config = config or (DetectorConfig.review() if candidate_pool else DetectorConfig())
    model = CandidateModel.load(candidate_model) if candidate_model else None
    manifest = json.loads((data_root / "manifest.json").read_text())
    results = []
    unfiltered_results = []
    predictions: dict[str, dict] = {}
    references = {}
    for entry in manifest["cases"]:
        case_id = entry["case_id"]
        directory = data_root / case_id
        for filename, digest in entry["files"].items():
            if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Dataset integrity check failed: {case_id}/{filename}")
        reference = json.loads((directory / "reference.json").read_text())
        result = (detect_pool if candidate_pool else detect)(
            read_nifti(str(directory / "orig.nii.gz")), read_nifti(str(directory / "mask.nii.gz")),
            config,
        )
        unfiltered_prediction = result.prediction(case_id)
        unfiltered_results.append(evaluate_case(unfiltered_prediction, reference, tolerance_mm))
        filter_start = perf_counter()
        model_decisions = filter_detection(result, model) if model else None
        filter_s = perf_counter() - filter_start if model else 0.0
        prediction = result.prediction(case_id)
        metrics = evaluate_case(prediction, reference, tolerance_mm)
        metrics["runtime_s"] = result.timings["total_s"] + filter_s
        metrics["family"] = entry.get("family")
        results.append(metrics)
        predictions[case_id] = {
            "prediction": prediction, "diagnostics": result.diagnostics(),
            "unfiltered_prediction": unfiltered_prediction, "candidate_model": model_decisions,
        }
        references[case_id] = reference
        print(f"{case_id}: TP={metrics['true_positives']} FP={metrics['false_positives']} "
              f"FN={metrics['false_negatives']} ({metrics['runtime_s']:.2f}s)", flush=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True, check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=Path(__file__).parent, capture_output=True, text=True, check=False,
    )
    sensitivity = {
        f"{alternative:g}": summarize_cases([
            evaluate_case(predictions[case]["prediction"], references[case], alternative)
            for case in predictions
        ])
        for alternative in (2, 3, 5)
    }
    report = {
        "dataset_source": manifest["source"], "dataset_usage": manifest.get("usage"),
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "working_tree_may_include_uncommitted_changes": bool(status.stdout) if status.returncode == 0 else None,
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "manifest_sha256": hashlib.sha256((data_root / "manifest.json").read_bytes()).hexdigest(),
        "detector_config": asdict(config),
        "candidate_pool": candidate_pool,
        "candidate_model": {
            "sha256": hashlib.sha256(candidate_model.read_bytes()).hexdigest(),
            "threshold": model.threshold, "split": model.split,
            "dataset_overlap": {
                partition: sorted(set(cases) & set(predictions))
                for partition, cases in model.split.items()
            },
        } if candidate_model and model else None,
        "unfiltered_summary": summarize_cases(unfiltered_results),
        "matching_tolerance_mm": tolerance_mm,
        "tolerance_sensitivity": sensitivity,
        "summary": summarize_cases(results), "cases": results, "predictions": predictions,
        "mean_runtime_s": float(np.mean([r["runtime_s"] for r in results])) if results else None,
        "maximum_runtime_s": float(np.max([r["runtime_s"] for r in results])) if results else None,
        "by_family": {
            family: summarize_cases([r for r in results if r["family"] == family])
            for family in sorted({r["family"] for r in results if r["family"] is not None})
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tolerance-mm", type=float, default=3)
    parser.add_argument("--candidate-model", type=Path, help="Evaluate an optional local candidate filter.")
    parser.add_argument("--candidate-pool", action="store_true", help="Evaluate the strict/review union.")
    parser.add_argument(
        "--variant",
        choices=("strict", "finer", "relaxed", "review", "partial-volume", "native-contrast", "wall-parallel"),
        default="native-contrast",
    )
    args = parser.parse_args()
    if not np.isfinite(args.tolerance_mm) or args.tolerance_mm <= 0:
        parser.error("Matching tolerance must be positive and finite.")
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    variants = {
        "strict": DetectorConfig(support_contrast_fraction=1.0, native_contrast_scale=0),
        "finer": DetectorConfig(spacing_mm=0.75, support_contrast_fraction=1.0, native_contrast_scale=0),
        "relaxed": DetectorConfig(blood_lower_scale=2.0, support_contrast_fraction=1.0, native_contrast_scale=0),
        "review": DetectorConfig.review(support_contrast_fraction=1.0, native_contrast_scale=0),
        "partial-volume": DetectorConfig(support_contrast_fraction=0.5, native_contrast_scale=0),
        "native-contrast": DetectorConfig(native_contrast_scale=1.2),
        "wall-parallel": DetectorConfig(native_contrast_scale=1.2, parallel_clearance_mm=2.5),
    }
    try:
        if args.candidate_pool and args.variant != "review":
            parser.error("--candidate-pool requires --variant review.")
        config = DetectorConfig.review() if args.candidate_pool else variants[args.variant]
        report = benchmark(
            args.data_root, args.output, args.tolerance_mm, config,
            args.candidate_model, args.candidate_pool,
        )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
