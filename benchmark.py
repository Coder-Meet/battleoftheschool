"""Evaluate a labeled case bundle offline, preserving per-case errors and predictions."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import SimpleITK as sitk

from detector import detect
from evaluate import evaluate_case, summarize_cases
from nifti_io import read_nifti


def benchmark(data_root: Path, output: Path, tolerance_mm: float) -> dict:
    manifest = json.loads((data_root / "manifest.json").read_text())
    results = []
    predictions = {}
    for entry in manifest["cases"]:
        case_id = entry["case_id"]
        directory = data_root / case_id
        reference = json.loads((directory / "reference.json").read_text())
        result = detect(read_nifti(str(directory / "orig.nii.gz")), read_nifti(str(directory / "mask.nii.gz")))
        prediction = result.prediction(case_id)
        metrics = evaluate_case(prediction, reference, tolerance_mm)
        metrics["runtime_s"] = result.timings["total_s"]
        results.append(metrics)
        predictions[case_id] = {"prediction": prediction, "diagnostics": result.diagnostics()}
        print(f"{case_id}: TP={metrics['true_positives']} FP={metrics['false_positives']} "
              f"FN={metrics['false_negatives']} ({metrics['runtime_s']:.2f}s)", flush=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True, check=False,
    )
    report = {
        "dataset_source": manifest["source"], "dataset_usage": manifest.get("usage"),
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "working_tree_may_include_uncommitted_changes": True,
        "matching_tolerance_mm": tolerance_mm,
        "summary": summarize_cases(results), "cases": results, "predictions": predictions,
        "mean_runtime_s": float(np.mean([r["runtime_s"] for r in results])) if results else None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tolerance-mm", type=float, default=3)
    args = parser.parse_args()
    if not np.isfinite(args.tolerance_mm) or args.tolerance_mm <= 0:
        parser.error("Matching tolerance must be positive and finite.")
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    try:
        report = benchmark(args.data_root, args.output, args.tolerance_mm)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
