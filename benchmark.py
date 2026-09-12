"""Evaluate a labeled case bundle offline, preserving per-case errors and predictions."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import SimpleITK as sitk

from detector import DetectorConfig, detect
from evaluate import evaluate_case, summarize_cases
from nifti_io import read_nifti


def benchmark(
    data_root: Path, output: Path, tolerance_mm: float, config: DetectorConfig | None = None,
) -> dict:
    if output.exists():
        raise FileExistsError("Report already exists; choose a new output to preserve prior evaluations.")
    config = config or DetectorConfig()
    manifest = json.loads((data_root / "manifest.json").read_text())
    results = []
    predictions = {}
    references = {}
    for entry in manifest["cases"]:
        case_id = entry["case_id"]
        directory = data_root / case_id
        for filename, digest in entry["files"].items():
            if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Dataset integrity check failed: {case_id}/{filename}")
        reference = json.loads((directory / "reference.json").read_text())
        result = detect(
            read_nifti(str(directory / "orig.nii.gz")), read_nifti(str(directory / "mask.nii.gz")), config,
        )
        prediction = result.prediction(case_id)
        metrics = evaluate_case(prediction, reference, tolerance_mm)
        metrics["runtime_s"] = result.timings["total_s"]
        metrics["family"] = entry.get("family")
        results.append(metrics)
        predictions[case_id] = {"prediction": prediction, "diagnostics": result.diagnostics()}
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
    parser.add_argument("--variant", choices=("strict", "finer", "relaxed", "review"), default="strict")
    args = parser.parse_args()
    if not np.isfinite(args.tolerance_mm) or args.tolerance_mm <= 0:
        parser.error("Matching tolerance must be positive and finite.")
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    variants = {
        "strict": DetectorConfig(),
        "finer": DetectorConfig(spacing_mm=0.75),
        "relaxed": DetectorConfig(blood_lower_scale=2.0),
        "review": DetectorConfig.review(),
    }
    try:
        report = benchmark(args.data_root, args.output, args.tolerance_mm, variants[args.variant])
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
