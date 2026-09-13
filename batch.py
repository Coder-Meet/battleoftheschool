"""Run the same detector on a directory of cases and record failures and runtimes."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import SimpleITK as sitk

from nifti_io import read_nifti
from pipeline import add_pipeline_arguments, run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("predictions"))
    parser.add_argument("--cases", nargs="*", help="Optional subset of case directory names.")
    add_pipeline_arguments(parser)
    args = parser.parse_args()
    if not args.data_root.is_dir():
        parser.error("The data directory does not exist.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    directories = sorted(p for p in args.data_root.iterdir() if p.is_dir())
    if args.cases:
        missing = set(args.cases) - {p.name for p in directories}
        if missing:
            parser.error(f"Unknown cases: {sorted(missing)}")
        directories = [p for p in directories if p.name in args.cases]
    if not directories:
        parser.error("No cases found.")
    records = []
    for directory in directories:
        start = perf_counter()
        try:
            images, masks = list(directory.glob("orig*.nii*")), list(directory.glob("mask*.nii*"))
            if len(images) != 1 or len(masks) != 1:
                raise ValueError("Expected exactly one CT and one mask.")
            result, workflow = run_pipeline(
                read_nifti(str(images[0])), read_nifti(str(masks[0])),
                workflow=args.pipeline, model_path=args.candidate_model, threshold=args.candidate_threshold,
            )
            (args.output_dir / f"{directory.name}.json").write_text(
                json.dumps(result.prediction(directory.name), indent=2, allow_nan=False) + "\n"
            )
            diagnostics = args.output_dir / "diagnostics"
            diagnostics.mkdir(exist_ok=True)
            (diagnostics / f"{directory.name}.json").write_text(
                json.dumps({**result.diagnostics(), "workflow": workflow}, indent=2, allow_nan=False) + "\n"
            )
            record = {
                "case_id": directory.name,
                "status": "ok",
                "branches": len(result.branches),
                "detection_s": result.timings["total_s"],
                "end_to_end_s": round(perf_counter() - start, 3),
            }
        except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
            record = {"case_id": directory.name, "status": "failed", "error": str(error)}
        records.append(record)
        print(json.dumps(record), flush=True)
    failures = sum(r["status"] == "failed" for r in records)
    report = {
        "cases": records,
        "failures": failures,
        "accuracy": None,
        "note": "Runtime smoke test only. Daughter reference annotations are required to measure accuracy.",
    }
    (args.output_dir / "batch_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
