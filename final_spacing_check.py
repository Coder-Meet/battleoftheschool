"""Compare fixed 1 and 1.5 mm grids on the frozen topology regression cohort."""

import argparse
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import SimpleITK as sitk

from detector import DetectorConfig, detect
from final_evaluation import ROOT, digest, read_json, score_prediction, summarize, write_json
from stress import generate_case as stress_case
from synthetic import generate_case as synthetic_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory to preserve prior evidence.")
    args.output.mkdir(parents=True)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    frozen_path = ROOT / "labels/final-eval/topology/experiment_config.json"
    cohort = read_json(frozen_path)["synthetic"]
    specs = [
        ("stress", family, seed)
        for seed in cohort["stress_seeds"] for family in cohort["stress_families"]
    ] + [("simple", scenario, cohort["simple_seed"]) for scenario in cohort["simple_scenarios"]]
    configs = {"strict": DetectorConfig(), "spacing1.5": DetectorConfig(spacing_mm=1.5)}
    records: dict[str, list[dict]] = {name: [] for name in configs}
    metadata = {
        "status": "POST-REFERENCE DEVELOPMENT; synthetic regression, not independent accuracy",
        "configs": {name: asdict(config) for name, config in configs.items()},
        "source_sha256": {name: digest(ROOT / name) for name in (
            "detector.py", "stress.py", "synthetic.py", "evaluate.py", "final_spacing_check.py",
        )},
        "cohort_sha256": digest(frozen_path),
    }
    write_json(args.output / "config.json", metadata)
    for kind, family, seed in specs:
        case = stress_case(family, seed) if kind == "stress" else synthetic_case(family, seed)
        for daughter, geometry in zip(case.reference["daughters"], case.provenance["geometry"]):
            daughter["centerline_xyz_mm"] = geometry["proximal_centerline_xyz_mm"]
        for name, config in configs.items():
            start = perf_counter()
            prediction = detect(case.image, case.parent, config).prediction(case.reference["case_id"])
            record = {
                "prediction": prediction, "reference": case.reference,
                "runtime_s": perf_counter() - start,
                "scores": {str(t): score_prediction(prediction, case.reference, t) for t in (2, 3, 5)},
            }
            records[name].append(record)
            write_json(args.output / name / f"{case.reference['case_id']}.json", record)
        print(case.reference["case_id"], flush=True)
    report: dict = {
        **metadata,
        "variants": {
            name: {
                "completed_cases": len(rows),
                "scores": {str(t): summarize([row["scores"][str(t)] for row in rows]) for t in (2, 3, 5)},
                "negative_control_predictions": sum(
                    len(row["prediction"]["daughters"]) for row in rows if not row["reference"]["daughters"]
                ),
                "total_runtime_s": sum(row["runtime_s"] for row in rows),
            } for name, rows in records.items()
        },
    }
    write_json(args.output / "report.json", report)
    print({name: row["scores"]["3"] for name, row in report["variants"].items()})


if __name__ == "__main__":
    main()
