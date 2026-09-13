"""Fixed post-reference comparison of wall connectivity and 0.75 mm sampling."""

import argparse
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import SimpleITK as sitk

from detector import DetectorConfig, detect
from final_evaluation import (
    CASES, ROOT, case_paths, digest, load_reference, read_json, score_prediction,
    summarize, write_json,
)
from nifti_io import read_nifti
from origin_recovery import detect_connected
from stress import FAMILIES, generate_case as stress_case
from synthetic import generate_case as synthetic_case

VARIANTS = (
    ("strict", 1.0, False), ("connected", 1.0, True),
    ("fine", 0.75, False), ("fine-connected", 0.75, True),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", choices=("references", "topology", "fresh", "confirmation"), required=True)
    parser.add_argument("--guarded-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory; previous evidence is immutable.")
    frozen_path = ROOT / "labels/final-eval/topology/experiment_config.json"
    cohort = read_json(frozen_path)["synthetic"]
    seeds = cohort["stress_seeds"] if args.cohort == "topology" else [28411, 91513]
    if args.cohort == "confirmation":
        seeds = [28412, 91514]
    variants = (("strict", 1.0, False), ("guarded", 1.0, True)) if args.guarded_only else VARIANTS
    families = cohort["stress_families"] if args.cohort == "topology" else list(FAMILIES)
    specs = (
        [("real", case, 0) for case in CASES] if args.cohort == "references" else
        [("stress", family, seed) for seed in seeds for family in families]
    )
    if args.cohort == "topology":
        specs += [("simple", family, cohort["simple_seed"]) for family in cohort["simple_scenarios"]]
    metadata = {
        "status": "POST-REFERENCE DEVELOPMENT; no independent clinical accuracy estimate",
        "cohort": args.cohort, "cases": specs,
        "require_original_root": args.guarded_only,
        "variants": {
            name: {"detector_config": asdict(DetectorConfig(spacing_mm=spacing)), "recovery": recovery}
            for name, spacing, recovery in variants
        },
        "source_sha256": {name: digest(ROOT / name) for name in (
            "detector.py", "origin_recovery.py", "accuracy_recheck.py", "stress.py",
            "synthetic.py", "final_evaluation.py", "evaluate.py", "nifti_io.py",
        )},
        "topology_cohort_sha256": digest(frozen_path),
        "gate": [
            "No threshold or model fitting on the five references.",
            "Improve reference F1 without losing baseline matches or worsening count MAE.",
            "No lost synthetic baseline matches, extra synthetic false positives, or negative-control detections.",
            "New procedural seeds use known generators and are not independent real-patient validation.",
            "Actual organizer hardware remains unverified.",
        ],
    }
    write_json(args.output / "config.json", metadata)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    rows: dict[str, list[dict]] = {name: [] for name, _, _ in variants}
    for kind, family, seed in specs:
        if kind == "real":
            image_path, mask_path = case_paths(str(family))
            image, mask = read_nifti(str(image_path)), read_nifti(str(mask_path))
            reference = load_reference(str(family))
        else:
            case = stress_case(str(family), seed) if kind == "stress" else synthetic_case(int(family), seed)
            image, mask, reference = case.image, case.parent, case.reference
            for daughter, geometry in zip(reference["daughters"], case.provenance["geometry"]):
                daughter["centerline_xyz_mm"] = geometry["proximal_centerline_xyz_mm"]
        for name, spacing, recovery in variants:
            start = perf_counter()
            config = DetectorConfig(spacing_mm=spacing)
            detection = (
                detect_connected(image, mask, config, require_root=args.guarded_only)
                if recovery else detect(image, mask, config)
            )
            prediction = detection.prediction(reference["case_id"])
            record: dict = {
                "prediction": prediction, "reference": reference, "diagnostics": detection.diagnostics(),
                "runtime_s": perf_counter() - start,
                "scores": {str(t): score_prediction(prediction, reference, t) for t in (2, 3, 5)},
            }
            rows[name].append(record)
            write_json(args.output / name / f"{reference['case_id']}.json", record)
            score = record["scores"]["3"]
            print(reference["case_id"], name, "TP/FP/FN",
                  *(score[key] for key in ("true_positives", "false_positives", "false_negatives")), flush=True)
    report = {
        **metadata,
        "results": {
            name: {
                "scores": {str(t): summarize([row["scores"][str(t)] for row in records]) for t in (2, 3, 5)},
                "negative_control_predictions": sum(
                    len(row["prediction"]["daughters"]) for row in records if not row["reference"]["daughters"]
                ),
                "maximum_in_process_runtime_s": max(row["runtime_s"] for row in records),
            } for name, records in rows.items()
        },
    }
    write_json(args.output / "report.json", report)
    print({name: result["scores"]["3"] for name, result in report["results"].items()}, flush=True)


if __name__ == "__main__":
    main()
