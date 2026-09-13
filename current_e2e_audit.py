"""Fresh current-source detector/filter development audit; never fits or enables a model.

Run with numerical thread pools capped at four. Outputs must be a new directory.
All threshold/ablation comparisons are retrospective development evidence.
"""

import argparse
from collections import Counter
import copy
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import SimpleITK as sitk

from autolabel import candidate_record, fingerprint
from detector import DetectorConfig, detect, detect_pool
from final_eval_audit import audit_case, verify_release
from final_eval_topology import diagnose, infer
from final_evaluation import (
    CASES, ROOT, case_paths, digest, filtered, load_reference, read_json,
    score_prediction, score_variant, summarize, write_json,
)
from learning import CandidateModel, FEATURE_NAMES, features, filter_detection, load_reviews
from nifti_io import read_nifti
from research_run import source_hashes, validate_tree_source
from tabular_learning import TreeModel


MODELS = {
    "synthetic": "labels/research/current-source-v1/models/logistic.json",
    "pseudo_original": "labels/candidate-model.json",
    "pseudo_augmented": "labels/candidate-model-synthetic.json",
    "pseudo_excluding_five": "labels/final-eval/tabular/training/logistic.json",
}
CONFIGS = {
    "strict": DetectorConfig(),
    "old_cutoff": DetectorConfig(native_contrast_scale=1.2),
    "review": DetectorConfig.review(),
    "review_origin2": DetectorConfig.review(minimum_origin_diameter_mm=2),
    "pool": None,
    "contrast030": DetectorConfig(support_contrast_fraction=0.3),
    "roots6": DetectorConfig(roots_per_contact=6),
    "parallel": DetectorConfig(parallel_clearance_mm=1.5),
    "recovery": DetectorConfig(
        support_contrast_fraction=0.3, roots_per_contact=6, parallel_clearance_mm=1.5,
    ),
}
THRESHOLDS = (0.05, 0.10, 0.15, 0.30, 0.50, 0.70, 0.85)


def csv_file(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def canonical_fingerprint(value):
    """Ignore JSON whitespace, preserving exact geometry and feature values."""
    return json.dumps(json.loads(value), separators=(",", ":"))


def proposal_coverage(metrics):
    """Union independently matched target IDs; never count duplicate profile proposals twice."""
    output = {}
    for tolerance in ("2", "3", "5"):
        rows = []
        for index, case in enumerate(CASES):
            reference_ids = {d["instance_id"] for d in load_reference(case)["daughters"]}
            matched = {
                match["reference_id"] for name, value in metrics.items() if name.endswith("/plain")
                for match in value[tolerance]["cases"][index]["matches"]
            }
            rows.append({"case_id": case, "matched_reference_ids": sorted(matched),
                         "unmatched_reference_ids": sorted(reference_ids - matched)})
        output[tolerance] = {"matched_targets": sum(len(row["matched_reference_ids"]) for row in rows),
                             "cases": rows}
    return {"by_tolerance": output,
            "note": "Observed union of independently matched target IDs, not a theoretical detector ceiling."}


def model_audit(models):
    report = {}
    for name, model in models.items():
        report[name] = {
            "path": MODELS[name], "sha256": digest(ROOT / MODELS[name]),
            "threshold": model.threshold,
            "reference_overlap": {key: sorted(set(CASES) & set(ids)) for key, ids in model.split.items()},
            "note": "Schema compatible; logistic artifacts do not enforce detector-source compatibility.",
        }
    trees = {}
    for path in sorted((ROOT / "labels/research/current-source-v1/models").glob("*.json")):
        if path.name in ("logistic.json", "freeze.json"):
            continue
        try:
            validate_tree_source(TreeModel.load(path), source_hashes())
            trees[path.name] = "compatible"
        except (ValueError, KeyError) as error:
            trees[path.name] = str(error)
    return {"logistics": report, "research_trees": trees}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--all-cases", action="store_true", help="Also measure proposal/review coverage on all 25 scans.")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    models = {key: CandidateModel.load(ROOT / path) for key, path in MODELS.items()}
    source_files = [
        "current_e2e_audit.py", "detector.py", "learning.py", "run.py", "evaluate.py",
        "final_evaluation.py", "final_eval_topology.py", "final_eval_audit.py",
        "nifti_io.py", "score_references.py", "autolabel.py", "research_run.py",
        "tabular_learning.py", "research_validation.py", "requirements.txt",
    ]
    source_identity = {path: digest(ROOT / path) for path in source_files}
    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "command": sys.argv, "python": sys.version, "platform": platform.platform(),
        "source_sha256": source_identity,
        "configurations": {k: asdict(v) if v else "detect_pool: review plus strict" for k, v in CONFIGS.items()},
        "models": model_audit(models), "thresholds": THRESHOLDS,
        "status": "Post-reference development; no fitting, no new independent test set.",
        "scoring": "Maximum-cardinality one-to-one LPS ostium matching at 2/3/5 mm; local proxy.",
    }
    write_json(out / "provenance.json", provenance)
    release = verify_release()
    if not release["all_verified"]:
        raise ValueError("Reference release verification failed")
    write_json(out / "release-verification.json", release)
    input_audits = [audit_case(case) for case in CASES]
    for row in input_audits:
        if not row["input_identity"]["release_image_identical"] or not row["input_identity"]["release_mask_identical"]:
            raise ValueError("Reference and inference image identity differs")
    write_json(out / "input-audit.json", {"cases": input_audits})
    predictions, candidate_rows, runtime_rows, score_rows, cache = {}, [], [], [], {}
    for case in CASES:
        paths = case_paths(case)
        image, mask = (read_nifti(str(p)) for p in paths)
        for profile, config in CONFIGS.items():
            start = time.perf_counter()
            result = detect(image, mask, config) if config else detect_pool(image, mask)
            elapsed = time.perf_counter() - start
            cache[case, profile] = result
            prediction = result.prediction(case)
            predictions.setdefault(profile + "/plain", {})[case] = prediction
            write_json(out / "cases" / case / f"{profile}-diagnostics.json", result.diagnostics())
            write_json(out / "cases" / case / f"{profile}.json", prediction)
            matched = {m["prediction_id"] for m in score_prediction(prediction, load_reference(case))["matches"]}
            matrix = np.asarray([features(b) for b in result.branches]).reshape(-1, len(FEATURE_NAMES))
            scores = {name: model.scores(matrix) for name, model in models.items()}
            for i, branch in enumerate(result.branches):
                candidate_rows.append({
                    "case": case, "profile": profile, "instance_id": branch.instance_id,
                    "matched_3mm": branch.instance_id in matched,
                    **dict(zip(FEATURE_NAMES, matrix[i])),
                    **{name + "_score": float(values[i]) for name, values in scores.items()},
                })
            for name, model in models.items():
                thresholds = sorted({*THRESHOLDS, model.threshold})
                for threshold in thresholds:
                    key = f"{profile}/{name}@{threshold:.17g}"
                    predictions.setdefault(key, {})[case] = filtered(prediction, scores[name], threshold)
                actual = copy.deepcopy(result)
                filter_detection(actual, model)
                if json.dumps(actual.prediction(case), sort_keys=True) != json.dumps(
                    filtered(prediction, scores[name], model.threshold), sort_keys=True,
                ):
                    raise AssertionError("Saved-score filtering differs from production filter_detection")
            runtime_rows.append({"case": case, "profile": profile, "inference_wall_s": elapsed,
                                 "proposed_roots": result.candidates, "retained": len(result.branches),
                                 "rejections": result.rejections})
            print(f"{case} {profile}: {result.candidates} roots -> {len(result.branches)} outputs ({elapsed:.2f}s)", flush=True)
        # Independent instrumented replay must exactly equal current production output before diagnosis.
        replay, audit, scan, evidence = infer(image, mask, CONFIGS["strict"])
        replay["case_id"] = case
        if json.dumps(replay, sort_keys=True) != json.dumps(predictions["strict/plain"][case], sort_keys=True):
            raise AssertionError("Instrumented resolver differs from production")
        write_json(out / "cases" / case / "strict-trace-audit.json", audit)
        write_json(out / "cases" / case / "strict-reference-diagnosis.json",
                   diagnose(load_reference(case), replay, audit, scan, evidence, CONFIGS["strict"]))
        del scan, evidence
        # Exercise the actual submission CLI and compare its JSON to the API run.
        for name in ("plain", "synthetic"):
            cli_path = out / "cli" / f"{case}-{name}.json"
            command = [sys.executable, str(ROOT / "run.py"), "--pipeline", "strict", "--image", str(paths[0]),
                       "--aorta-mask", str(paths[1]), "--case-id", case, "--output", str(cli_path)]
            if name != "plain":
                command += ["--candidate-model", str(ROOT / MODELS[name])]
            start = time.perf_counter()
            completed = subprocess.run(command, capture_output=True, text=True, check=True)
            expected_key = "strict/plain" if name == "plain" else f"strict/{name}@{models[name].threshold:.17g}"
            if read_json(cli_path) != json.loads(json.dumps(predictions[expected_key][case])):
                raise AssertionError("Submission CLI differs from API predictions")
            write_json(out / "cli" / f"{case}-{name}-run.json", {
                "command": command, "exit_code": completed.returncode,
                "wall_s": time.perf_counter() - start, "stdout": completed.stdout,
                "stderr": completed.stderr, "exact_api_equality": True,
            })
    metrics = {name: score_variant(values) for name, values in predictions.items()}
    for name, tolerances in metrics.items():
        for tolerance, block in tolerances.items():
            score_rows.append({"variant": name, "tolerance_mm": tolerance,
                               **{k: v for k, v in block["summary"].items() if k != "errors"}})
    write_json(out / "metrics.json", metrics)
    write_json(out / "runtime.json", {"rows": runtime_rows,
               "note": "Local macOS four-thread timing, loading excluded; not Windows acceptance evidence."})
    csv_file(out / "summary.csv", score_rows)
    csv_file(out / "candidates.csv", candidate_rows)
    # Selection diagnostics only: cases and configuration design have been seen before.
    eligible = [name for name in metrics if "/plain" in name or "/synthetic@" in name]
    folds = []
    for i, case in enumerate(CASES):
        def rank(name):
            rows = metrics[name]["3"]["cases"]
            summary = summarize([row for j, row in enumerate(rows) if i != j])
            return (-float(summary["f1"] or 0), summary["count_mae"], name)
        choice = min(eligible, key=rank)
        folds.append({"held_out_case": case, "choice": choice, "score": metrics[choice]["3"]["cases"][i]})
    write_json(out / "selection-stability.json", {"folds": folds,
               "pooled": summarize([row["score"] for row in folds]),
               "note": "Retrospective selection stability only, not independent held-out validation."})
    write_json(out / "proposal-coverage.json", proposal_coverage(metrics))
    if args.all_cases:
        reviews = load_reviews([ROOT / "labels/reviews.json"])
        lookup = {(r["case_id"], canonical_fingerprint(r["fingerprint"])): r for r in reviews}
        inventory, queue = [], []
        for directory in sorted((ROOT / "data").glob("subject*")):
            case = directory.name
            image_path, mask_path = case_paths(case)
            image, mask = read_nifti(str(image_path)), read_nifti(str(mask_path))
            for profile in ("strict", "pool"):
                result = cache.get((case, profile))
                if result is None:
                    result = detect(image, mask) if profile == "strict" else detect_pool(image, mask)
                write_json(out / "inventory" / case / f"{profile}.json", result.diagnostics())
                matrix = np.asarray([features(b) for b in result.branches]).reshape(-1, len(FEATURE_NAMES))
                scores = models["synthetic"].scores(matrix)
                counts = Counter()
                for branch, probability in zip(result.branches, scores):
                    fp = fingerprint(candidate_record(branch))
                    previous = lookup.get((case, canonical_fingerprint(fp)))
                    label = previous["label"] if previous else "unreviewed_current_identity"
                    counts[label] += 1
                    queue.append({"case": case, "profile": profile, "instance_id": branch.instance_id,
                                  "synthetic_score": float(probability), "exact_previous_label": label,
                                  "fingerprint": fp})
                inventory.append({
                    "case": case, "profile": profile, "proposals": len(result.branches),
                    "at_frozen_threshold": int(np.sum(scores >= models["synthetic"].threshold)),
                    "at_015": int(np.sum(scores >= 0.15)), "at_030": int(np.sum(scores >= 0.3)),
                    "at_050": int(np.sum(scores >= 0.5)),
                    "exact_confirmed": counts["confirmed"], "exact_rejected": counts["rejected"],
                    "unreviewed_current_identity": counts["unreviewed_current_identity"],
                    "image_sha256": digest(image_path), "mask_sha256": digest(mask_path),
                })
                print(f"inventory {case} {profile}: {len(result.branches)} -> {np.sum(scores >= 0.15)} at .15", flush=True)
        csv_file(out / "inventory.csv", inventory)
        csv_file(out / "review-queue.csv", queue)
    if source_identity != {path: digest(ROOT / path) for path in source_files}:
        raise RuntimeError("Sources changed during execution; results are not a single-source replay")
    write_json(out / "completion.json", {
        "complete": True, "reference_cases": list(CASES), "variants": len(metrics),
        "cli_comparisons": len(CASES) * 2, "all_cases_inventory": args.all_cases,
        "source_unchanged": True,
    })
    print(f"Complete: {out}", flush=True)


if __name__ == "__main__":
    main()
