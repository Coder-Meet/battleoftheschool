"""Reproducible tabular comparisons; never changes the production detector."""

import argparse
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import subprocess
from threading import Event, Thread
from time import perf_counter

import numpy as np
import psutil
import SimpleITK as sitk

from candidate_patches import EXTRA_FEATURE_NAMES
from detector import Detection, DetectorConfig, detect, detect_pool
from final_evaluation import (
    CASES, REFERENCE_ROOT, ROOT, THRESHOLDS, case_paths, digest, filtered,
    load_reference, read_json, score_prediction, score_variant, summarize, write_json,
)
from learning import CandidateModel, FEATURE_NAMES, features, load_reviews, train
from nifti_io import read_nifti
from research_run import score_detection, source_hashes, validate_tree_source
from tabular_learning import TreeModel, json_sha256

OUTPUT = ROOT / "labels/final-eval/tabular"
PROFILES = ("strict", "review-union")
THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
)


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def file_identity(path: Path) -> str:
    with path.open("rb") as stream:
        header = stream.read(256)
    if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
        match = re.search(rb"oid sha256:([a-f0-9]{64})", header)
        if match is None:
            raise ValueError(f"Malformed LFS pointer: {path}")
        return match[1].decode()
    return digest(path)


def input_inventory() -> tuple[dict, dict[str, str]]:
    verified = {row["case_id"]: row for row in read_json(REFERENCE_ROOT / "validation.json")["cases"]}
    inputs: dict = {}
    aliases: dict[str, str] = {}
    identities: dict[tuple[str, str], str] = {}
    for case in CASES:
        image, mask = case_paths(case)
        if image.stat().st_size < 1024 or mask.stat().st_size < 1024:
            raise ValueError(f"Unresolved or truncated image data for {case}; fetch its LFS objects.")
        hashes = {"image": digest(image), "mask": digest(mask)}
        if hashes != {"image": verified[case]["image_sha256"], "mask": verified[case]["mask_sha256"]}:
            raise ValueError(f"Input bytes differ from verified release: {case}")
        inputs[case] = hashes
        identities[hashes["image"], hashes["mask"]] = case
        aliases[case] = case
        aliases[f"case_{int(case[7:])}"] = case
    for directory in sorted((ROOT / "data").glob("subject*")):
        image, mask = case_paths(directory.name)
        pair = file_identity(image), file_identity(mask)
        aliases[directory.name] = identities.get(pair, directory.name)
    return inputs, aliases


def exclude_reference_reviews(
    rows: list[dict], aliases: dict[str, str], inputs: dict,
) -> tuple[list[dict], dict]:
    blocked_hashes = {value for hashes in inputs.values() for value in hashes.values()}
    initial: dict[int, list[str]] = {}
    for index, row in enumerate(rows):
        reasons = []
        if aliases.get(row["case_id"], row["case_id"]) in CASES:
            reasons.append("reference_case_or_byte_identical_alias")
        if aliases.get(row.get("group_id", ""), row.get("group_id", "")) in CASES:
            reasons.append("reference_group")
        if blocked_hashes & set(row.get("input_sha256", {}).values()):
            reasons.append("reference_input_hash")
        if reasons:
            initial[index] = reasons
    fingerprints = {rows[index]["fingerprint"] for index in initial}
    kept, removed = [], []
    for index, row in enumerate(rows):
        reasons = initial.get(index, []).copy()
        if row["fingerprint"] in fingerprints and "reference_case_or_byte_identical_alias" not in reasons:
            reasons.append("identical_excluded_candidate_fingerprint")
        if row["case_id"] not in aliases:
            reasons.append("unverified_pseudo_case_identity")
        if reasons:
            removed.append({
                "case_id": row["case_id"], "instance_id": row["instance_id"],
                "canonical_case_id": aliases.get(row["case_id"]),
                "fingerprint_sha256": json_sha256(row["fingerprint"]), "reasons": reasons,
            })
        else:
            kept.append(dict(row, case_id=aliases[row["case_id"]]))
    return kept, {
        "excluded_reference_case_ids": list(CASES),
        "reference_aliases": {name: case for name, case in aliases.items() if case in CASES},
        "removed_case_ids": sorted({row["case_id"] for row in removed}),
        "removed_rows": removed, "retained_rows": len(kept),
        "reference_cases_without_pseudo_rows": [
            case for case in CASES if not any(aliases.get(row["case_id"]) == case for row in rows)
        ],
        "alias_policy": "Case/group IDs, byte hashes (including LFS OIDs), exact candidate fingerprints; "
                        "unverified pseudo-case IDs are excluded, never inferred from branch numbers.",
        "identity_inventory": aliases,
    }


def fit_excluded(output: Path, inputs: dict, aliases: dict[str, str]) -> dict:
    path = ROOT / "labels/reviews.json"
    rows = load_reviews([path])
    selected, exclusions = exclude_reference_reviews(rows, aliases, inputs)
    historical_split = read_json(ROOT / "labels/split.json")
    cases = {row["case_id"] for row in selected}
    split = {part: [case for case in ids if case in cases] for part, ids in historical_split.items()}
    model, report = train(selected, split)
    directory = output / "training"
    model_path = directory / "logistic.json"
    write_json(directory / "split.json", split)
    model.save(model_path)
    report.update({
        "algorithm": "Existing learning.train L-BFGS-B logistic; mean log loss + 0.05*sum(weights**2).",
        "post_reference_refit": True, "reference_outcomes_used_for_fitting": False,
        "corpus_path": relative(path), "corpus_sha256": digest(path),
        "historical_split_sha256": digest(ROOT / "labels/split.json"),
        "split_policy": "Preserve existing case partitions after removing all reference identities.",
        "exclusions": exclusions, "model_sha256": digest(model_path),
        "source_sha256": source_hashes(), "driver_sha256": digest(Path(__file__)),
        "selected_records_sha256": json_sha256(selected),
        "selected_records": [
            {key: row[key] for key in ("case_id", "instance_id", "fingerprint", "label", "labeller")}
            for row in selected
        ],
        "calibration": {"method": "none", "fitting_case_ids": [], "clinical_probability_claim": False},
        "feature_lineage": "Frozen historical pseudo-review feature vectors. No labels transferred to "
                           "current proposals. Historical extraction commit is not recorded by these reviews; "
                           "current-source inference is a declared feature-schema transfer.",
        "fitting_source_counts": dict(Counter(row["labeller"] for row in selected)),
    })
    write_json(directory / "report.json", report)
    return report


def audit_models(output: Path) -> list[dict]:
    candidates = [
        ("tabular-original-logistic", ROOT / "labels/candidate-model.json"),
        ("tabular-synthetic-augmented-logistic", ROOT / "labels/candidate-model-synthetic.json"),
        ("tabular-excluded-five-logistic", output / "training/logistic.json"),
    ]
    for family in ("current-source-v1", "cohort-v1"):
        directory = ROOT / "labels/research" / family / "models"
        candidates.extend(
            (f"tabular-{family}-{path.stem}", path)
            for path in sorted(directory.glob("*.json")) if path.name != "freeze.json"
        )
    audit = []
    for name, path in candidates:
        entry = {
            "name": name, "path": relative(path),
            "eligible_for_selection": True, "ineligible_reason": "", "compatible": True,
            "failure": "",
        }
        try:
            payload = read_json(path)
            tree = payload.get("model_type") == "branchseed_candidate_trees"
            entry.update({
                "sha256": digest(path), "kind": "tree" if tree else "logistic",
                "threshold": payload["threshold"], "feature_names": payload["feature_names"],
                "split": payload["split"], "calibration": payload.get("calibration"),
                "reference_exposure_by_partition": {
                    part: sorted(set(cases) & set(CASES)) for part, cases in payload["split"].items()
                },
            })
            thresholds(payload["threshold"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            entry.update(compatible=False, eligible_for_selection=False,
                         failure=str(error), ineligible_reason=str(error))
            audit.append(entry)
            continue
        if "current-source-v1" in str(path):
            entry["lineage"] = "Synthetic-only current-source regeneration, no real review fitting."
            entry["lineage_sha256"] = {
                relative(p): digest(p) for p in (
                    path.parent / "freeze.json", path.parent.parent / "plan.json",
                    path.parent.parent / "reproduction.json",
                )
            }
        elif "cohort-v1" in str(path):
            entry["lineage"] = "Synthetic-only historical extraction; current-source re-extraction exists."
            entry["lineage_sha256"] = {
                relative(p): digest(p) for p in (path.parent / "freeze.json", path.parent.parent / "plan.json")
            }
        elif "excluded-five" in name:
            entry["lineage"] = relative(output / "training/report.json")
            entry["lineage_sha256"] = {entry["lineage"]: digest(output / "training/report.json")}
            entry["post_reference_refit"] = True
        else:
            entry["lineage"] = "Historical AI pseudo-reviews; no validated historical detector source contract."
            entry["eligible_for_selection"] = False
            entry["ineligible_reason"] = (
                "Retrospective real-image pseudo-label baseline. Reference images occurred in fitting/"
                "threshold selection/previous inspection; feature-schema transfer to current proposals."
            )
        try:
            if tree:
                model = TreeModel.load(path)
                validate_tree_source(model, source_hashes())
                entry["inference_contract"] = model.metadata["inference_contract"]
            else:
                CandidateModel.load(path)
                if "cohort-v1" in str(path):
                    replacement = ROOT / "labels/research/current-source-v1/models/logistic.json"
                    if digest(path) != digest(replacement):
                        raise ValueError("Historical logistic differs from the source-regenerated artifact.")
                    entry["identical_current_source_artifact"] = relative(replacement)
        except (OSError, ValueError, KeyError, TypeError) as error:
            entry.update(compatible=False, eligible_for_selection=False,
                         failure=str(error), ineligible_reason=str(error))
            replacement = ROOT / "labels/research/current-source-v1/models" / path.name
            if replacement.is_file():
                entry["reextracted_retrained_replacement"] = relative(replacement)
                entry["replacement_sha256"] = digest(replacement)
        audit.append(entry)
    write_json(output / "model-audit.json", {"models": audit})
    return audit


class MemorySample:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.done = Event()
        self.thread = Thread(target=self.sample, daemon=True)

    def sample(self) -> None:
        while not self.done.wait(0.01):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self) -> "MemorySample":
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.done.set()
        self.thread.join()


def candidate_records(result: Detection, case: str, inputs: dict, profile: str) -> list[dict]:
    records = []
    for branch in result.branches:
        vector = features(branch)
        fingerprint = json.dumps([
            branch.ostium_xyz_mm, branch.seed_xyz_mm, branch.direction_xyz, branch.radius_mm, vector,
        ])
        row = {
            "case_id": case, "instance_id": branch.instance_id, "features": vector,
            "fingerprint": fingerprint, "branch": asdict(branch),
        }
        row["candidate_sha256"] = json_sha256({
            **row, "input_sha256": inputs, "profile": profile, "source_sha256": source_hashes(),
        })
        records.append(row)
    return records


def extract_case(case: str, profile: str, inputs: dict, models: list[dict], output: Path) -> dict:
    with MemorySample() as memory:
        start = perf_counter()
        image_path, mask_path = case_paths(case)
        image, mask = read_nifti(str(image_path)), read_nifti(str(mask_path))
        result = detect(image, mask) if profile == "strict" else detect_pool(image, mask)
        detection_s = perf_counter() - start
        prediction = result.prediction(case)
        records = candidate_records(result, case, inputs, profile)
        scores: dict = {}
        failures = []
        for model in models:
            if not model["compatible"]:
                continue
            started = perf_counter()
            try:
                path = ROOT / model["path"]
                if model["kind"] == "tree":
                    probabilities, threshold, diagnostics = score_detection(
                        image, mask, result, case, case, inputs, tree_path=path, threads=4,
                    )
                    if [row["fingerprint"] for row in diagnostics["records"]] != [
                        row["fingerprint"] for row in records
                    ]:
                        raise ValueError("Tree extraction changed candidate identity or ordering.")
                    for record, extracted in zip(records, diagnostics["records"]):
                        if "extra_features" in extracted:
                            record["extra_features"] = extracted["extra_features"]
                    contract = diagnostics["contract"]
                else:
                    logistic = CandidateModel.load(path)
                    matrix = np.asarray([row["features"] for row in records], dtype=float).reshape(
                        len(records), len(FEATURE_NAMES),
                    )
                    probabilities = logistic.scores(matrix)
                    threshold = logistic.threshold
                    contract = {"feature_names": FEATURE_NAMES, "scaler": "saved train-only mean/std"}
                filtered(prediction, probabilities, threshold)
                scores[model["name"]] = {
                    "probabilities": probabilities.tolist(), "threshold": threshold,
                    "scoring_s": perf_counter() - started, "model_sha256": digest(path),
                    "contract": contract,
                }
            except (ValueError, OSError, KeyError, TypeError) as error:
                failures.append({"model": model["name"], "case_id": case, "profile": profile, "error": str(error)})
        cache = {
            "case_id": case, "profile": profile, "input_sha256": inputs,
            "source_sha256": source_hashes(), "driver_sha256": digest(Path(__file__)),
            "input_geometry": {
                name: {"size": list(volume.GetSize()), "spacing": list(volume.GetSpacing()),
                       "origin": list(volume.GetOrigin()), "direction": list(volume.GetDirection())}
                for name, volume in (("image", image), ("mask", mask))
            },
            "detector_config": asdict(result.config), "strict_config": asdict(DetectorConfig()),
            "feature_names": FEATURE_NAMES, "extra_feature_names": EXTRA_FEATURE_NAMES,
            "prediction": prediction, "records": records, "model_scores": scores, "failures": failures,
            "detection_and_input_load_s": detection_s, "total_case_job_s": perf_counter() - start,
        }
    cache["peak_rss_mb"] = memory.peak / 1024**2
    cache["records_sha256"] = json_sha256(records)
    write_json(output / "candidates" / profile / f"{case}.json", cache)
    return cache


def thresholds(frozen: float) -> list[float]:
    values = sorted({frozen, *THRESHOLDS})
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("Thresholds must be finite probabilities.")
    return values


def conditional_recall(predictions: dict, proposals: dict) -> dict:
    report = {}
    for tolerance in (2.0, 3.0, 5.0):
        cases = []
        for case in CASES:
            reference = load_reference(case)
            before = score_prediction(proposals[case], reference, tolerance)
            after = score_prediction(predictions[case], reference, tolerance)
            available = {match["reference_id"] for match in before["matches"]}
            retained = {match["reference_id"] for match in after["matches"]}
            cases.append({
                "case_id": case, "references": len(reference["daughters"]),
                "proposal_tp": before["true_positives"], "filtered_tp": after["true_positives"],
                "proposal_reference_ids": sorted(available), "retained_reference_ids": sorted(retained),
                "removed_reference_ids": sorted(available - retained),
                "matcher_reassigned_reference_ids": sorted(retained - available),
                "removed_candidate_ids": [
                    row["instance_id"] for row in proposals[case]["daughters"]
                    if row["instance_id"] not in {d["instance_id"] for d in predictions[case]["daughters"]}
                ],
            })
        total = sum(row["references"] for row in cases)
        proposed = sum(row["proposal_tp"] for row in cases)
        detected = sum(row["filtered_tp"] for row in cases)
        report[f"{tolerance:g}"] = {
            "proposal_recall": proposed / total, "end_to_end_recall": detected / total,
            "conditional_tp_retention": detected / proposed if proposed else None, "cases": cases,
            "interpretation": "Maximum-cardinality matches recomputed after filtering; ties can reassign IDs.",
        }
    return report


def write_variant(
    name: str, profile: str, model: dict | None, threshold: float,
    caches: dict[str, dict], output: Path,
) -> dict:
    proposals = {case: cache["prediction"] for case, cache in caches.items()}
    predictions, runtime = {}, {}
    for case in CASES:
        cache = caches[case]
        values = cache["model_scores"][model["name"]] if model else None
        start = perf_counter()
        predictions[case] = filtered(
            proposals[case], values["probabilities"] if values else np.ones(len(cache["records"])), threshold,
        )
        filter_s = perf_counter() - start
        runtime[case] = {
            "runtime_s": cache["detection_and_input_load_s"] + (values["scoring_s"] if values else 0) + filter_s,
            "peak_rss_mb": cache["peak_rss_mb"],
        }
        write_json(output / "predictions" / name / f"{case}.json", predictions[case])
    first = caches[CASES[0]]
    return {
        "name": name, "prediction_dir": relative(output / "predictions" / name),
        "configuration": {
            "profile": profile, "threshold": threshold, "model": model,
            "detector_config": first["detector_config"], "strict_config": first["strict_config"],
            "source_sha256": first["source_sha256"], "threads": 4,
            "candidate_caches": {
                case: {"path": relative(output / "candidates" / profile / f"{case}.json"),
                       "sha256": digest(output / "candidates" / profile / f"{case}.json")}
                for case in CASES
            },
        },
        "eligible_for_selection": model["eligible_for_selection"] if model else True,
        "ineligible_reason": model["ineligible_reason"] if model else "",
        "scores": score_variant(predictions), "runtime": runtime,
        "conditional_vs_proposal_recall": conditional_recall(predictions, proposals),
    }


def validate_cache(cache: dict, case: str, profile: str, inputs: dict) -> None:
    if cache["case_id"] != case or cache["profile"] != profile:
        raise ValueError("Candidate cache belongs to a different case/profile.")
    if cache["source_sha256"] != source_hashes() or cache["records_sha256"] != json_sha256(cache["records"]):
        raise ValueError("Candidate cache source or records changed; re-extract.")
    if cache["driver_sha256"] != digest(Path(__file__)):
        raise ValueError("Candidate cache driver changed; re-extract.")
    if cache["input_sha256"] != inputs[case]:
        raise ValueError("Candidate cache input identity changed.")
    if len(cache["records"]) != len(cache["prediction"]["daughters"]):
        raise ValueError("Candidate cache record/prediction count differs.")
    for record, prediction in zip(cache["records"], cache["prediction"]["daughters"]):
        if record["case_id"] != case or record["instance_id"] != prediction["instance_id"]:
            raise ValueError("Candidate cache ID or ordering differs.")
        if any(record["branch"][key] != value for key, value in prediction.items()):
            raise ValueError("Candidate cache prediction geometry differs.")
    for scores in cache["model_scores"].values():
        filtered(cache["prediction"], scores["probabilities"], scores["threshold"])


def report_variants(output: Path, models: list[dict]) -> dict:
    variants, failures = [], []
    inputs, _ = input_inventory()
    for profile in PROFILES:
        caches = {case: read_json(output / "candidates" / profile / f"{case}.json") for case in CASES}
        for case, cache in caches.items():
            validate_cache(cache, case, profile, inputs)
            failures.extend(cache["failures"])
        variants.append(write_variant(f"tabular-unfiltered-{profile}", profile, None, 0, caches, output))
        for model in models:
            if not model["compatible"]:
                continue
            missing = [case for case in CASES if model["name"] not in caches[case]["model_scores"]]
            if missing:
                failures.append({"model": model["name"], "profile": profile, "missing_cases": missing})
                continue
            if digest(ROOT / model["path"]) != model["sha256"]:
                raise ValueError("Model artifact differs from audit.")
            for cache in caches.values():
                if cache["model_scores"][model["name"]]["model_sha256"] != model["sha256"]:
                    raise ValueError("Model has changed since candidate scoring.")
            for threshold in thresholds(model["threshold"]):
                name = f"{model['name']}-{profile}-t{threshold:.17g}"
                variants.append(write_variant(name, profile, model, threshold, caches, output))
    report = {
        "schema_version": 1, "family": "tabular", "variants": variants,
        "failures": failures, "exclusions": [model for model in models if not model["compatible"]],
        "model_audit": relative(output / "model-audit.json"),
        "training_report": relative(output / "training/report.json"),
        "source_sha256": source_hashes(), "driver_sha256": digest(Path(__file__)),
        "input_sha256": inputs,
        "reference_sha256": {case: digest(REFERENCE_ROOT / "references" / f"{case}.json") for case in CASES},
        "environment": {
            "platform": platform.platform(), "python": platform.python_version(),
            "cpu": platform.processor(), "logical_cpu_count": os.cpu_count(),
            "threads": {name: os.environ.get(name) for name in THREAD_VARIABLES},
            "packages": {name: version(name) for name in (
                "numpy", "scipy", "SimpleITK", "scikit-image", "psutil",
            )},
        },
        "runtime_method": "Input loading + detection + individual model load/extraction/scoring + filtering. "
                          "Inference excludes evaluation and artifact writing. Peak RSS is the conservative "
                          "whole case/profile process sample at 10 ms, shared across its model variants.",
        "limitations": [
            "Local one-to-one 2/3/5 mm comparisons, not an official weighted challenge score.",
            "Judge-approved AI-assisted draft references may omit branches; only measured radii are scored.",
            "Legacy real-review models are ineligible retrospective baselines.",
            "Fresh refit uses only disjoint historical AI pseudo-labels, never expert negatives.",
            "No probabilities are claimed clinically calibrated; no calibration was fitted.",
            "Synthetic-only weights still inherit a detector developed on supplied images.",
            "Review-union includes proposals with relaxed origin eligibility; origin size is not seed radius.",
            "Linux CPU measurements do not establish Windows timing or clinical generalization.",
        ],
        "reproduction": [
            "OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "
            "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4 .venv313/bin/python final_eval_tabular.py run",
            ".venv313/bin/python final_eval_tabular.py report",
        ],
    }
    write_json(output / "report.json", report)
    return report


def rank_variant(variant: dict, held_out: str | None = None) -> tuple:
    rows = [row for row in variant["scores"]["3"]["cases"] if row["case_id"] != held_out]
    summary = summarize(rows)
    runtime = sum(row["runtime_s"] for case, row in variant["runtime"].items() if case != held_out)
    return (-summary["f1"], summary["count_mae"], runtime, variant["name"])


def paired_bootstrap(selected: dict, baselines: list[dict]) -> dict:
    indices = np.random.default_rng(20260913).integers(0, len(CASES), size=(10000, len(CASES)))
    report: dict = {
        "unit": "whole case", "seed": 20260913, "replicates": len(indices),
        "method": "Paired percentile 95% intervals; fixed held-out decisions, no reselection. "
                  "Five dependent development folds; intervals do not remove selection bias.",
        "tolerances": {},
    }
    for tolerance in ("2", "3", "5"):
        def samples(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
            counts = np.array([[row[key] for key in (
                "true_positives", "false_positives", "false_negatives",
            )] for row in rows])
            tp, fp, fn = counts[indices].sum(axis=1).T
            return 2 * tp / (2 * tp + fp + fn), np.abs(counts[:, 1] - counts[:, 2])[indices].mean(axis=1)
        f1, mae = samples(selected[tolerance])
        comparisons = {}
        for baseline in baselines:
            baseline_f1, baseline_mae = samples(baseline["scores"][tolerance]["cases"])
            comparisons[baseline["name"]] = {
                "f1_delta_95_percentile": np.quantile(f1 - baseline_f1, [0.025, 0.975]).tolist(),
                "count_mae_delta_95_percentile": np.quantile(mae - baseline_mae, [0.025, 0.975]).tolist(),
            }
        report["tolerances"][tolerance] = {
            "held_out_f1_95_percentile": np.quantile(f1, [0.025, 0.975]).tolist(),
            "held_out_count_mae_95_percentile": np.quantile(mae, [0.025, 0.975]).tolist(),
            "paired_comparisons": comparisons,
        }
    return report


def selection_summary(report: dict) -> dict:
    variants = [row for row in report["variants"] if row["eligible_for_selection"]]
    folds = []
    for held_out in CASES:
        chosen = min(variants, key=lambda variant: rank_variant(variant, held_out))
        folds.append({
            "held_out_case": held_out, "selected_variant": chosen["name"],
            "held_out_scores": {
                tolerance: next(row for row in value["cases"] if row["case_id"] == held_out)
                for tolerance, value in chosen["scores"].items()
            },
        })
    held_out_scores = {
        tolerance: [fold["held_out_scores"][tolerance] for fold in folds] for tolerance in ("2", "3", "5")
    }
    winner = min(variants, key=rank_variant)
    return {
        "scope": "Tabular-family-only leave-one-case-out configuration selection, not global family selection.",
        "eligible_variants": len(variants), "folds": folds,
        "selection_counts": dict(Counter(row["selected_variant"] for row in folds)),
        "held_out_summary": {
            tolerance: summarize(rows) for tolerance, rows in held_out_scores.items()
        },
        "all_five_development_winner": {
            "name": winner["name"], "scores": winner["scores"],
            "interpretation": "Descriptive reused-development maximum; not an independent held-out score.",
        },
        "uncertainty": paired_bootstrap(
            held_out_scores, [row for row in variants if row["configuration"]["model"] is None],
        ),
        "tie_break": "Pooled 3 mm F1, lower count MAE, lower measured four-case time, lexical name.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    relative(output)
    if args.command == "run":
        for name in THREAD_VARIABLES:
            if os.environ.get(name) not in ("1", "2", "3", "4"):
                parser.error(f"Set {name} to 1..4 before starting Python.")
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
        inputs, aliases = input_inventory()
        fit_excluded(output, inputs, aliases)
        models = audit_models(output)
        write_json(output / "run-provenance.json", {
            "starting_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "driver_sha256": digest(Path(__file__)), "input_sha256": inputs,
        })
        for case in CASES:
            for profile in PROFILES:
                print(f"Extracting {case} {profile}", flush=True)
                extract_case(case, profile, inputs[case], models, output)
    else:
        models = read_json(output / "model-audit.json")["models"]
    report = report_variants(output, models)
    write_json(output / "selection.json", selection_summary(report))
    print(json.dumps({
        "variants": len(report["variants"]), "failures": report["failures"],
        "excluded_artifacts": len(report["exclusions"]), "report": relative(output / "report.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
