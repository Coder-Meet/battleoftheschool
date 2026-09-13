"""Immutable synthetic E0/E1 research stages; never changes production proposals.

Run plan, generate, export, train-baselines, export --partition test, then
train-baselines --evaluate-test. Test predictions/labels are not computed until
the validation-selected models have been frozen. Outputs are research evidence,
not clinical accuracy. Completed stages cannot be overwritten; --resume verifies
and skips complete case transactions, never incomplete or modified artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter

try:
    import resource
except ImportError:
    resource = None  # type: ignore[assignment]

import numpy as np
import SimpleITK as sitk

from candidate_patches import EXTRA_FEATURE_NAMES, extract_candidate_data, preprocessing_metadata
from detector import DetectorConfig, detect, detect_pool
from evaluate import evaluate_case, summarize_cases
from learning import CandidateModel, FEATURE_NAMES, features
from nifti_io import read_nifti
from research_validation import paired_bootstrap
from stress import FAMILIES, generate_case, write_case
from synthetic_reviews import label_candidates
from tabular_learning import TreeModel

PARTITIONS = ("train", "validation", "test")
SOURCE_FILES = (
    "research_corpus.py", "stress.py", "detector.py", "candidate_patches.py",
    "synthetic_reviews.py", "evaluate.py", "learning.py", "nifti_io.py",
    "research_validation.py", "score_references.py", "tabular_learning.py", "train_trees.py",
    "requirements.txt", "requirements-trees.txt",
)
VERSIONS = ("numpy", "scipy", "SimpleITK", "scikit-image", "scikit-learn", "joblib", "threadpoolctl")
FORBIDDEN_SEEDS = {4001, 731927, 582743, 904117, 864203, 557891}
LIMITATIONS = [
    "Complete analytic synthetic references, not human anatomy or clinical accuracy.",
    "A filter cannot recover unproposed references; empty cases remain in every denominator.",
    "Ambiguous unmatched proposals within 5 mm are excluded only from candidate training.",
    "All 25 supplied real cases have prior inspection/pseudo-training exposure; new expert labels may overlap.",
    "No new patient labels, no clinical test and no target-Windows timing.",
    "Additional-paper variants are developed separately on exposed regression seeds, outside this cohort.",
]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def seal(path: Path) -> None:
    with path.with_name(path.name + ".sha256").open("x", encoding="ascii") as stream:
        stream.write(digest(path) + "\n")


def verify(path: Path) -> None:
    expected = path.with_name(path.name + ".sha256").read_text(encoding="ascii").strip()
    if digest(path) != expected:
        raise ValueError(f"Artifact integrity drift: {path}")


def save(path: Path, value: dict) -> None:
    write_json(path, value)
    seal(path)


def source_hashes() -> dict[str, str]:
    return {name: digest(Path(__file__).with_name(name)) for name in SOURCE_FILES}


def versions() -> dict[str, str]:
    result = {}
    for name in VERSIONS:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def new_seeds() -> list[int]:
    return [
        int.from_bytes(hashlib.sha256(f"branchseed-research-cohort-v1:{i}".encode()).digest()[:4], "big")
        for i in range(15)
    ]


def case_records(seeds: list[int]) -> list[dict]:
    if len(seeds) != 15 or len(set(seeds)) != 15 or any(
        type(seed) is not int or seed < 0 or seed in FORBIDDEN_SEEDS for seed in seeds
    ):
        raise ValueError("Require 15 distinct fresh nonnegative seed groups.")
    return [
        {
            "case_id": f"stress_{index:02d}_{family}_{seed}", "seed": seed,
            "family": family, "group_id": f"synthetic:{seed}",
            "partition": "train" if position < 10 else "validation" if position < 13 else "test",
        }
        for position, seed in enumerate(seeds) for index, family in enumerate(FAMILIES)
    ]


def validate_plan(plan: dict) -> None:
    if plan.get("schema_version") != 1 or plan.get("cases") != case_records(plan["seeds"]):
        raise ValueError("Plan case/seed/partition integrity drift.")
    if plan["source_sha256"] != source_hashes() or plan["versions"] != versions():
        raise ValueError("Source/dependency integrity drift; use the frozen source/environment.")
    if plan["python"] != platform.python_version():
        raise ValueError("Python version drift.")
    if plan["strict_config"] != asdict(DetectorConfig()) or plan["pool_config"] != asdict(DetectorConfig.review()):
        raise ValueError("Detector configuration drift.")
    if plan["preprocessing"] != preprocessing_metadata():
        raise ValueError("Physical preprocessing drift.")


def load_plan(root: Path) -> dict:
    verify(root / "plan.json")
    plan = read_json(root / "plan.json")
    validate_plan(plan)
    verify(root / "split.json")
    expected = {part: [r["case_id"] for r in plan["cases"] if r["partition"] == part] for part in PARTITIONS}
    if read_json(root / "split.json") != expected:
        raise ValueError("Split drift or group leakage.")
    return plan


def make_plan(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=False)
    seeds = new_seeds()
    plan: dict = {
        "schema_version": 1, "source": "analytic_synthetic_geometry", "seeds": seeds,
        "seed_derivation": "first four SHA256 bytes, big-endian, branchseed-research-cohort-v1:<0..14>",
        "cases": case_records(seeds), "source_sha256": source_hashes(), "versions": versions(),
        "python": platform.python_version(), "strict_config": asdict(DetectorConfig()),
        "pool_config": asdict(DetectorConfig.review()), "preprocessing": preprocessing_metadata(),
        "matching_tolerance_mm": 3.0, "negative_exclusion_mm": 5.0,
        "experiments": [
            {"name": f"{algorithm}-{feature_set}", "algorithm": algorithm, "features": feature_set}
            for algorithm in ("gradient_boosting", "random_forest") for feature_set in ("base", "extended")
        ],
        "training": {
            "source_weights": {"analytic": 1.0, "expert": 1.0, "pseudo": 0.0},
            "minimum_validation_recall": 1.0, "minimum_training_recall": 1.0,
            "calibration": "none; no separately adequate calibration cohort assumed",
            "broader_training_profile": None,
            "few_negatives_policy": "Report actual counts; train-only class weights; no invented negatives.",
            "legacy_ablation": "not run; synthetic-only first, no new real labels",
        },
        "selection": {
            "partition": "validation", "maximum_induced_misses": 0,
            "minimum_fp_reduction": 1, "minimum_fp_reduction_fraction": 0.1,
            "rank": "fewest full-pool FN, then FP, then preregistered model order",
            "cnn_gate": "Remove >=1 and >=10% pool FP, no pool/strict induced misses, and improve on logistic "
                        "without more FN, on validation and test.",
            "test_policy": "No test predictions or labels until all models and selection frozen; never retune.",
        },
        "bootstrap": {"samples": 2000, "seed": 42, "unit": "synthetic seed group"},
        "maximum_patch_partition_bytes": 512 * 1024**2,
        "limitations": LIMITATIONS,
    }
    validate_plan(plan)
    save(root / "plan.json", plan)
    save(root / "split.json", {
        part: [r["case_id"] for r in plan["cases"] if r["partition"] == part] for part in PARTITIONS
    })
    source = root / "source"
    source.mkdir()
    for name in SOURCE_FILES:
        shutil.copy2(Path(__file__).with_name(name), source / name)
    return plan


def thread_limit(threads: int) -> None:
    if not 1 <= threads <= 4:
        raise ValueError("Threads must be between one and four.")
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
        os.environ[variable] = str(threads)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(threads)


def peak_mib() -> float | None:
    if sys.platform != "linux" or resource is None:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def complete_directory(directory: Path, receipt_name: str) -> dict:
    verify(directory / receipt_name)
    receipt = read_json(directory / receipt_name)
    for name, expected in receipt["files"].items():
        path = directory / name
        if path.parent != directory or digest(path) != expected:
            raise ValueError(f"Case integrity drift: {path}")
    return receipt


def generate_one(job: tuple[str, dict, int, bool]) -> str:
    root_text, record, threads, resume = job
    thread_limit(threads)
    root = Path(root_text)
    load_plan(root)
    directory = root / "cases" / record["case_id"]
    if directory.exists():
        if not resume:
            raise FileExistsError(directory)
        complete_directory(directory, "input.json")
        return record["case_id"]
    start = perf_counter()
    case = generate_case(record["family"], record["seed"])
    if case.reference["case_id"] != record["case_id"]:
        raise ValueError("Generator case identity drift.")
    entry = write_case(case, directory)
    save(directory / "input.json", {
        **entry, **record, "generation_s": perf_counter() - start,
        "process_peak_mib": peak_mib(), "threads": threads,
    })
    return record["case_id"]


def run_jobs(function, jobs: list[tuple], workers: int) -> None:
    if workers not in (1, 2, 4):
        raise ValueError("Use one, two or four workers within the four-core budget.")
    thread_limit(4 // workers)
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1,
    ) as pool:
        for index, case_id in enumerate(pool.map(function, jobs), 1):
            print(f"{index}/{len(jobs)} {case_id}", flush=True)


def generate(root: Path, workers: int = 2, resume: bool = False) -> None:
    plan = load_plan(root)
    if (root / "manifest.json").exists():
        raise FileExistsError("Dataset manifest already exists.")
    (root / "cases").mkdir(exist_ok=True)
    run_jobs(generate_one, [(str(root), record, 4 // workers, resume) for record in plan["cases"]], workers)
    entries = [complete_directory(root / "cases" / r["case_id"], "input.json") for r in plan["cases"]]
    save(root / "manifest.json", {
        "schema_version": 1, "source": plan["source"], "plan_sha256": digest(root / "plan.json"),
        "source_sha256": plan["source_sha256"], "cases": entries,
    })


def validate_dataset(root: Path) -> dict:
    plan = load_plan(root)
    verify(root / "manifest.json")
    manifest = read_json(root / "manifest.json")
    if manifest["plan_sha256"] != digest(root / "plan.json"):
        raise ValueError("Dataset plan drift.")
    if [entry["case_id"] for entry in manifest["cases"]] != [entry["case_id"] for entry in plan["cases"]]:
        raise ValueError("Dataset case identity drift.")
    for entry in manifest["cases"]:
        receipt = complete_directory(root / "cases" / entry["case_id"], "input.json")
        if receipt != entry:
            raise ValueError("Dataset receipt drift.")
    return plan


def export_one(job: tuple[str, dict, int, bool]) -> str:
    root_text, record, threads, resume = job
    thread_limit(threads)
    root = Path(root_text)
    plan = load_plan(root)
    if record["partition"] == "test":
        verify_freeze(root)
    directory = root / "exports" / record["case_id"]
    if directory.exists():
        if not resume:
            raise FileExistsError(directory)
        complete_directory(directory, "receipt.json")
        return record["case_id"]
    source = root / "cases" / record["case_id"]
    receipt = complete_directory(source, "input.json")
    directory.mkdir()
    start = perf_counter()
    image = read_nifti(str(source / "orig.nii.gz"))
    mask = read_nifti(str(source / "mask.nii.gz"))
    reference = read_json(source / "reference.json")
    strict = detect(image, mask)
    pool = detect_pool(image, mask)
    extract_start = perf_counter()
    extracted = extract_candidate_data(image, mask, pool)
    extraction_s = perf_counter() - extract_start
    rows, metrics = label_candidates(
        pool, reference, plan["matching_tolerance_mm"], plan["negative_exclusion_mm"],
    )
    all_rows = []
    for index, branch in enumerate(pool.branches):
        vector = features(branch)
        all_rows.append({
            "case_id": record["case_id"], "instance_id": branch.instance_id,
            "features": vector, "extra_features": dict(zip(EXTRA_FEATURE_NAMES, extracted.extra_features[index].tolist())),
            "fingerprint": json.dumps([
                branch.ostium_xyz_mm, branch.seed_xyz_mm, branch.direction_xyz, branch.radius_mm, vector,
            ]),
            "group_id": record["group_id"], "family": record["family"],
            "local_patch_index": index, "input_sha256": receipt["files"],
            "detector_sha256": plan["source_sha256"]["detector.py"],
            "extractor_sha256": plan["source_sha256"]["candidate_patches.py"],
        })
    indexed = {r["instance_id"]: r for r in all_rows}
    rows = [{**row, **indexed[row["instance_id"]]} for row in rows]
    write_json(directory / "candidates.json", {
        **record, "records": rows, "all_candidates": all_rows,
        "pool_metrics": metrics, "strict_metrics": evaluate_case(strict.prediction(record["case_id"]), reference, 3),
        "extraction_s": extraction_s, "wall_s": perf_counter() - start,
        "process_peak_mib": peak_mib(), "threads": threads,
    })
    for name, result in (("strict", strict), ("pool", pool)):
        write_json(directory / f"{name}.json", result.prediction(record["case_id"]))
        write_json(directory / f"{name}-diagnostics.json", result.diagnostics())
    with (directory / "patches.npz").open("xb") as stream:
        np.savez_compressed(stream, patches=extracted.patches)
    save(directory / "receipt.json", {
        **record, "input_receipt_sha256": digest(source / "input.json"),
        "files": {path.name: digest(path) for path in sorted(directory.iterdir())},
    })
    return record["case_id"]


def assemble_partition(root: Path, part: str, plan: dict) -> None:
    records: list[dict] = []
    cases: list[dict] = []
    entries = [r for r in plan["cases"] if r["partition"] == part]
    for entry in entries:
        directory = root / "exports" / entry["case_id"]
        complete_directory(directory, "receipt.json")
        case = read_json(directory / "candidates.json")
        cases.append({k: v for k, v in case.items() if k not in ("records", "all_candidates")})
        for row in case["records"]:
            records.append({**row, "patch_index": len(records)})
    shape = (len(records), 12, 32, 32)
    if np.prod(shape) * 4 > plan["maximum_patch_partition_bytes"]:
        raise ValueError("Partition patch cache exceeds preregistered memory cap.")
    patches = np.empty(shape, dtype=np.float32)
    offset = 0
    for entry in entries:
        directory = root / "exports" / entry["case_id"]
        case = read_json(directory / "candidates.json")
        with np.load(directory / "patches.npz", allow_pickle=False) as archive:
            array = archive["patches"]
            validate_patches(array, len(case["all_candidates"]))
            for row in case["records"]:
                patches[offset] = array[row["local_patch_index"]]
                offset += 1
    path = root / f"patches-{part}.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, patches=patches)
    seal(path)
    save(root / f"patches-{part}.json", {
        "schema_version": 1, "scope": "candidate_reviews_only", "partition": part,
        "feature_names": FEATURE_NAMES, "extra_feature_names": EXTRA_FEATURE_NAMES,
        "preprocessing": plan["preprocessing"], "patches_sha256": digest(path),
        "manifest_sha256": digest(root / "manifest.json"), "source_sha256": plan["source_sha256"],
        "cases": cases, "records": records,
    })


def validate_patches(array: np.ndarray, count: int) -> None:
    if array.dtype != np.float32 or array.shape != (count, 12, 32, 32) or not np.isfinite(array).all():
        raise ValueError("Invalid physical patch cache shape, dtype or values.")


def assemble_reviews(root: Path, parts: tuple[str, ...], filename: str, plan: dict) -> None:
    records = []
    for part in parts:
        verify(root / f"patches-{part}.json")
        records.extend(read_json(root / f"patches-{part}.json")["records"])
    save(root / filename, {
        "schema_version": 1, "scope": "candidate_reviews_only", "feature_names": FEATURE_NAMES,
        "extra_feature_names": EXTRA_FEATURE_NAMES, "label_source": "analytic_synthetic_geometry",
        "profile": "pool", "source_sha256": plan["source_sha256"],
        "manifest_sha256": digest(root / "manifest.json"), "plan_sha256": digest(root / "plan.json"),
        "cases": plan["cases"], "records": records, "included_partitions": list(parts),
        "limitations": LIMITATIONS,
    })


def export(root: Path, partition: str = "development", workers: int = 2, resume: bool = False) -> None:
    plan = validate_dataset(root)
    parts = ("train", "validation") if partition == "development" else ("test",)
    if partition == "test":
        verify_freeze(root)
    for part in parts:
        if (root / f"patches-{part}.json").exists() or (root / f"patches-{part}.npz").exists():
            raise FileExistsError("Partition already exported; choose a new experiment.")
    (root / "exports").mkdir(exist_ok=True)
    entries = [record for record in plan["cases"] if record["partition"] in parts]
    run_jobs(export_one, [(str(root), record, 4 // workers, resume) for record in entries], workers)
    for part in parts:
        assemble_partition(root, part, plan)
    if partition == "development":
        assemble_reviews(root, parts, "reviews-development.json", plan)
    else:
        assemble_reviews(root, PARTITIONS, "reviews.json", plan)


def verify_freeze(root: Path) -> dict:
    verify(root / "models" / "freeze.json")
    freeze = read_json(root / "models" / "freeze.json")
    if freeze["plan_sha256"] != digest(root / "plan.json"):
        raise ValueError("Frozen model plan drift.")
    for name, expected in freeze["files"].items():
        if digest(root / "models" / name) != expected:
            raise ValueError(f"Frozen model integrity drift: {name}")
    for name, expected in freeze["inputs"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Frozen training input drift: {name}")
    return freeze


def filter_prediction(prediction: dict, rows: list[dict], model: TreeModel | CandidateModel) -> dict:
    if [row["instance_id"] for row in rows] != [branch["instance_id"] for branch in prediction["daughters"]]:
        raise ValueError("Candidate feature/prediction identity mismatch.")
    names = model.feature_names if isinstance(model, TreeModel) else FEATURE_NAMES
    vectors = []
    for row in rows:
        vector = list(row["features"])
        if len(names) > len(FEATURE_NAMES):
            extra = row["extra_features"]
            if set(extra) != set(EXTRA_FEATURE_NAMES):
                raise ValueError("Missing extended features; imputation prohibited.")
            vector.extend(extra[name] for name in EXTRA_FEATURE_NAMES)
        vectors.append(vector)
    matrix = np.asarray(vectors, dtype=float).reshape(len(rows), len(names))
    keep = model.scores(matrix) >= model.threshold
    return {**prediction, "daughters": [branch for branch, retained in zip(prediction["daughters"], keep) if retained]}


def comparisons(baseline: list[dict], variant: list[dict], groups: dict[str, str]) -> dict:
    base = {r["case_id"]: r for r in baseline}
    other = {r["case_id"]: r for r in variant}
    if set(base) != set(other):
        raise ValueError("Comparison must retain every case.")
    losses, recovered = {}, {}
    for case_id in base:
        before = {m["reference_id"] for m in base[case_id]["matches"]}
        after = {m["reference_id"] for m in other[case_id]["matches"]}
        losses[case_id] = sorted(before - after)
        recovered[case_id] = sorted(after - before)
    base_fp = sum(r["false_positives"] for r in baseline)
    fp_reduction = base_fp - sum(r["false_positives"] for r in variant)
    return {
        "induced_misses": sum(map(len, losses.values())), "lost_reference_ids": losses,
        "recovered_reference_ids": recovered, "fp_reduction": fp_reduction,
        "fp_reduction_fraction": fp_reduction / base_fp if base_fp else None,
        "paired_bootstrap": paired_bootstrap(base, other, groups),
    }


def evaluate_partition(root: Path, part: str, models: dict[str, TreeModel | CandidateModel]) -> dict:
    plan = load_plan(root)
    if part == "test":
        verify_freeze(root)
    entries = [r for r in plan["cases"] if r["partition"] == part]
    groups = {r["case_id"]: r["group_id"] for r in entries}
    reports: dict[str, list[dict]] = {name: [] for name in ("strict", "pool", *models)}
    predictions: dict[str, dict] = {}
    for record in entries:
        case_id = record["case_id"]
        directory = root / "exports" / case_id
        complete_directory(directory, "receipt.json")
        reference = read_json(root / "cases" / case_id / "reference.json")
        candidates = read_json(directory / "candidates.json")
        pool = read_json(directory / "pool.json")
        predictions[case_id] = {
            "strict": read_json(directory / "strict.json"), "pool": pool,
            **{name: filter_prediction(pool, candidates["all_candidates"], model) for name, model in models.items()},
        }
        for name, prediction in predictions[case_id].items():
            reports[name].append({
                **evaluate_case(prediction, reference, 3), "family": record["family"], "group_id": record["group_id"],
                "sensitivity": {
                    str(tolerance): evaluate_case(prediction, reference, tolerance) for tolerance in (2, 5)
                },
            })
    return {
        "partition": part, "scope": "end_to_end_complete_analytic_references", "cases": len(entries),
        "runs": {
            name: {
                "summary": summarize_cases(results), "cases": results,
                "families": {family: summarize_cases([r for r in results if r["family"] == family]) for family in FAMILIES},
                "versus_pool": comparisons(reports["pool"], results, groups),
                "versus_strict": comparisons(reports["strict"], results, groups),
            }
            for name, results in reports.items()
        },
        "predictions": predictions,
    }


def gate(run: dict, logistic: dict) -> bool:
    comparison = run["versus_pool"]
    fraction = comparison["fp_reduction_fraction"]
    return (
        comparison["induced_misses"] == 0 and comparison["fp_reduction"] >= 1
        and fraction is not None and fraction >= 0.1
        and run["versus_strict"]["induced_misses"] == 0
        and run["summary"]["false_negatives"] <= logistic["summary"]["false_negatives"]
        and run["summary"]["false_positives"] < logistic["summary"]["false_positives"]
    )


def load_models(root: Path, plan: dict) -> dict[str, TreeModel | CandidateModel]:
    models: dict[str, TreeModel | CandidateModel] = {
        experiment["name"]: TreeModel.load(root / "models" / f"{experiment['name']}.json")
        for experiment in plan["experiments"]
    }
    models["logistic"] = CandidateModel.load(root / "models" / "logistic.json")
    return models


def train_baselines(root: Path, evaluate_test: bool = False) -> None:
    plan = validate_dataset(root)
    directory = root / "models"
    if evaluate_test:
        freeze = verify_freeze(root)
        if (root / "summary.json").exists():
            raise FileExistsError("Test has already been evaluated.")
        verify(root / "reviews.json")
        report = evaluate_partition(root, "test", load_models(root, plan))
        save(root / "test-report.json", report)
        validation = read_json(directory / "validation-report.json")
        selected = freeze["selected_model"]
        passed = selected is not None and gate(report["runs"][selected], report["runs"]["logistic"])
        training = read_json(root / "development-report.json")
        save(root / "summary.json", {
            "schema_version": 1, "plan_sha256": digest(root / "plan.json"),
            "manifest_sha256": digest(root / "manifest.json"), "freeze_sha256": digest(directory / "freeze.json"),
            "source_sha256": plan["source_sha256"], "selected_model": selected,
            "cnn_stop_go": "go_research_only" if passed else "stop_E1_failed",
            "clinical_accuracy": False, "production_changed": False, "limitations": LIMITATIONS,
            "candidate_counts": {
                part: dict(Counter(row["label"] for row in read_json(root / f"patches-{part}.json")["records"]))
                for part in PARTITIONS
            },
            "unlabelled_ambiguous_counts": {
                part: sum(len(c["pool_metrics"]["excluded_ambiguous_candidate_ids"])
                          for c in read_json(root / f"patches-{part}.json")["cases"])
                for part in PARTITIONS
            },
            "validation": compact_report(validation), "test": compact_report(report),
            "E0_all_210": {
                name: {
                    "summary": summarize_cases([
                        case for partition_report in (training, validation, report)
                        for case in partition_report["runs"][name]["cases"]
                    ]),
                    "families": {
                        family: summarize_cases([
                            case for partition_report in (training, validation, report)
                            for case in partition_report["runs"][name]["cases"] if case["family"] == family
                        ]) for family in FAMILIES
                    },
                } for name in ("strict", "pool")
            },
            "resources": resource_summary(root, plan),
            "frozen_files": freeze["files"], "validation_gate": freeze["validation_gate"],
            "test_gate_by_model": {
                e["name"]: gate(report["runs"][e["name"]], report["runs"]["logistic"]) for e in plan["experiments"]
            },
            "selection_unchanged_after_test": True,
        })
        return
    if any((root / "exports" / r["case_id"]).exists() for r in plan["cases"] if r["partition"] == "test"):
        raise ValueError("Test exports already exist; refusing post-test training.")
    verify(root / "reviews-development.json")
    directory.mkdir(exist_ok=False)
    for experiment in plan["experiments"]:
        name = experiment["name"]
        subprocess.run([
            sys.executable, str(Path(__file__).with_name("train_trees.py")),
            "--reviews", str(root / "reviews-development.json"), "--split", str(root / "split.json"),
            "--algorithm", experiment["algorithm"], "--features", experiment["features"],
            "--pseudo-weight", "0", "--minimum-recall", "1", "--minimum-training-recall", "1",
            "--model", str(directory / f"{name}.json"), "--report", str(directory / f"{name}-training.json"),
        ], check=True)
    first_report = read_json(directory / f"{plan['experiments'][0]['name']}-training.json")
    logistic = first_report["logistic_comparator"]
    CandidateModel(
        logistic["mean"], logistic["scale"], logistic["weights"], logistic["bias"], logistic["threshold"],
        read_json(root / "split.json"),
    ).save(directory / "logistic.json")
    models = load_models(root, plan)
    report = evaluate_partition(root, "validation", models)
    save(directory / "validation-report.json", report)
    eligible = [
        e["name"] for e in plan["experiments"] if gate(report["runs"][e["name"]], report["runs"]["logistic"])
    ]
    selected = min(eligible, key=lambda name: (
        report["runs"][name]["summary"]["false_negatives"], report["runs"][name]["summary"]["false_positives"],
        [e["name"] for e in plan["experiments"]].index(name),
    )) if eligible else None
    save(directory / "freeze.json", {
        "schema_version": 1, "plan_sha256": digest(root / "plan.json"), "selected_model": selected,
        "validation_gate": {
            e["name"]: gate(report["runs"][e["name"]], report["runs"]["logistic"]) for e in plan["experiments"]
        },
        "test_status": "no_test_proposals_or_labels_computed",
        "files": {p.name: digest(p) for p in sorted(directory.iterdir())},
        "inputs": {name: digest(root / name) for name in (
            "reviews-development.json", "split.json", "patches-train.json", "patches-validation.json",
            "patches-train.npz", "patches-validation.npz", "manifest.json",
        )},
    })
    save(root / "development-report.json", evaluate_partition(root, "train", models))


def compact_report(report: dict) -> dict:
    return {
        name: {
            "summary": run["summary"], "families": run["families"],
            **{
                comparison: {key: value for key, value in run[comparison].items()
                             if key not in ("lost_reference_ids", "recovered_reference_ids")}
                for comparison in ("versus_pool", "versus_strict")
            },
        }
        for name, run in report["runs"].items()
    }


def resource_summary(root: Path, plan: dict) -> dict:
    cases = [read_json(root / "exports" / entry["case_id"] / "candidates.json") for entry in plan["cases"]]
    result: dict = {
        "platform": platform.platform(), "python": platform.python_version(),
        "scope": "Linux research measurements; worker RSS excludes parent and sibling processes",
        "target_Windows_verified": False,
    }
    for key in ("extraction_s", "wall_s", "process_peak_mib"):
        numbers = [case[key] for case in cases if case[key] is not None]
        result[key] = {
            "measured_cases": len(numbers), "mean": float(np.mean(numbers)) if numbers else None,
            "maximum": max(numbers) if numbers else None,
        }
    for profile in ("strict", "pool"):
        timings = [
            read_json(root / "exports" / entry["case_id"] / f"{profile}-diagnostics.json")["timings"]["total_s"]
            for entry in plan["cases"]
        ]
        result[f"{profile}_s"] = {"mean": float(np.mean(timings)), "maximum": max(timings)}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "generate", "export", "train-baselines"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--partition", choices=("development", "test"), default="development")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    thread_limit(4)
    root = args.root.resolve()
    try:
        if args.stage == "plan":
            make_plan(root)
        elif args.stage == "generate":
            generate(root, args.workers, args.resume)
        elif args.stage == "export":
            export(root, args.partition, args.workers, args.resume)
        else:
            train_baselines(root, args.evaluate_test)
    except (OSError, ValueError, KeyError, RuntimeError, TypeError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
