"""Frozen CNN comparison with checked source replay and reusable ordered probabilities."""

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import SimpleITK as sitk

from detector import DetectorConfig, detect, detect_pool, validate_geometry
from final_evaluation import (
    CASES, ROOT, THRESHOLDS, case_paths, digest, filtered, load_reference,
    read_json, score_variant, summarize, write_json,
)
from nifti_io import read_nifti
from patch_inference import BlendModel, PatchModel, blend_scores, score_batch
from research_resources import measure, network_probe
from research_run import score_detection, source_hashes, validate_tree_source
from tabular_learning import TreeModel, json_sha256

CURRENT = ROOT / "labels/research/current-source-v1"
OLD = ROOT / "labels/research/patch-cnn-retrospective-v1"
OUTPUT = ROOT / "labels/final-eval/cnn"
SNAPSHOT = ROOT / "outputs/final-cnn-frozen"
SNAPSHOT_COMMIT = "3395de51c91fe63882005a45e18f3fee54fce8ca"
BRIDGE_COMMIT = "2a825d89b66e7d5c61bea7f497be08fbc22cf526"
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
MODELS = {
    "current": CURRENT / "cnn-synthetic/model.onnx",
    "legacy-synthetic": OLD / "synthetic/model.onnx",
    "legacy-mixed": OLD / "mixed/model.onnx",
}
TREE = CURRENT / "models/gradient_boosting-base.json"
BLEND = CURRENT / "cnn-synthetic/blend.json"


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def require_source(model: PatchModel, sources: dict) -> None:
    expected = {name: sources[name] for name in ("detector.py", "candidate_patches.py")}
    if model.contract["source_sha256"] != expected:
        raise ValueError("ONNX extraction source differs; use its frozen source or re-extract/retrain.")


def inventory() -> dict:
    sources = source_hashes()
    rows = []
    found = [
        ROOT / name for name in subprocess.check_output(
            ["git", "ls-files", "-z", "--", "*.onnx"], cwd=ROOT, text=True,
        ).split("\0") if name
    ]
    for path in found:
        model = PatchModel.load(path)
        report_path = path.with_name("training-report.json")
        training = read_json(report_path)
        records = training["train_records"]
        expected = model.contract["source_sha256"]
        if any(
            row["detector_sha256"] != expected["detector.py"]
            or row["extractor_sha256"] != expected["candidate_patches.py"]
            for row in records
        ):
            raise ValueError(f"Training-record/source mismatch: {path}")
        mismatch = ""
        try:
            require_source(model, sources)
        except ValueError as error:
            mismatch = str(error)
        actual_fit = sorted({row["case_id"] for row in records})
        rows.append({
            "path": relative(path), "sha256": digest(path),
            "sidecar_sha256": digest(path.with_suffix(".json")),
            "training_report_sha256": digest(report_path),
            "threshold": model.threshold, "contract": model.contract,
            "training_config": training["config"], "selected_epoch": training["selected_epoch"],
            "training_records": len(records), "fit_cases": actual_fit,
            "training_labellers": sorted({row["labeller"] for row in records}),
            "released_cases_in_fit": sorted(set(CASES) & set(actual_fit)),
            "released_cases_in_declared_split": {
                key: sorted(set(CASES) & set(values)) for key, values in model.split.items()
            },
            "current_source_compatible": not mismatch, "current_source_rejection": mismatch,
            "frozen_extraction_commit": SNAPSHOT_COMMIT if mismatch else BRIDGE_COMMIT,
            "provider": model.session.get_providers(),
        })
    manifest = read_json(CURRENT / "manifest.json")
    model = PatchModel.load(MODELS["current"])
    if digest(CURRENT / "manifest.json") != model.contract["manifest_sha256"]:
        raise ValueError("Current CNN training-manifest mismatch.")
    if manifest["source"] != "analytic_synthetic_geometry":
        raise ValueError("Current training manifest is not synthetic-only.")
    if set(CASES) & set(model.split["train"] + model.split["validation"] + model.split["test"]):
        raise ValueError("Current CNN split overlaps released cases.")
    validate_tree_source(TreeModel.load(TREE), sources)
    frozen_blend = BlendModel.load(BLEND)
    if frozen_blend.tree_sha256 != digest(TREE) or frozen_blend.cnn_sha256 != digest(MODELS["current"]):
        raise ValueError("Frozen blend/model mismatch.")
    old_tree = ROOT / "labels/research/cohort-v1/models/gradient_boosting-base.json"
    try:
        validate_tree_source(TreeModel.load(old_tree), sources)
    except ValueError as error:
        old_tree_error = str(error)
    else:
        raise ValueError("Legacy tree unexpectedly acquired a current extraction contract.")
    return {
        "onnx_artifacts": rows, "onnx_count": len(rows), "current_source_sha256": sources,
        "training_manifest_sha256": digest(CURRENT / "manifest.json"),
        "training_manifest_content_sha256": json_sha256(manifest),
        "frozen_replay": {
            "extraction_commit": SNAPSHOT_COMMIT, "bridge_commit": BRIDGE_COMMIT,
            "directory": relative(SNAPSHOT),
            "extraction_files": ["detector.py", "candidate_patches.py", "learning.py", "nifti_io.py"],
            "bridge_files": ["research_run.py", "patch_inference.py", "tabular_learning.py"],
            "source_sha256": {
                path.name: digest(path) for path in sorted(SNAPSHOT.glob("*.py"))
            },
        },
        "source_evidence": {
            "current_training_report": relative(MODELS["current"].with_name("training-report.json")),
            "current_reproduction_receipt": relative(CURRENT / "reproduction.json"),
            "current_reproduction_sha256": digest(CURRENT / "reproduction.json"),
            "current_manifest": relative(CURRENT / "manifest.json"),
            "current_and_legacy_synthetic_graph_identical": digest(MODELS["current"]) == digest(
                MODELS["legacy-synthetic"]),
            "interpretation": "Equal graph bytes do not authorize sidecar substitution. Current training "
                              "records and fresh-generation receipts bind the current extraction source.",
        },
        "exclusions": [{
            "artifacts": [relative(OLD / kind / "blend.json") for kind in ("synthetic", "mixed")],
            "tree_path": relative(old_tree), "tree_sha256": digest(old_tree),
            "reason": old_tree_error,
            "retraining_path": "Regenerate source-hashed synthetic candidate records and physical patches with "
                               "research_corpus.py generate/export; run train-baselines and patch_learning.py "
                               "train on synthetic train/validation only, then export new paired sidecars and "
                               "validation-selected blends. Existing artifacts remain immutable. The current-source "
                               "synthetic family already provides this retrained comparator.",
        }],
        "limitations": [
            "Local 2/3/5 mm comparisons, not the official weighted score.",
            "Judge-approved references retain AI-assisted draft provenance and possible omissions.",
            "Only three reference radii are measured; unknown radii are masked by final_evaluation.",
            "All five images were historically inspected; synthetic-only fitting does not make them new patients.",
            "Legacy mixed CNN saw released images; all legacy runs are retrospective and excluded from selection.",
            "Legacy source predates the confirmed minimum origin-diameter policy.",
            "No model was trained or recalibrated on released candidates.",
            "Windows execution and organizer-machine performance are not established by Linux measurements.",
        ],
    }


def run_case(family: str, proposals: str, case_id: str, destination: Path) -> int:
    start = perf_counter()
    receipt: dict = {"status": "failed", "case_id": case_id, "family": family, "proposals": proposals}
    try:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
        model = PatchModel.load(MODELS[family], threads=4)
        require_source(model, source_hashes())
        paths = case_paths(case_id)
        hashes = {name: digest(path) for name, path in zip(("image", "aorta_mask"), paths)}
        verified = next(
            row for row in read_json(ROOT / "labels/organizer-v1/validation.json")["cases"]
            if row["case_id"] == case_id
        )
        if hashes != {"image": verified["image_sha256"], "aorta_mask": verified["mask_sha256"]}:
            raise ValueError("CT/mask hashes differ from the validated release.")
        hashed = perf_counter()
        image, mask = (read_nifti(str(path)) for path in paths)
        validate_geometry(image, mask)
        if list(image.GetSize()) != verified["size_xyz"] or any(
            not np.allclose(value, verified[key], rtol=0, atol=1e-6)
            for value, key in (
                (image.GetSpacing(), "spacing_xyz_mm"), (image.GetOrigin(), "origin_lps_mm"),
                (image.GetDirection(), "direction"),
            )
        ):
            raise ValueError("Physical geometry differs from the validated release.")
        loaded = perf_counter()
        result = detect(image, mask) if proposals == "strict" else detect_pool(image, mask)
        detected = perf_counter()
        receipt.update({
            "unfiltered_prediction": result.prediction(case_id),
            "branches": [asdict(branch) for branch in result.branches],
            "detector": result.diagnostics(),
            "strict_config": asdict(DetectorConfig()), "review_config": asdict(DetectorConfig.review()),
        })
        _, _, diagnostic = score_detection(
            image, mask, result, case_id, f"released:{case_id}", hashes,
            onnx_path=MODELS[family], tree_path=TREE if family == "current" else None,
            blend_path=BLEND if family == "current" else None, threads=4,
        )
        receipt.update(diagnostic)
        receipt.update({
            "status": "success", "hash_and_preflight_s": hashed - start, "input_load_s": loaded - hashed,
            "detection_s": detected - loaded, "worker_runtime_s": perf_counter() - start,
            "network_denied": network_probe(), "scorer_sha256": digest(Path(__file__)),
            "dependencies": {
                name: version(name) for name in ("numpy", "scipy", "SimpleITK", "scikit-image", "onnxruntime")
            },
        })
        write_json(destination, receipt)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        receipt.update(error=str(error), worker_runtime_s=perf_counter() - start)
        write_json(destination, receipt)
        return 1


def replay_values(receipt: dict, weight: float | None, frozen: bool = False) -> np.ndarray:
    if receipt["status"] != "success":
        raise ValueError("Failed candidate extraction cannot become an empty prediction.")
    records = receipt["records"]
    prediction = receipt["unfiltered_prediction"]
    if [row["instance_id"] for row in records] != [row["instance_id"] for row in prediction["daughters"]]:
        raise ValueError("Saved prediction and candidate ordering differ.")
    if len({row["instance_id"] for row in records}) != len(records):
        raise ValueError("Duplicate saved candidate identities.")
    for row, branch in zip(records, prediction["daughters"]):
        if row["case_id"] != prediction["case_id"]:
            raise ValueError("Saved candidate case differs from prediction.")
        fingerprint = read_fingerprint(row["fingerprint"])
        if fingerprint[:4] != [
            branch["ostium_xyz_mm"], branch["seed_xyz_mm"], branch["direction_xyz"], branch["radius_mm"],
        ] or fingerprint[4] != row["features"]:
            raise ValueError("Saved candidate geometry/features differ from fingerprint.")
    cnn = score_batch(
        np.asarray(receipt["component_scores"]["onnx"]), records, receipt["contract"],
        receipt["model_sha256"]["onnx"],
    )
    if weight is None:
        return cnn.scores
    tree = score_batch(
        np.asarray(receipt["component_scores"]["tree"]), records, receipt["contract"],
        receipt["model_sha256"]["tree"],
    )
    if frozen:
        blend = BlendModel.load(BLEND)
        if receipt["model_sha256"]["blend"] != digest(BLEND):
            raise ValueError("Saved frozen-blend hash differs.")
        return blend.scores(tree, cnn)
    return blend_scores(tree, cnn, weight).scores


def validate_saved(receipt: dict, family: str, proposals: str, case: str, model: PatchModel) -> None:
    if (receipt["family"], receipt["proposals"], receipt["case_id"]) != (family, proposals, case):
        raise ValueError("Saved candidate receipt has a different family, pool or case.")
    require_source(model, receipt["source_sha256"])
    if receipt["contract"]["source_sha256"] != model.contract["source_sha256"]:
        raise ValueError("Saved candidate extraction contract differs from the model.")
    if receipt["model_sha256"]["onnx"] != digest(MODELS[family]) or (
        receipt["model_sha256"]["onnx_sidecar"] != digest(MODELS[family].with_suffix(".json"))
    ):
        raise ValueError("Saved model or sidecar changed.")
    if family == "current" and receipt["model_sha256"]["tree"] != digest(TREE):
        raise ValueError("Saved tree model changed.")
    expected_manifest = json_sha256({
        "inputs": receipt["input_sha256"], "case_id": case, "group_id": f"released:{case}",
    })
    if receipt["contract"]["manifest_sha256"] != expected_manifest or any(
        row["input_sha256"] != receipt["input_sha256"] or row["group_id"] != f"released:{case}"
        for row in receipt["records"]
    ):
        raise ValueError("Saved input manifest differs from candidate identities.")


def read_fingerprint(value: str) -> list:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or len(parsed) != 5:
        raise ValueError("Malformed candidate fingerprint.")
    return parsed


def reference_losses(baseline: dict, variant: dict) -> dict:
    losses = {}
    for tolerance in ("2", "3", "5"):
        rows = []
        for before, after in zip(baseline[tolerance]["cases"], variant[tolerance]["cases"]):
            if before["case_id"] != after["case_id"]:
                raise ValueError("Loss comparisons must use identical case ordering.")
            original = {row["reference_id"] for row in before["matches"]}
            retained = {row["reference_id"] for row in after["matches"]}
            rows.append({
                "case_id": before["case_id"], "lost_reference_ids": sorted(original - retained),
                "newly_matched_reference_ids": sorted(retained - original),
                "true_positive_change": after["true_positives"] - before["true_positives"],
            })
        losses[tolerance] = rows
    return losses


def make_variant(
    name: str, receipts: dict[str, dict], resources: dict[str, dict], configuration: dict,
    output: Path, weight: float | None = None, threshold: float = 0.0,
    unfiltered: bool = False, frozen: bool = False,
) -> dict:
    if set(receipts) != set(CASES) or set(resources) != set(CASES):
        raise ValueError("All five cases and resource receipts are required.")
    predictions = {}
    baseline = {}
    timings = {}
    for case in CASES:
        receipt = receipts[case]
        values = replay_values(receipt, weight, frozen)
        if resources[case]["exit_code"] != 0:
            raise ValueError("Failed measurement cannot be scored.")
        baseline[case] = receipt["unfiltered_prediction"]
        predictions[case] = baseline[case] if unfiltered else filtered(baseline[case], values, threshold)
        write_json(output / name / f"{case}.json", predictions[case])
        runtime = resources[case]["cases"][case]
        timings[case] = {
            "runtime_s": runtime["runtime_s"], "peak_rss_mb": runtime["peak_rss_mb"],
            "input_load_s": receipt["input_load_s"], "detection_s": receipt["detection_s"],
            "model_load_s": receipt["model_load_s"], "extraction_s": receipt["extraction_s"],
            "scoring_s": receipt["scoring_s"],
        }
    current = configuration["family"] == "current"
    scores = score_variant(predictions)
    return {
        "name": name, "prediction_dir": relative(output / name), "configuration": {
            **configuration, "cnn_weight": weight, "threshold": threshold, "frozen_blend": frozen,
            "unfiltered": unfiltered,
            "source_sha256": receipts[CASES[0]]["source_sha256"],
            "model_sha256": receipts[CASES[0]]["model_sha256"],
            "detector_settings": {
                key: receipts[CASES[0]][key] for key in ("strict_config", "review_config")
            },
            "runtime_scope": "Measured full extraction worker including cold imports, both current models, "
                             "hash checks and diagnostics writing. Shared across replay thresholds/weights. "
                             "Unfiltered and weight endpoints conservatively include unused model costs.",
            "post_reference_algorithm": False,
            "eligibility_scope": "Synthetic-only frozen-family research selection; not deployment approval.",
            "origin_policy": (
                "Strict current 2-mm origin-diameter policy."
                if current and configuration["proposals"] == "strict"
                else "Research review override or historical source; final origin eligibility is not certified."
            ),
        },
        "eligible_for_selection": current,
        "ineligible_reason": "" if current else (
            "Historical source lacks confirmed origin eligibility; retrospective baseline."
            + (" Mixed CNN training included subject022/subject023." if configuration["family"] == "legacy-mixed" else "")
        ),
        "scores": scores, "runtime": timings,
        "true_reference_losses_vs_own_proposals": reference_losses(score_variant(baseline), scores),
    }


def replay(output: Path) -> dict:
    audit = read_json(output / "inventory.json")
    variants = []
    failures: list[dict] = []
    cnn_threshold = PatchModel.load(MODELS["current"]).threshold
    tree_threshold = TreeModel.load(TREE).threshold
    blend = BlendModel.load(BLEND)
    for family in MODELS:
        for proposals in ("strict", "review-union"):
            directory = output / "candidates" / family / proposals
            missing = [
                case for case in CASES
                if not (directory / f"{case}.json").exists()
                or not (directory / f"{case}.resources.json").exists()
            ]
            if missing:
                failures.append({
                    "family": family, "proposals": proposals, "missing_cases": missing,
                    "effect": "Incomplete extraction; entire family/pool matrix withheld until all five cases exist.",
                })
                continue
            receipts = {case: read_json(directory / f"{case}.json") for case in CASES}
            resources = {case: read_json(directory / f"{case}.resources.json") for case in CASES}
            failed = [case for case in CASES if receipts[case]["status"] != "success"
                      or resources[case]["exit_code"] != 0]
            if failed:
                failures.append({
                    "family": family, "proposals": proposals,
                    "cases": {case: receipts[case].get("error", "worker failed") for case in failed},
                    "effect": "Entire family/proposal matrix withheld; no synthetic empty predictions.",
                })
                continue
            model = PatchModel.load(MODELS[family])
            for case in CASES:
                validate_saved(receipts[case], family, proposals, case, model)
            configuration = {
                "family": family, "proposals": proposals, "candidate_dir": relative(directory),
                "candidate_sha256": {case: digest(directory / f"{case}.json") for case in CASES},
                "resource_sha256": {case: digest(directory / f"{case}.resources.json") for case in CASES},
            }
            prefix = f"cnn-{family}-{proposals}"
            variants.append(make_variant(
                f"{prefix}-unfiltered", receipts, resources, configuration, output / "predictions",
                unfiltered=True,
            ))
            if family == "current":
                thresholds = sorted(set((*THRESHOLDS, cnn_threshold, tree_threshold, blend.threshold)))
                for weight in WEIGHTS:
                    for threshold in thresholds:
                        variants.append(make_variant(
                            f"{prefix}-w{weight:g}-t{threshold:.16g}", receipts, resources, configuration,
                            output / "predictions", weight=weight, threshold=threshold,
                        ))
                variants.append(make_variant(
                    f"{prefix}-frozen-blend", receipts, resources, configuration, output / "predictions",
                    weight=blend.cnn_weight, threshold=blend.threshold, frozen=True,
                ))
            else:
                for threshold in sorted(set((*THRESHOLDS, receipts[CASES[0]]["threshold"]))):
                    variants.append(make_variant(
                        f"{prefix}-t{threshold:.16g}", receipts, resources, configuration,
                        output / "predictions", threshold=threshold,
                    ))
    selection = select_folds(variants)
    report = {
        "schema_version": 1, "family": "cnn", "variants": variants, "failures": failures,
        "status": "complete" if not failures else "partial",
        "exclusions": audit["exclusions"], "inventory_path": relative(output / "inventory.json"),
        "inventory_sha256": digest(output / "inventory.json"),
        "scoring_source_sha256": digest(ROOT / "final_evaluation.py"),
        "references_sha256": {
            case: digest(ROOT / "labels/organizer-v1/references" / f"{case}.json") for case in CASES
        },
        "reference_counts": {case: len(load_reference(case)["daughters"]) for case in CASES},
        "limitations": audit["limitations"],
        "selection": selection,
        "comparisons": comparison_summary(variants, selection),
    }
    write_json(output / "report.json", report)
    return report


def select_folds(variants: list[dict]) -> dict:
    eligible = [row for row in variants if row["eligible_for_selection"]]
    if not eligible:
        return {"status": "no eligible variants"}
    folds = []
    held_out: dict[str, list[dict]] = {key: [] for key in ("2", "3", "5")}
    for case in CASES:
        def key(row: dict) -> tuple:
            summary = summarize([r for r in row["scores"]["3"]["cases"] if r["case_id"] != case])
            dependencies = 0 if row["configuration"]["unfiltered"] else (
                1 if row["configuration"]["cnn_weight"] == 0 else 2)
            time = sum(row["runtime"][c]["runtime_s"] for c in CASES if c != case)
            return (-summary["f1"], summary["count_mae"], dependencies, time, row["name"])
        chosen = min(eligible, key=key)
        folds.append({"held_out_case": case, "selected_variant": chosen["name"]})
        for tolerance in held_out:
            held_out[tolerance].append(next(
                row for row in chosen["scores"][tolerance]["cases"] if row["case_id"] == case
            ))
    return {
        "scope": "Within CNN family only; parent comparison must rerun selection across all families.",
        "rule": "Four-case pooled 3-mm F1, count MAE, runtime dependencies, measured time, lexical name.",
        "folds": folds, "selection_stability": {
            name: sum(row["selected_variant"] == name for row in folds)
            for name in sorted({row["selected_variant"] for row in folds})
        },
        "held_out_scores": {key: {"summary": summarize(rows), "cases": rows} for key, rows in held_out.items()},
    }


def paired_bootstrap(left: dict, right: dict) -> dict:
    if [r["case_id"] for r in left["3"]["cases"]] != [r["case_id"] for r in right["3"]["cases"]]:
        raise ValueError("Bootstrap must resample identical paired cases.")
    indices = np.random.default_rng(42).integers(0, len(CASES), size=(10000, len(CASES)))
    intervals = {}
    for tolerance in ("2", "3", "5"):
        estimates = []
        for scores in (left, right):
            counts = np.array([
                [r["true_positives"], r["false_positives"], r["false_negatives"]]
                for r in scores[tolerance]["cases"]
            ])
            totals = counts[indices].sum(axis=1)
            tp, fp, fn = totals.T
            estimates.append(2 * tp / (2 * tp + fp + fn))
        intervals[tolerance] = {
            "left_f1_percentile_95": np.quantile(estimates[0], [0.025, 0.975]).tolist(),
            "paired_f1_difference_percentile_95": np.quantile(
                estimates[0] - estimates[1], [0.025, 0.975],
            ).tolist(),
        }
    return {
        "resamples": 10000, "seed": 42, "unit": "whole case", "intervals": intervals,
        "limitation": "Five reused cases; percentile intervals do not remove model-selection bias. "
                      "Selections are fixed during bootstrap, not refit for each resample.",
    }


def comparison_summary(variants: list[dict], selection: dict) -> dict:
    eligible = [row for row in variants if row["eligible_for_selection"]]
    if not eligible:
        return {"status": "no eligible variants"}
    best = min(eligible, key=lambda row: (
        -row["scores"]["3"]["summary"]["f1"], row["scores"]["3"]["summary"]["count_mae"],
        0 if row["configuration"]["unfiltered"] else 1 if row["configuration"]["cnn_weight"] == 0 else 2,
        sum(r["runtime_s"] for r in row["runtime"].values()), row["name"],
    ))
    baseline = next((row for row in eligible if row["name"] == "cnn-current-strict-unfiltered"), None)
    if baseline is None:
        return {"development_winner": best["name"], "status": "baseline unavailable", "deployment_selected": False}
    return {
        "development_winner": best["name"], "baseline": baseline["name"],
        "development_winner_vs_baseline": paired_bootstrap(best["scores"], baseline["scores"]),
        "within_family_held_out_vs_baseline": paired_bootstrap(
            selection["held_out_scores"], baseline["scores"],
        ),
        "deployment_selected": False,
    }


def extract(output: Path, offline: bool, resume: bool = False) -> None:
    write_json(output / "inventory.json", inventory())
    for family in MODELS:
        for proposals in ("strict", "review-union"):
            for case in CASES:
                directory = output / "candidates" / family / proposals
                destination = directory / f"{case}.json"
                resource_path = directory / f"{case}.resources.json"
                if resume and destination.exists() and resource_path.exists():
                    saved = read_json(destination)
                    measured = read_json(resource_path)
                    if saved["status"] == "success" and measured["exit_code"] == 0:
                        validate_saved(saved, family, proposals, case, PatchModel.load(MODELS[family]))
                        if offline and not saved.get("network_denied"):
                            raise ValueError("Existing receipt lacks requested offline verification.")
                    print(f"Preserved {family} {proposals} {case}: {saved['status']}", flush=True)
                    continue
                if destination.exists() or resource_path.exists():
                    raise ValueError("Extraction requires new paths; use replay for existing artifacts.")
                bootstrap = (
                    "import runpy,sys;"
                    + (f"sys.path.insert(0,{str(SNAPSHOT)!r});" if family != "current" else "")
                    + f"runpy.run_path({str(Path(__file__).resolve())!r},run_name='__main__')"
                )
                command = [
                    sys.executable, "-c", bootstrap, "worker", "--family", family,
                    "--proposals", proposals, "--case", case, "--output", str(destination),
                ]
                if offline:
                    trace = ROOT / "outputs" / f"cnn-{family}-{proposals}-{case}.log"
                    command = [
                        "strace", "-qq", "-f", "-e", "trace=%network", "-e",
                        "inject=%network:error=ENETUNREACH", "-o", str(trace), *command,
                    ]
                measurement = measure(command, case)
                if destination.exists() and read_json(destination).get("network_denied"):
                    measurement.update(network=False, network_evidence="Worker socket creation denied by syscall injection.")
                write_json(resource_path, measurement)
                if not destination.exists():
                    write_json(destination, {
                        "status": "failed", "error": "Worker exited before writing a receipt.",
                        "exit_code": measurement["exit_code"],
                    })
                print(f"{family} {proposals} {case}: exit {measurement['exit_code']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("extract", "replay", "worker", "audit"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--family", choices=tuple(MODELS), default="current")
    parser.add_argument("--proposals", choices=("strict", "review-union"), default="strict")
    parser.add_argument("--case", choices=CASES, default=CASES[0])
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "worker":
        raise SystemExit(run_case(args.family, args.proposals, args.case, args.output))
    if args.command == "audit":
        write_json(args.output / "inventory.json", inventory())
    elif args.command == "extract":
        extract(args.output, args.offline, args.resume)
    else:
        report = replay(args.output)
        print(f"{len(report['variants'])} variants; {len(report['failures'])} failed family/pool combinations.")


if __name__ == "__main__":
    main()
