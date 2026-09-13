"""Audit frozen candidate reviews and current-proposal agreement without relabelling."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess

from matplotlib.figure import Figure
import numpy as np
import SimpleITK as sitk

from autolabel import candidate_record, case_paths, fingerprint
from detector import detect, detect_pool
from learning import CandidateModel, FEATURE_NAMES, features, load_reviews
from nifti_io import read_nifti
from research_run import source_hashes, validate_tree_source
from tabular_learning import TreeModel

PARTITIONS = ("train", "validation", "test")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(value: str, vector: list[float]) -> tuple[float, ...]:
    parts = json.loads(value)
    if not isinstance(parts, list) or len(parts) != 5:
        raise ValueError("Expected ostium, seed, direction, radius and features in fingerprint.")
    arrays = [np.asarray(part, dtype=float) for part in parts]
    if [a.shape for a in arrays] != [(3,), (3,), (3,), (), (len(FEATURE_NAMES),)]:
        raise ValueError("Invalid fingerprint geometry or feature shape.")
    flat = np.concatenate([a.reshape(-1) for a in arrays])
    if not np.isfinite(flat).all() or not np.array_equal(arrays[-1], vector):
        raise ValueError("Fingerprint must be finite and match the stored feature vector.")
    return tuple(float(v) for v in flat)


def review_index(rows: list[dict]) -> dict[tuple[str, tuple[float, ...]], dict]:
    result = {}
    for row in rows:
        key = row["case_id"], identity(row["fingerprint"], row["features"])
        if key in result:
            raise ValueError(f"Ambiguous repeated candidate fingerprint in {row['case_id']}.")
        result[key] = row
    return result


def agreement(rows: list[dict], model_name: str) -> dict:
    counts = Counter(
        f"{row['label']}_{'retained' if row[f'{model_name}_keep'] else 'removed'}"
        for row in rows
    )
    confirmed = sum(row["label"] == "confirmed" for row in rows)
    rejected = len(rows) - confirmed
    retained = counts["confirmed_retained"] + counts["rejected_retained"]
    return {
        "reviewed_candidates": len(rows),
        **{key: counts[key] for key in (
            "confirmed_retained", "confirmed_removed", "rejected_retained", "rejected_removed",
        )},
        "agreement": (counts["confirmed_retained"] + counts["rejected_removed"]) / len(rows) if rows else None,
        "confirmed_retention": counts["confirmed_retained"] / confirmed if confirmed else None,
        "rejected_removal": counts["rejected_removed"] / rejected if rejected else None,
        "confirmed_share_of_retained": counts["confirmed_retained"] / retained if retained else None,
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def score_history(rows: list[dict], models: dict[str, CandidateModel]) -> list[dict]:
    matrix = np.asarray([row["features"] for row in rows])
    scores = {name: model.scores(matrix) for name, model in models.items()}
    result = []
    for index, row in enumerate(rows):
        fp = identity(row["fingerprint"], row["features"])
        item = {
            "case_id": row["case_id"], "instance_id": row["instance_id"],
            "label": row["label"], "labeller": row["labeller"], "reviewed_at": row["reviewed_at"],
            "ostium_x_mm": fp[0], "ostium_y_mm": fp[1], "ostium_z_mm": fp[2],
            "fingerprint_sha256": hashlib.sha256(json.dumps(fp).encode()).hexdigest(),
            **dict(zip(FEATURE_NAMES, row["features"])),
        }
        for name, model in models.items():
            partitions = [p for p, cases in model.split.items() if row["case_id"] in cases]
            if len(partitions) != 1:
                raise ValueError(f"Review {row['case_id']} must occur in exactly one {name} partition.")
            item[f"{name}_partition"] = partitions[0]
            item[f"{name}_score"] = float(scores[name][index])
            item[f"{name}_keep"] = bool(scores[name][index] >= model.threshold)
        result.append(item)
    return result


def current_rows(
    data_root: Path, output: Path, reviews: list[dict], models: dict[str, CandidateModel], tree: TreeModel,
) -> tuple[list[dict], list[dict]]:
    lookup = review_index(reviews)
    result_rows, case_rows = [], []
    for directory in sorted(data_root.glob("subject*")):
        image_path, mask_path = case_paths(directory)
        image, mask = read_nifti(str(image_path)), read_nifti(str(mask_path))
        candidates = [r for r in reviews if r["case_id"] == directory.name]
        historical_origins = np.asarray([
            identity(r["fingerprint"], r["features"])[:3] for r in candidates
        ]).reshape(-1, 3)
        case_output = output / "current" / directory.name
        case_output.mkdir(parents=True)
        for profile, run in (("strict", detect), ("review_union", detect_pool)):
            detection = run(image, mask)
            matrix = np.asarray([features(b) for b in detection.branches]).reshape(-1, len(FEATURE_NAMES))
            scores = {name: model.scores(matrix) for name, model in models.items()}
            scores["synthetic_tree"] = tree.scores(matrix)
            thresholds = {name: model.threshold for name, model in models.items()}
            thresholds["synthetic_tree"] = tree.threshold
            write_json(case_output / f"{profile}.json", detection.diagnostics())
            write_json(case_output / f"{profile}-prediction.json", detection.prediction(directory.name))
            local_rows = []
            for index, branch in enumerate(detection.branches):
                fp = identity(fingerprint(candidate_record(branch)), matrix[index].tolist())
                review = lookup.get((directory.name, fp))
                distances = np.linalg.norm(historical_origins - branch.ostium_xyz_mm, axis=1)
                nearest = int(np.argmin(distances)) if len(distances) else None
                same_id = next((r for r in candidates if r["instance_id"] == branch.instance_id), None)
                row = {
                    "case_id": directory.name, "profile": profile, "instance_id": branch.instance_id,
                    "label": review["label"] if review else "",
                    "historical_instance_id": review["instance_id"] if review else "",
                    "same_id_has_same_identity": (
                        identity(same_id["fingerprint"], same_id["features"]) == fp if same_id else ""
                    ),
                    "nearest_historical_id": candidates[nearest]["instance_id"] if nearest is not None else "",
                    "nearest_historical_distance_mm": float(distances[nearest]) if nearest is not None else "",
                    "historical_origins_within_3mm": int(np.sum(distances <= 3)),
                    "ostium_x_mm": branch.ostium_xyz_mm[0], "ostium_y_mm": branch.ostium_xyz_mm[1],
                    "ostium_z_mm": branch.ostium_xyz_mm[2],
                    "origin_diameter_mm": branch.features.get("origin_diameter_mm", ""),
                    "origin_diameter_upper_mm": branch.features.get("origin_diameter_upper_mm", ""),
                    "warnings": " | ".join(branch.warnings),
                    **dict(zip(FEATURE_NAMES, matrix[index].tolist())),
                }
                for name, values in scores.items():
                    row[f"{name}_score"] = float(values[index])
                    row[f"{name}_keep"] = bool(values[index] >= thresholds[name])
                local_rows.append(row)
            result_rows.extend(local_rows)
            case_rows.append({
                "case_id": directory.name, "profile": profile, "proposals": len(local_rows),
                "exact_confirmed": sum(r["label"] == "confirmed" for r in local_rows),
                "exact_rejected": sum(r["label"] == "rejected" for r in local_rows),
                "unlabelled_or_changed": sum(not r["label"] for r in local_rows),
                "origin_unresolved": sum(r["origin_diameter_mm"] == "" for r in local_rows),
                "image_sha256": sha256(image_path), "mask_sha256": sha256(mask_path),
                "warnings": " | ".join(detection.warnings), "detection_s": detection.timings["total_s"],
                **{f"{name}_retained": sum(r[f"{name}_keep"] for r in local_rows) for name in scores},
            })
            print(directory.name, profile, case_rows[-1]["proposals"],
                  "exact labels", sum(bool(r["label"]) for r in local_rows), flush=True)
    return result_rows, case_rows


def charts(output: Path, historical: list[dict], current: list[dict]) -> None:
    cases = sorted({r["case_id"] for r in historical})
    positive = [sum(r["case_id"] == case and r["label"] == "confirmed" for r in historical) for case in cases]
    negative = [sum(r["case_id"] == case and r["label"] == "rejected" for r in historical) for case in cases]
    figure = Figure(figsize=(12, 6), layout="constrained")
    axis = figure.subplots()
    axis.bar(cases, positive, color="#0072B2", label="AI confirmed")
    axis.bar(cases, negative, bottom=positive, color="#D55E00", label="AI rejected")
    axis.set(title="Steven's stored reviews: candidate decisions, not complete vessel labels",
             ylabel="Historical candidate decisions", xlabel="Case")
    axis.tick_params(axis="x", rotation=65)
    axis.legend()
    figure.savefig(output / "historical-labels.png", dpi=170)
    figure = Figure(figsize=(10, 5), layout="constrained")
    axis = figure.subplots()
    profiles = ["strict", "review_union"]
    bottom = np.zeros(2)
    for label, color, legend in (
        ("confirmed", "#0072B2", "Exact historical confirmed"),
        ("rejected", "#D55E00", "Exact historical rejected"),
        ("", "#A7ADB4", "Changed or unlabelled"),
    ):
        counts = [sum(r["profile"] == profile and r["label"] == label for r in current) for profile in profiles]
        axis.bar(profiles, counts, bottom=bottom, color=color, label=legend)
        for index, value in enumerate(counts):
            if value >= 12:
                axis.text(index, bottom[index] + value / 2, str(value), ha="center", va="center", color="white")
            elif value:
                axis.annotate(
                    str(value), (index - 0.4, bottom[index] + value / 2),
                    xytext=(-20, 12 if label == "rejected" else -12), textcoords="offset points",
                    ha="right", va="center", arrowprops={"arrowstyle": "-", "color": color},
                )
        bottom += counts
    axis.set(title="Current proposals cannot inherit labels from branch numbers",
             ylabel="Current proposals")
    axis.legend(loc="upper left")
    figure.savefig(output / "current-identity-coverage.png", dpi=170)


def report_text(summary: dict) -> str:
    rows = [
        "# Steven's candidate-label analytics",
        "",
        "Frozen-model agreement and current-candidate identity audit. No training, threshold tuning, "
        "new labels or production changes. These are AI-review comparisons, not clinical accuracy.",
        "",
        "## Historical agreement",
        "",
        "| Model | Original partition | N | Confirmed kept / removed | Rejected kept / removed |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for name, partitions in summary["historical_agreement"].items():
        for partition, counts in partitions.items():
            rows.append(
                f"| {name} | {partition} | {counts['reviewed_candidates']} | "
                f"{counts['confirmed_retained']} / {counts['confirmed_removed']} | "
                f"{counts['rejected_retained']} / {counts['rejected_removed']} |"
            )
    rows += [
        "", "Original test cases are now exposed development cases. Synthetic-augmented model rows "
        "above include only the original real-case reviews, not its synthetic training rows.",
        "", "## Current source: exact identity coverage", "",
        "| Profile | Proposals | Exact confirmed | Exact rejected | Changed / unlabelled |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for profile, counts in summary["current_coverage"].items():
        rows.append(
            f"| {profile} | {counts['proposals']} | {counts['exact_confirmed']} | "
            f"{counts['exact_rejected']} | {counts['unlabelled_or_changed']} |"
        )
    rows += [
        "", "Exact identity uses the entire stored ostium, seed, direction, radius and feature vector, "
        "within the same case. Whitespace and numeric JSON formatting are normalized; no floating-point "
        "tolerance or branch-ID fallback transfers a label. Nearest origins in the CSV are review aids "
        "only and are neither one-to-one assignments nor truth.",
        "", "## Current-source model agreement on the exactly matched subset", "",
        "| Profile | Model | Matched N | Confirmed kept / removed | Rejected kept / removed |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for profile, models in summary["current_agreement"].items():
        for name, counts in models.items():
            rows.append(
                f"| {profile} | {name} | {counts['reviewed_candidates']} | "
                f"{counts['confirmed_retained']} / {counts['confirmed_removed']} | "
                f"{counts['rejected_retained']} / {counts['rejected_removed']} |"
            )
    rows += [
        "", "The current tree was fitted to synthetic data and scored only on current-source vectors "
        "after its source contract passed. Both legacy logistic models saw many of these real cases "
        "during fitting. The exactly matched subset is selected and incomplete; no figure here "
        "establishes full proposal recall or real-patient generalization.",
        "", "## Files and method", "",
        "- historical-candidates.csv: all stored vectors, model scores, partitions and decisions.",
        "- historical-disagreements.csv: frozen models disagreeing with the stored reviewer.",
        "- current-candidates.csv: all current candidates, exact identities, scores and proximity aids.",
        "- current-cases.csv: per-case coverage, counts, warnings and input hashes.",
        "- current/: untouched strict/review-union predictions and full diagnostics.",
        "- summary.json: aggregates, source/model/review hashes and label provenance.",
        "",
        "The original raw labels remain unchanged. Unlabelled/changed proposals are not negatives; "
        "unmatched historical positives are not verified missed arteries. Historical labels did not "
        "independently establish 2 mm origin eligibility. The two empty-label cases remain in current "
        "inference and are not assumed to contain no real branches.",
    ]
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, default=Path("labels/reviews.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--original-model", type=Path, default=Path("labels/candidate-model.json"))
    parser.add_argument("--augmented-model", type=Path, default=Path("labels/candidate-model-synthetic.json"))
    parser.add_argument("--tree-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not any(path.is_dir() for path in args.data_root.glob("subject*")):
        parser.error("--data-root must contain prepared subject directories.")
    reviews = load_reviews([args.reviews])
    raw_count = len(json.loads(args.reviews.read_text())["records"])
    if raw_count != len(reviews):
        raise ValueError("Duplicate review rows must be audited before analysis.")
    review_index(reviews)
    paths = {"steven_logistic": args.original_model, "augmented_logistic": args.augmented_model}
    models = {name: CandidateModel.load(path) for name, path in paths.items()}
    tree = TreeModel.load(args.tree_model)
    validate_tree_source(tree, source_hashes())
    if tree.feature_names != FEATURE_NAMES:
        raise ValueError("This audit expects a base-feature tree; no extra features are fabricated.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    historical = score_history(reviews, models)
    write_csv(args.output_dir / "historical-candidates.csv", historical)
    write_csv(args.output_dir / "historical-disagreements.csv", [
        row for row in historical
        if any(row[f"{name}_keep"] != (row["label"] == "confirmed") for name in models)
    ])
    current, cases = current_rows(args.data_root, args.output_dir, reviews, models, tree)
    write_csv(args.output_dir / "current-candidates.csv", current)
    write_csv(args.output_dir / "current-cases.csv", cases)
    summary = {
        "scope": "AI candidate-review agreement only; no expert accuracy or promotion",
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {**source_hashes(), "review_analytics.py": sha256(Path(__file__))},
        "reviews_sha256": sha256(args.reviews),
        "models": {
            **{name: {"sha256": sha256(path), "threshold": models[name].threshold} for name, path in paths.items()},
            "synthetic_tree": {"sha256": sha256(args.tree_model), "threshold": tree.threshold},
        },
        "labels": dict(Counter(row["label"] for row in reviews)),
        "labellers": dict(Counter(row["labeller"] for row in reviews)),
        "reviewed_cases": len({row["case_id"] for row in reviews}),
        "historical_agreement": {
            name: {partition: agreement([
                row for row in historical if partition == "all" or row[f"{name}_partition"] == partition
            ], name) for partition in (*PARTITIONS, "all")} for name in models
        },
        "current_coverage": {
            profile: {key: sum(c[key] for c in cases if c["profile"] == profile) for key in (
                "proposals", "exact_confirmed", "exact_rejected", "unlabelled_or_changed", "origin_unresolved",
            )} for profile in ("strict", "review_union")
        },
        "current_agreement": {
            profile: {
                name: agreement([r for r in current if r["profile"] == profile and r["label"]], name)
                for name in (*models, "synthetic_tree")
            } for profile in ("strict", "review_union")
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "REPORT.md").write_text(report_text(summary))
    charts(args.output_dir, historical, current)
    print(json.dumps(summary["current_coverage"]), flush=True)


if __name__ == "__main__":
    main()
