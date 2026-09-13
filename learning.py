"""Train an optional CPU candidate classifier from labelled candidate features."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from scipy.special import expit

from detector import Branch, Detection

GEOMETRY_FEATURES = [
    "radius_mm", "mean_vesselness", "evidence_score",
    "path_length_mm", "seed_distance_mm", "tortuosity",
]
# Context features separate bone, veins and caps from arteries; the detector computes them per candidate.
CONTEXT_FEATURES = [
    "path_hu_relative", "bone_distance_mm", "parent_angle_degrees", "arc_position",
    "native_spacing_mm", "connector_gap", "candidate_volume_mm3",
]
FEATURE_NAMES = GEOMETRY_FEATURES + CONTEXT_FEATURES
FloatArray = npt.NDArray[np.float64]


def features(branch: Branch) -> list[float]:
    path = np.asarray(branch.path_xyz_mm)
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    missing = [name for name in CONTEXT_FEATURES if name not in branch.features]
    if missing:
        raise ValueError(f"Candidate lacks detector context features: {missing}")
    return [
        branch.radius_mm, branch.mean_vesselness, branch.evidence_score, length,
        float(np.linalg.norm(np.asarray(branch.seed_xyz_mm) - branch.ostium_xyz_mm)),
        length / max(float(np.linalg.norm(path[-1] - path[0])), 0.001),
        *(float(branch.features[name]) for name in CONTEXT_FEATURES),
    ]


@dataclass
class CandidateModel:
    mean: list[float]
    scale: list[float]
    weights: list[float]
    bias: float
    threshold: float
    split: dict[str, list[str]]

    def scores(self, matrix: npt.ArrayLike) -> FloatArray:
        array = np.asarray(matrix, dtype=float)
        if array.ndim != 2 or array.shape[1] != len(FEATURE_NAMES) or not np.isfinite(array).all():
            raise ValueError("Candidate features must be a finite matrix with the declared feature order.")
        return expit(((array - self.mean) / self.scale) @ self.weights + self.bias)

    def save(self, path: Path) -> None:
        write_json(path, {"schema_version": 1, "feature_names": FEATURE_NAMES, **asdict(self)})

    @classmethod
    def load(cls, path: Path) -> "CandidateModel":
        value = json.loads(path.read_text())
        if value["schema_version"] != 1 or value["feature_names"] != FEATURE_NAMES:
            raise ValueError("Unsupported model schema or feature order.")
        vectors = [np.asarray(value[key], dtype=float) for key in ("mean", "scale", "weights")]
        if any(v.shape != (len(FEATURE_NAMES),) or not np.isfinite(v).all() for v in vectors):
            raise ValueError(f"Model vectors must contain {len(FEATURE_NAMES)} finite values.")
        if np.any(vectors[1] <= 0) or not np.isfinite(value["bias"]):
            raise ValueError("Invalid model scale or bias.")
        if not np.isfinite(value["threshold"]) or not 0 <= value["threshold"] <= 1:
            raise ValueError("Model threshold must lie between zero and one.")
        split = validate_split(value["split"])
        return cls(
            vectors[0].tolist(), vectors[1].tolist(), vectors[2].tolist(),
            float(value["bias"]), float(value["threshold"]), split,
        )


def load_reviews(paths: list[Path]) -> list[dict]:
    rows: dict[tuple[str, str], dict] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        if payload["schema_version"] != 1 or payload["feature_names"] != FEATURE_NAMES:
            raise ValueError("Unsupported review schema or feature order.")
        if payload.get("scope") != "candidate_reviews_only":
            raise ValueError("Expected candidate reviews exported by Aorta Explorer.")
        for row in payload["records"]:
            if row["label"] not in ("confirmed", "rejected"):
                raise ValueError("Every training row needs an explicit confirmed/rejected label.")
            if any(not isinstance(row[key], str) or not row[key] for key in ("case_id", "instance_id")):
                raise ValueError("Case and instance identifiers must be nonempty strings.")
            vector = np.asarray(row["features"], dtype=float)
            if vector.shape != (len(FEATURE_NAMES),) or not np.isfinite(vector).all():
                raise ValueError(f"Review features must contain {len(FEATURE_NAMES)} finite values.")
            key = row["case_id"], row["instance_id"]
            if key in rows and rows[key] != row:
                raise ValueError(f"Conflicting reviews for {key}; resolve them before training.")
            rows[key] = row
    if not rows:
        raise ValueError("No labelled candidates supplied.")
    return [rows[key] for key in sorted(rows)]


def validate_split(value: dict) -> dict[str, list[str]]:
    if set(value) != {"train", "validation", "test"}:
        raise ValueError("Split must declare train, validation and test case IDs.")
    all_cases: list[str] = []
    for cases in value.values():
        if not isinstance(cases, list) or not cases or any(not isinstance(c, str) or not c for c in cases):
            raise ValueError("Every partition needs a nonempty list of case IDs.")
        all_cases.extend(cases)
    if len(set(all_cases)) != len(all_cases):
        raise ValueError("Patient/case leakage: split partitions must be disjoint.")
    return value


def split_cases(
    rows: list[dict], seed: int, test: list[str] | None = None, validation: list[str] | None = None,
) -> dict[str, list[str]]:
    cases = sorted({row["case_id"] for row in rows})
    if len(cases) < 6:
        raise ValueError("Review at least six independent cases before creating a three-way split.")
    chosen = [*(test or []), *(validation or [])]
    unknown = sorted(set(chosen) - set(cases))
    if unknown:
        raise ValueError(f"Requested split cases have no reviews: {unknown}")
    if len(set(chosen)) != len(chosen):
        raise ValueError("A case cannot be in both the test and validation partitions.")
    remaining = [c for c in np.random.default_rng(seed).permutation(cases).tolist() if c not in chosen]
    holdout = max(1, len(cases) // 5)
    test_cases = list(test) if test else remaining[:holdout]
    remaining = [c for c in remaining if c not in test_cases]
    validation_cases = list(validation) if validation else remaining[:max(1, len(test_cases))]
    train_cases = [c for c in remaining if c not in validation_cases]
    if not train_cases:
        raise ValueError("The requested test and validation cases leave nothing to train on.")
    return validate_split({"train": sorted(train_cases), "validation": validation_cases, "test": test_cases})


def metrics(labels: FloatArray, scores: FloatArray, threshold: float) -> dict:
    positive = scores >= threshold
    tp = int(np.sum(positive & (labels == 1)))
    fp = int(np.sum(positive & (labels == 0)))
    fn = int(np.sum(~positive & (labels == 1)))
    tn = int(np.sum(~positive & (labels == 0)))
    return {
        "reviewed_candidates": len(labels), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "brier_score": float(np.mean((scores - labels) ** 2)),
    }


def train(
    rows: list[dict], split: dict[str, list[str]], minimum_training_recall: float = 0.0,
) -> tuple[CandidateModel, dict]:
    if not np.isfinite(minimum_training_recall) or not 0 <= minimum_training_recall <= 1:
        raise ValueError("Minimum training recall must lie between zero and one.")
    split = validate_split(split)
    declared = {case for partition in split.values() for case in partition}
    if declared != {row["case_id"] for row in rows}:
        raise ValueError("Split must include exactly the reviewed cases; no missing or unknown cases.")
    partitions = {}
    for name, cases in split.items():
        selected = [row for row in rows if row["case_id"] in cases]
        x = np.asarray([row["features"] for row in selected], dtype=float)
        y = np.asarray([row["label"] == "confirmed" for row in selected], dtype=float)
        if set(y) != {0, 1}:
            raise ValueError(f"{name} needs both confirmed and rejected candidates from independent cases.")
        partitions[name] = x, y
    x, y = partitions["train"]
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-6)
    normalized = (x - mean) / scale

    def objective(parameters: FloatArray) -> tuple[float, FloatArray]:
        weights, bias = parameters[:-1], parameters[-1]
        logits = normalized @ weights + bias
        residual = expit(logits) - y
        loss = float(np.mean(np.logaddexp(0, logits) - y * logits) + 0.05 * np.sum(weights**2))
        gradient = np.r_[normalized.T @ residual / len(y) + 0.1 * weights, residual.mean()]
        return loss, gradient

    fitted = minimize(objective, np.zeros(len(FEATURE_NAMES) + 1), jac=True, method="L-BFGS-B")
    if not fitted.success:
        raise ValueError(f"Model optimization failed: {fitted.message}")
    model = CandidateModel(mean.tolist(), scale.tolist(), fitted.x[:-1].tolist(), float(fitted.x[-1]), 0.5, split)
    validation_x, validation_y = partitions["validation"]
    validation_scores = model.scores(validation_x)
    thresholds = sorted({0.0, 0.5, 1.0, *validation_scores.tolist()})
    if minimum_training_recall > 0:
        training_scores = model.scores(partitions["train"][0])
        training_y = partitions["train"][1]
        thresholds = sorted({*thresholds, *training_scores[training_y == 1].tolist()})
        thresholds = [
            threshold for threshold in thresholds
            if metrics(training_y, training_scores, threshold)["recall"] >= minimum_training_recall
        ]
    model.threshold = max(thresholds, key=lambda t: (
        metrics(validation_y, validation_scores, t)["f1"], -abs(t - 0.5),
    ))
    partition_reports: dict[str, dict] = {}
    for name, (x, y) in partitions.items():
        partition_reports[name] = {
            "classifier": metrics(y, model.scores(x), model.threshold),
            "keep_all_baseline": metrics(y, np.ones(len(y)), 0.5),
        }
    report = {
        "scope": "Labelled candidates only; misses never proposed by the detector are not measured.",
        "label_sources": {
            source: sum(row.get("labeller", "unspecified") == source for row in rows)
            for source in sorted({row.get("labeller", "unspecified") for row in rows})
        },
        "limitations": [
            "AI verdicts are pseudo-labels; synthetic labels are analytic geometry, not clinical validation.",
            "Not a calibrated clinical probability; validate on independent, complete daughter annotations.",
            "Do not tune against the test results. Group repeat scans of one patient into the same partition.",
        ],
        "feature_names": FEATURE_NAMES,
        "split": split,
        "threshold_selected_on": "validation",
        "minimum_training_recall": minimum_training_recall,
        "threshold": model.threshold,
        "partitions": partition_reports,
    }
    return model, report


def filter_detection(result: Detection, model: CandidateModel) -> dict:
    scores = model.scores([features(b) for b in result.branches]) if result.branches else np.empty(0)
    decisions = {b.instance_id: float(score) for b, score in zip(result.branches, scores)}
    kept = [b for b, score in zip(result.branches, scores) if score >= model.threshold]
    rejected = len(result.branches) - len(kept)
    result.branches = kept
    result.rejections["learned_candidate_filter"] = rejected
    result.warnings.append("Optional reviewed-candidate classifier applied; scores are not clinically calibrated.")
    return {"scores": decisions, "threshold": model.threshold, "rejected": rejected}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    split_parser = commands.add_parser("split", help="Persist a reproducible case-level split before training.")
    split_parser.add_argument("--reviews", required=True, nargs="+", type=Path)
    split_parser.add_argument("--output", required=True, type=Path)
    split_parser.add_argument("--seed", type=int, default=42)
    split_parser.add_argument("--test", nargs="+", help="Hold these case IDs out as the final test set.")
    split_parser.add_argument("--validation", nargs="+", help="Use these case IDs to pick the decision threshold.")
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--reviews", required=True, nargs="+", type=Path)
    train_parser.add_argument("--split", required=True, type=Path)
    train_parser.add_argument("--model", required=True, type=Path)
    train_parser.add_argument("--report", required=True, type=Path)
    train_parser.add_argument(
        "--minimum-training-recall", type=float, default=0.0,
        help="Optional retention constraint during threshold selection (0 disables it; not a recall guarantee).",
    )
    args = parser.parse_args()
    try:
        rows = load_reviews(args.reviews)
        if args.command == "split":
            write_json(args.output, split_cases(rows, args.seed, args.test, args.validation))
        else:
            model, report = train(rows, json.loads(args.split.read_text()), args.minimum_training_recall)
            model.save(args.model)
            write_json(args.report, report)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
