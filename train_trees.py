"""Optional conservative tree training; the test partition stays sealed by default.

Example:
  python train_trees.py --reviews corpus/reviews.json --split corpus/split.json \
      --algorithm gradient_boosting --features extended --model tree.json --report report.json

Use --pseudo-weight 0 for synthetic/expert-only fitting. Evaluation always keeps
the observed prevalence. Legacy reviews require an explicit --case-groups JSON
mapping case IDs to patient groups; repeat scans must share a group. Input files
are hashed and preserved in report provenance. Never overwrite existing outputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import sys
import warnings

import numpy as np

try:
    from scipy.optimize import minimize
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from candidate_patches import preprocessing_metadata
except ImportError:
    TRAINING_AVAILABLE = False
else:
    TRAINING_AVAILABLE = True

from tabular_learning import (
    EXTENDED_FEATURE_NAMES, EXTRA_FEATURE_NAMES, FEATURE_NAMES, FloatArray,
    PlattCalibration, Tree, TreeModel, feature_matrix, finite_number, json_sha256,
    nonempty_string, probability_logits, read_json, sigmoid, unit_interval,
    validate_split, write_json,
)

DEFAULT_SOURCE_WEIGHTS = {"analytic": 1.0, "expert": 1.0, "pseudo": 0.3}
SOURCE_CATEGORIES = {
    "analytic_synthetic_geometry": "analytic",
    "analytic": "analytic",
    "claude": "pseudo",
    "pseudo": "pseudo",
    "ai_pseudo": "pseudo",
    "expert": "expert",
    "human_expert": "expert",
    "organizer_expert": "expert",
}
LIMITATIONS = [
    "Conditional candidate classification cannot recover unproposed reference misses.",
    "Pseudo-label agreement and analytic synthetic performance are not clinical accuracy.",
    "Calibration fits observed candidate labels, not clinically calibrated probabilities.",
    "The 25 supplied scans were previously inspected/pseudo-trained; incoming expert cases may overlap.",
    "Disjoint current fitting partitions do not establish historical independence.",
    "Test results must not select model configuration, features, calibration or thresholds.",
]


def require_training() -> None:
    if not TRAINING_AVAILABLE:
        raise ValueError(
            "Optional tree training dependencies are absent. "
            "Install requirements-trees.txt before training; NumPy JSON inference needs no sklearn."
        )


def make_estimator(algorithm: str) -> GradientBoostingClassifier | RandomForestClassifier:
    require_training()
    if algorithm == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=64, learning_rate=0.05, max_depth=2, min_samples_leaf=8,
            subsample=1.0, init="zero", loss="log_loss", random_state=42,
        )
    if algorithm == "random_forest":
        return RandomForestClassifier(
            n_estimators=96, max_depth=6, min_samples_leaf=4, max_features=1.0,
            bootstrap=True, n_jobs=4, random_state=42,
        )
    raise ValueError("Algorithm must be gradient_boosting or random_forest.")


def export_estimator(
    estimator: GradientBoostingClassifier | RandomForestClassifier,
    names: list[str], split: dict[str, list[str]], metadata: dict | None = None,
) -> TreeModel:
    require_training()
    if not np.array_equal(estimator.classes_, [0, 1]) or estimator.n_features_in_ != len(names):
        raise ValueError("Estimator must have binary classes [0, 1] and the declared feature dimensions.")
    if isinstance(estimator, GradientBoostingClassifier):
        if estimator.init != "zero" or estimator.loss != "log_loss" or estimator.estimators_.shape[1:] != (1,):
            raise ValueError("Only binary log-loss boosting with explicit zero initialization is supported.")
        estimators = estimator.estimators_[:, 0]
        algorithm, rate = "gradient_boosting", float(estimator.learning_rate)
    elif isinstance(estimator, RandomForestClassifier):
        if estimator.n_outputs_ != 1:
            raise ValueError("Only single-output random forests are supported.")
        estimators = estimator.estimators_
        algorithm, rate = "random_forest", 1.0
    else:
        raise ValueError("Unsupported estimator type.")
    trees = []
    for fitted in estimators:
        tree = fitted.tree_
        if algorithm == "random_forest":
            if tree.value.shape[1:] != (1, 2):
                raise ValueError("Expected two class values at every forest node.")
            values = tree.value[:, 0, :]
            if not np.isfinite(values).all() or np.any(values < 0) or np.any(values.sum(axis=1) <= 0):
                raise ValueError("Invalid forest class values.")
            prediction = values[:, 1] / values.sum(axis=1)
        else:
            if tree.value.shape[1:] != (1, 1):
                raise ValueError("Expected scalar regression values in boosting trees.")
            prediction = tree.value[:, 0, 0]
        trees.append(Tree(
            tree.children_left.tolist(), tree.children_right.tolist(), tree.feature.tolist(),
            tree.threshold.tolist(), prediction.tolist(),
        ))
    model = TreeModel(algorithm, list(names), trees, 0.0, validate_split(split), metadata or {}, rate)
    model.validate()
    return model


def source_category(row: dict) -> str:
    labeller = nonempty_string(row.get("labeller"), "Review labeller provenance")
    declared = row.get("label_source", labeller)
    nonempty_string(declared, "Review label_source")
    if declared not in SOURCE_CATEGORIES:
        raise ValueError(f"Unknown label provenance {declared!r}; declare analytic, expert or pseudo.")
    category = SOURCE_CATEGORIES[declared]
    if labeller in SOURCE_CATEGORIES and SOURCE_CATEGORIES[labeller] != category:
        raise ValueError("Conflicting labeller and label_source provenance.")
    return category


def validate_row(row: dict, feature_set: str) -> list[float]:
    if not isinstance(row, dict):
        raise ValueError("Every review must be an object.")
    for name in ("case_id", "instance_id", "fingerprint", "group_id"):
        nonempty_string(row.get(name), f"Review {name}")
    if row.get("label") not in ("confirmed", "rejected"):
        raise ValueError("Every review needs an explicit confirmed/rejected label.")
    source = source_category(row)
    if source == "analytic":
        nonempty_string(row.get("family"), "Synthetic family")
        if not row["group_id"].startswith("synthetic:"):
            raise ValueError("Analytic groups must use synthetic:<generation seed> provenance.")
    if not isinstance(row.get("features"), list):
        raise ValueError("Review features must be a finite numeric list.")
    vector = feature_matrix([row["features"]], FEATURE_NAMES)[0].tolist()
    if feature_set == "extended":
        extra = row.get("extra_features")
        if not isinstance(extra, dict) or set(extra) != set(EXTRA_FEATURE_NAMES):
            raise ValueError("Extended training requires all three named extra_features; no zero imputation.")
        vector.extend(finite_number(extra[name], f"Extra feature {name}") for name in EXTRA_FEATURE_NAMES)
        feature_matrix([vector], EXTENDED_FEATURE_NAMES)
    return vector


def load_reviews(
    paths: list[Path], feature_set: str = "base", case_groups: dict[str, str] | None = None,
) -> tuple[list[dict], list[dict]]:
    if feature_set not in ("base", "extended"):
        raise ValueError("Feature selection must be base or extended.")
    rows: dict[tuple[str, str], dict] = {}
    provenance: list[dict] = []
    for path in paths:
        payload = read_json(path)
        if (type(payload.get("schema_version")) is not int or payload["schema_version"] != 1
                or payload.get("feature_names") != FEATURE_NAMES
                or payload.get("scope") != "candidate_reviews_only"):
            raise ValueError("Unsupported review schema, scope or feature order.")
        if not isinstance(payload.get("records"), list):
            raise ValueError("Reviews must contain a records list.")
        provenance.append({
            "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "metadata": {key: value for key, value in payload.items() if key != "records"},
        })
        for original in payload["records"]:
            if not isinstance(original, dict):
                raise ValueError("Every review must be an object.")
            row = dict(original)
            case = nonempty_string(row.get("case_id"), "Review case_id")
            if case_groups is not None and case in case_groups:
                group = nonempty_string(case_groups[case], "Declared case group")
                if "group_id" in row and row["group_id"] != group:
                    raise ValueError("Explicit case-group mapping conflicts with review group provenance.")
                row["group_id"] = group
            validate_row(row, feature_set)
            key = row["case_id"], row["instance_id"]
            if key in rows and rows[key] != row:
                raise ValueError(f"Conflicting reviews for {key}; preserve and resolve source provenance explicitly.")
            rows[key] = row
    if not rows:
        raise ValueError("No labelled candidates supplied.")
    return [rows[key] for key in sorted(rows)], provenance


def partition_rows(
    rows: list[dict], split: dict[str, list[str]], feature_set: str, provenance: list[dict],
) -> dict[str, list[dict]]:
    declared = {case: name for name, cases in split.items() for case in cases}
    reviewed = {row["case_id"] for row in rows}
    known = set(reviewed)
    for item in provenance:
        for case in item.get("metadata", {}).get("cases", []):
            known.add(nonempty_string(case.get("case_id"), "Provenance case_id"))
    if set(declared) != known:
        raise ValueError("Split must include exactly known reviewed/manifest cases, including empty cases.")
    groups: dict[str, str] = {}
    cases: dict[str, str] = {}
    fingerprints: set[str] = set()
    identifiers: set[tuple[str, str]] = set()
    result: dict[str, list[dict]] = {name: [] for name in split}
    for row in rows:
        validate_row(row, feature_set)
        case, group = row["case_id"], row["group_id"]
        name = declared[case]
        key = case, row["instance_id"]
        if key in identifiers:
            raise ValueError("Duplicate candidate rows must be resolved before fitting.")
        identifiers.add(key)
        if case in cases and cases[case] != group:
            raise ValueError("One case cannot have multiple patient/generation groups.")
        if group in groups and groups[group] != name:
            raise ValueError("Patient/generation group leakage across partitions.")
        if row["fingerprint"] in fingerprints:
            raise ValueError("Duplicate candidate fingerprint; possible leakage or duplicate candidate.")
        fingerprints.add(row["fingerprint"])
        cases[case], groups[group] = group, name
        result[name].append(row)
    return result


def arrays(rows: list[dict], feature_set: str) -> tuple[FloatArray, FloatArray]:
    names = FEATURE_NAMES if feature_set == "base" else EXTENDED_FEATURE_NAMES
    if not rows:
        return np.empty((0, len(names))), np.empty(0)
    return (
        feature_matrix([validate_row(row, feature_set) for row in rows], names),
        np.asarray([row["label"] == "confirmed" for row in rows], dtype=np.float64),
    )


def inference_contract(rows: list[dict]) -> dict | None:
    keys = ("detector_sha256", "extractor_sha256")
    if not any(key in row for row in rows for key in keys):
        return None
    sources = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("detector.py", "candidate_patches.py", "learning.py")
    }
    if any(
        row.get(key) != sources[name]
        for row in rows
        for key, name in zip(keys, ("detector.py", "candidate_patches.py"))
    ):
        raise ValueError("Training candidate extraction source drift; re-extract with the frozen source.")
    return {"source_sha256": sources, "preprocessing": preprocessing_metadata()}


def source_counts(rows: list[dict]) -> dict:
    return {
        source: {
            "rows": sum(source_category(row) == source for row in rows),
            "confirmed": sum(source_category(row) == source and row["label"] == "confirmed" for row in rows),
            "rejected": sum(source_category(row) == source and row["label"] == "rejected" for row in rows),
        }
        for source in DEFAULT_SOURCE_WEIGHTS
    }


def training_weights(rows: list[dict], source_weights: dict[str, float]) -> FloatArray:
    if set(source_weights) != set(DEFAULT_SOURCE_WEIGHTS):
        raise ValueError("Source weights must declare analytic, expert and pseudo.")
    for weight in source_weights.values():
        if not 0 <= finite_number(weight, "Source weight") <= 1:
            raise ValueError("Source weights must lie between zero and one.")
    weights = np.asarray([source_weights[source_category(row)] for row in rows], dtype=np.float64)
    labels = np.asarray([row["label"] == "confirmed" for row in rows])
    total = weights.sum()
    for label in (False, True):
        selected = labels == label
        mass = weights[selected].sum()
        if mass <= 0:
            raise ValueError("Training needs both classes with positive source weight.")
        weights[selected] *= total / (2 * mass)
    return weights


def metrics(labels: FloatArray, scores: FloatArray, threshold: float) -> dict:
    if (labels.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all()
            or not np.isin(labels, [0, 1]).all() or np.any((scores < 0) | (scores > 1))):
        raise ValueError("Metrics require aligned binary labels and finite probability scores.")
    unit_interval(threshold, "Metric threshold")
    predicted = scores >= threshold
    positive = labels == 1
    tp, fp = int(np.sum(predicted & positive)), int(np.sum(predicted & ~positive))
    fn, tn = int(np.sum(~predicted & positive)), int(np.sum(~predicted & ~positive))
    count = len(labels)
    both = set(labels.tolist()) == {0, 1}
    clipped = np.clip(scores, 1e-7, 1 - 1e-7)
    bins = []
    ece = 0.0
    for index in range(10):
        mask = (scores >= index / 10) & ((scores < (index + 1) / 10) if index < 9 else (scores <= 1))
        size = int(mask.sum())
        mean = float(scores[mask].mean()) if size else None
        observed = float(labels[mask].mean()) if size else None
        bins.append({"lower": index / 10, "upper": (index + 1) / 10,
                     "count": size, "mean_score": mean, "observed_positive_fraction": observed})
        if mean is not None and observed is not None:
            ece += size / count * abs(mean - observed)
    return {
        "reviewed_candidates": count, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "prevalence": float(labels.mean()) if count else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "retained_fraction": float(predicted.mean()) if count else None,
        "average_precision": float(average_precision_score(labels, scores)) if np.any(positive) else None,
        "roc_auc": float(roc_auc_score(labels, scores)) if both else None,
        "undefined": {
            "average_precision": None if np.any(positive) else "No positive labels.",
            "roc_auc": None if both else "Requires both classes.",
            "recall": None if np.any(positive) else "No positive labels.",
        },
        "calibration": {
            "brier_score": float(np.mean((scores - labels) ** 2)) if count else None,
            "log_loss": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)))
            if count else None,
            "ece_10_equal_width_bins": ece if count else None, "reliability_bins": bins,
            "scope": "Observed candidate-label diagnostics, not clinical calibration.",
        },
    }


def select_threshold(
    validation_y: FloatArray, validation_scores: FloatArray,
    training_y: FloatArray, training_scores: FloatArray,
    minimum_recall: float = 0.95, minimum_training_recall: float = 1.0,
) -> tuple[float, dict]:
    unit_interval(minimum_recall, "Minimum validation recall")
    unit_interval(minimum_training_recall, "Minimum training recall")
    if not len(validation_y):
        raise ValueError("Threshold selection needs nonempty validation candidates.")
    if set(validation_y.tolist()) != {0, 1}:
        return 0.0, {
            "status": "keep_all_single_class_validation",
            "reason": "Validation cannot establish discrimination with only one class.",
        }
    thresholds = sorted({0.0, 1.0, *validation_scores.tolist(), *training_scores[training_y == 1].tolist()})
    choices = []
    for threshold in thresholds:
        val_keep, train_keep = validation_scores >= threshold, training_scores >= threshold
        recall = float(val_keep[validation_y == 1].mean())
        training_recall = float(train_keep[training_y == 1].mean())
        if recall + 1e-12 < minimum_recall or training_recall + 1e-12 < minimum_training_recall:
            continue
        precision = float(validation_y[val_keep].mean()) if val_keep.any() else 0.0
        choices.append((precision, recall, threshold, training_recall))
    if not choices:
        raise ValueError("No threshold satisfies the declared validation/training recall constraints.")
    precision, recall, threshold, training_recall = max(choices)
    return threshold, {
        "status": "selected", "objective": "maximum validation precision, then recall, then threshold",
        "validation_precision": precision, "validation_recall": recall, "training_recall": training_recall,
    }


@dataclass
class LogisticBaseline:
    mean: FloatArray
    scale: FloatArray
    weights: FloatArray
    bias: float

    def scores(self, matrix: FloatArray) -> FloatArray:
        return sigmoid(((matrix[:, :len(FEATURE_NAMES)] - self.mean) / self.scale) @ self.weights + self.bias)


def fit_logistic(matrix: FloatArray, labels: FloatArray, weights: FloatArray) -> LogisticBaseline:
    x = matrix[:, :len(FEATURE_NAMES)]
    mean = np.average(x, axis=0, weights=weights)
    scale = np.maximum(np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights)), 1e-6)
    normalized = (x - mean) / scale
    normalized_weights = weights / weights.sum()

    def objective(parameters: FloatArray) -> tuple[float, FloatArray]:
        coefficients, bias = parameters[:-1], parameters[-1]
        logits = normalized @ coefficients + bias
        residual = normalized_weights * (sigmoid(logits) - labels)
        loss = float(np.sum(normalized_weights * (np.logaddexp(0, logits) - labels * logits))
                     + 0.05 * np.sum(coefficients**2))
        gradient = np.r_[normalized.T @ residual + 0.1 * coefficients, residual.sum()]
        return loss, gradient

    fitted = minimize(objective, np.zeros(len(FEATURE_NAMES) + 1), jac=True, method="L-BFGS-B")
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise ValueError(f"Logistic comparator optimization failed: {fitted.message}")
    return LogisticBaseline(mean, scale, fitted.x[:-1], float(fitted.x[-1]))


def fit_calibration(scores: FloatArray, labels: FloatArray, rows: list[dict]) -> PlattCalibration:
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Platt calibration needs both classes in its dedicated validation cases.")
    logits = probability_logits(scores)

    def objective(parameters: FloatArray) -> tuple[float, FloatArray]:
        transformed = parameters[0] * logits + parameters[1]
        residual = sigmoid(transformed) - labels
        loss = float(np.mean(np.logaddexp(0, transformed) - labels * transformed)
                     + 1e-6 * np.sum(parameters**2))
        gradient = np.asarray([np.mean(residual * logits), np.mean(residual)]) + 2e-6 * parameters
        return loss, gradient

    fitted = minimize(
        objective, np.asarray([1.0, 0.0]), jac=True, method="L-BFGS-B", bounds=((0, 1000), (-1000, 1000)),
    )
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise ValueError(f"Platt calibration optimization failed: {fitted.message}")
    return PlattCalibration(float(fitted.x[0]), float(fitted.x[1]), {
        "partition": "validation_calibration", "case_ids": sorted({row["case_id"] for row in rows}),
        "group_ids": sorted({row["group_id"] for row in rows}), "records_sha256": json_sha256(rows),
        "source_counts": source_counts(rows), "weighting": "unweighted_observed_prevalence",
        "regularization": 1e-6, "logit_clip": 1e-7,
        "scope": "Observed candidate-label calibration; not clinical calibration.",
    })


def lineage(rows: list[dict]) -> list[dict]:
    return [{
        "case_id": row["case_id"], "instance_id": row["instance_id"], "group_id": row["group_id"],
        "labeller": row["labeller"], "source_category": source_category(row),
        "fingerprint": row["fingerprint"], "record_sha256": json_sha256(row),
        "prior_exposure": row.get("prior_exposure"),
    } for row in rows]


def train(
    rows: list[dict], split: dict[str, list[str]], *,
    algorithm: str = "gradient_boosting", feature_set: str = "base",
    source_weights: dict[str, float] | None = None,
    minimum_recall: float = 0.95, minimum_training_recall: float = 1.0,
    calibration_cases: list[str] | None = None, evaluate_test: bool = False,
    provenance: list[dict] | None = None, prior_exposure: dict | None = None,
) -> tuple[TreeModel, dict]:
    require_training()
    if feature_set not in ("base", "extended"):
        raise ValueError("Feature selection must be base or extended.")
    unit_interval(minimum_recall, "Minimum validation recall")
    unit_interval(minimum_training_recall, "Minimum training recall")
    split = validate_split(split)
    partitions = partition_rows(rows, split, feature_set, provenance or [])
    calibration_cases = calibration_cases or []
    if (len(set(calibration_cases)) != len(calibration_cases)
            or not set(calibration_cases).issubset(split["validation"])
            or set(calibration_cases) == set(split["validation"])):
        raise ValueError("Calibration cases must be a distinct strict subset of validation cases.")
    calibration_rows = [row for row in partitions["validation"] if row["case_id"] in calibration_cases]
    validation_rows = [row for row in partitions["validation"] if row["case_id"] not in calibration_cases]
    if {row["group_id"] for row in calibration_rows} & {row["group_id"] for row in validation_rows}:
        raise ValueError("Calibration and threshold-selection groups must be disjoint.")
    if calibration_cases and {row["case_id"] for row in calibration_rows} != set(calibration_cases):
        raise ValueError("Every calibration case must have labelled candidates.")
    source_weights = dict(DEFAULT_SOURCE_WEIGHTS if source_weights is None else source_weights)
    all_weights = training_weights(partitions["train"], source_weights)
    active = all_weights > 0
    fit_rows = [row for row, keep in zip(partitions["train"], active) if keep]
    extraction_contract = inference_contract(fit_rows + validation_rows + calibration_rows)
    train_x, train_y = arrays(fit_rows, feature_set)
    fit_weights = all_weights[active]
    val_x, val_y = arrays(validation_rows, feature_set)
    if not len(val_y):
        raise ValueError("Threshold selection needs nonempty validation candidates.")
    estimator = make_estimator(algorithm)
    estimator.fit(train_x, train_y.astype(np.int64), sample_weight=fit_weights)
    names = FEATURE_NAMES if feature_set == "base" else EXTENDED_FEATURE_NAMES
    source_hashes = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("detector.py", "learning.py", "tabular_learning.py", "train_trees.py")
    }
    model = export_estimator(estimator, names, split, {
        "usage": "scores_only", "source_sha256": source_hashes, "fit_partition": "train",
        "fit_case_ids": sorted({row["case_id"] for row in fit_rows}),
        "fit_group_ids": sorted({row["group_id"] for row in fit_rows}),
        "fit_records_sha256": json_sha256(fit_rows), "source_weights": source_weights,
        "class_weighting": "Equal total source-weighted class mass, training rows only.",
        "estimator_parameters": estimator.get_params(deep=False),
        "versions": {name: importlib.metadata.version(name) for name in ("scikit-learn", "numpy", "scipy")},
        "python": platform.python_version(), "limitations": LIMITATIONS,
        "threshold_selected_on": "validation_threshold", "minimum_validation_recall": minimum_recall,
        "minimum_training_recall": minimum_training_recall,
        "threshold_case_ids": sorted(set(split["validation"]) - set(calibration_cases)),
        "prior_exposure": prior_exposure or {},
        "inference_contract": extraction_contract,
    })
    verification_x = np.concatenate((train_x, val_x))
    exported = model.raw_scores(verification_x)
    if isinstance(estimator, RandomForestClassifier):
        training_jobs = estimator.n_jobs
        expected = estimator.set_params(n_jobs=1).predict_proba(verification_x)[:, 1]
        estimator.set_params(n_jobs=training_jobs)
    else:
        expected = estimator.predict_proba(verification_x)[:, 1]
    difference = float(np.max(np.abs(exported - expected)))
    if not np.allclose(exported, expected, rtol=1e-12, atol=1e-12):
        raise ValueError(f"NumPy export parity failed; maximum probability error {difference}.")
    model.metadata["export_parity"] = {
        "max_absolute_error": difference, "rows": len(verification_x),
        "partitions": ["active_train", "validation_threshold"], "atol": 1e-12, "rtol": 1e-12,
        "sklearn_prediction_jobs": 1,
    }
    if calibration_cases:
        cal_x, cal_y = arrays(calibration_rows, feature_set)
        model.calibration = fit_calibration(model.raw_scores(cal_x), cal_y, calibration_rows)
    train_scores, val_scores = model.scores(train_x), model.scores(val_x)
    model.threshold, threshold_report = select_threshold(
        val_y, val_scores, train_y, train_scores, minimum_recall, minimum_training_recall,
    )
    model.metadata["threshold_selection"] = threshold_report
    logistic = fit_logistic(train_x, train_y, fit_weights)
    logistic_threshold, logistic_selection = select_threshold(
        val_y, logistic.scores(val_x), train_y, logistic.scores(train_x), minimum_recall, minimum_training_recall,
    )
    report: dict = {
        "schema_version": 1, "scope": "candidate_reviews_only", "limitations": LIMITATIONS,
        "algorithm": algorithm, "feature_names": names, "split": split, "threshold": model.threshold,
        "threshold_selection": threshold_report, "minimum_validation_recall": minimum_recall,
        "minimum_training_recall": minimum_training_recall,
        "model_metadata": model.metadata, "input_provenance": provenance or [],
        "prior_exposure": prior_exposure or {}, "source_weights": source_weights,
        "effective_training_source_counts": source_counts(fit_rows),
        "effective_training_class_mass": {
            "confirmed": float(fit_weights[train_y == 1].sum()), "rejected": float(fit_weights[train_y == 0].sum()),
        },
        "logistic_comparator": {
            "description": "13-feature train-only weighted logistic; same L2 coefficient as legacy CandidateModel.",
            "feature_names": FEATURE_NAMES, "mean": logistic.mean.tolist(), "scale": logistic.scale.tolist(),
            "weights": logistic.weights.tolist(), "bias": logistic.bias,
            "threshold": logistic_threshold, "threshold_selection": logistic_selection,
        },
        "calibration": model.calibration.provenance if model.calibration else None,
        "partitions": {},
    }
    reporting = {**partitions, "validation_threshold": validation_rows}
    if calibration_cases:
        reporting["validation_calibration"] = calibration_rows
    for name, selected in reporting.items():
        if name == "test" and not evaluate_test:
            report["partitions"][name] = {
                "status": "sealed_not_scored", "case_ids": split["test"],
                "records_sha256": json_sha256(selected),
            }
            continue
        x, y = arrays(selected, feature_set)
        scores = model.scores(x)
        report["partitions"][name] = {
            "status": "evaluated_once" if name == "test" else "development",
            "candidate_count": len(selected), "source_counts": source_counts(selected),
            "labeller_counts": dict(Counter(row["labeller"] for row in selected)),
            "classifier": metrics(y, scores, model.threshold),
            "keep_all_baseline": metrics(y, np.ones(len(y)), 0.0),
            "logistic_baseline": metrics(y, logistic.scores(x), logistic_threshold),
            "by_source": {
                source: metrics(
                    y[[source_category(row) == source for row in selected]],
                    scores[[source_category(row) == source for row in selected]], model.threshold,
                )
                for source in DEFAULT_SOURCE_WEIGHTS
            },
            "candidate_provenance": lineage(selected),
        }
    model.validate()
    return model, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reviews", nargs="+", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--algorithm", choices=("gradient_boosting", "random_forest"), default="gradient_boosting")
    parser.add_argument("--features", choices=("base", "extended"), default="base")
    parser.add_argument("--pseudo-weight", type=float, default=0.3)
    parser.add_argument("--analytic-weight", type=float, default=1.0)
    parser.add_argument("--expert-weight", type=float, default=1.0)
    parser.add_argument("--minimum-recall", type=float, default=0.95)
    parser.add_argument("--minimum-training-recall", type=float, default=1.0)
    parser.add_argument("--calibration-cases", nargs="+", default=[])
    parser.add_argument("--case-groups", type=Path, help="Explicit case ID -> patient group mapping for legacy rows.")
    parser.add_argument("--prior-exposure", type=Path, help="Case ID -> historical inspection/training provenance.")
    parser.add_argument("--evaluate-test", action="store_true", help="Unseal final results only after freezing choices.")
    args = parser.parse_args()
    try:
        require_training()
        if args.model.resolve() == args.report.resolve() or args.model.exists() or args.report.exists():
            raise ValueError("Model/report outputs must be distinct new paths; never overwrite frozen artifacts.")
        rows, provenance = load_reviews(
            args.reviews, args.features, read_json(args.case_groups) if args.case_groups else None,
        )
        model, report = train(
            rows, read_json(args.split), algorithm=args.algorithm, feature_set=args.features,
            source_weights={"pseudo": args.pseudo_weight, "analytic": args.analytic_weight, "expert": args.expert_weight},
            minimum_recall=args.minimum_recall, minimum_training_recall=args.minimum_training_recall,
            calibration_cases=args.calibration_cases, evaluate_test=args.evaluate_test, provenance=provenance,
            prior_exposure=read_json(args.prior_exposure) if args.prior_exposure else None,
        )
        report["invocation"] = sys.argv
        report["split_sha256"] = hashlib.sha256(args.split.read_bytes()).hexdigest()
        for name, path in (("case_groups", args.case_groups), ("prior_exposure", args.prior_exposure)):
            if path is not None:
                report[f"{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        model.save(args.model)
        report["model_file_sha256"] = hashlib.sha256(args.model.read_bytes()).hexdigest()
        write_json(args.report, report)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        parser.error(str(error))
    if args.evaluate_test:
        warnings.warn("Final test was explicitly unsealed. Do not use these results to tune this experiment.", stacklevel=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
