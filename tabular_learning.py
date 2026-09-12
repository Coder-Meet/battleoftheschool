"""Versioned candidate tree scores using NumPy only; never removes candidates.

Inputs follow the declared feature order, in the detector's physical units.
Tree routing reproduces sklearn's float32 input conversion and <= comparisons.
Scores describe candidate labels, not clinical probabilities or proposal recall.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt

FEATURE_NAMES = [
    "radius_mm", "mean_vesselness", "evidence_score", "path_length_mm",
    "seed_distance_mm", "tortuosity", "path_hu_relative", "bone_distance_mm",
    "parent_angle_degrees", "arc_position", "native_spacing_mm", "connector_gap",
    "candidate_volume_mm3",
]
EXTRA_FEATURE_NAMES = [
    "ostium_local_contrast", "patch_vesselness_std", "parent_wall_curvature",
]
EXTENDED_FEATURE_NAMES = FEATURE_NAMES + EXTRA_FEATURE_NAMES
FloatArray = npt.NDArray[np.float64]
MAX_MODEL_BYTES = 16 * 1024 * 1024
PREPROCESSING = {
    "input": "finite physical candidate features in feature_names order",
    "tree_input_dtype": "float32",
    "comparison_dtype": "float64",
    "normalization": "none",
    "missing_features": "error",
}


def finite_number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite JSON number.")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} exceeds the finite numeric range.") from error
    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite JSON number.")
    return number


def unit_interval(value: object, name: str) -> float:
    number = finite_number(value, name)
    if not 0 <= number <= 1:
        raise ValueError(f"{name} must lie between zero and one.")
    return number


def nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")
    return value


def validate_split(value: dict) -> dict[str, list[str]]:
    if not isinstance(value, dict) or set(value) != {"train", "validation", "test"}:
        raise ValueError("Split must declare train, validation and test case IDs.")
    result: dict[str, list[str]] = {}
    seen: set[str] = set()
    for name in ("train", "validation", "test"):
        cases = value[name]
        if not isinstance(cases, list) or not cases:
            raise ValueError("Every partition needs a nonempty list of case IDs.")
        for case in cases:
            nonempty_string(case, "Case ID")
            if case in seen:
                raise ValueError("Patient/case leakage: split partitions must be disjoint.")
            seen.add(case)
        result[name] = list(cases)
    return result


def feature_order(names: list[str]) -> list[str]:
    if names != FEATURE_NAMES and names != EXTENDED_FEATURE_NAMES:
        raise ValueError("Unsupported feature order; expected the base 13 or extended 16 features.")
    return list(names)


def feature_matrix(matrix: npt.ArrayLike, names: list[str]) -> FloatArray:
    feature_order(names)
    raw = np.asarray(matrix)
    if raw.dtype.kind not in "fiu":
        raise ValueError("Features must be a finite numeric matrix.")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(names) or not np.isfinite(array).all():
        raise ValueError(f"Features must be a finite matrix with {len(names)} columns in declared order.")
    if np.any(np.abs(array) > np.finfo(np.float32).max):
        raise ValueError("Features must be representable as finite float32 values.")
    return array


def sigmoid(values: FloatArray) -> FloatArray:
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1 + exponent)
    return result


def probability_logits(scores: FloatArray) -> FloatArray:
    clipped = np.clip(scores, 1e-7, 1 - 1e-7)
    return np.log(clipped) - np.log1p(-clipped)


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant: {value}")


def read_json(path: Path, maximum_bytes: int | None = None) -> dict:
    with path.open("rb") as stream:
        contents = stream.read(maximum_bytes + 1) if maximum_bytes is not None else stream.read()
    if maximum_bytes is not None and len(contents) > maximum_bytes:
        raise ValueError("JSON artifact exceeds the size limit.")
    try:
        value = json.loads(contents, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except (RecursionError, UnicodeError) as error:
        raise ValueError("Invalid JSON artifact.") from error
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value


def write_json(path: Path, value: dict, maximum_bytes: int | None = None) -> None:
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if maximum_bytes is not None and len(encoded.encode("utf-8")) > maximum_bytes:
        raise ValueError("JSON artifact exceeds the size limit.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")


def json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class Tree:
    children_left: list[int]
    children_right: list[int]
    feature: list[int]
    threshold: list[float]
    value: list[float]

    def validate(self, dimensions: int, probability: bool) -> None:
        arrays = (self.children_left, self.children_right, self.feature, self.threshold, self.value)
        if any(not isinstance(a, list) for a in arrays):
            raise ValueError("Tree arrays must be lists.")
        count = len(self.feature)
        if not 1 <= count <= 4095 or any(len(a) != count for a in arrays):
            raise ValueError("Tree arrays must have equal, bounded, nonzero dimensions.")
        for array in (self.children_left, self.children_right, self.feature):
            if any(type(v) is not int for v in array):
                raise ValueError("Tree indices must be integers.")
        for value in self.threshold:
            finite_number(value, "Tree threshold")
        for value in self.value:
            finite_number(value, "Tree value")
            if probability:
                unit_interval(value, "Class probability")
        pending = [(0, 0)]
        visited: set[int] = set()
        while pending:
            node, depth = pending.pop()
            if node in visited:
                raise ValueError("Trees must be acyclic with exactly one parent per non-root node.")
            if not 0 <= node < count or depth > 64:
                raise ValueError("Invalid tree child index or excessive depth.")
            visited.add(node)
            left, right = self.children_left[node], self.children_right[node]
            if left == right == -1:
                if self.feature[node] != -2 or self.threshold[node] != -2:
                    raise ValueError("Leaf feature and threshold must be the -2 sentinel.")
            else:
                if not (0 <= left < count and 0 <= right < count and 0 <= self.feature[node] < dimensions):
                    raise ValueError("Invalid tree child or feature index.")
                pending.extend(((left, depth + 1), (right, depth + 1)))
        if len(visited) != count:
            raise ValueError("Every tree node must be reachable from the root.")

    def predict(self, matrix: FloatArray) -> FloatArray:
        left = np.asarray(self.children_left)
        right = np.asarray(self.children_right)
        features = np.asarray(self.feature)
        thresholds = np.asarray(self.threshold, dtype=np.float64)
        nodes = np.zeros(len(matrix), dtype=np.int64)
        active = np.flatnonzero(left[nodes] != -1)
        while len(active):
            current = nodes[active]
            go_left = matrix[active, features[current]] <= thresholds[current]
            nodes[active] = np.where(go_left, left[current], right[current])
            active = active[left[nodes[active]] != -1]
        return np.asarray(self.value, dtype=np.float64)[nodes]


@dataclass
class PlattCalibration:
    slope: float
    intercept: float
    provenance: dict

    def validate(self, split: dict[str, list[str]]) -> None:
        slope = finite_number(self.slope, "Calibration slope")
        intercept = finite_number(self.intercept, "Calibration intercept")
        if not 0 <= slope <= 1000 or abs(intercept) > 1000:
            raise ValueError("Calibration coefficients exceed the supported range.")
        if not isinstance(self.provenance, dict):
            raise ValueError("Calibration requires explicit fitting provenance.")
        if self.provenance.get("partition") != "validation_calibration":
            raise ValueError("Calibration must declare its dedicated validation fitting partition.")
        cases = self.provenance.get("case_ids")
        if not isinstance(cases, list) or not cases or any(not isinstance(c, str) for c in cases):
            raise ValueError("Calibration must declare fitting cases.")
        if len(set(cases)) != len(cases) or not set(cases) < set(split["validation"]):
            raise ValueError("Calibration cases must be a strict subset of validation cases.")
        digest = nonempty_string(self.provenance.get("records_sha256"), "Calibration records hash")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Calibration records hash must be SHA256.")
        groups = self.provenance.get("group_ids")
        if not isinstance(groups, list) or not groups:
            raise ValueError("Calibration must declare fitting groups.")
        for group in groups:
            nonempty_string(group, "Calibration group")
        if len(set(groups)) != len(groups):
            raise ValueError("Calibration fitting groups must be unique.")
        counts = self.provenance.get("source_counts")
        if not isinstance(counts, dict) or set(counts) != {"analytic", "expert", "pseudo"}:
            raise ValueError("Calibration must declare source counts.")
        for count in counts.values():
            if (not isinstance(count, dict) or set(count) != {"rows", "confirmed", "rejected"}
                    or any(type(v) is not int or v < 0 for v in count.values())
                    or count["rows"] != count["confirmed"] + count["rejected"]):
                raise ValueError("Invalid calibration source counts.")
        if any(sum(count[label] for count in counts.values()) == 0 for label in ("confirmed", "rejected")):
            raise ValueError("Calibration fitting counts must include both classes.")
        if self.provenance.get("weighting") != "unweighted_observed_prevalence":
            raise ValueError("Calibration must preserve fitting prevalence.")

    def scores(self, scores: FloatArray) -> FloatArray:
        return sigmoid(self.slope * probability_logits(scores) + self.intercept)


@dataclass
class TreeModel:
    algorithm: str
    feature_names: list[str]
    trees: list[Tree]
    threshold: float
    split: dict[str, list[str]]
    metadata: dict
    learning_rate: float = 1.0
    initial_logit: float = 0.0
    calibration: PlattCalibration | None = None

    def validate(self) -> None:
        if self.algorithm not in ("gradient_boosting", "random_forest"):
            raise ValueError("Unsupported tree algorithm.")
        feature_order(self.feature_names)
        unit_interval(self.threshold, "Model threshold")
        validate_split(self.split)
        if not isinstance(self.metadata, dict):
            raise ValueError("Model metadata must be an object.")
        try:
            json.dumps(self.metadata, allow_nan=False)
        except (TypeError, ValueError, RecursionError) as error:
            raise ValueError("Model metadata must contain finite JSON values.") from error
        rate = finite_number(self.learning_rate, "Learning rate")
        base = finite_number(self.initial_logit, "Initial logit")
        if not 0 < rate <= 1 or base != 0:
            raise ValueError("Only positive learning rates <= 1 and zero initialization are supported.")
        if self.algorithm == "random_forest" and rate != 1:
            raise ValueError("Random forest learning rate must be one.")
        if not isinstance(self.trees, list) or not 1 <= len(self.trees) <= 256:
            raise ValueError("Model must contain between 1 and 256 trees.")
        for tree in self.trees:
            if not isinstance(tree, Tree):
                raise ValueError("Invalid tree object.")
            tree.validate(len(self.feature_names), self.algorithm == "random_forest")
        bound = sum(max(abs(v) for v in tree.value) for tree in self.trees) * rate
        if not np.isfinite(bound):
            raise ValueError("Tree ensemble can overflow.")
        if self.calibration is not None:
            self.calibration.validate(self.split)

    def raw_scores(self, matrix: npt.ArrayLike) -> FloatArray:
        self.validate()
        array = feature_matrix(matrix, self.feature_names).astype(np.float32).astype(np.float64)
        total = np.zeros(len(array), dtype=np.float64)
        if self.algorithm == "random_forest":
            for tree in self.trees:
                total += tree.predict(array)
            return total / len(self.trees)
        for tree in self.trees:
            total += self.learning_rate * tree.predict(array)
        return sigmoid(total + self.initial_logit)

    def scores(self, matrix: npt.ArrayLike) -> FloatArray:
        scores = self.raw_scores(matrix)
        return self.calibration.scores(scores) if self.calibration is not None else scores

    def to_dict(self) -> dict:
        self.validate()
        return {
            "schema_version": 1, "model_type": "branchseed_candidate_trees",
            "classes": [0, 1], "positive_class": 1, "preprocessing": PREPROCESSING,
            **asdict(self),
        }

    def save(self, path: Path) -> None:
        payload = self.to_dict()
        payload["model_sha256"] = json_sha256(payload)
        write_json(path, payload, MAX_MODEL_BYTES)

    @classmethod
    def load(cls, path: Path) -> "TreeModel":
        value = read_json(path, MAX_MODEL_BYTES)
        expected = {
            "schema_version", "model_type", "classes", "positive_class", "preprocessing",
            "algorithm", "feature_names", "trees", "threshold", "split", "metadata",
            "learning_rate", "initial_logit", "calibration", "model_sha256",
        }
        if set(value) != expected:
            raise ValueError("Invalid model fields.")
        digest = value.pop("model_sha256")
        if not isinstance(digest, str) or digest != json_sha256(value):
            raise ValueError("Model SHA256 integrity check failed.")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("Unsupported model schema.")
        if value["model_type"] != "branchseed_candidate_trees" or value["preprocessing"] != PREPROCESSING:
            raise ValueError("Unsupported model type or preprocessing.")
        classes = value["classes"]
        if (not isinstance(classes, list) or classes != [0, 1]
                or any(type(c) is not int for c in classes)
                or type(value["positive_class"]) is not int or value["positive_class"] != 1):
            raise ValueError("Only ordered binary classes [0, 1] with positive class 1 are supported.")
        if not isinstance(value["trees"], list):
            raise ValueError("Trees must be an array.")
        trees = []
        for tree in value["trees"]:
            if not isinstance(tree, dict) or set(tree) != {
                "children_left", "children_right", "feature", "threshold", "value",
            }:
                raise ValueError("Invalid tree fields.")
            trees.append(Tree(**tree))
        calibration = value["calibration"]
        if calibration is not None and (
            not isinstance(calibration, dict) or set(calibration) != {"slope", "intercept", "provenance"}
        ):
            raise ValueError("Invalid calibration fields.")
        result = cls(
            value["algorithm"], value["feature_names"], trees, value["threshold"],
            value["split"], value["metadata"], value["learning_rate"], value["initial_logit"],
            PlattCalibration(**calibration) if calibration is not None else None,
        )
        result.validate()
        return result
