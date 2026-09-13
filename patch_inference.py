"""Optional CPU ONNX candidate scores and declared probability blending.

Inputs are raw candidate_patches arrays, not HU volumes or standardized arrays.
The ONNX graph contains the train-only normalizer and sigmoid. Scores describe
candidate labels, not clinical probabilities. No detector decisions are changed.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import numpy.typing as npt

try:
    import onnxruntime as ort
except ImportError:
    INFERENCE_AVAILABLE = False
else:
    INFERENCE_AVAILABLE = True

from candidate_patches import EXTRA_FEATURE_NAMES, preprocessing_metadata
from tabular_learning import (
    FEATURE_NAMES, TreeModel, feature_matrix, finite_number, json_sha256,
    nonempty_string, read_json, unit_interval, validate_split, write_json,
)

PatchArray = npt.NDArray[np.float32]
Scores = npt.NDArray[np.float64]
PARITY_ATOL = 2e-6
PARITY_RTOL = 1e-5
MAX_MODEL_BYTES = 8 * 1024 * 1024


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha256_string(value: object) -> str:
    text = nonempty_string(value, "SHA256")
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise ValueError("Expected a lowercase SHA256 digest.")
    return text


def contract(metadata: dict) -> dict:
    """Validate the extraction/data contract independently of candidate labels."""
    prep = metadata.get("preprocessing")
    if not isinstance(prep, dict):
        raise ValueError("Missing physical preprocessing.")
    size = prep.get("size")
    spacing = finite_number(prep.get("spacing_mm"), "Patch spacing")
    if type(size) is not int or prep != preprocessing_metadata(size, spacing):
        raise ValueError("Unsupported physical/channel preprocessing contract.")
    if metadata.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Feature order mismatch.")
    if metadata.get("extra_feature_names") != EXTRA_FEATURE_NAMES:
        raise ValueError("Extended feature order mismatch.")
    sources = metadata.get("source_sha256")
    if not isinstance(sources, dict):
        raise ValueError("Missing source hashes.")
    return {
        "preprocessing": prep, "feature_names": FEATURE_NAMES,
        "extra_feature_names": EXTRA_FEATURE_NAMES,
        "manifest_sha256": sha256_string(metadata.get("manifest_sha256")),
        "source_sha256": {
            name: sha256_string(sources.get(name))
            for name in ("detector.py", "candidate_patches.py")
        },
    }


def validate_patches(patches: PatchArray, size: int) -> None:
    if (
        patches.dtype != np.float32 or patches.ndim != 4 or patches.shape[1:] != (12, size, size)
        or not np.isfinite(patches).all() or np.any(patches < 0) or np.any(patches > 1)
    ):
        raise ValueError("Patches must be finite float32 (N,12,size,size), bounded [0,1].")


def candidate_identity(row: dict) -> dict:
    names = ("case_id", "instance_id", "fingerprint", "group_id")
    identity = {key: nonempty_string(row.get(key), key) for key in names}
    features = row.get("features")
    if not isinstance(features, list):
        raise ValueError("Missing candidate features.")
    vector = feature_matrix([features], FEATURE_NAMES)[0].tolist()
    extra = row.get("extra_features")
    if not isinstance(extra, dict) or set(extra) != set(EXTRA_FEATURE_NAMES):
        raise ValueError("Missing extended features; zero imputation is prohibited.")
    identity["features_sha256"] = json_sha256({
        "features": vector,
        "extra_features": {name: finite_number(extra[name], name) for name in EXTRA_FEATURE_NAMES},
    })
    inputs = row.get("input_sha256")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("Missing candidate input hashes.")
    for value in inputs.values():
        sha256_string(value)
    identity["input_sha256"] = json_sha256(inputs)
    for name in ("detector_sha256", "extractor_sha256"):
        identity[name] = sha256_string(row.get(name))
    return identity


@dataclass(frozen=True)
class ScoreBatch:
    scores: Scores
    identities: list[dict]
    contract: dict
    model_sha256: str

    def validate(self) -> None:
        if (
            self.scores.shape != (len(self.identities),) or not np.isfinite(self.scores).all()
            or np.any((self.scores < 0) | (self.scores > 1))
        ):
            raise ValueError("Scores must align with ordered identities and lie in [0,1].")
        contract(self.contract)
        sha256_string(self.model_sha256)
        keys = [(r["case_id"], r["instance_id"]) for r in self.identities]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate candidate score identities.")
        for row in self.identities:
            for key in ("case_id", "instance_id", "fingerprint", "group_id"):
                nonempty_string(row.get(key), key)
            for key in ("features_sha256", "input_sha256", "detector_sha256", "extractor_sha256"):
                sha256_string(row.get(key))
            sources = self.contract["source_sha256"]
            if (row["detector_sha256"] != sources["detector.py"]
                    or row["extractor_sha256"] != sources["candidate_patches.py"]):
                raise ValueError("Candidate source hashes differ from score contract.")


def score_batch(scores: Scores, rows: list[dict], metadata: dict, model_hash: str) -> ScoreBatch:
    result = ScoreBatch(
        np.asarray(scores, dtype=np.float64), [candidate_identity(r) for r in rows],
        contract(metadata), model_hash,
    )
    result.validate()
    return result


class PatchModel:
    """Load a verified graph/sidecar pair with a bounded CPU-only session."""

    def __init__(self, path: Path, metadata: dict, threads: int) -> None:
        if not INFERENCE_AVAILABLE:
            raise ValueError("Install requirements-cnn-inference.txt for optional CPU ONNX inference.")
        if type(threads) is not int or not 1 <= threads <= 4:
            raise ValueError("ONNX threads must be between one and four.")
        self.metadata = metadata
        self.threshold = unit_interval(metadata.get("threshold"), "Threshold")
        split = metadata.get("split")
        if not isinstance(split, dict):
            raise ValueError("Missing model split.")
        self.split = validate_split(split)
        self.feature_names = list(FEATURE_NAMES)
        self.contract = contract(metadata["contract"])
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            path.read_bytes(), sess_options=options, providers=["CPUExecutionProvider"],
        )
        if self.session.get_providers() != ["CPUExecutionProvider"]:
            raise ValueError("Only CPUExecutionProvider is permitted.")
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        size = self.contract["preprocessing"]["size"]
        if (len(inputs) != 1 or inputs[0].name != "patches" or inputs[0].type != "tensor(float)"
                or inputs[0].shape != ["batch", 12, size, size]
                or len(outputs) != 1 or outputs[0].name != "probabilities"
                or outputs[0].type != "tensor(float)" or outputs[0].shape != ["batch"]):
            raise ValueError("Unexpected ONNX input/output signature.")

    @classmethod
    def load(cls, path: Path, threads: int = 4) -> "PatchModel":
        metadata = read_json(path.with_suffix(".json"), MAX_MODEL_BYTES)
        if (metadata.get("schema_version") != 1 or metadata.get("model_type") != "patch_cnn_2_5d"
                or metadata.get("classes") != [0, 1] or metadata.get("positive_class") != 1
                or metadata.get("normalization_location") != "inside_onnx"
                or metadata.get("threshold_selected_on") != "validation"):
            raise ValueError("Unsupported patch model sidecar.")
        model_contract = contract(metadata.get("contract", {}))
        size = model_contract["preprocessing"]["size"]
        if (metadata.get("opset") != 17 or metadata.get("output_name") != "probabilities"
                or metadata.get("input") != {
                    "name": "patches", "dtype": "float32", "shape": ["batch", 12, size, size],
                } or metadata.get("scope") != "research_only"
                or metadata.get("clinical_accuracy_claim") is not False
                or metadata.get("recall_loss_budget") != 0):
            raise ValueError("Unsupported graph/preprocessing/research metadata.")
        sha256_string(metadata.get("state_sha256"))
        expected = metadata.get("sidecar_sha256")
        if expected != json_sha256({k: v for k, v in metadata.items() if k != "sidecar_sha256"}):
            raise ValueError("Sidecar integrity mismatch.")
        if path.stat().st_size > MAX_MODEL_BYTES or file_sha256(path) != metadata.get("onnx_sha256"):
            raise ValueError("ONNX size/integrity mismatch.")
        if not 0 < finite_number(metadata.get("parameter_count"), "Parameter count") < 1_000_000:
            raise ValueError("Model exceeds the small classifier parameter budget.")
        normalizer = metadata.get("normalizer")
        if not isinstance(normalizer, dict):
            raise ValueError("Missing channel normalizer.")
        mean, scale = np.asarray(normalizer.get("mean")), np.asarray(normalizer.get("scale"))
        if (mean.shape != (12,) or scale.shape != (12,) or mean.dtype.kind not in "fiu"
                or scale.dtype.kind not in "fiu" or not np.isfinite(mean).all()
                or not np.isfinite(scale).all() or np.any(scale < 1e-3)
                or normalizer.get("fit_partition") != "train"):
            raise ValueError("Invalid train-only channel normalizer.")
        return cls(path, metadata, threads)

    def scores(self, patches: PatchArray, metadata: dict, batch_size: int = 64) -> Scores:
        supplied = contract(metadata)
        if any(supplied[key] != self.contract[key] for key in (
            "preprocessing", "source_sha256", "feature_names", "extra_feature_names",
        )):
            raise ValueError("Inference extraction source/preprocessing differs from the model.")
        validate_patches(patches, self.contract["preprocessing"]["size"])
        if type(batch_size) is not int or not 1 <= batch_size <= 256:
            raise ValueError("Batch size must be between one and 256.")
        result = np.empty(len(patches), dtype=np.float64)
        for start in range(0, len(patches), batch_size):
            block = np.ascontiguousarray(patches[start:start + batch_size])
            output = self.session.run(["probabilities"], {"patches": block})[0]
            if output.shape != (len(block),) or not np.isfinite(output).all():
                raise ValueError("ONNX returned malformed scores.")
            result[start:start + len(block)] = output
        if np.any((result < 0) | (result > 1)):
            raise ValueError("ONNX returned values outside [0,1].")
        return result

    def score_candidates(self, patches: PatchArray, rows: list[dict], metadata: dict) -> ScoreBatch:
        return score_batch(self.scores(patches, metadata), rows, metadata, self.metadata["onnx_sha256"])


def tree_scores(model: TreeModel, rows: list[dict], metadata: dict, model_hash: str) -> ScoreBatch:
    model.validate()
    supplied = contract(metadata)
    if model.metadata.get("source_sha256", {}).get("detector.py") != supplied["source_sha256"]["detector.py"]:
        raise ValueError("Tree and patch detector sources differ.")
    vectors = []
    for row in rows:
        candidate_identity(row)
        vector = list(row["features"])
        if len(model.feature_names) != len(FEATURE_NAMES):
            vector.extend(row["extra_features"][name] for name in EXTRA_FEATURE_NAMES)
        vectors.append(vector)
    matrix = np.asarray(vectors, dtype=np.float64).reshape(len(rows), len(model.feature_names))
    return score_batch(model.scores(matrix), rows, supplied, model_hash)


def blend_scores(tree: ScoreBatch, cnn: ScoreBatch, cnn_weight: float) -> ScoreBatch:
    weight = unit_interval(cnn_weight, "CNN weight")
    tree.validate()
    cnn.validate()
    if tree.identities != cnn.identities or tree.contract != cnn.contract:
        raise ValueError("Cannot blend mismatched case/candidate/order/features/data metadata.")
    identity = json_sha256({
        "tree": tree.model_sha256, "cnn": cnn.model_sha256, "cnn_weight": weight,
    })
    return ScoreBatch(
        (1 - weight) * tree.scores + weight * cnn.scores, list(tree.identities),
        tree.contract, identity,
    )


@dataclass(frozen=True)
class BlendModel:
    cnn_weight: float
    threshold: float
    tree_sha256: str
    cnn_sha256: str
    split: dict[str, list[str]]
    metadata: dict

    def validate(self) -> None:
        unit_interval(self.cnn_weight, "CNN weight")
        unit_interval(self.threshold, "Blend threshold")
        sha256_string(self.tree_sha256)
        sha256_string(self.cnn_sha256)
        validate_split(self.split)
        if self.metadata.get("selected_on") != "validation":
            raise ValueError("Blending requires explicit validation-only selection.")
        if self.metadata.get("case_ids") != self.split["validation"]:
            raise ValueError("Blend selection cases differ from validation split.")
        sha256_string(self.metadata.get("validation_identities_sha256"))
        contract(self.metadata["contract"])

    def scores(self, tree: ScoreBatch, cnn: ScoreBatch) -> Scores:
        self.validate()
        if tree.model_sha256 != self.tree_sha256 or cnn.model_sha256 != self.cnn_sha256:
            raise ValueError("Blending model hashes differ from the frozen selection.")
        expected = self.metadata["contract"]
        if any(tree.contract[k] != expected[k] for k in (
            "source_sha256", "preprocessing", "feature_names", "extra_feature_names",
        )):
            raise ValueError("Blend extraction contract mismatch.")
        return blend_scores(tree, cnn, self.cnn_weight).scores

    def save(self, path: Path) -> None:
        self.validate()
        payload = {
            "schema_version": 1, "model_type": "fixed_probability_blend",
            "cnn_weight": self.cnn_weight, "threshold": self.threshold,
            "tree_sha256": self.tree_sha256, "cnn_sha256": self.cnn_sha256,
            "split": self.split, "metadata": self.metadata,
        }
        write_json(path, {**payload, "model_sha256": json_sha256(payload)})

    @classmethod
    def load(cls, path: Path) -> "BlendModel":
        payload = read_json(path, MAX_MODEL_BYTES)
        if (payload.get("schema_version") != 1 or payload.get("model_type") != "fixed_probability_blend"
                or payload.get("model_sha256") != json_sha256(
                    {k: v for k, v in payload.items() if k != "model_sha256"})):
            raise ValueError("Invalid blend artifact or integrity.")
        model = cls(**{k: payload[k] for k in (
            "cnn_weight", "threshold", "tree_sha256", "cnn_sha256", "split", "metadata",
        )})
        model.validate()
        return model
