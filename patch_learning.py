"""Research-only CPU training on frozen, fingerprinted physical patch caches.

Example: python patch_learning.py --corpus archived/cohort-v1 --output experiment
Test evaluation requires a separate --evaluate-test invocation after model freeze.
Cached 2.5D planes cannot supply translated off-plane content; geometric
augmentation is disabled. HU jitter requires an explicit per-case HU scale.
"""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter
import SimpleITK as sitk

import candidate_patches
import detector

try:
    import onnx
    import torch
    from torch import nn
    from torch.nn import functional as functional
except ImportError:
    TRAINING_AVAILABLE = False
else:
    TRAINING_AVAILABLE = True

from patch_inference import (
    BlendModel, PARITY_ATOL, PARITY_RTOL, PatchArray, PatchModel, ScoreBatch,
    blend_scores, candidate_identity, contract, file_sha256, score_batch,
    tree_scores, validate_patches,
)
from evaluate import evaluate_case, summarize_cases
from learning import features
from nifti_io import read_nifti
from research_corpus import comparisons
from tabular_learning import TreeModel, json_sha256, read_json, validate_split, write_json
from train_trees import source_category, validate_row


@dataclass
class Partition:
    name: str
    patches: PatchArray
    records: list[dict]
    metadata: dict

    @property
    def labels(self) -> np.ndarray:
        return np.asarray([int(r["label"] == "confirmed") for r in self.records], dtype=np.int64)

    def validate(self, split: dict[str, list[str]]) -> None:
        validate_split(split)
        if self.name not in split:
            raise ValueError("Unknown patch partition.")
        expected = contract(self.metadata)
        validate_patches(self.patches, expected["preprocessing"]["size"])
        if len(self.patches) != len(self.records):
            raise ValueError("Patch/record count mismatch.")
        for i, row in enumerate(self.records):
            validate_row(row, "extended")
            candidate_identity(row)
            if row.get("patch_index") != i or row["case_id"] not in split[self.name]:
                raise ValueError("Partition-relative patch order or case assignment mismatch.")
            if (row["detector_sha256"] != expected["source_sha256"]["detector.py"]
                    or row["extractor_sha256"] != expected["source_sha256"]["candidate_patches.py"]):
                raise ValueError("Patch records disagree with detector/extractor provenance.")
        score_batch(np.zeros(len(self.records)), self.records, expected, "0" * 64)


def load_partition(root: Path, name: str, split: dict[str, list[str]], reviews: dict) -> Partition:
    metadata = read_json(root / f"patches-{name}.json")
    if (metadata.get("schema_version") != 1 or metadata.get("partition") != name
            or metadata.get("scope") != "candidate_reviews_only"):
        raise ValueError("Invalid patch cache metadata.")
    for key in ("feature_names", "extra_feature_names", "manifest_sha256", "source_sha256"):
        if reviews.get(key) != metadata.get(key):
            raise ValueError("Review/cache feature or provenance metadata mismatch.")
    if "cases" in metadata and {r["case_id"] for r in metadata["cases"]} != set(split[name]):
        raise ValueError("Patch case metadata omits split cases.")
    path = root / f"patches-{name}.npz"
    if file_sha256(path) != metadata.get("patches_sha256"):
        raise ValueError("Patch NPZ integrity mismatch.")
    with np.load(path, allow_pickle=False) as archive:
        if archive.files != ["patches"]:
            raise ValueError("Patch archive must contain only the numeric patches array.")
        patches = archive["patches"]
    records = metadata.get("records")
    if not isinstance(records, list):
        raise ValueError("Missing ordered patch records.")
    review_rows = {(r["case_id"], r["instance_id"]): r for r in reviews["records"]}
    if len(review_rows) != len(reviews["records"]):
        raise ValueError("Duplicate review candidate.")
    expected_rows = {key for key, r in review_rows.items() if r["case_id"] in split[name]
                     and r["label"] in ("confirmed", "rejected")}
    if {(r["case_id"], r["instance_id"]) for r in records} != expected_rows:
        raise ValueError("Patch candidates differ from the partition's classified review set.")
    for row in records:
        review = review_rows[(row["case_id"], row["instance_id"])]
        if {k: v for k, v in row.items() if k != "patch_index"} != {
            k: v for k, v in review.items() if k != "patch_index"
        }:
            raise ValueError("Patch fingerprint/features/labels/provenance differ from reviews.")
    part = Partition(name, patches, records, metadata)
    part.validate(split)
    return part


def check_group_split(parts: list[Partition], split: dict[str, list[str]]) -> None:
    groups: dict[str, str] = {}
    inputs: dict[str, str] = {}
    cases: dict[str, str] = {}
    for part in parts:
        part.validate(split)
        for row in part.records:
            group, case = row["group_id"], row["case_id"]
            if group in groups and groups[group] != part.name:
                raise ValueError("Group/seed leakage between partitions.")
            groups[group] = part.name
            if case in cases and cases[case] != group:
                raise ValueError("A case has inconsistent group provenance.")
            cases[case] = group
            images = [value for name, value in row["input_sha256"].items()
                      if name.startswith(("orig", "ct."))]
            for image in images:
                if image in inputs and inputs[image] != part.name:
                    raise ValueError("CT hash leakage between partitions.")
                inputs[image] = part.name


def load_development(root: Path) -> tuple[Partition, Partition, dict[str, list[str]], dict]:
    split = validate_split(read_json(root / "split.json"))
    review_path = root / "reviews-development.json"
    if not review_path.exists():
        review_path = root / "reviews.json"
    reviews = read_json(review_path)
    train = load_partition(root, "train", split, reviews)
    validation = load_partition(root, "validation", split, reviews)
    check_group_split([train, validation], split)
    if contract(train.metadata) != contract(validation.metadata):
        raise ValueError("Train and validation cache contracts differ.")
    manifest = read_json(root / "manifest.json")
    if file_sha256(root / "manifest.json") != train.metadata.get("manifest_sha256"):
        raise ValueError("Cache manifest hash mismatch.")
    case_rows = {r["case_id"]: r for r in manifest["cases"]}
    group_partition: dict[str, str] = {}
    for name, case_ids in split.items():
        for case in case_ids:
            info = case_rows.get(case)
            if not isinstance(info, dict) or not info.get("group_id"):
                raise ValueError(f"Missing complete group provenance for {case}.")
            group = info["group_id"]
            if group in group_partition and group_partition[group] != name:
                raise ValueError("Group/seed leakage in complete case manifest.")
            group_partition[group] = name
    for part in (train, validation):
        for row in part.records:
            info = case_rows[row["case_id"]]
            if row["group_id"] != info["group_id"] or row["input_sha256"] != info["files"]:
                raise ValueError("Candidate group/input provenance differs from manifest.")
    return train, validation, split, {
        "reviews_sha256": file_sha256(review_path),
        "split_sha256": file_sha256(root / "split.json"),
        "review_metadata": {k: v for k, v in reviews.items() if k != "records"},
        "manifest": manifest,
        "partition_metadata": {
            p.name: {k: v for k, v in p.metadata.items() if k != "records"} for p in (train, validation)
        },
    }


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 12
    batch_size: int = 32
    learning_rate: float = 0.001
    pseudo_weight: float = 0.3
    pseudo_smoothing: float = 0.1
    blur_sigma: float = 0.5
    seed: int = 42
    threads: int = 4

    def validate(self) -> None:
        for value, low, high in ((self.epochs, 1, 1000), (self.batch_size, 1, 256), (self.threads, 1, 4)):
            if type(value) is not int or not low <= value <= high:
                raise ValueError("Invalid bounded CPU training configuration.")
        for coefficient in (self.learning_rate, self.pseudo_weight, self.pseudo_smoothing, self.blur_sigma):
            if not math.isfinite(coefficient) or coefficient < 0:
                raise ValueError("Training coefficients must be finite and nonnegative.")
        if not 0 < self.learning_rate <= 1 or self.pseudo_smoothing >= 0.5 or self.blur_sigma > 1:
            raise ValueError("Invalid learning rate, smoothing or blur.")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("Seed must fit an unsigned 32-bit integer.")


def sampling_probabilities(records: list[dict], pseudo_weight: float) -> np.ndarray:
    """Equal total class mass, then source-weighted groups/cases within class."""
    if not math.isfinite(pseudo_weight) or pseudo_weight < 0:
        raise ValueError("Invalid pseudo source weight.")
    labels = [int(r["label"] == "confirmed") for r in records]
    sources = [source_category(r) for r in records]
    counts = Counter((y, r["group_id"], r["case_id"]) for y, r in zip(labels, records))
    group_cases: dict[tuple[int, str], set[str]] = {}
    for y, row in zip(labels, records):
        group_cases.setdefault((y, row["group_id"]), set()).add(row["case_id"])
    weights = np.asarray([
        (pseudo_weight if source == "pseudo" else 1.0) / (
            counts[(y, r["group_id"], r["case_id"])] * len(group_cases[(y, r["group_id"])])
        )
        for y, r, source in zip(labels, records, sources)
    ])
    for label in (0, 1):
        selected = np.asarray(labels) == label
        total = float(weights[selected].sum())
        if total <= 0:
            raise ValueError("Both training classes need positive source weight.")
        weights[selected] /= 2 * total
    return weights


def normalizer(patches: PatchArray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.zeros(12, dtype=np.float64)
    square = np.zeros(12, dtype=np.float64)
    for patch, weight in zip(patches, probabilities, strict=True):
        values = patch.astype(np.float64)
        mean += weight * values.mean(axis=(1, 2))
        square += weight * np.square(values).mean(axis=(1, 2))
    scale = np.sqrt(np.maximum(square - mean**2, 1e-6))
    return mean.astype(np.float32), scale.astype(np.float32)


def augment(
    patches: PatchArray, rng: np.random.Generator, blur_sigma: float = 0.5,
    hu_noise: float = 0, hu_jitter: float = 0, hu_scales: np.ndarray | None = None,
    translation_mm: float = 0,
) -> PatchArray:
    if translation_mm != 0:
        raise ValueError("Cached planes lack off-plane content; translations require source-volume reslicing.")
    if not all(math.isfinite(x) and x >= 0 for x in (blur_sigma, hu_noise, hu_jitter)):
        raise ValueError("Augmentation magnitudes must be finite and nonnegative.")
    result = patches.copy()
    if hu_noise or hu_jitter:
        if hu_scales is None or hu_scales.shape != (len(patches),) or (
            not np.isfinite(hu_scales).all() or np.any(hu_scales <= 0)
        ):
            raise ValueError("HU augmentation requires verified per-case normalization scales.")
    for i in range(len(result)):
        sigma = rng.uniform(0, blur_sigma)
        jitter = rng.uniform(-hu_jitter, hu_jitter)
        for channel in (0, 4, 8):
            values = result[i, channel]
            if sigma > 0:
                values = gaussian_filter(values, sigma=sigma, mode="constant", cval=0)
            if hu_scales is not None:
                values = values + (jitter + rng.normal(0, hu_noise, values.shape)) / hu_scales[i]
            result[i, channel] = np.clip(values, 0, 1)
    return result


if TRAINING_AVAILABLE:
    class PatchNetwork(nn.Module):
        mean: torch.Tensor
        scale: torch.Tensor

        def __init__(self, mean: np.ndarray, scale: np.ndarray) -> None:
            super().__init__()
            self.register_buffer("mean", torch.tensor(mean.reshape(1, 12, 1, 1)))
            self.register_buffer("scale", torch.tensor(scale.reshape(1, 12, 1, 1)))
            self.layers = nn.Sequential(
                nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
                nn.Conv2d(24, 40, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
                nn.Conv2d(40, 64, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 1),
            )

        def forward(self, patches: torch.Tensor) -> torch.Tensor:
            return self.layers((patches - self.mean) / self.scale).squeeze(1)

    class ProbabilityNetwork(nn.Module):
        def __init__(self, network: PatchNetwork) -> None:
            super().__init__()
            self.network = network

        def forward(self, patches: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.network(patches))


def require_training() -> None:
    if not TRAINING_AVAILABLE:
        raise ValueError("Install requirements-cnn-training.txt for optional CPU training/export.")


def retain_recall_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.shape != scores.shape or not np.isfinite(scores).all() or not np.any(labels == 1):
        raise ValueError("Recall selection requires finite validation scores and positives.")
    # A numerical guard below the smallest positive avoids export-rounding flips.
    return float(max(0, np.min(scores[labels == 1]) - (PARITY_ATOL + PARITY_RTOL)))


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    keep = scores >= threshold
    tp, fp = int(np.sum(keep & (labels == 1))), int(np.sum(keep & (labels == 0)))
    fn, tn = int(np.sum(~keep & (labels == 1))), int(np.sum(~keep & (labels == 0)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "candidate_recall": tp / (tp + fn) if tp + fn else None,
            "reference_recall": None, "unproposed_references_recoverable": False}


def native_scores(network: "PatchNetwork", patches: PatchArray, batch_size: int = 64) -> np.ndarray:
    require_training()
    network.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(patches), batch_size):
            result.extend(torch.sigmoid(network(torch.from_numpy(patches[start:start + batch_size]))).tolist())
    return np.asarray(result, dtype=np.float64)


def fit(
    train: Partition, validation: Partition, split: dict[str, list[str]], config: TrainingConfig,
) -> tuple["PatchNetwork", dict]:
    require_training()
    config.validate()
    if train.name != "train" or validation.name != "validation":
        raise ValueError("Fit accepts only train and validation partitions.")
    check_group_split([train, validation], split)
    if contract(train.metadata) != contract(validation.metadata):
        raise ValueError("Training and validation contracts differ.")
    if not len(validation.records) or not np.any(validation.labels == 1):
        raise ValueError("Validation needs candidates including positives.")
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(config.seed)
    probabilities = sampling_probabilities(train.records, config.pseudo_weight)
    mean, scale = normalizer(train.patches, probabilities)
    network = PatchNetwork(mean, scale)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    labels = train.labels.astype(np.float32)
    pseudo = np.asarray([source_category(r) == "pseudo" for r in train.records])
    labels[pseudo] = labels[pseudo] * (1 - 2 * config.pseudo_smoothing) + config.pseudo_smoothing
    best_state: dict[str, torch.Tensor] = {}
    best_key = (math.inf, math.inf)
    selected_epoch = 0
    history = []
    started = time.perf_counter()
    for epoch in range(config.epochs):
        network.train()
        indices = rng.choice(len(train.records), size=len(train.records), replace=True, p=probabilities)
        loss_sum = 0.0
        for start in range(0, len(indices), config.batch_size):
            batch = indices[start:start + config.batch_size]
            x = torch.from_numpy(augment(train.patches[batch], rng, blur_sigma=config.blur_sigma))
            y = torch.from_numpy(labels[batch])
            optimizer.zero_grad()
            loss = functional.binary_cross_entropy_with_logits(network(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
        scores = native_scores(network, validation.patches, config.batch_size)
        threshold = retain_recall_threshold(validation.labels, scores)
        result = metrics(validation.labels, scores, threshold)
        clipped = np.clip(scores, 1e-7, 1 - 1e-7)
        vloss = float(np.mean(-validation.labels * np.log(clipped) - (1 - validation.labels) * np.log1p(-clipped)))
        key = (result["fp"], vloss)
        history.append({"epoch": epoch + 1, "train_bce": loss_sum / len(indices),
                        "validation_bce": vloss, "threshold": threshold, "validation": result})
        if key < best_key:
            best_key = key
            best_state = {k: v.detach().clone() for k, v in network.state_dict().items()}
            selected_epoch = epoch + 1
    network.load_state_dict(best_state)
    network.eval()
    scores = native_scores(network, validation.patches)
    return network, {
        "config": asdict(config), "epochs": history, "selected_epoch": selected_epoch,
        "selection": "validation FP at zero lost candidate positives, then validation BCE, then earliest epoch",
        "threshold": retain_recall_threshold(validation.labels, scores),
        "normalizer": {"mean": mean.tolist(), "scale": scale.tolist(), "fit_partition": "train"},
        "sampler": {"method": "balanced class; inverse group/case candidate counts; source weighted",
                    "probabilities": probabilities.tolist(), "class_mass": [0.5, 0.5],
                    "source_weights": {"analytic": 1, "expert": 1, "pseudo": config.pseudo_weight}},
        "augmentation": {"CT_blur_sigma_pixels": [0, config.blur_sigma], "HU_noise": 0, "HU_jitter": 0,
                         "translations": "disabled: cached planes lack translated off-plane content",
                         "rotations": "disabled", "geometry_targets": "none; classification only"},
        "training_seconds_linux_or_host": time.perf_counter() - started,
        "train_records": train.records, "validation_records": validation.records,
    }


def export(
    network: "PatchNetwork", report: dict, train: Partition, validation: Partition,
    split: dict[str, list[str]], path: Path, provenance: dict,
) -> dict:
    require_training()
    if path.exists() or path.with_suffix(".json").exists():
        raise ValueError("Refusing to overwrite a frozen model.")
    path.parent.mkdir(parents=True, exist_ok=True)
    network.eval()
    wrapper = ProbabilityNetwork(network).eval()
    sample = torch.from_numpy(validation.patches[:2])
    torch.onnx.export(
        wrapper, (sample,), str(path), export_params=True, opset_version=17,
        input_names=["patches"], output_names=["probabilities"],
        dynamic_axes={"patches": {0: "batch"}, "probabilities": {0: "batch"}},
        dynamo=False, external_data=False,
    )
    onnx.checker.check_model(onnx.load(str(path)))
    model_hash = json_sha256({
        name: {"shape": list(tensor.shape), "sha256": hashlib.sha256(
            tensor.detach().numpy().tobytes()).hexdigest()}
        for name, tensor in network.state_dict().items()
    })
    metadata = {
        "schema_version": 1, "model_type": "patch_cnn_2_5d", "classes": [0, 1], "positive_class": 1,
        "architecture": "12->24->40->64 3x3 ReLU; two average pools; global mean; linear",
        "parameter_count": sum(p.numel() for p in network.parameters()),
        "output": "uncalibrated sigmoid probability of candidate label; not clinical accuracy",
        "input": {"name": "patches", "dtype": "float32",
                  "shape": ["batch", 12, train.metadata["preprocessing"]["size"], train.metadata["preprocessing"]["size"]]},
        "output_name": "probabilities", "opset": 17, "normalization_location": "inside_onnx",
        "normalizer": report["normalizer"], "threshold": report["threshold"],
        "threshold_selected_on": "validation", "recall_loss_budget": 0,
        "split": split, "contract": contract(train.metadata),
        "state_sha256": model_hash, "onnx_sha256": file_sha256(path),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "onnx": onnx.__version__, "numpy": np.__version__},
        "current_source_sha256": {name: file_sha256(Path(__file__).with_name(name)) for name in (
            "patch_learning.py", "patch_inference.py", "detector.py", "candidate_patches.py",
            "research_corpus.py", "evaluate.py", "research_validation.py", "train_trees.py",
        )},
        "training": report, "provenance": provenance,
        "parity_tolerance": {"atol": PARITY_ATOL, "rtol": PARITY_RTOL},
        "scope": "research_only", "clinical_accuracy_claim": False,
    }
    metadata["sidecar_sha256"] = json_sha256(metadata)
    write_json(path.with_suffix(".json"), metadata)
    loaded = PatchModel.load(path)
    native = native_scores(network, validation.patches)
    runtime = loaded.scores(validation.patches, validation.metadata, batch_size=7)
    if not np.allclose(native, runtime, atol=PARITY_ATOL, rtol=PARITY_RTOL):
        raise ValueError("PyTorch/ONNX probability parity failed.")
    if not np.array_equal(native >= loaded.threshold, runtime >= loaded.threshold):
        raise ValueError("PyTorch/ONNX threshold decisions differ.")
    if loaded.scores(validation.patches[:0], validation.metadata).shape != (0,):
        raise ValueError("Empty ONNX inference failed.")
    return {"max_absolute_error": float(np.max(np.abs(native - runtime))),
            "atol": PARITY_ATOL, "rtol": PARITY_RTOL, "threshold_decisions_equal": True,
            "empty_inference": True, "batch_size": 7, "provider": loaded.session.get_providers()}


def select_blend(
    tree: ScoreBatch, cnn: ScoreBatch, labels: np.ndarray, split: dict[str, list[str]],
    weights: tuple[float, ...] = (0, 0.25, 0.5, 0.75, 1),
) -> BlendModel:
    validate_split(split)
    if any(r["case_id"] not in split["validation"] for r in tree.identities):
        raise ValueError("Blend weights must be selected exclusively on validation candidates.")
    if labels.shape != tree.scores.shape or not np.isin(labels, [0, 1]).all() or not weights:
        raise ValueError("Invalid blend validation labels/grid.")
    options = []
    for weight in weights:
        combined = blend_scores(tree, cnn, weight).scores
        threshold = retain_recall_threshold(labels, combined)
        result = metrics(labels, combined, threshold)
        options.append((result["fp"], weight, threshold))
    _, weight, threshold = min(options)
    return BlendModel(weight, threshold, tree.model_sha256, cnn.model_sha256, split, {
        "selected_on": "validation", "case_ids": split["validation"],
        "validation_identities_sha256": json_sha256(tree.identities), "grid": list(weights),
        "rule": "fewest FP at zero lost validation candidate positives, then smaller CNN weight",
        "contract": tree.contract, "stacking": False,
    })


def compatible_splits(tree: dict[str, list[str]], cnn: dict[str, list[str]]) -> bool:
    return (tree["validation"] == cnn["validation"] and tree["test"] == cnn["test"]
            and set(tree["train"]) <= set(cnn["train"]))


def prepare_mixed_corpus(root: Path, output: Path, data: Path, reviews_path: Path, split_path: Path) -> dict:
    """Derive a separate cache using only exact historical training fingerprints.

Run with archived source ahead of this repository on PYTHONPATH. Runtime
detector/extractor hashes must match the immutable parent corpus before CT IO.
Only the historical real training cases are considered; no labels are changed.
"""
    train, validation, split, _ = load_development(root)
    expected = train.metadata["source_sha256"]
    if (file_sha256(Path(detector.__file__)) != expected["detector.py"]
            or file_sha256(Path(candidate_patches.__file__)) != expected["candidate_patches.py"]):
        raise ValueError("Mixed extraction requires the archive's exact detector/extractor source.")
    if output.exists():
        raise ValueError("Mixed corpus output already exists.")
    real_reviews = read_json(reviews_path)
    historical_split = validate_split(read_json(split_path))
    if real_reviews.get("feature_names") != train.metadata["feature_names"]:
        raise ValueError("Real review feature order differs from frozen cache.")
    all_reviews = read_json(root / "reviews.json")
    manifest = read_json(root / "manifest.json")
    audit: dict = {
        "parent_manifest_sha256": file_sha256(root / "manifest.json"),
        "real_reviews_sha256": file_sha256(reviews_path),
        "historical_split_sha256": file_sha256(split_path), "historical_split": historical_split,
        "prior_inspected_and_pseudo_trained_cases": sorted(
            r.name for r in data.iterdir() if r.is_dir() and r.name.startswith("subject")),
        "real_reference_completeness": "unknown; candidate pseudo-labels only",
        "cases": [], "new_labels": False,
    }
    arrays = [train.patches]
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    for case in historical_split["train"]:
        if any(case in ids for ids in split.values()):
            raise ValueError("Real/synthetic case ID collision.")
        records = [r for r in real_reviews["records"] if r["case_id"] == case]
        if not records:
            continue
        source = data / case
        images, masks = list(source.glob("orig*.nii*")), list(source.glob("mask*.nii*"))
        if len(images) != 1 or len(masks) != 1:
            raise ValueError("Real CT/mask files are not uniquely identified.")
        image, mask = read_nifti(str(images[0])), read_nifti(str(masks[0]))
        proposal = detector.detect_pool(image, mask, detector.DetectorConfig.review())
        by_id = {branch.instance_id: (i, branch) for i, branch in enumerate(proposal.branches)}
        accepted, rejected = [], []
        for row in records:
            match = by_id.get(row["instance_id"])
            if match is None:
                rejected.append(row["instance_id"])
                continue
            index, branch = match
            vector = features(branch)
            fingerprint = json.dumps([
                branch.ostium_xyz_mm, branch.seed_xyz_mm, branch.direction_xyz, branch.radius_mm, vector,
            ])
            if fingerprint != row["fingerprint"] or vector != row["features"]:
                rejected.append(row["instance_id"])
                continue
            accepted.append((index, row))
        case_audit: dict = {
            "case_id": case, "accepted_ids": [r["instance_id"] for _, r in accepted],
            "rejected_fingerprint_ids": rejected, "proposals": len(proposal.branches),
            "extraction_errors": [], "cached_ids": [],
        }
        audit["cases"].append(case_audit)
        if not accepted:
            continue
        selected = replace(proposal, branches=[proposal.branches[i] for i, _ in accepted])
        extracted_rows = []
        try:
            extracted = candidate_patches.extract_candidate_data(image, mask, selected)
        except ValueError:
            for index, row in accepted:
                single = replace(proposal, branches=[proposal.branches[index]])
                try:
                    extracted = candidate_patches.extract_candidate_data(image, mask, single)
                except ValueError as error:
                    case_audit["extraction_errors"].append({"instance_id": row["instance_id"], "error": str(error)})
                    continue
                extracted_rows.append((index, row, extracted.patches[0], extracted.extra_features[0]))
        else:
            extracted_rows = [(index, row, extracted.patches[i], extracted.extra_features[i])
                              for i, (index, row) in enumerate(accepted)]
        if not extracted_rows:
            continue
        inputs = {p.name: file_sha256(p) for p in (images[0], masks[0])}
        info = {"case_id": case, "group_id": f"real:{case}", "partition": "train",
                "files": inputs, "prior_inspection": True, "prior_pseudo_training": True}
        manifest["cases"].append(info)
        split["train"].append(case)
        all_reviews["cases"].append(info)
        for index, row, patch, extra in extracted_rows:
            record = {
                **row, "group_id": info["group_id"], "input_sha256": inputs,
                "detector_sha256": expected["detector.py"], "extractor_sha256": expected["candidate_patches.py"],
                "extra_features": dict(zip(candidate_patches.EXTRA_FEATURE_NAMES, extra.tolist())),
                "patch_index": len(train.records), "local_patch_index": index,
                "prior_inspection": True, "prior_pseudo_training": True,
            }
            train.records.append(record)
            all_reviews["records"].append(record)
            arrays.append(patch[np.newaxis])
            case_audit["cached_ids"].append(row["instance_id"])
    if len(arrays) == 1:
        raise ValueError("No real training fingerprints match; mixed experiment must remain unrun.")
    output.mkdir(parents=True)
    manifest["parent_manifest_sha256"] = audit["parent_manifest_sha256"]
    manifest["source"] = "analytic_synthetic_geometry_and_historical_candidate_pseudo"
    write_json(output / "manifest.json", manifest)
    manifest_hash = file_sha256(output / "manifest.json")
    all_reviews.update({"manifest_sha256": manifest_hash, "mixed_provenance": audit})
    train.patches = np.concatenate(arrays).astype(np.float32)
    for partition in (train, validation):
        partition.metadata["manifest_sha256"] = manifest_hash
    check_group_split([train, validation], split)
    for part in ("train", "validation", "test"):
        metadata = read_json(root / f"patches-{part}.json")
        metadata["manifest_sha256"] = manifest_hash
        if part == "train":
            np.savez_compressed(output / "patches-train.npz", patches=train.patches)
            metadata["records"] = train.records
            existing_cases = {r["case_id"] for r in metadata["cases"]}
            metadata["cases"].extend(r for r in manifest["cases"]
                                     if r["partition"] == "train" and r["case_id"] not in existing_cases)
        else:
            shutil.copyfile(root / f"patches-{part}.npz", output / f"patches-{part}.npz")
        metadata["patches_sha256"] = file_sha256(output / f"patches-{part}.npz")
        write_json(output / f"patches-{part}.json", metadata)
    write_json(output / "reviews.json", all_reviews)
    write_json(output / "split.json", split)
    shutil.copyfile(root / "summary.json", output / "summary.json")
    shutil.copytree(root / "exports", output / "exports")
    for record in manifest["cases"]:
        reference = root / "cases" / record["case_id"] / "reference.json"
        if reference.exists():
            destination = output / "cases" / record["case_id"]
            destination.mkdir(parents=True)
            shutil.copyfile(reference, destination / "reference.json")
    write_json(output / "mixed-cache-audit.json", audit)
    return audit


def evaluate_frozen(root: Path, output: Path, tree_path: Path | None) -> dict:
    model = PatchModel.load(output / "model.onnx")
    split = model.split
    reviews = read_json(root / "reviews.json")
    test = load_partition(root, "test", split, reviews)
    train, validation, current_split, _ = load_development(root)
    if current_split != split or contract(test.metadata) != model.contract:
        raise ValueError("Frozen test/model dataset or split mismatch.")
    check_group_split([train, validation, test], split)
    started = time.perf_counter()
    cnn = model.score_candidates(test.patches, test.records, test.metadata)
    result = {"cnn": metrics(test.labels, cnn.scores, model.threshold),
              "inference_seconds": time.perf_counter() - started,
              "candidate_count": len(test.records), "model_onnx_sha256": file_sha256(output / "model.onnx"),
              "records": test.records, "cnn_scores": cnn.scores.tolist(),
              "dataset_metadata": {k: v for k, v in test.metadata.items() if k != "records"},
              "retrospective_test_reuse": True, "clinical_accuracy_claim": False,
              "platform": platform.platform(), "windows_performance_verified": False}
    if tree_path is not None:
        tree_model = TreeModel.load(tree_path)
        tree = tree_scores(tree_model, test.records, test.metadata, file_sha256(tree_path))
        blend = BlendModel.load(output / "blend.json")
        combined = blend.scores(tree, cnn)
        result.update({"tree": metrics(test.labels, tree.scores, tree_model.threshold),
                       "blend": metrics(test.labels, combined, blend.threshold),
                       "tree_scores": tree.scores.tolist(), "blend_scores": combined.tolist()})
    result["reference_evaluation"] = evaluate_references(root, "test", model, output, tree_path)
    return result


def filter_scores(prediction: dict, rows: list[dict], scores: np.ndarray, threshold: float) -> dict:
    if ([r["instance_id"] for r in rows] != [b["instance_id"] for b in prediction["daughters"]]
            or len(rows) != len(scores) or any(r["case_id"] != prediction["case_id"] for r in rows)
            or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))):
        raise ValueError("Candidate scores and prediction geometry must be aligned.")
    return {**prediction, "daughters": [
        branch for branch, keep in zip(prediction["daughters"], scores >= threshold) if keep
    ]}


def evaluate_references(
    root: Path, part: str, model: PatchModel, output: Path, tree_path: Path | None,
) -> dict:
    """Score every archived proposal, including ambiguous rows, against all references."""
    if part not in ("validation", "test"):
        raise ValueError("Reference evaluation is for validation or frozen test.")
    if file_sha256(root / "manifest.json") != model.contract["manifest_sha256"]:
        raise ValueError("Reference corpus differs from model corpus.")
    manifest = read_json(root / "manifest.json")
    entries = [r for r in manifest["cases"] if r["case_id"] in model.split[part]]
    groups = {r["case_id"]: r["group_id"] for r in entries}
    if set(groups) != set(model.split[part]):
        raise ValueError("Reference manifest omits split cases.")
    tree_model = TreeModel.load(tree_path) if tree_path else None
    blend = BlendModel.load(output / "blend.json") if tree_path else None
    runs: dict[str, list[dict]] = {name: [] for name in ("pool", "strict", "cnn")}
    if tree_model is not None:
        if not compatible_splits(tree_model.split, model.split):
            raise ValueError("Tree/CNN split mismatch.")
        runs.update({"tree": [], "blend": []})
    for record in entries:
        case = record["case_id"]
        directory = root / "exports" / case
        receipt = read_json(directory / "receipt.json")
        for name in ("candidates.json", "patches.npz", "pool.json", "strict.json"):
            if file_sha256(directory / name) != receipt["files"][name]:
                raise ValueError("Archived per-case cache/prediction hash mismatch.")
        reference_path = root / "cases" / case / "reference.json"
        if file_sha256(reference_path) != record["files"]["reference.json"]:
            raise ValueError("Reference hash mismatch.")
        reference = read_json(reference_path)
        rows = read_json(directory / "candidates.json")["all_candidates"]
        for i, row in enumerate(rows):
            if (row["case_id"] != case or row["group_id"] != record["group_id"]
                    or row["local_patch_index"] != i or row["input_sha256"] != record["files"]):
                raise ValueError("Per-case patch/proposal identity mismatch.")
        with np.load(directory / "patches.npz", allow_pickle=False) as archive:
            patches = archive["patches"]
        cnn = model.score_candidates(patches, rows, model.contract)
        pool = read_json(directory / "pool.json")
        predictions = {
            "pool": pool, "strict": read_json(directory / "strict.json"),
            "cnn": filter_scores(pool, rows, cnn.scores, model.threshold),
        }
        if tree_model is not None and tree_path is not None and blend is not None:
            tree = tree_scores(tree_model, rows, model.contract, file_sha256(tree_path))
            predictions["tree"] = filter_scores(pool, rows, tree.scores, tree_model.threshold)
            predictions["blend"] = filter_scores(pool, rows, blend.scores(tree, cnn), blend.threshold)
        for name, prediction in predictions.items():
            runs[name].append({
                **evaluate_case(prediction, reference, 3), "family": record["family"],
                "group_id": record["group_id"],
                "sensitivity": {str(t): evaluate_case(prediction, reference, t) for t in (2, 5)},
            })
    return {
        "scope": "complete_analytic_references", "case_count": len(entries),
        "ambiguous_proposals": "scored using exact archived per-case numeric caches",
        "runs": {name: {
            "summary": summarize_cases(cases), "cases": cases,
            "versus_pool": comparisons(runs["pool"], cases, groups),
            "versus_strict": comparisons(runs["strict"], cases, groups),
        } for name, cases in runs.items()},
        "clinical_accuracy_claim": False, "retrospective_test_reuse": part == "test",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tree-model", type=Path)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pseudo-weight", type=float, default=0.3)
    parser.add_argument("--pseudo-smoothing", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--prepare-mixed", action="store_true")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--real-reviews", type=Path)
    parser.add_argument("--real-split", type=Path)
    args = parser.parse_args()
    try:
        if args.prepare_mixed:
            if args.data_root is None or args.real_reviews is None or args.real_split is None or args.evaluate_test:
                raise ValueError("Mixed preparation needs data-root, real-reviews and real-split.")
            prepare_mixed_corpus(args.corpus, args.output, args.data_root, args.real_reviews, args.real_split)
            return 0
        if args.evaluate_test:
            path = args.output / "test-report.json"
            if path.exists():
                raise ValueError("Frozen test report already exists; refusing repeat evaluation.")
            write_json(path, evaluate_frozen(args.corpus, args.output, args.tree_model))
            return 0
        if (args.output / "model.onnx").exists():
            raise ValueError("Frozen model exists; choose a unique output directory.")
        train, validation, split, provenance = load_development(args.corpus)
        gate = read_json(args.corpus / "summary.json")
        if gate.get("cnn_stop_go") != "go_research_only":
            raise ValueError("E1 CNN gate did not pass; E2/E3 training must remain unrun.")
        provenance["e1_gate_sha256"] = file_sha256(args.corpus / "summary.json")
        provenance["e1_gate"] = gate["cnn_stop_go"]
        provenance["retrospective_test_reuse"] = True
        config = TrainingConfig(epochs=args.epochs, seed=args.seed, pseudo_weight=args.pseudo_weight,
                                pseudo_smoothing=args.pseudo_smoothing, threads=args.threads)
        network, report = fit(train, validation, split, config)
        report["parity"] = export(network, report, train, validation, split, args.output / "model.onnx", provenance)
        model = PatchModel.load(args.output / "model.onnx")
        cnn = model.score_candidates(validation.patches, validation.records, validation.metadata)
        report["validation_cnn"] = metrics(validation.labels, cnn.scores, model.threshold)
        report["validation_cnn_scores"] = cnn.scores.tolist()
        if args.tree_model is not None:
            tree_model = TreeModel.load(args.tree_model)
            if not compatible_splits(tree_model.split, split):
                raise ValueError("Tree/CNN train-validation-test splits differ.")
            tree = tree_scores(tree_model, validation.records, validation.metadata, file_sha256(args.tree_model))
            blend = select_blend(tree, cnn, validation.labels, split)
            blend.save(args.output / "blend.json")
            report["validation_blend"] = metrics(validation.labels, blend.scores(tree, cnn), blend.threshold)
            report["validation_tree"] = metrics(validation.labels, tree.scores, tree_model.threshold)
            report["validation_tree_scores"] = tree.scores.tolist()
            report["tree_training_split"] = tree_model.split
        report["reference_validation"] = evaluate_references(
            args.corpus, "validation", model, args.output, args.tree_model,
        )
        write_json(args.output / "training-report.json", report)
        print(f"Frozen research model: {args.output / 'model.onnx'}")
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Patch research error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
