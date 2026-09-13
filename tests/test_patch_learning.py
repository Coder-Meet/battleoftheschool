"""Physical cache integrity, leakage controls and portable candidate inference."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from candidate_patches import EXTRA_FEATURE_NAMES, preprocessing_metadata
import patch_inference as inference
import patch_learning as training
from tabular_learning import FEATURE_NAMES, json_sha256, write_json


def metadata() -> dict:
    return {
        "schema_version": 1, "scope": "candidate_reviews_only", "partition": "train",
        "feature_names": FEATURE_NAMES, "extra_feature_names": EXTRA_FEATURE_NAMES,
        "preprocessing": preprocessing_metadata(),
        "manifest_sha256": "a" * 64,
        "source_sha256": {"detector.py": "b" * 64, "candidate_patches.py": "c" * 64},
    }


def row(case: str, index: int, positive: bool) -> dict:
    return {
        "case_id": case, "group_id": f"synthetic:{case}",
        "instance_id": f"branch_{index}", "fingerprint": f"{case}:{index}",
        "label": "confirmed" if positive else "rejected",
        "labeller": "analytic_synthetic_geometry", "family": "analytic_test",
        "features": [float(index)] * len(FEATURE_NAMES),
        "extra_features": dict.fromkeys(EXTRA_FEATURE_NAMES, 0.5), "patch_index": index,
        "input_sha256": {"orig.nii.gz": json_sha256(case)},
        "detector_sha256": "b" * 64, "extractor_sha256": "c" * 64,
    }


def partition(name: str, count: int = 6) -> training.Partition:
    rng = np.random.default_rng(42)
    patches = rng.uniform(0, 1, (count, 12, 32, 32)).astype(np.float32)
    records = [row(name, i, i % 2 == 0) for i in range(count)]
    return training.Partition(name, patches, records, {**metadata(), "partition": name})


SPLIT = {"train": ["train"], "validation": ["validation"], "test": ["test"]}


def save_partition(root: Path, part: training.Partition) -> dict:
    path = root / f"patches-{part.name}.npz"
    np.savez_compressed(path, patches=part.patches)
    payload = {**part.metadata, "patches_sha256": inference.file_sha256(path), "records": part.records}
    write_json(root / f"patches-{part.name}.json", payload)
    return {**part.metadata, "records": deepcopy(part.records)}


def test_exact_cache_roundtrip_and_review_fingerprint_rejection(tmp_path: Path) -> None:
    part = partition("train")
    reviews = save_partition(tmp_path, part)
    loaded = training.load_partition(tmp_path, "train", SPLIT, reviews)
    np.testing.assert_array_equal(part.patches, loaded.patches)
    reviews["records"][0]["fingerprint"] = "different geometry"
    with pytest.raises(ValueError, match="fingerprint/features/labels"):
        training.load_partition(tmp_path, "train", SPLIT, reviews)


@pytest.mark.parametrize("field", ["features", "label", "extra_features", "input_sha256"])
def test_review_metadata_mismatch_rejected(tmp_path: Path, field: str) -> None:
    part = partition("train")
    reviews = save_partition(tmp_path, part)
    reviews["records"][0][field] = reviews["records"][1][field]
    if field in ("extra_features", "input_sha256"):
        reviews["records"][0][field] = {}
    with pytest.raises(ValueError):
        training.load_partition(tmp_path, "train", SPLIT, reviews)


@pytest.mark.parametrize("kind", ["float64", "object", "nan", "shape", "range"])
def test_invalid_numeric_cache_rejected(tmp_path: Path, kind: str) -> None:
    part = partition("train")
    if kind in ("float64", "object"):
        part.patches = part.patches.astype(kind)
    elif kind == "nan":
        part.patches[0, 0, 0, 0] = np.nan
    elif kind == "range":
        part.patches[0, 0, 0, 0] = 2
    else:
        part.patches = part.patches[:, :3]
    reviews = save_partition(tmp_path, part)
    with pytest.raises(ValueError):
        training.load_partition(tmp_path, "train", SPLIT, reviews)


def test_npz_hash_and_partition_relative_order(tmp_path: Path) -> None:
    part = partition("train")
    reviews = save_partition(tmp_path, part)
    with (tmp_path / "patches-train.npz").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="integrity"):
        training.load_partition(tmp_path, "train", SPLIT, reviews)
    part.records[0]["patch_index"] = 3
    with pytest.raises(ValueError, match="order"):
        part.validate(SPLIT)


@pytest.mark.parametrize("leak", ["group", "ct", "case"])
def test_split_leakage_rejected(leak: str) -> None:
    train, val = partition("train"), partition("validation")
    for record in val.records:
        if leak == "group":
            record["group_id"] = train.records[0]["group_id"]
        elif leak == "ct":
            record["input_sha256"] = train.records[0]["input_sha256"]
        else:
            record["case_id"] = "train"
    with pytest.raises(ValueError, match="leakage|assignment"):
        training.check_group_split([train, val], SPLIT)


def test_balancing_limits_large_case_and_excludes_zero_weight_pseudo() -> None:
    records = [row("one", 0, True), row("one", 1, False)]
    records += [row("many", i, True) for i in range(100)]
    records += [row("many", 100, False)]
    records += [{**row("prior", 0, True), "labeller": "claude"}]
    weights = training.sampling_probabilities(records, 0)
    assert weights[-1] == 0
    assert weights[0] == pytest.approx(weights[2:102].sum())
    assert weights[1] == pytest.approx(weights[102])
    assert weights.sum() == pytest.approx(1)
    assert weights[[i for i, r in enumerate(records) if r["label"] == "confirmed"]].sum() == pytest.approx(0.5)


def test_normalization_excludes_zero_weight_and_preserves_input() -> None:
    patches = np.full((3, 12, 32, 32), 0.5, dtype=np.float32)
    patches[2] = 1
    mean, scale = training.normalizer(patches, np.array([0.5, 0.5, 0]))
    np.testing.assert_allclose(mean, 0.5)
    np.testing.assert_allclose(scale, 0.001)
    np.testing.assert_array_equal(patches[2], 1)


def test_ct_augmentations_keep_physical_geometry_channels_and_input_unchanged() -> None:
    part = partition("train")
    before = part.patches.copy()
    result = training.augment(
        part.patches, np.random.default_rng(1), hu_noise=10, hu_jitter=20,
        hu_scales=np.full(len(part.patches), 300),
    )
    structural = [i for i in range(12) if i % 4]
    np.testing.assert_array_equal(result[:, structural], before[:, structural])
    np.testing.assert_array_equal(part.patches, before)
    assert not np.array_equal(result[:, 0], before[:, 0])
    assert result.min() >= 0 and result.max() <= 1


def test_unsupported_physical_translations_and_unscaled_hu_jitter_fail() -> None:
    patches = partition("train").patches
    with pytest.raises(ValueError, match="source-volume reslicing"):
        training.augment(patches, np.random.default_rng(1), translation_mm=2)
    with pytest.raises(ValueError, match="per-case normalization"):
        training.augment(patches, np.random.default_rng(1), hu_noise=10)


@pytest.mark.parametrize("field", ["case_id", "instance_id", "fingerprint", "features", "group_id"])
def test_blend_rejects_candidate_identity_mismatch(field: str) -> None:
    records = partition("validation").records
    other = deepcopy(records)
    other[0][field] = [3.0] * 13 if field == "features" else "changed"
    tree = inference.score_batch(np.full(len(records), 0.7), records, metadata(), "d" * 64)
    cnn = inference.score_batch(np.full(len(records), 0.6), other, metadata(), "e" * 64)
    with pytest.raises(ValueError, match="mismatched"):
        inference.blend_scores(tree, cnn, 0.5)


def test_blend_order_data_hash_and_weight_checks() -> None:
    records = partition("validation").records
    tree = inference.score_batch(np.full(len(records), 0.7), records, metadata(), "d" * 64)
    cnn = inference.score_batch(np.full(len(records), 0.6), records[::-1], metadata(), "e" * 64)
    with pytest.raises(ValueError, match="mismatched"):
        inference.blend_scores(tree, cnn, 0.5)
    cnn = inference.score_batch(
        np.full(len(records), 0.6), records, {**metadata(), "manifest_sha256": "f" * 64}, "e" * 64,
    )
    with pytest.raises(ValueError, match="mismatched"):
        inference.blend_scores(tree, cnn, 0.5)
    with pytest.raises(ValueError):
        inference.blend_scores(tree, tree, float("nan"))


def test_validation_only_blend_freeze_roundtrip(tmp_path: Path) -> None:
    part = partition("validation")
    tree = inference.score_batch(np.where(part.labels, 0.8, 0.6), part.records, metadata(), "d" * 64)
    cnn = inference.score_batch(np.where(part.labels, 0.7, 0.1), part.records, metadata(), "e" * 64)
    model = training.select_blend(tree, cnn, part.labels, SPLIT)
    model.save(tmp_path / "blend.json")
    loaded = inference.BlendModel.load(tmp_path / "blend.json")
    np.testing.assert_array_equal(model.scores(tree, cnn), loaded.scores(tree, cnn))
    assert np.all(loaded.scores(tree, cnn)[part.labels == 1] >= loaded.threshold)
    with pytest.raises(ValueError, match="hashes"):
        loaded.scores(replace(tree, model_sha256="f" * 64), cnn)
    test = partition("test")
    wrong = inference.score_batch(tree.scores, test.records, metadata(), "d" * 64)
    with pytest.raises(ValueError, match="exclusively on validation"):
        training.select_blend(wrong, wrong, test.labels, SPLIT)


def test_filter_preserves_geometry_and_never_recovers_unproposed_branches() -> None:
    rows = [row("validation", i, True) for i in range(2)]
    branches = [{"instance_id": r["instance_id"], "ostium_xyz_mm": [i, 0, 1],
                 "radius_mm": 2, "path_xyz_mm": [[i, 0, 1], [i, 0, 6]]} for i, r in enumerate(rows)]
    original = {"case_id": "validation", "daughters": branches}
    snapshot = deepcopy(original)
    result = training.filter_scores(original, rows, np.array([0.1, 0.9]), 0.5)
    assert original == snapshot
    assert result["daughters"] == [branches[1]]
    assert result["daughters"][0] is branches[1]
    with pytest.raises(ValueError, match="geometry"):
        training.filter_scores(original, rows[::-1], np.array([0.1, 0.9]), 0.5)


@pytest.fixture(scope="module")
def frozen_model(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, training.Partition, dict]:
    if not training.TRAINING_AVAILABLE or not inference.INFERENCE_AVAILABLE:
        pytest.skip("Optional CPU Torch/ONNX/ORT packages not installed.")
    train, val = partition("train"), partition("validation")
    network, report = training.fit(train, val, SPLIT, training.TrainingConfig(epochs=2, threads=1))
    path = tmp_path_factory.mktemp("onnx") / "model.onnx"
    parity = training.export(network, report, train, val, SPLIT, path, {"prior_inspection": ["prior"]})
    assert parity["threshold_decisions_equal"]
    return path, val, report


def test_onnx_parity_empty_batches_threads_and_source_contract(
    frozen_model: tuple[Path, training.Partition, dict],
) -> None:
    path, val, _ = frozen_model
    model = inference.PatchModel.load(path, threads=1)
    assert model.session.get_providers() == ["CPUExecutionProvider"]
    assert model.metadata["parameter_count"] < 50_000
    one = model.scores(val.patches, val.metadata, batch_size=1)
    many = model.scores(val.patches, val.metadata, batch_size=4)
    np.testing.assert_allclose(one, many, atol=inference.PARITY_ATOL, rtol=inference.PARITY_RTOL)
    assert model.scores(val.patches[:0], val.metadata).shape == (0,)
    assert np.all(one[val.labels == 1] >= model.threshold)
    wrong = deepcopy(val.metadata)
    wrong["source_sha256"]["detector.py"] = "f" * 64
    with pytest.raises(ValueError, match="source/preprocessing"):
        model.scores(val.patches, wrong)
    for threads in (0, 5):
        with pytest.raises(ValueError, match="threads"):
            inference.PatchModel.load(path, threads=threads)


@pytest.mark.parametrize("mutation", ["onnx", "sidecar", "channels", "normalizer", "classes"])
def test_export_artifact_corruption_rejected(
    frozen_model: tuple[Path, training.Partition, dict], tmp_path: Path, mutation: str,
) -> None:
    path, _, _ = frozen_model
    target = tmp_path / "model.onnx"
    target.write_bytes(path.read_bytes())
    sidecar = json.loads(path.with_suffix(".json").read_text())
    if mutation == "onnx":
        target.write_bytes(target.read_bytes() + b"broken")
    elif mutation == "sidecar":
        sidecar["threshold"] = 0
    else:
        if mutation == "channels":
            sidecar["contract"]["preprocessing"]["channel_names"].reverse()
        elif mutation == "normalizer":
            sidecar["normalizer"]["scale"][0] = 0
        else:
            sidecar["classes"] = [1, 0]
        sidecar["sidecar_sha256"] = json_sha256({k: v for k, v in sidecar.items() if k != "sidecar_sha256"})
    write_json(target.with_suffix(".json"), sidecar)
    with pytest.raises(ValueError):
        inference.PatchModel.load(target)


def test_test_data_cannot_affect_fit_and_normalization(
    frozen_model: tuple[Path, training.Partition, dict],
) -> None:
    path, validation, report = frozen_model
    test = partition("test")
    test.patches[:] = 1
    for record in test.records:
        record["label"] = "rejected"
    train = partition("train")
    network, repeated = training.fit(train, validation, SPLIT, training.TrainingConfig(epochs=2, threads=1))
    assert report["normalizer"] == repeated["normalizer"]
    assert report["threshold"] == repeated["threshold"]
    assert report["selected_epoch"] == repeated["selected_epoch"]
    model = inference.PatchModel.load(path, threads=1)
    np.testing.assert_allclose(
        training.native_scores(network, validation.patches),
        model.scores(validation.patches, validation.metadata),
        atol=inference.PARITY_ATOL, rtol=inference.PARITY_RTOL,
    )
    with pytest.raises(ValueError, match="only train and validation"):
        training.fit(train, test, SPLIT, training.TrainingConfig(epochs=1))


def test_base_and_optional_modules_import_without_ml_dependencies() -> None:
    script = """
import importlib.abc
import sys
class BlockML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'onnx', 'onnxruntime', 'sklearn'}:
            raise ImportError('optional dependency unavailable')
sys.meta_path.insert(0, BlockML())
import run
import patch_learning
import patch_inference
assert not patch_learning.TRAINING_AVAILABLE
assert not patch_inference.INFERENCE_AVAILABLE
try:
    patch_learning.require_training()
except ValueError as error:
    assert 'requirements-cnn-training.txt' in str(error)
else:
    raise AssertionError('missing training dependencies should fail explicitly')
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
