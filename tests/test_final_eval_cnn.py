import copy
from pathlib import Path

import numpy as np
import pytest

from candidate_patches import EXTRA_FEATURE_NAMES, preprocessing_metadata
from final_eval_cnn import (
    MODELS, TREE, inventory, paired_bootstrap, read_fingerprint, reference_losses, replay_values, require_source,
)
from final_evaluation import CASES, filtered, read_json, score_variant
from patch_inference import PatchModel, blend_scores, score_batch
from research_run import source_hashes, validate_tree_source
from tabular_learning import FEATURE_NAMES, TreeModel


def receipt():
    sources = source_hashes()
    branch = {
        "instance_id": "branch_001", "parent_instance_id": "aorta",
        "ostium_xyz_mm": [1, 2, 3], "seed_xyz_mm": [1, 2, 8],
        "radius_mm": 2, "direction_xyz": [0, 0, 1],
    }
    vector = [1.0] * len(FEATURE_NAMES)
    row = {
        "case_id": CASES[0], "instance_id": "branch_001", "group_id": "unit:case",
        "fingerprint": f"[[1, 2, 3], [1, 2, 8], [0, 0, 1], 2, {vector}]",
        "features": vector, "extra_features": dict.fromkeys(EXTRA_FEATURE_NAMES, 0.1),
        "input_sha256": {"image": "a" * 64, "aorta_mask": "b" * 64},
        "detector_sha256": sources["detector.py"], "extractor_sha256": sources["candidate_patches.py"],
    }
    return {
        "status": "success", "records": [row],
        "unfiltered_prediction": {"case_id": CASES[0], "parent": {"instance_id": "aorta"}, "daughters": [branch]},
        "contract": {
            "preprocessing": preprocessing_metadata(), "feature_names": FEATURE_NAMES,
            "extra_feature_names": EXTRA_FEATURE_NAMES, "manifest_sha256": "c" * 64,
            "source_sha256": {name: sources[name] for name in ("detector.py", "candidate_patches.py")},
        },
        "component_scores": {"tree": [0.2], "onnx": [0.8]},
        "model_sha256": {"tree": "d" * 64, "onnx": "e" * 64},
    }


def test_convex_replay_keeps_probabilities_and_inclusive_threshold():
    data = receipt()
    before = copy.deepcopy(data)
    for weight in (0, 0.25, 0.5, 0.75, 1):
        values = replay_values(data, weight)
        assert values[0] == pytest.approx((1 - weight) * 0.2 + weight * 0.8)
    assert len(filtered(data["unfiltered_prediction"], [0.5], 0.5)["daughters"]) == 1
    assert data == before


@pytest.mark.parametrize("mutation", ["order", "geometry", "features", "source", "nonfinite", "failure"])
def test_replay_rejects_misalignment_and_failures(mutation):
    data = receipt()
    if mutation == "order":
        data["records"][0]["instance_id"] = "different"
    elif mutation == "geometry":
        data["unfiltered_prediction"]["daughters"][0]["seed_xyz_mm"][0] = 42
    elif mutation == "features":
        data["records"][0]["features"][0] = 42
    elif mutation == "source":
        data["records"][0]["extractor_sha256"] = "f" * 64
    elif mutation == "nonfinite":
        data["component_scores"]["onnx"] = [float("nan")]
    else:
        data["status"] = "failed"
    with pytest.raises(ValueError):
        replay_values(data, 0.5)


def test_blend_rejects_different_ordered_candidate_identity():
    data = receipt()
    left = score_batch(np.array([0.2]), data["records"], data["contract"], "a" * 64)
    changed = copy.deepcopy(data["records"])
    changed[0]["group_id"] = "other"
    right = score_batch(np.array([0.8]), changed, data["contract"], "b" * 64)
    with pytest.raises(ValueError, match="mismatched"):
        blend_scores(left, right, 0.5)


def test_empty_case_kept_and_unknown_radii_not_invented():
    predictions = {
        case: {"case_id": case, "parent": {"instance_id": "aorta"}, "daughters": []} for case in CASES
    }
    scores = score_variant(predictions)
    assert scores["3"]["summary"]["cases"] == 5
    assert scores["3"]["summary"]["false_negatives"] == 19
    assert scores["3"]["summary"]["errors"]["radius_error_mm"]["n"] == 0
    losses = reference_losses(scores, scores)
    assert len(losses["3"]) == 5
    assert all(not row["lost_reference_ids"] for row in losses["3"])


def test_actual_models_enforce_sources_and_fit_provenance():
    pytest.importorskip("onnxruntime")
    model = PatchModel.load(MODELS["current"], threads=1)
    require_source(model, source_hashes())
    validate_tree_source(TreeModel.load(TREE), source_hashes())
    for name in ("legacy-synthetic", "legacy-mixed"):
        old = PatchModel.load(MODELS[name], threads=1)
        with pytest.raises(ValueError, match="extraction source"):
            require_source(old, source_hashes())
    audit = inventory()
    assert audit["onnx_count"] == 3
    mixed = next(row for row in audit["onnx_artifacts"] if "/mixed/" in row["path"])
    assert set(mixed["released_cases_in_declared_split"]["train"]) == {"subject022", "subject023"}
    current = next(row for row in audit["onnx_artifacts"] if "current-source-v1" in row["path"])
    assert not current["released_cases_in_fit"]
    assert current["training_labellers"] == ["analytic_synthetic_geometry"]


def test_actual_onnx_cpu_empty_and_batched_parity():
    pytest.importorskip("onnxruntime")
    model = PatchModel.load(MODELS["current"], threads=1)
    patches = np.random.default_rng(17).uniform(size=(3, 12, 32, 32)).astype(np.float32)
    assert model.session.get_providers() == ["CPUExecutionProvider"]
    np.testing.assert_allclose(
        model.scores(patches, model.contract, batch_size=1),
        model.scores(patches, model.contract, batch_size=3), atol=2e-6, rtol=1e-5,
    )
    assert model.scores(patches[:0], model.contract).shape == (0,)


def test_fingerprint_rejects_wrong_structure():
    with pytest.raises(ValueError):
        read_fingerprint("{}")


def test_paired_bootstrap_same_predictions_have_zero_difference():
    predictions = {
        case: {"case_id": case, "parent": {"instance_id": "aorta"}, "daughters": []} for case in CASES
    }
    scores = score_variant(predictions)
    result = paired_bootstrap(scores, scores)
    assert all(
        row["paired_f1_difference_percentile_95"] == [0, 0] for row in result["intervals"].values()
    )
    other = copy.deepcopy(scores)
    other["3"]["cases"].reverse()
    with pytest.raises(ValueError, match="paired cases"):
        paired_bootstrap(scores, other)


def test_preserved_artifacts_replay_when_available():
    paths = list((Path(__file__).resolve().parents[1] / "labels/final-eval/cnn/candidates/current").glob("*/*.json"))
    if not paths:
        pytest.skip("Run the real-case matrix to verify its saved artifacts.")
    receipts = [read_json(path) for path in paths if not path.name.endswith(".resources.json")]
    assert len(receipts) == 10
    for data in receipts:
        if data["status"] == "success":
            for weight in (0, 0.25, 0.5, 0.75, 1):
                assert len(replay_values(data, weight)) == len(data["records"])
