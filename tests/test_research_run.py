import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import SimpleITK as sitk

from candidate_patches import preprocessing_metadata
from detector import detect
import patch_learning
from patch_inference import BlendModel, INFERENCE_AVAILABLE, file_sha256
from research_run import score_detection, select_prediction, source_hashes, validate_tree_source
from tabular_learning import FEATURE_NAMES, Tree, TreeModel
from test_detector import phantom
from test_patch_learning import partition, SPLIT
from train_trees import inference_contract

ROOT = Path(__file__).resolve().parents[1]


def model() -> TreeModel:
    sources = source_hashes()
    return TreeModel(
        "random_forest", FEATURE_NAMES.copy(),
        [Tree([-1], [-1], [-2], [-2.0], [0.1])], 0.9,
        {"train": ["train"], "validation": ["val"], "test": ["test"]},
        {"inference_contract": {
            "source_sha256": {
                name: sources[name] for name in ("detector.py", "candidate_patches.py", "learning.py")
            },
            "preprocessing": preprocessing_metadata(),
        }},
    )


def test_scores_only_preserves_geometry_ids_and_order_without_mutating_detection():
    image, mask = phantom()
    result = detect(image, mask)
    before = copy.deepcopy(result.prediction("case"))
    scores = np.array([0.1, 0.9])
    assert select_prediction(result, "case", scores, 0.5, "scores-only") == before
    filtered = select_prediction(result, "case", scores, 0.5, "filter")
    assert filtered["daughters"] == before["daughters"][1:]
    assert result.prediction("case") == before
    with pytest.raises(ValueError, match="ordered"):
        select_prediction(result, "case", np.ones(1), 0.5, "filter")
    with pytest.raises(ValueError, match="ordered"):
        select_prediction(result, "case", np.array([np.nan, 0.5]), 0.5, "scores-only")


@pytest.mark.parametrize("drift", ["detector.py", "candidate_patches.py", "learning.py", "preprocessing"])
def test_model_contract_rejects_source_or_physical_drift(drift):
    tree = model()
    contract = tree.metadata["inference_contract"]
    if drift == "preprocessing":
        contract["preprocessing"]["spacing_mm"] = 2
    else:
        contract["source_sha256"][drift] = "0" * 64
    with pytest.raises(ValueError, match="physical preprocessing/source"):
        validate_tree_source(tree, source_hashes())


def test_historical_model_cannot_silently_score_current_source():
    tree = model()
    tree.metadata.pop("inference_contract")
    with pytest.raises(ValueError, match="verified extraction contract"):
        validate_tree_source(tree, source_hashes())


@pytest.mark.parametrize("profile", ["strict", "review-union"])
def test_cli_preserves_no_filter_output_and_separates_scores(tmp_path, profile):
    image, mask = phantom()
    sitk.WriteImage(image, str(tmp_path / "image.nii.gz"))
    sitk.WriteImage(mask, str(tmp_path / "mask.nii.gz"))
    model().save(tmp_path / "tree.json")
    command = [
        sys.executable, str(ROOT / "research_run.py"), "--image", str(tmp_path / "image.nii.gz"),
        "--aorta-mask", str(tmp_path / "mask.nii.gz"), "--tree-model", str(tmp_path / "tree.json"),
        "--case-id", "fixture", "--proposals", profile, "--threads", "1",
    ]
    for mode in ("scores-only", "filter"):
        paths = [tmp_path / f"{mode}-{name}.json" for name in ("output", "diagnostics", "proposals")]
        result = subprocess.run([
            *command, "--mode", mode, "--output", str(paths[0]), "--diagnostics", str(paths[1]),
            "--proposals-output", str(paths[2]),
        ], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        output, diagnostics, proposals = [json.loads(path.read_text()) for path in paths]
        assert len(proposals["daughters"]) == 2
        assert diagnostics["unfiltered_prediction"] == proposals
        assert diagnostics["scores"] == [0.1, 0.1]
        assert set(output) == set(proposals)
        assert output == proposals if mode == "scores-only" else output["daughters"] == []
        assert diagnostics["promotion"] == "not_authorized"
        assert all("score" not in daughter for daughter in output["daughters"])
        assert diagnostics["source_sha256"] == source_hashes()
        assert len(diagnostics["history"]["prior_exposed_case_ids"]) == 25
        repeat = subprocess.run([
            *command, "--output", str(paths[0]), "--diagnostics", str(paths[1]),
            "--proposals-output", str(paths[2]),
        ], capture_output=True, text=True)
        assert repeat.returncode == 1
        assert "distinct new paths" in repeat.stderr


def test_empty_candidates_are_valid_and_combination_requires_frozen_blend(tmp_path):
    image, mask = phantom()
    result = detect(image, mask * 0)
    path = tmp_path / "tree.json"
    model().save(path)
    scores, threshold, report = score_detection(
        image, mask * 0, result, "empty", "unknown:empty", {"image": "a" * 64}, tree_path=path,
    )
    assert scores.shape == (0,)
    assert report["records"] == []
    assert select_prediction(result, "empty", scores, threshold, "filter")["daughters"] == []
    with pytest.raises(ValueError, match="Combined"):
        score_detection(image, mask, result, "empty", "empty", {}, tree_path=path, onnx_path=path)


def test_training_contract_rejects_mixed_or_missing_source_provenance():
    pytest.importorskip("sklearn")
    sources = source_hashes()
    row = {"detector_sha256": sources["detector.py"], "extractor_sha256": sources["candidate_patches.py"]}
    assert inference_contract([row])["preprocessing"] == preprocessing_metadata()
    with pytest.raises(ValueError, match="source drift"):
        inference_contract([row, dict(row, detector_sha256="a" * 64)])
    with pytest.raises(ValueError, match="source drift"):
        inference_contract([row, {}])


def test_actual_onnx_and_combined_inference_preserve_physical_candidates(tmp_path):
    if not patch_learning.TRAINING_AVAILABLE or not INFERENCE_AVAILABLE:
        pytest.skip("Optional CPU Torch/ONNX/ORT packages not installed.")
    train, validation = partition("train"), partition("validation")
    sources = source_hashes()
    for part in (train, validation):
        part.metadata["source_sha256"] = {
            name: sources[name] for name in ("detector.py", "candidate_patches.py")
        }
        for row in part.records:
            row.update(detector_sha256=sources["detector.py"], extractor_sha256=sources["candidate_patches.py"])
    network, report = patch_learning.fit(
        train, validation, SPLIT, patch_learning.TrainingConfig(epochs=1, threads=1),
    )
    onnx = tmp_path / "model.onnx"
    parity = patch_learning.export(network, report, train, validation, SPLIT, onnx, {"scope": "unit_fixture"})
    assert parity["threshold_decisions_equal"]
    tree = model()
    tree.split = copy.deepcopy(SPLIT)
    path = tmp_path / "tree.json"
    tree.save(path)
    blend = BlendModel(
        0.5, 0.5, file_sha256(path), file_sha256(onnx), SPLIT,
        {"selected_on": "validation", "case_ids": SPLIT["validation"],
         "validation_identities_sha256": "a" * 64, "contract": validation.metadata},
    )
    blend_path = tmp_path / "blend.json"
    blend.save(blend_path)
    image, mask = phantom()
    result = detect(image, mask)
    before = copy.deepcopy(result.prediction("physical_fixture"))
    cnn, _, cnn_report = score_detection(
        image, mask, result, "physical_fixture", "fixture", {"image": "a" * 64},
        onnx_path=onnx, threads=1,
    )
    combined, threshold, combined_report = score_detection(
        image, mask, result, "physical_fixture", "fixture", {"image": "a" * 64},
        tree_path=path, onnx_path=onnx, blend_path=blend_path, threads=1,
    )
    np.testing.assert_allclose(combined, 0.5 * (cnn + 0.1))
    assert cnn_report["records"] == combined_report["records"]
    assert result.prediction("physical_fixture") == before
    assert select_prediction(result, "physical_fixture", combined, threshold, "scores-only") == before
