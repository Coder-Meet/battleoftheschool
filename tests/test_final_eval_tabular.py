import copy
from pathlib import Path
import shutil

import numpy as np
import pytest
import SimpleITK as sitk

from detector import Detection, DetectorConfig
import final_eval_tabular as tabular
from final_evaluation import CASES, ROOT, filtered, load_reference, score_variant
from learning import CandidateModel, FEATURE_NAMES, load_reviews, train
from research_run import score_detection, source_hashes, validate_tree_source
from tabular_learning import TreeModel


def review(case: str, fingerprint: str, **extra: object) -> dict:
    return {
        "case_id": case, "instance_id": "branch_001", "fingerprint": fingerprint,
        "features": [0.0] * len(FEATURE_NAMES), "label": "confirmed", **extra,
    }


def test_exclusion_catches_renamed_bytes_groups_and_candidate_duplicates():
    rows = [
        review("subject019", "original"),
        review("renamed", "alias"),
        review("subject004", "original"),
        review("subject005", "group", group_id="subject022"),
        review("subject006", "hash", input_sha256={"image": "ct20"}),
        review("unregistered", "unknown"),
        review("subject001", "clean"),
    ]
    aliases = {
        **{f"subject{i:03d}": f"subject{i:03d}" for i in range(1, 26)},
        "renamed": "subject019",
    }
    kept, audit = tabular.exclude_reference_reviews(rows, aliases, {"subject020": {"image": "ct20"}})
    assert [row["case_id"] for row in kept] == ["subject001"]
    assert audit["removed_case_ids"] == [
        "renamed", "subject004", "subject005", "subject006", "subject019", "unregistered",
    ]
    assert audit["excluded_reference_case_ids"] == list(CASES)
    assert audit["removed_rows"][2]["reasons"] == ["identical_excluded_candidate_fingerprint"]
    assert "reference_input_hash" in audit["removed_rows"][4]["reasons"]


def test_lfs_oid_can_verify_alias_without_loading_pointer_as_image(tmp_path: Path):
    path = tmp_path / "image.nii.gz"
    path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:" + "a" * 64 + "\nsize 5000\n")
    assert tabular.file_identity(path) == "a" * 64
    path.write_text("version https://git-lfs.github.com/spec/v1\nbroken\n")
    with pytest.raises(ValueError, match="Malformed"):
        tabular.file_identity(path)


def test_fresh_model_excludes_all_five_and_scaler_is_fitted_only_on_train():
    aliases = {f"subject{i:03d}": f"subject{i:03d}" for i in range(1, 26)}
    rows = load_reviews([ROOT / "labels/reviews.json"])
    selected, audit = tabular.exclude_reference_reviews(rows, aliases, {})
    previous = tabular.read_json(ROOT / "labels/split.json")
    split = {part: [case for case in cases if case not in CASES] for part, cases in previous.items()}
    model, report = train(selected, split)
    training = np.array([row["features"] for row in selected if row["case_id"] in split["train"]])
    np.testing.assert_allclose(model.mean, training.mean(axis=0))
    assert not set(CASES) & {case for cases in model.split.values() for case in cases}
    assert audit["removed_case_ids"] == ["subject020", "subject021", "subject022", "subject023"]
    assert audit["reference_cases_without_pseudo_rows"] == ["subject019"]
    assert report["threshold_selected_on"] == "validation"
    assert report["label_sources"] == {"claude": len(selected)}
    perturbed = copy.deepcopy(selected)
    for row in perturbed:
        if row["case_id"] in split["test"]:
            row["features"] = [value + 100 for value in row["features"]]
            row["label"] = "confirmed" if row["label"] == "rejected" else "rejected"
    repeated, _ = train(perturbed, split)
    assert model == repeated


@pytest.mark.parametrize("frozen,expected", [(0.5, 5), (0.37631433544133275, 6)])
def test_frozen_threshold_is_preserved_without_rounded_name_collisions(frozen: float, expected: int):
    values = tabular.thresholds(frozen)
    assert frozen in values
    assert len(values) == expected
    assert all(float(f"{value:.17g}") == value for value in values)
    with pytest.raises(ValueError):
        tabular.thresholds(float("nan"))


@pytest.mark.parametrize("filename", [
    "gradient_boosting-base.json", "gradient_boosting-extended.json",
    "random_forest-base.json", "random_forest-extended.json",
])
def test_every_current_tree_validates_source_and_handles_empty_candidates(filename: str):
    path = ROOT / "labels/research/current-source-v1/models" / filename
    image = sitk.Image([8, 8, 8], sitk.sitkInt16)
    mask = sitk.Image([8, 8, 8], sitk.sitkUInt8)
    result = Detection([], {}, {}, 0, {}, [], DetectorConfig())
    scores, threshold, diagnostics = score_detection(
        image, mask, result, "empty", "empty", {}, tree_path=path,
    )
    assert scores.shape == (0,)
    assert diagnostics["records"] == []
    assert filtered(result.prediction("empty"), scores, threshold)["daughters"] == []
    tree = TreeModel.load(path)
    tree.metadata["inference_contract"]["source_sha256"]["detector.py"] = "0" * 64
    with pytest.raises(ValueError, match="source differs"):
        validate_tree_source(tree, source_hashes())


@pytest.mark.parametrize("path", [
    ROOT / "labels/candidate-model.json",
    ROOT / "labels/candidate-model-synthetic.json",
    ROOT / "labels/research/current-source-v1/models/logistic.json",
])
def test_saved_logistics_accept_correctly_shaped_zero_candidate_matrix(path: Path):
    assert CandidateModel.load(path).scores(np.empty((0, len(FEATURE_NAMES)))).shape == (0,)


def test_conditional_recall_records_exact_lost_targets_and_keeps_zero_cases():
    proposals, predictions = {}, {}
    for case in CASES:
        reference = load_reference(case)
        daughter = reference["daughters"][0]
        prediction: dict = {
            "case_id": case, "parent": {"instance_id": "aorta"},
            "daughters": [{
                key: value for key, value in daughter.items() if key in (
                    "instance_id", "parent_instance_id", "ostium_xyz_mm", "seed_xyz_mm",
                    "radius_mm", "direction_xyz",
                )
            }],
        }
        # Fixture prediction radius is independent of whether the reference radius is measured.
        prediction["daughters"][0]["radius_mm"] = 2.0
        proposals[case] = prediction
        predictions[case] = filtered(prediction, [0.1], 0.5)
    conditional = tabular.conditional_recall(predictions, proposals)
    scored = score_variant(predictions)
    for tolerance in ("2", "3", "5"):
        assert conditional[tolerance]["conditional_tp_retention"] == 0
        assert len(conditional[tolerance]["cases"]) == 5
        assert all(len(row["removed_reference_ids"]) == 1 for row in conditional[tolerance]["cases"])
        assert scored[tolerance]["summary"]["false_negatives"] == 19
        assert scored[tolerance]["summary"]["errors"]["radius_error_mm"]["n"] == 0


def test_all_committed_variants_replay_from_ordered_scores_and_retain_five_cases():
    report = tabular.read_json(tabular.OUTPUT / "report.json")
    models = {model["name"]: model for model in tabular.read_json(tabular.OUTPUT / "model-audit.json")["models"]}
    caches = {
        (profile, case): tabular.read_json(tabular.OUTPUT / "candidates" / profile / f"{case}.json")
        for profile in tabular.PROFILES for case in CASES
    }
    for (profile, case), cache in caches.items():
        tabular.validate_cache(cache, case, profile, report["input_sha256"])
        assert cache["input_geometry"]["image"] == cache["input_geometry"]["mask"]
        for name, saved in cache["model_scores"].items():
            artifact = models[name]
            path = ROOT / artifact["path"]
            assert tabular.digest(path) == saved["model_sha256"]
            vectors = [
                record["features"] + (
                    [record["extra_features"][feature] for feature in tabular.EXTRA_FEATURE_NAMES]
                    if len(artifact["feature_names"]) > len(FEATURE_NAMES) else []
                ) for record in cache["records"]
            ]
            matrix = np.asarray(vectors, dtype=float).reshape(len(vectors), len(artifact["feature_names"]))
            model = TreeModel.load(path) if artifact["kind"] == "tree" else CandidateModel.load(path)
            np.testing.assert_allclose(model.scores(matrix), saved["probabilities"], rtol=0, atol=0)
    names = [variant["name"] for variant in report["variants"]]
    assert len(names) == len(set(names))
    for variant in report["variants"]:
        configuration = variant["configuration"]
        model = configuration["model"]
        predictions = {}
        for case in CASES:
            cache = caches[configuration["profile"], case]
            probabilities = (
                cache["model_scores"][model["name"]]["probabilities"]
                if model else [1.0] * len(cache["records"])
            )
            prediction = tabular.read_json(ROOT / variant["prediction_dir"] / f"{case}.json")
            assert prediction == filtered(cache["prediction"], probabilities, configuration["threshold"])
            assert variant["runtime"][case]["runtime_s"] >= 0
            assert variant["runtime"][case]["peak_rss_mb"] > 0
            predictions[case] = prediction
        assert score_variant(predictions) == variant["scores"]


@pytest.mark.parametrize("mutation", ["ordering", "geometry", "driver", "probabilities", "records"])
def test_cache_corruption_is_rejected_before_prediction_generation(mutation: str):
    report = tabular.read_json(tabular.OUTPUT / "report.json")
    cache = tabular.read_json(tabular.OUTPUT / "candidates/review-union/subject022.json")
    if mutation == "ordering":
        cache["prediction"]["daughters"].reverse()
    elif mutation == "geometry":
        cache["prediction"]["daughters"][0]["radius_mm"] += 1
    elif mutation == "driver":
        cache["driver_sha256"] = "0" * 64
    elif mutation == "probabilities":
        next(iter(cache["model_scores"].values()))["probabilities"] = []
    else:
        cache["records"][0]["features"][0] += 1
    with pytest.raises(ValueError):
        tabular.validate_cache(cache, "subject022", "review-union", report["input_sha256"])


def test_failed_model_never_becomes_an_empty_prediction_variant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    shutil.copytree(tabular.OUTPUT / "candidates", tmp_path / "candidates")
    report = tabular.read_json(tabular.OUTPUT / "report.json")
    model = next(model for model in tabular.read_json(tabular.OUTPUT / "model-audit.json")["models"]
                 if model["name"] == "tabular-excluded-five-logistic")
    cache_path = tmp_path / "candidates/review-union/subject019.json"
    cache = tabular.read_json(cache_path)
    del cache["model_scores"][model["name"]]
    cache["failures"] = [{"model": model["name"], "error": "test failure", "case_id": "subject019"}]
    tabular.write_json(cache_path, cache)
    monkeypatch.setattr(tabular, "relative", lambda path: str(path))
    monkeypatch.setattr(tabular, "input_inventory", lambda: (report["input_sha256"], {}))
    rebuilt = tabular.report_variants(tmp_path, [model])
    assert len(rebuilt["variants"]) == 2 + len(tabular.thresholds(model["threshold"]))
    assert len(rebuilt["failures"]) == 2
    assert not any(row["configuration"]["model"] and row["configuration"]["profile"] == "review-union"
                   for row in rebuilt["variants"])


def test_fold_selection_and_bootstrap_do_not_reuse_held_out_case_for_ranking():
    report = tabular.read_json(tabular.OUTPUT / "report.json")
    selection = tabular.selection_summary(report)
    assert selection["uncertainty"] == tabular.selection_summary(report)["uncertainty"]
    for fold in selection["folds"]:
        altered = copy.deepcopy(report)
        for variant in altered["variants"]:
            variant["scores"]["3"]["cases"] = [
                row for row in variant["scores"]["3"]["cases"] if row["case_id"] != fold["held_out_case"]
            ]
            variant["runtime"][fold["held_out_case"]]["runtime_s"] += 10000
        eligible = [row for row in altered["variants"] if row["eligible_for_selection"]]
        selected = min(eligible, key=lambda row: tabular.rank_variant(row, fold["held_out_case"]))
        assert selected["name"] == fold["selected_variant"]
        for tolerance in ("2", "3", "5"):
            expected = next(row for row in selected["scores"][tolerance]["cases"]
                            if row["case_id"] == fold["held_out_case"]) if tolerance != "3" else None
            if expected:
                assert expected == fold["held_out_scores"][tolerance]
