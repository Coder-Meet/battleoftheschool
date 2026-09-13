import copy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from learning import FEATURE_NAMES as LEGACY_FEATURE_NAMES
from tabular_learning import (
    EXTENDED_FEATURE_NAMES, EXTRA_FEATURE_NAMES, FEATURE_NAMES, PREPROCESSING,
    MAX_MODEL_BYTES, PlattCalibration, Tree, TreeModel, json_sha256, read_json, write_json,
)

try:
    import sklearn
    from train_trees import (
        DEFAULT_SOURCE_WEIGHTS, arrays, export_estimator, load_reviews, make_estimator,
        metrics, select_threshold, train, training_weights,
    )
except ImportError:
    HAS_TRAINING = False
else:
    HAS_TRAINING = True

requires_training = pytest.mark.skipif(not HAS_TRAINING, reason="Optional requirements-trees.txt not installed.")
ROOT = Path(__file__).resolve().parents[1]
SPLIT = {
    "train": [f"case{i}" for i in range(8)],
    "validation": ["case8", "case9", "case10"],
    "test": ["case11"],
}


@pytest.fixture
def rows():
    rng = np.random.default_rng(827)
    result = []
    for case in range(12):
        source = "claude" if case % 3 == 1 else "analytic_synthetic_geometry"
        for index in range(20):
            vector = rng.normal(size=16)
            label = vector[0] * vector[1] + vector[2] * 0.4 > 0
            result.append({
                "case_id": f"case{case}", "instance_id": f"branch{index}",
                "group_id": f"synthetic:{case}" if source != "claude" else f"patient:{case}",
                "family": "mathematical_test_fixture", "labeller": source,
                "label": "confirmed" if label else "rejected",
                "features": vector[:13].tolist(),
                "extra_features": dict(zip(EXTRA_FEATURE_NAMES, vector[13:].tolist())),
                "fingerprint": json_sha256({"case": case, "candidate": index, "features": vector.tolist()}),
            })
    return result


def stump() -> TreeModel:
    return TreeModel(
        "random_forest", FEATURE_NAMES.copy(),
        [Tree([1, -1, -1], [2, -1, -1], [0, -2, -2], [1.0, -2.0, -2.0], [0.5, 0.1, 0.9])],
        0.4, copy.deepcopy(SPLIT), {"usage": "scores_only"},
    )


def save_changed(path: Path, value: dict) -> None:
    value.pop("model_sha256", None)
    value["model_sha256"] = json_sha256(value)
    write_json(path, value)


def test_feature_contract_and_float32_routing_do_not_mutate_candidates(tmp_path):
    assert FEATURE_NAMES == LEGACY_FEATURE_NAMES
    assert EXTENDED_FEATURE_NAMES == FEATURE_NAMES + [
        "ostium_local_contrast", "patch_vesselness_std", "parent_wall_curvature",
    ]
    model = stump()
    matrix = np.zeros((3, 13))
    matrix[:, 0] = [1, 1 + 3e-8, np.nextafter(np.float32(1), np.float32(2))]
    original = matrix.copy()
    np.testing.assert_allclose(model.scores(matrix), [0.1, 0.1, 0.9], rtol=0, atol=0)
    np.testing.assert_array_equal(matrix, original)
    path = tmp_path / "model.json"
    model.save(path)
    loaded = TreeModel.load(path)
    assert loaded.threshold == 0.4
    assert loaded.metadata["usage"] == "scores_only"
    np.testing.assert_array_equal(loaded.scores(matrix), model.scores(matrix))
    assert loaded.scores(np.empty((0, 13))).shape == (0,)
    assert read_json(path)["preprocessing"] == PREPROCESSING


@pytest.mark.parametrize("value", [
    np.zeros((2, 12)), np.zeros((2, 16)), np.zeros(13),
    [[float("nan")] * 13], [[float("inf")] * 13], [[1e39] * 13],
    [["1"] * 13], [[True] * 13], [[1 + 2j] * 13],
])
def test_runtime_rejects_invalid_feature_matrices(value):
    with pytest.raises(ValueError, match="finite|columns"):
        stump().scores(value)


@pytest.mark.parametrize("field,value", [
    ("algorithm", "unknown"), ("feature_names", list(reversed(FEATURE_NAMES))),
    ("classes", [1, 0]), ("classes", [False, True]), ("classes", [0, 1, 2]),
    ("positive_class", 0), ("positive_class", True), ("schema_version", True),
    ("threshold", -0.01), ("threshold", 1.01), ("threshold", "0.5"),
    ("threshold", True), ("learning_rate", 0.5), ("initial_logit", 2),
    ("trees", []), ("metadata", []), ("calibration", {}),
    ("preprocessing", {"tree_input_dtype": "float64"}),
    ("split", {"train": ["same"], "validation": ["same"], "test": ["different"]}),
])
def test_untrusted_model_fields_are_validated_beyond_checksum(tmp_path, field, value):
    path = tmp_path / "bad.json"
    payload = stump().to_dict()
    payload[field] = value
    save_changed(path, payload)
    with pytest.raises(ValueError):
        TreeModel.load(path)


@pytest.mark.parametrize("field,value", [
    ("children_left", [0, -1, -1]),
    ("children_left", [2, -1, -1]),
    ("children_left", [-1, -1, -1]),
    ("children_left", [3, -1, -1]),
    ("children_left", [1.0, -1, -1]),
    ("children_left", [True, -1, -1]),
    ("children_right", [1, -1, -1]),
    ("feature", [13, -2, -2]),
    ("feature", [-2, -2, -2]),
    ("feature", [0, 0, -2]),
    ("threshold", [1.0, -2]),
    ("threshold", [1.0, "nan", -2]),
    ("value", [0.5, -0.1, 0.9]),
    ("value", [0.5, 0.1, 1.1]),
    ("value", [[0.5], [0.1], [0.9]]),
])
def test_corrupt_tree_topology_indices_dimensions_and_probabilities(tmp_path, field, value):
    path = tmp_path / "bad.json"
    payload = stump().to_dict()
    payload["trees"][0][field] = value
    save_changed(path, payload)
    with pytest.raises(ValueError):
        TreeModel.load(path)


def test_rejects_unreachable_nodes_nonfinite_json_and_tampering(tmp_path):
    path = tmp_path / "model.json"
    payload = stump().to_dict()
    tree = payload["trees"][0]
    for name, value in (("children_left", -1), ("children_right", -1), ("feature", -2),
                        ("threshold", -2.0), ("value", 0.5)):
        tree[name].append(value)
    save_changed(path, payload)
    with pytest.raises(ValueError, match="reachable"):
        TreeModel.load(path)
    stump().save(path)
    payload = read_json(path)
    payload["threshold"] = 0.3
    write_json(path, payload)
    with pytest.raises(ValueError, match="integrity"):
        TreeModel.load(path)
    path.write_text('{"duplicate": 1, "duplicate": 2}')
    with pytest.raises(ValueError, match="Duplicate"):
        TreeModel.load(path)
    path.write_text('{"invalid": NaN}')
    with pytest.raises(ValueError, match="Nonfinite"):
        TreeModel.load(path)


def test_runtime_has_no_sklearn_scipy_or_detector_imports(tmp_path):
    path = tmp_path / "model.json"
    stump().save(path)
    result = subprocess.run([
        sys.executable, "-c",
        "import sys; sys.modules.update({name: None for name in "
        "['sklearn','scipy','SimpleITK','detector','learning']}); "
        "from pathlib import Path; from tabular_learning import TreeModel; "
        "assert TreeModel.load(Path(sys.argv[1])).scores([[0.0]*13]).tolist() == [0.1]",
        str(path),
    ], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("field,value", [
    ("partition", "test"), ("case_ids", ["case11"]), ("case_ids", SPLIT["validation"]),
    ("case_ids", ["case8", "case8"]), ("records_sha256", "missing"),
    ("group_ids", []), ("group_ids", ["same", "same"]), ("source_counts", {}),
    ("weighting", "balanced"),
])
def test_loader_rejects_invalid_calibration_fitting_provenance(tmp_path, field, value):
    model = stump()
    provenance = {
        "partition": "validation_calibration", "case_ids": ["case8"],
        "group_ids": ["synthetic:8"], "records_sha256": "a" * 64,
        "source_counts": {name: {"rows": 2, "confirmed": 1, "rejected": 1}
                          for name in ("analytic", "expert", "pseudo")},
        "weighting": "unweighted_observed_prevalence",
    }
    model.calibration = PlattCalibration(1.0, 0.0, provenance)
    path = tmp_path / "calibrated.json"
    model.save(path)
    assert TreeModel.load(path).calibration is not None
    payload = read_json(path)
    payload["calibration"]["provenance"][field] = value
    save_changed(path, payload)
    with pytest.raises(ValueError, match="Calibration"):
        TreeModel.load(path)


def test_leaf_threshold_sentinel_and_exact_artifact_size_limits(tmp_path):
    path = tmp_path / "bad.json"
    payload = stump().to_dict()
    payload["trees"][0]["threshold"][1] = 0.5
    save_changed(path, payload)
    with pytest.raises(ValueError, match="sentinel"):
        TreeModel.load(path)
    with path.open("wb") as stream:
        stream.truncate(MAX_MODEL_BYTES + 1)
    with pytest.raises(ValueError, match="size limit"):
        TreeModel.load(path)
    output = tmp_path / "output.json"
    with pytest.raises(ValueError, match="size limit"):
        write_json(output, {"data": [0] * 100}, maximum_bytes=100)
    assert not output.exists()


@requires_training
@pytest.mark.parametrize("algorithm", ["gradient_boosting", "random_forest"])
@pytest.mark.parametrize("dimensions", [13, 16])
def test_export_equals_sklearn_on_unseen_rows_and_split_boundaries(tmp_path, algorithm, dimensions):
    assert sklearn.__version__ == "1.7.2"
    rng = np.random.default_rng(313)
    x = rng.normal(size=(300, dimensions))
    y = (x[:, 0] * x[:, 1] + x[:, -1] > 0).astype(np.int64)
    fitted = make_estimator(algorithm).fit(x, y, sample_weight=np.linspace(0.1, 2, len(y)))
    names = FEATURE_NAMES if dimensions == 13 else EXTENDED_FEATURE_NAMES
    model = export_estimator(fitted, names, SPLIT)
    matrix = rng.normal(size=(160, dimensions))
    boundary_rows = []
    for tree in model.trees[:5]:
        for feature, threshold in zip(tree.feature, tree.threshold):
            if feature < 0:
                continue
            for value in (threshold, np.nextafter(np.float32(threshold), np.float32(-np.inf)),
                          np.nextafter(np.float32(threshold), np.float32(np.inf))):
                row = np.zeros(dimensions)
                row[feature] = value
                boundary_rows.append(row)
    matrix = np.concatenate((matrix, boundary_rows))
    path = tmp_path / "export.json"
    model.save(path)
    model = TreeModel.load(path)
    np.testing.assert_allclose(model.scores(matrix), fitted.predict_proba(matrix)[:, 1], rtol=1e-12, atol=1e-12)
    assert model.scores(np.empty((0, dimensions))).shape == (0,)


@requires_training
@pytest.mark.parametrize("algorithm", ["gradient_boosting", "random_forest"])
def test_test_changes_cannot_affect_training_calibration_or_threshold(rows, algorithm):
    model, report = train(rows, SPLIT, algorithm=algorithm, calibration_cases=["case8"])
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["case_id"] == "case11":
            row["features"] = [10000.0] * 13
            row["label"] = "confirmed" if row["label"] == "rejected" else "rejected"
    second, _ = train(changed, SPLIT, algorithm=algorithm, calibration_cases=["case8"])
    assert model.to_dict() == second.to_dict()
    assert report["partitions"]["test"]["status"] == "sealed_not_scored"
    assert "classifier" not in report["partitions"]["test"]
    assert model.calibration is not None
    assert model.calibration.provenance["case_ids"] == ["case8"]
    assert model.calibration.provenance["weighting"] == "unweighted_observed_prevalence"
    assert set(model.metadata["fit_case_ids"]) == set(SPLIT["train"])
    assert model.metadata["threshold_case_ids"] == ["case10", "case9"]
    assert model.metadata["threshold_selection"]["training_recall"] == 1


@requires_training
def test_validation_changes_do_not_change_fitted_estimator(rows):
    model, _ = train(rows, SPLIT)
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["case_id"] in SPLIT["validation"]:
            row["features"] = [1000.0] * 13
            row["label"] = "confirmed"
    second, report = train(changed, SPLIT)
    assert model.trees == second.trees
    assert report["threshold_selection"]["status"] == "keep_all_single_class_validation"
    assert second.threshold == 0


@requires_training
def test_source_weights_balance_training_without_changing_evaluation_prevalence(rows):
    selected = [row for row in rows if row["case_id"] in SPLIT["train"]]
    weights = training_weights(selected, DEFAULT_SOURCE_WEIGHTS)
    y = np.asarray([row["label"] == "confirmed" for row in selected])
    assert weights[y].sum() == pytest.approx(weights[~y].sum())
    for label in ("confirmed", "rejected"):
        analytic = next(i for i, row in enumerate(selected) if row["labeller"] != "claude" and row["label"] == label)
        pseudo = next(i for i, row in enumerate(selected) if row["labeller"] == "claude" and row["label"] == label)
        assert weights[pseudo] / weights[analytic] == pytest.approx(0.3)
    controls = {**DEFAULT_SOURCE_WEIGHTS, "pseudo": 0.0}
    model, report = train(rows, SPLIT, source_weights=controls)
    assert report["effective_training_source_counts"]["pseudo"]["rows"] == 0
    training = report["partitions"]["train"]
    assert training["source_counts"]["pseudo"]["rows"] > 0
    assert training["classifier"]["prevalence"] == pytest.approx(float(y.mean()))
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["case_id"] in SPLIT["train"] and row["labeller"] == "claude":
            row["features"] = [1e6] * 13
            row["label"] = "confirmed"
    second, _ = train(changed, SPLIT, source_weights=controls)
    assert model.to_dict() == second.to_dict()


@requires_training
@pytest.mark.parametrize("failure", [
    "group_leak", "case_leak", "fingerprint_leak", "missing_group",
    "missing_labeller", "unknown_source", "source_conflict", "missing_fingerprint", "conflicting_groups",
])
def test_provenance_and_patient_leakage_fail_before_fitting(rows, failure):
    split = copy.deepcopy(SPLIT)
    if failure == "group_leak":
        for row in rows[-20:]:
            row["group_id"] = rows[0]["group_id"]
    elif failure == "case_leak":
        split["test"].append(split["train"][0])
    elif failure == "fingerprint_leak":
        rows[-1]["fingerprint"] = rows[0]["fingerprint"]
    elif failure == "missing_group":
        del rows[0]["group_id"]
    elif failure == "missing_labeller":
        del rows[0]["labeller"]
    elif failure == "unknown_source":
        rows[0]["labeller"] = "unspecified"
    elif failure == "source_conflict":
        rows[0]["label_source"] = "expert"
    elif failure == "missing_fingerprint":
        del rows[0]["fingerprint"]
    elif failure == "conflicting_groups":
        rows[0]["group_id"] = "synthetic:different"
    with pytest.raises(ValueError, match="leakage|provenance|nonempty|groups|Conflicting"):
        train(rows, split)


@requires_training
def test_calibration_partition_cannot_overlap_threshold_groups_or_test(rows):
    for row in rows:
        if row["case_id"] == "case9":
            row["group_id"] = "synthetic:8"
    with pytest.raises(ValueError, match="disjoint"):
        train(rows, SPLIT, calibration_cases=["case8"])
    with pytest.raises(ValueError, match="subset"):
        train(rows, SPLIT, calibration_cases=["case11"])
    with pytest.raises(ValueError, match="subset"):
        train(rows, SPLIT, calibration_cases=SPLIT["validation"])


@requires_training
@pytest.mark.parametrize("extra", [None, {}, {name: 0.0 for name in EXTRA_FEATURE_NAMES[:-1]},
                                   {name: float("inf") for name in EXTRA_FEATURE_NAMES}])
def test_extended_feature_training_never_imputes_missing_or_nonfinite_values(rows, extra):
    rows[0]["extra_features"] = extra
    with pytest.raises(ValueError, match="extra_features|finite"):
        train(rows, SPLIT, feature_set="extended")


@requires_training
def test_loader_preserves_input_hashes_misses_fingerprints_and_explicit_legacy_grouping(rows, tmp_path):
    for row in rows:
        row.pop("group_id")
    payload = {
        "schema_version": 1, "feature_names": FEATURE_NAMES, "scope": "candidate_reviews_only",
        "manifest_sha256": "a" * 64, "source_sha256": {"detector.py": "b" * 64},
        "cases": [{"case_id": f"case{i}", "false_negatives": 2} for i in range(12)],
        "records": rows,
    }
    path = tmp_path / "reviews.json"
    write_json(path, payload)
    with pytest.raises(ValueError, match="group_id"):
        load_reviews([path])
    groups = {f"case{i}": f"synthetic:{i}" for i in range(12)}
    loaded, provenance = load_reviews([path, path], case_groups=groups)
    assert len(loaded) == len(rows)
    assert loaded[0]["fingerprint"] == rows[0]["fingerprint"]
    assert provenance[0]["metadata"]["cases"][0]["false_negatives"] == 2
    assert provenance[0]["metadata"]["source_sha256"] == {"detector.py": "b" * 64}
    exposure = {"case11": {"previously_pseudo_trained": True, "source": "earlier experiment"}}
    model, report = train(loaded, SPLIT, provenance=provenance, prior_exposure=exposure)
    assert model.metadata["prior_exposure"] == exposure
    assert report["input_provenance"] == provenance
    assert "group_id" not in read_json(path)["records"][0]
    rows[0]["label"] = "confirmed" if rows[0]["label"] == "rejected" else "rejected"
    second = tmp_path / "conflict.json"
    write_json(second, payload)
    with pytest.raises(ValueError, match="Conflicting reviews"):
        load_reviews([path, second], case_groups=groups)


@requires_training
def test_threshold_guard_preserves_training_positives_without_test_inputs():
    threshold, report = select_threshold(
        np.asarray([1.0, 1.0, 0.0, 0.0]), np.asarray([0.9, 0.8, 0.6, 0.1]),
        np.asarray([1.0, 1.0, 0.0]), np.asarray([0.5, 0.95, 0.01]),
        minimum_recall=1, minimum_training_recall=1,
    )
    assert threshold == 0.5
    assert report["training_recall"] == 1
    assert report["validation_recall"] == 1
    with pytest.raises(ValueError, match="Minimum"):
        select_threshold(np.ones(1), np.ones(1), np.ones(1), np.ones(1), minimum_recall=float("nan"))


@requires_training
def test_single_class_and_empty_metrics_are_honest_and_test_is_opt_in(rows):
    for row in rows[-20:]:
        row["label"] = "rejected"
    _, report = train(rows, SPLIT, evaluate_test=True)
    test = report["partitions"]["test"]["classifier"]
    assert test["roc_auc"] is None
    assert test["average_precision"] is None
    assert test["recall"] is None
    assert test["undefined"]["roc_auc"] == "Requires both classes."
    empty = metrics(np.empty(0), np.empty(0), 0.5)
    assert empty["calibration"]["brier_score"] is None
    assert empty["f1"] is None
    assert metrics(np.ones(3), np.asarray([0.1, 0.2, 0.3]), 0.5)["average_precision"] == 1
    for row in rows:
        if row["case_id"] in SPLIT["train"]:
            row["label"] = "confirmed"
    with pytest.raises(ValueError, match="both classes"):
        train(rows, SPLIT)


@requires_training
def test_cli_artifacts_are_reproducible_and_cannot_overwrite_inputs(rows, tmp_path):
    review = tmp_path / "reviews.json"
    split = tmp_path / "split.json"
    model_path = tmp_path / "model.json"
    report_path = tmp_path / "report.json"
    write_json(review, {"schema_version": 1, "feature_names": FEATURE_NAMES,
                        "scope": "candidate_reviews_only", "records": rows})
    write_json(split, SPLIT)
    command = [sys.executable, str(ROOT / "train_trees.py"), "--reviews", str(review),
               "--split", str(split), "--model", str(model_path), "--report", str(report_path),
               "--features", "extended", "--algorithm", "random_forest"]
    completed = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
    assert completed.returncode == 0, completed.stderr
    loaded = TreeModel.load(model_path)
    assert loaded.feature_names == EXTENDED_FEATURE_NAMES
    assert read_json(report_path)["partitions"]["test"]["status"] == "sealed_not_scored"
    before = model_path.read_bytes()
    repeated = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
    assert repeated.returncode != 0
    assert "never overwrite" in repeated.stderr
    assert model_path.read_bytes() == before
    x, _ = arrays(rows, "extended")
    assert np.isfinite(loaded.scores(x)).all()


def test_cli_reports_missing_optional_training_dependency():
    completed = subprocess.run([
        sys.executable, "-c",
        "import sys,runpy; sys.modules['sklearn']=None; "
        "sys.argv=['train_trees.py','--reviews','r.json','--split','s.json',"
        "'--model','m.json','--report','report.json']; runpy.run_module('train_trees',run_name='__main__')",
    ], cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode != 0
    assert "Optional tree training dependencies are absent" in completed.stderr
    assert "requirements-trees.txt" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_cli_reports_missing_scipy_dependency():
    completed = subprocess.run([
        sys.executable, "-c",
        "import sys,runpy; sys.modules['scipy']=None; "
        "sys.argv=['train_trees.py','--reviews','r.json','--split','s.json',"
        "'--model','m.json','--report','report.json']; runpy.run_module('train_trees',run_name='__main__')",
    ], cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode != 0
    assert "Optional tree training dependencies are absent" in completed.stderr
    assert "Traceback" not in completed.stderr
