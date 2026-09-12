import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import SimpleITK as sitk

from research_validation import compare, ingest_references, overlap_report, paired_bootstrap, seed_polyline_distance


def branch(identifier, x=0.0):
    return {
        "instance_id": identifier, "parent_instance_id": "aorta",
        "ostium_xyz_mm": [x, 0.0, 0.0], "seed_xyz_mm": [x, 5.0, 0.0],
        "direction_xyz": [0.0, 1.0, 0.0], "radius_mm": 2.0,
    }


def prediction(case, daughters):
    return {"case_id": case, "parent": {"instance_id": "aorta"}, "daughters": daughters}


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


@pytest.fixture
def bundle(tmp_path):
    for name in ("baseline", "variant", "proposals"):
        (tmp_path / name).mkdir()
    references = []
    for case in ("subject001", "subject002", "subject003"):
        ref = prediction(case, [branch("true")])
        ref["family"] = "straight"
        references.append(ref)
        write(tmp_path / "baseline" / f"{case}.json", prediction(case, [branch("true"), branch("fp", 30)]))
        write(tmp_path / "variant" / f"{case}.json", prediction(case, [branch("true")]))
        write(tmp_path / "proposals" / f"{case}.json", prediction(case, [branch("true"), branch("fp", 30)]))
    write(tmp_path / "refs.json", references)
    history = {"history_complete": True, "split": {"train": [], "validation": [], "test": []}}
    # These are test inputs to the guard, not measured performance.
    runtime = {
        "platform": "Windows", "cpu_cores": 4, "gpu": False, "network": False,
        "cases": {r["case_id"]: {"runtime_s": 10.0, "peak_rss_mb": 500.0} for r in references},
    }
    return {
        "schema_version": 1, "references": "refs.json", "baseline": "baseline",
        "variants": {"filter": "variant"}, "reference_provenance": "complete_expert",
        "references_complete": True, "bootstrap_samples": 100, "bootstrap_seed": 91,
        "histories": {"baseline": copy.deepcopy(history), "filter": copy.deepcopy(history)},
        "runtime_reports": {"baseline": copy.deepcopy(runtime), "filter": copy.deepcopy(runtime)},
        "compute_budget": {"max_runtime_s": 60, "max_peak_rss_mb": 8000, "max_cpu_cores": 4},
        "proposals": {"filter": "proposals"},
    }


def test_complete_expert_paired_improvement_and_family_counts(bundle, tmp_path):
    report = compare(bundle, tmp_path)
    assert report["comparisons"]["filter"]["promotion"]["pass"]
    assert report["clinical_accuracy_claim"] is False
    assert report["runs"]["baseline"]["by_family"]["straight"]["3"]["false_positives"] == 3
    assert report["runs"]["filter"]["summary_by_tolerance_mm"]["3"]["true_positives"] == 3
    changes = report["comparisons"]["filter"]["cases"]["subject001"]["3"]
    assert changes["corrected_false_positive_ids"] == ["fp"]
    assert changes["newly_lost_reference_ids"] == []
    family_changes = report["comparisons"]["filter"]["by_family"]["straight"]["3"]
    assert family_changes["delta"]["false_positives"] == -3
    assert family_changes["corrected_false_positive_ids"] == [
        {"case_id": case, "instance_id": "fp"} for case in ("subject001", "subject002", "subject003")
    ]
    assert report["runs"]["filter"]["summary_by_tolerance_mm"]["3"]["proposal_misses"]["count"] == 0
    assert "detector.py" in report["manifest"]["evaluator_source_sha256"]
    assert report["runs"]["baseline"]["prediction_source"] is None
    assert compare(report["manifest"]) == report


@pytest.mark.parametrize("provenance,complete", [
    ("candidate_pseudo", True), ("analytic_synthetic", True), ("complete_expert", False),
])
def test_nonexpert_or_incomplete_reference_cannot_promote(bundle, tmp_path, provenance, complete):
    bundle["reference_provenance"], bundle["references_complete"] = provenance, complete
    report = compare(bundle, tmp_path)
    assert not report["comparisons"]["filter"]["promotion"]["pass"]
    assert "insufficient_reference_provenance" in report["comparisons"]["filter"]["promotion"]["reasons"]
    assert "accuracy" not in report["metric_scope"]
    if provenance == "candidate_pseudo":
        assert report["metric_scope"] == "candidate_pseudo_agreement"
        assert report["references_complete"] is False


def test_existing_pseudo_provenance_cannot_be_laundered(bundle, tmp_path):
    references = json.loads((tmp_path / "refs.json").read_text())
    references[0]["provenance"] = {
        "kind": "pseudo_reference_from_candidate_labels", "labellers": ["claude"],
        "limitations": ["Unproposed branches absent"],
    }
    write(tmp_path / "refs.json", references)
    report = compare(bundle, tmp_path)
    assert any("pseudo" in e["error"] for e in report["reference_errors"])
    assert not report["comparisons"]["filter"]["promotion"]["pass"]
    bundle["reference_provenance"] = "candidate_pseudo"
    report = compare(bundle, tmp_path)
    assert report["references"]["subject001"]["_source"]["provenance"]["labellers"] == ["claude"]


def test_bundle_provenance_and_completeness_are_preserved(bundle, tmp_path):
    references = json.loads((tmp_path / "refs.json").read_text())
    write(tmp_path / "refs.json", {"cases": references, "references_complete": False, "dataset_sha256": "fixture"})
    report = compare(bundle, tmp_path)
    assert report["metric_scope"] == "incomplete_reference_agreement"
    assert report["references"]["subject001"]["_source"]["_bundle_metadata"]["dataset_sha256"] == "fixture"
    assert not report["comparisons"]["filter"]["promotion"]["pass"]


@pytest.mark.parametrize("exposure", ["training_cases", "tuning_cases", "inspected_cases", "pseudo_trained_cases"])
def test_expert_case_overlap_preserves_prior_exposure(bundle, tmp_path, exposure):
    bundle["histories"]["filter"][exposure] = ["1"]
    report = compare(bundle, tmp_path)
    overlap = report["runs"]["filter"]["overlap"]
    assert overlap["status"] == "contaminated"
    assert overlap["contaminated_reference_cases"] == ["subject001"]
    assert overlap["supplied_history"][exposure] == ["1"]
    assert "contaminated_evaluation" in report["comparisons"]["filter"]["promotion"]["reasons"]


def test_split_test_membership_does_not_erase_training_or_unknown_history():
    groups = {"subject001": "seed:1"}
    assert overlap_report(None, ["subject001"], groups)["status"] == "unknown"
    split = {"train": [], "validation": [], "test": ["subject001"]}
    assert overlap_report({"split": split}, ["subject001"], groups)["status"] == "unknown"
    assert overlap_report({"split": split, "history_complete": True}, ["subject001"], groups)["status"] == "no_overlap"
    history = {"split": split, "history_complete": True, "pseudo_trained_cases": ["subject001"]}
    assert overlap_report(history, ["subject001"], groups)["status"] == "contaminated"
    history = {"split": split, "history_complete": True, "training_groups": ["seed:1"]}
    assert overlap_report(history, ["subject001"], groups)["contaminated_reference_cases"] == ["subject001"]


def test_missing_unknown_and_malformed_predictions_do_not_disappear(bundle, tmp_path):
    (tmp_path / "variant" / "subject001.json").unlink()
    write(tmp_path / "variant" / "unknown.json", prediction("unknown", []))
    write(tmp_path / "variant" / "subject002.json", {"case_id": "subject002"})
    report = compare(bundle, tmp_path)
    errors = report["runs"]["filter"]["errors"]
    assert any("Unknown prediction case" in e["error"] for e in errors)
    assert report["runs"]["filter"]["missing_case_ids"] == ["subject001", "subject002"]
    assert report["comparisons"]["filter"]["excluded_case_ids"] == ["subject001", "subject002"]
    assert report["runs"]["filter"]["cases"]["subject001"]["status"] == "error"
    assert not report["comparisons"]["filter"]["promotion"]["pass"]


def test_missing_reference_is_reported_against_expected_case_inventory(bundle, tmp_path):
    bundle["expected_cases"] = ["subject001", "subject002", "subject003", "subject004"]
    report = compare(bundle, tmp_path)
    assert {"case_id": "subject004", "error": "Missing valid reference case."} in report["reference_errors"]
    assert not report["comparisons"]["filter"]["promotion"]["pass"]


def test_all_negative_cases_keep_fp_and_undefined_recall(bundle, tmp_path):
    write(tmp_path / "refs.json", [prediction(c, []) for c in ("subject001", "subject002", "subject003")])
    report = compare(bundle, tmp_path)
    summary = report["runs"]["baseline"]["summary_by_tolerance_mm"]["3"]
    assert (summary["true_positives"], summary["false_positives"], summary["false_negatives"]) == (0, 6, 0)
    assert summary["recall"] is None
    assert summary["negative_controls"]["cases_with_false_positives"] == 3
    interval = report["comparisons"]["filter"]["bootstrap_by_tolerance_mm"]["3"]["intervals"]["baseline"]["recall"]
    assert interval == {"low": None, "high": None, "valid_resamples": 0}
    assert "no_positive_references" in report["comparisons"]["filter"]["promotion"]["reasons"]
    json.dumps(report, allow_nan=False)


def test_duplicate_detections_match_one_reference_once(bundle, tmp_path):
    write(tmp_path / "variant" / "subject001.json", prediction("subject001", [branch("a"), branch("b", 0.1)]))
    report = compare(bundle, tmp_path)
    row = report["runs"]["filter"]["cases"]["subject001"]["by_tolerance"]["3"]
    assert (row["true_positives"], row["false_positives"], row["false_negatives"]) == (1, 1, 0)
    assert len({m["reference_id"] for m in row["matches"]}) == 1


def test_duplicate_reference_ids_and_normalized_case_ids_are_errors(bundle, tmp_path):
    write(tmp_path / "refs.json", [prediction("subject001", [branch("same"), branch("same", 10)])])
    report = compare(bundle, tmp_path)
    assert "Duplicate reference IDs" in report["reference_errors"][0]["error"]
    write(tmp_path / "refs.json", [prediction("1", []), prediction("subject001", [])])
    report = compare(bundle, tmp_path)
    assert "Duplicate normalized reference case" in report["reference_errors"][0]["error"]
    assert report["references"] == {}


def test_one_to_one_cardinality_and_no_angular_or_seed_gates(bundle, tmp_path):
    references = [prediction("subject001", [branch("r1", 3), branch("r2", 7)])]
    pred = prediction("subject001", [branch("p1", 0), branch("p2", 4)])
    pred["daughters"][0]["direction_xyz"] = [0, -1, 0]
    pred["daughters"][0]["seed_xyz_mm"] = [1000, 1000, 1000]
    write(tmp_path / "refs.json", references)
    write(tmp_path / "variant" / "subject001.json", pred)
    bundle["tolerance_mm"] = 4
    report = compare(bundle, tmp_path)
    row = report["runs"]["filter"]["cases"]["subject001"]["by_tolerance"]["4"]
    assert row["true_positives"] == 2
    assert row["matches"][0]["direction_error_degrees"] == 180


@pytest.mark.parametrize("point,polyline,expected", [
    ([5, 3, 0], [[0, 0, 0], [10, 0, 0]], 3),
    ([-2, 0, 0], [[0, 0, 0], [10, 0, 0]], 2),
    ([12, 0, 0], [[0, 0, 0], [10, 0, 0]], 2),
    ([5, 5, 2], [[0, 0, 0], [10, 10, 0]], 2),
    ([0, 3, 4], [[0, 0, 0], [0, 0, 0]], 5),
    ([7, 2, 0], [[0, 0, 0], [0, 0, 0], [10, 0, 0]], 2),
])
def test_seed_projects_to_segments_not_just_vertices(point, polyline, expected):
    assert seed_polyline_distance(point, polyline) == pytest.approx(expected)


@pytest.mark.parametrize("polyline", [[], [[1, 2, 3]], [[1, 2], [3, 4]], [[1, 2, 3], [np.nan, 0, 0]]])
def test_invalid_polylines_fail(polyline):
    with pytest.raises(ValueError, match="centreline"):
        seed_polyline_distance([0, 0, 0], polyline)


def test_geometry_uses_supplied_fields_and_physical_polyline(bundle, tmp_path):
    refs = json.loads((tmp_path / "refs.json").read_text())
    refs[0]["daughters"][0]["centreline_xyz_mm"] = [[0, 0, 0], [0, 10, 0]]
    refs[0]["daughters"][0]["radius_mm"] = 4
    refs[1]["daughters"] = [{"instance_id": "true", "ostium": [0, 0, 0], "direction": [0, 1, 0]}]
    refs[2]["daughters"] = [{"instance_id": "true", "ostium": [0, 0, 0], "seed": [0, 5, 0]}]
    write(tmp_path / "refs.json", refs)
    report = compare(bundle, tmp_path)
    cases = report["runs"]["filter"]["cases"]
    first = cases["subject001"]["by_tolerance"]["3"]["matches"][0]
    assert first["relative_radius_error"] == 0.5
    assert first["seed_to_reference_centreline_mm"] == 0
    second = cases["subject002"]["by_tolerance"]["3"]["matches"][0]
    assert second["seed_error_mm"] is None
    assert second["radius_error_mm"] is None
    assert second["relative_radius_error"] is None
    assert cases["subject003"]["by_tolerance"]["3"]["matches"][0]["direction_error_degrees"] is None
    metrics = report["runs"]["filter"]["summary_by_tolerance_mm"]["3"]["matched_errors_only"]
    assert metrics["radius_error_mm"] == {"n": 1, "mean": 2, "median": 2, "p95": 2, "geometric_mean": 2}
    assert metrics["seed_error_mm"]["n"] == 2


def test_voxel_organizer_ingestion_uses_image_origin_spacing_direction(tmp_path):
    image = sitk.Image(10, 10, 10, sitk.sitkInt16)
    image.SetSpacing((0.5, 2, 3))
    image.SetOrigin((10, -20, 100))
    image.SetDirection((0, -1, 0, 1, 0, 0, 0, 0, 1))
    raw = {
        "case": "1", "coordinate_space": "voxel",
        "branches": [{"id": "renal", "ostium": [2, 3, 1], "seed": [2, 6, 1], "diameter_mm": 6}],
    }
    write(tmp_path / "refs.json", raw)
    refs, errors, notes = ingest_references(tmp_path / "refs.json", "complete_expert", True, {"subject001": image})
    assert errors == []
    daughter = refs["subject001"]["daughters"][0]
    assert daughter["ostium_xyz_mm"] == [4, -19, 103]
    assert daughter["seed_xyz_mm"] == [-2, -19, 103]
    assert daughter["radius_mm"] == 3
    assert any("converted voxel" in note for note in notes)
    _, errors, _ = ingest_references(tmp_path / "refs.json", "complete_expert", True)
    assert "voxel coordinates need" in errors[0]["error"]


def test_seed_group_bootstrap_is_deterministic_and_preserves_pairing():
    rows = {
        "a": {"true_positives": 10, "false_positives": 4, "false_negatives": 1},
        "b": {"true_positives": 2, "false_positives": 2, "false_negatives": 7},
        "c": {"true_positives": 3, "false_positives": 8, "false_negatives": 1},
    }
    groups = {"a": "seed:1", "b": "seed:1", "c": "seed:2"}
    result = paired_bootstrap(rows, rows, groups, samples=200, seed=73)
    assert result == paired_bootstrap(dict(reversed(list(rows.items()))), rows, groups, samples=200, seed=73)
    assert result["units"] == 2
    for interval in result["intervals"]["paired_difference"].values():
        assert interval["low"] == interval["high"] == 0
    grouped = paired_bootstrap(rows, rows, dict.fromkeys(rows, "one"), samples=20)
    assert grouped["intervals"]["baseline"]["true_positives"]["low"] == 15
    assert grouped["intervals"]["baseline"]["true_positives"]["high"] == 15
    assert result["intervals"]["baseline"]["true_positives"]["low"] == 6
    assert result["intervals"]["baseline"]["true_positives"]["high"] == 24


def test_supplied_group_metadata_cannot_silently_become_independent_candidates(bundle, tmp_path):
    refs = json.loads((tmp_path / "refs.json").read_text())
    for ref in refs:
        ref["group_id"] = "synthetic:91"
    write(tmp_path / "refs.json", refs)
    report = compare(bundle, tmp_path)
    assert report["comparisons"]["filter"]["bootstrap_unit"] == "supplied_seed_group"
    assert report["comparisons"]["filter"]["bootstrap_by_tolerance_mm"]["3"]["units"] == 1
    assert "insufficient_independent_units" in report["comparisons"]["filter"]["promotion"]["reasons"]
    bundle["groups"] = {"subject001": "synthetic:91"}
    with pytest.raises(ValueError, match="cover exactly"):
        compare(bundle, tmp_path)


def test_fp_only_win_losing_true_branch_is_rejected(bundle, tmp_path):
    write(tmp_path / "variant" / "subject001.json", prediction("subject001", []))
    report = compare(bundle, tmp_path)
    row = report["comparisons"]["filter"]
    assert row["cases"]["subject001"]["3"]["newly_lost_reference_ids"] == ["true"]
    assert "true_branches_lost" in row["promotion"]["reasons"]
    assert not row["promotion"]["pass"]


def test_lost_reference_cannot_be_offset_by_recovery_and_proposal_misses_survive(bundle, tmp_path):
    refs = json.loads((tmp_path / "refs.json").read_text())
    refs[0]["daughters"] = [branch("first"), branch("missed", 20)]
    write(tmp_path / "refs.json", refs)
    write(tmp_path / "variant" / "subject001.json", prediction("subject001", [branch("recovered", 20)]))
    report = compare(bundle, tmp_path)
    diff = report["comparisons"]["filter"]["cases"]["subject001"]["3"]
    assert diff["newly_lost_reference_ids"] == ["first"]
    assert diff["newly_recovered_reference_ids"] == ["missed"]
    assert diff["delta"]["true_positives"] == 0
    row = report["runs"]["filter"]["cases"]["subject001"]["by_tolerance"]["3"]
    assert row["proposal_misses"] == ["missed"]
    assert report["runs"]["filter"]["by_family"]["straight"]["3"]["proposal_misses"]["count"] == 1
    assert not report["comparisons"]["filter"]["promotion"]["pass"]


def test_tolerance_sensitivity_catches_localization_loss(bundle, tmp_path):
    write(tmp_path / "variant" / "subject001.json", prediction("subject001", [branch("true", 2.5)]))
    report = compare(bundle, tmp_path)
    sensitivity = report["runs"]["filter"]["cases"]["subject001"]["by_tolerance"]
    assert [sensitivity[t]["true_positives"] for t in ("2", "3", "5")] == [0, 1, 1]
    assert "true_branches_lost" in report["comparisons"]["filter"]["promotion"]["reasons"]


@pytest.mark.parametrize("mutation,reason", [
    ("runtime", "compute_unverified_or_overrun"),
    ("memory", "compute_unverified_or_overrun"),
    ("cores", "compute_unverified_or_overrun"),
    ("missing_runtime", "compute_unverified_or_overrun"),
    ("linux", "compute_unverified_or_overrun"),
    ("unknown_history", "unknown_training_or_tuning_history"),
])
def test_promotion_guard_failures(bundle, tmp_path, mutation, reason):
    report = bundle["runtime_reports"]["filter"]
    if mutation == "runtime":
        report["cases"]["subject001"]["runtime_s"] = 61
    elif mutation == "memory":
        report["cases"]["subject001"]["peak_rss_mb"] = 8001
    elif mutation == "cores":
        report["cpu_cores"] = 5
    elif mutation == "linux":
        report["platform"] = "Linux"
    elif mutation == "missing_runtime":
        del bundle["runtime_reports"]
    else:
        del bundle["histories"]
    result = compare(bundle, tmp_path)
    assert reason in result["comparisons"]["filter"]["promotion"]["reasons"]


def test_input_mutation_prevents_reusing_frozen_manifest(bundle, tmp_path):
    manifest = compare(bundle, tmp_path)["manifest"]
    write(tmp_path / "variant" / "subject001.json", prediction("subject001", []))
    with pytest.raises(ValueError, match="input hashes changed"):
        compare(manifest)


def test_cli_replays_manifest_and_preserves_existing_artifacts(bundle, tmp_path):
    write(tmp_path / "manifest.json", bundle)
    script = Path(__file__).resolve().parents[1] / "research_validation.py"
    command = [sys.executable, str(script), "--manifest", str(tmp_path / "manifest.json"), "--output", str(tmp_path / "report.json")]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    original = (tmp_path / "report.json").read_bytes()
    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode == 1
    assert "already exists" in repeated.stderr
    assert (tmp_path / "report.json").read_bytes() == original
    (tmp_path / "variant" / "subject001.json").unlink()
    command[-1] = str(tmp_path / "missing.json")
    assert subprocess.run(command, capture_output=True, text=True).returncode == 2
    missing = json.loads((tmp_path / "missing.json").read_text())
    assert missing["runs"]["filter"]["errors"]


def test_reference_groups_cannot_be_fragmented_to_inflate_sample_count(bundle, tmp_path):
    refs = json.loads((tmp_path / "refs.json").read_text())
    for ref in refs:
        ref["group_id"] = "synthetic:1"
    write(tmp_path / "refs.json", refs)
    bundle["groups"] = {r["case_id"]: r["case_id"] for r in refs}
    with pytest.raises(ValueError, match="split an existing"):
        compare(bundle, tmp_path)


@pytest.mark.parametrize("change,message", [
    ({"coordinate_space": "RAS"}, "coordinate convention"),
    ({"parent": {"instance_id": "renal"}}, "direct aortic daughters"),
])
def test_unsupported_organizer_geometry_fails_explicitly(bundle, tmp_path, change, message):
    refs = json.loads((tmp_path / "refs.json").read_text())
    refs[0].update(change)
    write(tmp_path / "refs.json", refs)
    report = compare(bundle, tmp_path)
    assert any(message in e["error"] for e in report["reference_errors"])
    assert not report["comparisons"]["filter"]["promotion"]["pass"]


def test_zero_reference_radius_is_an_error_not_a_normalized_placeholder(bundle, tmp_path):
    refs = json.loads((tmp_path / "refs.json").read_text())
    refs[0]["daughters"][0]["radius_mm"] = 0
    write(tmp_path / "refs.json", refs)
    report = compare(bundle, tmp_path)
    assert "positive and finite" in report["reference_errors"][0]["error"]


def test_voxel_direction_convention_must_be_explicit(tmp_path):
    raw = prediction("subject001", [branch("true")])
    raw["coordinate_space"] = "voxel"
    write(tmp_path / "refs.json", raw)
    _, errors, _ = ingest_references(
        tmp_path / "refs.json", "complete_expert", True, {"subject001": sitk.Image(10, 10, 10, sitk.sitkInt16)},
    )
    assert "direction_coordinate_space" in errors[0]["error"]


def test_cli_output_cannot_pollute_prediction_inputs(bundle, tmp_path):
    write(tmp_path / "manifest.json", bundle)
    script = Path(__file__).resolve().parents[1] / "research_validation.py"
    output = tmp_path / "variant" / "report.json"
    result = subprocess.run(
        [sys.executable, str(script), "--manifest", str(tmp_path / "manifest.json"), "--output", str(output)],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "outside" in result.stderr
    assert not output.exists()


def test_absent_and_malformed_reference_files_preserve_failure(bundle, tmp_path):
    (tmp_path / "refs.json").unlink()
    report = compare(bundle, tmp_path)
    assert report["reference_errors"]
    assert report["references_complete"] is False
    assert not report["comparisons"]["filter"]["promotion"]["pass"]
    write(tmp_path / "refs.json", {"cases": [{"daughters": []}], "dataset_sha256": "fixture"})
    report = compare(bundle, tmp_path)
    assert "no case identifier" in report["reference_errors"][0]["error"]


def test_malformed_runtime_case_rejects_instead_of_crashing(bundle, tmp_path):
    bundle["runtime_reports"]["filter"]["cases"]["subject001"] = None
    report = compare(bundle, tmp_path)
    assert report["runs"]["filter"]["compute"]["status"] == "rejected"
    assert "compute_unverified_or_overrun" in report["comparisons"]["filter"]["promotion"]["reasons"]


@pytest.mark.parametrize("field", ["runtime_reports", "compute_budget"])
def test_malformed_compute_evidence_cannot_pass(bundle, tmp_path, field):
    if field == "runtime_reports":
        bundle[field]["filter"] = []
    else:
        bundle[field] = []
    report = compare(bundle, tmp_path)
    assert report["runs"]["filter"]["compute"]["status"] == "rejected"
    assert not report["comparisons"]["filter"]["promotion"]["pass"]


def test_case_exposure_contaminates_sibling_reference_seed_groups():
    groups = {"subject001": "seed:1", "subject002": "seed:1", "subject003": "seed:2"}
    history = {
        "history_complete": True, "training_cases": [], "tuning_cases": [],
        "inspected_cases": ["subject001"],
    }
    overlap = overlap_report(history, sorted(groups), groups)
    assert overlap["contaminated_reference_cases"] == ["subject001", "subject002"]
    assert overlap["group_overlap"] == ["seed:1"]


def test_explicit_synthetic_provenance_cannot_be_laundered(bundle, tmp_path):
    refs = json.loads((tmp_path / "refs.json").read_text())
    refs[0]["provenance"] = {"kind": "analytic_synthetic", "seed": 7}
    write(tmp_path / "refs.json", refs)
    report = compare(bundle, tmp_path)
    assert "Synthetic reference provenance" in report["reference_errors"][0]["error"]
    assert not report["comparisons"]["filter"]["promotion"]["pass"]
