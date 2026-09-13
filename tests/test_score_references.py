import json

import numpy as np
import pytest
import SimpleITK as sitk

from score_references import load_references, normalize_case_id, normalize_reference, score


def branch(x, identifier="branch_001"):
    return {
        "instance_id": identifier, "parent_instance_id": "aorta",
        "ostium_xyz_mm": [x, 0.0, 0.0], "seed_xyz_mm": [x, 5.0, 0.0],
        "radius_mm": 2.0, "direction_xyz": [0.0, 1.0, 0.0],
    }


def prediction(case_id, xs):
    return {
        "case_id": case_id, "parent": {"instance_id": "aorta"},
        "daughters": [branch(x, f"branch_{i:03d}") for i, x in enumerate(xs)],
    }


@pytest.mark.parametrize(
    "value, expected",
    [("18", "subject018"), ("subject18", "subject018"), ("orig7.nii.gz", "subject007"),
     ("Subject_003", "subject003"), ("subject001", "subject001"), ("left renal", "left renal")],
)
def test_case_ids_are_normalized_to_repository_names(value, expected):
    assert normalize_case_id(value) == expected


def test_aliases_missing_seed_and_diameter_are_normalized_with_notes():
    notes = []
    reference = normalize_reference(
        {"case": "2", "branches": [
            {"name": "celiac", "ostium": {"x": 1, "y": 2, "z": 3}, "direction": [0, 0, 2], "diameter_mm": 6},
            {"origin": [4, 5, 6], "seed": [4, 5, 11]},
        ]},
        notes,
    )
    assert reference["case_id"] == "subject002"
    first, second = reference["daughters"]
    assert first["instance_id"] == "celiac"
    assert first["direction_xyz"] == [0, 0, 1]
    assert first["seed_xyz_mm"] == [1, 2, 8]
    assert first["radius_mm"] == 3
    assert second["direction_xyz"] == [0, 0, 1]
    assert second["_radius_known"] is False
    assert any("renormalized" in note for note in notes)
    assert any("derived from seed" in note for note in notes)
    assert any("no radius" in note for note in notes)


def test_voxel_indices_are_converted_with_image_geometry():
    image = sitk.Image(10, 10, 10, sitk.sitkInt16)
    image.SetSpacing((0.5, 0.5, 2.0))
    image.SetOrigin((-10.0, 20.0, 100.0))
    notes = []
    reference = normalize_reference(
        {"case_id": "subject001", "coordinate_space": "voxel",
         "daughters": [{"ostium": [2, 4, 1], "direction": [1, 0, 0]}]},
        notes, image,
    )
    assert reference["daughters"][0]["ostium_xyz_mm"] == [-9.0, 22.0, 102.0]
    with pytest.raises(ValueError, match="data-root"):
        normalize_reference(
            {"case_id": "subject001", "coordinate_space": "voxel",
             "daughters": [{"ostium": [2, 4, 1], "direction": [1, 0, 0]}]},
            [], None,
        )


def test_directory_and_bundle_reference_layouts_are_accepted(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(prediction("subject001", [0])))
    (tmp_path / "b.json").write_text(json.dumps([prediction("subject002", [0]), prediction("subject003", [])]))
    (tmp_path / "c.json").write_text(json.dumps({"subject004": {"daughters": [branch(0)]}}))
    (tmp_path / "d.json").write_text(json.dumps({"daughters": [branch(0)]}))
    assert sorted(r["case_id"] if "case_id" in r else None for r in load_references(tmp_path)) == [
        "d", "subject001", "subject002", "subject003", "subject004"
    ]


def test_score_reports_tolerances_missing_cases_and_mirror_warning(tmp_path):
    refs, preds = tmp_path / "refs", tmp_path / "preds"
    refs.mkdir(), preds.mkdir()
    (preds / "subject001.json").write_text(json.dumps(prediction("subject001", [0, 1.5, 40])))
    (refs / "subject001.json").write_text(json.dumps(prediction("subject001", [0, 4, 60])))
    (refs / "subject009.json").write_text(json.dumps(prediction("subject009", [0])))
    report = score(refs, preds, None, [2, 3, 5])
    case = report["cases"][0]
    assert case["by_tolerance"]["2"]["true_positives"] == 1
    assert case["by_tolerance"]["3"]["true_positives"] == 2
    assert case["by_tolerance"]["5"]["true_positives"] == 2
    assert case["by_tolerance"]["5"]["false_positives"] == 1
    assert report["missing_predictions"] == ["subject009"]
    assert report["mirror_warning"] is False
    assert report["summary_by_tolerance_mm"]["3"]["recall"] == pytest.approx(2 / 3)

    mirrored = prediction("subject001", [0, 2.5, 40])
    for daughter in mirrored["daughters"]:
        daughter["ostium_xyz_mm"] = [-daughter["ostium_xyz_mm"][0], -30.0, 0.0]
        daughter["seed_xyz_mm"] = [-daughter["seed_xyz_mm"][0], -25.0, 0.0]
    (preds / "subject001.json").write_text(json.dumps(mirrored))
    reference = prediction("subject001", [0, 2.5, 40])
    for daughter in reference["daughters"]:
        daughter["ostium_xyz_mm"][1] = 30.0
        daughter["seed_xyz_mm"][1] = 35.0
    (refs / "subject001.json").write_text(json.dumps(reference))
    report = score(refs, preds, None, [3])
    assert report["cases"][0]["by_tolerance"]["3"]["true_positives"] == 0
    assert report["cases"][0]["mirror_xy_true_positives"] == 3
    assert report["mirror_warning"] is True


def test_unknown_radius_is_excluded_from_radius_error(tmp_path):
    refs, preds = tmp_path / "refs", tmp_path / "preds"
    refs.mkdir(), preds.mkdir()
    (preds / "subject001.json").write_text(json.dumps(prediction("subject001", [0, 10])))
    (refs / "subject001.json").write_text(json.dumps({"case_id": "subject001", "daughters": [
        {"ostium": [0, 0, 0], "direction": [0, 1, 0]},
        {"ostium": [10, 0, 0], "direction": [0, 1, 0], "radius_mm": 2.5},
    ]}))
    report = score(refs, preds, None, [3])
    radius = report["summary_by_tolerance_mm"]["3"]["matched_errors_only"]["radius_error_mm"]
    assert radius["mean"] == pytest.approx(0.5)
    errors = [m["radius_error_mm"] for m in report["cases"][0]["by_tolerance"]["3"]["matches"]]
    assert sorted(errors, key=lambda e: (e is None, e)) == [0.5, None]
    assert np.isfinite(report["summary_by_tolerance_mm"]["3"]["matched_errors_only"]["seed_error_mm"]["mean"])
