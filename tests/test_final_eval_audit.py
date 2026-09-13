import copy
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from final_eval_audit import (
    control_prediction, coordinate_controls, lps_affine, matching_controls,
    nifti_header, path_at_distance, release_path,
)
from final_evaluation import CASES, filtered, score_prediction, score_variant, summarize, write_json
from score_references import score


def test_rotated_nonzero_origin_and_voxel_direction_diagnostic():
    result = coordinate_controls()
    assert result["derived_direction_and_landmarks_pass"]
    assert result["supplied_voxel_direction_angle_error_degrees"] > 45
    assert not result["ras_marker_is_converted_to_lps"]
    assert not result["released_references_affected"]


def test_qform_sform_decode_rotated_nonzero_origin(tmp_path: Path):
    image = sitk.Image(7, 9, 11, sitk.sitkUInt8)
    image.SetOrigin((72, -114, 183))
    image.SetSpacing((0.7, 1.5, 2.3))
    image.SetDirection((0, -1, 0, 1, 0, 0, 0, 0, 1))
    path = tmp_path / "rotated.nii.gz"
    sitk.WriteImage(image, str(path))
    header = nifti_header(path)
    expected = np.diag([-1, -1, 1, 1]) @ lps_affine(image)
    assert header["qform_code"] > 0
    assert header["sform_code"] > 0
    assert np.allclose(header["qform_ras"], expected, rtol=0, atol=1e-5)
    assert np.allclose(header["sform_ras"], expected, rtol=0, atol=1e-5)
    assert header["shape_xyz"] == [7, 9, 11]
    assert header["spatial_units_code"] == 2


def test_unresolved_lfs_pointer_cannot_be_read_as_header(tmp_path: Path):
    path = tmp_path / "pointer.nii.gz"
    path.write_text("version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(ValueError, match="Unresolved LFS"):
        nifti_header(path)


def test_guide_seed_uses_curved_arc_not_straight_chord():
    guide = np.array([[0, 0, 0], [3, 0, 0], [3, 4, 0]], dtype=float)
    np.testing.assert_allclose(path_at_distance(guide, 5), [3, 2, 0])
    assert np.linalg.norm(path_at_distance(guide, 5)) < 5
    with pytest.raises(ValueError, match="contain"):
        path_at_distance(guide, 8)


def test_release_mapping_and_path_escape(tmp_path: Path):
    assert release_path("case_19/aorta19.nii.gz", tmp_path) == tmp_path / "eval/data/case_19/aorta19.nii.gz"
    assert release_path("REVIEWER_CHECKLIST.md", tmp_path) == tmp_path / "eval/docs/REVIEWER_CHECKLIST.md"
    with pytest.raises(ValueError, match="Invalid release"):
        release_path("../outside", tmp_path)


def test_matching_direct_parity_mirror_and_empty_denominators(tmp_path: Path):
    result = matching_controls(tmp_path)
    assert result["exhaustive_3d_assignments_checked"] == 48
    assert result["direct_shared_equal"]
    assert result["unknown_radius_error_is_null"]
    assert result["mirror_warning_triggered"]
    assert result["all_five_empty_false_negatives"] == 19
    assert result["missing_case_rejected"]
    assert result["filter_preserves_order_and_inclusive_threshold"]
    assert result["pooled_unknown_radius_observations"] == 6


@pytest.mark.parametrize("tolerance", [2.0, 3.0, 5.0])
def test_inclusive_tolerance_and_cardinality_before_cost(tolerance: float):
    reference = control_prediction([[0, 0, 0]])
    exact = control_prediction([[tolerance, 0, 0]])
    outside = control_prediction([[tolerance + 1e-6, 0, 0]])
    assert score_prediction(exact, reference, tolerance)["true_positives"] == 1
    assert score_prediction(outside, reference, tolerance)["true_positives"] == 0
    reference = control_prediction([[0, 0, 0], [tolerance, 0, 0]])
    prediction = control_prediction([[tolerance * 0.1, 0, 0], [-tolerance, 0, 0]])
    result = score_prediction(prediction, reference, tolerance)
    assert result["true_positives"] == 2
    assert {(m["prediction_id"], m["reference_id"]) for m in result["matches"]} == {
        ("control_0", "control_1"), ("control_1", "control_0"),
    }


@pytest.mark.parametrize("fault", ["nan", "radius", "duplicate", "case"])
def test_invalid_variants_raise_without_substitution(fault: str):
    predictions = {case: control_prediction([], case) for case in CASES}
    value = control_prediction([[1, 2, 3]], CASES[0])
    if fault == "nan":
        value["daughters"][0]["ostium_xyz_mm"][0] = float("nan")
    elif fault == "radius":
        value["daughters"][0]["radius_mm"] = 0
    elif fault == "duplicate":
        value["daughters"].append(copy.deepcopy(value["daughters"][0]))
    else:
        value["case_id"] = "incorrect"
    predictions[CASES[0]] = value
    with pytest.raises(ValueError):
        score_variant(predictions)


def test_filter_alignment_probability_validation_and_deep_copy():
    prediction = control_prediction([[1, 2, 3], [4, 5, 6]])
    for values, threshold in [([0.4], 0.5), ([np.nan, 0.4], 0.5), ([0.4, 1.1], 0.5), ([0.2, 0.5], 2)]:
        with pytest.raises(ValueError):
            filtered(prediction, values, threshold)
    retained = filtered(prediction, [0.5, 0.6], 0.6)
    retained["daughters"][0]["radius_mm"] = 9
    assert prediction["daughters"][1]["radius_mm"] == 2
    assert len(prediction["daughters"]) == 2


def test_unknown_radius_excluded_from_pooled_statistics():
    reference = control_prediction([[10, 20, 30], [40, 50, 60]])
    prediction = copy.deepcopy(reference)
    reference["daughters"][0]["radius_mm"] = None
    prediction["daughters"][0]["radius_mm"] = 99
    prediction["daughters"][1]["radius_mm"] = 3
    summary = summarize([score_prediction(prediction, reference)])
    assert summary["errors"]["radius_error_mm"]["n"] == 1
    assert summary["errors"]["radius_error_mm"]["mean"] == 1
    assert reference["daughters"][0]["radius_mm"] is None


def test_direct_scorer_missing_file_is_reported_but_omitted(tmp_path: Path):
    refs, predictions = tmp_path / "refs", tmp_path / "predictions"
    write_json(refs / "audit_control.json", control_prediction([[0, 0, 0]]))
    result = score(refs, predictions, None, [3])
    assert result["missing_predictions"] == ["audit_control"]
    assert result["cases"] == []
