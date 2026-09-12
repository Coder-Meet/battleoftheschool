import pytest

from evaluate import evaluate_case


def prediction(origins):
    return {
        "case_id": "phantom",
        "parent": {"instance_id": "aorta"},
        "daughters": [
            {
                "instance_id": f"branch_{i:03d}", "parent_instance_id": "aorta",
                "ostium_xyz_mm": [x, 0, 0], "seed_xyz_mm": [x, 5, 0],
                "radius_mm": 2, "direction_xyz": [0, 1, 0],
            }
            for i, x in enumerate(origins)
        ],
    }


def test_duplicate_detections_cannot_match_one_reference_twice():
    result = evaluate_case(prediction([0, 1, 30]), prediction([0, 60]))
    assert result["true_positives"] == 1
    assert result["false_positives"] == 2
    assert result["false_negatives"] == 1
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["recall"] == 0.5
    assert result["matches"][0]["direction_error_degrees"] == 0


def test_empty_predictions_preserve_false_negatives_without_nan():
    result = evaluate_case(prediction([]), prediction([0]))
    assert result["false_negatives"] == 1
    assert result["f1"] == 0
    assert result["precision"] is None
    assert evaluate_case(prediction([]), prediction([]))["f1"] is None


def test_matching_maximizes_valid_matches_before_minimizing_distance():
    result = evaluate_case(prediction([0, 4]), prediction([3, 7]), tolerance_mm=4)
    assert result["true_positives"] == 2


def test_wrong_case_and_nonphysical_predictions_are_rejected():
    wrong = prediction([0])
    wrong["case_id"] = "other"
    with pytest.raises(ValueError, match="case IDs"):
        evaluate_case(prediction([0]), wrong)
    invalid = prediction([0])
    invalid["daughters"][0]["direction_xyz"] = [0, 0, 0]
    with pytest.raises(ValueError, match="unit vectors"):
        evaluate_case(invalid, prediction([0]))
