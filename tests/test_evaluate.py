from itertools import combinations, permutations

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from evaluate import evaluate_case, summarize_cases


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


def test_assignment_cardinality_and_distance_against_exhaustive_search():
    rng = np.random.default_rng(8192)
    for predicted_count in range(1, 5):
        for expected_count in range(1, 5):
            for tolerance in (0.5, 2.0, 5.0):
                predicted = prediction(rng.uniform(0, 10, predicted_count))
                expected = prediction(rng.uniform(0, 10, expected_count))
                distances = cdist(
                    [p["ostium_xyz_mm"] for p in predicted["daughters"]],
                    [p["ostium_xyz_mm"] for p in expected["daughters"]],
                )
                best = (0, 0.0)
                for size in range(1, min(predicted_count, expected_count) + 1):
                    for rows in combinations(range(predicted_count), size):
                        for cols in permutations(range(expected_count), size):
                            costs = distances[rows, cols]
                            if np.all(costs <= tolerance):
                                best = min(best, (-size, float(costs.sum())))
                result = evaluate_case(predicted, expected, tolerance)
                assert result["true_positives"] == -best[0]
                assert sum(m["ostium_error_mm"] for m in result["matches"]) == pytest.approx(best[1])


def test_unmatched_ids_and_negative_controls_are_not_hidden():
    report = evaluate_case(prediction([0, 0.2, 50]), prediction([0, 100]), 3)
    assert report["unmatched_reference_ids"] == ["branch_001"]
    assert report["unmatched_prediction_ids"] == ["branch_001", "branch_002"]
    negative = evaluate_case(prediction([30]), prediction([]))
    summary = summarize_cases([report, negative, evaluate_case(prediction([]), prediction([]))])
    assert summary["negative_controls"] == {"cases": 2, "cases_with_false_positives": 1, "false_positives": 1}


def test_parent_id_and_tolerance_boundary_follow_contract():
    invalid = prediction([])
    invalid["parent"]["instance_id"] = "not_aorta"
    with pytest.raises(ValueError, match="aorta"):
        evaluate_case(invalid, prediction([]))
    assert evaluate_case(prediction([3]), prediction([0]), 3)["true_positives"] == 1
    assert evaluate_case(prediction([3.000001]), prediction([0]), 3)["true_positives"] == 0
