import copy

import pytest

from final_evaluation import filtered, score_prediction, summarize


def fixture():
    daughter = {
        "instance_id": "branch_001", "parent_instance_id": "aorta",
        "ostium_xyz_mm": [0, 0, 0], "seed_xyz_mm": [5, 0, 0],
        "radius_mm": 2.0, "direction_xyz": [1, 0, 0],
    }
    prediction = {"case_id": "subject019", "parent": {"instance_id": "aorta"}, "daughters": [daughter]}
    reference = copy.deepcopy(prediction)
    reference["daughters"][0]["radius_mm"] = None
    reference["daughters"][0]["centerline_xyz_mm"] = [[0, 0, 0], [10, 0, 0]]
    return prediction, reference


def test_unknown_radius_is_excluded_without_losing_reference_or_other_metrics():
    prediction, reference = fixture()
    result = score_prediction(prediction, reference)
    summary = summarize([result])
    assert summary["true_positives"] == 1
    assert summary["errors"]["radius_error_mm"]["n"] == 0
    assert summary["errors"]["radius_error_mm"]["mean"] is None
    assert summary["errors"]["seed_to_reference_centreline_mm"]["mean"] == 0


def test_filter_alignment_is_strict_and_does_not_mutate_original():
    prediction, _ = fixture()
    with pytest.raises(ValueError, match="aligned"):
        filtered(prediction, [], 0.5)
    with pytest.raises(ValueError, match="probabilities"):
        filtered(prediction, [2], 0.5)
    assert not filtered(prediction, [0.2], 0.5)["daughters"]
    assert len(prediction["daughters"]) == 1


def test_empty_predictions_remain_false_negatives_and_count_errors():
    prediction, reference = fixture()
    prediction["daughters"] = []
    result = summarize([score_prediction(prediction, reference)])
    assert result["false_negatives"] == 1
    assert result["count_mae"] == 1
    assert result["f1"] == 0
