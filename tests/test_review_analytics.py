import json

import pytest

from learning import FEATURE_NAMES
from review_analytics import agreement, identity, review_index


def review(case="case", instance="old_id", origin=(1, 2, 3)):
    vector = [0.5] * len(FEATURE_NAMES)
    return {
        "case_id": case, "instance_id": instance, "label": "confirmed", "features": vector,
        "fingerprint": json.dumps([origin, [4, 5, 6], [1, 0, 0], 0.5, vector]),
    }


def test_identity_ignores_ids_but_never_transfers_a_label_across_geometry_or_cases():
    row = review()
    lookup = review_index([row])
    same = review(instance="new_id")
    assert lookup[(same["case_id"], identity(same["fingerprint"], same["features"]))] == row
    changed = review(origin=(1.000001, 2, 3))
    assert ("case", identity(changed["fingerprint"], changed["features"])) not in lookup
    assert ("other_case", identity(same["fingerprint"], same["features"])) not in lookup


def test_mismatched_or_nonfinite_fingerprints_fail_instead_of_becoming_negative_labels():
    row = review()
    changed = row["features"].copy()
    changed[0] = 0.6
    with pytest.raises(ValueError, match="match the stored"):
        identity(row["fingerprint"], changed)
    with pytest.raises(ValueError, match="finite"):
        identity(review(origin=(float("nan"), 2, 3))["fingerprint"], row["features"])


def test_duplicate_geometry_with_different_branch_ids_is_ambiguous():
    with pytest.raises(ValueError, match="Ambiguous"):
        review_index([review(instance="one"), review(instance="two")])


def test_zero_denominators_and_confusion_cells_are_explicit():
    assert agreement([], "model")["confirmed_retention"] is None
    rows = [
        {"label": "confirmed", "model_keep": True},
        {"label": "confirmed", "model_keep": False},
        {"label": "rejected", "model_keep": True},
        {"label": "rejected", "model_keep": False},
    ]
    result = agreement(rows, "model")
    assert result["reviewed_candidates"] == 4
    assert result["confirmed_retained"] == result["confirmed_removed"] == 1
    assert result["rejected_retained"] == result["rejected_removed"] == 1
    assert result["agreement"] == result["confirmed_retention"] == result["rejected_removal"] == 0.5
