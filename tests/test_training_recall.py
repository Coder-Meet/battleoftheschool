from copy import deepcopy

import pytest

from learning import FEATURE_NAMES, train


def shifted_reviews():
    return [
        {
            "case_id": f"case{case}", "instance_id": f"branch_{label}",
            "label": "confirmed" if label else "rejected",
            "labeller": "analytic_synthetic_geometry" if case < 4 else "claude",
            "features": [
                (2 if label else 1) if case == 4 else (0.4 if label else -0.4),
                *([0] * (len(FEATURE_NAMES) - 1)),
            ],
        }
        for case in range(6) for label in (0, 1)
    ]


SPLIT = {"train": ["case0", "case1", "case2", "case3"], "validation": ["case4"], "test": ["case5"]}


def test_recall_constraint_prevents_shifted_validation_from_discarding_training_positives():
    rows = shifted_reviews()
    _, unconstrained = train(rows, SPLIT)
    model, constrained = train(rows, SPLIT, minimum_training_recall=0.95)
    assert unconstrained["partitions"]["train"]["classifier"]["recall"] == 0
    assert constrained["partitions"]["train"]["classifier"]["recall"] == 1
    assert constrained["threshold"] < unconstrained["threshold"]
    assert constrained["minimum_training_recall"] == 0.95
    assert constrained["label_sources"] == {"analytic_synthetic_geometry": 8, "claude": 4}
    changed = deepcopy(rows)
    for row in changed[-2:]:
        row["label"] = "rejected" if row["label"] == "confirmed" else "confirmed"
        row["features"][0] = 10000
    altered, _ = train(changed, SPLIT, minimum_training_recall=0.95)
    assert altered == model


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_recall_constraints_are_rejected(value):
    with pytest.raises(ValueError, match="Minimum training recall"):
        train(shifted_reviews(), SPLIT, minimum_training_recall=value)
