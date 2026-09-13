import numpy as np

from current_e2e_fusion import fuse
from learning import CONTEXT_FEATURES


class RadiusScore:
    def scores(self, matrix):
        return np.asarray(matrix)[:, 0] / 10


def candidate(x, radius):
    return {
        "instance_id": "branch_001", "ostium_xyz_mm": [x, 0, 0],
        "seed_xyz_mm": [x, 5, 0], "direction_xyz": [0, 1, 0],
        "radius_mm": radius, "path_xyz_mm": [[x, 0, 0], [x, 5, 0], [x, 10, 0]],
        "evidence_score": 0.8, "mean_vesselness": 0.8,
        "features": dict.fromkeys(CONTEXT_FEATURES, 1.0),
    }


def test_low_scoring_review_representation_cannot_remove_retained_strict_branch():
    strict, review = {"branches": [candidate(0, 8)]}, {"branches": [candidate(1, 1)]}
    prediction, origins = fuse(strict, review, RadiusScore(), 0.5)
    assert len(prediction["daughters"]) == 1
    assert prediction["daughters"][0]["ostium_xyz_mm"] == [0, 0, 0]
    assert origins[0]["profile"] == "strict"
    assert strict["branches"][0]["instance_id"] == "branch_001"


def test_score_priority_chooses_high_scoring_representation_without_duplicate():
    strict, review = {"branches": [candidate(0, 5)]}, {"branches": [candidate(1, 9)]}
    prediction, origins = fuse(strict, review, RadiusScore(), 0.5, "score")
    assert len(prediction["daughters"]) == 1
    assert prediction["daughters"][0]["ostium_xyz_mm"] == [1, 0, 0]
    assert origins[0]["profile"] == "review"


def test_threshold_equality_and_current_three_mm_merge_boundary_are_preserved():
    strict, review = {"branches": [candidate(0, 5)]}, {"branches": [candidate(3, 5)]}
    prediction, _ = fuse(strict, review, RadiusScore(), 0.5)
    assert len(prediction["daughters"]) == 2
    assert len({row["instance_id"] for row in prediction["daughters"]}) == 2
    empty, _ = fuse(strict, review, RadiusScore(), 0.6)
    assert empty["daughters"] == []
