import json

import numpy as np

from explorer import CaseData
from learning import FEATURE_NAMES
from review_page import ReviewLedger, page_case, render_candidate


def branch(instance="branch_001", radius=2.0):
    return {
        "instance_id": instance, "parent_instance_id": "aorta",
        "ostium_xyz_mm": [10.0, 10.0, 10.0], "seed_xyz_mm": [15.0, 10.0, 10.0],
        "radius_mm": radius, "direction_xyz": [1.0, 0.0, 0.0], "path_xyz_mm": [[10, 10, 10], [15, 10, 10]],
        "evidence_score": 0.8, "mean_vesselness": 0.5, "warnings": [],
        "features": {"path_hu_relative": 0.9, "bone_distance_mm": 12.0, "parent_angle_degrees": 80.0,
                     "arc_position": 0.5, "native_spacing_mm": 0.8, "connector_gap": 0.0,
                     "candidate_volume_mm3": 60.0},
        "feature_vector": [2.0, 0.5, 0.8, 5.0, 5.0, 1.0, 0.9, 12.0, 80.0, 0.5, 0.8, 0.0, 60.0],
    }


def case_data():
    shape = (24, 24, 24)
    ct = np.full(shape, 20, dtype="<i2")
    ct[:, 8:14, 8:14] = 330
    mask = np.zeros(shape, dtype=np.uint8)
    mask[:, 8:14, 8:14] = 1
    metadata = {
        "case_id": "subject_test", "size_xyz": [24, 24, 24], "origin_xyz": [0.0, 0.0, 0.0],
        "basis": np.eye(3).tolist(), "branches": [branch()],
    }
    return CaseData(metadata, ct.tobytes(), mask.tobytes())


def test_ledger_round_trips_in_the_training_schema_and_tracks_changed_candidates(tmp_path):
    ledger = ReviewLedger(tmp_path / "reviews.json")
    ledger.set("subject_test", branch(), "confirmed")
    ledger.set("subject_test", branch("branch_002"), "rejected")
    assert ledger.counts("subject_test") == {"confirmed": 1, "rejected": 1}
    reloaded = ReviewLedger(tmp_path / "reviews.json")
    assert reloaded.status("subject_test", branch()) == "confirmed"
    assert reloaded.status("subject_test", branch(radius=3.0)) == "unreviewed"
    payload = json.loads((tmp_path / "reviews.json").read_text())
    assert payload["scope"] == "candidate_reviews_only" and payload["feature_names"] == FEATURE_NAMES
    assert all(len(row["features"]) == len(FEATURE_NAMES) for row in payload["records"])
    reloaded.set("subject_test", branch(), "unreviewed")
    assert reloaded.counts("subject_test") == {"confirmed": 0, "rejected": 1}


def test_candidate_crops_render_and_case_page_lists_every_candidate(tmp_path):
    case = case_data()
    png = render_candidate(case, branch())
    assert png.startswith(b"\x89PNG")
    assert render_candidate(case, branch()) is png
    ledger = ReviewLedger(tmp_path / "reviews.json")

    class Store:
        class config:
            profile = "review"

    html = page_case(Store(), ledger, case).decode()
    assert "branch_001" in html and "/review/subject_test/branch_001.png" in html and "1 pending" in html
