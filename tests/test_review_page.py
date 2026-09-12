from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread

import numpy as np
import pytest

from explorer import CaseData, CaseStore, make_handler
from learning import FEATURE_NAMES
from review_page import ReviewLedger, _orientation, page_case, render_candidate


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


def test_pending_counts_ignore_changed_and_removed_candidates(tmp_path):
    ledger = ReviewLedger(tmp_path / "reviews.json")
    ledger.set("subject_test", branch(), "confirmed")
    ledger.set("subject_test", branch("branch_002"), "rejected")
    assert ledger.counts("subject_test", [branch(radius=3)]) == {"confirmed": 0, "rejected": 0}
    assert ledger.counts("subject_test", [branch()]) == {"confirmed": 1, "rejected": 0}


@pytest.mark.parametrize("basis,expected", [
    (np.eye(3), {"x_right": "L", "y_up": "P", "z_up": "S"}),
    (np.diag([-2, -2, -2]), {"x_right": "R", "y_up": "A", "z_up": "I"}),
    (np.array([[0, -2, 0], [2, 0, 0], [0, 0, 2]]), {"x_right": "P", "y_up": "R", "z_up": "S"}),
])
def test_review_orientation_respects_rotated_and_reflected_headers(basis, expected):
    assert _orientation({"basis": basis.tolist()}) == expected


def test_failed_disk_commit_preserves_previous_verdicts(tmp_path, monkeypatch):
    ledger = ReviewLedger(tmp_path / "reviews.json")
    ledger.set("subject_test", branch(), "confirmed")
    previous = ledger.path.read_bytes()

    def fail_replace(self, target):
        raise OSError("Read-only filesystem")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="Read-only"):
        ledger.set("subject_test", branch(), "rejected")
    assert ledger.path.read_bytes() == previous
    assert ledger.status("subject_test", branch()) == "confirmed"


def test_review_http_rejects_malformed_requests_without_losing_the_ledger(tmp_path):
    store = CaseStore(tmp_path)
    store.cache["subject_test"] = case_data()
    ledger = ReviewLedger(tmp_path / "reviews.json")
    http = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store, tmp_path, ledger))
    thread = Thread(target=http.serve_forever, daemon=True)
    thread.start()
    connection = HTTPConnection(*http.server_address[:2], timeout=10)
    try:
        for path, payload, expected in [
            ("/review", {}, 404),
            ("/review/subject_test/verdict", [], 400),
            ("/review/subject_test/verdict", {"instance_id": "branch_001", "label": "confirmed"}, 200),
        ]:
            connection.request("POST", path, json.dumps(payload), {"Content-Type": "application/json"})
            response = connection.getresponse()
            assert response.status == expected, response.read()
            response.read()
        assert ReviewLedger(ledger.path).status("subject_test", branch()) == "confirmed"
    finally:
        connection.close()
        http.shutdown()
        http.server_close()
        store.executor.shutdown(wait=True)
        thread.join(timeout=5)
