import json

import numpy as np
import pytest
import SimpleITK as sitk

from autolabel import apply_verdicts, candidate_record, render_candidate
from detector import DetectorConfig, detect_pool, prepare_roi
from learning import FEATURE_NAMES, load_reviews
from test_detector import phantom


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    image, mask = phantom()
    config = DetectorConfig.review()
    result = detect_pool(image, mask, config)
    grid, parent = prepare_roi(image, mask, config)
    ct = sitk.GetArrayFromImage(grid).astype(np.float32)
    records = [candidate_record(b) for b in result.branches]
    pool_dir = tmp_path_factory.mktemp("pool")
    case_dir = pool_dir / "phantom"
    case_dir.mkdir()
    (case_dir / "pool.json").write_text(json.dumps({"case_id": "phantom", "candidates": records}))
    return grid, ct, parent, records, pool_dir


def test_evidence_png_renders_for_every_pool_candidate(rendered):
    grid, ct, parent, records, _ = rendered
    assert len(records) >= 2
    for record in records:
        assert len(record["feature_vector"]) == len(FEATURE_NAMES)
        assert render_candidate(grid, ct, parent, record).startswith(b"\x89PNG")


def test_verdicts_land_in_the_training_schema_and_can_be_revised(rendered, tmp_path):
    _, _, _, records, pool_dir = rendered
    reviews = tmp_path / "reviews.json"
    first, second = records[0]["instance_id"], records[1]["instance_id"]
    written = apply_verdicts({"phantom": {first: "confirmed", second: "rejected"}}, pool_dir, reviews, "test")
    assert written == {"confirmed": 1, "rejected": 1, "cleared": 0}
    rows = load_reviews([reviews])
    assert {r["instance_id"]: r["label"] for r in rows} == {first: "confirmed", second: "rejected"}
    assert all(r["labeller"] == "test" and len(r["features"]) == len(FEATURE_NAMES) for r in rows)
    apply_verdicts({"phantom": {first: "unreviewed"}}, pool_dir, reviews, "test")
    assert [r["instance_id"] for r in load_reviews([reviews])] == [second]
    with pytest.raises(ValueError, match="not in the rendered pool"):
        apply_verdicts({"phantom": {"branch_999": "confirmed"}}, pool_dir, reviews, "test")
