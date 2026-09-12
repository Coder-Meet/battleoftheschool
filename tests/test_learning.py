import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from detector import Branch, Detection, DetectorConfig
from learning import CandidateModel, FEATURE_NAMES, features, filter_detection, load_reviews, split_cases, train


@pytest.fixture
def reviews():
    return [
        {
            "case_id": f"case{case}", "instance_id": f"branch_{label}",
            "label": "confirmed" if label else "rejected",
            "features": [2 + case * 0.01, 0.8 if label else 0.05, 0.9 if label else 0.4, 10, 5, 1],
        }
        for case in range(6) for label in (0, 1)
    ]


def partition():
    return {"train": ["case0", "case1", "case2", "case3"], "validation": ["case4"], "test": ["case5"]}


def test_training_uses_only_train_statistics_and_never_tunes_on_test(reviews):
    model, report = train(reviews, partition())
    assert np.allclose(model.mean, np.mean([r["features"] for r in reviews[:8]], axis=0))
    assert report["partitions"]["test"]["classifier"]["fp"] == 0
    assert report["partitions"]["test"]["keep_all_baseline"]["fp"] == 1
    for row in reviews[-2:]:
        row["label"] = "confirmed" if row["label"] == "rejected" else "rejected"
        row["features"][0] = 10000
    changed, _ = train(reviews, partition())
    assert changed == model


def test_group_split_is_deterministic_disjoint_and_rejects_leakage(reviews):
    split = split_cases(reviews, 42)
    assert split == split_cases(list(reversed(reviews)), 42)
    assert len({c for cases in split.values() for c in cases}) == 6
    split["test"] = split["train"]
    with pytest.raises(ValueError, match="leakage"):
        train(reviews, split)
    with pytest.raises(ValueError, match="six independent"):
        split_cases(reviews[:4], 42)


def test_training_requires_both_classes_and_all_cases(reviews):
    reviews[0]["case_id"] = "unknown"
    with pytest.raises(ValueError, match="exactly"):
        train(reviews, partition())
    reviews[0]["case_id"] = "case0"
    reviews[-1]["label"] = "rejected"
    with pytest.raises(ValueError, match="test needs both"):
        train(reviews, partition())


def test_model_round_trip_and_schema_validation(reviews, tmp_path):
    model, _ = train(reviews, partition())
    path = tmp_path / "model.json"
    model.save(path)
    loaded = CandidateModel.load(path)
    assert loaded == model
    assert np.allclose(loaded.scores([r["features"] for r in reviews]), model.scores([r["features"] for r in reviews]))
    value = json.loads(path.read_text())
    value["scale"][0] = 0
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="scale"):
        CandidateModel.load(path)
    with pytest.raises(ValueError, match="finite matrix"):
        model.scores([[float("nan")] * 6])


def test_review_loader_deduplicates_but_rejects_conflicting_or_nonfinite_rows(reviews, tmp_path):
    path = tmp_path / "reviews.json"
    payload = {"schema_version": 1, "scope": "candidate_reviews_only", "feature_names": FEATURE_NAMES, "records": reviews}
    path.write_text(json.dumps(payload))
    assert len(load_reviews([path, path])) == len(reviews)
    conflicting = tmp_path / "conflicting.json"
    reviews[0]["label"] = "confirmed"
    conflicting.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Conflicting"):
        load_reviews([path, conflicting])
    reviews[0]["features"][0] = float("nan")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="finite"):
        load_reviews([path])


def test_training_cli_and_inference_preserve_geometry(reviews, tmp_path):
    source = tmp_path / "reviews.json"
    source.write_text(json.dumps({
        "schema_version": 1, "scope": "candidate_reviews_only",
        "feature_names": FEATURE_NAMES, "records": reviews,
    }))
    split = tmp_path / "split.json"
    split.write_text(json.dumps(partition()))
    model_path, report_path = tmp_path / "model.json", tmp_path / "report.json"
    subprocess.run([
        sys.executable, str(Path(__file__).resolve().parents[1] / "learning.py"), "train",
        "--reviews", str(source), "--split", str(split), "--model", str(model_path), "--report", str(report_path),
    ], check=True, capture_output=True)
    branch = Branch(
        "branch_001", (1, 2, 3), (6, 2, 3), 2, (1, 0, 0),
        [(1, 2, 3), (6, 2, 3), (11, 2, 3)], 0.9, 0.8,
    )
    assert features(branch) == [2, 0.8, 0.9, 10, 5, 1]
    rejected = Branch(
        "branch_002", (1, 2, 3), (6, 2, 3), 2, (1, 0, 0),
        [(1, 2, 3), (6, 2, 3), (11, 2, 3)], 0.4, 0.05,
    )
    result = Detection([branch, rejected], {}, {}, 2, {}, [], DetectorConfig())
    original = branch.prediction()
    decisions = filter_detection(result, CandidateModel.load(model_path))
    assert decisions["rejected"] == 1
    assert result.prediction("case")["daughters"] == [original]
    assert filter_detection(Detection([], {}, {}, 0, {}, [], DetectorConfig()), CandidateModel.load(model_path))["scores"] == {}
