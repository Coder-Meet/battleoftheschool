import json
from pathlib import Path

import numpy as np
import pytest

from learning import CandidateModel, load_reviews, train


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("suffix", ["", "-synthetic"])
def test_committed_weights_reproduce_from_the_recorded_labels_and_split(suffix):
    label_paths = [ROOT / "labels/reviews.json"]
    if suffix:
        label_paths.extend([
            ROOT / "labels/synthetic/development-reviews.json",
            ROOT / "labels/synthetic/hard-five-reviews.json",
        ])
    committed = CandidateModel.load(ROOT / f"labels/candidate-model{suffix}.json")
    report_name = "model-synthetic-report.json" if suffix else "model-report.json"
    report = json.loads((ROOT / "labels" / report_name).read_text())
    reproduced, result = train(
        load_reviews(label_paths), committed.split, minimum_training_recall=report["minimum_training_recall"],
    )
    assert result["label_sources"] == report["label_sources"]
    assert reproduced.split == committed.split
    for saved, rebuilt in (
        (committed.mean, reproduced.mean),
        (committed.scale, reproduced.scale),
        (committed.weights, reproduced.weights),
    ):
        np.testing.assert_allclose(saved, rebuilt, rtol=0, atol=1e-12)
    assert reproduced.bias == pytest.approx(committed.bias, abs=1e-12)
    assert reproduced.threshold == pytest.approx(committed.threshold, abs=1e-12)
