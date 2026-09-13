import hashlib
import json

import pytest

from benchmark import benchmark
from detector import Branch, Detection, DetectorConfig
from learning import CandidateModel, FEATURE_NAMES, load_reviews
from synthetic import generate_case, write_case
from synthetic_reviews import export_reviews, label_candidates


def branch(identifier, x):
    return Branch(
        identifier, (x, 0, 0), (x + 5, 0, 0), 2, (1, 0, 0),
        [(x, 0, 0), (x + 10, 0, 0)], 0.8, 0.5,
        features={
            "path_hu_relative": 1, "bone_distance_mm": 20, "parent_angle_degrees": 90,
            "arc_position": 0.5, "native_spacing_mm": 1, "connector_gap": 0,
            "candidate_volume_mm3": 100,
        },
    )


def detection(branches):
    return Detection(branches, {"total_s": 0}, {}, len(branches), {}, [], DetectorConfig())


def test_duplicate_and_borderline_candidates_are_excluded_and_misses_stay_visible():
    reference = detection([branch("ref1", 0), branch("ref2", 20)]).prediction("case")
    result = detection([
        branch("p1", 0), branch("duplicate", 0.2), branch("borderline", 4), branch("negative", 100),
    ])
    rows, report = label_candidates(result, reference)
    assert [(r["instance_id"], r["label"]) for r in rows] == [
        ("p1", "confirmed"), ("negative", "rejected"),
    ]
    assert report["false_negatives"] == 1
    assert report["false_positives"] == 3
    assert report["unmatched_reference_ids"] == ["ref2"]
    assert report["excluded_ambiguous_candidate_ids"] == ["duplicate", "borderline"]
    assert rows[0]["reference_instance_id"] == "ref1"
    assert all(r["labeller"] == "analytic_synthetic_geometry" for r in rows)


def test_empty_reference_labels_candidates_negative_and_empty_prediction_retains_misses():
    result = detection([branch("p1", 0)])
    empty = detection([]).prediction("case")
    rows, report = label_candidates(result, empty)
    assert rows[0]["label"] == "rejected"
    assert report["false_positives"] == 1
    rows, report = label_candidates(detection([]), result.prediction("case"))
    assert rows == []
    assert report["false_negatives"] == 1
    for tolerance, exclusion in [(0, 5), (3, 3), (3, float("nan"))]:
        with pytest.raises(ValueError):
            label_candidates(result, empty, tolerance, exclusion)


@pytest.fixture
def bundle(tmp_path):
    case = generate_case(1, 2026)
    entry = write_case(case, tmp_path / case.reference["case_id"])
    (tmp_path / "manifest.json").write_text(json.dumps({
        "source": "analytic_synthetic_geometry", "cases": [entry],
    }))
    return tmp_path


def test_export_is_trainable_and_protects_source_hashes_and_existing_outputs(bundle):
    output = bundle / "reviews.json"
    payload = export_reviews(bundle, output, profile="strict")
    rows = load_reviews([output])
    assert len(rows) == 2
    assert all(r["label"] == "confirmed" for r in rows)
    assert payload["cases"][0]["false_negatives"] == 0
    assert payload["manifest_sha256"] == hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        export_reviews(bundle, output)
    manifest = json.loads((bundle / "manifest.json").read_text())
    case_id = manifest["cases"][0]["case_id"]
    (bundle / case_id / "reference.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        export_reviews(bundle, bundle / "corrupt.json")
    manifest["source"] = "unreviewed_proposals_not_ground_truth"
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Only analytic"):
        export_reviews(bundle, bundle / "unreviewed.json")


def test_model_benchmark_reports_full_discovery_loss_and_preserves_baseline(bundle):
    model = CandidateModel(
        [0] * len(FEATURE_NAMES), [1] * len(FEATURE_NAMES), [0] * len(FEATURE_NAMES),
        0, 1, {"train": ["train"], "validation": ["validation"], "test": ["test"]},
    )
    model_path = bundle / "model.json"
    model.save(model_path)
    report = benchmark(bundle, bundle / "metrics.json", 3, candidate_model=model_path)
    assert report["unfiltered_summary"]["true_positives"] == 2
    assert report["summary"]["true_positives"] == 0
    assert report["summary"]["false_negatives"] == 2
    assert report["candidate_model"]["sha256"] == hashlib.sha256(model_path.read_bytes()).hexdigest()
    prediction = next(iter(report["predictions"].values()))
    assert len(prediction["unfiltered_prediction"]["daughters"]) == 2
    assert prediction["prediction"]["daughters"] == []
    assert prediction["candidate_model"]["rejected"] == 2
