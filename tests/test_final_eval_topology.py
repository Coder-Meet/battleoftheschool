from dataclasses import asdict

import numpy as np
import pytest
import SimpleITK as sitk

import detector
from detector import DetectorConfig, enhance, normalize, propose
from final_eval_topology import (
    baseline_configs, experiment_settings, index_points, proximal_roots, traced_resolve,
)
from final_evaluation import filtered, score_prediction
from synthetic import generate_case


@pytest.mark.parametrize("scenario", [0, 1, 4, 5])
@pytest.mark.parametrize("config", list(baseline_configs().values()))
def test_instrumented_resolver_exactly_matches_frozen_source(scenario, config):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    case = generate_case(scenario, seed=87531)
    scan = normalize(case.image, case.parent, config)
    evidence = enhance(scan, config)
    candidates, _ = propose(scan, evidence, config)
    expected, counts = detector.resolve(scan, evidence, candidates, config)
    actual, records, rejections = traced_resolve(scan, evidence, candidates, config)
    assert [asdict(branch) for branch in actual] == [asdict(branch) for branch in expected]
    assert counts == rejections
    assert len(records) == len(candidates)
    assert sum(row["reason"] == "accepted" for row in records) == len(actual)


def test_alternate_roots_are_bounded_proximal_and_reference_free():
    case = generate_case(4, seed=87531)
    config = DetectorConfig(roots_per_contact=6)
    scan = normalize(case.image, case.parent, config)
    evidence = enhance(scan, config)
    candidates, _ = proximal_roots(scan, evidence, config, 2)
    assert candidates
    for _, root, _ in candidates:
        assert scan.outside[tuple(root)] <= config.root_depth_mm
        assert evidence.support[tuple(root)]
    for _, first, _ in candidates:
        for _, second, _ in candidates:
            if not np.array_equal(first, second):
                assert np.linalg.norm(first - second) * config.spacing_mm >= 2
    assert len(candidates) <= 12
    with pytest.raises(ValueError, match="2 mm"):
        proximal_roots(scan, evidence, config, 1.5)


def test_physical_lps_index_roundtrip_with_oblique_anisotropic_grid():
    case = generate_case(3, seed=87531)
    points = np.array([[5.25, 8.5, 12.75], [19, 21, 25]])
    world = np.array([case.image.TransformContinuousIndexToPhysicalPoint(p.tolist()) for p in points])
    assert np.allclose(index_points(case.image, world), points[:, ::-1])


def test_empty_outputs_and_unknown_radii_are_scored_without_fabrication():
    case = generate_case(1, seed=87531)
    reference = case.reference
    reference["daughters"][0]["radius_mm"] = None
    prediction = {"case_id": reference["case_id"], "parent": reference["parent"],
                  "daughters": [dict(row) for row in reference["daughters"]]}
    prediction["daughters"][0]["radius_mm"] = 2
    score = score_prediction(prediction, reference)
    assert score["matches"][0]["radius_error_mm"] is None
    empty = filtered(prediction, [0, 0], 0.5)
    assert empty["daughters"] == []
    score = score_prediction(empty, reference)
    assert score["false_negatives"] == 2


def test_all_experiments_preserve_origin_policy_and_are_post_reference():
    for settings in experiment_settings().values():
        config = DetectorConfig(**settings["detector"])
        assert config.minimum_origin_diameter_mm == 2
        assert config.minimum_path_mm == 5
        assert config.trace_length_mm == 10
        assert settings["development_status"] == "POST-REFERENCE DEVELOPMENT"
