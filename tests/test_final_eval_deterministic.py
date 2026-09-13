from dataclasses import asdict
import sys

import numpy as np
import pytest

from detector import DetectorConfig, detect, detect_pool
from final_eval_deterministic import CandidateAudit, ceiling, matrix, selection, size_status
from final_evaluation import CASES, score_variant
from synthetic import generate_case


def test_expanded_matrix_is_finite_deduplicated_and_preserves_challenge_geometry():
    specs = matrix()
    assert len(specs) == 26
    assert len({spec.name for spec in specs}) == len(specs)
    settings = {(spec.engine, spec.method, str(asdict(spec.config))) for spec in specs}
    assert len(settings) == len(specs)
    assert all(spec.config.minimum_path_mm == 5 and spec.config.trace_length_mm == 10 for spec in specs)
    for engine, profile in (("detect", "strict"), ("union", "review")):
        grid = {
            (spec.config.support_contrast_fraction, spec.config.native_contrast_scale)
            for spec in specs
            if spec.engine == engine and spec.config.profile == profile
            and spec.config.spacing_mm == 1 and spec.config.parallel_clearance_mm == 0
        }
        assert grid == {(fraction, native) for fraction in (0.3, 0.5, 0.7) for native in (0, 1.2)}
    standalone = [spec for spec in specs if spec.config.spacing_mm != 1 or spec.config.parallel_clearance_mm]
    assert len(standalone) == 6
    assert all(spec.config.support_contrast_fraction == 0.5 for spec in standalone)


@pytest.mark.parametrize("pool", [False, True])
def test_observational_audit_preserves_results_and_retains_trace_geometry(pool):
    case = generate_case(1)
    runner = detect_pool if pool else detect
    baseline = runner(case.image, case.parent).prediction("analytic")
    audit = CandidateAudit()
    previous = sys.getprofile()
    try:
        sys.setprofile(audit.profile)
        observed = runner(case.image, case.parent).prediction("analytic")
    finally:
        sys.setprofile(previous)
    assert observed == baseline
    assert len(audit.groups) == (2 if pool else 1)
    for group in audit.groups:
        assert [row["proposal_index"] for row in group["candidates"]] == list(range(len(group["candidates"])))
        assert all(row["trace_reason"] != "not_traced" for row in group["candidates"])
        assert sum(row["retained_after_resolve"] for row in group["candidates"]) > 0
    all_traces = ceiling({"case_id": "analytic", "proposal_audit": audit.groups})
    assert len(all_traces["daughters"]) >= len(observed["daughters"])
    assert all(branch["instance_id"].startswith("trace_") for branch in all_traces["daughters"])


def test_origin_unknown_and_borderline_are_not_confidently_undersized():
    assert size_status({}) == "unknown"
    assert size_status({"origin_diameter_mm": 1.3, "origin_diameter_upper_mm": 2.1}).startswith("borderline")
    assert size_status({"origin_diameter_mm": 1.1, "origin_diameter_upper_mm": 1.9}) == "confidently_below_2mm"
    assert size_status({"origin_diameter_mm": 2, "origin_diameter_upper_mm": 3}) == "at_least_2mm"
    assert DetectorConfig.review().minimum_origin_diameter_mm == 0


def test_shared_scoring_keeps_all_empty_cases_and_unknown_radius_mask():
    predictions = {
        case: {"case_id": case, "parent": {"instance_id": "aorta"}, "daughters": []} for case in CASES
    }
    scores = score_variant(predictions)
    for score in scores.values():
        assert score["summary"]["cases"] == 5
        assert score["summary"]["false_negatives"] == 19
        assert score["summary"]["errors"]["radius_error_mm"]["n"] == 0
    with pytest.raises(ValueError, match="all five"):
        score_variant({CASES[0]: predictions[CASES[0]]})


def test_loco_selection_excludes_held_out_outcomes_and_ineligible_ceiling():
    def variant(name, successes, eligible=True):
        rows = [
            {"case_id": case, "true_positives": tp, "false_positives": 0, "false_negatives": 1 - tp,
             "daughter_counts": {"absolute_error": 1 - tp}, "matches": []}
            for case, tp in zip(CASES, successes)
        ]
        return {
            "name": name, "eligible_for_selection": eligible,
            "runtime": {case: {"runtime_s": 1} for case in CASES},
            "scores": {tolerance: {"cases": rows} for tolerance in ("2", "3", "5")},
        }

    first = variant("first", [0, 1, 1, 1, 1])
    second = variant("second", [1, 0, 0, 0, 0])
    excluded = variant("ceiling", np.ones(5, dtype=int), False)
    chosen = selection([first, second, excluded])
    assert chosen["folds"][0]["selected_name"] == "first"
    assert chosen["held_out_scores"]["3"]["cases"][0]["true_positives"] == 0
    assert "ceiling" not in chosen["eligible_variants"]
    first["scores"]["3"]["cases"][0]["true_positives"] = 100
    assert selection([first, second, excluded])["folds"][0]["selected_name"] == "first"
