import numpy as np
import pytest
from scipy import ndimage as ndi
from scipy.special import expit

from detector import cross_section_radius, detect, wall_origin
from evaluate import evaluate_case, summarize_cases
from synthetic import generate_case


@pytest.mark.parametrize("scenario", range(1, 6))
def test_generated_training_geometries_have_complete_direct_daughter_recall(scenario):
    case = generate_case(scenario)
    result = detect(case.image, case.parent)
    report = evaluate_case(result.prediction(case.reference["case_id"]), case.reference, tolerance_mm=3)
    assert report["true_positives"] == 2
    assert report["false_positives"] == report["false_negatives"] == 0
    for match in report["matches"]:
        assert match["ostium_error_mm"] < 1.6
        assert match["seed_error_mm"] < 1.2
        assert match["radius_error_mm"] < 0.65
        assert match["direction_error_degrees"] < 18


@pytest.mark.parametrize("seed", [7, 43, 103])
def test_enhanced_vein_blob_calcification_and_crop_caps_are_negative_controls(seed):
    case = generate_case(0, seed)
    assert detect(case.image, case.parent).branches == []


def test_oblique_wall_origin_follows_the_vessel_instead_of_shortest_wall_distance():
    parent = np.indices((40, 40, 40))[2] <= 12
    signed = ndi.distance_transform_edt(~parent) - ndi.distance_transform_edt(parent)
    path = np.array([[19, 20, 15], [20, 20, 16], [21, 20, 17], [22, 20, 18], [23, 20, 19]], dtype=float)
    origin = wall_origin(path, parent, np.ones_like(parent), signed, 1)
    assert np.allclose(origin, [16.5, 20, 12.5])
    support = np.ones_like(parent)
    support[:, :, 14] = False
    assert wall_origin(path, parent, support, signed, 1) is None


@pytest.mark.parametrize("direction", [(1, 0, 0), (1, 2, 3), (0, 1, -1)])
def test_subvoxel_radius_uses_a_perpendicular_intensity_cross_section(direction):
    spacing = 0.5
    axis = np.asarray(direction, dtype=float)
    axis /= np.linalg.norm(axis)
    coordinates = np.moveaxis(np.indices((49, 49, 49)), 0, -1) * spacing - 12
    along = np.sum(coordinates * axis, axis=-1)
    distance = np.linalg.norm(coordinates - along[..., None] * axis, axis=-1)
    intensity = 25 + 300 * expit((2.3 - distance) / 0.2)
    measured = cross_section_radius(intensity, np.array([24, 24, 24]), axis, spacing, 175)
    assert measured == pytest.approx(2.3, abs=0.08)


def test_unbounded_or_low_contrast_cross_section_has_no_invented_radius():
    intensity = np.full((31, 31, 31), 320.0)
    for level in (175, 400):
        assert cross_section_radius(intensity, np.array([15, 15, 15]), np.array([0, 0, 1]), 1, level) is None


def test_benchmark_aggregates_counts_not_case_percentages_and_preserves_undefined_errors():
    summary = summarize_cases([
        {"true_positives": 1, "false_positives": 0, "false_negatives": 0, "matches": []},
        {"true_positives": 0, "false_positives": 9, "false_negatives": 3, "matches": []},
    ])
    assert summary["precision"] == 0.1
    assert summary["recall"] == 0.25
    assert summary["f1"] == pytest.approx(2 / 14)
    assert summary["matched_errors_only"]["ostium_error_mm"]["mean"] is None
    assert summarize_cases([])["f1"] is None
