import hashlib
import json

import numpy as np
import pytest
import SimpleITK as sitk

from detector import DetectorConfig, detect
from evaluate import validate_prediction
from paper_methods import METHODS, benchmark_papers, clean_wall_layers, detect_paper, proximal_pca_direction
from synthetic import generate_case


def test_outward_layer_chain_removes_attached_wall_sheet_but_keeps_radial_vessel():
    parent = np.indices((25, 25, 25))[2] <= 8
    support = parent.copy()
    support[10:15, 10:15, 9:20] = True
    support[3:10, 3:10, 9:11] = True
    original = support.copy()
    cleaned = clean_wall_layers(parent, support, 1.0)
    assert cleaned[parent].all()
    assert cleaned[12, 12, 9:20].all()
    assert not cleaned[3:8, 3:8, 9:11].any()
    assert np.array_equal(support, original)
    assert np.all(~cleaned | support)


def test_layer_rule_does_not_treat_long_wall_parallel_path_as_outward_clearance():
    parent = np.indices((25, 25, 25))[2] <= 8
    support = parent.copy()
    support[3:22, 11:14, 9:12] = True
    assert np.count_nonzero(clean_wall_layers(parent, support, 1.0) & ~parent) == 0
    assert np.count_nonzero(clean_wall_layers(parent, support, 2.0) & ~parent) > 0


@pytest.mark.parametrize("spacing,clearance", [(0, 5), (1, np.nan), (np.inf, 5), (1, -1)])
def test_invalid_layer_geometry_is_rejected(spacing, clearance):
    parent = np.indices((3, 3, 3))[2] == 0
    with pytest.raises(ValueError):
        clean_wall_layers(parent, parent, spacing, clearance)


def test_pca_is_physical_rotation_translation_and_sampling_density_invariant():
    path = np.column_stack([np.arange(11.0), np.zeros(11), np.zeros(11)])
    theta = 0.7
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    transformed = path @ rotation.T + [27, -42, 90]
    np.testing.assert_allclose(proximal_pca_direction(transformed, 2), rotation[:, 0], atol=1e-12)
    dense = np.repeat(transformed, 3, axis=0)
    np.testing.assert_allclose(proximal_pca_direction(dense, 2), rotation[:, 0], atol=1e-12)
    np.testing.assert_allclose(proximal_pca_direction(transformed[::-1], 2), -rotation[:, 0], atol=1e-12)


def test_radius_adaptive_prefix_ignores_a_distant_downstream_turn():
    path = [[0, 0, 0], [4, 0, 0], [4, 15, 0]]
    np.testing.assert_allclose(proximal_pca_direction(path, 1), [1, 0, 0], atol=1e-12)
    assert proximal_pca_direction(path, 4)[1] > 0.5


@pytest.mark.parametrize("path,radius", [
    ([[0, 0, 0], [0, 0, 0]], 1),
    ([[0, 0, 0], [1, np.nan, 0]], 1),
    ([[0, 0], [1, 0]], 1),
    ([[0, 0, 0], [1, 0, 0]], 0),
])
def test_invalid_direction_inputs_are_rejected(path, radius):
    with pytest.raises(ValueError):
        proximal_pca_direction(path, radius)


@pytest.mark.parametrize("method", METHODS)
def test_paper_runner_preserves_empty_mask_and_physical_contract(method):
    case = generate_case(1)
    empty = sitk.Image(case.parent.GetSize(), sitk.sitkUInt8)
    empty.CopyInformation(case.parent)
    result = detect_paper(case.image, empty, method)
    assert result.branches == []
    validate_prediction(result.prediction("empty"))


def test_pca_changes_only_direction_and_its_dependent_feature_without_mutating_inputs():
    case = generate_case(1)
    image_hash = hashlib.sha256(sitk.GetArrayFromImage(case.image).tobytes()).hexdigest()
    before = detect(case.image, case.parent)
    after = detect_paper(case.image, case.parent, "pca-direction")
    assert len(before.branches) == len(after.branches) == 2
    for baseline, candidate in zip(before.branches, after.branches):
        old, new = baseline.prediction(), candidate.prediction()
        old.pop("direction_xyz")
        new.pop("direction_xyz")
        assert old == new
    validate_prediction(after.prediction("pca"))
    assert hashlib.sha256(sitk.GetArrayFromImage(case.image).tobytes()).hexdigest() == image_hash
    assert DetectorConfig().profile == "strict"


@pytest.mark.parametrize("method", ["contact-growth", "border-cleaning"])
def test_classical_variants_run_on_an_analytic_two_daughter_case(method):
    case = generate_case(1)
    result = detect_paper(case.image, case.parent, method)
    validate_prediction(result.prediction("analytic"))
    assert result.branches
    assert all(len(branch.path_xyz_mm) >= 2 for branch in result.branches)


@pytest.mark.parametrize("method", METHODS)
def test_paper_variants_do_not_invent_branches_in_negative_control(method):
    case = generate_case(0)
    assert detect_paper(case.image, case.parent, method).branches == []


def test_paper_benchmark_requires_analytic_truth_and_preserves_existing_report(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"source": "candidate_pseudo", "cases": []}))
    output = tmp_path / "report.json"
    with pytest.raises(ValueError, match="analytic"):
        benchmark_papers(tmp_path, output)
    output.write_text("original")
    with pytest.raises(FileExistsError):
        benchmark_papers(tmp_path, output)
    assert output.read_text() == "original"
