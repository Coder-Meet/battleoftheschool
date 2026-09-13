from dataclasses import replace
import json

import numpy as np
import pytest
import SimpleITK as sitk

from candidate_patches import (
    CHANNEL_NAMES, EXTRA_FEATURE_NAMES, extract_candidate_data, preprocessing_metadata,
)
from detector import Branch, Detection, DetectorConfig, detect
from test_detector import phantom


def proposal(start=(0.0, 0.0, 0.0), end=(6.0, 0.0, 0.0)):
    direction = np.subtract(end, start)
    direction /= np.linalg.norm(direction)
    branch = Branch(
        "candidate", start, end, 1.5, tuple(direction), [start, end], 0.5, 0.2,
    )
    return Detection([branch], {}, {}, 1, {}, [], DetectorConfig())


def scene(spacing=(1.0, 1.0, 1.0), direction=np.eye(3), center=(0.0, 0.0, 0.0), shape=(49, 49, 49)):
    indices = np.moveaxis(np.indices(shape, dtype=float), 0, -1)[..., ::-1]
    origin = np.asarray(center) - direction @ ((np.asarray(shape[::-1]) // 2) * spacing)
    xyz = (indices * spacing) @ direction.T + origin
    relative = xyz - center
    parent = relative[..., 0] < -0.1
    ct = 100 + relative[..., 0] + 0.6 * relative[..., 1] + 0.3 * relative[..., 2]
    ct[parent] = 330
    image, mask = sitk.GetImageFromArray(ct), sitk.GetImageFromArray(parent.astype(np.uint8))
    image.SetSpacing(spacing)
    image.SetOrigin(origin.tolist())
    image.SetDirection(direction.ravel().tolist())
    mask.CopyInformation(image)
    return image, mask


def test_native_anisotropic_oblique_grid_samples_physical_axes_and_nearest_mask():
    angle = np.deg2rad(27)
    direction = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    origin = (121.0, -43.0, 550.0)
    image, mask = scene((0.7, 1.1, 1.8), direction, origin)
    result = proposal(origin, tuple(np.asarray(origin) + [6, 0, 0]))
    data = extract_candidate_data(image, mask, result, size=33, spacing_mm=0.8)
    # All these points are beyond the interpolated parent wall, where CT is an affine physical field.
    for view, physical_slopes in enumerate(((1, 0.6), (1, 0.3), (0.6, 0.3))):
        ct = data.patches[0, view * 4]
        if view < 2:
            horizontal = ct[23, 31] - ct[23, 29]
            vertical = ct[25, 29] - ct[23, 29]
            assert horizontal / vertical == pytest.approx(physical_slopes[0] / physical_slopes[1], rel=1e-4)
    view_axes = ((0, 1), (0, 2), (1, 2))
    native = sitk.GetArrayViewFromImage(mask)
    for view, (horizontal, vertical) in enumerate(view_axes):
        expected = np.zeros((33, 33))
        for row, column in np.ndindex(expected.shape):
            point = np.asarray(origin, dtype=float)
            point[horizontal] += (column - 16) * 0.8
            point[vertical] += (row - 16) * 0.8
            index = image.TransformPhysicalPointToIndex(point.tolist())
            if all(0 <= value < extent for value, extent in zip(index, image.GetSize())):
                expected[row, column] = native[index[::-1]]
        assert np.array_equal(data.patches[0, view * 4 + 1], expected)
    assert data.extra_features[0, 2] < 0.05


def test_reoriented_native_storage_preserves_all_channels_in_world_space():
    image, mask = scene(shape=(33, 33, 33), center=(70, -20, 600))
    result = proposal((70.0, -20.0, 600.0), (76.0, -20.0, 600.0))
    expected = extract_candidate_data(image, mask, result, size=16)
    reoriented_image = sitk.Flip(sitk.PermuteAxes(image, [2, 0, 1]), [True, False, True])
    reoriented_mask = sitk.Flip(sitk.PermuteAxes(mask, [2, 0, 1]), [True, False, True])
    actual = extract_candidate_data(reoriented_image, reoriented_mask, result, size=16)
    np.testing.assert_allclose(actual.patches, expected.patches, atol=1e-6)
    np.testing.assert_allclose(actual.extra_features, expected.extra_features, atol=1e-6)


@pytest.mark.parametrize("size", [15, 16])
def test_path_polyline_uses_mm_and_crosses_aligned_views(size):
    image, mask = scene(spacing=(0.7, 1.1, 1.8))
    result = proposal(end=(6.0, 4.0, 0.0))
    result.branches[0].path_xyz_mm = [(0.0, 0.0, 0.0), (6.0, 0.0, 0.0), (6.0, 4.0, 0.0)]
    data = extract_candidate_data(image, mask, result, size=size)
    mid = size // 2
    xy, xz, yz = data.patches[0, [3, 7, 11]]
    assert xy[mid, mid] == xz[mid, mid] == yz[mid, mid] == 1
    assert np.all(xy[mid, mid:mid + 7] == 1)
    assert np.all(xy[mid:mid + 5, mid + 6] == 1)
    assert xy[mid + 2, mid + 3] < 0.05
    assert yz[mid, mid + 3] == 0
    assert xz[mid + 1, mid + 3] == pytest.approx(np.exp(-1 / (2 * 0.75**2)))


def test_boundary_padding_is_finite_zero_and_does_not_create_a_parent_cap():
    image, mask = scene(center=(0.0, 0.0, 24.0))
    result = proposal()
    data = extract_candidate_data(image, mask, result, size=16)
    assert np.isfinite(data.patches).all() and np.isfinite(data.extra_features).all()
    assert np.all(data.patches[0, 4:12, :8] == 0)
    assert data.patches[0, 7, 8, 8] == 1
    assert set(np.unique(data.patches[0, [1, 5, 9]])) == {0, 1}
    assert data.extra_features[0, 2] == pytest.approx(0, abs=1e-8)


def test_scan_relative_window_resists_a_remote_outlier_and_affine_hu_change():
    image, mask = scene()
    expected = extract_candidate_data(image, mask, proposal(), size=16)
    changed = image * 1.5 + 17
    actual = extract_candidate_data(changed, mask, proposal(), size=16)
    np.testing.assert_allclose(actual.patches, expected.patches, atol=2e-6)
    np.testing.assert_allclose(actual.extra_features, expected.extra_features, atol=2e-6)
    changed[0, 0, 0] = 1e12
    outlier = extract_candidate_data(changed, mask, proposal(), size=16)
    np.testing.assert_allclose(outlier.patches, expected.patches, atol=2e-6)


def test_production_candidates_extract_without_mutation_or_reference_proxies():
    image, mask = phantom()
    result = detect(image, mask)
    before = result.diagnostics()
    data = extract_candidate_data(image, mask, result)
    assert result.diagnostics() == before
    assert data.patches.shape == (2, 12, 32, 32)
    assert data.patches.dtype == np.float32 and data.extra_features.dtype == np.float64
    assert np.all(data.extra_features[:, 0] > 0.5)
    assert np.all(data.extra_features[:, 1] > 0)
    assert np.allclose(data.extra_features[:, 2], 1 / 16, rtol=0.35)
    result.branches.reverse()
    for branch in result.branches:
        branch.features = {"expert_label": 1e9}
        branch.evidence_score = -999
        branch.mean_vesselness = -999
        branch.warnings = ["pseudo"]
    result.blood_model = {"lower_hu": -999, "upper_hu": -998}
    actual = extract_candidate_data(image, mask, result)
    np.testing.assert_array_equal(actual.patches, data.patches[::-1])
    np.testing.assert_array_equal(actual.extra_features, data.extra_features[::-1])


@pytest.mark.parametrize("radius", [8.0, 12.0])
def test_wall_curvature_has_inverse_mm_scale_for_an_analytic_sphere(radius):
    image, _ = scene(spacing=(0.75,) * 3, shape=(65,) * 3)
    z, y, x = np.indices(image.GetSize()[::-1]) * 0.75 - 24
    parent = (x + radius)**2 + y**2 + z**2 <= radius**2
    mask = sitk.GetImageFromArray(parent.astype(np.uint8))
    mask.CopyInformation(image)
    ct = sitk.GetImageFromArray(np.where(parent, 330, 20).astype(np.float32))
    ct.CopyInformation(image)
    data = extract_candidate_data(ct, mask, proposal(), size=16, spacing_mm=0.75)
    assert data.extra_features[0, 2] == pytest.approx(1 / radius, rel=0.2)


def test_numeric_artifact_round_trip_and_documented_channel_contract(tmp_path):
    image, mask = scene()
    data = extract_candidate_data(image, mask, proposal(), size=16)
    destination = tmp_path / "patches.npz"
    np.savez_compressed(destination, patches=data.patches, extra_features=data.extra_features)
    with np.load(destination, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["patches"], data.patches)
        np.testing.assert_array_equal(archive["extra_features"], data.extra_features)
    metadata = json.loads(json.dumps(preprocessing_metadata(16)))
    assert metadata["channel_names"] == CHANNEL_NAMES
    assert metadata["extra_feature_names"] == EXTRA_FEATURE_NAMES
    assert metadata["center_pixel"] == 8
    assert metadata["sato_sigmas_mm"] == [0.8, 1.5, 2.5]


def test_empty_results_have_correct_shapes_even_without_parent():
    image, mask = scene()
    data = extract_candidate_data(image, mask * 0, replace(proposal(), branches=[]), size=11)
    assert data.patches.shape == (0, 12, 11, 11) and data.patches.dtype == np.float32
    assert data.extra_features.shape == (0, 3) and data.extra_features.dtype == np.float64


@pytest.mark.parametrize("label", [0.25, 255.0])
def test_binary_foreground_encoding_does_not_change_patches(label):
    image, mask = scene(shape=(33, 33, 33))
    expected = extract_candidate_data(image, mask, proposal(), size=16)
    actual = extract_candidate_data(image, sitk.Cast(mask, sitk.sitkFloat64) * label, proposal(), size=16)
    np.testing.assert_array_equal(actual.patches, expected.patches)
    np.testing.assert_array_equal(actual.extra_features, expected.extra_features)


@pytest.mark.parametrize("size,spacing", [(0, 1), (2.5, 1), (True, 1), (32, 0), (32, np.nan), (32, np.inf), (32, 1e-8), (10000, 1)])
def test_invalid_or_unbounded_patch_grids_fail_before_allocation(size, spacing):
    image, mask = scene(shape=(17, 17, 17))
    with pytest.raises(ValueError):
        extract_candidate_data(image, mask, proposal(), size=size, spacing_mm=spacing)


@pytest.mark.parametrize("target", ["image", "mask"])
def test_nonfinite_voxels_are_rejected_even_for_empty_detections(target):
    image, mask = scene()
    mask = sitk.Cast(mask, sitk.sitkFloat32)
    if target == "image":
        image[0, 0, 0] = np.nan
    else:
        mask[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="nonfinite"):
        extract_candidate_data(image, mask, replace(proposal(), branches=[]))


@pytest.mark.parametrize("kind", ["origin", "direction", "size", "multilabel", "negative", "empty_parent"])
def test_invalid_input_geometry_and_parent_masks_are_rejected(kind):
    image, mask = scene()
    if kind == "origin":
        mask.SetOrigin((10, 20, 30))
    elif kind == "direction":
        image.SetDirection((1, 0.2, 0, 0, 1, 0, 0, 0, 1))
        mask.CopyInformation(image)
    elif kind == "size":
        mask = mask[:, :, :-1]
    elif kind == "multilabel":
        mask[10, 10, 10] = 2
    elif kind == "negative":
        mask = sitk.Cast(mask, sitk.sitkInt16)
        mask[10, 10, 10] = -1
    else:
        mask *= 0
    with pytest.raises(ValueError):
        extract_candidate_data(image, mask, proposal())


@pytest.mark.parametrize("path", [
    [], [(0, 0, 0)], [(0, 0, 0), (0, 0, 0)], [(0, 0), (1, 1)],
    [(0, 0, 0), (np.nan, 0, 0)], [(1, 0, 0), (5, 0, 0)], [(0, 0, 0), (1e12, 0, 0)],
])
def test_malformed_or_mismatched_branch_paths_raise(path):
    image, mask = scene()
    result = proposal()
    result.branches[0].path_xyz_mm = path
    with pytest.raises(ValueError):
        extract_candidate_data(image, mask, result)


def test_missing_feature_support_raises_instead_of_filling_zeros():
    image, mask = scene()
    result = proposal((-15.0, 0.0, 0.0), (-10.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="non-parent tissue"):
        extract_candidate_data(image, mask, result)
