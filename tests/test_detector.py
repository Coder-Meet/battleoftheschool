import gzip
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import SimpleITK as sitk
from scipy import ndimage as ndi

import detector
from detector import (
    DetectorConfig, branch_junctions, detect, physical_points, stop_at_junction, truncate_path, validate_geometry,
)
from nifti_io import read_nifti


def phantom(branches=(20, 39), spacing=(1.0, 1.0, 1.0)):
    z, y, x = np.indices((64, 52, 64)) * np.asarray(spacing)[::-1, None, None, None]
    parent = ((x - 25) ** 2 + (y - 25) ** 2 <= 8**2) & (z >= 5) & (z <= 56)
    vessels = parent.copy()
    for height in branches:
        vessels |= ((y - 25) ** 2 + (z - height) ** 2 <= 2.5**2) & (x >= 25) & (x <= 49)
    ct = np.full(parent.shape, 20, dtype=np.float32)
    ct[vessels] = 330
    image = sitk.GetImageFromArray(ct)
    mask = sitk.GetImageFromArray(parent.astype(np.uint8))
    image.SetSpacing(spacing)
    mask.CopyInformation(image)
    return image, mask


def test_two_daughters_have_independent_origins_and_physical_measurements():
    image, mask = phantom()
    result = detect(image, mask)
    assert len(result.branches) == 2
    assert {b.instance_id for b in result.branches} == {"branch_001", "branch_002"}
    for b, height in zip(result.branches, (20, 39)):
        assert np.linalg.norm(np.asarray(b.ostium_xyz_mm) - (33, 25, height)) < 2
        assert np.linalg.norm(b.direction_xyz) == pytest.approx(1)
        assert b.direction_xyz[0] > 0.9
        assert 1.5 < b.radius_mm < 3.5
        seed = truncate_path(np.asarray(b.path_xyz_mm), 5)[-1]
        assert np.allclose(seed, b.seed_xyz_mm)
        assert b.parent_instance_id == "aorta"


def test_empty_parent_produces_empty_daughters():
    image, mask = phantom()
    result = detect(image, mask * 0)
    assert result.prediction("empty")["daughters"] == []


def test_no_branches_and_bright_crop_caps_are_not_origins():
    image, mask = phantom(branches=())
    data = sitk.GetArrayFromImage(image)
    z, y, x = np.indices(data.shape)
    data[(x - 25) ** 2 + (y - 25) ** 2 <= 8**2] = 330
    continuous = sitk.GetImageFromArray(data)
    continuous.CopyInformation(image)
    assert detect(continuous, mask).branches == []


def test_rotated_translated_geometry_preserves_detection_in_physical_space():
    image, mask = phantom()
    baseline = detect(image, mask)
    image.SetOrigin((121.5, -48.2, 505))
    image.SetDirection((0, -1, 0, 1, 0, 0, 0, 0, 1))
    mask.CopyInformation(image)
    moved = detect(image, mask)
    matrix = np.asarray(image.GetDirection()).reshape(3, 3)
    assert len(moved.branches) == len(baseline.branches)
    for before, after in zip(baseline.branches, moved.branches):
        assert np.allclose(matrix @ before.ostium_xyz_mm + image.GetOrigin(), after.ostium_xyz_mm)
        assert np.allclose(matrix @ before.direction_xyz, after.direction_xyz)


def test_geometry_rejects_misaligned_inputs():
    image, mask = phantom()
    mask.SetOrigin((1, 0, 0))
    with pytest.raises(ValueError, match="physical geometry"):
        validate_geometry(image, mask)


def test_anisotropic_physical_transform_matches_simpleitk():
    image, _ = phantom(spacing=(0.8, 1.1, 1.4))
    image.SetOrigin((-34, 17, 209))
    point = (10, 12, 17)
    assert np.allclose(physical_points(image, [point])[0], image.TransformIndexToPhysicalPoint(point[::-1]))


def test_seed_interpolates_along_curved_arc_not_straight_line():
    path = np.array([[0, 0, 0], [3, 0, 0], [3, 4, 0]], dtype=float)
    clipped = truncate_path(path, 5)
    assert np.allclose(clipped[-1], (3, 2, 0))
    assert np.linalg.norm(np.diff(clipped, axis=0), axis=1).sum() == pytest.approx(5)


def test_gzip_magic_and_lfs_pointer(tmp_path):
    image, _ = phantom()
    archive = tmp_path / "scan.nii.gz"
    sitk.WriteImage(image, str(archive))
    disguised = tmp_path / "scan.nii"
    disguised.write_bytes(archive.read_bytes())
    assert read_nifti(str(disguised)).GetSize() == image.GetSize()
    assert gzip.decompress(disguised.read_bytes())
    pointer = tmp_path / "pointer.nii"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(ValueError, match="Git LFS pointer"):
        read_nifti(str(pointer))


def test_cli_creates_output_parent_and_correct_double_extension_id(tmp_path):
    image, mask = phantom(branches=())
    sitk.WriteImage(image, str(tmp_path / "example.nii.gz"))
    sitk.WriteImage(mask, str(tmp_path / "mask.nii.gz"))
    output = tmp_path / "nested" / "prediction.json"
    command = [
        sys.executable, str(Path(__file__).resolve().parents[1] / "run.py"),
        "--image", str(tmp_path / "example.nii.gz"), "--aorta-mask", str(tmp_path / "mask.nii.gz"),
        "--output", str(output),
    ]
    subprocess.run(command, check=True, capture_output=True)
    assert json.loads(output.read_text()) == {
        "case_id": "example", "parent": {"instance_id": "aorta"}, "daughters": [],
    }


def test_settings_reject_nonfinite_values():
    with pytest.raises(ValueError):
        DetectorConfig(minimum_radius_mm=float("nan"))
    with pytest.raises(ValueError):
        DetectorConfig(minimum_origin_diameter_mm=-1)


def test_origin_diameter_is_separate_from_seed_radius_eligibility(monkeypatch):
    image, mask = phantom(branches=(20,))
    disabled = DetectorConfig(minimum_radius_mm=0.3, minimum_origin_diameter_mm=0)
    enforced = DetectorConfig(minimum_radius_mm=0.3, minimum_origin_diameter_mm=2)
    monkeypatch.setattr(detector, "cross_section_radius", lambda *args: 0.4)
    assert len(detect(image, mask, disabled).branches) == 1
    result = detect(image, mask, enforced)
    assert result.branches == []
    assert result.rejections["origin_below_minimum_diameter"] == 1

    monkeypatch.setattr(detector, "cross_section_radius", lambda *args: 1.1)
    branch = detect(image, mask, enforced).branches[0]
    assert branch.features["origin_diameter_mm"] >= 2
    assert branch.features["origin_diameter_upper_mm"] >= branch.features["origin_diameter_mm"]


def test_anisotropic_detection_keeps_five_mm_physical_seed():
    image, mask = phantom(spacing=(0.8, 1.1, 1.4))
    result = detect(image, mask)
    assert len(result.branches) == 2
    for branch, height in zip(result.branches, (20, 39)):
        assert np.linalg.norm(np.asarray(branch.ostium_xyz_mm) - (33, 25, height)) < 2.5
        assert np.allclose(truncate_path(np.asarray(branch.path_xyz_mm), 5)[-1], branch.seed_xyz_mm)


def test_nearby_independent_openings_remain_separate():
    image, mask = phantom(branches=(20, 28))
    result = detect(image, mask)
    assert len(result.branches) == 2
    assert abs(result.branches[1].ostium_xyz_mm[2] - result.branches[0].ostium_xyz_mm[2]) > 5


@pytest.mark.parametrize("heights", [(20, 28), (20, 39)])
def test_one_returning_vessel_has_two_distinct_aortic_openings(heights):
    image, mask = phantom(branches=heights)
    data = sitk.GetArrayFromImage(image)
    z, y, x = np.indices(data.shape)
    data[((x - 47) ** 2 + (y - 25) ** 2 <= 2.5**2) & (z >= heights[0]) & (z <= heights[1])] = 330
    external = (data > 250) & ~sitk.GetArrayFromImage(mask).astype(bool)
    assert ndi.label(external, structure=np.ones((3, 3, 3)))[1] == 1
    image = sitk.GetImageFromArray(data)
    image.CopyInformation(mask)
    result = detect(image, mask)
    assert len(result.branches) == 2
    for branch, height in zip(result.branches, heights):
        assert np.linalg.norm(np.asarray(branch.ostium_xyz_mm) - (33, 25, height)) < 2
        assert branch.direction_xyz[0] > 0.9


def test_overlapping_openings_keep_a_valid_single_or_double_representation():
    image, mask = phantom(branches=(24, 28))
    result = detect(image, mask)
    assert 1 <= len(result.branches) <= 2
    for branch in result.branches:
        assert 22 <= branch.ostium_xyz_mm[2] <= 30
        assert branch.direction_xyz[0] > 0.9


def test_second_generation_vessel_is_not_a_second_aortic_origin():
    image, mask = phantom(branches=(20,))
    data = sitk.GetArrayFromImage(image)
    z, y, x = np.indices(data.shape)
    data[((x - 44) ** 2 + (z - 20) ** 2 <= 2**2) & (y >= 25) & (y <= 43)] = 330
    image = sitk.GetImageFromArray(data)
    image.CopyInformation(mask)
    assert len(detect(image, mask).branches) == 1


def test_a_noncontacting_enhanced_vessel_is_rejected():
    image, mask = phantom(branches=())
    data = sitk.GetArrayFromImage(image)
    z, y, x = np.indices(data.shape)
    data[((x - 40) ** 2 + (y - 25) ** 2 <= 2**2) & (z >= 12) & (z <= 48)] = 330
    image = sitk.GetImageFromArray(data)
    image.CopyInformation(mask)
    assert detect(image, mask).branches == []


def test_shared_prefix_marks_a_common_trunk_but_not_a_neighbouring_opening():
    from detector import shares_prefix

    trunk = np.array([[0, 0, 0], [4, 0, 0], [8, 3, 0], [10, 6, 0]], dtype=float)
    other_child = np.array([[0.5, 0.3, 0], [4, 0.4, 0], [8, -3, 0]], dtype=float)
    neighbour = np.array([[0, 2.5, 0], [4, 4, 0], [8, 6, 0]], dtype=float)
    assert shares_prefix(other_child, trunk)
    assert not shares_prefix(neighbour, trunk)


def test_proximal_path_stops_at_first_associated_bifurcation():
    path = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0]], dtype=float)
    junctions = np.array([[9, 0, 0], [7, 0, 0], [2, 4, 0]], dtype=float)
    result = stop_at_junction(path, junctions, 1.5)
    assert np.allclose(result[-1], [7, 0, 0])
    assert np.allclose(truncate_path(result, 5)[-1], [5, 0, 0])


def test_skeleton_junction_requires_three_substantial_arms():
    support = np.zeros((31, 31, 31), dtype=bool)
    support[15, 15, 5:26] = True
    support[15, 15:26, 15] = True
    parent = np.zeros_like(support)
    junctions = branch_junctions(support, parent, 1)
    assert len(junctions) == 1
    assert np.linalg.norm(junctions[0] - [15, 15, 15]) < 1.5
    support[15, 16:26, 15] = False
    assert len(branch_junctions(support, parent, 1)) == 0


@pytest.mark.parametrize("spacing,radius", [(1.0, 1.8), (1.5, 1.4)])
def test_partial_volume_daughters_survive_both_contrast_levels(spacing, radius):
    size = tuple(np.ceil(np.array([72, 64, 72]) / spacing).astype(int))
    z, y, x = np.indices(size) * spacing
    parent = ((x - 26)**2 + (y - 30)**2 <= 8**2) & (z >= 5) & (z <= 65)
    subvoxels = 3
    fine_size = tuple(value * subvoxels for value in size)
    fz, fy, fx = (np.indices(fine_size) - (subvoxels - 1) / 2) * (spacing / subvoxels)
    lumen = ((fx - 26)**2 + (fy - 30)**2 <= 8**2) & (fz >= 5) & (fz <= 65)
    for height in (24, 48):
        lumen |= ((fy - 30)**2 + (fz - height)**2 <= radius**2) & (fx >= 26) & (fx <= 52)
    occupancy = ndi.gaussian_filter(lumen.astype(float), 0.6 * subvoxels / spacing)
    occupancy = occupancy.reshape(
        size[0], subvoxels, size[1], subvoxels, size[2], subvoxels,
    ).mean(axis=(1, 3, 5))
    for hu in (150, 550):
        image = sitk.GetImageFromArray((20 + (hu - 20) * occupancy).astype(np.float32))
        image.SetSpacing((spacing,) * 3)
        mask = sitk.GetImageFromArray(parent.astype(np.uint8))
        mask.CopyInformation(image)
        result = detect(image, mask, DetectorConfig(native_contrast_scale=1.2))
        assert len(result.branches) == 2
        for branch, height in zip(result.branches, (24, 48)):
            assert np.linalg.norm(np.asarray(branch.ostium_xyz_mm) - (34, 30, height)) < 3
            assert np.allclose(truncate_path(np.asarray(branch.path_xyz_mm), 5)[-1], branch.seed_xyz_mm)
        if spacing == 1.5:
            assert result.blood_model["support_fraction"] < 0.5


def test_background_overlap_is_reported_without_claiming_a_noise_measurement():
    image, mask = phantom(branches=())
    data = sitk.GetArrayFromImage(image)
    z, y, x = np.indices(data.shape)
    data[sitk.GetArrayViewFromImage(mask) == 0] = (40 + 7 * x)[sitk.GetArrayViewFromImage(mask) == 0]
    image = sitk.GetImageFromArray(data)
    image.CopyInformation(mask)
    result = detect(image, mask, DetectorConfig(native_contrast_scale=1.2))
    assert result.blood_model["contrast_to_background_mad"] <= 1
    assert any("Parent/background intensities overlap" in warning for warning in result.warnings)


def test_same_relative_daughter_contrast_keeps_both_origins():
    template, mask = phantom()
    parent = sitk.GetArrayFromImage(mask) > 0
    daughters = (sitk.GetArrayFromImage(template) > 20) & ~parent
    for hu in (150, 550):
        intensity = np.full(parent.shape, 20, dtype=np.float32)
        intensity[parent] = hu
        intensity[daughters] = 20 + 0.65 * (hu - 20)
        image = sitk.GetImageFromArray(intensity)
        image.CopyInformation(mask)
        result = detect(image, mask)
        assert len(result.branches) == 2
        for branch, height in zip(result.branches, (20, 39)):
            assert np.linalg.norm(np.asarray(branch.ostium_xyz_mm) - (33, 25, height)) < 3
