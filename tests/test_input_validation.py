import numpy as np
import pytest
import SimpleITK as sitk

from detector import detect, validate_geometry
from test_detector import phantom


@pytest.mark.parametrize("contrast,noise", [(330, 25), (180, 20), (120, 20), (90, 20)])
def test_direct_origins_survive_noise_and_contrast_variation(contrast, noise):
    image, mask = phantom()
    source = sitk.GetArrayFromImage(image)
    ct = np.where(source > 100, contrast, 20).astype(np.float32)
    ct += np.random.default_rng(8).normal(0, noise, ct.shape).astype(np.float32)
    image = sitk.GetImageFromArray(ct)
    image.CopyInformation(mask)
    result = detect(image, mask)
    assert len(result.branches) == 2
    for branch, height in zip(result.branches, (20, 39)):
        assert np.linalg.norm(np.asarray(branch.ostium_xyz_mm) - [33, 25, height]) < 2


def test_multilabel_or_sheared_inputs_fail_before_detection():
    image, mask = phantom()
    labels = sitk.GetArrayFromImage(mask)
    labels[20, 25, 25] = 2
    invalid = sitk.GetImageFromArray(labels)
    invalid.CopyInformation(mask)
    with pytest.raises(ValueError, match="binary parent-aorta"):
        validate_geometry(image, invalid)
    image.SetDirection((1, 0.2, 0, 0, 1, 0, 0, 0, 1))
    mask.CopyInformation(image)
    with pytest.raises(ValueError, match="orthonormal"):
        validate_geometry(image, mask)


@pytest.mark.parametrize("blood,daughter", [(330, 210), (450, 285), (180, 135)])
def test_less_enhanced_daughters_remain_connected_without_admitting_separate_vessels(blood, daughter):
    image, mask = phantom()
    source = sitk.GetArrayFromImage(image)
    parent = sitk.GetArrayFromImage(mask) > 0
    ct = np.where(source > 100, daughter, 20).astype(np.float32)
    ct[parent] = blood
    z, y, x = np.indices(ct.shape)
    ct[((x - 13)**2 + (y - 25)**2 <= 1.5**2) & (z > 12) & (z < 48)] = daughter
    ct += np.random.default_rng(931).normal(0, 8, ct.shape).astype(np.float32)
    image = sitk.GetImageFromArray(ct)
    image.CopyInformation(mask)
    result = detect(image, mask)
    assert len(result.branches) == 2
    for branch, height in zip(result.branches, (20, 39)):
        assert np.linalg.norm(np.asarray(branch.ostium_xyz_mm) - [33, 25, height]) < 2
        assert np.linalg.norm(np.asarray(branch.seed_xyz_mm) - [38, 25, height]) < 2
