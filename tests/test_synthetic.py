import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import SimpleITK as sitk

from evaluate import validate_prediction
from synthetic import generate_case, write_case


@pytest.mark.parametrize("scenario", range(1, 6))
def test_analytic_labels_and_geometry_are_independent_of_detector(scenario):
    case = generate_case(scenario)
    validate_prediction(case.reference)
    parent = sitk.GetArrayFromImage(case.parent)
    labels = sitk.GetArrayFromImage(case.instances)
    assert np.array_equal(labels == 1, parent > 0)
    assert set(np.unique(labels)) == {0, 1, 2, 3}
    assert case.image.GetSpacing() == case.instances.GetSpacing() == case.parent.GetSpacing()
    for index, branch in enumerate(case.reference["daughters"], 2):
        ostium = np.asarray(branch["ostium_xyz_mm"])
        seed = np.asarray(branch["seed_xyz_mm"])
        assert np.linalg.norm(seed - ostium) == pytest.approx(5)
        assert np.allclose((seed - ostium) / 5, branch["direction_xyz"])
        seed_index = case.image.TransformPhysicalPointToIndex(seed.tolist())
        assert case.instances[seed_index] == index
        assert case.parent[seed_index] == 0
        local = np.asarray(case.image.TransformPhysicalPointToContinuousIndex(ostium.tolist()))
        local *= case.image.GetSpacing()
        assert np.linalg.norm(local[:2] - [32, 32]) == pytest.approx(8)
    assert case.provenance["source"] == "analytic_synthetic_geometry"
    assert "training" in case.provenance["intended_use"]


def test_seed_reproducibility_and_common_trunk_provenance():
    first, again, different = (generate_case(4, seed) for seed in (19, 19, 20))
    assert np.array_equal(sitk.GetArrayFromImage(first.image), sitk.GetArrayFromImage(again.image))
    assert first.reference == again.reference
    assert first.reference != different.reference
    assert len(first.reference["daughters"]) == 2
    assert len(first.provenance["geometry"][0]["downstream_centerlines_xyz_mm"]) == 2


def test_dataset_write_roundtrip_hashes_and_overwrite_protection(tmp_path):
    case = generate_case(3)
    directory = tmp_path / case.reference["case_id"]
    entry = write_case(case, directory)
    assert entry["partition"] == "train"
    assert len(entry["files"]) == 5
    for name, digest in entry["files"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    restored = sitk.ReadImage(str(directory / "instances.nii.gz"))
    assert np.array_equal(sitk.GetArrayFromImage(case.instances), sitk.GetArrayFromImage(restored))
    assert json.loads((directory / "reference.json").read_text()) == case.reference
    with pytest.raises(FileExistsError):
        write_case(case, directory)


def test_cli_protects_existing_ground_truth(tmp_path):
    sentinel = tmp_path / "do-not-overwrite.json"
    sentinel.write_text('{"human_annotation": true}')
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "synthetic.py"),
         "--output-dir", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "protect labels" in result.stderr
    assert sentinel.read_text() == '{"human_annotation": true}'
