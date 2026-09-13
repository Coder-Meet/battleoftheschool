import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import SimpleITK as sitk

from benchmark import benchmark
from evaluate import validate_prediction
from stress import generate_case, write_case


@pytest.mark.parametrize("family", [
    "curved_daughters", "cropped_short_segment", "common_trunk_early_split", "wall_parallel_descending",
])
def test_independent_references_remain_on_the_analytic_lumen(family):
    case = generate_case(family, 4001)
    validate_prediction(case.reference)
    assert case.image.GetDirection() == case.parent.GetDirection()
    for branch, geometry in zip(case.reference["daughters"], case.provenance["geometry"]):
        assert branch["instance_id"] == geometry["instance_id"]
        line = np.asarray(geometry["proximal_centerline_xyz_mm"])
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
        seed = np.array([np.interp(5.0, arc, line[:, i]) for i in range(3)])
        assert np.allclose(seed, branch["seed_xyz_mm"])
        index = case.image.TransformPhysicalPointToIndex(seed.tolist())
        assert case.parent[index] == 0
        assert case.image[index] > 100
    if family == "cropped_short_segment":
        occupied = np.argwhere(sitk.GetArrayFromImage(case.parent) > 0)
        coverage = np.ptp(occupied[:, 0]) * case.parent.GetSpacing()[2]
        assert coverage <= 29
        assert len(case.reference["daughters"]) == 1


def test_wall_parallel_daughter_hugs_the_parent_and_is_only_found_with_the_fallback():
    from detector import DetectorConfig, detect

    case = generate_case("wall_parallel_descending", 4001)
    hugging, ordinary = case.reference["daughters"]
    line = np.asarray(case.provenance["geometry"][0]["proximal_centerline_xyz_mm"])
    distance = sitk.SignedMaurerDistanceMap(case.parent, insideIsPositive=False, squaredDistance=False, useImageSpacing=True)
    clearance = [distance.EvaluateAtPhysicalPoint(point.tolist()) for point in line]
    assert max(clearance) < 5, "the hugging daughter must stay within 5 mm of the parent over its proximal path"

    def ostia(config):
        return [np.asarray(branch.ostium_xyz_mm) for branch in detect(case.image, case.parent, config).branches]

    baseline = ostia(DetectorConfig(native_contrast_scale=1.2))
    assert any(np.linalg.norm(o - ordinary["ostium_xyz_mm"]) < 3 for o in baseline)
    assert not any(np.linalg.norm(o - hugging["ostium_xyz_mm"]) < 3 for o in baseline)
    fallback = ostia(DetectorConfig(native_contrast_scale=1.2, parallel_clearance_mm=2.5))
    assert len(fallback) == 2
    assert any(np.linalg.norm(o - hugging["ostium_xyz_mm"]) < 2 for o in fallback)


def test_generator_does_not_depend_on_python_hash_randomization():
    command = (
        "import hashlib, json, SimpleITK as sitk; from stress import generate_case; "
        "c=generate_case('negative_controls_only',4001); "
        "print(hashlib.sha256(sitk.GetArrayFromImage(c.image).tobytes()).hexdigest()); "
        "print(json.dumps(c.reference,sort_keys=True))"
    )
    results = [
        subprocess.run(
            [sys.executable, "-c", command], env={**os.environ, "PYTHONHASHSEED": seed},
            check=True, capture_output=True, text=True,
        ).stdout for seed in ("1", "987")
    ]
    assert results[0] == results[1]


def test_input_integrity_and_report_overwrite_are_protected(tmp_path):
    case = generate_case("negative_controls_only", 4001)
    entry = write_case(case, tmp_path / case.reference["case_id"])
    for filename, digest in entry["files"].items():
        assert hashlib.sha256((tmp_path / entry["case_id"] / filename).read_bytes()).hexdigest() == digest
    (tmp_path / "manifest.json").write_text(json.dumps({"cases": [entry]}))
    (tmp_path / entry["case_id"] / "reference.json").write_text("{}")
    report = tmp_path / "metrics.json"
    with pytest.raises(ValueError, match="integrity"):
        benchmark(tmp_path, report, 3)
    report.write_text("protected")
    with pytest.raises(FileExistsError):
        benchmark(tmp_path, report, 3)
    assert report.read_text() == "protected"


def test_unknown_family_and_negative_seed_fail():
    with pytest.raises(ValueError, match="Unknown"):
        generate_case("not-a-family", 4001)
    with pytest.raises(ValueError, match="nonnegative"):
        generate_case("curved_daughters", -1)
