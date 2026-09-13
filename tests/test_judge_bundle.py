import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile

import pytest
import SimpleITK as sitk

from application.build import (
    BUNDLE_NAME, ROOT, SOURCE_HASHES, build, runtime_files, supported_wheel, windows_wheels,
)
from test_detector import phantom


def test_source_archive_is_reproducible_and_has_only_verified_runtime(tmp_path):
    first = build(ROOT, tmp_path / "first")[0]
    second = build(ROOT, tmp_path / "second")[0]
    assert first.read_bytes() == second.read_bytes()
    with ZipFile(first) as archive:
        files = {name.removeprefix(f"{BUNDLE_NAME}/"): archive.read(name) for name in archive.namelist()}
    assert set(files) == set(SOURCE_HASHES) | {"requirements.txt", "START_HERE.md", "MANIFEST.json"}
    manifest = json.loads(files.pop("MANIFEST.json"))
    assert manifest["pipeline"] == "score-before-merge"
    assert manifest["files_sha256"] == {
        name: hashlib.sha256(content).hexdigest() for name, content in files.items()
    }
    for name, expected in SOURCE_HASHES.items():
        assert hashlib.sha256(files[name]).hexdigest() == expected


def test_changed_runtime_is_rejected_before_writing_archive(tmp_path):
    for name in SOURCE_HASHES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    (tmp_path / "pipeline.py").write_text("raise RuntimeError('wrong algorithm')\n")
    with pytest.raises(ValueError, match="pipeline.py"):
        runtime_files(tmp_path)


def test_existing_output_is_never_overwritten(tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        build(ROOT, tmp_path)
    assert marker.read_text() == "keep"
    assert list(tmp_path.iterdir()) == [marker]


def test_wheel_selection_excludes_extra_packages_and_rejects_missing_windows_wheel(tmp_path):
    wanted = tmp_path / "numpy-2.5.3-cp313-cp313-win_amd64.whl"
    wanted.write_bytes(b"selected")
    (tmp_path / "numpy-2.5.3-cp313-cp313-manylinux_2_27_x86_64.whl").write_bytes(b"linux")
    (tmp_path / "matplotlib-3.11.2-cp313-cp313-win_amd64.whl").write_bytes(b"unneeded")
    assert windows_wheels(tmp_path, b"numpy==2.5.3\n") == {f"windows-wheelhouse/{wanted.name}": b"selected"}
    with pytest.raises(ValueError, match="exactly one"):
        windows_wheels(tmp_path, b"numpy==2.5.3\nscipy==1.16.2\n")
    with pytest.raises(ValueError, match="pinned"):
        windows_wheels(tmp_path, b"numpy>=2\n")


@pytest.mark.parametrize(("filename", "supported"), [
    ("numpy-2.5.3-cp313-cp313-win_amd64.whl", True),
    ("simpleitk-2.5.6-cp311-abi3-win_amd64.whl", True),
    ("imageio-2.37.4-py3-none-any.whl", True),
    ("numpy-2.5.3-cp312-cp312-win_amd64.whl", False),
    ("numpy-2.5.3-cp313-cp313t-win_amd64.whl", False),
    ("simpleitk-2.5.6-cp314-abi3-win_amd64.whl", False),
    ("numpy-2.5.3-cp313-cp313-win_arm64.whl", False),
])
def test_windows_wheels_match_python_313_and_architecture(filename, supported):
    assert supported_wheel(Path(filename)) is supported


def test_extracted_application_runs_from_unrelated_directory_and_requires_bundled_model(tmp_path):
    archive_path = build(ROOT, tmp_path / "build")[0]
    isolated = tmp_path / "isolated"
    with ZipFile(archive_path) as archive:
        archive.extractall(isolated)
    bundle = isolated / BUNDLE_NAME
    image, mask = phantom()
    image_path, mask_path = tmp_path / "scan.nii.gz", tmp_path / "mask.nii.gz"
    sitk.WriteImage(image, str(image_path))
    sitk.WriteImage(mask, str(mask_path))
    output, diagnostics = tmp_path / "prediction.json", tmp_path / "diagnostics.json"
    args = ["--image", str(image_path), "--aorta-mask", str(mask_path)]
    subprocess.run(
        [sys.executable, str(bundle / "run.py"), *args, "--output", str(output),
         "--diagnostics", str(diagnostics)], cwd=tmp_path, check=True, capture_output=True,
    )
    baseline = tmp_path / "baseline.json"
    subprocess.run(
        [sys.executable, str(ROOT / "run.py"), *args, "--output", str(baseline)],
        cwd=tmp_path, check=True, capture_output=True,
    )
    assert json.loads(output.read_text()) == json.loads(baseline.read_text())
    workflow = json.loads(diagnostics.read_text())["workflow"]
    assert workflow["name"] == "score-before-merge"
    assert workflow["candidate_model"]["threshold"] == 0.15
    assert Path(workflow["candidate_model"]["path"]).is_relative_to(bundle)
    assert json.loads(output.read_text())["daughters"]
    (bundle / "models/production-v1/logistic.json").write_text("{}")
    failed = subprocess.run(
        [sys.executable, str(bundle / "run.py"), *args, "--output", str(tmp_path / "bad.json")],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert failed.returncode == 1
    assert "hash differs" in failed.stderr
    assert not (tmp_path / "bad.json").exists()
