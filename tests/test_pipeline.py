import json
from pathlib import Path
import subprocess
import sys

import pytest
import SimpleITK as sitk

from detector import Branch, Detection, DetectorConfig
from learning import CONTEXT_FEATURES, CandidateModel, FEATURE_NAMES
import pipeline


def branch(x, radius):
    return Branch('original', (x, 0, 0), (x, 5, 0), radius, (0, 1, 0),
                  [(x, 0, 0), (x, 5, 0), (x, 10, 0)], .8, .8,
                  features=dict.fromkeys(CONTEXT_FEATURES, 1.0))


def detection(branches, config):
    return Detection(branches, {'total_s': .1}, {}, len(branches), {}, [], config)


@pytest.fixture
def model_path(tmp_path):
    model = CandidateModel([0.] * len(FEATURE_NAMES), [1.] * len(FEATURE_NAMES),
                           [1.] + [0.] * (len(FEATURE_NAMES) - 1), -4., .5,
                           {'train': ['train'], 'validation': ['validation'], 'test': ['test']})
    path = tmp_path / 'model.json'
    model.save(path)
    return path


def test_production_scores_before_merging_and_records_every_decision(monkeypatch, model_path):
    original = branch(0, 8)
    strict = [original]
    review = [branch(1, 1), branch(2, 9), branch(3, 8)]
    monkeypatch.setattr(pipeline, 'detect', lambda image, mask, config: detection(
        strict if config.profile == 'strict' else review, config))
    result, diagnostics = pipeline.run_pipeline(None, None, model_path=model_path)
    assert [b.ostium_xyz_mm for b in result.branches] == [(0, 0, 0), (3, 0, 0)]
    assert [r['decision'] for r in diagnostics['decisions']] == [
        'retained', 'below_threshold', 'merged', 'retained',
    ]
    assert diagnostics['decisions'][2]['suppressor_instance_id'] == 'branch_001'
    assert original.instance_id == 'original'
    assert diagnostics['candidate_model']['threshold'] == .5


def test_shared_physical_settings_and_two_mm_origin_policy(monkeypatch):
    seen = []
    def detect(image, mask, config):
        seen.append(config)
        return detection([], config)
    monkeypatch.setattr(pipeline, 'detect', detect)
    config = DetectorConfig(spacing_mm=.75, minimum_radius_mm=.8, parallel_clearance_mm=1.5)
    result, diagnostics = pipeline.run_pipeline(None, None, config)
    assert len(seen) == 2
    for current in seen:
        assert current.spacing_mm == .75
        assert current.minimum_radius_mm == .8
        assert current.minimum_origin_diameter_mm == 2
        assert current.parallel_clearance_mm == 1.5
    assert result.branches == []
    assert diagnostics['candidate_model']['threshold'] == .15
    assert diagnostics['candidate_model']['saved_threshold'] != .15


def test_strict_escape_hatch_and_explicit_custom_threshold(monkeypatch, model_path):
    calls = []
    def detect(image, mask, config):
        calls.append(config)
        return detection([branch(0, 1)], config)
    monkeypatch.setattr(pipeline, 'detect', detect)
    plain, _ = pipeline.run_pipeline(None, None, workflow='strict')
    assert len(plain.branches) == 1 and len(calls) == 1
    filtered, metadata = pipeline.run_pipeline(None, None, workflow='strict', model_path=model_path, threshold=0)
    assert len(filtered.branches) == 1
    assert metadata['candidate_model']['threshold'] == 0


@pytest.mark.parametrize('threshold', [float('nan'), float('inf'), -1, 1.1])
def test_invalid_threshold_fails_before_detection(monkeypatch, threshold):
    monkeypatch.setattr(pipeline, 'detect', lambda *args: pytest.fail('should not detect'))
    with pytest.raises(ValueError, match='threshold'):
        pipeline.run_pipeline(None, None, threshold=threshold)


def test_model_missing_or_modified_never_falls_back_to_unfiltered(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, 'detect', lambda *args: pytest.fail('should not detect'))
    missing = tmp_path / 'missing.json'
    monkeypatch.setattr(pipeline, 'DEFAULT_MODEL', missing)
    with pytest.raises(OSError):
        pipeline.run_pipeline(None, None)
    missing.write_text('{}')
    with pytest.raises(ValueError, match='hash'):
        pipeline.run_pipeline(None, None)


def test_cli_default_model_resolves_outside_repository_and_batch_matches(tmp_path):
    root = Path(__file__).resolve().parents[1]
    case = tmp_path / 'data' / 'subject001'
    case.mkdir(parents=True)
    image = sitk.Image([8, 8, 8], sitk.sitkInt16)
    mask = sitk.Image([8, 8, 8], sitk.sitkUInt8)
    sitk.WriteImage(image, str(case / 'orig.nii'))
    sitk.WriteImage(mask, str(case / 'mask.nii'))
    output, diagnostics = tmp_path / 'prediction.json', tmp_path / 'diagnostics.json'
    subprocess.run([sys.executable, str(root / 'run.py'), '--image', str(case / 'orig.nii'),
                    '--aorta-mask', str(case / 'mask.nii'), '--output', str(output),
                    '--diagnostics', str(diagnostics)], cwd=tmp_path, check=True, capture_output=True)
    assert json.loads(output.read_text())['daughters'] == []
    metadata = json.loads(diagnostics.read_text())['workflow']
    assert metadata['name'] == 'score-before-merge'
    assert metadata['candidate_model']['threshold'] == .15
    batch = tmp_path / 'batch'
    subprocess.run([sys.executable, str(root / 'batch.py'), '--data-root', str(case.parent),
                    '--output-dir', str(batch)], cwd=tmp_path, check=True, capture_output=True)
    assert json.loads((batch / 'subject001.json').read_text()) == json.loads(output.read_text())
