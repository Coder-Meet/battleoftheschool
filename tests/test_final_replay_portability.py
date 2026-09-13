import copy
import json

import pytest

import final_eval_select as select


def test_verify_allows_final_bit_reductions_but_preserves_manifest_and_score_checks(tmp_path, monkeypatch):
    saved = {"score": 9.728707802260844, "true_positives": 8, "variant": "strict"}
    replayed = copy.deepcopy(saved)
    replayed["score"] = 9.728707802260837
    (tmp_path / "report.json").write_text(json.dumps(saved))
    (tmp_path / "rankings.json").write_text(json.dumps(saved))
    (tmp_path / "manifest.json").write_text('{"sha256": "unchanged"}')
    (tmp_path / "FINAL_EVALUATION_RESULTS.md").write_text("Results")
    monkeypatch.setattr(select, "evaluate", lambda root: (replayed, replayed, {}, {}, {}))
    monkeypatch.setattr(select, "render_results", lambda report: "Results")
    monkeypatch.setattr(select, "build_manifest", lambda *args: {"sha256": "unchanged"})

    assert select.verify(tmp_path, tmp_path) == replayed
    replayed["score"] += 0.001
    with pytest.raises(ValueError, match="Selection report"):
        select.verify(tmp_path, tmp_path)
    replayed.update(saved, true_positives=9)
    with pytest.raises(ValueError, match="Selection report"):
        select.verify(tmp_path, tmp_path)
    replayed.update(saved, variant="review")
    with pytest.raises(ValueError, match="Selection report"):
        select.verify(tmp_path, tmp_path)
    replayed.update(saved)
    (tmp_path / "manifest.json").write_text('{"sha256": "tampered"}')
    with pytest.raises(ValueError, match="manifest"):
        select.verify(tmp_path, tmp_path)
