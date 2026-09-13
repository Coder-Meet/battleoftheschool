import json
from pathlib import Path

import compare_e2e
from stress import generate_case, write_case


def test_windows_memory_measurement_is_explicitly_unavailable(monkeypatch):
    monkeypatch.setattr(compare_e2e.sys, "platform", "win32")
    assert compare_e2e.peak_rss_mib() is None


def test_comparison_keeps_predictions_when_memory_is_unavailable(tmp_path: Path, monkeypatch):
    case = generate_case("negative_controls_only", 4001)
    case_id = case.reference["case_id"]
    write_case(case, tmp_path / "data" / case_id)
    monkeypatch.setattr(compare_e2e, "peak_rss_mib", lambda: None)
    runs = compare_e2e.run_cases(tmp_path / "data", [case_id], None, tmp_path / "output")
    run = runs[case_id]
    assert run["peak_rss_mb_so_far"] is None
    assert run["rss_grew_mb"] is None
    for name in ("plain", "filtered"):
        prediction = json.loads((tmp_path / "output" / name / f"{case_id}.json").read_text())
        assert prediction["daughters"] == []
        assert prediction["case_id"] == case_id
    report = {
        "profile": "strict", "partitions": {}, "cases": [],
        "mean_detection_s": run["detection_s"], "max_detection_s": run["detection_s"],
        "peak_rss_mb": None,
    }
    assert "peak RSS - MiB" in compare_e2e.render(report, "3")
