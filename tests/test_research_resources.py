import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from research_resources import measure, psutil

pytestmark = pytest.mark.skipif(psutil is None, reason="Optional requirements-resources.txt not installed.")


def test_measurement_includes_descendants_and_inherits_cpu_limits():
    child = (
        "import os,time; "
        "names=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',"
        "'NUMEXPR_NUM_THREADS','ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS','VECLIB_MAXIMUM_THREADS'); "
        "values={os.environ[name] for name in names}; assert len(values)==1; "
        "effective=int(values.pop()); assert 1<=effective<=2; "
        "memory=bytearray(48*1024*1024); time.sleep(0.15)"
    )
    parent = (
        "import subprocess,sys; memory=bytearray(32*1024*1024); "
        f"subprocess.run([sys.executable,'-c',{child!r}],check=True)"
    )
    result = measure([sys.executable, "-c", parent], "fixture", cores=2, interval=0.005)
    assert result["exit_code"] == 0
    assert 1 <= result["cpu_cores"] <= 2
    assert set(result["thread_environment"].values()) == {str(result["cpu_cores"])}
    if result["cpu_affinity"] is None:
        assert result["cpu_limit_method"] == "thread_environment_only"
    else:
        assert len(result["cpu_affinity"]) == result["cpu_cores"]
        assert result["cpu_limit_method"] == "os_affinity_and_thread_environment"
    row = result["cases"]["fixture"]
    assert row["max_processes"] >= 3
    assert row["peak_rss_mib"] > 80
    assert row["runtime_s"] >= 0.15
    assert row["samples"] > 1
    assert result["gpu"] is None


def test_cli_records_command_failure_and_does_not_overwrite(tmp_path):
    path = tmp_path / "resource.json"
    command = [
        sys.executable, str(Path(__file__).resolve().parents[1] / "research_resources.py"),
        "--output", str(path), "--", sys.executable, "-c", "raise SystemExit(7)",
    ]
    completed = subprocess.run(command, env=os.environ.copy(), capture_output=True, text=True)
    assert completed.returncode == 7
    assert json.loads(path.read_text())["exit_code"] == 7
    previous = path.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode == 1
    assert path.read_bytes() == previous
