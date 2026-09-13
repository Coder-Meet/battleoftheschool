"""Portable optional process-tree RSS sampler and CPU-affinity wrapper for research commands."""

import argparse
import errno
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from time import perf_counter

try:
    import psutil
except ImportError:
    psutil = None

THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def require_psutil() -> None:
    if psutil is None:
        raise RuntimeError("Install requirements-resources.txt for process-tree measurements.")


def sample_tree(pid: int) -> tuple[int, int]:
    require_psutil()
    total, count = 0, 0
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0, 0
    for process in processes:
        try:
            total += process.memory_info().rss
            count += 1
        except psutil.NoSuchProcess:
            continue
    return total, count


def worker(command: list[str], cores: int) -> int:
    require_psutil()
    process = psutil.Process()
    allowed = process.cpu_affinity()
    process.cpu_affinity(allowed[:cores])
    return subprocess.call(command)


def network_probe() -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM):
            return False
    except OSError as error:
        return error.errno in (errno.EPERM, errno.EACCES)


def measure(command: list[str], case_id: str, cores: int = 4, interval: float = 0.01) -> dict:
    require_psutil()
    if not command or type(cores) is not int or not 1 <= cores <= 4 or not 0.001 <= interval <= 1:
        raise ValueError("Supply a command, 1–4 cores, and a sampling interval from 0.001 to 1 second.")
    allowed = psutil.Process().cpu_affinity()
    affinity = allowed[:cores]
    env = dict(os.environ, **{key: str(len(affinity)) for key in THREAD_VARIABLES})
    start = perf_counter()
    process = subprocess.Popen([
        sys.executable, str(Path(__file__).resolve()), "--worker", "--cores", str(cores), "--", *command,
    ], env=env)
    peak, samples, max_processes = 0, 0, 0
    try:
        while True:
            rss, count = sample_tree(process.pid)
            peak, max_processes = max(peak, rss), max(max_processes, count)
            samples += 1
            try:
                code = process.wait(timeout=interval)
                break
            except subprocess.TimeoutExpired:
                continue
    except BaseException:
        for child in reversed(psutil.Process(process.pid).children(recursive=True)):
            child.terminate()
        process.terminate()
        process.wait()
        raise
    return {
        "schema_version": 1, "platform": platform.system(), "platform_detail": platform.platform(),
        "python": platform.python_version(), "cpu_cores": len(affinity), "cpu_affinity": affinity,
        "gpu": None, "network": False if network_probe() else None,
        "network_evidence": "socket creation denied" if network_probe() else "not verified",
        "target_Windows_verified": platform.system() == "Windows",
        "command": command, "exit_code": code, "thread_environment": {k: env[k] for k in THREAD_VARIABLES},
        "cases": {case_id: {
            "runtime_s": perf_counter() - start, "peak_rss_mb": peak / 1_000_000,
            "peak_rss_mib": peak / 2**20, "samples": samples, "max_processes": max_processes,
        }},
        "measurement": {
            "scope": "sum of sampled RSS for wrapper and all live descendants, excluding outer sampler",
            "interval_s": interval,
            "limitations": "Sampling can miss short peaks; shared pages may be counted more than once. "
                          "Wall time includes process startup. GPU use is not verified by this generic wrapper.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case-id", default="command")
    parser.add_argument("--cores", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--interval", type=float, default=0.01)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        if args.worker:
            return worker(command, args.cores)
        if args.output is None or args.output.exists():
            raise ValueError("Choose a new --output report path.")
        report = measure(command, args.case_id, args.cores, args.interval)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return int(report["exit_code"])
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Resource measurement: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
