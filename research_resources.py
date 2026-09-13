"""Portable optional process-tree RSS sampler and CPU-limit wrapper for research commands."""

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


def cpu_limit(cores: int, apply: bool = False) -> tuple[list[int] | None, int, str]:
    """Return an honest process/thread CPU limit, applying affinity where supported."""
    require_psutil()
    if type(cores) is not int or not 1 <= cores <= 4:
        raise ValueError("CPU limit must be an integer from 1 to 4.")
    process = psutil.Process()
    affinity_method = getattr(process, "cpu_affinity", None)
    if affinity_method is not None:
        try:
            allowed = affinity_method()
        except NotImplementedError:
            pass
        else:
            affinity = [int(cpu) for cpu in allowed[:cores]]
            if not affinity:
                raise RuntimeError("No CPUs are available to the resource-measurement process.")
            if apply:
                affinity_method(affinity)
            return affinity, len(affinity), "os_affinity_and_thread_environment"
    logical_cpus = psutil.cpu_count(logical=True) or os.cpu_count() or 1
    effective = min(cores, int(logical_cpus))
    return None, effective, "thread_environment_only"


def worker(command: list[str], cores: int) -> int:
    cpu_limit(cores, apply=True)
    return subprocess.call(command)


def network_probe() -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM):
            return False
    except OSError as error:
        return error.errno in (errno.EPERM, errno.EACCES, errno.ENETUNREACH)


def measure(command: list[str], case_id: str, cores: int = 4, interval: float = 0.01) -> dict:
    require_psutil()
    if not command or type(cores) is not int or not 1 <= cores <= 4 or not 0.001 <= interval <= 1:
        raise ValueError("Supply a command, 1–4 cores, and a sampling interval from 0.001 to 1 second.")
    affinity, effective_cores, limit_method = cpu_limit(cores)
    env = dict(os.environ, **{key: str(effective_cores) for key in THREAD_VARIABLES})
    start = perf_counter()
    process = subprocess.Popen([
        sys.executable, str(Path(__file__).resolve()), "--worker", "--cores", str(effective_cores), "--", *command,
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
        try:
            children = psutil.Process(process.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            children = []
        for child in reversed(children):
            child.terminate()
        process.terminate()
        process.wait()
        raise
    network_denied = network_probe()
    affinity_limitation = (
        "OS process affinity and cooperative library thread ceilings were applied."
        if affinity is not None else
        "OS process affinity is unavailable; the CPU ceiling is cooperative for runtimes honoring the recorded "
        "thread environment."
    )
    return {
        "schema_version": 1, "platform": platform.system(), "platform_detail": platform.platform(),
        "python": platform.python_version(), "cpu_cores": effective_cores, "cpu_affinity": affinity,
        "cpu_limit_method": limit_method,
        "gpu": None, "network": False if network_denied else None,
        "network_evidence": "socket creation denied" if network_denied else "not verified",
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
            "Wall time includes process startup. GPU use is not verified by this generic wrapper. "
            + affinity_limitation,
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
