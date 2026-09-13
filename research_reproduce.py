"""One-command current-source synthetic replay. The cohort's seed groups are already exposed."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from research_corpus import SOURCE_FILES, digest, read_json, source_hashes, versions
from research_resources import THREAD_VARIABLES


def reproduce(root: Path, workers: int = 2, cnn: bool = False) -> dict:
    if root.exists():
        raise ValueError("Use a new corpus root; frozen experiments must not be overwritten.")
    if workers not in (1, 2, 4):
        raise ValueError("Workers must be 1, 2 or 4.")
    code = Path(__file__).resolve().parent
    env = dict(os.environ, **{key: str(4 // workers) for key in THREAD_VARIABLES})
    commands = [
        ["research_corpus.py", "plan"],
        ["research_corpus.py", "generate", "--workers", str(workers)],
        ["research_corpus.py", "export", "--workers", str(workers)],
        ["research_corpus.py", "train-baselines"],
        ["research_corpus.py", "export", "--partition", "test", "--workers", str(workers)],
        ["research_corpus.py", "train-baselines", "--evaluate-test"],
    ]
    receipts = []
    for arguments in commands:
        command = [sys.executable, str(code / arguments[0]), *arguments[1:], "--root", str(root)]
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True, env=env)
        receipts.append(command)
    summary = read_json(root / "summary.json")
    cnn_status = "not_requested"
    if cnn:
        if summary["cnn_stop_go"] != "go_research_only":
            cnn_status = "blocked_by_e1_gate"
        else:
            selected = read_json(root / "models" / "freeze.json")["selected_model"]
            for suffix in ([], ["--evaluate-test"]):
                command = [
                    sys.executable, str(code / "patch_learning.py"), "--corpus", str(root),
                    "--output", str(root / "cnn-synthetic"), "--threads", "4",
                    "--tree-model", str(root / "models" / f"{selected}.json"), *suffix,
                ]
                subprocess.run(command, check=True, env=env)
                receipts.append(command)
            cnn_status = "retrospective_research_evaluated"
    evidence = {
        "schema_version": 1, "source_sha256": source_hashes(), "versions": versions(),
        "driver_sha256": digest(Path(__file__)), "commands": receipts,
        "cohort_status": "previously_exposed_seeds_current_source_replay",
        "clinical_accuracy_claim": False, "cnn_status": cnn_status,
        "summary_sha256": digest(root / "summary.json"),
        "plan_sha256": digest(root / "plan.json"),
        "source_files": list(SOURCE_FILES),
        "historical_detector_sha256": "17495723df18c79e4302edc19bd949bb16f615b902403a88c68d6e74fa200063",
    }
    output = root / "reproduction.json"
    output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    output.with_suffix(".json.sha256").write_text(
        hashlib.sha256(output.read_bytes()).hexdigest() + "\n", encoding="ascii",
    )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--cnn", action="store_true", help="Run retrospective optional CNN only if E1 allows it.")
    args = parser.parse_args()
    try:
        reproduce(args.root.resolve(), args.workers, args.cnn)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Reproduction error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
