#!/usr/bin/env python3
"""
Human review workflow: run the loose review-profile detector, record every candidate it proposes,
then open the Aorta Explorer so an engineer can confirm or reject each one.

    python review.py --cases subject001 subject002        # detect, record, open the Explorer
    python review.py --all                                # every available case
    python review.py --status                             # per-case progress of labels/reviews.json

The submission CLI (run.py) is untouched: the review profile exists only to build labels.
"""

import argparse
import json
from pathlib import Path
import sys
from threading import Timer
import webbrowser
from http.server import ThreadingHTTPServer

import SimpleITK as sitk

from detector import DetectorConfig, detect_pool
from explorer import ROOT, CaseStore, make_handler
from learning import FEATURE_NAMES, features as candidate_features
from nifti_io import read_nifti
from review_page import ReviewLedger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--cases", nargs="*", default=[], help="Case directory names to review.")
    parser.add_argument("--all", action="store_true", help="Review every case with resolved scan files.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "review",
                        help="Where the proposed candidate pool is recorded, one JSON per case.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--skip-detect", action="store_true", help="Open the Explorer without re-recording the pool.")
    parser.add_argument("--reviews", type=Path, default=ROOT / "labels" / "reviews.json",
                        help="Verdict file, git-tracked so labels are shared; written on every click.")
    parser.add_argument("--status", action="store_true", help="Print per-case progress of --reviews and exit.")
    return parser.parse_args()


def case_paths(directory: Path) -> tuple[Path, Path]:
    images, masks = sorted(directory.glob("orig*.nii*")), sorted(directory.glob("mask*.nii*"))
    if len(images) != 1 or len(masks) != 1:
        raise ValueError(f"{directory.name}: expected one orig*.nii[.gz] and one mask*.nii[.gz].")
    return images[0], masks[0]


def record_pool(directories: list[Path], output_dir: Path, config: DetectorConfig) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for directory in directories:
        image_path, mask_path = case_paths(directory)
        result = detect_pool(read_nifti(str(image_path)), read_nifti(str(mask_path)), config)
        pool = {
            "case_id": directory.name,
            "profile": config.profile,
            "feature_names": FEATURE_NAMES,
            "candidates": [
                {**b.prediction(), "evidence_score": b.evidence_score, "warnings": b.warnings,
                 "features": dict(zip(FEATURE_NAMES, candidate_features(b)))}
                for b in result.branches
            ],
            "rejections": result.rejections,
            "blood_model": result.blood_model,
            "timings": result.timings,
        }
        (output_dir / f"{directory.name}.json").write_text(json.dumps(pool, indent=2, allow_nan=False) + "\n")
        record = {"case_id": directory.name, "candidates": len(result.branches),
                  "rejected": sum(result.rejections.values()), "seconds": result.timings["total_s"]}
        records.append(record)
        print(json.dumps(record), flush=True)
    (output_dir / "pool_summary.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


def print_status(reviews_path: Path, output_dir: Path) -> None:
    payload = json.loads(reviews_path.read_text())
    verdicts: dict[str, dict[str, int]] = {}
    for row in payload.get("records", []):
        counts = verdicts.setdefault(row["case_id"], {"confirmed": 0, "rejected": 0})
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    pools = {p.stem: json.loads(p.read_text()) for p in sorted(output_dir.glob("subject*.json"))}
    case_ids = sorted(set(verdicts) | set(pools))
    print(f"{'case':<12}{'pool':>6}{'confirmed':>11}{'rejected':>10}{'pending':>9}")
    total_pending = 0
    for case_id in case_ids:
        pool = len(pools.get(case_id, {}).get("candidates", []))
        counts = verdicts.get(case_id, {"confirmed": 0, "rejected": 0})
        pending = max(0, pool - counts["confirmed"] - counts["rejected"]) if pool else 0
        total_pending += pending
        print(f"{case_id:<12}{pool or '-':>6}{counts['confirmed']:>11}{counts['rejected']:>10}{pending:>9}")
    print(f"\n{len(payload.get('records', []))} verdicts across {len(verdicts)} cases; {total_pending} candidates still pending.")


def main() -> int:
    args = parse_args()
    if args.status:
        if not args.reviews.is_file():
            print(f"No verdicts yet at {args.reviews}.", file=sys.stderr)
            return 1
        print_status(args.reviews, args.output_dir)
        return 0
    if not args.data_root.is_dir():
        print("The data directory does not exist.", file=sys.stderr)
        return 2
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        print("Frontend not built. Run: npm --prefix web install && npm --prefix web run build", file=sys.stderr)
        return 2
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    config = DetectorConfig.review()
    store = CaseStore(args.data_root, config)
    available = [c["id"] for c in store.list_cases() if c["available"]]
    selected = available if args.all else args.cases
    unknown = sorted(set(selected) - set(available))
    if unknown:
        print(f"Unknown or unresolved cases: {unknown}. Run git lfs pull first.", file=sys.stderr)
        return 2
    if not selected:
        print("Pass --cases subjectNNN ... or --all.", file=sys.stderr)
        return 2
    if not args.skip_detect:
        record_pool([args.data_root / c for c in selected], args.output_dir, config)
        print(f"Candidate pool recorded under {args.output_dir}", flush=True)
    ledger = ReviewLedger(args.reviews)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store, ROOT / "web" / "dist", ledger))
    url = f"http://127.0.0.1:{args.port}/review/{selected[0]}"
    print(f"\nReview {len(selected)} case(s) at {url}", flush=True)
    print(f"Verdicts save to {ledger.path} as you click; the 3D explorer stays at http://127.0.0.1:{args.port}/", flush=True)
    print("Ctrl+C stops the server.", flush=True)
    print("Progress: python review.py --status   ·   share labels: git add labels/reviews.json && git commit", flush=True)
    if not args.no_browser:
        Timer(1.0, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        store.executor.shutdown(wait=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
