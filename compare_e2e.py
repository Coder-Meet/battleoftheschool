#!/usr/bin/env python3
"""
Run every case through the detector end to end, with and without the candidate filter, and score both
JSON outputs against a directory of references.

    python compare_e2e.py --candidate-model labels/candidate-model.json --references labels/pseudo_references \\
        --split labels/split.json --output-dir outputs/e2e

Detection runs once per case through the same detect()/filter_detection() functions run.py uses; the plain
prediction is written before the filter is applied, the filtered one after. Scores are pooled per split
partition so cases whose labels trained the weights are never mixed with held-out ones.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import sys

import SimpleITK as sitk

from detector import detect
from evaluate import summarize_cases
from explorer import ROOT
from learning import CandidateModel, filter_detection
from nifti_io import read_nifti
from score_references import score

PARTITION_ORDER = ("test", "validation", "train", "unsplit")


def case_paths(directory: Path) -> tuple[Path, Path] | None:
    images, masks = sorted(directory.glob("orig*.nii*")), sorted(directory.glob("mask*.nii*"))
    if len(images) != 1 or len(masks) != 1:
        return None
    for path in (images[0], masks[0]):
        with path.open("rb") as source:
            if source.read(40).startswith(b"version https://git-lfs"):
                return None
    return images[0], masks[0]


def run_cases(data_root: Path, cases: list[str], model: CandidateModel | None, output_dir: Path) -> dict[str, dict]:
    plain_dir, filtered_dir = output_dir / "plain", output_dir / "filtered"
    plain_dir.mkdir(parents=True, exist_ok=True)
    filtered_dir.mkdir(parents=True, exist_ok=True)
    runs: dict[str, dict] = {}
    for case_id in cases:
        paths = case_paths(data_root / case_id)
        if paths is None:
            print(f"{case_id}: skipped (missing or unresolved scan files)", flush=True)
            continue
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = detect(read_nifti(str(paths[0])), read_nifti(str(paths[1])))
        plain = result.prediction(case_id)
        (plain_dir / f"{case_id}.json").write_text(json.dumps(plain, indent=2, allow_nan=False) + "\n")
        decisions = filter_detection(result, model) if model else None
        filtered = result.prediction(case_id)
        (filtered_dir / f"{case_id}.json").write_text(json.dumps(filtered, indent=2, allow_nan=False) + "\n")
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)
        runs[case_id] = {
            "plain_daughters": len(plain["daughters"]), "filtered_daughters": len(filtered["daughters"]),
            "removed_by_filter": decisions["rejected"] if decisions else 0,
            "detection_s": result.timings["total_s"], "peak_rss_mb_so_far": round(peak_mb, 1),
            "rss_grew_mb": round(peak_mb - before / (1024 * 1024 if sys.platform == "darwin" else 1024), 1),
        }
        print(f"{case_id}: {len(plain['daughters'])} -> {len(filtered['daughters'])} daughters "
              f"({result.timings['total_s']:.1f}s)", flush=True)
    return runs


def partition_of(split: dict[str, list[str]] | None, case_id: str) -> str:
    if split:
        for name, cases in split.items():
            if case_id in cases:
                return name
    return "unsplit"


def pooled(cases: list[dict], tolerance: str) -> dict:
    return summarize_cases([c["by_tolerance"][tolerance] for c in cases])


def fmt(value, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def render(report: dict, tolerance: str) -> str:
    lines = [f"tolerance {tolerance} mm · pooled counts per partition (plain -> filtered)", ""]
    lines.append(f"{'partition':<11}{'cases':>6}{'refs':>6}   {'TP':>7}{'FP':>9}{'FN':>7}{'precision':>14}{'recall':>13}{'F1':>13}")
    for name in PARTITION_ORDER:
        block = report["partitions"].get(name)
        if not block:
            continue
        p, f = block["plain"][tolerance], block["filtered"][tolerance]
        lines.append(
            f"{name:<11}{block['cases']:>6}{block['reference_daughters']:>6}   "
            f"{p['true_positives']:>3}->{f['true_positives']:<3}{p['false_positives']:>4}->{f['false_positives']:<3}"
            f"{p['false_negatives']:>3}->{f['false_negatives']:<3}"
            f"{fmt(p['precision']):>7}->{fmt(f['precision']):<5}{fmt(p['recall']):>7}->{fmt(f['recall']):<5}"
            f"{fmt(p['f1']):>7}->{fmt(f['f1']):<5}"
        )
    lines += ["", f"{'case':<11}{'part':<11}{'refs':>5}{'plain':>7}{'filt':>6}   {'TP':>6}{'FP':>8}{'FN':>7}{'sec':>7}"]
    for row in report["cases"]:
        p, f = row["plain"][tolerance], row["filtered"][tolerance]
        lines.append(
            f"{row['case_id']:<11}{row['partition']:<11}{row['reference_daughters']:>5}{row['plain_daughters']:>7}"
            f"{row['filtered_daughters']:>6}   {p['true_positives']:>2}->{f['true_positives']:<2}"
            f"{p['false_positives']:>3}->{f['false_positives']:<2}{p['false_negatives']:>3}->{f['false_negatives']:<2}"
            f"{row['detection_s']:>7.1f}"
        )
    lines += ["", f"mean detection {report['mean_detection_s']:.1f} s · max {report['max_detection_s']:.1f} s · "
              f"peak RSS {report['peak_rss_mb']:.0f} MB"]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--split", type=Path, help="learning.py split JSON; cases are pooled per partition.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "e2e")
    parser.add_argument("--cases", nargs="*", default=[])
    parser.add_argument("--tolerances-mm", type=float, nargs="+", default=[2, 3, 5])
    parser.add_argument("--report-tolerance-mm", type=float, default=3)
    args = parser.parse_args()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    try:
        model = CandidateModel.load(args.candidate_model)
        split = json.loads(args.split.read_text()) if args.split else None
        cases = args.cases or sorted(p.name for p in args.data_root.iterdir() if p.is_dir())
        runs = run_cases(args.data_root, cases, model, args.output_dir)
        scored = {
            name: score(args.references, args.output_dir / name, args.data_root, args.tolerances_mm)
            for name in ("plain", "filtered")
        }
        by_case = {name: {c["case_id"]: c for c in scored[name]["cases"]} for name in scored}
        rows = []
        for case_id, run in runs.items():
            if case_id not in by_case["plain"]:
                print(f"{case_id}: no reference, not scored", flush=True)
                continue
            rows.append({
                "case_id": case_id, "partition": partition_of(split, case_id),
                "reference_daughters": by_case["plain"][case_id]["reference_daughters"], **run,
                "plain": by_case["plain"][case_id]["by_tolerance"],
                "filtered": by_case["filtered"][case_id]["by_tolerance"],
            })
        partitions = {}
        for name in PARTITION_ORDER:
            members = [r for r in rows if r["partition"] == name]
            if members:
                partitions[name] = {
                    "cases": len(members), "reference_daughters": sum(r["reference_daughters"] for r in members),
                    "plain": {f"{t:g}": pooled([{"by_tolerance": r["plain"]} for r in members], f"{t:g}") for t in args.tolerances_mm},
                    "filtered": {f"{t:g}": pooled([{"by_tolerance": r["filtered"]} for r in members], f"{t:g}") for t in args.tolerances_mm},
                }
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "candidate_model": str(args.candidate_model), "references": str(args.references),
            "note": "Train/validation partitions were used to fit the filter; only 'test' is a held-out estimate.",
            "partitions": partitions, "cases": rows,
            "mean_detection_s": sum(r["detection_s"] for r in rows) / max(len(rows), 1),
            "max_detection_s": max((r["detection_s"] for r in rows), default=0.0),
            "peak_rss_mb": max((r["peak_rss_mb_so_far"] for r in rows), default=0.0),
            "missing_predictions": scored["plain"]["missing_predictions"],
        }
        (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print()
        print(render(report, f"{args.report_tolerance_mm:g}"))
        print(f"\nreport: {args.output_dir / 'comparison.json'}")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"compare_e2e: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
