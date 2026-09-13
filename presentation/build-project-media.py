"""Build public project graphics from recorded UI and frozen evaluation receipts."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
BG, PANEL, INK, MUTED, MINT, AMBER = (
    "#0D1B22", "#172B33", "#F4F0E7", "#B3C5C8", "#80E3C5", "#F3CA83",
)
RECEIPTS = {
    "references": "labels/finalization/accuracy-recheck/v2-references/report.json",
    "topology": "labels/finalization/accuracy-recheck/v2-topology/report.json",
    "batch": "labels/finalization/batch_report.json",
    "resources": "labels/finalization/final-batch-resource.json",
}


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class Graphic:
    def __init__(self, fonts: Path, height: int = 1080):
        self.image = Image.new("RGB", (1920, height), BG)
        self.draw = ImageDraw.Draw(self.image)
        self.fonts = fonts

    def text(self, x: int, y: int, value: str, size: int = 30, color: str = INK,
             display: bool = False, width: int = 1760) -> None:
        font = ImageFont.truetype(str(self.fonts / ("display.ttf" if display else "body.ttf")), size)
        if self.draw.textlength(value, font=font) > width:
            raise ValueError(f"Text exceeds allocated width: {value}")
        self.draw.text((x, y), value, font=font, fill=color)

    def card(self, box: tuple[int, int, int, int]) -> None:
        self.draw.rounded_rectangle(box, radius=24, fill=PANEL)

    def brand(self, subtitle: str) -> None:
        self.text(80, 45, "BRANCHSEED / AORTA EXPLORER", 25, MINT)
        self.text(80, 94, subtitle, 57, display=True)

    def save(self, path: Path) -> None:
        self.image.save(path, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, default=ROOT / "outputs/live-presentation-kit")
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/presentation-kit/source-footage.mp4")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/media")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fonts = args.kit / "assets"
    reports = {name: json.loads((ROOT / path).read_text()) for name, path in RECEIPTS.items()}
    reference = reports["references"]["results"]["strict"]["scores"]["3"]
    topology = reports["topology"]["results"]["strict"]["scores"]["3"]
    cases = reports["batch"]["cases"]
    assert len(cases) == 25 and all(case["status"] == "ok" for case in cases)
    times = [float(case["end_to_end_s"]) for case in cases]
    mean_time, maximum_time = sum(times) / len(times), max(times)
    rss = float(reports["resources"]["cases"]["all-25"]["peak_rss_mib"])
    for scores in (reference, topology):
        tp, fp, fn = (int(scores[key]) for key in ("true_positives", "false_positives", "false_negatives"))
        assert abs(float(scores["f1"]) - 2 * tp / (2 * tp + fp + fn)) < 1e-12
    assert reports["resources"]["platform"] == "Linux"
    assert reports["resources"]["cpu_cores"] == 4
    assert reports["resources"]["target_Windows_verified"] is False
    source_hash = digest(args.source)
    shutil.copy2(fonts / "explorer-poster.png", args.output / "explorer-overview.png")
    for name, second in (("ct-evidence", 68), ("wall-map", 90), ("interior-tour", 121)):
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(second),
            "-i", str(args.source), "-frames:v", "1", str(args.output / f"{name}.png"),
        ], check=True)

    hero = Graphic(fonts)
    hero.text(80, 60, "BRANCHSEED", 32, MINT)
    hero.text(80, 154, "Find the branch.", 76, display=True, width=710)
    hero.text(80, 247, "Keep the evidence.", 70, display=True, width=730)
    for y, line in enumerate(("Automatic aortic branch discovery.",
                              "Linked 3D and CT inspection.",
                              "Physical measurements. Offline.")):
        hero.text(80, 380 + y * 53, line, 32, MUTED, width=710)
    hero.text(80, 609, "CT + parent mask", 32, MINT)
    hero.text(80, 663, "Origins / seeds / radii / directions", 28, width=710)
    with Image.open(args.output / "explorer-overview.png") as source:
        screenshot = ImageOps.contain(source.convert("RGB"), (1030, 640))
        hero.image.paste(screenshot, (825, 150 + (640 - screenshot.height) // 2))
    hero.text(825, 804, "Recorded interface / historical predictions", 24, MUTED, width=1030)
    for x, value, label in ((80, "25 / 25", "supplied scans completed"),
                            (690, f"{mean_time:.2f} s", "mean time per scan"),
                            (1300, "CPU / offline", "no external inference service")):
        hero.card((x, 877, x + 540, 1010))
        hero.text(x + 25, 890, value, 45, MINT, width=495)
        hero.text(x + 25, 954, label, 25, width=495)
    hero.text(80, 1030, "Runtime: four-core Linux development run. Organizer-Windows timing unverified.", 24, MUTED)
    hero.save(args.output / "branchseed-cover.png")

    scorecard = Graphic(fonts)
    scorecard.brand("Evidence you can inspect.")
    for x, scores, title, cohort in (
        (80, reference, "LOCAL REFERENCE AGREEMENT", "5 reused cases / 19 AI-assisted targets"),
        (1010, topology, "SYNTHETIC TOPOLOGY REGRESSION", "24 procedural cases / 49 analytic targets"),
    ):
        scorecard.card((x, 213, x + 830, 846))
        scorecard.text(x + 35, 242, title, 28, MINT, width=760)
        scorecard.text(x + 35, 301, cohort, 28, MUTED, width=760)
        scorecard.text(x + 35, 371, f"F1 {float(scores['f1']):.3f}", 111, display=True, width=760)
        scorecard.text(x + 35, 540,
                       f"{scores['true_positives']} TP / {scores['false_positives']} FP / {scores['false_negatives']} FN",
                       40, width=760)
        scorecard.text(x + 35, 626, f"Precision {float(scores['precision']):.3f}   Recall {float(scores['recall']):.3f}",
                       30, MUTED, width=760)
        scorecard.text(x + 35, 693, f"Count MAE: {float(scores['count_mae']):g} daughters / case", 30, width=760)
    scorecard.text(115, 781, "References may omit eligible branches.", 27, AMBER, width=760)
    scorecard.text(1045, 781, "Regression evidence, not patient accuracy.", 27, AMBER, width=760)
    scorecard.text(80, 889, "SUBMITTED CONFIGURATION / deterministic strict / 1 mm grid / native contrast 1.2", 27, MINT)
    scorecard.text(80, 948, "Both use local 3 mm one-to-one ostium matching. These are separate evaluations.", 29)
    scorecard.text(80, 1000, "No independent hidden-test score or clinical validation is available.", 29, MUTED)
    scorecard.save(args.output / "evidence-scorecard.png")

    runtime = Graphic(fonts)
    runtime.brand("All 25 supplied scans completed.")
    for x, value, label in ((80, f"{mean_time:.2f} s", "mean per-case time"),
                            (690, f"{maximum_time:.2f} s", "maximum per-case time"),
                            (1300, f"{rss:.0f} MiB", "maximum sampled process-tree RSS")):
        runtime.card((x, 210, x + 540, 365))
        runtime.text(x + 25, 220, value, 65, MINT, width=495)
        runtime.text(x + 25, 312, label, 24, width=495)
    left, top, bottom, plot_width = 150, 470, 855, 1650
    for tick in range(0, 26, 5):
        y = bottom - round((bottom - top) * tick / 25)
        runtime.draw.line((left, y, left + plot_width, y), fill="#34515C", width=1)
        runtime.text(82, y - 18, str(tick), 24, MUTED, width=60)
    runtime.text(80, 411, "SECONDS / PER-CASE END-TO-END TIME", 25, MUTED)
    step = plot_width / len(cases)
    for index, seconds in enumerate(times):
        x = round(left + step * index + 12)
        y = bottom - round((bottom - top) * seconds / 25)
        runtime.draw.rectangle((x, y, x + 39, bottom), fill=AMBER if seconds == maximum_time else MINT)
        runtime.text(x - 5, 868, cases[index]["case_id"].removeprefix("subject"), 21, MUTED, width=60)
    runtime.text(80, 943, "Four-core Linux affinity + thread limits. Per-case batch times; sampled RSS can miss peaks.", 28)
    runtime.text(80, 995, "Organizer-Windows timing remains unverified. Completing a scan does not establish detection accuracy.", 27, MUTED)
    runtime.save(args.output / "runtime.png")

    pipeline = Graphic(fonts, height=720)
    pipeline.brand("One parent mask. Explicit geometric decisions.")
    stages = (
        ("01 / ANCHOR", "Validate + normalize", ("CT and parent geometry", "Physical 1 mm working grid")),
        ("02 / DISCOVER", "Find wall contacts", ("Scan-relative contrast", "Multiscale tubular support")),
        ("03 / TRACE", "Connect + measure", ("Supported proximal paths", "Size and origin checks")),
        ("04 / INSPECT", "Review + export", ("3D / CT / wall map", "Physical-coordinate JSON")),
    )
    for index, (label, title, lines) in enumerate(stages):
        x = 80 + 455 * index
        pipeline.card((x, 260, x + 410, 560))
        pipeline.text(x + 25, 280, label, 27, MINT, width=360)
        pipeline.text(x + 25, 352, title, 30, display=True, width=360)
        for number, line in enumerate(lines):
            pipeline.text(x + 25, 429 + number * 44, line, 25, MUTED, width=360)
        if index < 3:
            pipeline.draw.line((x + 417, 410, x + 445, 410), fill=MINT, width=3)
            pipeline.draw.polygon(((x + 445, 404), (x + 454, 410), (x + 445, 416)), fill=MINT)
    pipeline.text(80, 620, "Default: deterministic strict. No learned weights or runtime cloud inference.", 31)
    pipeline.save(args.output / "pipeline.png")

    assert digest(args.source) == source_hash
    manifest = {
        "strict_reference_3mm": reference,
        "strict_synthetic_topology_3mm": topology,
        "runtime": {"cases": len(cases), "mean_per_case_s": mean_time, "max_per_case_s": maximum_time,
                    "max_sampled_process_tree_rss_mib": rss, "platform": "Linux", "cpu_cores": 4},
        "receipts_sha256": {path: digest(ROOT / path) for path in RECEIPTS.values()},
        "recording_sha256": source_hash,
        "recorded_frame_seconds": {"ct-evidence": 68, "wall-map": 90, "interior-tour": 121},
        "image_sha256": {path.name: digest(path) for path in sorted(args.output.glob("*.png"))},
        "scope": "Recorded UI is historical; charts use frozen strict receipts. Not clinical or hidden-test accuracy.",
    }
    (args.output / "metrics.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Built project images in {args.output}; evidence and source hashes recorded.")


if __name__ == "__main__":
    main()
