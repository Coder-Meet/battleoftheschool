#!/usr/bin/env python3
"""Build the offline presentation from actual predictions and local media."""

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import SimpleITK as sitk
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import find_contours, marching_cubes

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[1]
INK = "101C23"
PAPER = "F1EDE3"
MINT = "7BE7C2"
CORAL = "FF927D"
MUTED = "9FACAA"
LINE = "34444A"


@dataclass
class Element:
    kind: str
    x: float
    y: float
    w: float
    h: float
    text: str = ""
    size: float = 32
    color: str = PAPER
    font: str = "body"
    stroke: float = 0
    source: str = ""


@dataclass
class Slide:
    title: str
    speaker: int
    seconds: int
    background: str
    elements: list[Element]
    notes: str = ""

    def text(self, x, y, w, text, size=32, color=PAPER, font="body"):
        self.elements.append(Element("text", x, y, w, size * 1.25 * len(text.splitlines()),
                                     text=text, size=size, color=color, font=font))

    def rect(self, x, y, w, h, color):
        self.elements.append(Element("rect", x, y, w, h, color=color))

    def circle(self, x, y, diameter, color, stroke=0):
        self.elements.append(Element("circle", x, y, diameter, diameter, color=color, stroke=stroke))

    def line(self, x, y, xx, yy, color=LINE, stroke=2):
        self.elements.append(Element("line", x, y, xx - x, yy - y, color=color, stroke=stroke))

    def image(self, x, y, w, h, source):
        self.elements.append(Element("image", x, y, w, h, source=source))


def fonts(output: Path):
    assets = output / "assets"
    sources = {
        "body": ROOT / "web/node_modules/@fontsource-variable/manrope",
        "display": ROOT / "presentation/node_modules/@fontsource-variable/fraunces",
    }
    for family, source in sources.items():
        name = "manrope-latin-wght-normal.woff2" if family == "body" else "fraunces-latin-wght-normal.woff2"
        shutil.copy2(source / "files" / name, assets / f"{family}.woff2")
        shutil.copy2(source / "LICENSE", assets / f"{family}-LICENSE.txt")
        font = TTFont(source / "files" / name)
        fixed = instantiateVariableFont(font, {"wght": 500}, inplace=False)
        family_name = "Branchseed Text" if family == "body" else "Branchseed Display"
        for name_id, value in {
            1: family_name, 2: "Regular", 4: family_name + " Regular",
            6: family_name.replace(" ", "") + "-Regular", 16: family_name, 17: "Regular",
        }.items():
            fixed["name"].setName(value, name_id, 3, 1, 0x409)
            fixed["name"].setName(value, name_id, 1, 0, 0)
        fixed.flavor = None
        fixed.save(assets / f"{family}.ttf")


def evidence(output: Path, predictions: Path):
    for index in (1, 2, 3):
        case = f"subject{index:03}"
        image = sitk.ReadImage(str(ROOT / f"data/{case}/orig{index}.nii"))
        mask = sitk.ReadImage(str(ROOT / f"data/{case}/mask{index}.nii"))
        ct = sitk.GetArrayFromImage(image)
        parent = sitk.GetArrayFromImage(mask) > 0
        pred = json.loads((predictions / f"{case}.json").read_text())
        zz, yy, xx = np.nonzero(parent)
        padding = np.ceil(22 / np.asarray(image.GetSpacing())).astype(int)
        bounds = [(max(0, int(v.min()) - int(p)), min(n, int(v.max()) + int(p) + 1))
                  for v, p, n in zip((xx, yy, zz), padding, image.GetSize())]
        (x0, x1), (y0, y1), (z0, z1) = bounds
        mip = ct[z0:z1, y0:y1, x0:x1].max(axis=1)
        outline = parent[z0:z1, y0:y1, x0:x1].max(axis=1)
        fig, axis = plt.subplots(figsize=(5, 6.5), dpi=180)
        fig.patch.set_facecolor("#" + INK)
        axis.set_facecolor("#" + INK)
        sx, _, sz = image.GetSpacing()
        axis.imshow(mip, cmap="gray", vmin=70, vmax=450, origin="lower",
                    extent=(x0 * sx, x1 * sx, z0 * sz, z1 * sz), interpolation="bilinear")
        for contour in find_contours(outline.astype(float), 0.5):
            axis.plot((contour[:, 1] + x0) * sx, (contour[:, 0] + z0) * sz,
                      color="#" + MINT, linewidth=1.2)
        for branch in pred["daughters"]:
            start = np.array(image.TransformPhysicalPointToContinuousIndex(branch["ostium_xyz_mm"]))
            end_physical = np.array(branch["ostium_xyz_mm"]) + 10 * np.array(branch["direction_xyz"])
            end = np.array(image.TransformPhysicalPointToContinuousIndex(end_physical.tolist()))
            axis.scatter(start[0] * sx, start[2] * sz, s=24, color="#" + CORAL, zorder=6)
            axis.annotate("", xy=(end[0] * sx, end[2] * sz), xytext=(start[0] * sx, start[2] * sz),
                          arrowprops={"arrowstyle": "-|>", "color": "#" + CORAL, "lw": 1.8})
        axis.set_aspect("equal")
        axis.set_axis_off()
        fig.subplots_adjust(0, 0, 1, 1)
        fig.savefig(output / f"assets/check-{index:03}.png", facecolor=fig.get_facecolor())
        plt.close(fig)
        shutil.copy2(predictions / f"{case}.json", output / f"evidence/{case}.json")
        if index == 1:
            verts, faces, _, _ = marching_cubes(parent.transpose(2, 1, 0), 0.5,
                                                 spacing=image.GetSpacing(), step_size=3)
            verts = verts @ np.asarray(image.GetDirection()).reshape(3, 3).T + image.GetOrigin()
            fig = plt.figure(figsize=(7, 8), dpi=190)
            axis3 = fig.add_subplot(projection="3d")
            surface = Poly3DCollection(verts[faces], alpha=1, linewidth=0, shade=True,
                                      lightsource=LightSource(azdeg=305, altdeg=40), facecolors="#71B6A6")
            surface.set_edgecolor("none")
            axis3.add_collection3d(surface)
            for branch in pred["daughters"]:
                ostium = np.array(branch["ostium_xyz_mm"])
                seed = np.array(branch["seed_xyz_mm"])
                axis3.scatter(*ostium, s=110, color="#" + CORAL, depthshade=False)
                axis3.plot(*np.stack([ostium, seed]).T, color="#" + PAPER, linewidth=4)
            lo, hi = verts.min(axis=0), verts.max(axis=0)
            axis3.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]))
            axis3.set_box_aspect(hi - lo, zoom=1.15)
            axis3.view_init(elev=10, azim=-65, roll=12)
            axis3.set_axis_off()
            fig.subplots_adjust(0, 0, 1, 1)
            fig.savefig(output / "assets/vessel.png", transparent=True)
            plt.close(fig)


def base(title, number, speaker, seconds, paper=False):
    slide = Slide(title, speaker, seconds, PAPER if paper else INK, [])
    fg = INK if paper else PAPER
    sub = "526367" if paper else MUTED
    accent = "1A7465" if paper else MINT
    slide.text(86, 54, 800, "BRANCHSEED  /  AORTA EXPLORER", 23, fg)
    slide.text(1320, 54, 520, "TORALIS LABS CHALLENGE", 21, sub)
    slide.line(86, 112, 1834, 112, "C6CEC4" if paper else LINE)
    slide.line(86, 1006, 1834, 1006, "C6CEC4" if paper else LINE)
    slide.line(86, 1006, 86 + 1748 * number / 8, 1006, accent, 4)
    slide.text(86, 1029, 500, f"SPEAKER {speaker}  /  {seconds:02}s", 20, sub)
    slide.text(1370, 1029, 400, f"{number:02}  /  08", 20, sub)
    return slide


def tree(slide, x, y, scale=1):
    def segment(a, b, color=MINT, width=24):
        slide.line(x + a[0] * scale, y + a[1] * scale,
                   x + b[0] * scale, y + b[1] * scale, color, width * scale)
    for a, b in [((120, 0), (120, 330)), ((120, 330), (85, 410)),
                 ((120, 330), (160, 410)), ((120, 90), (220, 130)),
                 ((220, 130), (285, 100)), ((220, 130), (282, 172)),
                 ((120, 220), (20, 230)), ((120, 260), (25, 275))]:
        segment(a, b, MINT if a[0] == 120 else "52786F", 35 if b == (120, 330) else 13)
    for px, py in ((120, 90), (120, 220), (120, 260)):
        slide.circle(x + (px - 8) * scale, y + (py - 8) * scale, 16 * scale, CORAL)


def create_deck():
    verification = json.loads((ROOT / "presentation/evidence/judge-fusion-verification.json").read_text())
    validation = json.loads((ROOT / "docs/fusion-restored-validation.json").read_text())
    reference = verification["reference_scores_3mm"]
    topology = validation["synthetic"]["variants"]["score-before-merge"]["scores"]["3"]
    rss = verification["resource_receipt"]["cases"]["all-25-fusion"]["peak_rss_mib"]
    slides: list[Slide] = []
    s = base("Find the branch. Keep the evidence.", 1, 1, 25)
    s.text(90, 190, 975, "Find the branch.\nKeep the evidence.", 105, PAPER, "display")
    s.text(95, 504, 860, "From a CT and one parent mask\nto inspectable daughter instances.", 36, MUTED)
    s.text(95, 750, 800, "PHYSICAL COORDINATES.  LOCAL COMPUTE.", 23, MINT)
    s.text(95, 820, 860, "Fusion discovery. Built to run offline.", 29)
    s.image(1060, 118, 710, 812, "assets/vessel.png")
    s.text(1118, 924, 650, "Subject001 · parent mask + predicted origins", 21, MUTED)
    s.notes = """Every branch begins with a small opening. The challenge is to find that opening reliably when anatomy and scan coverage change. We built Branchseed: a CPU detector that turns a CT and one aorta mask into individual daughter instances, with an Explorer that lets you inspect the evidence behind each result."""
    slides.append(s)

    s = base("Connection is the challenge.", 2, 1, 50)
    s.text(90, 170, 1500, "Connection is the challenge.", 86, PAPER, "display")
    s.text(94, 295, 1350, "Brightness finds candidates. The aortic wall defines the instance.", 34, MUTED)
    tree(s, 1110, 415, 1.15)
    s.text(94, 430, 950, "01   Two wall openings: two daughters", 34)
    s.text(94, 531, 950, "02   A common trunk: one direct origin", 34)
    s.text(94, 632, 950, "03   Exclude crop caps and distal children", 34)
    s.text(95, 815, 890, "2 mm origin diameter. 5 mm visible continuation.", 30, MINT)
    s.text(95, 878, 910, "We concentrated effort where the challenge puts the weight.", 25, MUTED)
    s.text(1450, 506, 355, "Common trunk\n1 aortic origin", 25, CORAL)
    s.text(1135, 900, 650, "Schematic · not patient anatomy", 21, MUTED)
    s.notes = """The parent mask is a search anchor; it does not label the daughters. Bright structures alone are not enough. Two separate wall openings are two instances, including a vessel that returns to the aorta. A common trunk has one direct origin even when it divides. Distal children and crop caps must be excluded. The judge confirmed a two-millimeter origin diameter and at least five millimeters of visible continuation. Overlapping openings can receive scoring allowances, but the official evaluator was not supplied. Daughter discovery and count are the priority. We search locally, check connection and topology, and preserve physical coordinates. Speaker two will show how the measurements work."""
    slides.append(s)

    s = base("Anchor. Trace. Measure.", 3, 2, 40, True)
    s.text(90, 176, 1600, "Anchor. Trace. Measure.", 94, INK, "display")
    s.text(94, 320, 1550, "Geometry discovers candidates. A small learned filter selects them.", 32, "526367")
    labels = [("01", "Anchor", "Validate the grid.\nSearch around the parent."),
              ("02", "Discover", "Strict + review proposals.\nFind supported wall contacts."),
              ("03", "Trace", "Follow the proximal path.\nStop at the first bifurcation."),
              ("04", "Measure", "Ostium, 5 mm seed,\nradius and unit direction.")]
    for i, (number, title, description) in enumerate(labels):
        x = 96 + i * 440
        s.circle(x, 492, 58, "1A7465", 2)
        s.text(x + 11, 503, 50, number, 22, "1A7465")
        if i < 3:
            s.line(x + 72, 521, x + 415, 521, "AABDB3", 3)
        s.text(x, 596, 400, title, 49, INK, "display")
        s.text(x, 695, 402, description, 26, "526367")
    s.text(95, 880, 1620, "FUSION DEFAULT     /     1 mm working grid     /     native contrast scale 0.9", 28, "1A7465")
    s.notes = """Our submission combines geometric discovery with a small learned filter. We validate physical geometry, crop around the parent and work on a one-millimeter grid. Scan-relative contrast and multiscale vesselness find wall contacts in two automatic passes: strict and broader review proposals. Both use native contrast scale point nine. We trace supported proximal paths, checking connection, crop caps and downstream branching. The result contains an ostium, five-millimeter seed, radius and unit direction in the original physical frame. The Explorer and judge application use this same fusion workflow."""
    slides.append(s)

    s = base("Score first. Then merge.", 4, 2, 35, True)
    s.text(90, 171, 1750, "Score first. Then merge.", 94, INK, "display")
    s.rect(138, 372, 222, 495, "D8E4DC")
    s.line(360, 372, 360, 867, "1A7465", 4)
    s.line(360, 640, 810, 450, "B5DACA", 74)
    s.line(360, 640, 810, 450, "1A7465", 3)
    s.circle(347, 627, 26, "C6503D")
    s.circle(603, 519, 26, "1A7465")
    s.line(360, 699, 616, 590, "526367", 2)
    s.text(427, 698, 290, "5 mm along path", 25, INK)
    s.text(148, 901, 770, "Schematic · seed follows the path, not a voxel offset", 23, "526367")
    s.text(398, 399, 220, "Wall origin", 24, "C6503D")
    s.line(435, 442, 362, 622, "C6503D", 2)
    s.text(677, 595, 230, "Seed + radius", 24, "1A7465")
    s.text(1050, 410, 770, "Keep the physical rules", 40, INK, "display")
    s.text(1050, 479, 770, "2 mm origin diameter. 5 mm continuation.", 27, "526367")
    s.text(1050, 580, 770, "Filter each proposal pass", 40, INK, "display")
    s.text(1050, 649, 770, "Bundled logistic model. Threshold 0.15.", 27, "526367")
    s.text(1050, 750, 770, "Merge survivors within 3 mm", 40, INK, "display")
    s.text(1050, 819, 770, "Keep strict survivors first. Export instances.", 27, "526367")
    s.notes = """The seed follows five millimeters of vessel, not a fixed voxel offset. We score both proposal passes with a bundled thirteen-feature logistic model and filter at point one five before merging. Strict survivors are kept first; review survivors add origins at least three millimeters away. Filtering first prevents a weak overlapping proposal from displacing a surviving strict candidate. This improves our measured reference result, but broader proposals also introduce synthetic false positives. The model runs locally without downloads. Now, the live Explorer."""
    slides.append(s)

    s = base("From detection to inspection.", 5, 3, 75)
    s.text(90, 164, 1750, "From detection to inspection.", 78, PAPER, "display")
    s.text(94, 276, 1650, "Live, 75 seconds. Switch to the Explorer; poster shows historical footage.", 29, MUTED)
    s.image(368, 357, 1184, 596, "assets/explorer-poster.png")
    s.text(95, 949, 1700, "ORBIT  /  SELECT  /  CHECK CT  /  WALL MAP  /  EXPORT JSON", 23, MINT)
    s.notes = """[0:00–0:05] Switch to the preloaded Explorer using the fusion default. [0:05–0:18] Orbit the parent aorta: one CT and one parent mask produce inspectable branch candidates. [0:18–0:30] Select a branch and point to its origin, radius and direction. [0:30–0:46] Open linked CT views and inspect the candidate against the scan. [0:46–0:56] Show the wall map. [0:56–1:03] Briefly show the interior tour if the app is ready; skip it if behind. [1:03–1:10] Export challenge JSON. [1:10–1:15] Return to slide 6. These are predictions, not expert-confirmed anatomy. No video plays in this presentation."""
    slides.append(s)

    s = base("Three scans. Traceable evidence.", 6, 4, 25)
    s.text(90, 169, 1690, "Three scans. Traceable evidence.", 80, PAPER, "display")
    s.text(94, 286, 1680, "Parent outline in mint. Predicted ostia and 10 mm direction arrows in coral.", 28, MUTED)
    for index in range(3):
        x = 100 + index * 594
        s.image(x, 382, 520, 532, f"assets/check-{index + 1:03}.png")
        s.text(x, 337, 510, f"SUBJECT {index + 1:03}", 25, MINT)
    s.text(94, 947, 1750, "Native-grid coronal MIPs · selected fusion outputs · visual checks, not ground truth", 24, MUTED)
    s.notes = """These are three supplied scans, with the parent outline, predicted ostia and direction arrows. Each image links back to an exported result in the kit. They satisfy visual verification, not an accuracy score. We do not have complete expert daughter annotations for these scans, so a plausible overlay is not treated as ground truth."""
    slides.append(s)

    s = base("Measured progress. Bounded claims.", 7, 4, 30)
    s.text(90, 169, 1720, "Measured progress. Bounded claims.", 76, PAPER, "display")
    s.line(1004, 360, 1004, 895, LINE, 2)
    s.text(95, 344, 830, "RELEASED REFERENCES · 5 REUSED CASES", 24, MINT)
    s.text(95, 432, 890, f"F1 {reference['f1']:.3f}", 92, PAPER, "display")
    s.text(100, 575, 850, f"Fusion: {reference['true_positives']} TP / {reference['false_positives']} FP / {reference['false_negatives']} FN", 29)
    s.text(100, 650, 850, "19 AI-assisted targets; labels may be incomplete.", 26, MUTED)
    s.text(100, 719, 850, "Not independent hidden-test accuracy.", 25, CORAL)
    s.text(100, 799, 850, f"Precision {reference['precision']:.3f} · recall {reference['recall']:.3f} · count MAE {reference['count_mae']:g}", 25, MUTED)
    s.text(1080, 344, 740, "SYNTHETIC TOPOLOGY · 24 CASES", 24, MINT)
    s.text(1080, 432, 740, f"F1 {topology['f1']:.4f}", 92, PAPER, "display")
    s.text(1085, 575, 710, f"{topology['true_positives']} TP / {topology['false_positives']} FP / {topology['false_negatives']} FN", 30)
    s.text(1085, 650, 710, "4 negative-control false positives.", 29, CORAL)
    s.text(1085, 727, 710, "Procedural regression; not clinical evidence.", 24, MUTED)
    s.text(95, 900, 1740, "Local one-to-one matching at 3 mm. Real-reference agreement and synthetic topology are separate measures.", 24, MUTED)
    s.notes = """Fusion matches fourteen of nineteen targets, with four extras and five misses: F1 point seven five seven, precision seventy-eight percent, and average count error one point eight. These five reused, AI-assisted cases may omit branches. Separately, synthetic F1 is point eight seven eight five, with eleven false positives, including four negative-control detections. This is the tradeoff behind the higher reference score. Neither result establishes hidden-test or clinical accuracy."""
    slides.append(s)

    s = base("A branch you can inspect.", 8, 4, 20)
    s.text(90, 169, 1700, "A branch you can inspect.", 98, PAPER, "display")
    s.text(95, 333, 1670, "Local CPU inference. Offline after setup.", 38, MINT)
    s.text(100, 494, 810, "25 of 25 scans completed", 42, PAPER, "display")
    s.text(100, 574, 800, f"{verification['mean_end_to_end_s']:.2f} s mean · {verification['max_end_to_end_s']:.2f} s maximum\n{rss:.0f} MiB sampled RSS · four-core Linux", 28, MUTED)
    s.text(100, 674, 810, "Organizer-Windows timing remains pending.", 26, CORAL)
    s.text(1050, 494, 780, "Known limits", 42, PAPER, "display")
    s.text(1050, 574, 770, "Weak contrast · small vessels · complex junctions\nNext: complete expert labels + unseen patients.", 26, MUTED)
    s.text(100, 795, 1720, "Find every eligible origin. Keep each daughter separate.\nMake the result possible to verify.", 47, PAPER, "display")
    s.text(100, 943, 1740, "github.com/Coder-Meet/battleoftheschool   ·   prototype, not clinically validated", 23, MUTED)
    s.notes = """All twenty-five scans completed offline: eleven point six seconds average, fifty-two point seven maximum, and about fifteen hundred mebibytes sampled memory on four-core Linux. Organizer-Windows timing remains pending. Next: complete expert labels, fewer false positives, and unseen patients. Branch instances you can inspect. Thank you."""
    slides.append(s)
    return slides


def render(slide: Slide, output: Path) -> Image.Image:
    image = Image.new("RGB", (1920, 1080), "#" + slide.background)
    draw = ImageDraw.Draw(image)
    for e in slide.elements:
        box = (round(e.x), round(e.y), round(e.x + e.w), round(e.y + e.h))
        if e.kind == "text":
            font = ImageFont.truetype(str(output / f"assets/{e.font}.ttf"), round(e.size))
            for index, line in enumerate(e.text.splitlines()):
                if draw.textlength(line, font=font) > e.w + 2:
                    raise ValueError(f"Text exceeds width in {slide.title!r}: {line!r}")
                draw.text((e.x, e.y + index * e.size * 1.25), line, fill="#" + e.color, font=font, anchor="lt")
        elif e.kind == "rect":
            draw.rectangle(box, fill="#" + e.color)
        elif e.kind == "circle":
            draw.ellipse(box, fill=None if e.stroke else "#" + e.color,
                         outline="#" + e.color, width=max(1, round(e.stroke)))
        elif e.kind == "line":
            draw.line(box, fill="#" + e.color, width=max(1, round(e.stroke)))
        elif e.kind == "image":
            source = Image.open(output / e.source).convert("RGBA")
            source.thumbnail((round(e.w), round(e.h)), Image.Resampling.LANCZOS)
            image.paste(source, (round(e.x + (e.w - source.width) / 2),
                                 round(e.y + (e.h - source.height) / 2)), source)
        else:
            raise ValueError(e.kind)
    return image


def write_html(slides: list[Slide], output: Path):
    template = (ROOT / "presentation/player.html").read_text()
    serialized = json.dumps([asdict(slide) for slide in slides]).replace("<", "\\u003c")
    (output / "index.html").write_text(template.replace("__DECK_DATA__", serialized))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/fusion-presentation-kit")
    parser.add_argument("--predictions", type=Path, default=ROOT / "outputs/fusion-presentation-predictions")
    parser.add_argument("--poster", type=Path)
    parser.add_argument("--skip-evidence", action="store_true")
    args = parser.parse_args()
    for directory in ("assets", "slides", "evidence"):
        (args.output / directory).mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).with_name("favicon.svg"), args.output / "assets/favicon.svg")
    for name in ("README.md", "LIVE_DEMO_CUES.md"):
        shutil.copy2(Path(__file__).with_name(name), args.output / name)
    for source in (ROOT / "presentation/evidence/judge-fusion-verification.json",
                   ROOT / "docs/fusion-restored-validation.json"):
        shutil.copy2(source, args.output / "evidence" / source.name)
    fonts(args.output)
    if not args.skip_evidence:
        evidence(args.output, args.predictions)
    if args.poster:
        shutil.copy2(args.poster, args.output / "assets/explorer-poster.png")
    elif not (args.output / "assets/explorer-poster.png").exists():
        shutil.copy2(args.output / "assets/vessel.png", args.output / "assets/explorer-poster.png")
    slides = create_deck()
    assert sum(s.seconds for s in slides) == 300
    assert all(sum(s.seconds for s in slides if s.speaker == who) == 75 for who in range(1, 5))
    for slide in slides:
        for element in slide.elements:
            if element.kind != "line":
                assert 0 <= element.x <= element.x + element.w <= 1920, (slide.title, element)
                assert 0 <= element.y <= element.y + element.h <= 1080, (slide.title, element)
    images = []
    for i, slide in enumerate(slides):
        image = render(slide, args.output)
        image.save(args.output / f"slides/{i + 1:02}.png")
        images.append(image)
    images[0].save(args.output / "branchseed-slides.pdf", save_all=True, append_images=images[1:],
                   resolution=144, title="Branchseed — Find the branch. Keep the evidence.")
    contact = Image.new("RGB", (1920, 1080), "#" + INK)
    for index, image in enumerate(images):
        thumb = image.resize((640, 360), Image.Resampling.LANCZOS)
        contact.paste(thumb, ((index % 3) * 640, (index // 3) * 360))
    contact.save(args.output / "slide-overview.jpg", quality=95)
    (args.output / "deck.json").write_text(json.dumps([asdict(slide) for slide in slides], indent=2))
    write_html(slides, args.output)
    start = 0
    script = "# Branchseed: five minutes, four speakers\n\n"
    script += "The live Explorer demo runs INSIDE Speaker 3's 75 seconds; no video is played.\n\n"
    for index, slide in enumerate(slides):
        end = start + slide.seconds
        script += (f"## {start // 60}:{start % 60:02}–{end // 60}:{end % 60:02} · Speaker {slide.speaker}"
                   f" · Slide {index + 1}: {slide.title}\n\n{slide.notes}\n\n")
        start = end
    (args.output / "SPEAKER_SCRIPT.md").write_text(script)
    print(f"Built {len(slides)} slides; 300 seconds; four equal speaking slots at {args.output}")


if __name__ == "__main__":
    main()
