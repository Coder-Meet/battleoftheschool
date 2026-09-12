#!/usr/bin/env python3
"""Build the offline presentation from actual predictions and local media."""

import argparse
import json
import shutil
import subprocess
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
    slides: list[Slide] = []
    s = base("Find the branch. Keep the evidence.", 1, 1, 25)
    s.text(90, 190, 975, "Find the branch.\nKeep the evidence.", 105, PAPER, "display")
    s.text(95, 504, 860, "From a CT and one parent mask\nto inspectable daughter instances.", 36, MUTED)
    s.text(95, 750, 800, "PHYSICAL COORDINATES.  LOCAL COMPUTE.", 23, MINT)
    s.text(95, 820, 860, "A focused prototype for a difficult anatomical task.", 27)
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
    s.text(95, 815, 890, "70% of the score is discovery + ostium location.", 30, MINT)
    s.text(95, 878, 910, "We concentrated effort where the challenge puts the weight.", 25, MUTED)
    s.text(1450, 506, 355, "Common trunk\n1 aortic origin", 25, CORAL)
    s.text(1135, 900, 650, "Schematic · not patient anatomy", 21, MUTED)
    s.notes = """The parent mask is a search anchor; it does not label the daughters. Bright structures alone are not enough. Two nearby openings at the wall are two instances. A common trunk has one direct origin even when it divides. Distal children and acquisition crop caps must be excluded. We also cannot guess vessels outside the visible scan. Discovery and origin localization account for seventy percent of the challenge score, so our design concentrates there. It searches locally, checks connection and topology, and preserves physical coordinates. That focus is why we built a geometric detector before claiming a large learned model. Speaker two will show how the measurements work."""
    slides.append(s)

    s = base("Anchor. Trace. Measure.", 3, 2, 40, True)
    s.text(90, 176, 1600, "Anchor. Trace. Measure.", 94, INK, "display")
    s.text(94, 320, 1550, "A local geometric pipeline, with explicit decisions at every step.", 32, "526367")
    labels = [("01", "Anchor", "Validate the grid.\nSearch around the parent."),
              ("02", "Discover", "Enhance local vessels.\nFind supported wall contacts."),
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
    s.text(95, 880, 1620, "CT + binary aorta mask     /     one JSON file     /     visual verification", 28, "1A7465")
    s.notes = """There are four stages. First, validate the image and mask geometry and create a local search region around the aorta. Second, enhance vascular structure and find supported contacts at the wall. Third, trace the proximal lumen while checking topology, stopping at ten millimeters or the first downstream bifurcation. Finally, measure the origin, the five-millimeter seed, local radius and initial unit direction. The result is a machine-readable JSON file. Everything here uses local image processing on CPU. The optional classifier can rank reviewed candidates, but the demonstrated detector does not require trained weights or an online service."""
    slides.append(s)

    s = base("Small geometry. Big consequences.", 4, 2, 35, True)
    s.text(90, 171, 1750, "Small geometry. Big consequences.", 79, INK, "display")
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
    s.text(1050, 410, 770, "Keep small openings", 40, INK, "display")
    s.text(1050, 479, 770, "Contact threshold scales with physical size.", 27, "526367")
    s.text(1050, 580, 770, "Find the wall crossing", 40, INK, "display")
    s.text(1050, 649, 770, "Use the proximal tangent to localize the ostium.", 27, "526367")
    s.text(1050, 750, 770, "Measure across the lumen", 40, INK, "display")
    s.text(1050, 819, 770, "Sample perpendicular rays; keep a fallback.", 27, "526367")
    s.notes = """Our latest changes target three concrete errors. A physical-size contact threshold retains small openings that a fixed voxel-volume cutoff rejected. A proximal tangent improves the estimated crossing of the aortic wall, especially for oblique branches. At the five-millimeter seed, perpendicular intensity samples estimate the lumen radius, with a distance-transform fallback when the boundary cannot be measured. These are general geometry changes, not case-specific edits. We tested them against analytic phantoms and negative controls. Speaker three will show how a reviewer can inspect what the algorithm actually produced."""
    slides.append(s)

    s = base("From detection to inspection.", 5, 3, 75)
    s.text(90, 164, 1750, "From detection to inspection.", 78, PAPER, "display")
    s.text(94, 276, 1650, "A 60-second film of the real local Explorer. Press V to play.", 29, MUTED)
    s.image(368, 357, 1184, 596, "assets/explorer-poster.png")
    s.notes = """[0:00–0:06] Now follow a real local result through the Explorer. Start the sixty-second film.
[During film, speak naturally; the film has no narration.] The 3D view gives us spatial context. Selecting a branch links its origin, direction and radius. We can return to CT rather than trust a mesh alone. The wall map gives a second way to inspect where openings lie. The interior tour adds another view of the supplied aorta. These are predicted candidates, not anatomical labels or confirmed truth. The result remains exportable as the challenge JSON.
[After film, by 1:15] The interface makes a result inspectable; expert annotation is still needed to establish accuracy. Speaker four will separate our evidence from what we have not yet proved."""
    slides.append(s)

    s = base("Three scans. Traceable evidence.", 6, 4, 25)
    s.text(90, 169, 1690, "Three scans. Traceable evidence.", 80, PAPER, "display")
    s.text(94, 286, 1680, "Parent outline in mint. Predicted ostia and 10 mm direction arrows in coral.", 28, MUTED)
    for index in range(3):
        x = 100 + index * 594
        s.image(x, 382, 520, 532, f"assets/check-{index + 1:03}.png")
        s.text(x, 337, 510, f"SUBJECT {index + 1:03}", 25, MINT)
    s.text(94, 947, 1750, "Native-grid coronal MIPs · current detector outputs · no expert daughter labels available", 24, MUTED)
    s.notes = """These are three supplied scans, with the parent outline, predicted ostia and direction arrows. Each image links back to an exported result in the kit. They satisfy visual verification, not an accuracy score. We do not have complete expert daughter annotations for these scans, so a plausible overlay is not treated as ground truth."""
    slides.append(s)

    s = base("Measured progress. Bounded claims.", 7, 4, 30)
    s.text(90, 169, 1720, "Measured progress. Bounded claims.", 76, PAPER, "display")
    s.line(1004, 360, 1004, 895, LINE, 2)
    s.text(95, 344, 830, "SYNTHETIC DEVELOPMENT · 5 CASES", 24, MINT)
    s.text(95, 432, 890, "8/10 to 10/10", 92, PAPER, "display")
    s.text(100, 575, 850, "Direct daughters recovered · 0 false positives", 29)
    s.text(100, 650, 850, "Additional seed: 9/10 to 10/10, with 0 FP.", 26, MUTED)
    s.text(100, 719, 850, "Same procedural families; not clinical validation.", 25, CORAL)
    s.text(1080, 344, 740, "REAL-SCAN EXECUTION · 25 CASES", 24, MINT)
    s.text(1080, 432, 740, "25/25", 98, PAPER, "display")
    s.text(1085, 575, 710, "Scans completed", 30)
    s.text(1085, 650, 710, "1.0–20.4 s / case", 44, PAPER, "display")
    s.text(1085, 727, 710, "Development machine · numerical threads = 4", 24, MUTED)
    s.text(95, 900, 1740, "Local scoring: one-to-one ostium matching at 3 mm. Real-scan detection accuracy remains unmeasured.", 24, MUTED)
    s.notes = """On five synthetic development cases, recovered daughters improved from eight to ten, with zero false positives. Additional variants improved from nine to ten. These simplified families are not clinical validation. Separately, all twenty-five real scans completed in about one to twenty seconds per case, with four numerical threads. That measures execution on our machine, not real detection accuracy or guaranteed organizer performance."""
    slides.append(s)

    s = base("A branch you can inspect.", 8, 4, 20)
    s.text(90, 169, 1700, "A branch you can inspect.", 98, PAPER, "display")
    s.text(95, 333, 1670, "No GPU. No cloud inference. Prepared dependencies required.", 35, MINT)
    s.text(100, 494, 810, "Built for the constraints", 42, PAPER, "display")
    s.text(100, 574, 800, "Target: 4 CPU cores · 8 GB · offline\n50 tests passed with networking denied.", 28, MUTED)
    s.text(1050, 494, 780, "Known limits", 42, PAPER, "display")
    s.text(1050, 574, 770, "Weak contrast · very small vessels · complex junctions\nNext: expert labels + a frozen patient-level test split.", 26, MUTED)
    s.text(100, 795, 1720, "Find every eligible origin. Keep each daughter separate.\nMake the result possible to verify.", 47, PAPER, "display")
    s.text(100, 943, 1740, "github.com/Coder-Meet/battleoftheschool   ·   prototype, not clinically validated", 23, MUTED)
    s.notes = """We target four cores, eight gigabytes and offline execution after setup. Fifty tests passed with networking denied. Weak contrast, small vessels and complex junctions remain risks. Next: expert labels and a frozen patient-level test split. We built branch instances you can inspect and verify. Thank you."""
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


def reel_plates(output: Path):
    specs = [
        ("Find the branch.\nKeep the evidence.", "AORTA EXPLORER", "Real scans. Physical geometry. Local compute."),
        ("CT + parent mask.\nInspectable branch instances.", "THE METHOD", "Anchor / discover / trace / measure"),
        ("See the origin in 3D.", "ACTUAL EXPLORER CAPTURE", "Subject001 · parent surface and predicted branches"),
        ("Go back to the CT.", "ACTUAL EXPLORER CAPTURE", "Predictions stay connected to the image evidence."),
        ("Map the aortic wall.", "ACTUAL EXPLORER CAPTURE", "A second view of where candidate origins lie."),
        ("Explore from the inside.", "ACTUAL EXPLORER CAPTURE", "A local tour through the supplied parent aorta."),
        ("25 real scans processed.", "EXECUTION EVIDENCE", "1.0–20.4 s/case on our development machine · not an accuracy score"),
        ("No GPU.\nNo cloud inference.", "BRANCHSEED", "Prepared dependencies required. Expert validation comes next."),
    ]
    for index, (title, tag, detail) in enumerate(specs):
        s = Slide(title, 0, 0, INK, [])
        s.text(90, 64, 1750, tag, 25, MINT)
        if 2 <= index <= 5:
            s.text(90, 114, 1750, title, 55, PAPER, "display")
            s.rect(237, 187, 1446, 816, LINE)
            s.text(240, 1027, 1500, detail, 23, MUTED)
        else:
            s.text(94, 290, 1600, title, 112, PAPER, "display")
            s.text(99, 815, 1740, detail, 29, MUTED)
            s.line(99, 961, 1820, 961, MINT, 4)
            if index in (0, 7):
                s.image(1390, 205, 410, 680, "assets/vessel.png")
        render(s, output).save(output / f"assets/reel-{index}.png")


def movie(output: Path, shots: list[str]):
    if len(shots) != 4:
        raise ValueError("Supply four video paths with start seconds: PATH@SECONDS")
    work = output / "render-work"
    work.mkdir(exist_ok=True)
    durations = [6, 6, 8, 8, 8, 8, 8, 8]
    for index, seconds in enumerate(durations):
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", "4",
                   "-loop", "1", "-framerate", "30", "-i", str(output / f"assets/reel-{index}.png")]
        if 2 <= index <= 5:
            filename, start = shots[index - 2].rsplit("@", 1)
            probe = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                             "-of", "default=noprint_wrappers=1:nokey=1", filename], text=True)
            if float(start) < 0 or float(start) + seconds > float(probe):
                raise ValueError(f"Shot extends beyond source: {shots[index - 2]}")
            command += ["-ss", start, "-i", filename, "-filter_complex_threads", "1",
                        "-filter_complex", "[1:v]scale=1440:810:force_original_aspect_ratio=decrease,"
                        "pad=1440:810:(ow-iw)/2:(oh-ih)/2:color=0x101C23,setsar=1[clip];"
                        "[0:v][clip]overlay=240:190,fade=t=in:st=0:d=0.25,"
                        f"fade=t=out:st={seconds - 0.25}:d=0.25[v]", "-map", "[v]"]
        else:
            frames = seconds * 30
            command += ["-vf", f"zoompan=z='1+0.02*on/{frames}':"
                        "x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2':d=1:s=1920x1080:fps=30,"
                        f"fade=t=in:st=0:d=0.25,fade=t=out:st={seconds - 0.25}:d=0.25"]
        command += ["-an", "-t", str(seconds), "-r", "30", "-c:v", "libx264", "-preset", "fast",
                    "-crf", "19", "-pix_fmt", "yuv420p", "-threads", "4", str(work / f"clip-{index}.mp4")]
        subprocess.run(command, check=True)
        print(f"Rendered reel scene {index + 1}/8", flush=True)
    (work / "concat.txt").write_text("".join(f"file 'clip-{i}.mp4'\n" for i in range(8)))
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
                    "-safe", "0", "-i", str(work / "concat.txt"), "-c", "copy",
                    "-movflags", "+faststart", str(output / "branchseed-film.mp4")], check=True)
    (output / "evidence/footage-provenance.json").write_text(json.dumps({
        "shots": shots, "durations_s": durations, "notes": "Actual Explorer capture; cuts only, no speed changes.",
        "audio": "Silent by design for four-speaker live narration; no stock footage or synthetic patient imagery."
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/presentation-kit")
    parser.add_argument("--predictions", type=Path, default=ROOT / "outputs/accuracy-real-smoke")
    parser.add_argument("--poster", type=Path)
    parser.add_argument("--shot", action="append", default=[])
    parser.add_argument("--skip-evidence", action="store_true")
    args = parser.parse_args()
    for directory in ("assets", "slides", "evidence"):
        (args.output / directory).mkdir(parents=True, exist_ok=True)
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
    reel_plates(args.output)
    if args.shot:
        movie(args.output, args.shot)
    start = 0
    script = "# Branchseed: five minutes, four speakers\n\n"
    script += "The 60-second film is INCLUDED in Speaker 3's 75 seconds. It is silent for live narration.\n\n"
    for index, slide in enumerate(slides):
        end = start + slide.seconds
        script += (f"## {start // 60}:{start % 60:02}–{end // 60}:{end % 60:02} · Speaker {slide.speaker}"
                   f" · Slide {index + 1}: {slide.title}\n\n{slide.notes}\n\n")
        start = end
    (args.output / "SPEAKER_SCRIPT.md").write_text(script)
    print(f"Built {len(slides)} slides; 300 seconds; four equal speaking slots at {args.output}")


if __name__ == "__main__":
    main()
