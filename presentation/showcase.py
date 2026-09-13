"""Render a standalone product edit from the original Explorer recording."""

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
FPS = 30
TRANSITION_FRAMES = 10
WIDTH, HEIGHT = 1920, 1080
VIEWPORT = (128, 248, 1664, 788)


@dataclass(frozen=True)
class Scene:
    title: str
    source_start: float
    seconds: int
    crop: tuple[int, int, int, int]


SCENES = (
    Scene("See the anatomy.", 20, 10, (244, 250, 1650, 780)),
    Scene("Check the CT evidence.", 64, 10, (244, 250, 1650, 780)),
    Scene("Map the origins.", 88, 8, (650, 340, 780, 370)),
    Scene("Explore from within.", 116, 13, (280, 285, 1330, 630)),
)


def run(arguments: list[str]) -> None:
    subprocess.run(arguments, check=True)


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def background(scene: Scene, index: int, fonts: Path, output: Path) -> None:
    y, x = np.mgrid[:HEIGHT, :WIDTH]
    glow = np.exp(-((x - 1400) ** 2 / 900**2 + (y - 850) ** 2 / 500**2))
    pixels = np.stack([10 + 5 * glow, 21 + 13 * glow, 27 + 13 * glow], axis=-1)
    canvas = Image.fromarray(pixels.astype(np.uint8))
    draw = ImageDraw.Draw(canvas)
    body = ImageFont.truetype(str(fonts / "body.ttf"), 28)
    title = ImageFont.truetype(str(fonts / "display.ttf"), 84)
    draw.text((128, 65), "BRANCHSEED  /  AORTA EXPLORER", font=body, fill="#7BE7C2")
    draw.text((1370, 65), "RECORDED PRODUCT TOUR", font=body, fill="#9FACAA")
    if draw.textlength(scene.title, font=title) > 1664:
        raise ValueError("Showcase title exceeds the safe width")
    draw.text((128, 115), scene.title, font=title, fill="#F1EDE3")
    for number in range(len(SCENES)):
        start = 128 + number * 422
        draw.rectangle((start, 220, start + 396, 222),
                       fill="#7BE7C2" if number == index else "#2A3A40")
    vx, vy, vw, vh = VIEWPORT
    draw.rectangle((vx - 2, vy - 2, vx + vw + 1, vy + vh + 1), fill="#34484D")
    canvas.save(output)


def soundtrack(output: Path, seconds: float) -> None:
    rate = 48000
    time = np.arange(round(seconds * rate)) / rate
    channels = []
    chords = (
        (146.832, 220.000, 293.665, 369.994),
        (123.471, 184.997, 293.665, 369.994),
        (97.999, 146.832, 246.942, 293.665),
        (110.000, 164.814, 277.183, 329.628),
    )
    for detune in (-0.12, 0.12):
        channel = np.zeros_like(time)
        for index, chord in enumerate(chords):
            local = time - index * 10
            envelope = np.clip(local / 1.5, 0, 1) * np.clip((12 - local) / 2, 0, 1)
            for frequency in chord:
                phase = 2 * np.pi * (frequency + detune) * time
                tone = np.sin(phase) + 0.12 * np.sin(2 * phase)
                channel += 0.016 * envelope * tone
        fade = np.clip(time / 0.7, 0, 1) * np.clip((seconds - time) / 1.4, 0, 1)
        channels.append(channel * fade)
    samples = np.column_stack(channels)
    assert np.isfinite(samples).all() and np.abs(samples).max() < 1
    wavfile.write(output, rate, (samples * 32767).astype(np.int16))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--kit", type=Path, default=ROOT / "outputs/live-presentation-kit")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/product-showcase")
    args = parser.parse_args()
    source = args.source.resolve()
    source_hash = sha256(source)
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(source),
    ]))
    video = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
    if (video["width"], video["height"]) != (WIDTH, HEIGHT):
        raise ValueError("The edit requires the original 1920×1080 Explorer source capture")
    if float(probe["format"]["duration"]) < max(s.source_start + s.seconds for s in SCENES):
        raise ValueError("The original source recording is too short for this edit")
    args.output.mkdir(parents=True, exist_ok=True)
    work = args.output / "render-work"
    work.mkdir(exist_ok=True)
    vx, vy, vw, vh = VIEWPORT
    clips = []
    for index, scene in enumerate(SCENES):
        frame = work / f"background-{index}.png"
        clip = work / f"scene-{index}.mp4"
        background(scene, index, args.kit / "assets", frame)
        cx, cy, cw, ch = scene.crop
        scene_graph = (
            f"[0:v]fps={FPS},crop={cw}:{ch}:{cx}:{cy},"
            f"scale={vw}:{vh}:flags=lanczos,setsar=1,format=yuv420p[ui];"
            f"[1:v]format=yuv420p[bg];[bg][ui]overlay={vx}:{vy}:shortest=1[v]"
        )
        run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(scene.source_start), "-t", str(scene.seconds), "-i", str(source),
            "-loop", "1", "-i", str(frame), "-filter_complex_threads", "1",
            "-filter_complex", scene_graph, "-map", "[v]", "-t", str(scene.seconds),
            "-r", str(FPS), "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-threads", "4", str(clip),
        ])
        clips.append(clip)
    frames = sum(s.seconds * FPS for s in SCENES) - TRANSITION_FRAMES * (len(SCENES) - 1)
    seconds = frames / FPS
    audio = work / "original-ambient.wav"
    soundtrack(audio, seconds)
    graph = [f"[{index}:v]settb=AVTB,setpts=PTS-STARTPTS[v{index}]" for index in range(len(clips))]
    elapsed = SCENES[0].seconds * FPS
    previous = "v0"
    for index in range(1, len(SCENES)):
        offset = (elapsed - TRANSITION_FRAMES) / FPS
        graph.append(
            f"[{previous}][v{index}]xfade=transition=fade:"
            f"duration={TRANSITION_FRAMES / FPS:.6f}:offset={offset:.6f}[x{index}]"
        )
        previous = f"x{index}"
        elapsed += SCENES[index].seconds * FPS - TRANSITION_FRAMES
    graph.append(f"[{previous}]fade=t=out:st={seconds - 0.3}:d=0.3:color=0x101C23,format=yuv420p[out]")
    graph.append(f"[{len(clips)}:a]loudnorm=I=-25:TP=-3:LRA=7[audio]")
    final = args.output / "branchseed-showcase.mp4"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for clip in clips:
        command.extend(["-i", str(clip)])
    run(command + [
        "-i", str(audio), "-filter_complex_threads", "1", "-filter_complex", ";".join(graph),
        "-map", "[out]", "-map", "[audio]", "-t", str(seconds), "-r", str(FPS),
        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-threads", "4",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart",
        "-metadata", "title=Branchseed — Aorta Explorer",
        "-metadata", "comment=Edited historical UI capture; original synthesized ambient soundtrack.",
        str(final),
    ])
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", "3",
        "-i", str(final), "-frames:v", "1", str(args.output / "showcase-poster.jpg"),
    ])
    if sha256(source) != source_hash:
        raise RuntimeError("Original source recording changed")
    report = {
        "source": str(source), "source_sha256": source_hash,
        "output_sha256": sha256(final), "duration_seconds": seconds, "frames": frames,
        "width": WIDTH, "height": HEIGHT, "fps": FPS, "source_speed": 1,
        "scenes": [asdict(scene) for scene in SCENES],
        "music": "Original deterministic sine-pad composition; no third-party samples or voice.",
        "context": "Historical interface demonstration, not current accuracy or runtime evidence.",
    }
    (args.output / "edit-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Rendered {seconds:.2f}s product showcase at {final}; original source preserved")


if __name__ == "__main__":
    main()
