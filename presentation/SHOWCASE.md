# Standalone product showcase

The current deck uses a **live demo**. The showcase is a separate sharing
asset for Drive, social posts or a product page.

## Delivered files

- `branchseed-showcase.mp4`: 40-second, 1920×1080, 30 fps H.264/AAC MP4,
  with fast-start metadata for web playback.
- `showcase-poster.jpg`: a frame suitable for a cover image.
- `edit-report.json`: exact source intervals, crops, hashes and encoding.
- The original `branchseed-film.mp4` remains unchanged on the final release.

The edit removes the old opening title/method cards and ending metric/offline
cards. It starts immediately with the product and finishes on the interior
view. It uses the higher-quality original capture instead of enlarging the
small app window already embedded in the original film.

## Edit decisions

| Shot | Original capture interval | Message |
|---|---|---|
| 3D orbit | 20–30 s | See the anatomy. |
| Linked CT planes | 64–74 s | Check the CT evidence. |
| Wall map close-up | 88–96 s | Map the origins. |
| Interior tour | 116–129 s | Explore from within. |

Ten-frame dissolves join the shots; total duration is exactly 40 seconds.
The footage runs at its original speed. Static framing crops focus attention
on the relevant workspace; they do not alter the displayed prediction data.
The 30 fps output repeats frames from the original 15 fps capture.

Branding uses the deck's local Fraunces/Manrope fonts. Captions, progress marks
and dark framing are consistent across scenes. The soft ambient bed is an
original synthesized chord composition, with no downloaded music, samples,
voice clone or third-party recording. It fades gently and is normalized to
approximately -25 LUFS.

This is an edited historical interface recording, not a new inference run.
Do not use the footage's old instance counts or displayed timing as current
fusion performance. Current results live in the deck and evaluation reports.

## Rebuild

Prepare the existing presentation kit first so its licensed fonts are present.
The source is the preserved full-resolution `source-footage.mp4` from the
original presentation kit, not the 60-second titled film.

```bash
.venv313/bin/python presentation/showcase.py \
  --source outputs/presentation-kit/source-footage.mp4
```

Output: `outputs/product-showcase/`. FFmpeg/ffprobe must be installed.
The script uses the existing NumPy, SciPy and Pillow dependencies; it does not
add runtime dependencies to the detector. It checks the recording dimensions
and duration before editing and verifies the source hash afterward.

The original capture is included in the historical
[presentation kit](https://github.com/Coder-Meet/battleoftheschool/releases/download/branchseed-presentation-2026-09-12/branchseed-presentation-kit.zip).
Keep both the capture and original film; publish the newly named showcase.
