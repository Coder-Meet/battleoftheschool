# Branchseed presentation studio

An eight-slide, five-minute presentation with four equal speaking slots and a
60-second film of the actual Explorer. The visual system combines warm paper,
deep ink, mint geometry and coral origins, with Fraunces display type and the
Explorer's Manrope body type. Generated media stays outside Git.

## Start with the delivered kit

Extract the entire ZIP before opening `index.html`. Keep `assets/` and
`branchseed-film.mp4` beside it. The HTML uses local fonts and media; no account,
CDN, analytics or internet connection is required. It is the recommended live
presentation format.

| Control | Action |
|---|---|
| Left / Right, Page Up / Down, Space | Change slide |
| Home / End | First / last slide |
| F | Fullscreen |
| N | Speaker notes |
| V, on slide 5 | Play the film |
| A | Start, pause or resume timed rehearsal |
| Escape | Close film or notes |

The timed rehearsal includes the film inside Speaker 3's 75 seconds. The film
starts six seconds into that slot and lasts exactly sixty seconds. Pausing the
rehearsal also pauses the film. Native video controls are available for manual
playback; timed playback uses the shared rehearsal controls.

The film is intentionally silent for live team narration. It is not a narrated
five-minute standalone submission. Use the script while presenting or record
the four teammates over the deck if the organizers request an uploaded talk.
No additional presentation file-size, codec, speaker or upload constraints
were found in the supplied challenge brief.

## Four-person running order

Names were not supplied, so the editable deck uses Speaker 1–4.

| Time | Speaker / role | Slides | Responsibility |
|---|---|---|---|
| 0:00–1:15 | 1 — problem lead | 1–2 | Opening, challenge, topology, scoring priority |
| 1:15–2:30 | 2 — algorithm lead | 3–4 | Pipeline, physical geometry, recent improvements |
| 2:30–3:45 | 3 — demo lead | 5 | Run and narrate the 60-second Explorer film |
| 3:45–5:00 | 4 — validation lead | 6–8 | Three cases, evidence, runtime, limits, close |

Speaker 3 operates the laptop throughout to avoid switching drivers. Speaker 1
owns the opening and first handoff; Speaker 2 owns the method questions;
Speaker 4 owns accuracy/runtime questions. Everyone should learn the final
sentence in case the clock forces an early close.

`SPEAKER_SCRIPT.md` in the kit contains the complete timed script. The same
notes are in PowerPoint and the HTML notes dialog. Slide durations are
25, 50, 40, 35, 75, 25, 30 and 20 seconds, totaling exactly 300 seconds.

## PowerPoint, PDF and media

`branchseed-editable.pptx` contains editable text and geometric shapes,
speaker notes, CT images and the embedded film. Click the film to play it in
PowerPoint slideshow mode; HTML keyboard shortcuts do not apply. Install `assets/body.ttf`
and `assets/display.ttf` before opening PowerPoint; they are static instances
named **Branchseed Text** and **Branchseed Display**. Restart PowerPoint after
installing fonts. Their original OFL licenses are included. The HTML does
not need font installation.

`branchseed-slides.pdf` is the consistent visual fallback. It has eight pages
and cannot play video. If presenting from PDF, open the MP4 separately during
slide 5 and return to the next slide afterward. Keep the whole demonstration
within five minutes.

`source-footage.mp4` in the kit is a real-time, unannotated recording of the
Explorer, useful for recutting. The reel uses cuts at 22, 63, 85 and 119 seconds
for 3D, CT, wall map and tour. Each excerpt plays for eight seconds at original
speed. Each scene's 0.25-second fade-in and fade-out are included in its
allocated duration.

## Build from this repository

Use the repository's verified Python environment and frontend dependencies.
The extra font and PowerPoint packages are development-only; they do not
affect inference.

```bash
npm --prefix presentation ci
uv pip install --python .venv313/bin/python -r presentation/requirements.txt

# Generate current detector results if this output directory is absent.
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  .venv313/bin/python batch.py --data-root data --output-dir outputs/accuracy-real-smoke

# Use the real screenshot and raw capture from the delivered kit.
.venv313/bin/python presentation/build.py \
  --poster /path/to/explorer-screenshot.png \
  --shot /path/to/source-footage.mp4@22 \
  --shot /path/to/source-footage.mp4@63 \
  --shot /path/to/source-footage.mp4@85 \
  --shot /path/to/source-footage.mp4@119
node presentation/build-pptx.cjs
```

Outputs default to `outputs/presentation-kit`. To change copy, timing or composition,
edit `create_deck()` in `build.py`; to change controls, edit `player.html`.
All formats share the same layout data. Supply `--skip-evidence` when only
updating text or controls, and omit the four `--shot` arguments to retain the
existing film. Re-run the PowerPoint builder after editing the deck or film.

The script needs local CTs, current predictions, FFmpeg/FFprobe, Pillow and
FontTools (the latter two are installed with the project's Matplotlib stack).
It does not fabricate an Explorer screenshot: without `--poster`, an existing
poster is retained or the actual parent-mask rendering is used temporarily.
Do not deliver that temporary poster as product footage.

## Evidence and wording

The deck's metrics are frozen to the algorithm improvement report from
`29c843e`; product capture uses `4d59d12`, with the same detector and a formatted
Vite configuration. The kit contains the supporting reports.

- Development seed 2026: five synthetic cases, ten direct daughters.
  Baseline 8 TP / 0 FP / 2 FN; updated 10 TP / 0 FP / 0 FN.
- Additional seed 7183: the same five procedural families, with changed
  geometry/noise/rotation. Baseline 9 TP / 0 FP / 1 FN; updated 10 TP / 0 FP / 0 FN.
- Local matching is one-to-one at 3 mm. It is not a reproduction of hidden
  organizer scoring. Both results remain synthetic.
- All 25 supplied scans completed; measured range 0.998–20.385 seconds on the
  development machine with four numerical-library threads. This is not a
  dedicated four-core/8 GB benchmark or a guarantee on the organizer's machine.
- 50 Python tests passed with network system calls denied. Dependencies,
  scans, models if explicitly selected, and frontend assets must be prepared
  before disconnection.
- The three CT visuals use the supplied scans, parent outlines and current
  predicted origins/directions. They are native-grid projections, not expert
  annotations or proof of clinical accuracy.

Do not say “100% accurate,” “clinically validated,” “state of the art,”
“trained on 25 scans,” or “guaranteed to win.” The core demonstrated detector
is classical image processing. The optional classifier needs human-reviewed
positive/negative candidates and cannot recover a branch never proposed.

## Rehearsal and recovery

1. Copy the extracted kit and the repository to the presentation laptop.
2. Disconnect external internet and open the HTML. Start the video once.
3. Assign the four names in the notes or regenerate the deck with names.
4. Rehearse once at full pace, then once with the speaker handoffs. Keep the
   film within Speaker 3's slot; never add sixty seconds after the five-minute talk.
5. Keep PDF and MP4 open as a fallback. If playback fails, Speaker 3 uses the
   poster and explains linked 3D, CT, wall-map and export interactions.
6. If showing the live Explorer instead, preload a case and avoid promising a
   particular candidate count. Subject018 currently has 13 candidates.

Likely questions:

| Question | Grounded answer |
|---|---|
| Is this a learned model? | The demonstrated detector is a CPU geometric pipeline. An optional reviewed-candidate classifier exists but is not required. |
| Why only five labeled cases? | They are analytic training phantoms with labels defined before image generation. Real annotation is the next step. |
| Why is a common trunk one instance? | The instance is defined at the aortic wall, before downstream splitting. |
| What fails? | Weak contrast, very small vessels, junction ambiguity and geometry/topology errors remain concerns. A candidate classifier cannot recover missed proposals. |
| What proves real accuracy? | Independent expert references, a frozen patient-level split and comparison against strong baselines. We do not have that evidence yet. |
| Will it run on the final machine? | Inference is local and CPU-only. The brief targets four cores/8 GB; measured development times are separate from final-hardware validation. |

## Skills research and source requirements

Reviewed on 2026-09-12:

- [Official Branchseed challenge](https://docs.google.com/document/d/1oRb2R9pauvsC-9hDIfr23ojLx90JpCt0jVZjCD5l5Cg/edit):
  “a five-minute demonstration explaining your method, its runtime and known
  failure cases”; visual checks on at least three cases. Hidden environment:
  four CPU cores, 8 GB RAM, no GPU, no internet. The initial average target is
  no more than 60 seconds/case; the final limit is to be confirmed.
- [Frontend Slides](https://github.com/zarazhangrui/frontend-slides):
  fixed 1920×1080 stage, distinctive typography, local assets, keyboard
  navigation, reduced motion and print fallback. We inferred a speaker-led
  deck from the challenge instead of adding a style-selection approval round.
- [Anthropic PPTX skill](https://github.com/anthropics/skills/tree/main/skills/pptx):
  editable shapes/text, explicit layout, independent option objects and
  rendered slide validation.
- [Remotion skills](https://github.com/remotion-dev/skills):
  frame-driven motion, one focal point per video scene, safe margins, local
  assets and still-frame validation. Remotion is a strong future option for
  complex React motion scenes; this film uses FFmpeg with deterministic
  durations and original-speed footage, avoiding another browser-render stack.

No personal plugins were installed. Local plugin search found no suitable
presentation skill. External skill guidance informed the authoring; this kit
does not pretend the FFmpeg renderer is a Remotion project.

## Authoring checks

```bash
.venv313/bin/ruff check presentation/build.py
.venv313/bin/mypy presentation/build.py
npm --prefix presentation run lint
node presentation/build-pptx.cjs
ffprobe -v error -show_entries format=duration:stream=codec_name,width,height \
  -of json outputs/presentation-kit/branchseed-film.mp4
```

`build.py` checks stage bounds, font text widths and speaker timing.
The PowerPoint generator permits PNG inputs only. The locked PptxGenJS
dependency has upstream `image-size` parser advisories for other image formats;
the build uses generated PNGs and does not process uploaded/untrusted images.
No audit settings or repository security controls were disabled.

The exporter sets the embedded video's generated cover relationship to the
local poster file before writing the package. This adapter targets the pinned
PptxGenJS 4.0.1 data model and checks its expected media type. Revalidate the
video cover and embedded MP4 if upgrading that dependency.
