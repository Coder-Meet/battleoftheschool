# Branchseed presentation studio

An eight-slide, five-minute presentation with four equal speaking slots and a
75-second live demo of the actual Explorer, driven from a real browser window
alongside the deck. The visual system combines warm paper, deep ink, mint
geometry and coral origins, with Fraunces display type and the Explorer's
Manrope body type. Generated media stays outside Git.

## Start with the delivered kit

Published downloads: [current offline kit](https://github.com/Coder-Meet/battleoftheschool/releases/download/branchseed-final-2026-09-13/branchseed-final-presentation.zip)
and [separate product showcase](https://github.com/Coder-Meet/battleoftheschool/releases/download/branchseed-final-2026-09-13/branchseed-showcase.mp4).
The unchanged [original 60-second film](https://github.com/Coder-Meet/battleoftheschool/releases/download/branchseed-final-2026-09-13/branchseed-film.mp4)
is preserved separately. Neither video is included in or played by the current
presentation. Upload the showcase to Drive or other destinations as needed.

Extract the entire ZIP before opening `index.html`. Keep `assets/`
beside it. The HTML uses local fonts and media; no account,
CDN, analytics or internet connection is required. It is the recommended live
presentation format.

| Control | Action |
|---|---|
| Left / Right, Page Up / Down, Space | Change slide |
| Home / End | First / last slide |
| F | Fullscreen |
| N | Speaker notes |
| A | Start, pause or resume timed rehearsal |
| Escape | Close notes |

The timed rehearsal covers all eight slides, including Speaker 3's 75-second
slot, but it does not try to time or control the live demo itself — that runs
in a separate window under the presenter's own hands. Pausing the rehearsal
simply pauses the on-screen clock.

Slide 5 is a cue card, not the demo: it stays on screen only until the
presenter switches to the live Explorer, and it's what the audience sees again
when they switch back. See [LIVE_DEMO_CUES.md](LIVE_DEMO_CUES.md) for the
second-by-second run of show and suggested narration.

## Four-person running order

Names were not supplied, so the editable deck uses Speaker 1–4.

| Time | Speaker / role | Slides | Responsibility |
|---|---|---|---|
| 0:00–1:15 | 1 — problem lead | 1–2 | Opening, challenge, topology, scoring priority |
| 1:15–2:30 | 2 — algorithm lead | 3–4 | Pipeline, physical geometry, recent improvements |
| 2:30–3:45 | 3 — demo lead | 5 | Run and narrate the live Explorer demo |
| 3:45–5:00 | 4 — validation lead | 6–8 | Three cases, evidence, runtime, limits, close |

Speaker 3 operates the laptop and the Explorer window throughout, to avoid
switching drivers mid-demo. Speaker 1
owns the opening and first handoff; Speaker 2 owns the method questions;
Speaker 4 owns accuracy/runtime questions. Everyone should learn the final
sentence in case the clock forces an early close.

`SPEAKER_SCRIPT.md` in the kit contains the complete timed script. The same
notes are in PowerPoint and the HTML notes dialog. Slide durations are
25, 50, 40, 35, 75, 25, 30 and 20 seconds, totaling exactly 300 seconds.

## PowerPoint, PDF and media

`branchseed-editable.pptx` contains editable text and geometric shapes,
speaker notes and CT images. Install `assets/body.ttf` and `assets/display.ttf`
before opening PowerPoint; they are static instances named **Branchseed Text**
and **Branchseed Display**. Restart PowerPoint after installing fonts. Their
original OFL licenses are included. The HTML does not need font installation.

`branchseed-slides.pdf` is the consistent visual fallback. It has eight pages.
If presenting from PDF or PowerPoint, alt-tab to the live Explorer at slide 5
exactly as you would from the HTML deck, then return to the next slide
afterward. Keep the whole demonstration within five minutes.

## Build from this repository

Use the repository's verified Python environment and frontend dependencies.
The extra font and PowerPoint packages are development-only; they do not
affect inference.

```bash
npm --prefix presentation ci
uv pip install --python .venv313/bin/python -r presentation/requirements.txt

# Generate strict detector results if this output directory is absent.
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  .venv313/bin/python batch.py --data-root data --output-dir outputs/accuracy-real-smoke

# Use a real Explorer screenshot from the delivered kit as slide 5's cue card.
.venv313/bin/python presentation/build.py \
  --predictions outputs/accuracy-real-smoke \
  --poster /path/to/explorer-screenshot.png
node presentation/build-pptx.cjs
```

Outputs default to `outputs/live-presentation-kit`. To change copy, timing or composition,
edit `create_deck()` in `build.py`; to change controls, edit `player.html`.
All formats share the same layout data. Supply `--skip-evidence` when only
updating text or controls. Re-run the PowerPoint builder after editing the deck.

The script needs local CTs, current predictions, Pillow and FontTools (the
latter two are installed with the project's Matplotlib stack). It does not
fabricate an Explorer screenshot: without `--poster`, an existing poster is
retained, or the actual parent-mask rendering is used temporarily as a
placeholder cue card.

## Evidence and wording

The deck reflects the accepted **deterministic strict** configuration:
1 mm working grid, native contrast scale 1.2, no trained weights.

- Five reused cases, 19 judge-approved AI-assisted/non-exhaustive targets:
  8 TP / 3 FP / 11 FN; precision 0.7273, recall 0.4211, F1 0.5333,
  daughter-count MAE 2.0. This is development-reference agreement.
- Frozen topology, 24 synthetic cases: 46 TP / 0 FP / 3 FN; F1 0.9684.
  Report this separately from real-reference results.
- Guarded curved-wall recovery: local F1 0.5806, but count MAE rises to 2.2.
  No added errors across 80 synthetic comparisons; it remains experimental.
- Strict ran all 25 scans with four-core Linux affinity: maximum 23.21 seconds
  and 1481 MiB sampled peak RSS. Native organizer-Windows timing is pending.
- The three CT visuals use released strict outputs for subjects001–003.
  Their overlays are visual checks, not complete annotations.

Sources: [final evaluation](../FINAL_EVALUATION_RESULTS.md),
[accuracy recheck](../ACCURACY_RECHECK.md) and
[strict resource receipt](../labels/finalization/final-batch-resource.json).
The old [robustness supplement](ROBUSTNESS_UPDATE.md) and original film are
historical; use the current deck's notes rather than old replacement scripts.

Do not say “100% accurate,” “clinically validated,” “state of the art,”
“trained on 25 scans,” or “guaranteed to win.” The core demonstrated detector
is classical image processing. The optional classifier needs human-reviewed
positive/negative candidates and cannot recover a branch never proposed.

## Rehearsal and recovery

1. Copy the extracted kit, the repository, and a working Python/Explorer
   environment to the presentation laptop.
2. Disconnect external internet, open the HTML, and separately start the
   Explorer (`python explorer.py --port 8000`) with a case preloaded in its
   own window or tab.
3. Assign the four names in the notes or regenerate the deck with names.
4. Rehearse once at full pace, then once with the speaker handoffs, including
   the actual window switch at slide 5. Keep the live segment within
   Speaker 3's 75-second slot; never add time after the five-minute talk.
5. Keep the PDF open as a fallback. If the live Explorer fails to launch or
   hangs mid-demo, Speaker 3 stays on slide 5's poster and narrates the linked
   3D, CT, wall-map and tour interactions from memory instead.
6. Use the strict default, preload subject018 and keep subject001 as backup.
   Do not promise a fixed daughter count as anatomical truth. Export JSON
   before returning to slide 6; skip the optional tour if behind.

Likely questions:

| Question | Grounded answer |
|---|---|
| Is this a learned model? | The demonstrated detector is a CPU geometric pipeline. An optional reviewed-candidate classifier exists but is not required. |
| Why only five reference cases? | The released package has 19 AI-assisted targets approved by the judge. They may be incomplete and have been reused for development. |
| Why is a common trunk one instance? | The instance is defined at the aortic wall, before downstream splitting. |
| What fails? | Weak contrast, very small vessels, junction ambiguity and geometry/topology errors remain concerns. A candidate classifier cannot recover missed proposals. |
| Why not the higher-F1 recovery? | Its local daughter-count MAE worsens. Synthetic checks do not establish a gain on unseen patients. |
| What proves unseen accuracy? | Complete expert references and previously unused patients. Current development scores do not establish it. |
| Will it run on the final machine? | Inference is local and CPU-only. The brief targets four cores/8 GB; measured development times are separate from final-hardware validation. |

## Skills research and source requirements

Reviewed for the current revision:

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
- [Remotion video layout](https://github.com/remotion-dev/skills/blob/main/skills/remotion/rules/video-layout.md):
  one focal point per scene, readable short captions, safe margins, consistent
  composition and still-frame inspection. The separate showcase follows these
  principles with the existing FFmpeg tools, without adding a Remotion app.

No personal plugins were installed. Local plugin search found no suitable
presentation skill. External skill guidance informed the authoring; this kit
does not fabricate UI footage. See [SHOWCASE.md](SHOWCASE.md) for source
timings, render commands and soundtrack provenance.

## Authoring checks

```bash
.venv313/bin/python -m ruff check presentation
.venv313/bin/python -m mypy presentation/build.py presentation/showcase.py
npm --prefix presentation run lint
node presentation/build-pptx.cjs
```

`build.py` checks stage bounds, font text widths and speaker timing.
The PowerPoint generator permits PNG inputs only. The locked PptxGenJS
dependency has upstream `image-size` parser advisories for other image formats;
the build uses generated PNGs and does not process uploaded/untrusted images.
No audit settings or repository security controls were disabled.
