# Project media and provenance

The PNGs are ready for the README and a Devpost gallery. Use `branchseed-cover.png`
as the cover. [Submission captions](../../DEVPOST_SUBMISSION.md#5-cover-image-and-gallery-order)
keep the distinction between recorded interface views and measured performance.

- `explorer-overview.png`: the preserved full Explorer poster.
- `ct-evidence.png`, `wall-map.png`, `interior-tour.png`: full, uncropped frames
  at 68, 90 and 121 seconds of the original capture.
- `branchseed-cover.png`: brand composition containing the complete scaled
  Explorer screenshot; no simulated scan or fabricated detection.
- `evidence-scorecard.png`: strict local reference and synthetic topology
  results, in separate equally labelled panels.
- `runtime.png`: all 25 recorded per-case end-to-end times, with mean,
  maximum and sampled process-tree memory.
- `pipeline.png`: a conceptual software diagram, not a patient image.
- `metrics.json`: source receipts, exact metrics, input/output hashes and
  recording timestamps.

Visible screenshot counts, scores and timing belong to the historical
recording; do not use them as current accuracy or runtime evidence. The
performance graphics use the frozen strict reports named in `metrics.json`.
Matched geometry summaries are conditional on eight matches; missed branches
are excluded. No clinical or hidden-test accuracy is claimed.

## Rebuild

Use the existing project Python dependencies and FFmpeg. Extract the current
[presentation kit](https://github.com/Coder-Meet/battleoftheschool/releases/download/branchseed-final-2026-09-13/branchseed-final-presentation.zip)
to `outputs/live-presentation-kit/` for its local font files and poster.
The preserved source recording is included in the
[original presentation kit](https://github.com/Coder-Meet/battleoftheschool/releases/download/branchseed-presentation-2026-09-12/branchseed-presentation-kit.zip).
Put its `source-footage.mp4` at `outputs/presentation-kit/source-footage.mp4`.

From the repository root:

```bash
.venv313/bin/python presentation/build-project-media.py
.venv313/bin/python -m ruff check presentation/build-project-media.py
.venv313/bin/python -m mypy presentation/build-project-media.py
```

The script reads local JSON receipts, checks TP/FP/FN against F1, validates
case completion/platform, checks text widths, preserves the source hash and
writes the manifest. No database or SQL is used. It does not rerun inference,
fit models or select thresholds.

Typography uses the presentation kit's Manrope and Fraunces fonts; their
licences remain in that kit. The graphics and recorded interface are project
assets; credit Toralis Labs for the supplied challenge data. Consult the
sponsor brief for any additional dataset-use conditions.
