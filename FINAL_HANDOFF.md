> Historical strict-release evidence: the current default was restored to fusion on 2026-09-13. See [PRODUCTION_WORKFLOW.md](PRODUCTION_WORKFLOW.md) for current settings and validation. Figures and decisions below remain tied to their original source.

# Final hackathon handoff

## Decision

Keep **deterministic strict, 1 mm working spacing,
`native_contrast_scale=1.2`**. Use the default CLI without model or research
flags. No new fitted weights were promoted.

The subsequent [accuracy recheck](ACCURACY_RECHECK.md) found a guarded
wall-connection recovery method with local F1 0.5806 and precision 0.7500.
It preserves baseline matches across 80 synthetic controls but increases
reference count MAE from 2.0 to 2.2. It is available only through the experimental
`--recover-connected-origins` CLI flag; the release and Explorer remain strict.

The [selection report](FINAL_EVALUATION_RESULTS.md) remains authoritative.
New production outputs for all five reference cases are exactly equal to the
frozen selected outputs. Final replay verifies 280 variants, 180 eligible.

The remaining fixed 1.5 mm challenger was compared against strict on the
same frozen 24-case synthetic topology cohort:

| Grid | TP / FP / FN at 3 mm | F1 | Count MAE |
|---|---:|---:|---:|
| Strict 1 mm | 46 / 0 / 3 | 0.9684 | 0.1250 |
| 1.5 mm | 44 / 53 / 5 | 0.6027 | 2.3333 |

The coarser grid fails the topology gate. This is post-reference development
evidence, not independent validation. Reproduce it with:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4 \
.venv313/bin/python final_spacing_check.py --output outputs/spacing-replay
```

Frozen inputs, full prediction checkpoints and the result are stored under
`labels/finalization/spacing/`. No additional threshold search was performed.

## Final verification

| Check | Result |
|---|---|
| Python regression suite | 472 passed, 11 skipped |
| Ruff | Passed |
| Mypy | 29 configured and 5 explicitly checked sources passed |
| Injected network-denial regression | 63 passed |
| Frontend tests | 36 passed |
| Frontend lint, typecheck, production build | Passed |
| Explorer HTTP | Strict health, 25 available cases, static assets, subject018 analysis, CT/mask buffers and schema-valid JSON passed |
| Final selector replay | Passed; deployment remains deterministic strict |
| All 25 supplied inputs | Completed; 150 schema-valid daughter predictions |
| Released-case prediction identity | All five exactly equal the frozen selected JSON values |
| Windows wheel resolution | Complete pinned runtime closure resolves offline for CPython 3.13/x64 |
| Presentation generation | Eight slides, 300 seconds, 75 seconds per speaker; HTML, PDF and PPTX generated |
| Product showcase | Separate 40-second 1080p edit; original 60-second film preserved |

Ten normal-suite skips are optional Torch/ONNX/ONNX Runtime tests; the
remaining test requires denied networking and passed in the separate
network-denial run. Optional model training/inference was not required by the
chosen deployment. Browser-driven UI testing was not repeated in this final
pass; the final website checks used tests, a production build and HTTP.

The all-25 resource run applied OS affinity to four CPUs and thread ceilings
of four. Average end-to-end time per case was **5.316 seconds**, maximum
**23.212 seconds**. Maximum sampled RSS across the batch's process tree was
**1480.7 MiB**. This was Linux development hardware, with other checks running;
it is not an organizer-laptop benchmark or a hard memory-cap proof. Sampling
can miss brief peaks. The release carries the batch and resource reports.

The production website builds successfully with a nonfatal bundle-size
warning. The separate PowerPoint authoring dependency tree reports two high
audit findings for `image-size` malformed ICNS/JXL/HEIF parsing. The deck
builder only accepts generated, signature-checked PNGs; it does not accept
user uploads, and these packages are absent from inference/runtime wheels.
No forced dependency downgrade or security-policy exception was applied.

## Changes during finalization

- Reused the scorer's existing `1e-12` floating-point equivalence for replayed
  report/ranking aggregates. Prediction identity and manifests remain exact.
  The regression rejects materially changed scores, counts, IDs and hashes.
- Regenerated the selector manifest/reports after that fix.
- Formatted the three frontend files that failed Prettier.
- Installed the existing pinned resource requirements in CI so final-evaluation
  test collection has `psutil`; runtime inference requirements are unchanged.
- Fixed Windows checkout line-ending conversion of source/evidence hashes.
  Code and generated evidence retain LF; imported organizer originals retain
  their exact bytes, including CRLF. Hash validation was not weakened.
- Updated slide results and speaker notes to distinguish the five reused
  AI-assisted references from synthetic topology evidence.
- Added [one complete demo guide](DEMO_GUIDE.md), including Windows offline
  commands, local case layout, presentation files, timing and recovery.
- Refreshed the five-minute deck for the accepted strict configuration and the
  guarded-recovery tradeoff. Speaker 3 demonstrates the Explorer live, with CT
  inspection and JSON export; a still poster is the on-stage fallback.
- Separated the [product showcase](presentation/SHOWCASE.md) from the talk.
  The 40-second edit removes the old opening/ending cards, reframes the recorded
  interface, adds short feature captions and an original ambient soundtrack.
  Source footage and the original film remain unchanged.

## Submission contents and remaining team actions

Use the **Final hackathon submission** release on
[GitHub Releases](https://github.com/Coder-Meet/battleoftheschool/releases).
The submission ZIP contains source, built frontend, 25 prediction JSONs,
three visual checks, Windows runtime wheels, verification and checksums.
The presentation ZIP contains the local HTML/PPTX/PDF deck, script, live-demo
cues and CT evidence, with **no MP4 or embedded video**. Download
`branchseed-showcase.mp4` separately for Drive or other uploads.
Neither ZIP includes raw CT/mask volumes or a Python installer.

1. Run the offline CLI and website once on the actual Windows laptop using
   [DEMO_GUIDE.md](DEMO_GUIDE.md). Actual organizer-machine validation remains open.
2. Assign four speaker names and preload the demo case.
3. Upload the final ZIPs/video to the organizer or Devpost as required. Publishing
   a GitHub Release does not submit an entry to either destination.

All code/documentation work was committed directly on `main`; no PR or work
branch was created. Each archive records its exact source commit. GitHub CI
status is separate from the local checks above; inspect the latest workflow
for the recorded commit rather than assuming it has passed.
