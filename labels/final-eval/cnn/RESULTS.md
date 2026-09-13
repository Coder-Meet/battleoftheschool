# Frozen CNN comparison on the released five cases

**CNN weighting did not improve the selected family result.** Every
leave-one-case-out fold chose CNN weight zero. The within-family held-out
3-mm result is TP 8 / FP 3 / FN 11, F1 0.5333, equal to the unfiltered strict
baseline's pooled F1. These are local development-reference comparisons, not an
official score or independent clinical estimate. No model was promoted.

## Scope and reproducibility

- `report.json`: 112 variants, each scored on all five cases at 2/3/5 mm,
  per-case matches and missing reference IDs, true-reference losses against its
  own proposal baseline, ordered-probability replay configuration and runtime.
- 84 current-source rows: two pools × (five weights × eight thresholds +
  unfiltered baseline + frozen blend).
- 28 historical rows: two CNN artifacts × two pools × (six thresholds +
  unfiltered baseline). They are retrospective and ineligible for clean selection.
- `candidates/`: 30 successfully measured extraction/inference runs. All input
  bytes and physical geometries match the validated release. Every worker used
  at most four CPU cores and four numerical-library threads, CPU-only ONNX, and
  denied network syscalls. No CT was uploaded.
- `predictions/`: 560 complete, schema-valid JSON files. Valid empty predictions
  are retained. `inventory.json` audits all three repository ONNX models.
- `HANDOFF.md` gives exact extraction, resume, source archive and replay commands.
  Replay does not load CTs. No model was fitted or calibrated on these cases.

Eight current thresholds: `0.15`, frozen CNN `0.2103264733200073`, `0.3`,
`0.5`, frozen tree `0.5630076829578097`, `0.7`, frozen blend
`0.7008606038842237`, `0.85`. CNN weights are `0`, `.25`, `.5`, `.75`, `1`.
Historical models use their frozen CNN threshold plus the five requested probes.
Pairs are scored on exactly the same ordered identities, features, inputs and
extraction contract; preserved geometry is checked against candidate fingerprints.

## Selected comparisons at 3 mm

| Configuration | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| Current strict unfiltered | 8 | 3 | 11 | 0.5333 |
| Current review-union unfiltered | 10 | 14 | 9 | 0.4651 |
| Current strict CNN, frozen threshold | 6 | 1 | 13 | 0.4615 |
| Current review-union CNN, frozen threshold | 7 | 7 | 12 | 0.4242 |
| Current strict frozen blend | 7 | 0 | 12 | 0.5385 |
| Current review-union frozen blend | 6 | 1 | 13 | 0.4615 |
| Development winner: review-union, weight 0, frozen tree threshold | 9 | 2 | 10 | 0.6000 |
| Strict weight 0, threshold 0.3 | 8 | 0 | 11 | 0.5926 |
| Within-family held-out selection | 8 | 3 | 11 | 0.5333 |

The saved frozen blend already has CNN weight zero; evaluating its declared
threshold remains distinct from the frozen tree threshold. The all-five winner
is a **development winner**, not its own held-out score. Three folds select
review-union at the frozen tree threshold, one selects review-union at 0.3,
and one selects strict at 0.3. The same fixed selection has F1 0.4000 at 2 mm
and 0.5333 at 5 mm. Parent integration must rerun selection across all families.

Paired whole-case bootstrap uses 10,000 resamples with seed 42. The 3-mm
held-out-minus-strict-baseline F1 interval is approximately [-0.0716, 0.0435].
The development winner's F1 interval is [0.1111, 0.8500]; its paired difference
from strict baseline is [0, 0.2353]. Selection stays fixed during resampling.
With five reused cases these intervals are descriptive and do not remove
selection bias.

## Missed references and origin policy

At 3 mm strict proposals find 8/19 references, review-union 10/19. Neither pool
finds any matched reference in subject019 or subject020. Pure classifiers cannot
recover references that never entered the proposal pool. Frozen CNN filtering
loses two strict true positives and three review-union true positives relative
to those pools; every variant records the exact reference identities lost.

Review-union retains the existing research origin-diameter override. Its clean
selection eligibility describes synthetic-only model fitting, **not final
2-mm-origin deployment eligibility**. Strict uses the current 2-mm origin policy.
Legacy source predates it. All paths preserve the detector's 5-mm seed /
up-to-10-mm tracing contract. Only three reference radii are measured; radius
summaries include only known matched radii, never invented replacements.

## Artifact provenance and exclusions

Current synthetic training is verified through source hashes in training
records, the synthetic manifest, and fresh-corpus reproduction receipts.
Current and historical synthetic ONNX graph bytes happen to match, but their
sidecars bind different detector sources. Both were run through their own
verified contracts; no sidecar was substituted.

Historical synthetic and mixed models ran with detector/extractor/learning/
NIfTI source archived from `3395de51c91fe63882005a45e18f3fee54fce8ca`,
using the unchanged validated inference bridge from preparation commit
`2a825d89b66e7d5c61bea7f497be08fbc22cf526`. Mixed fitting includes subject022
and subject023; it is contaminated and ineligible. Both old blend manifests
reference a tree without a verified extraction contract, so those blends were
excluded rather than run through weakened checks. The inventory documents a
source-hashed re-extraction/retraining path.

Driver hashes differ across checkpoint/resume edits. Detector, extractor,
learning and inference hashes remained unchanged during the matrix. The source
hash in each receipt identifies the driver file present when the receipt was
written; the authoritative extraction and inference contract hashes are recorded
separately.

## Resource evidence and checks

Measured full-worker wall time ranges from 2.36 to 28.28 seconds; the maximum
sampled process-tree RSS is 715.27 MB. Separate image-load, detection, model-load,
physical-patch extraction and combined scoring stages are saved. Full-worker
times are conservative shared costs across all threshold/weight replays,
including both current models even when their weight is zero. They include
startup, diagnostics and syscall-tracing overhead, and are not native Windows
benchmarks. Sampling may miss short peaks; host contention was not isolated.

Ruff and mypy passed for assigned source/tests. Focused evaluator/inference and
artifact tests: 28 passed, 2 skipped. Detector/physical-patch regressions:
91 passed, 7 skipped. All 560 prediction schemas, exact score reproductions,
ordered probability replays, receipt hashes and RAM bounds were independently
verified. Optional training-package tests are skipped because sklearn/Torch/ONNX
export packages are absent from this inference environment. The actual CPU ONNX
model load, empty-input path and batched inference parity tests pass.

No production source, requirements, existing model artifact, or other family's
files were modified. Remaining decisions belong to parent-wide comparison:
cross-family selection, any proposal-recovery work, final origin eligibility,
synthetic promotion gates and native Windows verification.
