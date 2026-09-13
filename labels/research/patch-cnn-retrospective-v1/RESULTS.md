# Optional patch CNN: retrospective research result

**Reject promotion of both CNNs and both selected blends.** They lose true
branches on the archived test cases. No model, threshold, blend weight, label
or archived cache was changed after seeing these results. The original frozen
tree remains the stronger comparator within this historical synthetic cohort;
this is not permission to deploy it.

## Protocol and freeze

The E1 gate was `go_research_only`. The protocol was committed at `267a436`.
Both CNN/sidecar pairs, training reports, blend weights and thresholds were
committed and pushed at `a8bfefb` before the separate test commands.
`freeze.json` records their hashes; all remained unchanged afterward.
The archive's test results were already public to this research workflow:
this is retrospective test reuse, not a new sealed test or clinical validation.

The classifier has **34,465 trainable parameters**: 12 input channels,
24/40/64 convolution widths, two average pools, global mean, binary head.
Both runs used 12 epochs, batch size 32, learning rate 0.001, seed 42 and four
CPU threads. Validation selected epoch 10 for synthetic-only and epoch 12 for
mixed. Normalization and weighted class/group/case sampling use train only.
Mixed pseudo source weight is 0.3 with 0.1 target smoothing.

The frozen arrays contain 307 synthetic training candidates (297 positive,
10 negative), 95 validation candidates (93/2), and 66 test candidates (63/3).
Ambiguous proposals are excluded from fitting but included in full-reference
evaluation; unproposed references remain false negatives.

## Full-reference results

Counts are TP / FP / FN from one-to-one ostium matching, including every
archived proposal and every analytic reference.

| Variant | Validation, 42 cases | Test, 28 cases | Test true branches lost vs pool |
|---|---:|---:|---:|
| Archived proposal pool | 93 / 4 / 10 | 63 / 4 / 7 | 0 |
| Archived strict detector | 91 / 0 / 12 | 61 / 1 / 9 | 2 |
| Original frozen gradient-boosting tree | 93 / 1 / 10 | 63 / 0 / 7 | 0 |
| Synthetic-only CNN | 93 / 3 / 10 | 59 / 2 / 11 | 4 |
| Mixed historical-pseudo CNN | 93 / 3 / 10 | 58 / 2 / 12 | 5 |
| Synthetic selected blend | 93 / 0 / 10 | 62 / 0 / 8 | 1 |
| Mixed selected blend | 93 / 0 / 10 | 62 / 0 / 8 | 1 |

Both blends selected **CNN weight 0** from the declared grid
`[0, 0.25, 0.5, 0.75, 1]`. Their threshold is
`0.7008606038842237`, selected on validation to retain its candidate positives.
These are rethresholded tree-only scores. The validation improvement therefore
does not demonstrate a CNN contribution. The selected threshold loses a test
positive. No in-sample stacking meta-model was trained.

Per-case/family counts, geometric errors, paired seed-group bootstrap intervals,
lost reference IDs and candidate scores are in the immutable training reports
and separate test reports. Only two test seed groups and very few negatives
make generalization estimates weak. No new patient reference accuracy was
measured.

## Mixed historical data audit

Only the 13 cases in the original real training split were considered.
There were 46 exact candidate ID/fingerprint/13-feature matches and
76 mismatches. One matched candidate (`subject010`, `branch_001`) could not
support the required parent-wall curvature feature and was excluded with
`Insufficient observed parent wall for curvature.` No missing feature or patch
was replaced with zero.

The resulting 45 cached candidates contain 24 confirmed and 21 rejected
historical Claude labels. Their image/mask hashes, ordered indices, source
hashes, prior-inspection/pseudo-training flags and original review hash are
preserved. No new labels were made. All 25 previously inspected/pseudo-trained
cases remain in the exposure ledger, including those absent from this fit.
They may overlap future expert cases. Validation/test remain the identical
synthetic arrays; the mixed manifest receives its own hash and parent link.

The frozen mixed review header inherited the original aggregate
`label_source=analytic_synthetic_geometry`. That field alone is incomplete:
the manifest declares mixed sources, added rows retain `labeller=claude`,
and `mixed_provenance` records the real cases. Sampling used row labellers,
so the 0.3 pseudo weight was applied correctly. Future cache preparation
corrects this header; frozen caches and model sidecars were not rewritten.

## Physical and source boundary

The extractor contract is `physical-candidate-patches-v1`, float32
`(N,12,32,32)` at 1 mm spacing: XY/XZ/YZ times CT/parent/vesselness/path.
Numeric NPZ is loaded with `allow_pickle=False`. The shared extractor hashes,
channel order, finite ranges, identity, features, split membership and source
hashes are checked rather than inferred.

Training applies CT-only blur, up to 0.5 pixel sigma. HU noise/jitter are disabled
because these caches do not provide a verified per-case HU normalization scale.
Translations and rotations are disabled: the cached 2.5D planes do not contain
translated off-plane samples. No regression, radius, direction or geometry
targets are fabricated.

Archived detector SHA256:
`17495723df18c79e4302edc19bd949bb16f615b902403a88c68d6e74fa200063`.
Shared extractor SHA256:
`568d8bd9f0d417768cb22fdd56d054a26ca215cc1741d3d9914ff4052eaa534a`.
Current detector SHA256 at this experiment:
`9f96f3eb3c06f5e06d5b7db79b199be70e742130fc3885243d25d066adc1c92e`.

The archive predates the organizer-confirmed **2 mm origin diameter** rule.
It was not relabelled or re-exported. These results do not validate current-main
eligibility. Mixed patch extraction explicitly required the archived detector
and identical extractor source; training consumes cached patches.
No production detector, existing labels, base dependencies or CLI integration
were changed by this work.

## ONNX and CPU observations

Native/ONNX maximum absolute errors were `1.1920928955078125e-7`
(synthetic) and `1.7881393432617188e-7` (mixed), below explicit tolerances
`atol=2e-6`, `rtol=1e-5`. Validation threshold decisions agreed. Empty input and
batch size 7 were verified. CPU inference permits only `CPUExecutionProvider`,
1–4 intra-op threads and one inter-op thread.

Four-core-affinity Linux training/export processes took 3.327 s / 552.36 MiB
peak RSS and 3.502 s / 550.21 MiB for synthetic/mixed. Fresh inference processes
used 252.54 / 252.71 MiB peak RSS, including imports and input. Across 50 repeats
on 95 cached patches, mean batch scoring was 32.97 / 19.93 ms; p95 was
142.82 / 23.44 ms. Host variability is visible in these measurements.
Inference imported neither Torch nor ONNX.

These observations exclude detector/extraction latency on clinical-size
volumes. **Native Windows four-core/8-GB/offline execution remains unverified.**
The chosen Torch 2.8.0 CPU, ONNX 1.19.0 and ONNX Runtime 1.23.2 versions have
published Windows CPython 3.13 x64 wheels. Wheel availability is not a Windows
runtime or performance test.

## Checks and handoff

Ruff and mypy passed for the assigned Python files. Targeted tests: 36 passed.
With every network syscall forced to `ENETUNREACH` using Linux `strace`, all
36 patch tests plus the network-denial assertion passed (37 total), including
tiny training/export, CPU inference and base imports without optional packages.
Before the final added case-coverage regression and aggregate-header correction,
the full suite was 339 passed / 27 skipped / 51 upstream warnings in 125.30 s;
26 skips were optional tree-training dependency tests. Windows-platform mypy
passed before those final changes. No native Windows or CI pass is claimed.

The handoff includes exact model/sidecar pairs, reports, freeze hashes,
source snapshots, reproducible CPU benchmark and numeric mixed caches.
The data bundle contains derived candidate patches and analytic references,
not full patient CTs. Base operation needs none of the optional ML packages.
The pipeline and rejected research artifacts are available for reproducibility;
further model selection requires new prospectively held-out data.
