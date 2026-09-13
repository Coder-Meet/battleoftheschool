# Robustness protocol, frozen before variant selection

The five released phantoms remain training/development data. This stress suite
is also procedural, not clinical anatomy or the organizer's hidden evaluation.
No synthetic score establishes clinical accuracy.

## Candidate filter study — 2026-09-12

The `2e17139` label ledger contains 52 **Claude pseudo-labels** from six real
cases; no trained weights were present in that revision. The following are
new local fits, not a reproduction of Steven's missing weight file.

Training uses subject001/002/003/007. Subject004 supplies only five validation
rows (four positive, one negative). Subject006 is held out from fitting, but
its initial failed result has already been inspected; treat that case as
development evidence rather than a fresh final accuracy estimate.

The blended fit adds 40 analytic candidate labels from seed 4001 and the
five hard seed-731927 phantoms, all in training. No synthetic generation seed
straddles a partition. Unreviewed real packets are not training labels.

Unconstrained validation-F1 selection chose thresholds 0.9105 (AI only) and
0.9213 (blended). Both got all five validation rows right but discarded
15/17 and 19/56 positive **training** rows, respectively. This is enough to
reject them before independent synthetic evaluation. Tiny validation sets
can reward an excessively restrictive operating point.

The optional 95% training-recall constraint selects threshold 0.5 for the
same blended coefficients: 54/56 training positives retained, with 3/19
training negatives retained. Validation retains all five proposals. This
constraint does not establish unseen recall or calibration.

Before opening independent predictions, freeze the constrained model and
new 14-family synthetic seed 864203:

- Model SHA-256: `de1170092db85199f2cd3fdf0c8b37c60edc0dad27e866d6c673c286e19d44d1`
- Manifest SHA-256: `a9eef26c633ecb453d545bfdb193c5706196cdbf8dac122dd9608c8a7833bb3a`
- Real review SHA-256: `5110230f2547609f4d2113c41212f89d6e3287396204788d27f5861ffde5b020`

Evaluate strict + filter and review-union + filter against the unchanged
production detector, at 3 mm with 2/5 mm sensitivity. Existing frozen sets
are additional regression checks. Do not change the coefficients or
threshold after opening the new seed's results. Keep default inference
unchanged if the filter drops true branches or the wider pool introduces
additional false positives.

### Expanded labels and Steven's supplied weights

Steven's upstream commit `0207bf7` subsequently supplied the model, split and
237 Claude verdicts across 23 nonempty candidate pools. Subject017/019 had
empty pools. These assets are now also on this repository's main branch.
The supplied 13-feature schema, normalization and split match `learning.py`;
local retraining differs only at approximately floating-point roundoff in
coefficients. Benchmark the original supplied bytes, not the local refit.

Keep Steven's five validation and five test cases unchanged. Add the same
40 analytic synthetic rows only to training; use the previously chosen 95%
training-recall constraint. This expanded blended model selects threshold
0.5, retains 86/89 positive training labels, and retains 17/21 positive
validation labels. It does not promise 95% validation or test recall.

Seed 864203 has already been inspected in the smaller-label experiment, so
it is now development evidence. Before evaluating the expanded models,
freeze a new seed **557891**, all 14 stress families, and both models:

- Steven model SHA-256: `a45747a0382ae41d0a37f2104f2348190905c847a1b89347f9c9703448ddccc6`
- Expanded blended SHA-256: `9d7026550b834b5f427d4b6404cd6ff02b182b9db7ffd207f0f24b802a83e2f1`
- Expanded review SHA-256: `319138ab8e6778f4378d5938901e934d5997dc303adc01f54e3ed7be901a7b5f`

Evaluate each with strict and review-union candidates; also run strict
regressions on seeds 582743/904117. Do not retune either model after viewing
these outcomes. Scores on AI verdicts are candidate-label agreement; only
the analytic cases measure complete branch discovery here.

### Measured outcomes

**Decision: keep the production detector unchanged.** The blended weights
are a safer optional experiment than the supplied weights on these synthetic
cases, but neither has established real branch-discovery accuracy.

At 3 mm, the new 14-case seed 557891 produced:

| Pipeline | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| Strict production | 26 | 1 | 5 | 0.8966 |
| Strict + Steven filter | 25 | 1 | 6 | 0.8772 |
| Strict + blended filter | 26 | 1 | 5 | 0.8966 |
| Review union, unfiltered | 28 | 0 | 3 | 0.9492 |
| Review union + Steven filter | 27 | 0 | 4 | 0.9310 |
| Review union + blended filter | 28 | 0 | 3 | 0.9492 |

The additional discoveries come from the broader proposal stage, not the
filter. The supplied filter removes a true thick-slice branch. The blended
model keeps it. All coefficients and thresholds remained frozen.

Strict regression results, reported as TP/FP/FN:

| Seed | Production | Steven filter | Blended filter |
|---|---|---|---|
| 582743 | 26/1/6 | 25/1/7 | 26/1/6 |
| 904117 | 27/0/4 | 26/0/5 | 27/0/4 |

Across these 40 cases, adding synthetic training labels avoids all three
true-branch losses caused by Steven's filter. The blended strict pipeline
keeps every production prediction, including its two false positives.
This is an improvement over the supplied filter, not over production
branch discovery.

The attractive fresh-seed result does not justify switching to the review
union. Additional regression checks with the frozen blended model found:

| Seed | Strict production TP/FP/FN | Blended review union TP/FP/FN |
|---|---|---|
| 864203 (now development) | 31/0/3 | 31/2/3 |
| 582743 | 26/1/6 | 28/3/4 |
| 904117 | 27/0/4 | 28/1/3 |

It recovers some branches but adds false positives, including dense-field
and curved-daughter distractors. These follow-up results were inspected
without further tuning. The broader mode remains a benchmark experiment.

On the five held-out **AI-labelled candidate pools**, both expanded models
retain 20/21 AI-confirmed candidates and reject 45/47 AI-rejected candidates
(candidate F1 0.9302). The blended model removes one additional false
positive on the validation labels (F1 0.8293 → 0.8500), but has no test
classification-count improvement. These incomplete pseudo-labels cannot
measure branches absent from the pool or establish clinical accuracy.

Rerunning all 25 supplied scans reproduced every frozen production JSON
after JSON serialization: **151 daughters**. Applying the optional filters
to strict proposals yields **98** (Steven) or **93** (blended) daughters.
Those removals are unadjudicated. In particular, both filters remove the
only production proposal in subject020; do not interpret that empty output
as proof that the case contains no eligible branch.

The detailed per-case synthetic metrics, 2/3/5 mm sensitivity, source/model
hashes and split-overlap checks are in `labels/evaluation-summary.json`.
Source checks passed: Ruff, mypy, 105 tests plus one expected skip, and
19 selected tests with network calls denied. Native organizer Windows
runtime and real-reference accuracy remain unmeasured.

## Evaluation audit

The evaluator uses maximum-cardinality, minimum-distance one-to-one ostium
assignment. Its invalid-edge cost is `(min(predictions, references) + 1) * tolerance`;
one extra valid assignment saves more than every valid distance can cost.
Duplicate predictions count as false positives. Unmatched IDs are exposed.
Empty-case precision/recall are undefined, not 100%; negative-control false
positives are reported separately. There is no ignored-prediction exemption:
predictions at ineligible short stubs also count as false positives.

Discovery uses 3 mm tolerance, with 2 and 5 mm sensitivity from the same saved
predictions. Geometric errors (ostium, seed, radius and direction) apply only
to matched pairs and must always accompany TP/FP/FN and recall. These rules
are a local proxy; organizer tolerances, minimum eligible size and instance
quality scoring have not been supplied.

## Dataset freeze

`stress.py` uses NumPy SeedSequence with the fixed family index, independent
of Python's randomized hash seed. Analytic references precede noise and
voxelization and never use detector predictions.

- Development: all 13 families, seed 4001.
- Frozen evaluation: all 13 families, seeds 90817 and 112213 (26 cases).
- Do not inspect frozen detector results until the selected implementation and
  its complete configuration have been committed.
- Run the baseline and selected implementation once each on those frozen cases.
  Disclose failures; do not tune and rerun on this set.
- If later changes are needed, label this set development and freeze a new set.

Families include tortuous/tapered parents, curved daughters, low contrast,
2.5 mm slices, touching veins, calcification, aneurysm/thrombus, common trunks,
downstream daughters, nearby ostia, ineligible stubs, short cropped masks,
mask perturbations, dense branches, negative controls and high-noise small
branches. Negative structures and references have independent provenance.
Mask perturbations deliberately disagree with the anatomical wall reference.
The voxelized lumens are still simplified, with no patient-derived texture or
expert adjudication of visibility.

The exploratory pre-protocol `stress-dev` run used a nondeterministic generator
and is discarded; it is not a baseline for any claimed improvement.

## Alternatives and selection

Start with four defensible approaches on development only:

1. Strict production baseline at 1 mm.
2. Finer 0.75 mm sampling, testing geometric resolution against CPU cost.
3. Relaxed blood contrast threshold, testing partial-volume sensitivity.
4. Loose human-review proposal profile, measuring its recall/false-positive
   tradeoff; it is not silently promoted to submission inference.

An additional focused fix may be derived from development failures, but it must
retain genuine aorta connection and respect the physical geometry contract.
Select using branch-level micro F1 at 3 mm. Prefer the baseline if gains require
more false positives or regress the existing topology/geometry tests. Report all
variants and runtime, not only the winner. Freeze the final choice in a commit
before opening frozen results.

## Reproduction

```bash
python stress.py --output-dir outputs/robust-dev --seed 4001
python benchmark.py --data-root outputs/robust-dev --output outputs/robust-strict.json --variant strict
python benchmark.py --data-root outputs/robust-dev --output outputs/robust-finer.json --variant finer
python benchmark.py --data-root outputs/robust-dev --output outputs/robust-relaxed.json --variant relaxed
python benchmark.py --data-root outputs/robust-dev --output outputs/robust-review.json --variant review
```

Generator directories and benchmark reports refuse overwrite. Reports include
input hashes, source hashes, Git revision and working-tree status. Benchmark
checks input hashes before inference. Hold thread limits at four throughout.
Use fresh per-case processes under Linux `/usr/bin/time -v` for peak resident
memory; Python allocation tracing is not a substitute for native memory and
should not distort inference timing.

## Release checks

Run Python tests, Ruff, mypy, network-denied algorithm/training tests, frontend
tests/lint/build, the five development phantoms, all 25 real scans, and the
required CLI under four-core CPU affinity. Validate every saved challenge JSON.
Real-scan execution measures resources and robustness to inputs, not detection
accuracy without complete daughter annotations. Preserve the presentation's
historical development claims, and add a dated stress-results supplement rather
than replacing them with an unqualified accuracy number.

## Selection record — before frozen inference

Baseline revision: `406ab01`. Development seed 4001 has 32 reference daughters.

| Variant | TP | FP | FN | F1 | Mean detector seconds |
|---|---:|---:|---:|---:|---:|
| Strict baseline | 21 | 0 | 11 | 0.7925 | 1.00 |
| 0.75 mm sampling | 22 | 0 | 10 | 0.8148 | 1.82 |
| Relaxed HU threshold | 28 | 0 | 4 | 0.9333 | 1.14 |
| Loose review proposals (without strict-pool union) | 28 | 2 | 4 | 0.9032 | 1.08 |
| Half-contrast support | 28 | 0 | 4 | 0.9333 | 1.09 |

Selected: half-contrast support at 1 mm. It ties the relaxed HU threshold on
discovery and uses the measured blood/background contrast rather than a
fixed HU decrement. Threshold = the midpoint between background and blood
(subject to the existing 30 HU lower bound). Strict wall connectivity,
tubularity, radius, topology and cap filters remain required. The original
five development phantoms still have 10 TP / 0 FP / 0 FN. The full Python
suite has 69 passed and the expected one network-sandbox skip; frontend
lint/types/build and six review tests passed before selection.

Four development misses remain: two thick-slice small branches and two
high-noise small branches, all rejected for lacking a supported distal path.
The procedural references describe analytic geometry; expert judgement of
5 mm visibility after partial-volume loss is unavailable. They remain misses
in the report, not ignored predictions or removed cases.

Frozen manifest SHA-256 values:

- Development 4001: `49ad6220758c7a07d8db836ffeeb83a4ed9a3eb329d0c96dd04a819ae297a00f`
- Evaluation 90817: `f4ba6252877b940c82578a4ba442a77d04cea1d47a0c5c4bb5a4787e6d8f1a82`
- Evaluation 112213: `fe5f96ae9656e062a5080b801470d42122557a687b672abd69b5087f1b0b7e7e`

Both evaluation datasets were generated before selection. Neither detector's
evaluation predictions were opened before this decision was committed.

## Frozen results — 2026-09-12

Baseline predictions were generated at `406ab01`; selected predictions at
`fd8f67f`. Both recorded clean working trees. The selected detector source
SHA-256 is `ecc058e475bad05d67be6ab8e8c06973a41a9ae38e05bb4476a5e20e5738b7e7`.
The later review/UI changes do not change this detector. Neither frozen set
was used for subsequent threshold tuning.

| Implementation / frozen seed | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline, both seeds | 46 | 0 | 18 | 1.0000 | 0.7188 | 0.8364 |
| Selected, 90817 | 29 | 1 | 4 | 0.9667 | 0.8788 | 0.9206 |
| Selected, 112213 | 27 | 0 | 4 | 1.0000 | 0.8710 | 0.9310 |
| Selected, both seeds | 56 | 1 | 8 | 0.9825 | 0.8750 | 0.9256 |

These are 26 procedural cases with 64 analytic reference daughters, not 26
patients. The improvement recovered ten additional references and introduced
one false positive. Both negative controls remained empty. Selected mean
detector time was 1.0115 seconds; the slowest case was 1.409 seconds.

Failures remain visible: four misses in high-noise small branches, three in
thick-slice acquisitions, one miss and one false positive in dense branch
fields. Do not remove these cases, redefine their eligibility after observing
the result, or describe this as perfect detection.

The same saved selected predictions at 2 mm tolerance give 54 TP / 3 FP / 10 FN;
3 mm and 5 mm each give 56 TP / 1 FP / 8 FN. Baseline at 2 mm gives
45 TP / 1 FP / 19 FN; at 3 and 5 mm it gives 46 TP / 0 FP / 18 FN.

| Matched-pair error only | Baseline mean | Selected mean | Selected p95 |
|---|---:|---:|---:|
| Ostium distance, mm | 0.746 | 0.844 | 1.760 |
| Seed distance, mm | 0.609 | 0.648 | 1.144 |
| Absolute radius error, mm | 0.191 | 0.217 | 0.467 |
| Direction error, degrees | 8.471 | 10.560 | 24.854 |

Matched geometry did not improve across every metric. The matched populations
also differ, since the selected detector includes harder recovered branches.
Discovery and geometry must be discussed together.

## Real-input resource audit

All 25 supplied scans completed through the required CLI under Linux CPU
affinity 0–3, four numerical-library threads, and an 8 GiB virtual-address-space
limit. Each ran in a fresh process with `/usr/bin/time` capturing elapsed
seconds, maximum RSS in KiB and exit status. All exit statuses were zero; all
25 output JSONs passed the evaluator's physical-output validation.

- Mean end-to-end elapsed time: 5.837 seconds.
- Slowest end-to-end elapsed time: 23.35 seconds (`subject025`).
- Mean detector-reported time: 5.159 seconds.
- Maximum detector-reported time: 22.345 seconds.
- Maximum peak RSS: 1,495,320 KiB, approximately 1.43 GiB (`subject018`).
- Total predictions: 146, including empty daughter lists.

The 8 GiB virtual-memory limit is a conservative process limit, not an
emulation of the organizer's complete operating system. CPU affinity fixes
the available logical cores, not their speed. These observations meet the
initial local runtime target but do not guarantee organizer hardware results.
No complete real daughter annotations were available: real precision, recall,
instance quality and clinical validity remain unmeasured.

The delivery evidence ZIP includes unchanged per-case predictions, resource
files, development/frozen reports and dataset manifests. Regenerate datasets
with the recorded seeds in fresh directories to verify the report hashes.

## Native-resolution experiment — selection before new frozen inference

The teammate's fixed-65-HU diagnosis describes the earlier baseline. The
current default already uses half of the measured parent/background contrast.
The new experiment lowers that fraction only when native resolution warrants it:

```text
p = r² / (r² + (0.6 × max(native_spacing))²)
f = min(0.5, scale × p)
lower = max(30 HU, background + f × (parent - background))
```

Here `r` is the configured minimum radius, not a case-specific anatomical label.
The formula approximates partial volume; it is not a calibrated scanner PSF.
At nonpositive parent/background contrast the existing conservative fallback
remains. The old midpoint clamp is absent from the experimental path.
`native_contrast_scale=0` preserves the existing detector; the benchmark's
`native-contrast` variant selects scale 1.2. It remained opt-in until frozen
evaluation; the promotion is recorded below.

Development seed 4001, before any new frozen predictions:

| Native scale | TP | FP | FN | F1 | Negative-control FP |
|---|---:|---:|---:|---:|---:|
| Disabled (existing half-contrast) | 28 | 0 | 4 | 0.9333 | 0 |
| 1.2 | 29 | 0 | 3 | 0.9508 | 0 |
| 0.9 | 30 | 2 | 2 | 0.9375 | 0 |
| 0.6 | 29 | 20 | 3 | 0.7160 | 8 |

Select **1.2 only** for the independent comparison. Do not retune using the
new frozen cases. Promote only if aggregate frozen F1 improves without more
false positives or negative-control detections, the original geometry tests
pass, and real-input runtime/memory remain within the existing limits.
Otherwise retain the existing production default and report the rejected result.

New datasets were generated before this selection:

- Seed 582743: `d98c8d368881567c0ab95dcd1a303106fa60c2f7aab1677643fbcf94545ca857`
- Seed 904117: `2a457b54796d8a0bfb54b19a75297b97d08a63993211209392b1a6738170f349`

These remain procedural draws from the same generator, not independent
clinical data. The old frozen sets are regression data for this investigation.

All three scales were also swept over all 25 supplied scans. Counts and their
correlation with parent HU are diagnostics, not optimization targets: there
are no real reference counts, and anatomy, crop coverage and acquisition
protocol are confounded. In particular, neither more detections nor zero
correlation proves improvement. The aggressive scale's eight false positives
on an empty control disqualify it despite its increased candidate recovery.

| Real-scan variant | HU/count correlation | Mean count | Count range | Mean / max detector seconds | Peak RSS, KiB |
|---|---:|---:|---:|---:|---:|
| Existing midpoint | -0.6399 | 5.84 | 0–14 | 4.990 / 21.093 | 1,505,784 |
| Native 1.2 | -0.5727 | 6.04 | 0–16 | 5.018 / 21.316 | 1,505,676 |
| Native 0.9 | -0.6067 | 6.32 | 0–19 | 5.713 / 24.616 | 1,539,784 |
| Native 0.6 | -0.5092 | 9.32 | 0–24 | 5.840 / 24.164 | 1,500,416 |

The midpoint timing is from an explicit `native_contrast_scale=0` replay at
`09f753b`; all 25 predictions exactly matched the initial midpoint run.
The original initial-run checksum was sampled after execution, so the replay
also pins an unchanged source checksum across the run. These are single
local sequential sweeps, not replicated speed comparisons. The final
production batch and fresh required-CLI tail checks are in the submission
audit; all production predictions exactly matched the selected experiment.
Per-case counts, diagnostics, predictions, resource measurements and frozen
reports are supplied in the native-evidence archive.

Diagnostics now expose background MAD and parent/background contrast divided
by that MAD, and warn when their intensity distributions overlap. This MAD
measures heterogeneous tissue, **not image noise**. It does not set the intensity
threshold or establish a clinical CNR/visibility cutoff. A low absolute parent
HU warning is retained; decisions still require complete expert references.

### New frozen result and promotion

Selection and implementation were committed at `d9fc6c7` before opening the new
frozen predictions. Both variants' four reports record that revision, clean
working trees and the manifest hashes above.

| Variant / seed | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| Existing midpoint / 582743 | 25 | 1 | 7 | 0.8621 |
| Native 1.2 / 582743 | 26 | 1 | 6 | 0.8814 |
| Existing midpoint / 904117 | 26 | 0 | 5 | 0.9123 |
| Native 1.2 / 904117 | 27 | 0 | 4 | 0.9310 |
| Existing midpoint / combined | 51 | 1 | 12 | 0.8870 |
| Native 1.2 / combined | 53 | 1 | 10 | 0.9060 |

The fixed candidate recovered one thick-slice reference per seed without new
false positives. Both empty controls remained empty. Ten misses and one false
positive remain, including high-noise branches and vessel/distractor contacts.
The absolute score is lower than the earlier frozen result because these are
different procedural draws; all results are retained.

Promote the preselected scale 1.2 as the default, subject to the release checks.
Historical benchmark variants explicitly disable native adaptation so their
commands remain reproducible. The controlled contrast regressions and
voxel-integrated phantoms test separate photometric and sampling behavior.
The point-sampled, one-voxel-width development probe still exposes subvoxel
phase sensitivity; it is not evidence of complete small-vessel recovery.

Matched-pair mean errors for the new frozen comparison:

| Metric | Prior midpoint | Native 1.2 |
|---|---:|---:|
| Ostium, mm | 0.863 | 0.878 |
| Seed, mm | 0.546 | 0.551 |
| Absolute radius, mm | 0.190 | 0.182 |
| Direction, degrees | 10.163 | 10.440 |

Matched populations differ; do not claim every geometric measurement improved.
The older frozen seeds were rerun only after selection: native results were
30 TP / 0 FP / 3 FN for 90817 and 27 TP / 0 FP / 4 FN for 112213.
The five training cases retained 10 TP / 0 FP / 0 FN.

## Wall-parallel daughter experiment — opt-in, not promoted

Hypothesis: the strict tracer requires a daughter to move monotonically away
from the parent, so a daughter that leaves the wall and then runs alongside it
(clearance under 5 mm for its first 10 mm) is rejected as `wall_hugging_path`.
Some real daughters do this; so do veins, mural thrombus and calcified wall.

Implementation (`DetectorConfig.parallel_clearance_mm`, default `0` = off):
only after the ordinary endpoint search fails, endpoints are accepted by
geodesic path length through supported vessel voxels instead of Euclidean
parent clearance, provided the path clears the parent by at least the
configured margin and the lumen radius stays above the minimum. The traced
path is extended upstream to where it first enters the contact shell, paths
whose mean HU exceeds the parent blood pool are rejected
(`hyperdense_wall_structure`), and an opening lying on another daughter's
proximal path is folded into that daughter (`opening_on_another_path`).
`benchmark.py --variant wall-parallel` enables it with 2.5 mm clearance.

A new procedural family, `wall_parallel_descending`, generates one hugging
daughter (analytic clearance 1.5–3 mm, checked by the test) plus one ordinary
daughter. Ten seeds, 20 eligible references, 3 mm tolerance:

| Variant | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| Native 1.2 (production) | 10 | 0 | 10 | 0.667 |
| Wall-parallel 2.5 mm | 18 | 0 | 2 | 0.947 |

Frozen seeds 582743 and 904117 (63 references, all 13 earlier families,
including touching vein/calcification, mural thrombus and nearby pairs): the
wall-parallel variant reproduced the production counts exactly (26/1/6 and
27/0/4) and identical mean ostium error. An earlier draft without the
hyperdense rejection and path folding produced 4 and 1 false positives there;
those drafts were discarded before this record.

Real 25-case batch, production versus wall-parallel (151 → 152 daughters, per-case
detection time unchanged within noise): four openings were added (subject004,
017, 023, 025) and three were folded into another daughter's path (two in
subject005, one in subject013). Visual review of the additions: subject004 is a
small contrast-filled vessel leaving the anterior wall at the superior mask cap
(plausible inferior mesenteric origin, but adjacent to the cap); subject017 and
subject023 are small anterolateral wall-adjacent structures with weak
evidence; subject025 sits beside an isolated mask island. The folded openings
in subject005/013 shared a traced lumen with a neighbouring daughter 4–8 mm
away. None of these can be resolved without expert references, so the variant
stays opt-in. Production predictions are unchanged. Once organizer references
arrive, score both frozen prediction sets with `score_references.py` and
promote only if the wall-parallel set is not worse on any reference case.

Commit `5c2b9a3` proposed the same geodesic endpoint rule *always on* with a
1.5 mm clearance and no hyperdense or path-folding guard, plus a
`shares_prefix` common-trunk merge. Measured on the same data before merging:

| Data | Production | Always-on geodesic (`5c2b9a3`) |
|---|---|---|
| Frozen 582743 | 26 / 1 / 6 | 26 / **4** / 6 (thrombus, vein/calcification) |
| Frozen 904117 | 27 / 0 / 4 | 27 / **1** / 4 (thrombus) |
| Wall-parallel family, 10 seeds | 10 / 0 / 10 | 17 / 0 / 3 |
| Real 25 cases, daughters | 151 | 163 (+13 small additions, 4 ostium shifts) |

The frozen false positives are mural-thrombus and wall-adjacent vein
structures accepted because a supported path of 5 mm exists along the wall;
they gain no true positives. The always-on rule was therefore not kept. The
`shares_prefix` common-trunk merge was kept: with it, frozen counts and all 25
production predictions are unchanged.

The photometric regression keeps anatomy/background fixed and gives daughters
65% of the parent/background contrast. The historical gate detects two origins
at parent 150 HU and zero at 550 HU; the selected detector detects both at each
level. This is a targeted synthetic regression, not a general invariance proof.
The absolute 30 HU floor, upper intensity bound, noise, calcification and
partial-volume sampling can still affect detection.
