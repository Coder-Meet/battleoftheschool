# Branchseed — Toralis Labs Challenge (Battle of the Schools)

Team repo for the Toralis Labs track: detect every artery that directly
branches off the abdominal aorta from a CT volume + aorta mask, and report
each one as a separate daughter instance.

Full problem statement: see the [Branchseed challenge doc](https://docs.google.com/document/d/1oRb2R9pauvsC-9hDIfr23ojLx90JpCt0jVZjCD5l5Cg/edit).

**Current production default:** strict + review proposals, scored separately with the bundled synthetic logistic at **0.15**, then merged with strict-first priority. Fresh five-case result: **14 TP / 4 FP / 5 FN, F1 0.757**. CLI, batch and normal Explorer use this shared workflow. See [the current workflow and commands](PRODUCTION_WORKFLOW.md) and [the audit/next-step plan](CURRENT_E2E_REVIEW.md). Use `--pipeline strict` to reproduce the earlier classical default; annotation/review mode stays unfiltered.

## Working agreement

New teammates: start with the [setup, backend API and training handoff guide](TEAMMATE_GUIDE.md).
For a complete detector and ML walkthrough, read the [algorithm review for Steven](STEVEN_ALGORITHM_REVIEW.md).

To audit Steven's frozen candidate reviews and compare them with current proposals:

```bash
python review_analytics.py --tree-model labels/research/current-source-v1/models/gradient_boosting-base.json --output-dir outputs/review-analytics
```

Use a fresh output directory and the prepared `data/subject*` cases. This writes
CSV scores, exact candidate-identity coverage, charts, input/source hashes and
untouched strict/review-union predictions. It does not train or enable a model,
transfer labels by branch number, or measure accuracy against complete references.
The frozen findings and review priorities are in [Steven's label analytics](STEVEN_LABEL_ANALYTICS.md).

Research: see the [ML implementation plan](RESEARCH_IMPLEMENTATION.md) and the
[three additional paper experiments and measured limitations](ADDITIONAL_PAPERS.md).
The [research commands below](#optional-ml-research) preserve the submission CLI
and keep model scores outside the daughter JSON schema.

The [latest judge clarifications](SUBMISSION_AUDIT.md#latest-judge-clarifications)
specify 2 mm minimum **origin diameter** (confirmed directly with the judge),
one opening for a common trunk, and two openings for a returning vessel.
Discovery/count accuracy remains the priority within the four-core, 8 GB,
offline Windows limits. Do not equate the seed-radius CLI setting with origin size.
`--minimum-origin-diameter-mm` defaults to 2; zero reproduces the earlier
eligibility policy. The estimate uses a cross-section 2 mm along the proximal
path, local half-maximum contrast and one native voxel of diameter allowance.
Unresolved and borderline estimates remain visible in diagnostics instead of
being rejected as confidently undersized. This is an approximate origin
measurement, awaiting calibration against the organizer's references.

Presenters: use the [five-minute presentation guide](presentation/README.md)
and [live demo cues](presentation/LIVE_DEMO_CUES.md). The authoring source builds
an offline HTML deck, editable PowerPoint and PDF; the Explorer segment runs live.

**We are committing directly to `main`. There are no feature branches.**
Pull before you start working, commit small and often, push as soon as
something works. If you break `main`, fix it forward — don't force-push.

```bash
git pull
# ... do work ...
git add <files>
git commit -m "short description"
git pull --rebase   # pick up anyone else's commits
git push
```

Because everyone pushes to `main` directly, keep commits small and pull
often to minimize conflicts. If you hit a conflict, resolve it locally
before pushing — don't push broken code.

## What's in this repo

```
data/                   # 25 subjects, paired CT + aorta mask (tracked via Git LFS)
  subject001/
    orig1.nii           # CT volume
    mask1.nii           # binary aorta mask (1 = aorta, 0 = everything else)
  ...
run.py                  # entry point — see CLI contract below
detector.py             # physical-space intensity/vesselness/topology baseline
explorer.py             # local analysis API and production web server
web/                    # TypeScript + Three.js Aorta Explorer
batch.py                # run every case with diagnostics and a failure report
evaluate.py             # one-to-one matching against daughter reference JSON
tests/                  # synthetic geometry, topology, evaluation and API tests
synthetic.py             # five synthetic CT/instance-label/analytic-reference training cases
benchmark.py             # labeled-bundle detection and geometry evaluation
requirements.txt
.gitattributes          # routes *.nii / *.nii.gz through Git LFS
```

The `.nii` files are large (the whole `data/` folder is ~2 GB). They are
tracked with **Git LFS**, not raw git. See setup below.

## Setup

### Windows submission quick start

The organizer confirmed **Windows, four CPU cores, 8 GB RAM, no GPU and no
internet during evaluation**. Install 64-bit Python **3.13.3** before setup
(confirm the organizer's CPU architecture; the prepared wheel bundle targets
Windows x64). In a terminal where `python --version` reports that interpreter,
the single online dependency-setup command is:

```powershell
python -m pip install --only-binary=:all: -r requirements.txt
```

The required run command works without a shell-specific interpreter path:

```powershell
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
```

To prepare a fresh Windows machine for installation without internet, run
`python -m pip download --only-binary=:all: -r requirements.txt --dest wheelhouse`
on a matching Windows/Python machine while online, then copy the wheels and
source. Install offline with:

```powershell
python -m pip install --no-index --find-links wheelhouse -r requirements.txt
```

Do not copy a Linux virtual environment onto Windows. SimpleITK defaults to
four threads in the CLI. In PowerShell, cap numerical-library thread pools
before execution with `$env:OPENBLAS_NUM_THREADS="4"` and
`$env:OMP_NUM_THREADS="4"`. The Windows CI job checks native installation and
Python tests; the Linux network-denied test separately checks offline execution.
Neither replaces a timed run on the organizer's actual 8 GB machine.

### Linux/macOS development setup

Submission quick start (Git checkout, `uv` and package access already available):

```bash
uv venv --python 3.13.3 .venv313 && uv pip install --python .venv313/bin/python -r requirements.txt
```

Then run `.venv313/bin/python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json`.
Prepare this environment before the offline evaluation. Node and the supplied
development scans are only needed for the Explorer/demo, not new-case inference.
The [robustness protocol](ROBUSTNESS_PROTOCOL.md) records the frozen detector selection.
The [submission audit](SUBMISSION_AUDIT.md) maps the deliverables to both challenge
guides and lists the remaining team actions.

1. Install Git LFS (one-time, per machine):
   ```bash
   brew install git-lfs      # macOS
   # or: https://git-lfs.com for other platforms
   git lfs install
   ```
2. Clone / pull as normal — LFS pointers resolve automatically once LFS is
   installed:
   ```bash
   git clone https://github.com/Coder-Meet/battleoftheschool.git
   cd battleoftheschool
   ```
3. Python environment (Python **3.13.3** is verified with the pinned packages):
   ```bash
   # Install uv from https://docs.astral.sh/uv/getting-started/installation/
   uv python install 3.13.3
   uv venv --python 3.13.3 .venv313
   uv pip install --python .venv313/bin/python -r requirements-dev.txt
   source .venv313/bin/activate
   ```
   The system Python 3.10 cannot install these NumPy pins. Use the environment
   above. `requirements.txt` is sufficient for inference without development tools.
4. Explorer frontend (Node **20.18.1** and npm are verified):
   ```bash
   npm --prefix web install
   npm --prefix web run build
   ```
   Dependencies and fonts are bundled locally; no network or API keys are needed
   during inference or when serving the built Explorer. If the volumes remain
   small Git LFS pointer files, run `git lfs pull`.

## Run

### Offline operation

Inference, candidate-model training and the Explorer run **without internet**.
There are no hosted models, inference APIs, runtime weight downloads, CDNs or
remote font requests. First install the pinned Python/frontend dependencies,
resolve Git LFS scans and build `web/dist` while connected. Preserve that
environment and local data before disconnecting; setup commands such as
`npm install` and `git lfs pull` require the relevant packages/data to be available.
The Explorer still needs its local Python server running; disconnecting the
internet must not block loopback/localhost.

```bash
npm --prefix web run build
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv313/bin/python explorer.py
```

On Linux, reproduce the network-denied algorithm/training regression check
with `strace` installed:

```bash
BRANCHSEED_NETWORK_BLOCKED=1 strace -f -e trace=%network \
  -e inject=%network:error=ENETUNREACH -o offline-network.log \
  .venv313/bin/python -m pytest -q tests/test_offline.py tests/test_detector.py \
  tests/test_input_validation.py tests/test_learning.py
```

Include `tests/test_synthetic.py tests/test_accuracy.py` in the traced test list
to verify synthetic ground-truth generation and the expanded accuracy regressions.
The sandbox test verifies sockets are actually blocked, and the trace applies to
child CLI processes too. API/server tests are excluded because they deliberately
need localhost networking. Bundled frontend assets are served by the local server.

### Labelling workflow: AI verdicts on detector candidates, then train the filter

**AI verdicts are provisional pseudo-labels, not expert ground truth.** A
`confirmed` status describes that reviewer's decision, not clinical validation.
Keep the `labeller` provenance, hold out patients, and do not report evaluation
against these decisions as real branch-detection accuracy. The default submission CLI loads the bundled synthetic logistic model. Candidate reviews are not loaded during inference.

There are no branch annotations for the 25 scans, so labels come from judging the
detector's own proposals against the CT. This is done by an AI reviewer reading
rendered evidence, not by a web page. `autolabel.py render` runs the loose
**review profile** of the detector merged with the strict result, saves each
case's candidate pool with its thirteen features, and writes one PNG per
candidate showing whole-aorta locators, ±2 mm slab views through the origin,
consecutive axial slices and the traced path:

```bash
python autolabel.py render --cases subject001      # or --all; output under outputs/autolabel/<case>/
```

The reviewer answers one question per PNG: does a bright tube leave the aorta
outline at the yellow dot, along the red path, for at least 5 mm? Verdicts go in
a small JSON file, `{"subject001": {"branch_001": "confirmed", "branch_002": "rejected"}}`,
and are recorded in the training schema with the candidate's features and a
fingerprint, so a candidate that later changes loses its stale label:

```bash
python autolabel.py apply --verdicts verdicts.json --labeller claude
python autolabel.py status
git add labels/reviews.json && git commit -m "Label subject001" && git pull --rebase && git push
```

Candidates the reviewer cannot decide are left out rather than guessed. Then
hold out five patients and train:

```bash
python learning.py split --reviews labels/reviews.json --output labels/split.json \
  --test subject005 subject013 subject018 subject021 subject025
python learning.py train --reviews labels/reviews.json --split labels/split.json \
  --model labels/candidate-model.json --report labels/model-report.json
```

The production CLI now runs both strict and review profiles, scores before merging, and uses the bundled filter at 0.15. `--pipeline strict` restores the previous single-profile workflow; `--candidate-model` selects a custom logistic. Each
candidate carries thirteen features: six geometric ones and seven context ones
(brightness relative to the aorta, distance to bone, angle against the aortic
axis, position along the aorta, native spacing, wall-connector gap, contact
volume). External datasets that could supply independent truth are listed in
[EXTERNAL_DATA_SOURCES.md](EXTERNAL_DATA_SOURCES.md).

### Scoring the pipeline end to end against the labels

Confirmed labels can be turned into reference JSONs and every case run through
the detector with and without the filter, so the two outputs are scored the
same way using our local matching proxy. The official evaluator and its overlap
allowances are still needed to establish organizer scoring:

```bash
python labels_to_references.py --cases $(ls data | grep subject) --output-dir labels/pseudo_references
python compare_e2e.py --candidate-model labels/candidate-model.json \
  --references labels/pseudo_references --split labels/split.json --output-dir outputs/e2e
```

`compare_e2e.py` writes `outputs/e2e/plain/<case>.json` and
`outputs/e2e/filtered/<case>.json`, then pools precision, recall and F1 per split
partition. `--profile strict|review|pool` picks the proposal rules: `strict` is
the submission default, `review` the loose labelling rules, `pool` their union. Only the **test** partition is a held-out estimate; train and
validation labels fitted the weights. These references carry the reviewer's
judgement, not organiser truth: a branch the detector never proposed cannot be
in them, and their geometry is the detector's own, so ostium and radius errors
against them are lower bounds. When the organiser's references arrive, point
`--references` at them instead and nothing else changes. Two prediction files
can also be compared directly with `evaluate.py --prediction a.json --reference b.json`.

### Human review and optional ML

The Explorer now supports **Confirm**, **Reject**, **Clear**, **Next unreviewed**,
review filters, and **Export training reviews**. Reviews persist in this browser
and export across cases. Export regularly: clearing browser storage removes them.
Review labels do not modify the raw challenge prediction. A changed candidate's
measurements invalidate its prior review in the UI.

AI candidate reviews and optional trained weights are included; complete expert
daughter references are not yet available. `learning.py`
trains an L2-regularized logistic candidate classifier using CPU NumPy/SciPy
once an expert has reviewed candidates. It consumes thirteen physical/evidence/context features,
not CT images. It cannot discover vessels missed by the classical proposal stage.

Review at least six independent cases, including both true and false candidates
in each partition. Export the review file, then freeze the case split **before**
training. Group repeat scans of a patient in the same partition; the automatic
split assumes each case is a different patient.

```bash
python learning.py split --reviews branchseed-reviews.json --output outputs/split.json --seed 42
python learning.py train --reviews branchseed-reviews.json --split outputs/split.json \
  --model outputs/candidate-model.json --report outputs/model-report.json
python run.py --image data/subject001/orig1.nii --aorta-mask data/subject001/mask1.nii \
  --candidate-model outputs/candidate-model.json --output outputs/prediction.json \
  --diagnostics outputs/diagnostics.json
```

Scaling and weights use training cases only; the decision threshold uses validation
F1; test cases are evaluated afterward. The report compares against keeping every
candidate. Split overlap, conflicting reviews, missing classes, and invalid features
fail explicitly. Test results must not be used for parameter selection. Candidate
metrics are **not whole-vessel detection recall**; use complete independent daughter
references with `evaluate.py` for that. Synthetic regression fixtures only test the
training machinery, not real-scan accuracy. Custom models are opt-in; the bundled synthetic logistic is enabled by default. Normal Explorer shows production results; `--review-mode` exposes the unfiltered pool for annotation.

`--spacing-mm` selects the detector's working resolution (default 1.0 mm);
finer grids increase CPU/memory requirements. Multi-label masks and sheared
geometry are rejected before measurement.

Frontend review tests run with `npm --prefix web test`. GitHub Actions checks
Python tests/lint/types and frontend tests/lint/build on every push.

The task spec requires the program to support this CLI contract:

```bash
python run.py --image data/subject001/orig1.nii --aorta-mask data/subject001/mask1.nii --output prediction.json
```

Optional `--diagnostics outputs/subject001-diagnostics.json` exports candidate
counts, rejection reasons, proximal paths, timings, and the adaptive blood model.
`--threads 4` controls SimpleITK threads; `--minimum-radius-mm 0.7` is the default.
To bound threaded numerical libraries as well, set `OPENBLAS_NUM_THREADS=4` and
`OMP_NUM_THREADS=4` before starting Python.

Output is one JSON file per case with the parent aorta and a list of
daughter branch instances (ostium, seed point, radius, direction). See the
challenge doc for the exact schema and definitions (ostium centre, daughter
seed, daughter radius, etc).

## Aorta Explorer

After building the frontend, run:

```bash
python explorer.py --port 8000
```

Open **http://127.0.0.1:8000** in your browser. Select a local case to run the
detector and generate its surface. The app includes:

- An orbitable parent-aorta surface, colored proximal branch paths, ostia,
  direction arrows, labels, and layer controls.
- A branch inspector with physical coordinates, 5 mm seed offset, radius,
  and a clearly labeled heuristic evidence score.
- Linked acquisition-plane CT views with crosshairs, window presets, scrolling,
  and parent-mask overlays.
- An approximate unwrapped origin map and an inside-aorta camera tour.
- Challenge-format JSON export.

Processing uses one worker with a bounded queue and a two-case memory cache.
The default server binds to loopback. `--host 0.0.0.0` enables a remote development
preview; this is a local research server without user accounts or authentication.
Use `--data-root PATH` for another directory of paired case folders.

For frontend development, keep the Python server running and run
`npm --prefix web run dev`; Vite proxies `/api` to port 8000.

## Batch processing, visual checks, and evaluation

### Synthetic ground truth for training

```bash
python synthetic.py --output-dir outputs/ground-truth-five --seed 2026
python benchmark.py --data-root outputs/ground-truth-five \
  --output outputs/synthetic-metrics.json --tolerance-mm 3
```

This produces five synthetic CTs with parent masks, daughter instance masks,
analytic physical-coordinate references and provenance/file hashes. Labels
come from independently defined vessel geometry, never from detector predictions.
All five are training/development data; they are **not annotations of the real
scans** or an independent clinical test set. The generator refuses to overwrite
an existing directory. See [the handoff guide](TEAMMATE_GUIDE.md#4-generate-the-five-ground-truth-training-cases)
for label definitions, training use and limitations.

The detector now admits small physically supported wall contacts, follows the
proximal vessel tangent back to the wall for oblique ostia, and measures radius
on a perpendicular intensity cross-section where a bounded lumen is observable.
It retains distance-transform radius as a fallback. These changes are validated
on synthetic geometry; they do not establish real-scan accuracy.

### Real-case processing

```bash
python batch.py --data-root data --output-dir predictions
python batch.py --cases subject001 subject002 subject003 --output-dir predictions
python viz.py --subject subject001 --pred predictions/subject001.json --output outputs/subject001.png
python viz.py --subject subject002 --pred predictions/subject002.json --output outputs/subject002.png
python viz.py --subject subject003 --pred predictions/subject003.json --output outputs/subject003.png
```

Batch processing emits per-case challenge JSON, a diagnostics subdirectory, and
`batch_report.json`. It continues after a failed case and exits nonzero if any
case failed. Visual checks show the supplied aorta mask, predicted ostia, and
direction arrows over CT projections.

When **actual daughter reference annotations** are available in the same JSON
schema, local evaluation is:

```bash
python evaluate.py --prediction predictions/subject001.json \
  --reference labels/subject001.json --tolerance-mm 5 --output outputs/metrics.json
```

This evaluator uses maximum-cardinality one-to-one ostium matching followed by
minimum total distance. It reports precision, recall, F1, and matched ostium,
seed, radius, and direction errors. Undefined metrics are JSON `null`. This is
a transparent local metric, not a claim to reproduce the organizer's scoring.
Do not evaluate against the detector's own outputs as if they were ground truth.

When organizer reference files arrive, freeze the current predictions first
(`python batch.py --output-dir predictions/frozen-<commit>`; committed sets for
the current revision live in [`frozen/`](frozen/README.md)), then score the
whole set in one command regardless of their exact field names:

```bash
python score_references.py --references organizer-refs/ \
  --predictions predictions/frozen-<commit> --data-root data \
  --output outputs/reference-score.json
```

While waiting for labels, [prepare five difficult cases for expert review](ANNOTATION_GUIDE.md).
`prepare_annotations.py` builds blinded native-slice surveys, proposed-opening
CT sheets and editable review worksheets from frozen predictions. It leaves all
labels and reference counts unconfirmed; the scorer rejects these packets as
ground truth. The guide also provides a reproducible five-case **hard synthetic**
bundle with analytic labels for development.

It accepts one JSON per case, a list of cases, or a directory; maps common
aliases (`ostium`/`origin`, `branches`/`daughters`, `diameter_mm`, case ids
like `18` or `orig18`); converts voxel indices to millimetres with the case
image when `coordinate_space` says so; derives a missing seed or direction;
reports TP/FP/FN at 2, 3 and 5 mm; and warns when an LPS/RAS mirror would match
far better. Every normalization it applied is listed in the report JSON, so
check that list before quoting any number.

## Verification

```bash
python -m pytest -q
ruff check .
mypy
npm --prefix web run lint
npm --prefix web run typecheck
npm --prefix web run build
npm --prefix web audit --audit-level=moderate
```

The regression suite covers rotated and anisotropic geometry, separate nearby
openings, a downstream vessel sharing one origin, crop caps, disconnected
vessels, empty masks, skeleton junctions, 5 mm arc interpolation, CLI output,
gzip/LFS handling, reference matching, and the HTTP API.

## Current scope and limitations

The production workflow combines classical proposals with a trained synthetic logistic filter. There are 25 CT/aorta pairs and 19 released reference targets across five development cases. The current workflow has local 3 mm F1 0.757 on those reused cases; independent accuracy remains unmeasured. Scores are uncalibrated. See `PRODUCTION_WORKFLOW.md` for current defaults and limitations.

The detector resamples a tight ROI to 1 mm, estimates blood intensity from the
parent, enhances tubular structures at multiple scales, finds wall-contact
components, and traces supported outward paths. It suppresses parent end caps,
deduplicates shared origins, and truncates paths at estimated skeleton junctions
or 10 mm. The seed lies 5 mm along the estimated proximal path. All output
coordinates are transformed through the input geometry.

Intensity support uses a fraction of measured parent/background contrast,
adjusted for the largest **native** voxel spacing and minimum supported radius.
The conservative scale was selected before new frozen procedural evaluation;
see [the robustness protocol](ROBUSTNESS_PROTOCOL.md) for the comparison and
remaining failures. Background-overlap diagnostics measure tissue heterogeneity,
not scanner noise or a clinically validated CNR.

Topology and bifurcation locations are estimates from image evidence. Small or
short vessels, touching openings, calcification, veins, and low-contrast scans
can still cause misses or false detections. A radius is a local distance-transform
fallback or a local intensity cross-section estimate, not a validated lumen measurement. The colored tubes are proximal
path illustrations, not daughter segmentations. The wall map and interior tour
use an approximate parent curve; disconnected masks may interrupt that curve.
Review the CT evidence before interpreting any result.

## Constraints to keep in mind while building

- No case-specific edits — the same code must run on all cases and the
  hidden eval set.
- No GPU. Target environment: 4 CPU cores, 8 GB RAM, no internet.
- Runtime target: ~60s/case average.
- Coordinates must be physical mm (use `SimpleITK.TransformIndexToPhysicalPoint`),
  not voxel indices.
- Don't try to name vessels anatomically — just `branch_001`, `branch_002`, ...
- Provide a visual check (aorta mask + detected ostia + direction arrows)
  for at least 3 cases before submission.

## Submission deadline

Sunday 11:00 AM. Have the repo pushed and a working demo before then.

## Optional ML research

Current-source replay, exact checks, rejected CNN/blend results and remaining
gates: [research integration results](labels/research/current-source-v1/RESULTS.md).

`run.py` now defaults to the versioned synthetic logistic score-before-merge workflow. Tree/CNN research models remain disabled; `--pipeline strict --candidate-model PATH` preserves the legacy custom-filter behavior. `research_run.py` scores a variable number of
classical proposals using a source-compatible tree, optional CPU ONNX model,
or frozen probability blend. Synthetic performance and pseudo-label agreement
are not clinical accuracy. Complete organizer references and measurements on
the organizer's Windows hardware are still promotion gates.

Install only the extras you need, alongside `requirements.txt`:

```powershell
python -m pip install -r requirements-trees.txt
python -m pip install -r requirements-cnn-inference.txt
python -m pip install -r requirements-resources.txt
```

Tree *inference* needs no scikit-learn. Tree training uses the first extra;
CNN training/export additionally uses `requirements-cnn-training.txt` (CPU
Torch/ONNX). These packages are never added to base submission requirements.

The safe scoring command retains every daughter and writes scores separately:

```powershell
python research_run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --tree-model current-tree.json --mode scores-only --proposals strict --output scored.json --diagnostics scores.json --proposals-output unfiltered.json
```

Use `--proposals review-union` for the existing broad review pool plus any
strict-only detections. It uses the unchanged strict 2 mm **origin diameter** /
0.7 mm seed-radius policy and the existing review origin-diameter override
of zero. Only explicit `--mode filter` removes below-threshold daughters;
the original proposals are always retained. All three destinations must be new.

Replace `--tree-model` with `--onnx-model model.onnx` for CNN-only scores.
For combined scores, supply `--tree-model`, `--onnx-model`, and
`--blend-model blend.json` together. The blend must match both model hashes,
candidate identities, physical preprocessing, sources and held-out splits.
No inference flag changes a frozen threshold. All models remain research-only.

Physical features are never zero-filled. If a model is incompatible or an
extraction fails, exit status is 1 and diagnostics list every unscored candidate
with the error. Scores-only still writes the original daughters. Filter mode
withholds the filtered prediction, leaving preserved proposals and the error
report. Check exit status before consuming results.

Archived cohort/CNN models retain their original source/configuration boundary.
They cannot be passed silently to current-source image inference. New tree
training binds detector/extractor/feature sources and physical preprocessing;
models without this contract require their frozen workflow or fresh extraction
and training. Do not rewrite archived metadata to bypass the check.

### Reproduce synthetic research with one command

After installing tree requirements (plus CNN training extras only if requested):

```powershell
python research_reproduce.py --root outputs/current-source-replay --workers 2
```

This creates a new plan and source snapshot, generates all 210 CTs, exports
development candidates and numeric patch caches, freezes four trees/logistic
and validation selection, then exports/evaluates the test partition. The
seed groups are the **already exposed** historical cohort: this is a
current-source replay, not a new independent test or official 2 mm eligibility
validation. `--cnn` runs the optional retrospective synthetic CNN/blend only
if the unchanged E1 gate permits it. A failed gate is recorded without retuning.
Never overwrite a frozen root. Each stage is also available through
`research_corpus.py` for audited resumption; its source/dependency checks must
still match the original plan.

### Organizer-reference comparison with one command

Prepare an evaluation manifest with prediction directories and explicit
reference provenance/completeness, then run:

```powershell
python research_validation.py --manifest comparison-input.json --output comparison-report.json
```

The minimal manifest shape is:

```json
{
  "schema_version": 1,
  "references": "organizer-references",
  "reference_provenance": "complete_expert",
  "references_complete": true,
  "baseline": "strict-predictions",
  "variants": {"tree": "tree-predictions"},
  "proposals": {"tree": "unfiltered-predictions"}
}
```

Set `complete_expert`/`true` only after verifying completeness independently.
For existing candidate-derived labels use `candidate_pseudo` and `false`;
they cannot support promotion or measure missed unproposed branches.
Add `histories` for `baseline` and every variant, with `training_cases`,
`tuning_cases`, `inspected_cases`, `pseudo_trained_cases`, split/group IDs,
and `history_complete`. Preserve all 25 previously inspected/pseudo-trained
supplied subjects, including matching renamed/grouped cases. A complete list
of current training cases alone does not establish historical independence.
Missing histories or Windows resource evidence conservatively reject promotion.

The report includes one-to-one discovery counts, proposal misses, matched
geometry and seed-to-polyline errors when references supply geometry,
2/3/5 mm tolerance sensitivity, seed-group bootstrap intervals, overlap and
promotion reasons. A valid comparison exits 0 even when promotion is rejected;
case errors exit 2; configuration/I/O failures exit 1. Do not treat exit 0 as
permission to activate a model.

### Portable resource command and Windows smoke

Set numerical thread variables **before** Python starts. On PowerShell:

```powershell
$env:OPENBLAS_NUM_THREADS="4"; $env:OMP_NUM_THREADS="4"; $env:MKL_NUM_THREADS="4"
python research_resources.py --cores 4 --case-id subject001 --output resources.json -- python run.py --image data/subject001/orig1.nii --aorta-mask data/subject001/mask1.nii --output prediction.json --threads 4
```

The optional psutil wrapper assigns at most four inherited CPU-affinity cores,
caps numerical/ITK thread environments, and samples the sum of RSS for the
wrapper and all live descendants every 10 ms. Shared pages can be counted
twice and short peaks can be missed; reports state this limitation. Wall time
includes process startup. It does not impose an 8 GB allocation limit or
verify GPU use. Normal runs report network status unknown; external
network-syscall denial is separately tested on Linux.

The manual GitHub Actions input `optional_research=true` runs a bounded,
20-minute Python 3.13.3 Windows CPU tree/CNN/export/resource smoke job. It
downloads binary wheels online and installs them from a local wheelhouse.
The ordinary base jobs remain free of optional Torch dependencies. Successful
Linux wheel downloads or tests do not prove native Windows execution,
Windows network denial, or performance on the organizer's machine.
