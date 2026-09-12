# Branchseed — Toralis Labs Challenge (Battle of the Schools)

Team repo for the Toralis Labs track: detect every artery that directly
branches off the abdominal aorta from a CT volume + aorta mask, and report
each one as a separate daughter instance.

Full problem statement: see the [Branchseed challenge doc](https://docs.google.com/document/d/1oRb2R9pauvsC-9hDIfr23ojLx90JpCt0jVZjCD5l5Cg/edit).

## Working agreement

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
requirements.txt
.gitattributes          # routes *.nii / *.nii.gz through Git LFS
```

The `.nii` files are large (the whole `data/` folder is ~2 GB). They are
tracked with **Git LFS**, not raw git. See setup below.

## Setup

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

### Human review and optional ML

The Explorer now supports **Confirm**, **Reject**, **Clear**, **Next unreviewed**,
review filters, and **Export training reviews**. Reviews persist in this browser
and export across cases. Export regularly: clearing browser storage removes them.
Review labels do not modify the raw challenge prediction. A changed candidate's
measurements invalidate its prior review in the UI.

No real daughter annotations or trained weights are included. `learning.py`
trains an L2-regularized logistic candidate classifier using CPU NumPy/SciPy
once an expert has reviewed candidates. It consumes six physical/evidence features,
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
training machinery, not real-scan accuracy. The model is opt-in at the CLI; the
Explorer continues to show all classical candidates for unbiased human review.

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

This is an executable **classical research baseline**, not a trained ML model.
There are 25 CT/aorta pairs and no daughter labels in this repository. Precision
and recall remain unmeasured; the evidence score is not a calibrated probability.
Preserve a held-out patient split once annotations exist before fitting a ranker.

The detector resamples a tight ROI to 1 mm, estimates blood intensity from the
parent, enhances tubular structures at multiple scales, finds wall-contact
components, and traces supported outward paths. It suppresses parent end caps,
deduplicates shared origins, and truncates paths at estimated skeleton junctions
or 10 mm. The seed lies 5 mm along the estimated proximal path. All output
coordinates are transformed through the input geometry.

Topology and bifurcation locations are estimates from image evidence. Small or
short vessels, touching openings, calcification, veins, and low-contrast scans
can still cause misses or false detections. A radius is a local distance-transform
estimate, not a validated lumen measurement. The colored tubes are proximal
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
