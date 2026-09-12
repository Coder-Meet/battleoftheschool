# Branchseed teammate handoff

## What is committed

The backend and frontend are in this repository's `main` branch:

| Entry point | Purpose |
|---|---|
| `run.py`, `detector.py` | CPU detector and challenge JSON |
| `explorer.py` | Local HTTP API, job queue, CT buffers and frontend serving |
| `web/` | Explorer source, local fonts and locked npm dependencies |
| `synthetic.py` | Five reproducible synthetic CT/ground-truth training cases |
| `benchmark.py`, `evaluate.py` | One-to-one reference matching and geometric errors |
| `learning.py` | Optional classifier trained from human candidate reviews |
| `tests/`, `.github/workflows/checks.yml` | Regression and offline checks |

Generated volumes, predictions, local environments and `web/dist` are not
committed. Build/download these using the commands below. No API keys or app
login are required.

## 1. Start on a teammate's machine

Prerequisites: Git, Git LFS, Node **20.18.1**, npm and **uv 0.7.9**.
Install uv from <https://docs.astral.sh/uv/getting-started/installation/>.
The commands below use a POSIX shell (Linux/macOS); Windows users can use WSL.

```bash
git clone https://github.com/Coder-Meet/battleoftheschool.git
cd battleoftheschool
git lfs install
git lfs pull
uv python install 3.13.3
uv venv --python 3.13.3 .venv313
uv pip install --python .venv313/bin/python -r requirements-dev.txt
npm --prefix web ci
npm --prefix web run build
```

For an existing clone, use `git pull` instead of cloning. The 25 original
CT/mask pairs live under `data/subject001` through `data/subject025`.
Do not substitute a tiny Git LFS pointer file for a CT volume.

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  .venv313/bin/python explorer.py --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000** on that same machine. The Python server serves both
the frontend and API. It must remain running. The Devin preview is temporary;
it is not a permanent production deployment.

When pulling changes, rebuild frontend assets if `web/` changed, and restart
Python if backend/detector code changed. Restarting clears the in-memory
analysis cache; reload the browser to use the new assets.

## 2. Run the detector without the website

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv313/bin/python run.py \
  --image data/subject001/orig1.nii \
  --aorta-mask data/subject001/mask1.nii \
  --output outputs/subject001.json \
  --diagnostics outputs/subject001-diagnostics.json
```

Outputs use physical **LPS millimeters**, not voxel indices. Each direct
daughter has an ostium, a seed 5 mm along the proximal path, a radius and a unit
direction. The detector traces up to 10 mm or an estimated downstream
bifurcation. CT and binary aorta masks must have matching physical geometry.

`--spacing-mm 0.8` selects a finer working grid; it costs more CPU/memory and
must be evaluated rather than assumed more accurate. `--candidate-model`
applies an explicitly supplied reviewed-candidate model; it is off by default.

## 3. Use the backend from another program

Base URL: `http://127.0.0.1:8000`.

| Request | Response |
|---|---|
| `GET /api/health` | Health JSON |
| `GET /api/cases` | Available local cases |
| `POST /api/cases/subject001/analyze` | Start/queue analysis; HTTP 202 |
| `GET /api/cases/subject001/status` | `idle`, `queued`, `running`, `ready` or `failed` |
| `GET /api/cases/subject001` | Geometry, mesh, branches and diagnostics after `ready` |
| `GET /api/cases/subject001/ct` | Little-endian signed int16 CT buffer |
| `GET /api/cases/subject001/mask` | uint8 binary parent-mask buffer |
| `GET /api/cases/subject001/prediction` | Challenge-format JSON after `ready` |

```bash
curl http://127.0.0.1:8000/api/health
curl -X POST http://127.0.0.1:8000/api/cases/subject001/analyze
curl http://127.0.0.1:8000/api/cases/subject001/status
# Once status is ready:
curl http://127.0.0.1:8000/api/cases/subject001/prediction
```

CT/mask buffers use x-fastest storage; reshape as `(z, y, x)` using metadata
`size_xyz`. Use `origin_xyz` and `basis` to transform coordinates. Do not treat
the native scan grid as the resampled review grid.

There is one analysis worker, at most three queued/running jobs and a two-case
cache. Fetch results before eviction. A full queue returns an actionable error.
The local server has no user accounts; binding to `0.0.0.0` makes it reachable
on available network interfaces and is intended for a controlled development demo.

## 4. Generate the five ground-truth training cases

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv313/bin/python synthetic.py \
  --output-dir outputs/ground-truth-five --seed 2026
```

Use a new output directory: the generator refuses to overwrite existing labels.

Each case contains:

- `orig.nii.gz`: synthetic noisy CT.
- `mask.nii.gz`: binary parent-aorta mask.
- `instances.nii.gz`: background/distractors = 0, parent = 1, daughter instances = 2+.
- `reference.json`: analytic ostia, 5 mm seeds, directions and radii.
- `provenance.json`: generator seed, geometry, label mapping and limitations.

The manifest contains file hashes. Cases cover radial branches, oblique
branches, small low-contrast vessels with anisotropic acquisition, a common
trunk, and nearby independent openings with distractors. Distal children of a
common trunk retain their direct parent's instance label.

**These labels come from the geometry used to generate the images, not from
detector predictions. They are synthetic training/development ground truth,
not expert annotations of the 25 real scans.** Noise, sampling and blur
deliberately make CT boundaries differ from the continuous analytic surface.

For a future image model, use CT and parent mask as inputs. A binary daughter
target is `instances > 1`; instance methods can use the individual labels.
IDs are per-case instances, not stable anatomical vessel names. All five
provided cases are assigned to training/development. Reserve independently
annotated real patients for validation/test and keep repeat scans together.
Five simplified phantoms cannot establish real-world generalization.

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv313/bin/python benchmark.py \
  --data-root outputs/ground-truth-five \
  --output outputs/synthetic-metrics.json --tolerance-mm 3
```

This measures one-to-one ostium matches, false positives/negatives and matched
ostium, seed, radius and direction errors. It is a local metric, not verified
organizer scoring. Do not call a result on these training cases hidden-test
or clinical accuracy. The Explorer can open them with
`--data-root outputs/ground-truth-five`.

## 5. Human reviews and the existing candidate classifier

In the Explorer, select a branch, inspect its CT evidence, then Confirm/Reject.
Use Next unreviewed and filters to work through cases. Export training reviews
regularly: labels live in that browser's local storage. Changed candidates
invalidate old reviews. Raw prediction export stays separate from reviews.

The existing `learning.py` consumes reviewed candidate features, **not**
segmentation volumes. It needs positive and negative human reviews and at least
six cases for its automatic three-way split:

```bash
.venv313/bin/python learning.py split --reviews branchseed-reviews.json \
  --output outputs/split.json --seed 42
.venv313/bin/python learning.py train --reviews branchseed-reviews.json \
  --split outputs/split.json --model outputs/candidate-model.json \
  --report outputs/model-report.json
```

Freeze case/patient partitions before fitting. Scaling uses training data;
threshold selection uses validation; test data is evaluated afterward.
This ranker cannot recover daughters the proposal stage missed. Do not
relabel algorithm outputs as human truth or mix synthetic labels with human
reviews without explicit provenance.

For real ground truth, a qualified reviewer needs to independently inspect
all direct origins—including missed ones—resolve common trunks, record
physical measurements and adjudicate uncertain cases. Merely confirming
displayed candidates does not establish complete detection recall.

## 6. Offline checks and regression tests

Prepare dependencies, fonts, Git LFS volumes and frontend build while connected.
After preparation, inference, local training, generation and the Explorer
do not need internet. Keep loopback networking available for the local server.

```bash
.venv313/bin/ruff check .
.venv313/bin/mypy
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv313/bin/python -m pytest -q
npm --prefix web run lint
npm --prefix web test
npm --prefix web run build
```

Linux network-denied algorithm/training checks (requires `strace`):

```bash
BRANCHSEED_NETWORK_BLOCKED=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
strace -f -e trace=%network -e inject=%network:error=ENETUNREACH \
  -o offline-network.log .venv313/bin/python -m pytest -q \
  tests/test_offline.py tests/test_detector.py tests/test_input_validation.py \
  tests/test_learning.py tests/test_synthetic.py tests/test_accuracy.py
```

The socket-sandbox test is intentionally skipped in ordinary pytest and must
pass in this traced run. Server/API tests need localhost and are checked
separately. Real-case batch processing is a runtime smoke test, not accuracy:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv313/bin/python batch.py \
  --data-root data --output-dir outputs/batch
```

## 7. Collaborate without breaking the demo

The team agreement is direct development on `main`. Pull before starting,
keep commits small, run the relevant checks and push after they pass. Never
force-push or reset a teammate's work. Coordinate ownership of shared files.

```bash
git pull
# Make and test changes.
git add <specific-files>
git commit -m "Describe the working change"
git pull --rebase
git push
```

Do not commit environments, `node_modules`, generated outputs, model artifacts
or screenshots by accident. Keep reviewed labels/model versions in an agreed
versioned data location, and retain their provenance and frozen splits.

## 8. Troubleshooting

| Symptom | Action |
|---|---|
| NumPy installation fails with system Python | Use the verified Python 3.13.3 virtual environment. |
| Scan is a small text file / LFS-pointer error | Run `git lfs pull` with repository access. |
| Website says build the frontend / 404 at `/` | Run `npm --prefix web ci` and `npm --prefix web run build`. |
| Old code/results after pulling | Restart Python, rebuild changed frontend and reload the browser. |
| Incomplete CT/mask download or timeout | Keep server running; use Try again. Check server output. |
| Port 8000 already in use | Stop your old server normally or use `--port 8001`. |
| Empty parent mask | CLI supports empty output; Explorer needs a parent surface to render. |
| Geometry/binary-mask validation error | Supply aligned scalar CT and a binary mask; do not guess alignment. |
| No candidates | Inspect CT contrast, mask and diagnostics. Zero predictions do not prove no daughters. |

## Claims for the presentation

We can demonstrate offline CPU execution, exact synthetic ground truth,
regression-tested geometric improvements and interactive real-scan review.
Real precision/recall and clinical validity remain unmeasured. Do not describe
the detector as clinically validated, a trained image network or state of the
art without an independent comparison supporting those claims.
