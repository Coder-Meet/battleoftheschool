# Final submission and demo guide

## Start here

**Submit the deterministic strict detector with `native_contrast_scale=1.2`.**
These are already the defaults in `run.py`, `batch.py` and `explorer.py`.
Use no candidate model, review mode, spacing override or experimental flag.
The [final evaluation](FINAL_EVALUATION_RESULTS.md) documents the decision.

For the public entry, use the [ready-to-paste Devpost submission](DEVPOST_SUBMISSION.md).
For rehearsal, use the [complete presenter briefing and judge Q&A](PRESENTER_BRIEFING.md).
The README's [media pack](docs/media/README.md) contains cover art, authentic
screenshots and graphics with explicitly scoped accuracy/runtime evidence.

The final download bundles are on the repository's
[GitHub Releases page](https://github.com/Coder-Meet/battleoftheschool/releases).
Use the **Final hackathon submission** release for the submission ZIP and
updated presentation kit. The earlier September 12 deck is historical.

| Download | Purpose |
|---|---|
| `branchseed-final-submission.zip` | Current inference source, 25 prediction JSONs, three visual checks, built website and Windows x64 runtime wheels |
| `branchseed-final-presentation.zip` | Updated five-minute live-demo deck, editable PowerPoint, PDF, speaker script and CT evidence; no video |
| `branchseed-showcase.mp4` | Separate edited product showcase for Drive and sharing |
| `branchseed-film.mp4` | Unchanged original 60-second UI film, preserved separately |

The showcase uses recorded UI footage from an earlier revision. It is a
standalone product edit, not the five-minute talk or evidence of current
accuracy. **Present the Explorer live at slide 5; no video plays in the deck.**
The original film remains available. Upload the showcase to Drive or other
destinations as needed; organizer/Devpost uploads remain a team action.

## Fastest website launch: downloaded submission

Install **64-bit Python 3.13.3** before the event. Extract the submission ZIP
fully; do not run files inside Windows' ZIP viewer. In PowerShell, open the
extracted `source` folder:

```powershell
python --version
python -m venv .venv313
.\.venv313\Scripts\python.exe -m pip install --no-index --find-links ..\windows-wheelhouse -r requirements.txt
$env:OPENBLAS_NUM_THREADS="4"
$env:OMP_NUM_THREADS="4"
$env:MKL_NUM_THREADS="4"
$env:ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="4"
.\.venv313\Scripts\python.exe explorer.py --data-root "C:\path\to\data" --port 8000
```

Replace the data path with your real case directory. **On that same laptop**,
open **http://127.0.0.1:8000** in Chrome or Edge. Keep the terminal running.
The built website is included: no Node install, Vite server, API key, cloud
service or internet connection is needed to serve it.

The Windows wheels require x64 CPython 3.13; they cannot be installed on
macOS/Linux, Windows ARM or another Python minor version. Python itself and
the supplied scans are not included.

## Launch from a Git checkout

Run commands from the repository root. Pull the final `main` before setup.
Resolve the actual scans while connected; tiny Git LFS pointers are not CTs.

```bash
git pull --rebase
git lfs install
git lfs pull
```

### Windows / PowerShell

Confirm `python --version` reports 3.13.3. Prepare dependencies while online:

```powershell
python -m venv .venv313
.\.venv313\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.txt
npm --prefix web ci
npm --prefix web run build
$env:OPENBLAS_NUM_THREADS="4"
$env:OMP_NUM_THREADS="4"
$env:MKL_NUM_THREADS="4"
$env:ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="4"
.\.venv313\Scripts\python.exe explorer.py --port 8000
```

### macOS / Linux

Use [uv](https://docs.astral.sh/uv/getting-started/installation/) and Node
20.18.1/npm. These setup steps need package access:

```bash
uv python install 3.13.3
uv venv --python 3.13.3 .venv313
uv pip install --python .venv313/bin/python -r requirements.txt
npm --prefix web ci
npm --prefix web run build
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4 .venv313/bin/python explorer.py --port 8000
```

Open **http://127.0.0.1:8000 on the machine running the command**.
After setup, subsequent launches require only the last command.
The Python server serves both the frontend and API; a separate frontend
server is unnecessary for presenting.

For frontend development only, run Python as above plus
`npm --prefix web run dev` in another terminal. Open the URL Vite prints;
its `/api` proxy targets port 8000.

## Cases and new input pairs

The Explorer reads folders from its local data root; it has no browser upload
form. A folder name may contain letters, digits, underscores and hyphens.
Each case needs exactly one `orig*.nii` or `orig*.nii.gz` CT and exactly one
`mask*.nii` or `mask*.nii.gz` binary parent-aorta mask:

```text
data/
  subject001/
    orig1.nii
    mask1.nii
  demo_case/
    orig.nii.gz
    mask.nii.gz
```

Use an existing CT/mask pair with matching physical geometry. Do not rename
an unrelated segmentation to make it look like a parent mask. Use
`--data-root "/absolute/path/to/cases"` for a different location, then refresh
the page after adding a case.

1. Select a case in the left library and wait for analysis.
2. Inspect the aorta in 3D; select a candidate in the branch list.
3. Check the linked CT planes and traced path, then show the wall map.
4. Show the interior tour briefly if time allows.
5. Click **Export JSON** to download the prediction. The camera button exports
   a visual check. Browser downloads normally go to the laptop's Downloads folder.

Review labels are saved in that browser. **Export training reviews** is a
different output and does not change the challenge prediction. Do not use it
as the challenge submission.

Preload subject018 before the talk and keep subject001 as a backup. The
backend caches two cases; returning to an older case may require analysis
again. Do not promise a specific daughter count as anatomical truth.

## Exact challenge command

With the prepared environment activated:

```bash
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
```

If you did not activate the environment, replace `python` with
`.\.venv313\Scripts\python.exe` on Windows or `.venv313/bin/python` on
macOS/Linux. Quote paths containing spaces.

For a supplied example:

```bash
python run.py --image data/subject001/orig1.nii --aorta-mask data/subject001/mask1.nii --output prediction.json --case-id subject001
```

Coordinates are the original SimpleITK LPS physical millimetres. Each daughter
has an ID, parent ID, ostium, 5 mm seed, seed radius and unit direction.

The submission ZIP's `predictions/` folder contains only the 25 challenge
JSONs from this same strict configuration. Its diagnostics and measurements
are separate under `verification/`. Do not submit a fold-selected composite,
review pool, synthetic labels or the retrospective RF predictions.

## Presentation and four-speaker run of show

Extract the **whole final presentation ZIP** and open `index.html` in a
browser. Keep the `assets` directory beside it. For a Git checkout, the
generated current kit is under `outputs/live-presentation-kit/`; generated
media are distributed through Releases rather than committed to Git.

| File | Use |
|---|---|
| `index.html` | Recommended offline slide deck |
| `branchseed-editable.pptx` | Editable PowerPoint; install the two TTFs in `assets` for matching fonts |
| `branchseed-slides.pdf` | Eight-page visual fallback |
| `SPEAKER_SCRIPT.md` | Full timed narration with final reference results |
| `LIVE_DEMO_CUES.md` | Detailed 75-second Explorer sequence |

Deck controls: arrows/Space change slides, **F** fullscreen, **N** notes,
**A** timed rehearsal, Home/End first/last slide. The rehearsal clock does
not control the live app or play the video.

| Time | Presenter | Content |
|---|---|---|
| 0:00–1:15 | Speaker 1 | Problem, eligible origins and topology |
| 1:15–2:30 | Speaker 2 | Detector, physical geometry and measurements |
| 2:30–3:45 | Speaker 3 | Switch to the preloaded Explorer, inspect and export |
| 3:45–5:00 | Speaker 4 | Three CT checks, final evidence, runtime and limits |

Assign your names before presenting. Keep one person operating the laptop.
If live rendering fails, use the still poster and then the three CT checks.
The separate showcase is not part of the talk. Do not debug on stage or add
another minute after the five-minute talk.

## What to say about accuracy

The fixed strict detector scored **8 TP / 3 FP / 11 FN**, F1 **0.5333**, at our
local 3 mm ostium tolerance on 19 targets across five reused cases. The judge
approved the package for scoring; its annotations are AI-assisted and may
omit valid branches. This is development-reference agreement, not hidden-test
or complete clinical accuracy. Only three reference radii are known.

The RF hybrid's all-five F1 **0.6207** is retrospective. The case-separated
selection composite scored **0.4444** and is not one deployable algorithm.
The fixed strict detector remains the defensible submission. Synthetic
topology F1 **0.9684** is reported separately. No official weighted score was
available. Avoid converting any of these numbers into “percent accurate.”

The later guarded wall-recovery experiment reached F1 **0.5806** but worsened
count MAE from **2.0 to 2.2**. It preserved baseline matches across 80 synthetic
comparisons, but that does not establish better unseen-patient performance.
Strict is the accepted submission choice; recovery remains opt-in.

## Troubleshooting and the final laptop check

| Symptom | Action |
|---|---|
| Connection refused | Start `explorer.py`; keep the terminal open |
| Address already in use | Use `--port 8001`, then open that port locally |
| “Build the frontend” / blank 404 page | Run `npm --prefix web ci` and `npm --prefix web run build`; do not open the app's HTML directly |
| Cases unavailable | Run `git lfs pull` while connected; verify files contain scan bytes |
| Empty library | Check the `--data-root` and the exact CT/mask naming pattern |
| Missing Python module | Install requirements using the same interpreter used to launch the app |
| Geometry error | Use the matching CT and binary parent mask; do not change the header to suppress the error |
| Analysis failed | Read the error and terminal; try the preloaded backup case |
| 3D unavailable | Enable browser hardware acceleration; CT, wall map and JSON remain available |

Stop Python and any optional Vite server with **Ctrl+C** in their terminals.
Restarting clears the backend cache, so preload the demo again.

On the actual Windows laptop, disconnect external internet after setup, run
the CLI on a supplied case, open the website, analyze/select/export, and rehearse
the deck-to-app switch. Check the showcase separately if sharing it.
To time a CLI run in PowerShell:

```powershell
Measure-Command { .\.venv313\Scripts\python.exe run.py --image data\subject001\orig1.nii --aorta-mask data\subject001\mask1.nii --output prediction.json }
```

Linux measurements and Windows CI do not establish runtime or peak memory on
the organizer's actual four-core, 8 GB machine. That native laptop check is
still required.

## Reproduce engineering checks

Install `requirements-dev.txt` and `requirements-resources.txt` in the prepared
Python environment. Optional tree/CNN experiments have separate requirements.

```bash
python -m ruff check .
python -m mypy
python -m pytest -q -rs
npm --prefix web test
npm --prefix web run lint
npm --prefix web run typecheck
npm --prefix web run build
python final_eval_select.py verify
```

Keep four-thread environment variables set for replay. Full replay needs the
repository's tracked evaluation evidence; the compact submission ZIP is
intended for inference/demo and links back to that audit in Git.
