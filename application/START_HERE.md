# Branchseed — standalone judge application

**Algorithm: score-before-merge fusion.** This is the selected merged detector,
with native contrast scale 0.9, bundled logistic scoring at 0.15 and strict-first
merging within 3 mm. Run the commands below without additional model flags.
The model is included and loads locally; nothing is trained or downloaded during
inference.

This package has no website, Node/npm, visualization server, matplotlib,
scikit-learn, PyTorch, ONNX, development tools or CT dataset. The required numeric
libraries are SimpleITK, NumPy, SciPy and scikit-image, plus their six pinned
runtime dependencies. Some small optional helper functions remain because the
unchanged inference modules import them.

## Windows x64: install completely offline

Install **64-bit CPython 3.13.3** before going offline. Python itself is not
included. Use the **Windows x64 ZIP**, extract it fully and open PowerShell in
the extracted `branchseed-judge-fusion` directory containing `run.py`.
Do not execute files inside Windows' ZIP viewer.

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-index --only-binary=:all: --find-links windows-wheelhouse -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

These wheels require ordinary x64 CPython 3.13. They do not support Windows ARM,
the free-threaded interpreter, another Python minor version, macOS or Linux.

## Run one CT and parent-aorta mask

Set the thread limits **before starting Python**. Run cases sequentially.

```powershell
$env:OPENBLAS_NUM_THREADS="4"
$env:OMP_NUM_THREADS="4"
$env:MKL_NUM_THREADS="4"
$env:NUMEXPR_NUM_THREADS="4"
$env:ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="4"
.\.venv\Scripts\python.exe run.py --image "C:\cases\image.nii.gz" --aorta-mask "C:\cases\aorta_mask.nii.gz" --output "C:\results\prediction.json"
```

Replace the three paths with the organizer's input/output paths. The unchanged
challenge contract is:

```text
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
```

Optional `--diagnostics diagnostics.json` records the algorithm name, applied
threshold, model hash, timings, paths and decisions. A successful run exits 0.
Inputs must be a CT and aligned binary parent mask with matching physical
geometry. Output is challenge JSON with the case ID, parent and a variable
number of daughters, each with ostium, 5 mm seed, radius and direction in
physical LPS coordinates.

For the supplied directory layout, batch execution is also included:

```powershell
.\.venv\Scripts\python.exe batch.py --data-root "C:\cases" --output-dir "C:\results"
```

Each case subdirectory must contain one `orig*.nii`/`.nii.gz` and one
`mask*.nii`/`.nii.gz`. Batch processing is sequential. A Git LFS pointer is not a
scan; the organizer must provide the actual image files.

## Source-only ZIP / online setup

The source ZIP contains the same inference code and model but no wheels.
With Python 3.13.3, create a virtual environment and install its pinned
dependencies while connected:

```bash
python -m venv .venv
# Activate .venv for your platform, then:
python -m pip install --only-binary=:all: -r requirements.txt
python -m pip check
```

For offline Linux/macOS installation, first download matching wheels on that
platform:

```bash
python -m pip download --only-binary=:all: -r requirements.txt --dest wheelhouse
python -m pip install --no-index --only-binary=:all: --find-links wheelhouse -r requirements.txt
```

The included Windows wheelhouse cannot be reused on those platforms.

## Evidence and resource limits

On five reused, AI-assisted reference cases, the selected fusion workflow
previously reproduced **14 TP / 4 FP / 5 FN, precision 0.7778, recall 0.7368,
F1 0.7568 and count MAE 1.8** at 3 mm matching tolerance. These are development
results, not independent test accuracy or an official weighted score.

The separate 24-case synthetic regression returned 47 TP / 11 FP / 2 FN
(F1 0.8785), including four negative-control false positives. The earlier strict
detector had 46/0/3 on that cohort. Removing website dependencies does not fix
those false positives or change predictions.

Fusion makes two detector passes. A smaller download does not imply less CT
processing memory. Four library threads are configured above; this is not a
hard memory cap. Native organizer-Windows/four-core/8-GB acceptance remains a
team check. Consult the published package verification receipt for actual
measured runtime, memory and platform rather than reusing strict timings.

`MANIFEST.json` identifies the source commit, algorithm and SHA-256 of each
packaged file. Keep `models/production-v1/logistic.json` in place; missing or
modified bundled weights fail explicitly.

The earlier `branchseed-final-submission.zip` uses **strict**, not fusion.
Use this **judge-fusion** download for the selected submission. The website and
presentation remain separate demo resources in the repository.
