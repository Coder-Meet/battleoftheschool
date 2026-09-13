# Fusion judge package — verification

## Selected application

The standalone application uses the unchanged runtime files from
`6e67aca527fac6eaa2e2f1dc97cea3a62a7b5677`:

- strict and review proposals at native contrast scale 0.9;
- bundled logistic model, threshold 0.15;
- scoring before strict-first merging within 3 mm;
- 1 mm working spacing, 2 mm minimum origin diameter and 5 mm seed path.

The runtime/model SHA-256 values are pinned in `build.py` and reproduced in
each archive's `MANIFEST.json`. All eight source/model files match. The model
SHA-256 is
`e9f864956aa65c3f05c038b13b0c8ca9e7289e4a44f6f0eb7837ab85814c730e`.
There is no change to detector logic, weights or thresholds in this packaging.

## Download and installation checks

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Source ZIP | 31,079 | `68220554ad76324445b1faf5a7c57cae735c779fdda2a811269e8b28558bb127` |
| Windows x64 ZIP | 90,713,183 | `55d8bcc7fbe6346fcf313e4894952f5ed72a5955cf6fed27d5106004751dc304` |

Two local rebuilds were byte-identical. The Windows CI build was also
byte-identical to the published Windows ZIP, including every wheel and the
model. [Windows and Linux package CI](https://github.com/Coder-Meet/battleoftheschool/actions/runs/34760619075)
passed with Python 3.13.3.

The source ZIP has only eight runtime/model files, the pinned requirements,
judge instructions and manifest. The Windows ZIP adds ten runtime wheels.
No website, Node/npm, training data, matplotlib, scikit-learn, PyTorch or ONNX
is present. Helper code imported by the unchanged runtime remains included.

Checks performed:

- Clean Python 3.13.3 environment with only the ten runtime packages and pip.
- Local wheel installation and `pip check`, with network syscalls denied.
- Complete Windows x64 CPython 3.13 dependency resolution with network denied.
- Native Windows and Linux CI installation without a package index.
- Extracted CLI run from outside the repository on a nonempty vessel phantom.
- Missing/modified model fails instead of silently changing the algorithm.
- Source pinning, archive membership, checksums, repeatable builds and rejection
  of incompatible Python/architecture wheels.
- Repository Ruff and Mypy checks; 22 focused packaging/pipeline tests.

The [full repository Linux/Windows regression workflow](https://github.com/Coder-Meet/battleoftheschool/actions/runs/34760619058)
also passed for the validated packaging commit.

## All 25 supplied cases: isolated offline replay

The **extracted Windows ZIP's Python source** was executed on Linux with matching
Linux wheels. It ran outside the checkout using the clean numerical environment.
`strace` denied network syscalls, and the resource wrapper verified that socket
creation was denied. OS CPU affinity was restricted to four cores, with all
recorded numerical-library thread ceilings set to four.

| Measurement | Result |
|---|---:|
| Cases completed with valid challenge JSON | 25 / 25 |
| Predictions byte-identical to a separate full-checkout replay | 25 / 25 |
| Failures | 0 |
| Average per-case end-to-end time | 11.604 seconds |
| Slowest case (subject025) | 52.670 seconds |
| Whole batch, including process startup | 290.981 seconds |
| Maximum sampled process-tree RSS | 1498.99 MiB (about 1.46 GiB) |

These are **Linux development measurements**, including syscall tracing.
RSS includes the resource wrapper and its live descendants, excludes the outer
sampler, and can miss brief peaks or count shared pages twice. No 8 GB hard
memory cap was imposed. Windows CI validates installation and synthetic CLI
execution; it does not measure these CT cases on the organizer's laptop.
**Native organizer-Windows/four-core/8-GB timing remains outstanding.**

## Fresh reference score

The packaged outputs reproduce the recorded TP/FP/FN, F1 and daughter-count
scores at all three local matching tolerances (2, 3 and 5 mm). At **3 mm**:

| TP / FP / FN | Precision | Recall | F1 | Count MAE |
|---|---:|---:|---:|---:|
| 14 / 4 / 5 | 0.77778 | 0.73684 | **0.75676** | 1.8 |

The evidence ZIP contains the newly generated detailed geometric-error
summaries; do not substitute historical geometry summaries for this replay.
The 3 mm matching tolerance is separate from the 2 mm **diameter** eligibility
rule. At the tighter 2 mm matching tolerance, F1 is 0.43243.

These are five **reused, AI-assisted, potentially incomplete development
references**, not independent hidden-test accuracy or the official weighted
score. The user selected fusion for its higher measured reference F1.
The existing 24-case synthetic comparison recorded 47/11/2 for fusion versus
46/0/3 for strict, including four fusion detections in negative controls.

The separate evidence download includes all 25 predictions, diagnostics, three
CT overlays, the fresh scores and the resource receipt. Overlays show automatic
predictions for inspection; they do not certify each daughter as correct.

## Reproduction

Build with the [maintainer commands](README.md#rebuild), then extract the ZIP and
use the [judge commands](START_HERE.md). Run `batch.py` against the 25 case
directories. For the Linux network/CPU measurement, use the resource wrapper
`research_resources.py` from git history before the 2026-09-13 cleanup commit (it is no longer in the working tree) around the isolated Python/batch command:

```bash
strace -f -e trace=%network -e inject=%network:error=ENETUNREACH \
  -o network.log /path/to/dev/python /path/to/repo/research_resources.py \
  --output resources.json --case-id all-25-fusion --cores 4 -- \
  /path/to/clean/python /path/to/extracted/batch.py \
  --data-root /path/to/cases --output-dir /path/to/predictions
```

The wrapper and its `requirements-resources.txt` live in that same git history;
neither is part of the judge runtime.
