# CNN final-evaluation handoff

## Checkpoint status: PARTIAL — do not present as a completed comparison

Implementation commit: `546daa1` (main). This checkpoint adds the preserved
candidate receipts, source inventory, and continuation support. No training on
released cases, model edits, production changes, branches, or PRs.

- Current synthetic CNN + synthetic gradient-boosting + frozen blend extraction:
  all five cases completed on strict and review-union candidates.
- Historical synthetic CNN: all five strict cases completed; review-union
  extraction underway at checkpoint.
- Historical mixed CNN: extraction still pending at checkpoint.
- A running sequential extraction process may finish more cases after this
  checkpoint; the committed receipts are the authoritative completion inventory.
- `report.json`, when present, includes only complete five-case family/pool
  combinations. Its `failures` records missing/failed combinations. It is partial
  until all six combinations have succeeded. Never fill missing cases with empty
  predictions; valid zero-candidate outputs are explicitly preserved.
- Final text interpretation, complete report verification and final push remain
  pending.

## Continue on main

Follow root README and FINAL_EVALUATION_PROTOCOL.md. Ownership is limited to
`final_eval_cnn.py`, `tests/test_final_eval_cnn.py`, and this directory. Pull before
push, never overwrite other agents' changes. No root handoff changes.

```bash
git pull --rebase
uv venv --python 3.13.3 .venv313
uv pip install --python .venv313/bin/python -r requirements-dev.txt \
  -r requirements-cnn-inference.txt -r requirements-resources.txt
git lfs pull --include='data/subject019/*,data/subject020/*,data/subject021/*,data/subject022/*,data/subject023/*' --exclude=''
mkdir -p outputs/final-cnn-frozen
git archive 3395de51c91fe63882005a45e18f3fee54fce8ca \
  detector.py candidate_patches.py learning.py nifti_io.py \
  | tar -x -C outputs/final-cnn-frozen
git archive 2a825d89b66e7d5c61bea7f497be08fbc22cf526 \
  research_run.py patch_inference.py tabular_learning.py \
  | tar -x -C outputs/final-cnn-frozen
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4 VECLIB_MAXIMUM_THREADS=4
.venv313/bin/python final_eval_cnn.py extract --offline --resume
.venv313/bin/python final_eval_cnn.py replay
.venv313/bin/ruff check final_eval_cnn.py tests/test_final_eval_cnn.py
.venv313/bin/mypy final_eval_cnn.py tests/test_final_eval_cnn.py
.venv313/bin/python -m pytest -q tests/test_final_eval_cnn.py \
  tests/test_research_run.py tests/test_final_evaluation.py \
  tests/test_detector.py tests/test_candidate_patches.py tests/test_patch_learning.py -rs
git add final_eval_cnn.py tests/test_final_eval_cnn.py labels/final-eval/cnn
git commit -m "Complete frozen CNN five-case comparison"
git pull --rebase
git push origin main
```

`strace` must be installed for `--offline` on Linux. It denies all network
syscalls to each worker; measured workers confirm socket creation was denied.
Do not run a second extractor concurrently against the same output directory.
`--resume` preserves complete receipts, validating successful model/source
contracts. Failed receipts remain failed; use a fresh `--output` directory for
a diagnosed retry rather than deleting evidence. Replay needs no CTs or training
corpora. Full extraction requires the five LFS image/mask pairs.

## Reproducibility and interpretation

The grid is CNN weights `{0,.25,.5,.75,1}` crossed with
`{.15,.3,.5,.7,.85}` and the frozen CNN/tree/blend thresholds, for current strict
and review-union pools. Each pool also includes the unfiltered proposals and
declared frozen blend. Historical CNN-only rows use their frozen threshold and
the same five threshold probes, plus their own unfiltered proposal baseline.

Probabilities, features, ordered identities, complete branch paths, input hashes,
model hashes, extraction source hashes, model/extraction/inference timings and
resource receipts are in `candidates/<family>/<pool>/subjectNNN*.json`. Threshold
and blend replay validates identity/order/features/data contracts through
`patch_inference`. Runtime is conservatively shared across variants using the
same extraction; it includes both current models even at weight endpoints.

Current source hashes:

- detector.py: `9f96f3eb3c06f5e06d5b7db79b199be70e742130fc3885243d25d066adc1c92e`
- candidate_patches.py: `568d8bd9f0d417768cb22fdd56d054a26ca215cc1741d3d9914ff4052eaa534a`
- learning.py: `d7acdaa6f78ea9f3b6703d0b457880de37804ee0fc06ff6aeb4c9bfe7fe82c1e`
- current/legacy synthetic ONNX graph: `a562a0f4027b028eeda9c8e1af9c5b792d29cbc31d586d9c0814f32573a8353d`
- mixed ONNX: `8787a20a9738f12c4016c1b7d61874836974e55d759cf17b45aa61eb18d4096f`
- frozen detector: `17495723df18c79e4302edc19bd949bb16f615b902403a88c68d6e74fa200063`

`inventory.json` binds all remaining source/sidecar/training-report hashes.
Identical current/legacy synthetic graph bytes do not allow sidecar substitution:
current training records, manifest and reproduction receipts bind current source.
Historical mixed training includes subject022 and subject023, so its rows are
ineligible. Both old blends are excluded: their tree lacks a verified extraction
contract even before source comparison. The inventory gives the retraining path.

Review-union outputs are research proposals with the existing origin-diameter
override; they are not a deployment eligibility certification. Current strict
uses the 2-mm origin-diameter policy. Legacy source predates that policy. Confirm
these distinctions when integrating with other families.

Use local 2/3/5-mm scores and actual known-radius masking. Only three radii are
measured; all five reused cases remain development data, not clinical evidence.
Within-family leave-one-case-out results do not replace selection across the
parent's entire model matrix. Native Windows performance is unverified.

## Local-only files and checks

- Local checkout: `/home/ubuntu/repos/final-cnn`.
- `.venv313/` is rebuildable from pinned requirements.
- `outputs/final-cnn-frozen/` is ignored, recreated by the archive commands above.
- `outputs/cnn-*.log` are local network syscall traces. Portable proof is the
  committed per-worker `network_denied` and measurement command/evidence.
- Running extraction shell at checkpoint: `2c3dbc` on the original session VM.
- Initial Ruff/mypy passed. Focused checks: 26 passed, 3 skipped before the full
  matrix existed. Detector/patch regressions: 91 passed, 7 optional Torch/ONNX
  training-package tests skipped. Actual ONNX CPU inference and batching tests
  passed with installed ONNX Runtime; no Torch is needed for this evaluation.
- Direct notification to parent through session MCP returned 403; structured
  output and pushed repository artifacts are the supported handoff.
