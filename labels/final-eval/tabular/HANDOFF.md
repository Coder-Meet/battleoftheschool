# Tabular evaluation checkpoint — INCOMPLETE

## Status

The initial complete run produced **106 variants, no scoring failures, and four
explicit historical-tree exclusions**. Nine compatible models were scored on
strict and review-union proposals at each frozen threshold plus the deduplicated
0.15/0.3/0.5/0.7/0.85 grid; two unfiltered baselines are included.

A second complete run is currently regenerating results after strengthening
cache validation, adding input geometry, and adding paired case-bootstrap
intervals. At checkpoint time it had reached subject021/review-union.
**Candidate caches and aggregate reports may therefore belong to different
driver revisions in this checkpoint. Do not combine them or promote its scores.
Run the complete command below before relying on the report.**

All authored source, tests, the fresh model, available predictions and caches are
committed here. No useful authored code exists only on the VM.

## Commands

From the repository root, with Python 3.13.3:

```bash
uv venv --python 3.13.3 .venv313
uv pip install --python .venv313/bin/python -r requirements-dev.txt -r requirements-resources.txt
git lfs pull --include="data/subject019/*,data/subject020/*,data/subject021/*,data/subject022/*,data/subject023/*"
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4 \
  .venv313/bin/python final_eval_tabular.py run
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4 \
  .venv313/bin/python -m pytest -q tests/test_final_eval_tabular.py \
  tests/test_final_evaluation.py tests/test_learning.py tests/test_tabular_learning.py
.venv313/bin/ruff check final_eval_tabular.py tests/test_final_eval_tabular.py
.venv313/bin/mypy --follow-imports=silent final_eval_tabular.py tests/test_final_eval_tabular.py
```

The `report` command replays existing caches without CT processing but deliberately
rejects stale sources, driver hashes, model hashes or altered input identities.
The full `run` command currently takes a few minutes on this four-thread CPU VM.
It uses existing NumPy tree inference; optional sklearn is not needed for saved
model evaluation. Install `requirements-trees.txt` to enable the additional
existing sklearn training/export tests.

## Completed and pending

- Completed: saved-model/source audit; fresh regularized logistic after excluding
  all five released identities and aliases; first 106-variant run.
- Completed: Ruff and mypy on both authored files after the latest edits.
- Earlier focused/relevant pytest run: 81 passed, 26 skipped (optional sklearn
  training dependency absent). A test annotation error was fixed.
- Pending: finish second run; run new cache corruption, replay, failure-contract,
  and leave-one-case-out tests; inspect compact results; update this handoff.
- Pending: final commit, `git pull --rebase`, and push directly to MAIN.
- No branches, PRs, reference-driven candidate generation, or production detector
  changes are authorized.

## Provenance

Starting commit: `2a825d89b66e7d5c61bea7f497be08fbc22cf526`.

SHA256 at checkpoint:

```text
final_eval_tabular.py 0b09877d8548d4751769b626a0e7185ae5cd7e15cd9cfa985ae21e8dffdc1f5f
detector.py          9f96f3eb3c06f5e06d5b7db79b199be70e742130fc3885243d25d066adc1c92e
candidate_patches.py 568d8bd9f0d417768cb22fdd56d054a26ca215cc1741d3d9914ff4052eaa534a
learning.py          d7acdaa6f78ea9f3b6703d0b457880de37804ee0fc06ff6aeb4c9bfe7fe82c1e
```

Fresh fitting retained 208 AI pseudo-reviewed candidates; subject019 had no
pseudo-review rows. Exact removed IDs, fingerprints and historical split are in
`training/report.json`; scaler, coefficients and validation threshold are in
`training/logistic.json`. No reference outcomes or unmatched draft-reference
proposals were used as training labels. No calibration was fitted.

The four excluded cohort-v1 trees lack validated current extraction contracts.
Their current-source-v1 re-extracted/retrained equivalents were all evaluated.
The byte-identical cohort logistic was also evaluated. Historical real-review
logistics remain retrospective and ineligible for clean selection.

## Artifacts and limitations

- `report.json`: shared contract, all variants, scores, runtime and exclusions.
- `selection.json`: tabular-only leave-one-case-out choices and, after rerun,
  paired bootstrap intervals. Global family selection belongs to the parent.
- `candidates/`: replayable ordered features, extended features, probabilities,
  fingerprints, input/source/model hashes and resource measurements.
- `predictions/`: all five challenge-schema outputs per valid variant.
- `model-audit.json`, `run-provenance.json`, `training/`: model and data lineage.

These are LOCAL 2/3/5 mm reference comparisons, not official challenge scores.
Only measured reference radii are scored. Five reused development images cannot
establish hidden-set accuracy. Linux CPU timing does not validate Windows timing.
Do not promote a model from these results without the parent's synthetic topology
regressions and overall resource review.

Local-only state: `/home/ubuntu/repos/tabular-eval/.venv313`, fetched LFS objects
and the currently running shell job. Dependencies and images are reproducible
with the commands above. The initial snapshot referenced an unrelated
StevenTB1 checkout; the work is in a fresh Coder-Meet clone.

Commit only `final_eval_tabular.py`, `tests/test_final_eval_tabular.py` and
`labels/final-eval/tabular/`. Other agents own all other files.
