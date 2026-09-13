# Tabular evaluation — COMPLETE bounded family comparison

## Status

The final complete run produced **106 variants, no scoring failures, and four
explicit historical-tree exclusions**. Nine compatible models were scored on
strict and review-union proposals at each frozen threshold plus the deduplicated
0.15/0.3/0.5/0.7/0.85 grid; two unfiltered baselines are included.

A second complete run finished after strengthening cache validation, adding input
geometry, and adding paired case-bootstrap intervals. All final caches and
aggregate reports share the driver hash recorded below. All 530 predictions,
source, tests and the fresh model are committed. No authored code or result needed
for replay exists only on the VM. The earlier partial checkpoint is
`d0a99d302093c7dbf2885012d73c2fe2a7ba3eea`; use its successor completion commit.

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
  all five released identities and aliases; both complete 106-variant runs.
- Completed: Ruff and mypy on both authored files.
- Focused/relevant pytest: 89 passed, 26 skipped (optional sklearn training
  dependency absent). Includes complete five-case prediction/score replay,
  exact saved probability reproduction, cache corruption rejection, failed-model
  withholding, empty proposals and held-out ranking checks.
- All requested compatible tabular model/profile/threshold variants are complete.
  The four historical tree exclusions are explicit in `model-audit.json`, with
  evaluated source-compatible regenerated replacements.
- Optional follow-up: install `requirements-trees.txt` and exercise the existing
  training/export tests. No new estimator family or training run is needed here.
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
- `selection.json`: tabular-only leave-one-case-out choices and paired bootstrap
  intervals. Global family selection belongs to the parent.
- `candidates/`: replayable ordered features, extended features, probabilities,
  fingerprints, input/source/model hashes and resource measurements.
- `predictions/`: all five challenge-schema outputs per valid variant.
- `model-audit.json`, `run-provenance.json`, `training/`: model and data lineage.

These are LOCAL 2/3/5 mm reference comparisons, not official challenge scores.
Only measured reference radii are scored. Five reused development images cannot
establish hidden-set accuracy. Linux CPU timing does not validate Windows timing.
Do not promote a model from these results without the parent's synthetic topology
regressions and overall resource review.

Local-only state: `/home/ubuntu/repos/tabular-eval/.venv313` and fetched LFS objects.
Dependencies and images are reproducible
with the commands above. The initial snapshot referenced an unrelated
StevenTB1 checkout; the work is in a fresh Coder-Meet clone.

Commit only `final_eval_tabular.py`, `tests/test_final_eval_tabular.py` and
`labels/final-eval/tabular/`. Other agents own all other files.
The repo ignores all directories named `predictions`; explicitly stage only this
family's generated prediction directory with
`git add -f labels/final-eval/tabular/predictions` when refreshing artifacts.

## Results for the parent

At 3 mm, strict proposals achieved TP/FP/FN = 8/3/11 (F1 0.5333); review-union
achieved 10/14/9 (F1 0.4651). The development winner was current-source extended
random forest on review-union at threshold 0.85: 9/1/10, F1 0.6207. It retained
9 of the 10 targets available in those proposals; classifiers cannot recover the
other nine targets missed by that proposal pool.

Across 84 eligible variants, tabular-only leave-one-case-out selection achieved
8/2/11, F1 0.5517, count MAE 1.8. F1 at 2/5 mm was 0.4138/0.5517. Three folds
selected the development winner, one selected base gradient boosting with its
frozen threshold, and one selected synthetic logistic on strict proposals at 0.3.
This is not a stable universal model selection result. The paired 95% case
bootstrap interval for held-out F1 is [0, 0.8485]; relative F1 change versus strict
is [0, 0.0476], and versus review-union is [-0.2, 0.2441].

Fresh reference-excluded logistic retained 208 pseudo-reviewed candidates and
froze validation threshold 0.5. Its highest all-five development F1 was 0.5517
at the predeclared 0.15 threshold on strict proposals; do not reinterpret that
reference-selected threshold as the validation-frozen threshold.

Maximum measured per-case variant time was 20.6631 seconds and conservative
whole-job peak RSS was 631.4922 MiB, with four configured CPU threads. These
measurements passed the stated memory budget on Linux; Windows remains untested.
No deployment promotion or claim of independent challenge accuracy is made.
