# Current workflow — score before merge

Restored at the user's request on 2026-09-13. The five-case CLI/batch replay exactly reproduces the historical `2a498a4` predictions: **14 TP / 4 FP / 5 FN, F1 0.75676**, precision 77.78%, recall 73.68%, count MAE 1.8 at 3 mm. This is a selected development operating point on five reused, AI-assisted and potentially incomplete references.

The [standalone judge package](application/README.md) ships this workflow and
its bundled model separately from the website, with a ten-wheel Windows
runtime. [Judge installation and run instructions](application/START_HERE.md).

## What runs

`run.py`, `batch.py`, `find_daughter_branches()` and normal Explorer use `pipeline.run_pipeline()`:

1. Generate strict proposals with native contrast scale **0.9**, on a 1 mm working grid.
2. Generate review proposals with looser settings, sharing the physical options and 2 mm origin eligibility policy.
3. Score each completed proposal independently with the bundled synthetic-trained logistic model and the existing 13 features in `learning.FEATURE_NAMES`.
4. Keep scores **≥0.15**. These scores are uncalibrated.
5. Retain strict survivors first, then review survivors whose origins are at least **3 mm** from every retained origin.
6. Assign sequential IDs and export the source geometry unchanged.

Scoring before merging prevents a low-scoring review representation from displacing a valid strict representation before filtering. The 3 mm deduplication distance and the 3 mm evaluation tolerance serve different purposes.

| Setting | Value |
|---|---|
| Model | `models/production-v1/logistic.json` |
| Model SHA-256 | `e9f864956aa65c3f05c038b13b0c8ca9e7289e4a44f6f0eb7837ab85814c730e` |
| Effective / saved model threshold | 0.15 / 0.08833933 |
| Minimum seed radius / origin diameter | 0.7 mm / 2 mm with existing native-voxel uncertainty allowance |
| Seed distance / maximum trace | 5 mm / 10 mm or estimated bifurcation |
| Runtime | Local model, CPU, base pinned dependencies; no model download |

The raw `detector.detect()` and `DetectorConfig()` retain the 1.2 strict baseline for research reproducibility. The production pipeline applies 0.9 to both fusion passes explicitly. Review mode remains an unfiltered annotation workflow and is not a final prediction.

## Commands

Run from the repository root after activating your environment (`source .venv/bin/activate` in this macOS workspace). Cap library threads before starting Python:

```bash
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4

# Required challenge interface: now uses fusion by default.
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json

# One supplied subject.
python run.py --image data/subject019/orig19.nii --aorta-mask data/subject019/mask19.nii \
  --output outputs/test019/prediction.json --diagnostics outputs/test019/diagnostics.json

# All five released reference subjects, then score at 3 mm.
python batch.py --cases subject019 subject020 subject021 subject022 subject023 \
  --output-dir outputs/test-labelled
python score_references.py --references labels/organizer-v1/references \
  --predictions outputs/test-labelled --data-root data --output outputs/test-labelled/metrics.json

# All 25 scans, or the normal Explorer.
python batch.py --data-root data --output-dir outputs/fusion-all
python explorer.py --data-root data --port 8000
```

Use a fresh output directory for each experiment. Diagnostics include model hash, effective/saved thresholds, both detector configurations, proposal features and geometry, scores, filtering decisions and merge suppressors.

## Explicit alternatives

- `--pipeline strict` reproduces the previous unfiltered 1.2 baseline (five-case F1 0.5333).
- `--candidate-model PATH` overrides the bundled model and uses that artifact's saved threshold. `--candidate-threshold VALUE` overrides the threshold. These options exist in CLI, batch and Explorer.
- CLI `--pipeline strict --recover-connected-origins` preserves guarded recovery. Combining recovery with fusion is rejected because that combination has not been validated.
- Explorer `--review-mode` is unfiltered and rejects model/threshold overrides.

Missing or changed bundled weights fail explicitly. A threshold without a model in strict mode is rejected. The bundled model resolves relative to the source file, including when launched outside the repository.

## Evidence and limits

Fresh five-case predictions and full diagnostics are in `outputs/fusion-restored-20260913/`; the compact checked-in receipt is [fusion-restored-validation.json](docs/fusion-restored-validation.json). At 2 mm matching, fusion scores 8/10/11 (F1 0.4324); at 5 mm it matches the 3 mm score. No coordinates were adjusted to labels.

The earlier strict release evidence and its replay scripts were removed from the working tree in the 2026-09-13 cleanup and remain in git history before that commit; the four receipts the slides cite are kept under `presentation/evidence/`. Its selection gates, 24-case synthetic F1 0.9684, packaged downloads and presentation describe the earlier strict configuration; they do not establish fusion performance. The fresh 24-case synthetic comparison gives fusion **47 TP / 11 FP / 2 FN, F1 0.8785**, versus strict **46/0/3, F1 0.9684**. Fusion adds seven FPs in `touching_vein_and_calcification` and four in `negative_controls_only`, both seed 31415. Thus the old zero-negative-control claim does not apply to fusion. This known FP regression is retained with the requested development operating point and is the next filter-hardening target. Full results are recorded separately in the receipt. Native organizer-Windows/four-core/8-GB acceptance remains open.

The original fusion trial, its revert and the strict-release plans are documented in git history before the 2026-09-13 cleanup commit.

Validation: 460 Python tests passed (38 skipped); Ruff, mypy, frontend lint/typecheck, all 36 frontend tests and the production build passed. CLI predictions from outside the repository match batch outputs on all five references.
