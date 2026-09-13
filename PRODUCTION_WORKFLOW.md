# Current production workflow — score before merge, v1

The default workflow is now **strict + review proposals → logistic scoring → threshold 0.15 → strict-first merge within 3 mm**. This replaces the earlier strict-only default. The user selected the highest observed five-case development F1 configuration after reviewing its FP tradeoffs.

`run.py`, `batch.py`, `find_daughter_branches()` and normal Explorer analysis all call `pipeline.run_pipeline()`. Raw `detector.detect()` remains the classical proposal generator. `explorer.py --review-mode` and `autolabel.py` retain the unfiltered annotation pool.

| Component | Selected setting |
|---|---|
| Strict detector | Existing `DetectorConfig()`, native contrast scale 0.9, 1 mm working grid |
| Review detector | Existing loose review settings, with the same physical options and origin eligibility as strict |
| Final origin eligibility | 2 mm default on both passes, with existing native-voxel uncertainty policy |
| Model | `models/production-v1/logistic.json`, byte-identical copy of the audited synthetic logistic |
| Model SHA-256 | `e9f864956aa65c3f05c038b13b0c8ca9e7289e4a44f6f0eb7837ab85814c730e` |
| Features | Existing 13-feature vector in `learning.FEATURE_NAMES` |
| Effective threshold | **0.15**, inclusive; the original artifact's 0.08833933 remains recorded separately |
| Merge | Score first; retain strict survivors first, then review survivors at least 3 mm from every retained origin |
| IDs and geometry | Sequential output IDs; source branch coordinates, radius and direction preserved |
| Dependencies | Base requirements only; local model, offline CPU inference |

The review pass's 2 mm eligibility enforcement was checked against the audit's original review policy. All five final predictions are exactly equal to the audited winner, including coordinates and IDs. Unknown/borderline origin measurements retain the existing uncertainty handling.

```bash
# Default production inference
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz \
  --output prediction.json --diagnostics diagnostics.json

# Same workflow for every prepared case
python batch.py --data-root data --output-dir predictions/production-v1

# Normal Explorer displays production-filtered results
python explorer.py

# Annotation: retain weak/rejected proposals for review
python explorer.py --review-mode
python autolabel.py render --cases subject019
```

`diagnostics.json` contains `workflow`: actual model hash, effective and saved thresholds, both detector configurations, per-profile rejection counts, and every completed proposal's source profile, source ID, feature/geometry record, probability, decision and merge suppressor. The challenge prediction JSON retains its existing schema. The UI identifies filtered results; review mode remains explicitly unfiltered.

Compatibility and overrides:

```bash
# Previous classical default, no model required
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz \
  --pipeline strict --output strict.json

# Reproduce the previous strict-plus-custom-model command
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz \
  --pipeline strict --candidate-model model.json --output strict-filtered.json

# Explicit alternative operating point with the production model
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz \
  --candidate-threshold 0.30 --output tighter.json
```

These pipeline/model/threshold options also exist in batch and Explorer detection mode. An explicit custom model uses its own saved threshold unless `--candidate-threshold` is supplied. With no custom model, production always uses 0.15. Strict mode without a model is unfiltered and rejects a threshold-only override. Invalid thresholds, missing models and altered bundled model hashes fail explicitly. Paths to the bundled model resolve relative to the code, including when invoked from another working directory.

Ship `pipeline.py` and `models/production-v1/logistic.json` alongside the existing inference modules. Do not omit the model when preparing an offline submission. No tree/CNN artifact, training environment or online download is needed at inference time.

The fresh five-case CLI and batch replay returns **14 TP / 4 FP / 5 FN**, precision **77.78%**, recall **73.68%**, F1 **0.75676** at the local 3 mm ostium tolerance. Per-case output counts are 2, 2, 4, 9 and 1. Count MAE is 1.8; higher discovery F1 does not imply the closest count on every case. At 2 mm the score is 8/10/11; at 5 mm it matches 3 mm. The released reference package is judge-approved but AI-assisted and potentially incomplete.

Known tradeoff: the audit's 14-family synthetic cohort gives fusion 31/3/3 versus strict-plus-filter 31/1/3. The user selected the stronger real-reference development F1 operating point for now; this does not establish independent accuracy or resolve the synthetic FP regression. Four saved research trees still reject their old detector-source contracts. Their guards remain intact. Windows/four-core/8-GB acceptance measurements and a newly adjudicated patient cohort remain outstanding.

See [CURRENT_E2E_REVIEW.md](CURRENT_E2E_REVIEW.md) for the audit and updated next-step status. Compact model provenance and exact replay evidence are stored in `models/production-v1/`; larger local run receipts are in `outputs/production-v1-validation/`.
