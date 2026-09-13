# Branchseed: algorithm review for Steven

**Engineering handoff • 13 September 2026 • reviewed source: `f43165b`**

## 1. What to know before changing anything

The submission still uses the deterministic, CPU-only detector. It takes a CT and a parent-aorta mask and discovers a variable number of direct daughter openings. It does not load a learned model unless explicitly requested. The Explorer visualizes and reviews its results; it is not the detection algorithm.

We have added an optional ML research pipeline, stronger synthetic experiments, physical candidate patches, source-compatibility checks and reference-evaluation tools. The selected tree filter is worth evaluating on complete organizer references. The CNN and combined-model experiments lost true branches and were rejected for promotion. More complex models have not established better real-scan accuracy.

Steven's imported review dataset currently contains **237 candidate decisions across 23 cases: 92 confirmed and 145 rejected**. Every stored labeller is `claude`. These are useful pseudo-labels, not independent expert references. The model's agreement with them cannot tell us how many real arteries the proposal stage never found.

**Recommended operating choice:** keep the strict detector as the submission baseline; use review-union and ML scores to investigate disagreements. Freeze outputs before evaluating the incoming references.

### Important changes

| Area | Current behavior and reason |
| --- | --- |
| Eligibility | Separate **2 mm origin diameter** setting; seed/tracing radius remains **0.7 mm**. Unresolved origin measurements are retained with warnings. |
| Opening count | Common trunks are deduplicated; nearby separate openings and returning-vessel openings have regression coverage. Count errors are reported alongside one-to-one matches. |
| Learned filtering | Original logistic model is preserved. Optional random forests, gradient boosting, patch CNN and frozen blends have separate tooling. No new model is enabled by default. |
| Research data | A 210-case, group-separated synthetic corpus replaces dependence on the earlier tiny, positive-heavy training set. Negatives are actual unmatched proposals. |
| Reproducibility | Source/model hashes, feature order, candidate identity, splits and physical preprocessing are checked. Archived weights are not silently treated as current-source models. |
| Deployment | Base inference remains offline and CPU-only. Windows functional CI exists; actual organizer-laptop timing is still outstanding. |

## 2. How the production detector works

The code is organized as `normalize → enhance → propose → resolve` in `detector.py`.

### A. Normalize the CT and parent geometry

`validate_geometry` checks the CT and mask before processing. `prepare_roi` crops around the supplied parent with a **20 mm margin**, then resamples onto a **1 mm isotropic** working grid. CT uses linear interpolation; the binary parent uses nearest-neighbor interpolation. Origin and direction are preserved for physical-coordinate conversion.

`normalize` smooths the image and computes distances inside and outside the parent. `blood_window` estimates blood and surrounding-tissue intensity from the scan rather than assuming one universal contrast threshold. Low-contrast and overlapping-intensity conditions generate warnings.

The mask supplies the parent location, not daughter segmentation. Poor parent masks, thick slices and low contrast still affect the downstream result.

### B. Enhance supported vessel-like structures

`enhance` combines scan-relative CT support with multiscale Sato tubularity at physical scales **0.8, 1.5 and 2.5 mm**. Distance transforms estimate local lumen support/radius; skeleton structure supplies estimated downstream junctions. Crop-end exclusion suppresses contacts at acquisition caps.

This creates plausible image support, not a semantic artery-versus-vein classifier. Contrast-enhanced veins and other bright structures can resemble arteries. The judge's reported allowance for some vein predictions does not make size alone an artery/vein discriminator.

### C. Propose aortic wall contacts

The strict profile searches supported, tubular components in a shell **1.5–3.5 mm outside the parent**. Roots are ranked using local radius and vesselness. Strict mode normally chooses one root per contact component and processes at most **160 candidates**. A limit warning is emitted if that budget is exceeded.

The wider review profile extends the shell to **6 mm**, lowers vesselness cutoffs, permits some connector gaps and allows up to six separated roots per contact. It also disables origin-diameter rejection. `detect_pool` unions the review result with strict detections so strict-only proposals remain available.

The review pool is an investigation tool. Its looser connection and size settings can create extra proposals and do not establish final challenge eligibility.

### D. Trace and verify each proposal

`_trace` uses a minimum-cost path on a local supported region. Unsupported voxels and the parent interior are excluded. The path cost favors lumen interiors and stronger tubularity:

```text
cost = 1 / (0.5 + local_radius) + 0.7 * (1 - vesselness)
```

The path is connected back to the parent wall to estimate the ostium. The physical path is limited to **10 mm** and stopped at an estimated downstream bifurcation. It must retain at least **5 mm** before the seed is interpolated along its arc length.

Filters reject disconnected ostia, short paths, weak tubularity, excessively broad regions, implausible radius and duplicated openings. Rejection reasons are retained in diagnostics.

**Two consequential limitations:** the strict endpoint search requires outward wall clearance of approximately 5.5 mm on the default grid; it also requires at least 3.5 mm straight-line displacement from ostium to seed. Thus a valid curved or wall-parallel daughter can have 5 mm of visible centerline yet fail strict tracing. Experimental parallel tracing exists but previously added false positives, so it stays off.

If an estimated first bifurcation leaves less than 5 mm of proximal path, the implementation rejects that candidate. The judge's exact seed convention for such early common trunks remains important.

### E. Measure, deduplicate and export

- **Ostium:** the estimated parent-wall crossing, not the parent centerline.
- **Seed:** 5 mm along the accepted proximal path, not necessarily 5 mm of straight-line displacement.
- **Direction:** the normalized ostium-to-seed vector. This is a chord, not a fitted local tangent.
- **Seed radius:** a perpendicular CT cross-section estimate when available; otherwise the support distance-transform estimate. The cross-section uses 32 rays and the median of opposing-ray diameters, not an area-equivalent radius.
- **Instances:** nearby origins are compared using origin separation, seed separation and shared proximal paths. A common trunk should produce one direct opening; separately supported openings can survive.

These are geometry heuristics with regression coverage, not guarantees for every anatomy. Overlapping ostia may be scored as one or two according to the judge; the official evaluator is still needed to encode that allowance exactly.

Branches are sorted and numbered after resolution. **`branch_003` is not a stable anatomical identity across profiles or algorithm versions.** Match reviews using stored candidate fingerprints; use physical proximity only as an explicitly approximate review aid.

## 3. The confirmed 2 mm rule

The two settings have different jobs:

| Setting | Default | Meaning |
| --- | --- | --- |
| `minimum_origin_diameter_mm` | 2.0 | Organizer's minimum eligible proximal origin diameter. Zero disables this check. |
| `minimum_radius_mm` | 0.7 | Minimum tracing/seed lumen radius used by the detector. |

The origin estimate is sampled **2 mm along the proximal path**. Its cross-section uses the local halfway intensity between proximal lumen and background. If a radius can be measured, diameter is twice that radius.

```text
estimated_upper_diameter = measured_diameter + largest_native_voxel_spacing
reject only when estimated_upper_diameter < 2 mm
```

The voxel allowance is a conservative engineering rule, **not a statistical confidence interval**. The measurement is an approximate proximal-origin surrogate; it needs reference-based calibration. The plane follows the ostium-to-seed direction, which can also bias measurements on curved vessels.

Unresolved estimates remain in the output with a warning. Borderline estimates receive a separate warning. On the 25 supplied cases, the current strict batch produced **150 daughters**, including **82 unresolved origin estimates**, one below-2-mm estimate within the allowance and no confident size rejections. Prediction JSONs were unchanged by this size-policy update.

This preserves candidates under partial volume. It does not prove that every retained candidate satisfies the official diameter rule.

## 4. What the ML modules do

### Original logistic filter

`learning.py` fits an L2-regularized logistic classifier to 13 candidate features: radius, mean vesselness, evidence score, path length, seed displacement, tortuosity, relative path intensity, bone distance, parent angle, position along the parent, native spacing, connector gap and contact volume.

It learns feature scaling on training cases and selects a threshold using validation cases. The score is not a calibrated clinical probability. The original model and the AI-plus-synthetic model are preserved separately.

The legacy `run.py --candidate-model ...` option can remove strict candidates. It is explicit and off by default; do not confuse its older feature-schema checks with the stronger current research source-contract checks.

### Tree models

`tabular_learning.py` and `train_trees.py` add random forests and gradient boosting. Training uses optional scikit-learn; inference uses validated JSON artifacts and NumPy rather than requiring scikit-learn or pickle.

Models can use the original 13 features or three additional measured features: local ostium contrast, patch vesselness standard deviation and parent-wall curvature. The extra features have not demonstrated an advantage in the completed experiment.

### Physical patches and CNN

`candidate_patches.py` extracts **12 channels of 32 × 32 pixels at 1 mm spacing**: physical XY/XZ/YZ views, each containing normalized CT, parent mask, vesselness and the proposed path. The ostium anchors the patch. The path channel represents the proposal, never a reference segmentation.

Local extraction preserves physical directions, interpolation conventions and scan-level normalization. Missing support for a required feature raises an error instead of inventing a zero.

The optional CNN has **34,465 parameters**, convolution widths 24/40/64, two average-pooling stages and global mean pooling. It classifies candidates; it does not predict missing ostia, segment a complete vessel tree or regress seed/radius/direction. CPU ONNX export and native/ONNX parity are tested.

### Current-source research runner

`research_run.py` supports tree, CNN and frozen combined scoring with two explicit modes:

- `scores-only`: preserve all selected proposals and put model scores in diagnostics.
- `filter`: write a separate filtered prediction while preserving original proposals.

It verifies preprocessing, source/model hashes and candidate ordering. Scoring failures are explicit; scores-only preserves daughters, while failed filter mode withholds the filtered output. Historical artifacts with incompatible contracts are rejected.

**Every classifier is limited by proposal recall.** A filter may remove false positives, but it cannot recover an artery absent from its input pool.

## 5. What the experiments actually showed

The current-source replay has **210 synthetic cases**, 14 stress families and 15 seed groups: 140 train, 42 validation and 28 test cases. These seed groups have already been inspected, so the replay is development evidence, not a new sealed test.

Counts below use complete analytic synthetic references and a local **3 mm one-to-one ostium tolerance**. They are not official challenge scores or real-patient accuracy.

| Method | Validation TP / FP / FN | Replayed test TP / FP / FN |
| --- | --- | --- |
| Strict detector | 91 / 0 / 12 | 61 / 1 / 9 |
| Review-union pool | 93 / 4 / 10 | 63 / 4 / 7 |
| Selected gradient boosting, base features | 93 / 1 / 10 | 63 / 0 / 7 |
| Logistic comparator | 93 / 2 / 10 | 63 / 3 / 7 |
| Synthetic CNN | See frozen training report | 59 / 2 / 11 |
| Selected tree/CNN blend | See frozen training report | 62 / 0 / 8 |

The selected tree removed four test-pool false positives without losing pool true positives. Seven reference daughters were still absent from the pool. Across all 210 cases, strict produced **437 TP / 5 FP / 72 FN** and the pool **453 / 28 / 56**.

The CNN lost four pool true positives. The selected blend lost one and chose **CNN weight zero**; that does not demonstrate an ensemble benefit. The separate historical mixed-pseudo CNN was also rejected, at **58 / 2 / 12** on its archived test.

No learned model passed the conditions for production promotion. Complete expert references, case-exposure accounting and target hardware evidence are still missing. Synthetic generator geometry and real CTA appearance remain substantially different.

### The three requested papers

| Paper | Implemented experiment | Decision |
| --- | --- | --- |
| Danilov et al., 2016 | Near-aorta border cleanup based on outward support layers | Optional; can remove wall-parallel vessels. |
| Riffaud et al., 2022 | Origin-anchored PCA direction | Optional geometry comparator; requires adaptation from already segmented vessel trees. |
| Tahoces et al., 2020, online 2019 | Parent-connected contact growth | Optional proposal comparator; named-vessel exclusions from the paper were not copied. |

None of our three adaptations beat the production detector in the recorded synthetic comparison. Their published results concern different inputs/tasks and must not be quoted as our performance.

## 6. How to interpret Steven's labels

The stored decisions describe rendered candidates. A `confirmed` row means that the recorded AI reviewer accepted that proposal. It does not certify 2 mm eligibility, complete coverage or independently measured geometry.

The original logistic split has **13 training, five validation and five test cases**. The test case IDs are subject005, subject013, subject018, subject021 and subject025. They were held out for that original fit, but have subsequently been examined; they are not fresh evaluation cases now.

The useful immediate analyses are:

1. Audit label counts, labeller provenance, duplicates, splits and candidate fingerprints.
2. Recompute frozen-model decisions on the exact stored feature vectors.
3. List confirmed candidates removed by a model and rejected candidates retained.
4. Run the current detector and distinguish exact fingerprint matches from changed or unlabelled candidates.
5. Prioritize uncertain cases for re-review without overwriting the original labels.

Do not label unmatched current proposals as false positives. Do not interpret an unmatched historical confirmed proposal as a verified missed artery. Do not use detector-generated pseudo-reference coordinates to claim independent localization accuracy.

A count match alone is insufficient: predicting 12 openings for a case with 12 references can still contain two extras and miss two true openings. We therefore report count error and one-to-one detection matches separately.

## 7. Commands and hardware

Use the pinned Python environment in the README. These commands run from the repository root after dependencies and data are prepared.

### Submission inference

```text
python run.py --image data/subject001/orig1.nii --aorta-mask data/subject001/mask1.nii --output outputs/subject001.json --diagnostics outputs/subject001-diagnostics.json
```

### Add research scores without removing strict predictions

```text
python research_run.py --image data/subject001/orig1.nii --aorta-mask data/subject001/mask1.nii --case-id subject001 --proposals strict --mode scores-only --tree-model labels/research/current-source-v1/models/gradient_boosting-base.json --output outputs/subject001-scored.json --diagnostics outputs/subject001-scores.json --proposals-output outputs/subject001-original.json
```

Use fresh output paths. See the README for CNN dependencies, combined-model arguments and source-contract troubleshooting.

### Full research reproduction

```text
python research_reproduce.py --root outputs/new-research-run --workers 4 --cnn
```

Install optional tree/CNN dependencies first. The command refuses an existing root. Do not overwrite frozen archives or retune their exposed test partitions.

### Portable resource measurement

```text
python research_resources.py --cores 4 --output outputs/windows-resources.json -- python run.py --image data/subject001/orig1.nii --aorta-mask data/subject001/mask1.nii --output outputs/windows-prediction.json
```

Install `requirements-resources.txt` first. The wrapper measures wall time and sampled process-tree RSS with affinity/thread limits; it does not enforce an 8 GB allocation cap.

The organizer target remains **Windows, four CPU cores, 8 GB RAM, no GPU, offline**. The measured strict Linux batch completed 25/25 cases in **129.24 seconds**, with **21.612 seconds maximum per case** and **1,514,432 KiB peak RSS**, under four-core affinity and an 8,000,000,000-byte address-space cap.

Separate sampled Linux research runs took about 30 seconds on the two large representative real cases with combined scores. These timings do not establish Windows-laptop performance.

Functional integration checks recorded **381 passed / one skipped** with optional packages, plus **136 network-denied targeted tests passed**. Ruff and Windows-targeted mypy passed. Base Windows CI passed; optional Windows ML execution remains unverified because workflow dispatch was denied. An authorized user can start the workflow with `optional_research=true`.

## 8. Where Steven should focus next

**First: protect discovery.** Inspect disagreement candidates, especially branches a filter removes, wall-parallel paths, early common trunks and nearby ostia. Record a hypothesis and a regression case before changing thresholds.

**When references arrive:** verify case IDs, physical coordinates, units, completeness and ignore/overlap rules; freeze source and predictions; score strict first, then the pool and optional models separately. Account for all prior real-case pseudo-label exposure. Use the official evaluator when supplied.

**Before submission:** measure the actual Windows laptop, prepare offline wheels, confirm output names and required case coverage, and retain the raw baseline JSONs. Presentation weighting and exact official matching still need the final organizer materials.

**Do not change the default just because a model agrees better with AI labels.** A learned filter needs evidence that it preserves eligible daughters and improves the complete-reference outcome.

Heavy 3D segmentation, reinforcement learning, regression heads, external-data training, quantization and out-of-fold stacking remain gated by labels, permissions or evidence. They have not been represented as completed accuracy improvements.

### Code and evidence map

| Read this | For this question |
| --- | --- |
| `detector.py`, `run.py` | Production stages, measurements, thresholds and output contract |
| `learning.py`, `labels/model-report.json` | Original feature model and pseudo-label split |
| `candidate_patches.py`, `tabular_learning.py`, `patch_inference.py` | Physical features, tree artifacts and CPU CNN scoring |
| `research_run.py`, `research_reproduce.py` | Optional inference and reproducible experiments |
| `research_validation.py`, `score_references.py`, `evaluate.py` | Complete-reference comparisons, matching and counts |
| `SUBMISSION_AUDIT.md` | Judge clarifications, runtime evidence and remaining requirements |
| `ADDITIONAL_PAPERS.md`, `RESEARCH_IMPLEMENTATION.md` | Paper adaptations and recommendation dispositions |
| `labels/research/current-source-v1/RESULTS.md` | Current-source counts, rejection decisions and detailed test evidence |
| `TEAMMATE_GUIDE.md`, `ANNOTATION_GUIDE.md` | Explorer/API operation and review procedures |

Repository: https://github.com/Coder-Meet/battleoftheschool

Reviewed implementation: https://github.com/Coder-Meet/battleoftheschool/tree/f43165b

Detailed research evidence: https://github.com/Coder-Meet/battleoftheschool/blob/f43165b/labels/research/current-source-v1/RESULTS.md
