> Superseded fusion trial from commit `2a498a4`; reverted by `e06f8b0`. This is historical evidence, not the current release workflow.

# Detector and filter E2E review — 13 September 2026

The main bottleneck is proposal/tracing recall, followed by a fixable interaction between proposal merging and filtering. The current strict detector finds 11 of 19 supplied targets. A broad pool finds 14, but its filter discards two of those. A new **score-before-merge experiment retains all 14 with four false positives**, improving development F1 from 0.706 to 0.757 relative to the existing pool/filter workflow. Its synthetic FP regression means it should remain experimental.

This audit uses **main, commit `1b944925dc25690e9b1c1e6357583935ff2cb4b8`**, as confirmed by the user. It freshly processes the CTs, rather than reusing historical prediction metrics. Production detector settings and saved model weights were not changed.

The five judge-approved reference cases are subject019–subject023, with 3/4/3/6/3 targets. All release checksums and CT/parent-mask byte identities passed. Scoring is maximum-cardinality one-to-one physical-LPS ostium matching at 2/3/5 mm. “FP” below means unmatched against this released key; the AI-assisted source package retains an incomplete-annotation warning. These are reused development cases, not independent clinical or hidden-test results.

**Current detector/filter status.** The strict detector uses a 1 mm working grid and native-spacing-adaptive intensity cutoff (`native_contrast_scale=0.9`), wall-contact roots, supported proximal tracing and geometric rejection. The optional logistic filter uses 13 features and can only remove completed proposals. The default submission CLI has no filter. The synthetic logistic artifact has a saved threshold of **0.08833933**, whereas the stronger documented result uses **0.15**; passing the model to `run.py` does not silently select 0.15.

The saved original pseudo-label logistic and synthetic-augmented pseudo-label logistic contain reference cases in their historical train/validation splits. The refitted pseudo-label model excludes these five cases but uses old candidate features and still transfers poorly. The synthetic logistic has no five-case split overlap; this does not make the later detector tuning on these scans independent. All four saved current-source research trees reject the new detector hash. Their source guards should be preserved; regenerate compatible extraction/training artifacts before using them. CNN source compatibility is also documented as stale in `DETECTOR_FLOW.md`; CNN inference was not rerun in this audit.

**Fresh E2E results at 3 mm.** “Filter” in this table is the synthetic logistic, except where named otherwise.

| Pipeline | Outputs | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Previous intensity cutoff, replayed on current code | 11 | 8 | 3 | 11 | 72.7% | 42.1% | 0.533 |
| Current strict | 15 | 11 | 4 | 8 | 73.3% | 57.9% | 0.647 |
| Strict + saved filter threshold 0.08834 | 14 | 11 | 3 | 8 | 78.6% | 57.9% | 0.667 |
| Strict + filter 0.15 | 13 | 11 | 2 | 8 | 84.6% | 57.9% | 0.688 |
| Broad review/strict pool | 39 | 14 | 25 | 5 | 35.9% | 73.7% | 0.483 |
| Pool + filter 0.08834 | 17 | 12 | 5 | 7 | 70.6% | 63.2% | 0.667 |
| Pool + filter 0.15 | 15 | 12 | 3 | 7 | 80.0% | 63.2% | 0.706 |
| Pool + filter 0.05 | 19 | 13 | 6 | 6 | 68.4% | 68.4% | 0.684 |
| **Score before merge, filter 0.15 — experiment** | **18** | **14** | **4** | **5** | **77.8%** | **73.7%** | **0.757** |
| Strict + original pseudo-label model, saved threshold | 8 | 7 | 1 | 12 | 87.5% | 36.8% | 0.519 |
| Strict + five-case-excluded pseudo model, saved threshold | 2 | 2 | 0 | 17 | 100% | 10.5% | 0.190 |

The nine detector/profile settings and four logistic artifacts produced **279** complete profile/model/threshold variants. The subsequent merging experiment has 14 separately recorded variants. Neither model weights nor reference labels were fitted during this work.

| Case | References | Strict TP/FP/FN | Pool + 0.15 TP/FP/FN | Score-before-merge + 0.15 TP/FP/FN |
|---|---:|---|---|---|
| subject019 | 3 | 1/0/2 | 2/0/1 | 2/0/1 |
| subject020 | 4 | 0/1/4 | 2/0/2 | 2/0/2 |
| subject021 | 3 | 3/1/0 | 3/2/0 | 3/1/0 |
| subject022 | 6 | 6/2/0 | 5/1/1 | 6/3/0 |
| subject023 | 3 | 1/0/2 | 0/0/3 | 1/0/2 |

**Where the gap happens.** Strict proposal recall is 11/19 = 57.9%; its filter retains all 11 at 0.15. Pool recall is 14/19 = 73.7%, followed by filter retention of 12/14 = 85.7%, giving 63.2% end-to-end recall. A better filter alone cannot recover the five targets absent from the raw pool. Across the nine independently scored settings, the union of matched reference identities is still 14/19: five targets remain unmatched in every setting. A naive concatenation of all profiles counted one additional match by retaining duplicate representations; it is explicitly excluded from the coverage assessment. Neither this observed coverage nor the raw pool is a theoretical ceiling for a redesigned detector.

The instrumented strict resolver was checked for exact equality with current production outputs on all five cases before using its diagnostics. Nearby-root associations locate likely failure stages; they do not independently prove anatomical identity.

| Missing strict target | Evidence and failure stage | Recovered by current pool? |
|---|---|---|
| 019 / branch_001 | Root 3.09 mm from reference origin; `disconnected_ostium` despite substantial downstream support | No |
| 019 / branch_003 | Root 3.77 mm away; `short_proximal_segment` after tracing/junction handling | Yes |
| 020 / branch_001 | No nearby selected root (nearest 20.64 mm); guide remains within 2.32 mm of parent, below radial endpoint requirement | No |
| 020 / branch_002 | Root 3.99 mm away; `no_supported_5mm_path` | Yes |
| 020 / branch_003 | Root 2.30 mm away; `wall_hugging_path` rejects short ostium-to-seed chord | Yes |
| 020 / branch_004 | Root 2.85 mm away; `small_radius` rejects the traced seed estimate | No |
| 023 / branch_001 | Selected roots displaced over 5 mm along a shared contact; nearest 17.26 mm away; guide remains near wall | No |
| 023 / branch_003 | Root 2.89 mm away; `short_proximal_segment`; guide also stays near wall | No |

The five real scans have 1.5 mm native voxels. Partial volume and smoothing weaken narrow-vessel support; changing the intensity cutoff recovered three targets but added one FP compared with the old cutoff. Other losses arise from using radial clearance to find endpoints, a 3.5 mm chord requirement in addition to the 5 mm path requirement, seed-radius rejection, and estimated junction truncation. Improving these needs separate controlled experiments; a supplied reference's unknown radius is not evidence that its origin is undersized.

Simply loosening current rules performed poorly: contrast fraction 0.30 yielded **10/9/9**, six roots/contact **11/15/8**, parallel-clearance fallback **11/6/8**, and their tested combination **11/28/8**. Extra roots can follow the same connected blob or competing tissue. The parallel fallback only runs when no ordinary radial endpoint is available; it cannot rescue a missed origin that never gets a suitable root. Lowering thresholds can also merge support blobs or change the chosen path, so recall is not monotonic.

**A concrete filter/merge defect and measured improvement.** `detect_pool()` starts with the review profile and adds a strict candidate only when every review origin is at least 3 mm away. That selects a representation before the classifier evaluates it. The discarded strict representation can have a much better score:

| Reference | Strict contact volume / score | Pool contact volume / score | Outcome at 0.15 |
|---|---|---|---|
| 022 / branch_001 | 38 mm³ / 0.243 | 344 mm³ / 0.018 | Strict survives; pool loses it |
| 023 / branch_002 | 12 mm³ / 0.481 | 331 mm³ / 0.056 | Strict survives; pool loses it |

The contact-volume feature measures the proposal component, which can expand substantially under the review settings. Its negative logistic contribution combines with weak tubularity and evidence to suppress these targets. For 023, radius, mean tubularity and heuristic evidence are identical between representations. This is a feature-stability problem tied to proposal construction, not proof that the vessel disappeared.

`current_e2e_fusion.py` scores both profiles first, retains above-threshold branches, then applies the existing 3 mm merge distance. Strict-first and score-first choices produce the same pooled counts on these five cases. Every output geometry must equal an input branch geometry; references are used only in scoring. The experiment gains **two targets for one additional FP** over pool+0.15, and **three targets for two additional FPs** over strict+0.15.

It remains experimental. On the 14 existing procedural stress families (seed 4001; 34 analytic targets), strict+0.15 is **31/1/3**, pool+0.15 **31/2/3**, and fusion+0.15 **31/3/3**. Fusion preserves synthetic recall but adds FPs, including in the mural-thrombus and daughter-of-daughter scenarios. These procedural cases were previously exposed; they are regression evidence.

**Operating points for the scaling tradeoff.** For current strict and pool, 0.15 is a useful development operating point. Reducing pool threshold to 0.05 gains one target but adds three FPs. Fusion at 0.15 dominates that particular operating point here: 14/4 versus 13/6. For a tighter observed FP budget, fusion at 0.30 yields 11/2; at 0.50 it yields 9/0. These are measured options, not calibrated probabilities or guaranteed FP rates.

![Measured target-retention versus FP tradeoff](../../outputs/archive/pre-release-2a498a4/current-e2e-20260913/tradeoff.png)

**Training-data scale and label drift.** Across all 25 CTs:

| Current proposal population | Before filter | At saved 0.08834 | At 0.15 | At 0.30 | At 0.50 |
|---|---:|---:|---:|---:|---:|
| Strict | 157 | 143 | 136 | 123 | 101 |
| Broad pool | 337 | 169 | 154 | 126 | 96 |

These are candidate counts, not 25-case TP estimates: only five cases have released references. The historical review set contains 237 AI decisions: 92 confirmed and 145 rejected. On today's pool, **140** candidates retain exact old geometry/features (57 confirmed, 83 rejected), while **197** need review of their current identity. Whole-fingerprint numeric comparison agrees with these counts; an all-field 1e-8 roundoff check restores no additional matches. Do not transfer reviews by branch number or nearest origin.

Of the 154 pool survivors at 0.15, 54 have an exact previous confirmation, 19 an exact previous rejection, and **81 are unreviewed current candidates**. The filter also drops three exact previously confirmed candidates. These disagreements concern AI reviews, not independent truth. They form a useful review queue; survivors should not automatically become positive training labels, and unmatched reference candidates should not automatically become negatives.

Prioritized next work:

1. **Preserve and score alternate representations before merging.** The experiment provides a concrete candidate implementation and five-case outputs. Next resolve the synthetic FP regressions and evaluate on a newly frozen patient set before enabling it in production.
2. **Make filter features stable across proposal profiles.** Replace whole-component volume with a local proximal measurement, or explicitly model profile/measurement provenance; re-extract candidate features on frozen current source before retraining. Include weak thin branches as positives and wall/vein/secondary-branch confounders as reviewed hard negatives.
3. **Improve tracing where it actually loses targets.** Investigate proximal root coverage in long merged contacts, simultaneous geodesic and radial endpoint searches, and confidence-aware junction/radius handling. Preserve the 5 mm path and 2 mm origin rules. Avoid a global lower intensity threshold as the only intervention.
4. **Expand adjudicated data with a review budget.** Review the 81 unknown survivors first for positive yield, plus the 19 retained prior rejections and three removed prior confirmations for errors. Sample low-score/filtered-out candidates and perform whole-parent sweeps to find unproposed vessels. Otherwise training repeats the detector's blind spots. The other 116 unknown pool candidates are a useful stratified negative/uncertainty sampling population.
5. **Select using FP budget and target retention on patient-held-out data.** Track proposal recall, filter retention, final TP/FP/FN, and reviewed positives per annotation hour. Keep all candidates and scores available for annotation. Freeze patient splits before new labels; do not choose a threshold solely by aggregate F1 on these same five scans. Retrain source-bound tree/CNN artifacts only with compatible extraction and provenance.

**Precision of the evidence and verification.** At 2 mm, strict is 7/8/12, pool+0.15 is 7/8/12, and fusion+0.15 is 8/10/11. At 5 mm, the corresponding results equal 3 mm results. Thus localization remains consequential: fusion goes from 14 to eight matched targets when tolerance tightens. For strict's 11 matched pairs at 3 mm, mean ostium error is 1.64 mm, seed error 1.36 mm and direction error 13.76°; only one matched radius is known (0.26 mm error). Fusion's mean ostium error is 1.66 mm over 14 pairs. Count MAE is 2.0 for strict+0.15, 1.6 for pool+0.15, and 1.8 for fusion+0.15; improved discovery F1 does not improve every metric.

Retrospective leave-one-case-out selection among the initial unfiltered/synthetic-filter matrix yields 9/6/10, F1 0.529, with unstable choices. This is a selection-stability warning, not clean held-out accuracy: the configuration design already used these cases. Paired five-case bootstrap 95% intervals for fusion's F1 change include zero: −0.072 to +0.392 versus strict+0.15, and −0.017 to +0.205 versus pool+0.15. Bootstrap does not remove the adaptive experiment's selection bias.

Ten actual `run.py` invocations (five plain, five saved-threshold-filtered) succeeded and exactly matched API outputs. Strict inference averaged 2.48 seconds, maximum 5.44 seconds on this Mac, excluding file loading. These are local four-thread observations, not Windows/four-core/8-GB acceptance measurements; current memory limits were not benchmarked. The focused detector, geometry, scorer, learning and topology suite passed **78 tests**, and the new fusion/duplicate-coverage regression suite passed **four**. Four existing tree integration tests fail because saved tree contracts pin the previous detector source. The optional pinned `psutil` dependency was installed to run that check. No source guard was bypassed.

Reproduce with a fresh output directory:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4 \
  .venv/bin/python current_e2e_audit.py --output-dir outputs/current-e2e-repeat --all-cases
.venv/bin/python current_e2e_fusion.py \
  --audit-dir outputs/current-e2e-repeat --output-dir outputs/current-e2e-repeat/fusion
```

Detailed local artifacts are in `outputs/current-e2e-20260913/`: `completion.json`, `provenance.json`, `release-verification.json`, `input-audit.json`, `metrics.json`, `summary.csv`, `candidates.csv`, per-case prediction/trace/diagnosis JSONs, ten CLI equality receipts, `inventory.csv`, `review-queue.csv`, `label-identity-supplement.json`, and the fusion predictions/stress/bootstrap reports. The two audit scripts and their regression tests are new reviewable workspace files. Existing frozen reports and production code remain unchanged.
