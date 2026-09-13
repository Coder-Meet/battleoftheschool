# Detector/filter plan — fusion restored

The user requested restoration of the highest observed five-reference F1 workflow on 2026-09-13. **The current default is strict + review proposals, native contrast scale 0.9, logistic scoring at 0.15 before strict-first 3 mm merging.** Its fresh replay returns 14/4/5, F1 0.75676 and count MAE 1.8; all five predictions exactly match the historical trial. This deliberately supersedes the default selected by the earlier strict release, without rewriting its frozen evidence.

Read [PRODUCTION_WORKFLOW.md](PRODUCTION_WORKFLOW.md) for settings and commands and [the validation receipt](docs/fusion-restored-validation.json) for current evidence.

| Original priority | Completed work | Current disposition / next step |
|---|---|---|
| Score alternate representations before merging | The trial integrated and replayed strict/review scoring at 0.15. Five-case output was 14/4/5, F1 0.757, on the earlier adaptive-cutoff source. It increased synthetic FPs versus strict-plus-filter. | Restored at the user’s request on 2026-09-13. Fresh five-case predictions exactly match the trial; default CLI, batch and Explorer use fusion. Strict baseline remains explicit. See the new validation receipt for current checks. |
| Stabilize filter features | Four fits used 140 exact current-at-the-time AI-reviewed candidates, with all five reference patients excluded. Contact-volume exclusion and training-recall constraints were tested. Best reference result was 8/3/11, F1 0.533; none beat that trial's selected model. | Experiments completed, no new weights promoted. Current source differs; re-extract identities/features before a new fit. A local physical volume feature and independent calibration remain research work. |
| Improve roots and tracing | Earlier alternate-root/parallel experiments reached at best 14/5/5, below the fusion trial. Later current-release work implemented guarded connected-origin recovery and tested 80 synthetic cases without losing baseline matches or adding FPs. | Recovery stays opt-in: it improves local F1 but worsens count MAE. Continue only against the declared count, topology and resource gates. |
| Expand adjudicated data | A 337-candidate review queue was prepared on the earlier source: 22 verdict/filter disagreements, 81 unknown survivors, 116 unknown rejected candidates, 118 agreements. No new annotation verdicts were fabricated. | That queue is archived and source-stale. Regenerate candidates on the frozen current source, then obtain new reviews and whole-parent sweeps for unproposed origins. Preserve patient splits and reviewer provenance. |
| Select with FP budgets and held-out patients | Threshold/retention comparisons and patient-level stability checks were completed for the earlier trial. Final release selection separately replayed 280 frozen variants and the 24-case topology gate. | Fusion is now the user-selected development default; the old frozen selector continues to describe strict. New data must be split before fitting/tuning; these repeatedly inspected five scans cannot become an independent test set. |

The earlier follow-up evidence is retained under `outputs/archive/pre-release-2a498a4/next-steps-v1/`. Those source-specific candidate queues must be regenerated before new annotation or training. Original CTs, released references and frozen strict evaluations are unchanged.

## Completed restoration

- Restored the bundled model and score-before-merge orchestration across CLI, batch, Python API and Explorer.
- Retained explicit strict-baseline mode and the guarded-recovery experiment; review mode remains unfiltered.
- Replayed all five references and checked exact geometry/ID equality to the prior fusion winner.
- Replayed the frozen 24-case synthetic cohort on both workflows: fusion 47/11/2 (F1 0.8785), strict 46/0/3 (F1 0.9684). Fusion adds seven FPs in the touching-vein/calcification case and four in the negative-control case at seed 31415; those regressions remain unresolved.
- Passed 460 Python tests (38 skipped), 36 frontend tests, lint/type checks and the frontend build.
- Added regression checks for scoring order, merge boundary, model integrity, shared settings, threshold overrides, strict baseline and recovery guards.
- Updated workflow, run commands and current documentation; marked strict release evidence and presentation as historical to this default.

## Prioritized next steps

1. Freeze this restored source and patient splits; regenerate the adjudication queue from its actual candidates. Review retained/rejected proposals and sweep whole parents for unproposed branches.
2. Prioritize the touching-vein/calcification and negative-control FP failures. Stabilize filter features using those reviewed candidates: evaluate local physical volume features and calibration. The earlier four fitted alternatives did not beat the selected synthetic model; no new weights were promoted.
3. Compare tracing/filter changes against fusion and strict together, with FP budgets, count MAE, 2/3/5 mm sensitivity and synthetic topology. Do not combine guarded recovery with fusion until independently evaluated.
4. Run actual Windows/four-core/8-GB acceptance and measure the two-pass runtime/memory. The old strict runtime measurements do not transfer to fusion.
5. Refresh presentation and submission packages from the new source/evidence before any new publication. Existing downloaded ZIPs and the 0.533 slide still represent the strict release.

The five references have already been used for tuning and selection; additional patients must be split before model fitting or threshold tuning to support an independent estimate.
