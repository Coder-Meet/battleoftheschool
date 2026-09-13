# Final five-case development evaluation

Frozen before model benchmarking. Baseline implementation: `4a43dd4`.

## Authority, data and metrics

The user explicitly confirmed **“The judge has approved these as scoring
references”** after being shown the package's draft warning. We therefore use the
19 released instances as scoring targets. The original AI-assisted authorship,
incomplete-annotation warning and absence of clinical sign-off remain in the
archived source; judge approval is not relabelled as clinical expert authorship.

Cases subject019–subject023 have 3, 4, 3, 6 and 3 references. Verify byte identity of
the CT and parent masks, all package hashes and physical SimpleITK LPS geometry.
Unknown radii remain unknown. No official evaluator or numeric overlap-matching
rule was supplied. Use our existing one-to-one ostium matcher at 2, 3 and 5 mm;
these are local comparison tolerances, not claimed official tolerances.

Report TP/FP/FN, precision/recall/F1, absolute and signed daughter-count error,
exact-count cases, matched ostium/seed/direction errors, known-radius error and
seed-to-reference-guide distance. Do not infer an official weighted final score.
Record CPU, time and memory. Final inference must remain offline, CPU-only and
compatible with Windows/four cores/8 GB.

## Declared experiment families

1. Frozen strict, review, review-union and all three existing paper adaptations.
2. Finite detector sensitivity grid: support-contrast fraction {0.3,0.5,0.7},
   native-contrast scale {0,1.2}, with strict and review-union proposals. Add
   working-spacing {0.75,1.5} and wall-parallel tracing as individually identified
   ablations. Preserve 5 mm/10 mm geometry; distinguish review proposals from
   final 2 mm origin eligibility.
3. Every saved logistic/tree artifact on compatible features: original,
   synthetic-augmented, synthetic logistic, gradient boosting and random forest
   with base/extended features. Refit real-label models with all five released
   cases excluded. Do not inherit old branch IDs as labels.
4. Current-source CNN and compatible frozen blends. Audit older CNN artifacts;
   evaluate them with matching extraction source or retrain rather than bypassing
   source checks. Explicitly record any artifact that cannot be validly compared.
5. Candidate-score thresholds: each frozen threshold plus
   {0.15,0.3,0.5,0.7,0.85}. Tree/CNN convex blend weights
   {0,0.25,0.5,0.75,1}, applied only to identical ordered proposals.
6. Root-connectivity/topology analysis and bounded general-purpose recovery
   experiments. These are post-reference development, reported separately from
   the frozen-family selection results.

No fixed per-case counts, reference-coordinate lookup, injected guide paths or
patient-specific rules may be used by any detector.

## Selection without reusing the scored case

Exclude all five reference cases from pseudo-label fitting and calibration in
new transferable models. Existing models that saw these images are
retrospective baselines, identified as such and ineligible for the clean
selection comparison.

For the finite compatible frozen matrix, leave one whole case out; select the
configuration on the remaining four; score it once on the held-out case.
Rank by pooled discovery F1 at 3 mm, then lower count MAE, then fewer runtime
dependencies/lower measured time. Include the unfiltered baseline and deterministic
tie-breaking. Report selected configuration per fold and selection stability.
For any fitting that uses reference outcomes, nest all training, calibration and
hyperparameter selection inside the four development cases; never use the held-out
case for preprocessing or calibration.

The best all-five aggregate is a **development winner**, not a held-out score.
Five reused development images cannot establish independent clinical or hidden
challenge accuracy. Report paired case-bootstrap uncertainty and sensitivity to
2/5 mm matching; do not claim bootstrap removes selection bias.

Only choose a deployment configuration after synthetic topology regressions and
resource checks. Do not promote an unstable complex model merely because it wins
one aggregate. Preserve exact artifacts, thresholds, source/input hashes, commands,
failed variants and exclusions. Keep changes together on main with no PRs or branches.
