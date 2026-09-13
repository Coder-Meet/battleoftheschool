> Historical strict-release evidence: the current default was restored to fusion on 2026-09-13. See [PRODUCTION_WORKFLOW.md](PRODUCTION_WORKFLOW.md) for current settings and validation. Figures and decisions below remain tied to their original source.

# Accuracy recheck: a useful recovery method, with a scoring tradeoff

## Decision

**Keep strict as the submitted default.** A new guarded wall-connection method
improves local precision from **0.7273 to 0.7500** and F1 from **0.5333 to
0.5806**, preserving every baseline match. However, daughter-count MAE rises
from **2.0 to 2.2**. The judge emphasized counts, so this fails the experiment's
predefined promotion gate. The release ZIP and Explorer default remain strict.

The new method is available on main as an explicit experimental CLI option:

```bash
python run.py --image image.nii.gz --aorta-mask mask.nii.gz \
  --output experimental.json --diagnostics experimental-diagnostics.json \
  --threads 4 --recover-connected-origins
```

Omit `--recover-connected-origins` for the submitted algorithm. No learned
weights, patient-specific thresholds, reference coordinates or fixed daughter
counts are used during recovery.

This is evidence that the detector has room to improve. It does not establish
a higher official challenge score or accuracy on the unseen cases.

## Where the strict misses occur

At the local 3 mm matching tolerance, strict matches all three references in
subject021 and five of six in subject022. It matches none in subjects019,
020 or 023. Eleven missed references have these nearest-candidate diagnoses:

| Nearest-candidate observation | Misses |
|---|---:|
| No supported path reaching the endpoint criterion | 4 |
| Straight connector to the parent wall fails | 3 |
| No candidate root within 5 mm of the ostium or guide | 2 |
| Wall-hugging path rejected | 1 |
| Seed radius rejected | 1 |

These are associations from the frozen diagnostics, not clinical adjudications.
Five missed guides have enhanced support at fewer than half their samples.
There is no indication that a learned filter is responsible for these baseline
misses: strict does not use one. A classifier that only removes candidates
cannot recover an absent proposal or a path rejected before classification.

The full per-reference records and input hashes are in
[the audit](labels/finalization/accuracy-recheck/audit.json).

## The structural change

The original wall test tries two straight connectors: the reverse proximal
tangent and the nearest parent voxel. A genuinely supported curved connector
can fail both.

The experimental recovery:

1. Leaves the original proposals and accepted branches in place.
2. Considers only proposals rejected as `disconnected_ostium`.
3. Finds a face-connected route through existing support, outside the parent
   and excluded caps, to a voxel adjacent to the parent wall.
4. Limits that connector to 6 mm, matching the existing wall-search extent.
5. Retraces from the wall contact through the existing strict pipeline.
6. Requires the retraced centerline to pass within the measured local lumen
   radius of the **original proposal root**.
7. Applies the existing size, path, intensity, tubularity, common-trunk and
   duplicate checks.

It does not fill gaps, lower contrast thresholds or waive the 2 mm origin and
5 mm continuation rules.

The root constraint matters. The first version could relocate to a valid wall
contact and then follow a different vessel. It added two false positives on
new procedural cases. Their new paths missed the original roots by 2.0 and
1.63 mm despite local radii of only 1 mm. Requiring the recovered path to retain
the original root removed these errors. This constraint uses the existing
measured radius, not a threshold fitted to those errors.

The refinement was made after observing those failures. They remain development
data. A second set of procedural seeds was reserved in the refinement's source
before running it.

## Measured reference results

All rows use the same five judge-approved, AI-assisted/non-exhaustive references
and the same one-to-one 3 mm scorer.

| Algorithm | TP / FP / FN | Precision | Recall | F1 | Count MAE |
|---|---:|---:|---:|---:|---:|
| Strict, 1 mm | 8 / 3 / 11 | 0.7273 | 0.4211 | 0.5333 | 2.0 |
| Unguarded connection recovery | 9 / 4 / 10 | 0.6923 | 0.4737 | 0.5625 | 2.0 |
| Strict, 0.75 mm | 9 / 3 / 10 | 0.7500 | 0.4737 | 0.5806 | 2.2 |
| 0.75 mm plus unguarded recovery | 10 / 5 / 9 | 0.6667 | 0.5263 | 0.5882 | 2.4 |
| **Guarded connection recovery, 1 mm** | **9 / 3 / 10** | **0.7500** | **0.4737** | **0.5806** | **2.2** |

The guarded method recovers subject022/branch_001. It adds no other reference
matches and no new reference false positives. Subject022 already had seven
predictions for six annotations; the extra matched branch raises its predicted
count to eight. This explains why F1 improves while count MAE worsens.

This one-case gain is too little evidence to infer a reliable hidden-case gain.
The unlisted predictions may themselves include unannotated eligible vessels;
these reference scores do not establish clinical false-positive rates.

## Robustness checks

| Cohort | Strict TP / FP / FN | Guarded TP / FP / FN |
|---|---:|---:|
| Frozen topology, 24 cases | 46 / 0 / 3 | 46 / 0 / 3 |
| First additional procedural set, 28 cases | 57 / 0 / 7 | 57 / 0 / 7 |
| Second procedural set, 28 cases | 62 / 1 / 9 | 62 / 1 / 9 |

Across these **80 synthetic cases**, guarded recovery loses no baseline matched
instance, adds no false positive and produces no negative-control detections.
Checks compare individual matched IDs as well as aggregate counts.

The alternatives expose why the extra testing was necessary:

- The unguarded 1 mm method passes the frozen cohort but adds two false
  positives on the first additional set.
- The 0.75 mm grid also passes the frozen cohort, but loses two true detections
  on the first additional set, including a thick-slice case.
- Combining the finer grid with unguarded recovery adds four false positives
  on the frozen cohort, including a negative-control detection.

The second procedural set uses seeds 28412 and 91514 across all 14 stress
families. These cases were not inspected before freezing the root constraint,
but they use familiar generators. They do not substitute for new patients.

## CLI, offline and resource checks

The opt-in CLI completed all 25 supplied scans with networking denied by
`strace`, four-thread library limits and OS affinity to CPUs 0–3:

- Maximum measured process runtime: **23.69 seconds**, including CLI startup.
- Maximum sampled process-tree RSS: **1476.23 MiB**.
- Output: **162 daughters**, compared with 150 in the released strict package.
- Twelve additions across nine cases; every original daughter's measurements
  remain exactly unchanged apart from instance renumbering.
- Only one addition is matched by the released references. The other eleven
  are unadjudicated and must not be called accuracy improvements.

All outputs passed coordinate finiteness, direction, radius, parent-link and
unique-instance checks. The default CLI still reproduces frozen strict output;
the opt-in CLI reproduces the guarded reference checkpoint.

These are Linux measurements. They do not establish native organizer-Windows
timing. See [the resource report](labels/finalization/accuracy-recheck/resources/report.json)
for commands, source hashes, per-case records and predictions.

Local checks: Ruff, configured and explicit Mypy checks, **477 passing tests**
(11 skipped), and the frozen selector verification of 280 variants / 180
eligible. The five new unit tests cover curved connectors, gaps, excluded caps,
physical distance and original-root identity in a rotated image frame.

## Reproduction and evidence

The first comparison is preserved from commit `987e9a4`; the guarded refinement
is preserved from `2acc79c`. Each run records its source hashes, exact cases,
configuration, complete predictions, diagnostics and scores at 2/3/5 mm.
All **398 prediction records** were rescored at all three tolerances; aggregate
scores and source hashes were checked against the recorded commits.

Evidence:

- [Initial references](labels/finalization/accuracy-recheck/v1-references/report.json)
- [Initial topology](labels/finalization/accuracy-recheck/v1-topology/report.json)
- [Initial additional cases](labels/finalization/accuracy-recheck/v1-fresh/report.json)
- [Guarded references](labels/finalization/accuracy-recheck/v2-references/report.json)
- [Guarded topology](labels/finalization/accuracy-recheck/v2-topology/report.json)
- [Guarded first additional set](labels/finalization/accuracy-recheck/v2-fresh/report.json)
- [Guarded second additional set](labels/finalization/accuracy-recheck/v2-confirmation/report.json)
- [Replay, per-case regressions and promotion gates](labels/finalization/accuracy-recheck/audit.json)

Use new output directories; the runner refuses to overwrite prior evidence:

```bash
python accuracy_recheck.py --cohort references --output outputs/recheck-initial
python accuracy_recheck.py --guarded-only --cohort references --output outputs/recheck-reference
python accuracy_recheck.py --guarded-only --cohort topology --output outputs/recheck-topology
python accuracy_recheck.py --guarded-only --cohort fresh --output outputs/recheck-first
python accuracy_recheck.py --guarded-only --cohort confirmation --output outputs/recheck-second
```

Set `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and
`ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS` to 4 before launching these commands.
The recorded cohort runs also used Linux OS affinity for CPUs 0–3. Timings in
those cohort files exclude process startup and image loading.

## What would justify the next promotion

The main remaining opportunity is supported branch discovery in subjects019,
020 and 023, especially weak-contrast and wall-parallel continuations. Another
filter trained on the current proposals does not address this failure stage.

The highest-value additional evidence is complete review of those missed
openings and the unlisted strict predictions, followed by evaluation on
previously unused patients with different contrast and slice thickness.
Until the official evaluator is available, improving local F1 and improving the
judge's count-heavy score cannot be assumed to be the same objective.

Exceeding 0.7 is not a definition of overfitting. Selecting increasingly complex
rules on the same five cases creates selection bias regardless of whether the
result is 0.58, 0.7 or 0.9. The rejected alternatives and the optional status of
the guarded method are retained so that the team can see the tradeoffs.
