# Topology evaluation — COMPLETE / DIAGNOSTIC ONLY

## Final status

The topology evaluation is complete. All **9 recovery variants × 5 real cases**
finished with zero failures, and the frozen synthetic gate contains the exact
**11 variants × 24 cases = 264/264 checkpoints**. Every recovery variant is
post-reference development and remains ineligible for frozen-family selection.
No continuation work or production promotion is pending.

Frozen source revision:
`76ce2f13b09da1f66cbeb6d8dd63dcdd5b00b1db`.
Frozen experiment config SHA-256:
`db719d635fd57f724cc4ede61edcdc1851d1688073c7c2af96f1da338bb96a94`.

## Real-case results

| Variant suffix (`topology-`) | 3 mm TP / FP / FN | F1 |
| --- | --- | ---: |
| alternate-roots | 8 / 14 / 11 | 0.3902 |
| connector-gap | 8 / 4 / 11 | 0.5161 |
| parallel-path | 8 / 4 / 11 | 0.5161 |
| contrast-support | 10 / 9 / 9 | 0.5263 |
| combined | 14 / 34 / 5 | 0.4179 |
| combined-no-roots | 11 / 13 / 8 | 0.5116 |
| combined-no-gap | 14 / 32 / 5 | 0.4308 |
| combined-no-parallel | 15 / 38 / 4 | 0.4167 |
| combined-no-contrast | 8 / 16 / 11 | 0.3721 |

None improves the strict real-case baseline at 3 mm (8 / 3 / 11, F1 0.5333).
The review-origin2 baseline is also not a replacement (9 / 14 / 10, F1 0.4286).

## Frozen synthetic gate

| Variant suffix (`topology-`) | 3 mm TP / FP / FN | F1 | Negative-control predictions |
| --- | --- | ---: | ---: |
| strict-audit | 46 / 0 / 3 | 0.9684 | 0 |
| review-origin2-audit | 47 / 0 / 2 | 0.9792 | 0 |
| alternate-roots | 47 / 9 / 2 | 0.8952 | 0 |
| connector-gap | 46 / 0 / 3 | 0.9684 | 0 |
| parallel-path | 48 / 3 / 1 | 0.9600 | 0 |
| contrast-support | 45 / 31 / 4 | 0.7200 | 14 |
| combined | 46 / 101 / 3 | 0.4694 | 30 |
| combined-no-roots | 47 / 31 / 2 | 0.7402 | 14 |
| combined-no-gap | 48 / 90 / 1 | 0.5134 | 30 |
| combined-no-parallel | 47 / 110 / 2 | 0.4563 | 34 |
| combined-no-contrast | 47 / 16 / 2 | 0.8393 | 0 |

The strict frozen snapshots are TP/FP/FN **45/1/4 at 2 mm** and **46/0/3 at
3 mm and 5 mm**, with zero strict detections on both `negative_controls_only`
cases. Contrast-support recovery causes major false-positive regressions,
including negative-control detections; combined variants amplify them. Alternate
roots and wall-parallel recovery also reduce precision. Connector-gap is neutral
on the synthetic gate and worse on real cases. These procedural regressions are
diagnostic robustness evidence, not independent clinical evidence.

## Durable evidence

- `experiment_config.json`: immutable nine-variant settings, 24-case synthetic
  cohort settings, source identities, and initial diagnosis identity.
- `baseline_report.json`: complete strict and review-origin2 real baselines.
- `report.json`: all nine complete real-case recovery variants.
- `synthetic/` and `synthetic_report.json`: all 264 checkpoints, aggregate
  2/3/5-mm scores, runtime maps, and zero failures.
- `diagnosis/`: per-candidate real-case diagnostic evidence.
- `resources/`: isolated process-tree samples for all 45 recovery runs; baseline
  RSS remains unknown and baseline runtime excludes loading/diagnosis.

The selector derives all 24 case IDs from the frozen settings, requires exact
variant/case path sets, replays every stored prediction against its stored
reference at 2/3/5 mm, and compares aggregate/runtime maps. It binds every
checkpoint and supporting artifact in the final evidence manifest.

## Verification and frozen-source behavior

Routine raw verification uses the stored evidence and performs no inference:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4 \
.venv313/bin/python final_eval_select.py verify
```

A current-main raw topology rerun is **intentionally rejected** by the frozen
source guard. `research_resources.py` changed after the freeze only to make CPU
limiting portable and report cooperative-only fallback honestly; its current hash
therefore differs from the hash in `experiment_config.json`. The selector permits
that documented wrapper mismatch only for replay of already-frozen evidence and
requires every other frozen topology source identity to match. It does not claim a
current-main inference rerun is equivalent.

Do **not** refreeze or bypass the source guard. If raw experiments must be repeated,
use an isolated detached checkout whose `HEAD` is exactly
`76ce2f13b09da1f66cbeb6d8dd63dcdd5b00b1db`, preserve the immutable experiment
config, set the four thread limits above, and run the frozen `experiments` or
`synthetic` command into isolated output. Do not overwrite accepted checkpoints to
reproduce timing. Any source or algorithm change requires a new explicitly
versioned experiment.
