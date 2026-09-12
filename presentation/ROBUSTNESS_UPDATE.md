# Later robustness audit — 2026-09-12

The existing deck and film preserve the earlier development evidence at
`29c843e`. The production detector was subsequently selected at `fd8f67f`
using development seed 4001 before inspecting the frozen evaluation sets.
See [the complete protocol](../ROBUSTNESS_PROTOCOL.md) for all variants,
source hashes, tolerance sensitivity and retained failures.

## Speaker 4: replacement for the 30-second slide 7 narration

> The slide shows our original five-case development experiment. In a later,
> frozen stress test across twenty-six procedural cases, we recovered
> fifty-six of sixty-four reference daughters, with one false positive.
> Small vessels in noise and thick slices still fail. All twenty-five supplied
> scans completed under four-core affinity, within twenty-four seconds and
> one-point-five GiB peak memory. Real accuracy still needs expert annotations.

Keep this inside the existing 30 seconds; do not add another segment after
the five-minute demonstration. The old slide numbers refer to the original
experiment; the supplement is the source for the later results.

## Questions and answers

- **Is it perfect?** No. The frozen set retains eight misses and one false
  detection. Synthetic micro precision is 98.25%, recall 87.5%, F1 92.56%.
- **What changed?** Contrast-normalized partial-volume support improved
  discovery while retaining strict connection, cap and topology checks.
- **What is the tradeoff?** Ten additional frozen references recovered
  compared with baseline, one extra false positive; matched geometric errors
  did not all improve.
- **Does that predict clinical accuracy?** No. Procedural phantoms do not
  reproduce full patient anatomy, scanner artifacts or clinical adjudication.
- **What is next?** Complete expert references, a frozen patient-level split,
  independent baselines and confirmation of the organizer's final thresholds.
