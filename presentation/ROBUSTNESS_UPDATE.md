# Later robustness audit — 2026-09-12

The existing deck and film preserve the earlier development evidence at
`29c843e`. The production detector was subsequently selected at `fd8f67f`
using development seed 4001 before inspecting the frozen evaluation sets.
See the complete protocol (removed in the 2026-09-13 cleanup; see git history) for all variants,
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

## Current default: native-resolution support at `09f753b`

Use this narration in the **same 30-second slot** when demonstrating the current
default. The film and original slide metrics remain historical evidence.

> We tested our teammate's contrast proposal on newly frozen procedural cases.
> The conservative version recovered two additional origins without adding
> false positives: fifty-three of sixty-three references, with one false
> detection. Ten misses remain. All twenty-five real scans completed locally
> under the CPU and memory limits. These are engineering checks; real accuracy
> still needs complete expert annotations.

This is a different 26-case draw from the earlier frozen set, not a replacement
of its score. Stronger threshold relaxation produced false positives and was
rejected. Before presenting, finish the coarse-scan 3D/selection/export smoke
on the demo laptop: the final cloud-browser run stalled after rendering the
Subject018 CT panels. See the submission audit (removed in the 2026-09-13 cleanup; see git history).
