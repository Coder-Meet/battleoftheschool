# Frozen real-case predictions

Predictions for all 25 supplied cases, written before any organizer reference
was seen, so that reference scoring cannot be tuned after the fact. Each
directory is named `<commit>-<variant>`; the commit is the detector revision
that produced it.

| Directory | Command | Notes |
|---|---|---|
| `2a40d10-production` | `python batch.py --output-dir ...` (defaults) | Submission behaviour. Byte-identical to the earlier native-contrast release batch. |
| `2a40d10-wall-parallel` | `DetectorConfig(parallel_clearance_mm=2.5)` / `run.py --parallel-clearance-mm 2.5` | Opt-in experiment, see ROBUSTNESS_PROTOCOL.md. Differs from production in subjects 004, 005, 013, 017, 023, 025. |

Score either set the moment references arrive:

```bash
python score_references.py --references organizer-refs/ \
  --predictions frozen/2a40d10-production --data-root data \
  --output outputs/reference-score-production.json
python score_references.py --references organizer-refs/ \
  --predictions frozen/2a40d10-wall-parallel --data-root data \
  --output outputs/reference-score-wall-parallel.json
```

Do not edit these files. Add a new directory for a new revision instead.
