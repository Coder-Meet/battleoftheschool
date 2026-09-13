# Deterministic family handoff — COMPLETE

## Final status

The immutable deterministic evaluation is complete: **26 extraction
configurations × 5 cases = 130/130 successful receipts**. It produced 52 scored
variants (26 final predictions and 26 diagnostic trace ceilings), with five
predictions per variant and local 2/3/5-mm scores. `failures.json`, the report
failure list, and the worker-failure list are empty.

Durable artifacts:

- `matrix.json`: frozen specification, inputs, references, source identities, and
  runner identity.
- `runs/`: all 130 inference receipts, including candidate audits, final
  predictions, runtime/RSS, and matrix/runner identity.
- `predictions/`: all 260 prediction files (52 variants × 5 cases).
- `report.json`: complete scores, eligibility, family LOCO, bootstrap, runtimes,
  prediction hashes, and limitations.
- `reference-diagnostics.json`: spatial diagnostics for every extraction
  configuration/reference pair.
- `failures.json`: zero final worker failures.

`checkpoint.json` is a superseded historical snapshot. The authoritative
completion state is the exact 130-receipt set plus `report.json`; no continuation
work remains.

## Frozen provenance

Raw inference was produced from detached frozen revision
`a3cf586031caf67b3f6ef7965ed30d0db717e257`. Recovery reused verified existing
receipts and generated only missing pairs under the frozen source guard. No guard
was bypassed, no empty prediction replaced a failure, and no frozen hash was
changed.

| Item | SHA-256 |
| --- | --- |
| Frozen runner | `69bc364d810bf402a7035e5abccfdf6a360cf6d5643dbda6ebaf5d6274233de1` |
| Frozen matrix | `fcf4c35fb1aacbd54141a1b09e2d69b9f889b4325c80ad3235c6063108aa8c38` |
| Production detector | `9f96f3eb3c06f5e06d5b7db79b199be70e742130fc3885243d25d066adc1c92e` |
| Shared scorer | `be2488c1123e0f3ae7cbff7a319ac360fab60c9502ebe5ce490bd7c918d7c990` |

Completion spans Linux and macOS receipts. The report-level `platform` and `cpu`
fields describe the report-generation host only; individual receipts did not
record host identity, so per-receipt host attribution is unavailable. Runtime is
therefore descriptive and is not organizer-Windows/four-core validation.

## Interpretation

At 3 mm, deterministic-family leakage-safe LOCO is TP/FP/FN **8/9/11**, precision
**0.470588**, recall **0.421053**, F1 **0.444444**, and count MAE **3.2**. The
all-five development-only winner is `deterministic-strict-spacing1.5`; that reused
maximum is not an independent held-out estimate.

The cross-family selector keeps `deterministic-strict` as the fixed deployment
algorithm only after exact matrix/source/receipt and synthetic topology-gate
replay. See `FINAL_EVALUATION_RESULTS.md` and
`labels/final-eval/selection/report.json`.

Origin-disabled rows and trace ceilings remain ineligible. Unknown origin diameter
is uncertainty, not evidence of a sub-2-mm origin. The 19 judge-approved targets
retain AI-assisted/non-exhaustive provenance, only three radii are measured, and
the scorer masks unknown-radius errors. No official weighted score, hidden-test
accuracy, clinical claim, or native organizer-hardware timing is available.

## Verification and optional raw rerun

Routine verification performs no inference and does not require refreezing:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4 \
.venv313/bin/python final_eval_select.py verify
```

Do **not** run `freeze` for final reproduction. If raw deterministic inference must
be repeated, use an isolated detached checkout whose `HEAD` is exactly
`a3cf586031caf67b3f6ef7965ed30d0db717e257`, preserve the existing immutable
`matrix.json`, use the four thread limits above, and run only the frozen runner's
`run` command into an isolated output. Do not replace the accepted receipts or
final report merely to reproduce timing. A source or algorithm change requires a
new versioned evaluation rather than edits to this frozen evidence.
