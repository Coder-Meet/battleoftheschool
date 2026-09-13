> Historical strict-release evidence: the current default was restored to fusion on 2026-09-13. See [PRODUCTION_WORKFLOW.md](PRODUCTION_WORKFLOW.md) for current settings and validation. Figures and decisions below remain tied to their original source.

# Final evaluation results

## Recommendation

**Keep the fixed production `deterministic-strict` algorithm.** Its deployment eligibility and origin-policy certification are derived from an exact full production configuration, pinned frozen specification, raw deterministic receipts, and the replayed topology gate—not from its name. The global LOCO result below estimates a selection procedure whose five predictions can come from different algorithms; it is not a single deployable configuration. The all-five winner is development-only.

No official evaluator was supplied, so no official weighted challenge score is reported. These are local maximum-cardinality one-to-one ostium matches against 19 judge-approved but AI-assisted/non-exhaustive targets.

## Leakage-safe global leave-one-case-out selection

Validated 280 variants from all five required reports; 180 were eligible clean frozen candidates. Each fold was ranked using only the other four cases: pooled 3-mm F1, count MAE, effective model dependencies, algorithm components, measured runtime, then global lexical name.

| Held out | Four-case selected variant | Dev F1 | Dev count MAE | Dependencies | Dev runtime (s) | Held-out TP/FP/FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| subject019 | `tabular:tabular-current-source-v1-random_forest-extended-review-union-t0.84999999999999998` | 0.6923 | 1.5000 | 1 | 38.885 | 0/0/3 |
| subject020 | `tabular:tabular-current-source-v1-gradient_boosting-base-review-union-t0.56300768295780967` | 0.7200 | 1.2500 | 1 | 24.327 | 0/1/4 |
| subject021 | `deterministic:deterministic-strict-contrast0.3-native1.2` | 0.5333 | 2.5000 | 0 | 13.427 | 2/3/1 |
| subject022 | `deterministic:deterministic-strict-spacing1.5` | 0.5000 | 1.5000 | 0 | 5.151 | 6/5/0 |
| subject023 | `tabular:tabular-current-source-v1-logistic-strict-t0.29999999999999999` | 0.6667 | 2.0000 | 1 | 13.721 | 0/0/3 |

Selection stability: `tabular:tabular-current-source-v1-random_forest-extended-review-union-t0.84999999999999998` 1/5, `tabular:tabular-current-source-v1-gradient_boosting-base-review-union-t0.56300768295780967` 1/5, `deterministic:deterministic-strict-contrast0.3-native1.2` 1/5, `deterministic:deterministic-strict-spacing1.5` 1/5, `tabular:tabular-current-source-v1-logistic-strict-t0.29999999999999999` 1/5.

| Tolerance | Composite TP/FP/FN | Precision | Recall | F1 | Count MAE | Mean signed count bias | Strict F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 mm | 7/10/12 | 0.4118 | 0.3684 | 0.3889 | 3.2000 | -0.4000 | 0.4000 |
| 3 mm | 8/9/11 | 0.4706 | 0.4211 | 0.4444 | 3.2000 | -0.4000 | 0.5333 |
| 5 mm | 8/9/11 | 0.4706 | 0.4211 | 0.4444 | 3.2000 | -0.4000 | 0.5333 |

At 3 mm, the fixed-decision paired whole-case bootstrap (10,000 resamples, seed 20260913) gives a composite-minus-strict F1 interval of [-0.3173, 0.0188]. With only five reused cases this is descriptive uncertainty, not a correction for selection bias.

Exact gained/lost reference IDs versus strict at 2/3/5 mm are preserved in `labels/final-eval/selection/report.json`.

## All-five development winner (not held out)

`tabular:tabular-current-source-v1-random_forest-extended-review-union-t0.84999999999999998` is the descriptive all-five maximum. It is explicitly **not independent held-out accuracy** and is not the deployment recommendation.

| Tolerance | TP/FP/FN | Precision | Recall | F1 | Count MAE | Origin-policy certified |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 mm | 7/3/12 | 0.7000 | 0.3684 | 0.4828 | 1.8000 | false |
| 3 mm | 9/1/10 | 0.9000 | 0.4737 | 0.6207 | 1.8000 | false |
| 5 mm | 9/1/10 | 0.9000 | 0.4737 | 0.6207 | 1.8000 | false |

Its exact threshold/weight, detector settings, model/source hashes, prediction hashes, and runtime are in the machine-readable report. The complete clean and retrospective rankings are in `rankings.json`.

## Origin-certified challengers without symmetric promotion evidence

Some origin-certified candidates score higher than strict on the same five reused development cases. They remain unpromoted because they lack a symmetric promotion gate, not because their descriptive score is lower:

| Challenger | All-five 3 mm F1 | Origin certified | Model dependencies | Symmetric topology/native-resource gate |
| --- | ---: | --- | ---: | --- |
| `deterministic:deterministic-strict-spacing1.5` | 0.5946 | true | 0 | no |
| best strict-proposal learned rows (14 tied rows; e.g. `cnn:cnn-current-strict-w0-t0.3`) | 0.5926 | true | 1 | no |

`deterministic-strict-spacing1.5` is approximately F1 0.5946; the best strict-proposal learned rows are approximately F1 0.5926. Neither class was exercised through the exact same frozen 24-case topology gate and native organizer hardware measurement as a promotion candidate, so no symmetric replacement decision is supported.

## Synthetic topology and deployment gate

The selector enumerated and semantically replayed the exact frozen 11×24 checkpoint tree (264/264), including case/config/root-separation/runtime identity, 2/3/5-mm scores, aggregate/runtime maps, and zero failures. At 3 mm:

| Variant | TP/FP/FN | F1 | Count MAE |
| --- | ---: | ---: | ---: |
| `topology-strict-audit` | 46/0/3 | 0.9684 | 0.1250 |
| `topology-review-origin2-audit` | 47/0/2 | 0.9792 | 0.0833 |
| `topology-alternate-roots` | 47/9/2 | 0.8952 | 0.4583 |
| `topology-connector-gap` | 46/0/3 | 0.9684 | 0.1250 |
| `topology-parallel-path` | 48/3/1 | 0.9600 | 0.1667 |
| `topology-contrast-support` | 45/31/4 | 0.7200 | 1.4583 |
| `topology-combined` | 46/101/3 | 0.4694 | 4.0833 |
| `topology-combined-no-roots` | 47/31/2 | 0.7402 | 1.3750 |
| `topology-combined-no-gap` | 48/90/1 | 0.5134 | 3.7083 |
| `topology-combined-no-parallel` | 47/110/2 | 0.4563 | 4.5833 |
| `topology-combined-no-contrast` | 47/16/2 | 0.8393 | 0.5833 |

Strict had 46/0/3 (F1 0.9684) and zero negative-control detections. Although review-origin2 reached 47/0/2, topology recovery variants were post-reference diagnostics; several added substantial synthetic false positives. None is promoted. Final learned candidates were not all exercised through this identical gate, so score gains alone cannot certify deployment.

The deployment bundle contains five predictions from the single fixed strict algorithm at `labels/final-eval/selection/predictions/deployment/`. The held-out fold composite is separately stored under `predictions/held-out/` and must not be submitted as one algorithm.

## Resources, exclusions, and limitations

| Configuration | Mean runtime/case (s) | Runtime range (s) | Maximum peak RSS (MiB) |
| --- | ---: | ---: | ---: |
| Fixed strict | 3.278 | 1.564–7.169 | 624.57 |
| Development-only winner | 8.476 | 3.194–20.656 | 631.49 |

Strict also outperformed the adaptive held-out composite at 3 mm: 8/3/11, F1 0.5333 and count MAE 2.0 versus 8/9/11, F1 0.4444 and count MAE 3.2. The all-five development winner's higher F1 0.6207 is retrospective and uses the relaxed review-union origin policy.

Runtime scopes differ by family (deterministic source inference/audit, tabular filtering/model scope, CNN cold worker extraction, and topology stage timing), so runtime is only a late tie-breaker. Linux/macOS measurements do not establish native organizer Windows/four-core/8-GB performance.

All upstream failure lists were empty at aggregation. Incompatible old tree/blend artifacts, historical models with released-case overlap or source-contract mismatch, origin-disabled rows, trace ceilings, all post-reference topology rows, and the audit duplicate were excluded from clean selection. Review-union rows remain research candidates where their frozen report permits them, but are not final-origin-policy certified.

Only 3 of 19 reference radii are measured; the other 16 remain unknown and are masked from radius error. Unmatched predictions are false positives only relative to the non-exhaustive local reference set. Five previously inspected cases cannot support hidden-test, clinical, or external-generalization claims.

## Reproduction and provenance

Run `.venv313/bin/python final_eval_select.py verify` under the four-thread environment to replay all 1,400 source prediction files, all 130 deterministic receipts and 260 deterministic outputs, and all 264 topology checkpoints; it also validates family lineage, recomputes every 2/3/5-mm score, and verifies the complete evidence manifest. Canonical pinned Git LFS pointers are accepted for selector-only identity replay and are reported as unresolved; inference still requires payload bytes. The evaluated scorer is pinned to `be2488c1123e0f3ae7cbff7a319ac360fab60c9502ebe5ce490bd7c918d7c990` and the production detector to `9f96f3eb3c06f5e06d5b7db79b199be70e742130fc3885243d25d066adc1c92e`.

The historical audited download source identifier is `EVAL_SET-20260913T015512Z-1-001.zip`, SHA-256 `7cec39de73c8447864b6103f858768781a201279ea808bd75f153f6fd5eecdbc`. The selector does not reopen that archive; it validates the portable basename/digest record and repository evidence. Exact per-file evidence hashes and tree-root digests are in `labels/final-eval/selection/manifest.json`.
