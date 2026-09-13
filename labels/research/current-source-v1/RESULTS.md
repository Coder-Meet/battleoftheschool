# Current-source ML research integration

**Implemented, verified and research-only. No learned model is promoted.**
Production inference, existing labels and prediction fields are unchanged.
Functional integration was verified at `ef344557650c1b2b6faf924275217df447c7ec96`.
This directory contains compact frozen models and reports; the attachment
contains source, detailed checks and comparison inputs, without CTs or wheels.

[Download the verified bundle](https://app.devin.ai/attachments/d986927b-1ad8-4d7a-97c6-6cc1b79a86d4/branchseed-research-integration-v1.zip)
or read the [handoff](https://app.devin.ai/attachments/fb3980ad-b1e0-4d58-bce3-ab6ad6b02742/HANDOFF.md).
The bundle is 6,137,260 bytes, with 2,728 individually verified files;
SHA256 `7b24dc5c1819bfc846aa1176c965e440b6fb66c7862421e4eb3e488f1c103a46`.
Its relocated reference-comparison command reproduced every run metric and
promotion decision. The archived report predates only this download paragraph.

## Source and experiment boundary

| Source | SHA256 |
| --- | --- |
| `detector.py` | `9f96f3eb3c06f5e06d5b7db79b199be70e742130fc3885243d25d066adc1c92e` |
| `candidate_patches.py` | `568d8bd9f0d417768cb22fdd56d054a26ca215cc1741d3d9914ff4052eaa534a` |
| `learning.py` | `d7acdaa6f78ea9f3b6703d0b457880de37804ee0fc06ff6aeb4c9bfe7fe82c1e` |
| `run.py` | `6827f2df94282aa5d0a0c1cbff4b90e561528802d5fbbc3c4aefeb87ee881a55` |

The confirmed strict policy is **2 mm origin diameter**, separately from
0.7 mm seed radius. Review uses the existing diameter override of zero.
No detector logic was changed by this integration. The previous corpus used
detector hash `17495723df18c79e4302edc19bd949bb16f615b902403a88c68d6e74fa200063`.
Its models are not silently applied to current-source image inference:
the actual historical tree is rejected for a missing verified contract and the
actual historical CNN is rejected for extraction-source mismatch.

The 210 cases reuse the already exposed 15 seed groups and 14 families.
Fresh generation, detection, physical extraction, tree training and optional
CNN execution completed under the current source/configuration. The matching
counts reproduce the earlier results; this is now observed from fresh exports,
not inferred from the historical archive. Analytic references are retained
without retrospective relabelling. They are neither clinical references nor
official 2 mm eligibility validation.

`plan.json`, `models/freeze.json`, `reproduction.json` and `summary.json`
record source/dependency/configuration hashes, train/validation/test sequencing
and frozen weights. Model selection was unchanged after tree test evaluation.
CNN execution was explicitly retrospective, after E1 allowed research-only
work; this is not a new sealed test or an externally timestamped preregistration.

The full-corpus audit passed: **210 input/export receipts, 437 SHA256 sidecars**,
all partition-to-case patch identities, finite float32 numeric caches loaded
with `allow_pickle=False`, disjoint groups and frozen model/input hashes.
Cache shapes are train `(307,12,32,32)`, validation `(95,12,32,32)`, and
test `(66,12,32,32)` across 10/3/2 seed groups. Complete CT/NPZ files are omitted
from the compact handoff; regenerate a fresh corpus before running its full audit.

## Observed discovery counts and gates

Counts are TP/FP/FN against all analytic branch references at the disclosed
3 mm proxy tolerance, including missed branches never proposed.

| Variant | Validation, 42 cases | Replayed test, 28 cases |
| --- | --- | --- |
| Strict classical | 91/0/12 | 61/1/9 |
| Review union / pool | 93/4/10 | 63/4/7 |
| Gradient boosting, base, selected | 93/1/10 | 63/0/7 |
| Gradient boosting, extended | 93/2/10 | 63/0/7 |
| Random forest, base | 93/1/10 | 63/0/7 |
| Random forest, extended | 93/1/10 | 63/0/7 |
| Logistic | 93/2/10 | 63/3/7 |
| Synthetic CNN | See frozen training report | 59/2/11 |
| Frozen tree/CNN blend | See frozen training report | 62/0/8 |

Across all 210 cases strict is **437/5/72**, pool **453/28/56**.
The selected tree removes four test pool FPs without adding misses; it cannot
recover the seven unproposed test references. E1 is `go_research_only`.
Extended gradient boosting fails its validation gate; no extra-feature benefit
was demonstrated. Only ten rejected training rows and two test groups limit
generalization.

The CNN loses four pool TPs and the blend loses one, so both are **rejected for
promotion without retuning**. The blend chooses CNN weight **0** and tree
threshold `0.7008606038842237`; it demonstrates no CNN ensemble benefit.
Historical mixed-CNN results remain separate: 58/2/12 and rejected, with all
25 previously inspected/pseudo-trained real cases retained in provenance.
No new labels, mixed retraining or clinical transfer experiment was performed.

The separate `research_validation.py` comparison completed with **zero
reference errors and zero run errors** across strict, pool, logistic and four
trees. All promotion gates reject: synthetic rather than complete expert
provenance, exposed cases, incomplete historical knowledge and missing
organizer-Windows compute evidence. Pool/logistic additionally fail relevant
FP/paired-improvement checks. Scores are research evidence, not clinical accuracy.
The full E0–E4/action status table is in `RESEARCH_IMPLEMENTATION.md`.

## Exact verification

| Check | Observed result |
| --- | --- |
| Ruff | Passed |
| Full mypy | Passed, 27 source modules |
| Full `mypy --platform win32` | Passed, 27 source modules |
| Base-only pytest | **344 passed, 38 skipped, 49 warnings, 141.17 s** |
| Full optional pytest | **381 passed, 1 skipped, 53 warnings, 150.25 s** |
| Network-denied targeted optional pytest | **136 passed, 4 warnings, 37.09 s** |
| Current trained CNN native/ONNX parity | Max absolute error **1.1920928955078125e-7**; threshold decisions equal; empty/batched CPU inference passed |
| Historical source-boundary rejection | Actual archived tree and CNN rejected explicitly |
| Representative CLI/resource runs | 12 ordinary + 1 network-denied, all exit 0 |
| Scores-only preservation | Byte-identical to production JSON for tree/CNN/combined on subject001 and combined on subjects018/025; offline combined also identical |
| Integration-evidence helper checks | Ruff/mypy passed for five artifact scripts; full audit and receipt assertions passed |

Base-only skips cover absent optional packages and the explicit network
injection assertion. The full optional run skips only that injection assertion;
the separate denied run exercises it with training/export/inference. Physical,
source-contract, score-parity, leakage and preserved synthetic failure/regression
tests run without changing their thresholds or existing tests.

[Linux and base Windows CI](https://github.com/Coder-Meet/battleoftheschool/actions/runs/34729551640)
passed for the functional commit. The new bounded optional Windows job was
**skipped**, and manual dispatch returned HTTP 403, “Resource not accessible by
integration.” An authorized user can run `checks.yml` with
`optional_research=true`. Native optional Windows execution is not claimed.

**38 Windows-target binary wheels** downloaded successfully and have recorded
SHA256 hashes, including CPython 3.13 x64 Torch `2.8.0+cpu`, ONNX `1.19.0`,
ONNX Runtime `1.23.2`, scikit-learn `1.7.2`, and compatible psutil `7.1.0`.
Wheel availability is not execution evidence. Optional dependencies remain
outside base requirements.

## Linux resource observations

Serialized representative runs use four-core inherited affinity, numerical/ITK
thread caps and the optional psutil process-tree wrapper. Wall time includes
startup; RSS includes wrapper plus descendants. Numbers below are sampled
summed RSS, **not an exact high-water mark or an enforced 8 GB allocation cap**.

| Workload | Wall seconds | Sampled peak MiB |
| --- | ---: | ---: |
| Complete 210-case reproduction, trees and retrospective CNN | 773.374 | 3608.72 |
| Synthetic strict CLI | 1.657 | 312.64 |
| Synthetic review union + combined scores | 3.792 | 326.40 |
| subject001 strict CLI | 2.983 | 519.78 |
| subject001 tree scores | 3.086 | 533.43 |
| subject001 CNN scores | 7.497 | 533.32 |
| subject001 combined scores | 7.622 | 533.79 |
| subject001 combined scores, network denied | 7.899 | 533.26 |
| subject018 strict CLI | 21.684 | 1477.88 |
| subject018 combined scores | 30.213 | 1492.64 |
| subject025 strict CLI | 21.598 | 1439.35 |
| subject025 combined scores | 29.566 | 1453.66 |

Subject001 explicit tree filtering retains 7/8 proposals and blend filtering
4/8; every original proposal is saved separately. These are output counts,
not accuracy judgments. Rejected experimental models were not enabled in
production. Unscorable extraction or incompatible models return exit 1,
preserve proposals and report all unscored IDs; scores-only preserves daughters
and filter mode withholds its output.

Sampling can miss short peaks and count shared pages twice. Corpus timings
include other development activity; the representative CLI runs are serialized.
GPU state is unknown to the generic wrapper; the ONNX provider is explicitly CPU.
Network is normally unknown. The denied Linux run injects `ENETUNREACH` into
network syscalls and verifies socket denial and identical real output.
No native Windows network denial, organizer-hardware latency or guaranteed
under-ten-second overhead is claimed.

## Reproduction and remaining conditions

See the root README for `research_reproduce.py`, `research_run.py`,
`research_validation.py` and the portable Windows resource command.
The compact attachment includes an executable audit, comparison-preparation
helper, source snapshot, model weights/sidecars, raw logs/JUnit, runtime reports,
dependency versions, wheel hashes and whole-bundle SHA256 index.

Complete expert references, adjudicated eligibility, case/group overlap
resolution and organizer-native Windows measurements remain prerequisites.
All 25 real-case exposure IDs remain recorded even if later cases are renamed.
Heavy 3D segmentation, DRL, graph/topology, external-data training, regression
heads, quantization and group-OOF stacking remain explicitly gated prerequisites.
No poor-model gate was reversed and no clinical accuracy claim is made.
