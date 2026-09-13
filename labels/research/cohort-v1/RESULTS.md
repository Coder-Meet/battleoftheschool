# Preregistered synthetic E0/E1 results

**Verdict: `go_research_only`, with unchanged pre-test selection.**
The selected base-feature gradient boosting model preserved every pool/strict
matched reference and removed 4/4 pool false positives on the sealed test.
This is synthetic evidence under the original analytic definitions, not
clinical accuracy, official challenge scoring or production promotion.

## Artifacts and chronology

- [Full archive: synthetic CTs, patches, frozen source, predictions and reports](https://app.devin.ai/attachments/5316d337-4395-4ce2-997a-0d162b6c6915/branchseed-research-v1.zip)
- [Detailed handoff and reproduction commands](https://app.devin.ai/attachments/70f44860-8f64-40fa-8c78-b71122d1a79f/HANDOFF.md)
- ZIP SHA256: `dc54f317a59e0426bffeb9199f3cf97ecf8e453922fa31b70d9afe52f2faa2ff`
- [Preregistration before generation](https://github.com/Coder-Meet/battleoftheschool/commit/3d9aa9dee9aedf66f18f23253b7e41fd298c89c3)
- [Model/validation freeze before any test detection or candidate labels](https://github.com/Coder-Meet/battleoftheschool/commit/c64fc6f7242d8c546cd06a50afe15e1533420da1)

Fifteen fresh seed groups cover all 14 existing stress families: 140 train,
42 validation and 28 test CTs. All stages completed. Analytic references/CTs
were generated before model fitting; sealed test proposals and candidate
labels were computed only after the immutable freeze. No pseudo/legacy data,
invented negatives, calibration search, test tuning or broadened proposal
profile was introduced.

## End-to-end results at 3 mm

| Partition | Method | TP | FP | FN |
|---|---|---:|---:|---:|
| All 210 | Strict | 437 | 5 | 72 |
| All 210 | Pool | 453 | 28 | 56 |
| Validation, 42 | Strict | 91 | 0 | 12 |
| Validation | Pool | 93 | 4 | 10 |
| Validation | Selected GB-base | 93 | 1 | 10 |
| Validation | Logistic | 93 | 2 | 10 |
| Test, 28 | Strict | 61 | 1 | 9 |
| Test | Pool | 63 | 4 | 7 |
| Test | Selected GB-base | 63 | 0 | 7 |
| Test | Logistic | 63 | 3 | 7 |

The fixed CNN gate required zero induced misses versus both pool and strict,
at least one and >=10% pool FP reduction, and fewer FP than logistic without
more FN, on validation and test. GB-base met it. All four trees reached
63/0/7 on test; extended features did not demonstrate added value. GB-extended
failed the relative-logistic validation gate and was never selected.

Training rows contain 297 confirmed and **10 real unmatched negative proposals**;
validation 93/2; test 63/3. The 10/2/1 ambiguous proposals are excluded only
from candidate training, retained in full inference. No missed reference is
turned into a candidate. All empty cases remain in every case-level split.
Weights use training rows only: equal source-weighted class mass, analytic 1,
expert 1, pseudo 0.

Full reports preserve per-case/family metrics, predictions, missed-reference
IDs, strict/pool lost/recovered references, and 2/5-mm tolerance sensitivity.
Paired bootstrap uses seed groups: test FP difference interval [-6,-2], recall
difference [0,0] (95%, 2,000 samples). Only two test seed groups and 10 training
negatives limit inference. Current strict/pool recall across 210 cases is
85.85%/89.00%; filtering cannot recover the 56 unproposed pool references.

## Source and compatibility boundary

Frozen detector SHA256:
`17495723df18c79e4302edc19bd949bb16f615b902403a88c68d6e74fa200063`.
Exact strict/review configs are in `plan.json`; both used
`minimum_radius_mm=0.7`, `minimum_path_mm=5`, with no explicit origin-diameter
eligibility field. The archived extractor hash is
`568d8bd9f0d417768cb22fdd56d054a26ca215cc1741d3d9914ff4052eaa534a`.
The frozen corpus source hash is
`344733304025ef3d0338d744aefde9e6c71ac3e341806fcb40aabe57dd2aca42`.

The judge later confirmed **2 mm diameter at the origin**, >=5 mm visible
length, one count for a common-trunk opening, two for distinct re-entry
openings, and overlap counting as one or two. Vein acceptance and the exact
official evaluator remain unsettled. These clarifications did not relabel or
retune this run. The parent's newer origin-diameter detector requires a new
source/config plan and separate evaluation. This cohort is now exposed.

The parent's paper variants remain outside this cohort and unpromoted. Retain
its `ADDITIONAL_PAPERS.md` links and `paper_methods.py` pyproject entry.
No patient images are in this archive. The 25 supplied real cases retain
their prior inspection/pseudo-training history; incoming expert cases may overlap.

After the frozen run, current corpus code received a Windows static typing
guard plus stronger future-plan dependency/freeze checks. Frozen scientific
artifacts were not rewritten. Reproduce with archived source, not future main.
Dependencies absent at planning remain recorded as `not-installed`; install
the optional tree requirements before planning a training experiment.

## Reproduce

From the extracted archive directory, using Python 3.13.3:

```bash
uv venv --python 3.13.3 .venv
uv pip install --python .venv/bin/python -r source/requirements.txt -r source/requirements-trees.txt
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
.venv/bin/python source/research_corpus.py plan --root ../reproduced-corpus
.venv/bin/python source/research_corpus.py generate --root ../reproduced-corpus --workers 2
.venv/bin/python source/research_corpus.py export --root ../reproduced-corpus --workers 2
.venv/bin/python source/research_corpus.py train-baselines --root ../reproduced-corpus
.venv/bin/python source/research_corpus.py export --root ../reproduced-corpus --partition test --workers 2
.venv/bin/python source/research_corpus.py train-baselines --root ../reproduced-corpus --evaluate-test
.venv/bin/python audit.py
```

The executed Linux run additionally used `taskset -c 0-3` on every stage.
Caches are float32 `(307,12,32,32)`, `(95,12,32,32)` and `(66,12,32,32)`,
loaded with `allow_pickle=False`; each ordered record has a partition-relative
index and original 13 features plus explicit three physical extras.

Audit passed for 210 cases, 436 sidecars, the immutable freeze, all caches and
every per-case/partition patch identity. Current-source Ruff and both default/
Windows-targeted mypy pass; focused suite 198 passed and full suite 328 passed/
1 skipped before the final optional-dependency regression. All 20 corpus tests
passed afterward. Full-suite warnings were 49 upstream NumPy/scikit-image
deprecations.

Mean synthetic Linux strict/pool/extraction times: 1.010/1.989/1.210 seconds.
Maximum worker peak RSS: 322.61 MiB (excludes parent/sibling processes).
This does not validate native Windows, full-size real CTs or the organizer's
four-core/8-GB limit.
