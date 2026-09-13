# Topology checkpoint — PARTIAL / INCOMPLETE

Checkpoint requested because usage credits are nearly exhausted. No production
promotion or clean model-selection claim is justified. All new experiments are
**POST-REFERENCE DEVELOPMENT**, ineligible for the frozen-family comparison.

## Latest continuation status

Initial partial checkpoint was pushed successfully as
`76ce2f13b09da1f66cbeb6d8dd63dcdd5b00b1db`. Subsequently all nine recovery
variants completed all five cases with zero execution failures. `report.json`
now contains all nine variants; the original progress table below is historical.
The synthetic cohort and final interpretation remain pending.

| Variant suffix (topology-) | 3 mm TP / FP / FN | F1 |
| --- | --- | --- |
| alternate-roots | 8 / 14 / 11 | 0.3902 |
| connector-gap | 8 / 4 / 11 | 0.5161 |
| parallel-path | 8 / 4 / 11 | 0.5161 |
| contrast-support | 10 / 9 / 9 | 0.5263 |
| combined | 14 / 34 / 5 | 0.4179 |
| combined-no-roots | 11 / 13 / 8 | 0.5116 |
| combined-no-gap | 14 / 32 / 5 | 0.4308 |
| combined-no-parallel | 15 / 38 / 4 | 0.4167 |
| combined-no-contrast | 8 / 16 / 11 | 0.3721 |

No recovery configuration improves strict baseline aggregate 3 mm F1 (0.5333).
Contrast sensitivity drives recovered references but introduces unmatched
origins. Adding gap tolerance to the combined configuration adds two unmatched
origins with no new matches. The synthetic evidence is still needed.

## What is durable

- `final_eval_topology.py`: faithful instrumented detector resolver, physical LPS
  diagnosis, bounded alternate-root experiment, frozen configurations, resumable
  real/synthetic runners. References enter only diagnosis and scoring.
- `tests/test_final_eval_topology.py`: 12 tests including exact equality against
  the actual frozen resolver on four synthetic scenarios under strict/review.
- `baseline_report.json`: complete strict and review-origin2 baselines, all five
  case outputs and per-candidate diagnosis under their respective directories.
- `experiment_config.json`: nine configurations and a 24-case synthetic cohort
  frozen before recovery scoring, including source hashes and rationale.
- `report.json`: **partial** recovery report, containing only complete five-case
  variants. Do not interpret absent variants as empty predictions.
- `resources/`: isolated four-core process-tree RSS samples and command metadata.
  Baseline RSS is explicitly unknown; baseline runtime excludes loading/diagnosis.

## Historical initial-checkpoint status (superseded above)

| Variant | Status | Local 3 mm TP / FP / FN | F1 |
| --- | --- | --- | --- |
| topology-strict-audit | complete, five cases | 8 / 3 / 11 | 0.5333 |
| topology-review-origin2-audit | complete, five cases | 9 / 14 / 10 | 0.4286 |
| topology-alternate-roots | complete, five cases | 8 / 14 / 11 | 0.3902 |
| topology-connector-gap | complete, five cases | 8 / 4 / 11 | 0.5161 |
| topology-parallel-path | subject019 complete; rest pending | incomplete | — |
| topology-contrast-support | pending | incomplete | — |
| topology-combined | pending | incomplete | — |
| topology-combined-no-roots | pending | incomplete | — |
| topology-combined-no-gap | pending | incomplete | — |
| topology-combined-no-parallel | pending | incomplete | — |
| topology-combined-no-contrast | pending | incomplete | — |

No synthetic cohort evaluation has run yet. The 37 existing-detector plus new
topology tests passed; those tests are separate from the planned 24-case cohort.
The experiment process was deliberately stopped for a consistent checkpoint
during the parallel-path variant. No inference process remains running.
Completed resource checkpoints are reused on resume; an interrupted case without
its resource JSON reruns. Do not manufacture missing case results.

Preliminary evidence: extra roots and a 20% connector-gap tolerance increased
unmatched origins without recovering a new 3 mm match. Strict fails through
disconnected ostia, unsupported 5 mm paths and low support on several reference
guides. Remaining wall-parallel/contrast ablations must finish before any verdict.
No final per-reference narrative or consolidated final family report exists yet.

## Exact continuation commands

Run from the repository root on **main**, after `git pull --rebase`. Only this
module, its new test, and `labels/final-eval/topology/` are assigned to this agent.
Never change `detector.py`, production configuration, existing models, or others'
files. Do not create branches/PRs. Commit only assigned files, pull with rebase,
and push main frequently.

```bash
git lfs pull --include='data/subject019/*,data/subject020/*,data/subject021/*,data/subject022/*,data/subject023/*'
uv venv --python 3.13.3 .venv313
uv pip install --python .venv313/bin/python -r requirements-dev.txt -r requirements-resources.txt
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4
.venv313/bin/ruff check final_eval_topology.py tests/test_final_eval_topology.py
.venv313/bin/mypy final_eval_topology.py tests/test_final_eval_topology.py
.venv313/bin/python -m pytest -q tests/test_detector.py tests/test_final_eval_topology.py
.venv313/bin/python final_eval_topology.py experiments
.venv313/bin/python final_eval_topology.py synthetic
```

Skip creating/installing the environment if the matching pinned environment
already exists. On Windows use its corresponding interpreter path and set the
four environment variables before starting Python. Windows was not tested here.
Real experiments run sequentially, each worker restricted to four CPU cores.

Do **not** rerun `freeze` or overwrite the frozen config. Do **not** regenerate
baselines unnecessarily: the frozen config hashes the initial baseline report.
Do not edit hashed source then bypass `load_frozen` validation. An essential
algorithm repair requires a new explicitly versioned experiment with documented
provenance; unchanged settings/results must retain their original hashes.

After completing the remaining runs:

1. Verify nine complete recovery variants and all five case files for each.
2. Verify 24 synthetic cases for all eleven variants and inspect failures.
3. Compare standalone and leave-one-component-out ablations; inspect matched
   reference identities and extra-origin features, not only aggregate F1.
4. Write a concise per-reference diagnosis and synthetic regression verdict.
   Unknown reference radii must remain unknown. All reported scores are local
   one-to-one 2/3/5 mm comparisons, not an official challenge score.
5. Confirm `detector.py` remains unchanged, inspect `git diff`, run Ruff/mypy/tests,
   then commit, `git pull --rebase`, push main, and return SHA/report path.

## Source and input provenance

Starting main: `2a825d89b66e7d5c61bea7f497be08fbc22cf526`.

| File | SHA-256 |
| --- | --- |
| final_eval_topology.py | b7de4488c0b5771dbf9a10223c008e3e03f22b068bd26c16afcd2eb45aad7fbe |
| tests/test_final_eval_topology.py | 96e6cb3f6690591483e8e06c34047e39cc09edb4b10a20ff240f6fafd84078d5 |
| detector.py | 9f96f3eb3c06f5e06d5b7db79b199be70e742130fc3885243d25d066adc1c92e |
| experiment_config.json | db719d635fd57f724cc4ede61edcdc1851d1688073c7c2af96f1da338bb96a94 |

Full dependency/source/input/reference hashes are in the JSON reports. There are
no fitted model files. The 2 mm-origin policy is the unchanged detector policy
using explicit measurement uncertainty, not fabricated radii. Diagnostic support
connectivity uses positive connected-component labels; unsupported label zero
does not count as parent-connected.

## Local-only state and blockers

The checkout used was `/home/ubuntu/repos/topology-final`, not the pre-existing
checkout pointing to the separate StevenTB1 remote. Local `.venv313`, caches and
the selectively resolved five-case LFS payloads are reproducible dependencies,
not unique unpublished work. No CT volumes are added by this work.

There is no access or code blocker. Work is incomplete because of the requested
credit-preservation checkpoint. The stopped shell ID is not a continuation
mechanism; use the commands above. The checkpoint commit itself is the durable
revision identifier (reported to the parent after a successful push).
