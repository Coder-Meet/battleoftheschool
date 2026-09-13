# Deterministic family handoff

## Status: partial checkpoint; extraction still running

The source and exact 26-extraction matrix were committed and pushed before scoring
in `a3cf586031caf67b3f6ef7965ed30d0db717e257`. This checkpoint preserves completed
worker records; it does **not** claim a complete comparison or selected winner.
`checkpoint.json` is the authoritative inventory of committed completed and pending
case/configuration pairs at each checkpoint. Each completed worker record includes
predictions, runtime/RSS, candidate ordering/features, rejected trace evidence,
and matrix/source identity. No scores have been inspected at this checkpoint.

Seven focused tests passed (5.79 seconds); Ruff and mypy passed for both assigned
Python files. The existing detector, paper, accuracy, stress, and input-validation
regressions passed: 77 tests in 98.65 seconds.

## Scope and immutable provenance

Own only `final_eval_deterministic.py`, `tests/test_final_eval_deterministic.py`,
and `labels/final-eval/deterministic/`. Work directly on `main`, commit only these
files, then `git pull --rebase` and `git push origin main`. No branches or PRs.
Do not edit production detector, scorer, requirements, model artifacts, or other
agents' files. Do not amend, force-push, skip hooks, or change Git configuration.
Resolve only clerical concurrent-push conflicts.

SHA-256:

| Item | Hash |
| --- | --- |
| Runner | `69bc364d810bf402a7035e5abccfdf6a360cf6d5643dbda6ebaf5d6274233de1` |
| Frozen matrix | `fcf4c35fb1aacbd54141a1b09e2d69b9f889b4325c80ad3235c6063108aa8c38` |
| Production detector | `9f96f3eb3c06f5e06d5b7db79b199be70e742130fc3885243d25d066adc1c92e` |
| Shared scorer | `be2488c1123e0f3ae7cbff7a319ac360fab60c9502ebe5ce490bd7c918d7c990` |

All other source, input, and reference hashes are in `matrix.json`. The detector
is byte-identical to baseline `4a43dd4`. The runner verifies frozen sources and
settings; never bypass a mismatch. Keep the runner itself at the hash above when
resuming this matrix. The added seventh test changes no extraction source.

## Resume commands

Run from the repository root. On this VM that is
`/home/ubuntu/repos/battleoftheschool-final-deterministic`, not the older
`StevenTB1` checkout at `/home/ubuntu/repos/battleoftheschool`.

```bash
git pull --rebase
git lfs pull --include="data/subject019/*,data/subject020/*,data/subject021/*,data/subject022/*,data/subject023/*"
uv venv --python 3.13.3 .venv313
uv pip install --python .venv313/bin/python -r requirements-dev.txt
.venv313/bin/ruff check final_eval_deterministic.py tests/test_final_eval_deterministic.py
.venv313/bin/mypy final_eval_deterministic.py tests/test_final_eval_deterministic.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4 .venv313/bin/python -m pytest -q tests/test_final_eval_deterministic.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4 .venv313/bin/python -m pytest -q tests/test_detector.py tests/test_paper_methods.py tests/test_accuracy.py tests/test_stress.py tests/test_input_validation.py
.venv313/bin/python final_eval_deterministic.py run
.venv313/bin/python final_eval_deterministic.py report
```

Skip virtualenv creation/install when this VM's existing environment remains
usable. Do **not** run `freeze` again: the committed matrix is immutable.
The `run` command resumes completed matching worker records without rerunning
CT processing, caps worker numerical threads at four, enforces 7168 MB RSS and
1800 seconds per worker, and records explicit failures. Before starting on the
same VM, check whether its original runner is still alive:

```bash
ps -eo pid,etime,args | rg 'final_eval_deterministic.py (run|worker)'
```

Do not start a second runner against the same output directory. The original
Linux invocation is pinned with `taskset -c 0-3`; the portable command above
works without `taskset` and retains library thread caps.

## Remaining work

1. Finish pending pairs listed in `checkpoint.json` and inspect `failures.json`.
   There are 26 extraction configurations x five cases. Never substitute an
   empty prediction for a failed extraction.
2. Generate `report.json`, `reference-diagnostics.json`, and five predictions
   for every valid final and trace-ceiling variant (52 derived/scored variants
   if all extractions succeed).
3. Validate local 2/3/5 mm scores, origin-size eligibility, reference IDs lost
   versus strict, support/size rejection evidence, LOCO folds, and bootstrap
   caveats. Add a concise human-readable interpretation under this directory.
4. Commit compact JSON results and update this handoff. Pull with rebase before
   pushing. Return the final pushed SHA and relative `report.json` path.

The 19 judge-approved targets retain AI-assisted provenance and may omit vessels.
Only three radii are measured. Shared scoring masks unknown radii. No hidden-test,
clinical, or official weighted-score claim is supported. Origin-disabled and
trace-ceiling variants are ineligible proposal/recall baselines. Unknown origin
diameter is uncertain; it is not knowingly below 2 mm. Spatial rejected-root
diagnostics are evidence of nearby failures, not proof of vessel identity.

## Local-only files and blockers

- `.venv313/` and resolved five-case NIfTI LFS files are local setup/data;
  recreate them using the commands above.
- `execution.log` and `regressions.log` in this directory are ignored local logs.
  Durable predictions/diagnostics are in the committed `runs/` records; the test
  verdict is recorded above.
- Results completed after the latest checkpoint may still be untracked until the
  next checkpoint. Inspect `git status --short` and commit only owned files.
- No task blocker is known. The initial stale checkout's setup failure was
  avoided by cloning the requested `Coder-Meet` repository and installing its
  pinned development dependencies. No credentials are required beyond existing
  repository/LFS access.
