# Completed independent audit

Status: **complete**. The expanded audit ran successfully: every structural check
passed and the failure list is empty. Ruff, mypy and 39 focused tests passed.
Results include explicit structural summaries, deeper model metadata checks and
strict-first execution to avoid inherited parent RSS floors. Initial checkpoint
`3c34ef61d92d425a438106bac29a0e2678be204d` is already pushed to MAIN; this update
supersedes its partial status. No training or audit variant remains pending.

## Scope and ownership

Only `final_eval_audit.py`, `tests/test_final_eval_audit.py`, and this directory
belong to this agent. Work on MAIN; no branches/PRs. Never edit production
detector/scorer, accepted references, models, requirements or another agent's
files. Commit only owned files, `git pull --rebase` before push, no force push.
Initial parent preparation commit: `2a825d89b66e7d5c61bea7f497be08fbc22cf526`.

## Completed evidence

- All 55 release files and CRLF checksum inventory verified, plus archived
  originals. CT/masks are byte-identical to subjects019–023.
- 19 targets (3/4/3/6/3), 3 measured radii, 16 unknown. Unknowns remain null.
- Active NIfTI affines match SimpleITK LPS; all physical/voxel guides round-trip.
  All labels are connected, contact parent faces, do not overlap parent, and
  contain their 5 mm seeds. Directions are unit seed-chord vectors.
- All 20 inspected headers have inactive qform=0, active sform=2 and unspecified
  spatial units=0. Millimetres come from release documentation/JSON, not a header
  unit declaration. This metadata limitation is recorded as a finding.
- Direct/shared scorer parity passed on synthetic controls and all five strict
  outputs at 2/3/5 mm. Strict control: TP/FP/FN = 6/5/13 at 2 mm and 8/3/11 at
  3 and 5 mm. This audit-only duplicate is **ineligible for selection**.
- 48 exhaustive assignment controls verify cardinality before distance.
  Empty-case denominators and invalid/missing-variant rejection verified.
- Ruff and mypy passed; 39 focused tests passed.
- Review checklist, exclusions and review notes preserved without anatomical
  adjudication. Prior pseudo-label exposure and source/model hashes recorded.

Findings needing aggregator awareness:

- Supplied voxel directions are not rotated/scaled by the normalizer (108.8°
  error in rotated anisotropic control); RAS marker is not automatically
  converted. Accepted LPS targets are unaffected. Direct mirror control warns.
- Direct `score_references.score` omits missing files from aggregation although
  it lists them; use `score_variant` with all five cases for comparison.
- Shared summary omits pooled signed count error (per-case signed errors exist).
- Older logistic fitting includes 022/023, calibration includes 020; mixed CNN
  fitting includes 022/023. These models are retrospective. Older cohort/CNN
  detector source differs from current baseline; do not bypass compatibility.
- Original review is unsigned and incomplete despite judge scoring approval;
  unmatched predictions are not established anatomical negatives.

## Reproduction commands

Checkout used: `/home/ubuntu/repos/battleoftheschool-audit`. Local-only `.venv313`
contains the pinned environment. No useful result exists only in that environment.
The earlier `/home/ubuntu/repos/battleoftheschool` checkout had untracked files
blocking pull; it was left untouched. Use a clean clone, never destructive cleanup.

```bash
uv venv --python 3.13.3 .venv313
uv pip install --python .venv313/bin/python -r requirements-dev.txt
git lfs pull --include="data/subject019/**,data/subject020/**,data/subject021/**,data/subject022/**,data/subject023/**,eval/data/case_19/**,eval/data/case_20/**,eval/data/case_21/**,eval/data/case_22/**,eval/data/case_23/**"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4
.venv313/bin/python final_eval_audit.py
.venv313/bin/ruff check final_eval_audit.py tests/test_final_eval_audit.py
.venv313/bin/mypy final_eval_audit.py
.venv313/bin/pytest -q tests/test_final_eval_audit.py tests/test_final_evaluation.py tests/test_score_references.py tests/test_evaluate.py
git diff --check
git add final_eval_audit.py tests/test_final_eval_audit.py labels/final-eval/audit
git commit -m "Complete independent released-reference audit"
git pull --rebase
git push origin main
```

The default audit regenerates only owned artifacts; strict inference reads only
CT/mask and baseline config, never reference coordinates/counts. It takes roughly
20 seconds here, uses CPU only and caps threads at four. No training remains.
`--skip-strict` is a partial diagnostic and intentionally yields no strict variant.
Model ONNX bytes and sidecar hashes are inspected; CNN inference is outside scope.
Offline Windows runtime validation remains for the parent, not this Linux audit.

Final source SHA256 (also recorded in `report.json`):

| File | SHA256 |
|---|---|
| final_eval_audit.py | 2cf69e08661464ff75b91732c1a2e9176e37c79875dda4eb501096611f34ad3e |
| tests/test_final_eval_audit.py | f2546997cc42c2045d8d80b9e038bcacec5b12bc2a5a1ff72785baf3787f8d3b |
| detector.py | 9f96f3eb3c06f5e06d5b7db79b199be70e742130fc3885243d25d066adc1c92e |
| final_evaluation.py | be2488c1123e0f3ae7cbff7a319ac360fab60c9502ebe5ce490bd7c918d7c990 |
| score_references.py | 483d216d6a3a9993a0ca8f6c2d614f7a85702e51ee4894437369e72487b22e38 |

No unresolved operational blocker. The parent can consume `report.json` and
its `findings`, retain audit-only variants as ineligible, and use the source/model
compatibility and training-overlap classifications during aggregation. No
production scorer or accepted target was changed.
