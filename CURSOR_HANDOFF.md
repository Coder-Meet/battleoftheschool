# Branchseed handoff for Cursor or another agent

## Paste this into the next agent

> Continue `Coder-Meet/battleoftheschool` from the latest `main`. Read
> `CURSOR_HANDOFF.md`, `FINAL_EVALUATION_PROTOCOL.md`, and every available
> `labels/final-eval/*/HANDOFF.md` first. Pull before doing anything that could
> duplicate a worker's work. The user explicitly requires **main only: no branches
> and no PRs**, with frequent focused commits, `git pull --rebase`, then `git push`.
> The immediate objective is to finish a reproducible, bounded comparison of
> detector variants and existing ML models on judge-approved cases 19–23, using
> case-separated selection without overfitting. Do not promote a model merely
> because it wins the five-case aggregate. Preserve source/model contracts and
> the default detector unless evidence clears the regression/resource gates.
> Five Devin workers were active at the handoff and have been asked to checkpoint
> everything to main. Check their committed state before rerunning anything.
> Finish remaining matrix runs, integrate the audit, select on other cases only,
> run the complete checks, and publish the measured recommendation directly on main.
> Do not resume the cancelled manual candidate-labelling task.

## 1. Source of truth and current stopping point

- Repository: https://github.com/Coder-Meet/battleoftheschool
- Main: https://github.com/Coder-Meet/battleoftheschool/tree/main
- Original session: https://app.devin.ai/sessions/d87579c343bb4443b7d75602150c9f01
- Previous complete algorithm review and label analytics: `4a43dd4`.
- Teammate's release-data and Explorer update: `8d9a1f9`.
- Reference validation and frozen comparison protocol: `2a825d89b66e7d5c61bea7f497be08fbc22cf526`.
- First deterministic matrix checkpoint observed: `a3cf586`.
- Subsequent worker and handoff commits may already be on main. **Use latest main,
  not a reset to one of these checkpoints.**

The parent finished reference ingestion, hash/geometry validation, normalized
scoring helpers and the predeclared comparison protocol. No learned model was
promoted. At the initial handoff snapshot the new five-case benchmark had **not**
yet produced a final integrated result. Partial worker results must not be called
the completed evaluation.

The user's last request was to preserve everything for a transition to Cursor or
another agent because usage credits were running low. This document and the
workflow sources are intentionally committed at the user's request.

## 2. Active workers: avoid duplicate work

Workflow name: `branchseed-final-reference-comparison`  
Run ID: `wfr-2df2f65092404cd9a38c48fbea5496b8`  
Portable workflow source: `.agents/workflows/final_reference_comparison.py`

| Family | Session | Agent result ID | Owned files |
|---|---|---|---|
| Deterministic/papers | https://app.devin.ai/sessions/17fe7b9ccd714580a1c57684160500a5 | `wfar-f805866bc7bf4c96a9eb5d149c8110ae` | `final_eval_deterministic.py`, its new tests, `labels/final-eval/deterministic/` |
| Tabular/logistic/trees | https://app.devin.ai/sessions/00c025d5fbb34046a5a6067470b4c116 | `wfar-e6a907da5d83489fbb0c1f3f385ac948` | `final_eval_tabular.py`, its new tests, `labels/final-eval/tabular/` |
| CNN/blends | https://app.devin.ai/sessions/77cca365565348d2b20229415759cb17 | `wfar-2f4dbe5cf6144ce29b83cfd63c8b94c5` | `final_eval_cnn.py`, its new tests, `labels/final-eval/cnn/` |
| Topology/recovery | https://app.devin.ai/sessions/efdd65e6f2fc461ebc4048311b1dcde9 | `wfar-4e6a69776e4f457b9a50b07878d54ef4` | `final_eval_topology.py`, its new tests, `labels/final-eval/topology/` |
| Independent audit | https://app.devin.ai/sessions/5b3a5c2d38e44ea382aa009a500960bf | `wfar-dd7eb4549f554edbbe5db840ab7699e6` | `final_eval_audit.py`, its new tests, `labels/final-eval/audit/` |

The workflow automatically starts a sixth **selection** agent after all five
finish. It owns `final_eval_select.py`, its new tests,
`labels/final-eval/selection/`, and `FINAL_EVALUATION_RESULTS.md`. Its prompt
consumes all five preceding reports. The committed workflow contains all exact
prompts, scope boundaries and output contracts, so Cursor can perform any
unfinished unit without access to Devin tools.

Every worker was instructed to immediately commit/push partial code, frozen plans,
available results and a family `HANDOFF.md`, then continue with regular checkpoints.
Those files, not this initial static snapshot, record its latest completed runs.
Workers own separate files and must not change the production detector, root
requirements, `pyproject.toml`, or one another's modules.

### Verified checkpoint update — 2026-09-13 02:16 UTC

All five workers have pushed source/results checkpoints to main. The latest
pulled checkpoint for this update is `39db197b2687ecd344ad755a9cd8ac06a3408878`.
The independent audit and tabular comparison have finished. Deterministic, CNN
and topology workers remain active; selection has not started.

- **Audit complete:** structural checks passed; limitations and generic
  coordinate-normalizer issues are recorded separately below.
- **Tabular complete:** 106 variants, 530 prediction files, no scoring failures,
  four explicitly excluded old tree artifacts. Tabular-family held-out selection
  has F1 0.551724 at 3 mm; the all-five development winner has F1 0.620690. These
  are different estimates; fold choices are unstable.
- **Deterministic checkpoint:** frozen 26-extraction matrix, resumable
  `checkpoint.json`, and completed case records are committed.
- **CNN checkpoint:** current model extraction finished for both proposal pools;
  historical-model replay remains in progress.
- **Topology checkpoint:** nine recovery variants completed all five cases;
  synthetic regressions and interpretation remain pending. None beats strict
  aggregate 3 mm F1. No recovery change is promoted.

Ruff over the current repository and the configured 29-file mypy check passed at
this checkpoint. A complete final-integrated pytest run remains for the next stage.
Future family handoffs supersede this snapshot.

For another Devin session:

1. Inspect the existing run with `get_workflow_output(run_id=..., timeout_secs=1)`.
2. A running agent can be contacted with `message_workflow_agent` and the IDs above.
3. Do not launch a duplicate while this run is still active.
4. If interrupted, invoke the `dynamic-workflows` skill, then resume via
   `run_workflow(run_id=..., workflow_name=..., script_path=<absolute committed
   workflow path>)`. Completed agent results replay. Keep the workflow structure
   unchanged for a resume.
5. If stopping is needed, `kill_workflow` records finished results and puts
   remaining children to sleep. Stopping a run does not itself copy an uncommitted
   worker filesystem into Git: checkpoint first.

For Cursor: pull main, read the family handoffs, and run their ordinary Python
CLIs. The orchestration script itself requires Devin's injected runtime; it is
not a standalone Python application. The earlier research workflow is archived
at `.agents/workflows/research_implementation.py` for provenance, not for rerunning
already completed work.

## 3. What the application does

Given a CT and a binary parent abdominal-aortic lumen mask, discover a variable
number of direct daughter openings. Output for each daughter:

- `instance_id`, `parent_instance_id="aorta"`;
- `ostium_xyz_mm`, `seed_xyz_mm`, `radius_mm`, `direction_xyz`;
- original SimpleITK **LPS physical millimetres**, without axis flips.

Submission command:

```sh
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
```

Production is deterministic, CPU-only and model-free by default. It crops and
resamples around the parent, estimates scan-relative contrast support, computes
multiscale Sato vesselness, proposes wall contacts, traces supported paths,
localizes ostia/seeds/radii/directions, and resolves topology/duplicates.
Read `STEVEN_ALGORITHM_REVIEW.md` for the full algorithm and history.

Confirmed judge rules:

- Minimum eligible **origin diameter 2 mm**; not 2 mm radius.
- Visible supported lumen must continue at least **5 mm** beyond the wall.
- Trace to the first downstream bifurcation or up to 10 mm.
- A common trunk counts as one direct origin; daughters of daughters do not.
- Two separate aortic openings of a returning vessel may count as two.
- Overlapping ostia may receive one-or-two allowance in the official evaluator.
- Some vein detections may receive credit. Local strict matching does not invent
  such official allowances.
- Daughter count carries substantial accuracy weight; do not assume the older
  score table reflects every oral update. No official scoring script was supplied.

Keep the 0.7 mm trace/seed-radius setting separate from the 2 mm origin-diameter
eligibility setting.

## 4. Reference data and provenance

Original release:
https://drive.google.com/drive/folders/1GoIKqCKIMhtqzHNwLgtNkBhmxT_96ZR6

| Release case | Repository case | Daughters | Known seed radii |
|---|---|---:|---:|
| 19 | subject019 | 3 | 0 |
| 20 | subject020 | 4 | 2 |
| 21 | subject021 | 3 | 0 |
| 22 | subject022 | 6 | 1 |
| 23 | subject023 | 3 | 0 |
| Total | | **19** | **3** |

The user explicitly confirmed: **“The judge has approved these as scoring
references.”** Preserve that authorization AND the release's own statement:
AI-assisted draft annotations, no clinical expert sign-off, possible omissions.
Unmatched predictions count as FP against this inventory, but are not automatically
proven anatomically false. Sixteen unknown radii must remain unknown.

Tracked files:

- `eval/data/case_19/` through `case_23/`: teammate's original release copy;
  images use Git LFS.
- `eval/docs/`: original release documentation.
- `labels/organizer-v1/raw/`: frozen compact metadata and annotation JSON.
- `labels/organizer-v1/references/`: normalized physical-coordinate references.
- `labels/organizer-v1/validation.json`: pairing, geometry, source hashes and
  normalization validation.

Verified: 55 manifest files with matching size and SHA-256; checksum-list equality;
all five CT/parent pairs exactly match `data/subject019..023`; 1.5 mm isotropic
release images; image and label geometry; instance-label sets; voxel-guide to
physical-path conversion; landmark bounds; count inventory; known-radius coverage.
NIfTI qform/sform metadata is recorded. The independent audit additionally checks
mask contact/connectivity and guide/path semantics.

Release manifest SHA-256:
`78c2cd3af2c451c60a91e947fd32c740704d02f94d63cd566e0b927d50ce4850`.
The checksum list contains CRLF. On Unix verify through
`tr -d '\r' < SHA256SUMS.txt | sha256sum -c -`; do not modify the original list.

On the original VM, the complete flat release is at
`/home/ubuntu/attachments/organizer-release-2026-09-13/`.
Revalidate there with:

```sh
python final_evaluation.py --ingest-release /home/ubuntu/attachments/organizer-release-2026-09-13
```

On a fresh machine, either download that release or stage its existing tracked
copies so that `manifest.json`, `SHA256SUMS.txt` and the five `case_XX/` folders
share one parent directory. The repository stores metadata under `eval/docs/`
and case folders under `eval/data/`; do not pass either subdirectory directly as
the complete flat release. Run `git lfs pull` before reading LFS-backed images.

## 5. Frozen comparison and selection contract

Read `FINAL_EVALUATION_PROTOCOL.md` before looking for a “best” score.

Families: strict, review, review-union/pool, existing paper adaptations
(`contact-growth`, `border-cleaning`, `pca-direction`), contrast/support/spacing/
wall-parallel ablations, logistic filters, gradient boosting, random forest,
physical/context features, CNN, CNN/tree blends, and unfiltered proposal ceilings.

Declared grids:

- Support contrast: `{0.3, 0.5, 0.7}`.
- Native contrast scale: `{0, 1.2}`.
- Explicit spacing ablations: `{0.75, 1.5}` mm.
- Candidate thresholds: `{0.15, 0.3, 0.5, 0.7, 0.85}` plus frozen model thresholds.
- CNN blend weights: `{0, 0.25, 0.5, 0.75, 1}`.

Use one-to-one ostium matching at **2, 3 and 5 mm local tolerances**. These are
not verified official challenge tolerances. Use `final_evaluation.score_variant`
and related helpers so empty cases count, unknown radii are excluded, and
seed-to-reference-guide distance is reported.

For each held-out case, choose a configuration using **only the other four**:
3 mm pooled F1, then lower count MAE, then simplicity/runtime, deterministic tie
resolution. Retain that fold's selected prediction unchanged for held-out scoring.
Report fold selections/stability and count/localization errors. An all-five
winner is a **development/reference-set result**, not independent test accuracy.

Old AI-label models that saw these images are retrospective/exposed baselines.
Verify training lineage rather than assuming every saved model is contaminated:
current-source synthetic models may be eligible. Train new candidate filters
without subject019–023 or renamed equivalents; never turn unannotated draft
proposals into expert-negative training examples.

Post-reference recovery hypotheses from the topology worker are separate
development experiments, ineligible for the clean frozen-family selection.
Filters cannot recover branches absent from the proposal pool.

Each family report must preserve all five predictions per variant, complete
configuration, hashes, selection eligibility, 2/3/5 mm scores, candidate ordering
and probabilities where applicable, runtime/memory measurements, exclusions and
reference IDs lost by filtering. No silently missing matrix rows.

## 6. Environments and commands

Original Linux checkout: `/home/ubuntu/repos/battleoftheschool`. Verified interpreter:
`.venv313/bin/python`, Python **3.13.3**. Do not copy this Linux venv onto Windows.
Follow README and pinned requirements.

Linux/macOS:

```sh
git pull --rebase
git lfs pull
uv venv --python 3.13.3 .venv313
uv pip install --python .venv313/bin/python -r requirements-dev.txt
uv pip install --python .venv313/bin/python -r requirements-trees.txt -r requirements-cnn-inference.txt -r requirements-resources.txt
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4
.venv313/bin/python -m ruff check .
.venv313/bin/python -m mypy
.venv313/bin/python -m pytest -q
```

Inspect each worker CLI using `python final_eval_<family>.py --help` and follow its
`HANDOFF.md`; do not invent arguments or assume a partial checkpoint is runnable.
CNN training dependencies are separate from ONNX inference; install only when a
remaining experiment actually requires training.

For Windows, use Python 3.13.3 x64 and pinned requirements. Set the same thread
variables with PowerShell `$env:NAME="4"` before launching Python.
Final inference must be **offline, four cores, 8 GB RAM, no GPU**. Initial average
target is <=60 seconds/case. Actual organizer-laptop timing remains outstanding.
Linux measurements and Windows CI do not establish performance on that laptop.

Parent checks before handoff:

- Focused evaluator/reference tests: 14 passed.
- Ruff: passed before worker integration.
- Mypy on `final_evaluation.py`: passed.
- Ingestion with the final mask-geometry/label validation: passed.
- No claim that the final integrated full suite or final reference matrix passed.

Prior production resource evidence: 25/25 cases, 5.15 seconds average, 21.612
seconds slowest, approximately 1.45 GiB peak RSS on the tested Linux setup.
Re-run the selected configuration; do not transfer these timings to a new model.

## 7. Work still required

Known audit findings for the next agent: the generic `score_references.py`
normalizer does not rotate/scale supplied voxel-space directions and does not
automatically convert a RAS marker into LPS. The released references are already
verified LPS, so their present scores are unaffected. Its direct aggregate omits
missing prediction files; use the stricter complete-case shared helper for this
matrix. All NIfTI spatial-unit tags are unspecified; mm units come from the
release documentation. See `labels/final-eval/audit/report.json` for controls and
evidence. Do not silently rewrite release headers or misstate these as fixed.

1. Pull all worker checkpoints and inspect their family handoffs and blockers.
2. Finish every declared supported comparison, or record a precise incompatibility
   and reproduction path. Never bypass source-contract mismatches.
3. Address the independent reference/scoring audit before accepting aggregate scores.
4. Run/review the selection stage; check held-out selection cannot use held-out labels.
5. Compare selected candidates against fixed synthetic topology/negative-control
   regressions and strict production. Keep experimental gains separate.
6. Run Ruff, mypy, full pytest, offline/network-denied checks, Windows compatibility,
   and four-core resource measurements. Test actual Windows hardware when available.
7. Publish exact chosen configuration/weights, all five challenge JSONs, per-case
   metrics and lost/recovered IDs, source/model/input hash manifest, reproducible
   commands and limitations. If strict remains best supported, say so.
8. Commit, pull with rebase and push main. Verify a clean working tree and remote
   equality. Do not create a PR or branch.

Do not overwrite teammates' recent Explorer changes. Do not use destructive Git
commands, force pushes, amend, skipped hooks, or blanket `git add .`. Stage
explicit files. Never commit secrets, venvs, node_modules or new raw CT copies.

## 8. Recovery assets and previous deliverables

Durable downloads (Devin login required; download in your browser before giving
them to Cursor):

- [Local generated outputs — 366 MiB](https://app.devin.ai/attachments/d5248c34-7550-46d7-8786-c2560d9c7e50/local-outputs.tar.gz)
- [Research, release and submission assets — 465 MiB](https://app.devin.ai/attachments/68c07258-e74c-46f2-9d77-a7b83e436b4c/research-and-submission-assets.tar.gz)
- [Source and materialized dataset snapshot — 700 MiB](https://app.devin.ai/attachments/7f907237-f9a8-4710-8ae4-2b4c83ef017e/source-snapshot.zip)

`HANDOFF_RECOVERY.json` records exact sizes and SHA-256 hashes. Both gzip streams
were verified. Extract the first at the repository root. Extract the second into
a separate recovery directory; its relative paths reflect the original home
directory. Do not overwrite current source with older bundled submission copies.
The ZIP captures commit `7781d7cb1ec3df710cb4959ff4707d287251a0eb`, including
materialized LFS data; its CRC test passed. It has no `.git` directory. Prefer a
normal clone for continued commits, and use the ZIP as a recovery copy. Later
worker commits on main supersede this snapshot.

The handoff message supplies recovery archives for local outputs and selected
outside-repository research artifacts. They supplement Git; they do not replace
the latest worker commits. Generated caches, Python environments, browser profiles,
credentials and node_modules are excluded. The 25 supplied CT volumes remain
available through the repository's LFS data; they are not duplicated in a handoff
commit. The original five-case release remains at the Drive URL above.

Important original VM-only locations, until downloaded from the recovery archives:

- `outputs/`: generated comparisons, synthetic cohorts, candidate audit, presentation
  kit and Steven's PDF/Word/ZIP handoff.
- `/home/ubuntu/contrast-audit`, `parallel-audit`, `paper-audit`, `steven-audit`,
  `merge-audit`, `refs-dryrun`, `refactor-check`: previous local experiments/checks.
- `/home/ubuntu/hard-real-review-v2`, `hard-synthetic-731927`: prepared review and
  analytic-ground-truth bundles.
- `/home/ubuntu/branchseed-development-submission`: earlier submission bundle;
  rebuild it if the selected algorithm changes.
- `/home/ubuntu/windows-wheelhouse`: earlier pinned Windows x64 dependency wheels.
- `/home/ubuntu/attachments/organizer-release-2026-09-13`: validated release and
  public-download helper.

Existing shareable deliverables:

- Algorithm review: `STEVEN_ALGORITHM_REVIEW.md`.
- Label analytics: `STEVEN_LABEL_ANALYTICS.md`; pseudo-label agreement is not
  expert-reference accuracy.
- Research results: `labels/research/current-source-v1/RESULTS.md`.
- Presentation MP4:
  https://github.com/Coder-Meet/battleoftheschool/releases/download/branchseed-presentation-2026-09-12/branchseed-film.mp4
- Complete presentation kit:
  https://github.com/Coder-Meet/battleoftheschool/releases/download/branchseed-presentation-2026-09-12/branchseed-presentation-kit.zip

Devpost submission and native organizer Windows timing were not completed in this
session. Manual/AI candidate labelling was explicitly cancelled.
