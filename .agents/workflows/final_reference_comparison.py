# ruff: noqa: F821
"""Devin injects the workflow runtime functions when executing this script."""

import asyncio
import json

REPO = "github.com/Coder-Meet/battleoftheschool"
SCHEMA = {
    "type": "object",
    "properties": {
        "commit": {"type": "string"},
        "report_path": {"type": "string"},
        "findings": {"type": "string"},
        "checks": {"type": "string"},
        "blockers": {"type": "array", "items": {"type": "string"}},
        "files": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["commit", "report_path", "findings", "checks", "blockers", "files"],
}

COMMON = """
Final Branchseed model comparison in github.com/Coder-Meet/battleoftheschool.
User: "eval is out go all out test all possibel comobitaiton all adanvced algos
we have differenet testign and eveythgin get the best possible weights! ddo not
overfit! use the best algo!!!!" Judge approval was explicitly confirmed by user.

Read README.md, FINAL_EVALUATION_PROTOCOL.md, final_evaluation.py and relevant
model source. Start on current MAIN, including parent preparation commit
2a825d89b66e7d5c61bea7f497be08fbc22cf526. Baseline detector.py is unchanged from
4a43dd4. Never modify production detector.py, run.py, root README, pyproject.toml,
requirements, existing model artifacts or other agents' files. You may add only
your assigned source/test/artifact files. NO BRANCHES OR PRS: user explicitly
requires MAIN, regular focused commits, pull --rebase before push. This overrides
the default feature-branch workflow. Avoid all destructive git commands, amend,
force-push, git config changes and skipped hooks. On main each agent owns disjoint
files. Commit only your files; resolve only clerical concurrent-push conflicts.
Return commit SHA and relative report path after a successful push.

The released 5 cases map by exact image/mask byte hashes to data/subject019..023.
CTs: orig*.nii*, parent masks mask*.nii*. The prep verified all 55 manifest files,
checksum list, geometries and labels. Source release now ALSO exists in repo
eval/data/case_19..23 and eval/docs (teammate commit 8d9a1f9); NIfTI uses Git LFS.
Only fetch LFS data you need. Do not treat unresolved LFS pointers as image data.
Canonical reference JSON: labels/organizer-v1/references/subject019.json etc.
Reference metadata and original JSON: labels/organizer-v1/raw/ and validation.json.
The 19 targets (3/4/3/6/3) are judge-approved scoring references with AI-assisted
draft provenance. Original release says omissions are possible. Only 3 of 19
radii are measured! Never substitute made-up radius values in reported metrics.
No official evaluator was supplied; our 2/3/5 mm one-to-one scores are LOCAL
reference comparisons, not the official weighted challenge score.

Use final_evaluation.CASES, case_paths(), load_reference(), score_variant(),
score_prediction(), summarize(), filtered(), digest(), write_json().
score_variant takes {case_id: prediction_dict} for ALL FIVE cases, empty outputs
included, and returns {"2":{"summary":...,"cases":[...]}, "3":..., "5":...}.
It masks unknown radii and measures seed-to-guide distance. Do not replace this
with a metric that drops zero-prediction cases or silently treats missing radii
as 1 mm. Prediction schema remains existing challenge schema.

Shared output contract: your family report has "variants": list of objects:
{"name": <globally unique family-prefixed name>,
 "prediction_dir": <repo-relative dir containing all five subjectNNN.json files>,
 "configuration": <full reproducible settings/model hashes/thresholds>,
 "eligible_for_selection": <bool>,
 "ineligible_reason": <string, empty when eligible>,
 "scores": <score_variant result>,
 "runtime": {<case_id>: {"runtime_s": <measured seconds>, "peak_rss_mb": <number or null>}}}.
Also record source/model/input hashes and explicit failures/exclusions. Preserve
candidate scores/features/ordering so decisions can be replayed without new CT
processing. Invalid variants are failures, never replaced with empty predictions.
Use labels/final-eval/<your-family>/ for compact committed artifacts; no CTs or
giant training corpora. Existing real pseudo-labels are not expert negatives.
Models that saw these five images are retrospective baselines, not eligible for
the clean leave-one-case-out selection. Never train against the entire 19-reference
set and report it as independent. Do not use reference coordinates/counts to
generate detector outputs. No fixed case counts, top-k by expected count or
patient-specific thresholds. Post-reference new algorithms must be tagged as such.

Runtime deployment: offline Windows, 4 CPU cores, 8GB RAM, no GPU. Training may
be separate. Configure ITK/OMP/OpenBLAS/MKL threads <=4, avoid memory explosions.
Use existing pinned dependency environments per README. Relevant installed paths
on parent are .venv313 (Python 3.13) and .venv; your own VM may need setup. Optional
CPU ONNX/sklearn requirements are committed. Do not upload patient CTs for inference.
Run Ruff, mypy on your own files, and meaningful focused tests. Do not edit
existing tests to pass. Source checks on tree/CNN models are intentional: do not
bypass or fake them. Source mismatch requires faithful frozen-source replay or
re-extraction/retraining, with explicit provenance. Do not message the user.
"""

JOBS = [
    {
        "name": "deterministic",
        "minutes": 35,
        "task": """
Own final_eval_deterministic.py, tests/test_final_eval_deterministic.py and
labels/final-eval/deterministic/ only. Build and RUN a reproducible finite detector
matrix, using the declared protocol: strict, review, review-union; all 3 existing
paper_methods.py METHODS; contrast fraction {0.3,0.5,0.7} x native_contrast_scale
{0,1.2} for strict and review-union, deduplicating identical defaults; standalone
working spacing {0.75,1.5} and wall-parallel option ablations. Read DetectorConfig,
detect_pool and paper signatures rather than guessing parameters. Hold 5/10 mm
path requirements fixed; preserve all candidates as recall ceilings and separately
tag outputs with origin-size eligibility disabled. Such unfiltered review output
is a proposal baseline, not automatically a deployable final algorithm.
Freeze exact expanded matrix JSON before examining scores. Score all valid variants
at 2/3/5 mm, retain per-case diagnostics and predictions. The model agents run
their own strict/pool once; do not coordinate mutable caches with them.
No detector implementation edits; error insights go in report. If a variant loses
references, identify IDs, rejection reasons and downstream size/support effects.
Set eligible_for_selection false for configurations that violate challenge
eligibility; explain the distinction between unknown diameter and knowingly
below-2mm diameter. Report clean finite-family outputs without claiming hidden
test accuracy. Run appropriate existing detector/paper regressions as well.
""",
    },
    {
        "name": "tabular",
        "minutes": 35,
        "task": """
Own final_eval_tabular.py, tests/test_final_eval_tabular.py and
labels/final-eval/tabular/ only. Audit and RUN every compatible saved logistic
and tree model on strict and review-union outputs. At least original
labels/candidate-model.json, candidate-model-synthetic.json, current-source-v1
models/logistic.json, gradient_boosting-{base,extended}.json and
random_forest-{base,extended}.json. Apply each frozen threshold plus
{0.15,0.3,0.5,0.7,0.85}, deduplicate equal thresholds. Handle 0-candidate cases.
Use research_run.score_detection for validated tree source/preprocessing; preserve
extra-feature ordering. Current-source models are synthetic-only. The historical
logistic models saw some reference images; retain these as ineligible retrospective
baselines. Also train a fresh regularized logistic with all subject019..023 and
renamed equivalents excluded from the existing pseudo-label corpus; case-separated
training/validation selects threshold. Do NOT label unmatched draft-reference
proposals as expert negatives. Freeze split, scaler, weights, threshold and source
fingerprints; report training lineage and exact removed reference IDs for every
filter, plus conditional-vs-proposal recall. No new sklearn estimator families
beyond existing models are needed. Calibration only from disjoint development
cases; any claimed calibration must include fitting case IDs.
Record replayable per-candidate probabilities/features with IDs and complete
fingerprints. Return all variants through shared report contract, not just best.
""",
    },
    {
        "name": "cnn",
        "minutes": 40,
        "task": """
Own final_eval_cnn.py, tests/test_final_eval_cnn.py and labels/final-eval/cnn/ only.
RUN current-source-v1/cnn-synthetic/model.onnx on strict and review-union candidates,
plus its frozen tree/CNN blend. Evaluate CNN and synthetic gradient-boosting
convex probability blends weights {0,.25,.5,.75,1} and thresholds frozen plus
{.15,.3,.5,.7,.85}; pairs must be same ordered candidate identities/extraction
contract. Use validated research_run/patch_inference APIs. Score extraction/load
costs and offline ONNX inference. Do not weaken source/manifest/feature checks.
Also inventory older labels/research/patch-cnn-retrospective-v1/{mixed,synthetic}
models and any other ONNX artifacts. If stale, use a frozen matching source
snapshot (git archive into a separate ignored directory is permitted; no branch)
to run faithfully and mark old mixed models as contaminated retrospective baselines.
Do not silently count incompatible artifacts as tested. If a matching snapshot
cannot be located, demonstrate mismatch and explain exact retraining path; optional
retrain only if practical with data and current source. Existing synthetic CNN
was trained from the current source; verify that rather than assume it.
No training on held-out release candidates. Frozen/blended outputs and thresholds
must be replayable from saved candidate probabilities and source/model hashes.
Return all matrix variants, exclusions and true-reference losses.
""",
    },
    {
        "name": "topology",
        "minutes": 40,
        "task": """
Own final_eval_topology.py, tests/test_final_eval_topology.py and
labels/final-eval/topology/ only. Investigate proposal misses and extra origins
on all five references, independently tracing current detector normalize/enhance/
propose/resolve rejection reasons. Use physical LPS coordinates and CT/parent
connectivity. Distinguish cropping, contrast support, small ostium, merged roots,
wall-parallel tracing and actual missing proposal support. Reference masks/guide
paths are allowed only for scoring/diagnosis, never runtime candidate generation.
Develop bounded GENERAL-PURPOSE recovery hypotheses in this separate experimental
module if findings justify them, without editing detector.py. Examples should
follow actual evidence, not case-specific tuning. Freeze an experiment config
before scoring it; label every newly designed algorithm POST-REFERENCE DEVELOPMENT
and ineligible_for_selection in the clean frozen-family comparison.
Run promising general recovery variants across ALL FIVE and a fixed synthetic
regression cohort covering negative controls, common trunks, nearby openings,
daughter-of-daughter, wall-parallel paths, low contrast and mask imperfections.
Include ablations showing why an improvement occurs. Enforce minimum 2 mm origin
diameter without treating partial-volume unknowns as fabricated measurements.
Do not hard-code expected counts, specific CT geometry or reference coordinates.
Return complete prediction/scoring artifacts with evidence, not speculative fixes.
No production promotion: parent will weigh these against frozen selection evidence.
""",
    },
    {
        "name": "audit",
        "minutes": 25,
        "task": """
Own final_eval_audit.py, tests/test_final_eval_audit.py and
labels/final-eval/audit/ only. Independently audit the released references,
normalization and matching/selection protocol. Confirm 19 instances and hashes
against eval/data/case_19..23 + eval/docs; verify input pairing, NIfTI qform/sform,
LPS physical/voxel paths, mask labels/connectivity, 5mm guide seeds, unit directions,
known radii and parent contact. Read excluded_candidate, review_notes and reviewer
checklist. Do not adjudicate anatomy or change the accepted scoring targets.
Reference thresholds/diameters are uncertain; preserve unknown fields and explain
the 3 measured radii/16 missing. Validate shared scoring against direct
score_references.py on a small control and strict output, including LPS/RAS mirror
diagnostic. Challenge coordinate bugs with rotated/nonzero-origin controls.
Audit image overlap with old pseudo-label training and source/model hashes.
Write any scientific/methodological issues as machine-readable findings. If
final_evaluation.py has a bug, report it to parent in structured output rather
than editing its file. A checksum file has Windows CRLF: splitlines or strip CR
while verifying, do not alter originals. No model score tuning.
""",
    },
]


async def run_unit(job):
    log("Starting " + job["name"])
    result = await agent(
        COMMON + job["task"], phase="comparisons", label=job["name"], schema=SCHEMA,
        repos=[REPO], soft_time_limit_minutes=job["minutes"],
    )
    log(job["name"] + " completed: " + json.dumps(result, sort_keys=True))
    return {"family": job["name"], **result}


async def main():
    await register_workflow({
        "name": "branchseed-final-reference-comparison",
        "description": "Frozen parallel algorithm comparisons followed by case-separated selection.",
        "phases": [
            {"title": "comparisons", "detail": "Five independent model, topology and reference analyses",
             "labels": [job["name"] for job in JOBS]},
            {"title": "selection", "detail": "Consume every family result, select per held-out case and validate",
             "labels": ["selection"], "soft_time_limit_minutes": 30},
        ],
    })
    # User forbids branches; isolated VMs contribute only disjoint files to main.
    # Agent failure aborts before selection; completed agents are resumable.
    results = await parallel([lambda job=job: run_unit(job) for job in JOBS])
    log("All comparison reports available; starting case-separated selection.")
    selected = await agent(
        COMMON + """
Own final_eval_select.py, tests/test_final_eval_select.py,
labels/final-eval/selection/ and FINAL_EVALUATION_RESULTS.md only.
Pull main and consume ALL five recorded result reports below. Investigate failure
or audit findings explicitly; do not silently omit any family or production-blocking
issue. No edits to their source/production files. Build a reproducible aggregator
that validates identical case inventory, prediction hashes and provenance.
Replay scores from predictions using final_evaluation; do not trust cached summaries.
Use eligible finite frozen variants for whole-case leave-one-out selection:
for each case choose best F1 at 3mm on OTHER FOUR, count MAE tie-breaker, then
lower complexity/runtime and stable name. Each held-out prediction must come from
that fold-selected variant, not the all-five winner. Report fold choices and
stability. Keep contaminated or post-reference-designed algorithms in a separate
retrospective/development ranking. Final all-five best configuration/weight report
must be explicitly DEVELOPMENT, not independent held-out accuracy.
Report paired case-bootstrap intervals for baseline differences with fixed seed,
2/3/5mm sensitivity, counts, lost/recovered IDs, geometry and unknown radii. Show
all failures/exclusions and computational limits. Do not create official numeric
weighted score (official evaluator unavailable) or pretend native Windows timing.
Compare deployment recommendation to strict baseline and fixed synthetic
regressions. If an experimental topology recovery improves significantly, describe
evidence and remaining promotion requirements; parent integrates production only
after independent verification. Preserve exact chosen weights/config/model source
hashes, reproducible commands and 5-case challenge prediction bundle for the
recommended deployment. If evidence favors strict, publish that clearly rather
than force an ML promotion. All-five best vs held-out results must remain separate.
Run meaningful leakage tests: altering held-out outcomes cannot change that fold's
selected configuration; tie handling deterministic; missing variants/cases fail.
Run Ruff/mypy on your files and focused tests; commit/pull --rebase/push MAIN.
Do not call entire project done or message user. Parent runs final complete suite,
offline/resource validation and handles any production integration.

Recorded comparison outputs:
""" + json.dumps(results, sort_keys=True),
        phase="selection", label="selection", schema=SCHEMA, repos=[REPO],
        soft_time_limit_minutes=30,
    )
    log("Selection report: " + json.dumps(selected, sort_keys=True))


asyncio.run(main())
