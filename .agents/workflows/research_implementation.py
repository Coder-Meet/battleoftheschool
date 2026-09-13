# ruff: noqa: F821
"""Archived workflow; runtime functions are injected by Devin."""

import asyncio
import json

REPO = "github.com/Coder-Meet/battleoftheschool"
PDF = "https://app.devin.ai/attachments/b52a6c42-ae12-4106-9437-66ceba72e088/%23%20Research%20task_%20ML%20for%20direct%20abdominal-aorta%20dau.pdf"
SCHEMA = {
    "type": "object",
    "properties": {
        "commit": {"type": "string"},
        "files": {"type": "array", "items": {"type": "string"}},
        "interfaces": {"type": "string"},
        "checks": {"type": "string"},
        "findings": {"type": "string"},
        "blockers": {"type": "array", "items": {"type": "string"}},
        "artifact_urls": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["commit", "files", "interfaces", "checks", "findings", "blockers", "artifact_urls"],
}

COMMON = f"""
Implement the user's supplied ML research report for Branchseed.
ATTACHMENT:"{PDF}"
Repo: {REPO}. Read README and your relevant source. Start from current main
(parent inspected ff91eed). This is an existing application, not a greenfield
rewrite. Python 3.13.3, pinned NumPy/SciPy/SimpleITK. Final runtime Windows,
4 CPU cores, 8 GB RAM, no GPU or network; training may be separate.

The user explicitly requires work on MAIN. This overrides default feature
branch/PR instructions. Work and commit only assigned files on main, pull
--rebase before pushing, retry non-destructive rebase/push if another agent
pushes first. No force push, amend, resetting, git config edits or skipped hooks.
Other agents own nonoverlapping files. Do not touch their files, root README,
requirements.txt, pyproject.toml, run.py, benchmark.py, learning.py or detector.py.
The final integration agent handles common wiring and configuration. No PR.
Do not perform the cancelled manual/AI labelling task or alter existing labels.

Production detector must remain unchanged. Neither pseudo-label agreement nor
synthetic accuracy is clinical accuracy. Incoming expert cases can overlap the
25 already inspected/pseudo-trained cases. Do not discard that provenance.
Existing full suite was 105 passed/1 skipped before recent teammate changes.
Current main added same_trunk logic and pseudo-reference comparison scripts;
the old frozen production metrics are NOT automatically current-main metrics.
Inspect current source and hash it for all experiments.

Use apply_patch for manual code; no Any/getattr/setattr or nested imports.
Keep dependencies optional and pinned to versions released at least 7 days ago.
Use documented published APIs. Run Ruff, targeted tests, and mypy for your files.
New tests must test physical correctness, artifact invariants, leakage or
real failure cases, not implementation trivia. Do not modify existing tests.
Commit working source/tests and compact reproducible artifacts, not patient
images or multi-GB data. Upload non-code handoff artifacts with upload_attachment
and return exact URLs. Other sessions cannot access your local files.
Escalate actual blockers in your result rather than inventing successful tests.
Do not notify the user directly. Return the requested structured output.

Shared module contracts (coordinate with these, don't invent replacements):
1. candidate_patches.py defines EXTRA_FEATURE_NAMES =
   ["ostium_local_contrast", "patch_vesselness_std", "parent_wall_curvature"].
   extract_candidate_data(image: sitk.Image, mask: sitk.Image,
   result: detector.Detection, size: int=32, spacing_mm: float=1.0)
   returns CandidateData with patches float32 (N,12,size,size) and
   extra_features float64 (N,3), in result.branches order.
   12 channels are three orthogonal views times CT/parent/vesselness/path.
2. research reviews keep the existing 13 values in row["features"], and add
   row["extra_features"] mapping the 3 names to finite floats,
   row["group_id"] = "synthetic:<seed>" and row["family"].
   Original schema_version=1/scope=candidate_reviews_only/feature_names remain.
   Optional patch metadata row["patch_index"] indexes numeric NPZ patches.
   Never synthesize missing extended features as zeros.
3. TreeModel in tabular_learning.py has load(Path), save(Path),
   scores(matrix), feature_names, threshold, split and metadata.
   Pure NumPy JSON inference; no sklearn needed for scores/load.
   train_trees.py owns optional sklearn training CLI. It accepts review files
   and the existing split JSON train/validation/test lists, with base/extended
   feature selection, source weights and case/group leakage checks.
4. research corpus publishes reviews.json, split.json, per-partition
   patches-{{train,validation,test}}.npz containing numeric array "patches",
   and patches-<partition>.json containing ordered "records". Their
   patch_index values are relative to that partition's NPZ.
   NPZ loading always allow_pickle=False. Preserve candidate fingerprints,
   dataset/detector hashes and unproposed-reference misses.
5. patch_learning.py is optional training; patch_inference.py is optional
   CPU ONNX inference and sidecar metadata. Export explicit physical/channel
   preprocessing and model hashes. Base run.py must not require torch/onnx.
"""

JOBS = [
    {
        "label": "physical-patches",
        "task": """
Own only candidate_patches.py and tests/test_candidate_patches.py.
Implement the shared patch contract and 3 additional features. Use physical
resampling with input direction/origin/spacing, robust scan-relative CT
normalization, finite padding and bounded local memory. Reuse existing
normalize/enhance helpers where appropriate; do not resample a full CT per
candidate. Do not assume zyx==physical xyz. Avoid arbitrary candidate caps.
Ensure parent mask nearest-neighbor and path channels stay aligned; vesselness
must be computed consistently for training/inference and not from reference
labels. Curvature is a local physical parent-wall feature, not a label proxy.
Empty results must return correctly shaped empty arrays; reject invalid grids,
nonfinite input and malformed branch paths. Include native anisotropic and
rotated-origin geometry tests and boundary/padding tests.
Return exact documented interfaces and resource observations.
""",
        "minutes": 20,
    },
    {
        "label": "tree-models",
        "task": """
Own tabular_learning.py, train_trees.py, requirements-trees.txt and
tests/test_tabular_learning.py only. Implement both sklearn
GradientBoostingClassifier and RandomForestClassifier training with small
fixed conservative configurations, plus a safe versioned JSON export and
NumPy-only runtime. Serialize public tree_ arrays, never pickle/joblib.
Strictly validate dimensions, feature order, finite values, indices, acyclic
trees, classes and thresholds. Prove exported scores equal sklearn scores.
Support base 13 and extended 16 features. Keep legacy CandidateModel intact.
Training must use only train rows for estimator fitting, downweight pseudo
sources (default 0.3 vs analytic/expert 1), balance classes within training
without changing evaluation prevalence, reject unknown/missing provenance or
overlapping cases/groups as appropriate. Add optional Platt calibration with
explicit fitting provenance; do not claim clinical calibration. Validation
selects threshold with a declared minimum recall plus training-retention guard;
test cannot choose thresholds/configs. Report keep-all/logistic comparators,
source counts, AP/ROC where defined, calibration diagnostics and undefined
single-class metrics honestly. Declared models should be usable in scores-only
mode without removing candidates. CLI should fail clearly if optional training
deps absent. requirements-trees.txt pinned sklearn version verified published.
Use source-weight controls to permit synthetic-only and mixed experiments.
Do not inspect final sealed test results while iterating model configuration.
""",
        "minutes": 25,
    },
    {
        "label": "reference-evaluation",
        "task": """
Own research_validation.py and tests/test_research_validation.py.
Implement a CLI/library comparing a baseline prediction directory with named
variant directories against provided references, reusing evaluate.py and
score_references.py normalization rather than duplicating them. Require
explicit reference provenance complete_expert / analytic_synthetic /
candidate_pseudo; pseudo or incomplete references must not yield a promotion
pass or be labelled branch-discovery accuracy. Preserve unknown/missing case
errors. Report full per-case/family TP/FP/FN, proposal misses, newly lost/recovered
reference IDs, extra/corrected FP IDs, 2/3/5 mm sensitivity, geometric mean/
median/p95, seed-to-reference-centreline distance and relative radius errors
where references supply the required fields. Do not invent organizer angular/
seed gating rules; discovery matching stays existing one-to-one ostium-based.
Case or supplied seed-group bootstrap confidence intervals with fixed RNG and
paired baseline differences; do not silently bootstrap candidates as independent.
Model training/tuning case overlap report must disclose contaminated reference
cases; distinguish no overlap from unknown split history. Provide a conservative
promotion decision object that rejects FP-only wins obtained by losing true
branches, contaminated evaluation, insufficient provenance, or compute overruns.
Runtime reports may be supplied, but do not invent measurements. Include
organizer reference ingestion validation and a machine-readable comparison
manifest suitable for a later one-command reference evaluation. Use no ML deps.
Test contamination, missing cases, all-negative cases, duplicate matching,
seed polyline projection, deterministic paired intervals and guard failures.
""",
        "minutes": 20,
    },
    {
        "label": "source-audit",
        "task": """
Own RESEARCH_IMPLEMENTATION.md only. Thoroughly read the PDF and verify key
primary papers/data links with web tools. Audit questionable claims: CPU
numbers are not established on our machine; ROC is not calibration; 90 degree
augmentation must transform channels coherently; five tuned cases are not an
independent test; pseudo geometries cannot establish localization; strict
79/(79+15) is about 84% recall rather than an assured high-recall proposal stage.
Map every actionable report recommendation and E0-1 through E4-1 to the shared
module contracts, acceptance gate or explicit data/hardware prerequisite.
Verify annotation granularity/licence/access for the most promising datasets;
do not claim absent datasets globally or download patient data. Record verified
facts with primary links and uncertain/unavailable items as such. Explain why
whole 3D segmentation, DRL, graph learning and regression heads need different
labels/evidence, and what adapter/data contract enables them later. Include
concrete organizer questions. Do not call planned implementation completed
before integration; mark status pending. No speculative skills installation.
""",
        "minutes": 20,
    },
]

async def run_unit(job):
    log("Starting " + job["label"])
    result = await agent(
        COMMON + job["task"],
        phase="foundations",
        label=job["label"],
        schema=SCHEMA,
        repos=[REPO],
        soft_time_limit_minutes=job["minutes"],
    )
    log(job["label"] + " finished: " + json.dumps(result, sort_keys=True))
    return result

async def main():
    await register_workflow({
        "name": "branchseed-research-implementation",
        "description": "Implement the report as optional models, reproducible data and evidence-gated evaluation.",
        "phases": [
            {"title": "foundations", "detail": "Physical patches, CPU trees, evaluation and source verification",
             "labels": [job["label"] for job in JOBS]},
            {"title": "corpus", "detail": "Generate disjoint synthetic data and measure tree experiments",
             "labels": ["synthetic-corpus"]},
            {"title": "appearance", "detail": "Prepare and test optional patch CNN and ONNX deployment",
             "labels": ["patch-cnn"]},
            {"title": "integration", "detail": "Integrate, run offline/resource checks and publish evidence",
             "labels": ["integration"]},
        ],
    })
    # Separate machines; disjoint source files are handed off through the
    # user's explicitly required main branch, with rebase before every push.
    foundations = await parallel([lambda job=job: run_unit(job) for job in JOBS])
    if any(result["blockers"] for result in foundations):
        log("Foundation blockers require investigation: " + json.dumps(foundations, sort_keys=True))
        raise RuntimeError("Foundation work reported blockers; do not conceal or bypass them.")
    log("Building corpus from recorded foundation contracts")
    corpus = await agent(
        COMMON + """
Own research_corpus.py and tests/test_research_corpus.py only, plus compact
research-artifact manifests/reports in a new labels/research/ subdirectory.
Pull main to get foundation commits/interfaces in the structured results below.
Implement plan/generate/export/train-baselines CLI stages. Preregister 15 new
seed groups (10 training, 3 validation, 2 sealed test), all 14 existing stress
families = 210 CTs, before evaluating models. Use fresh deterministic seeds
not 4001, 731927, 582743, 904117, 864203 or 557891. Preserve seed grouping and
case IDs, hash manifests/source/models and refuse overwrites/integrity drift.
Generate all 210 and actually measure current strict and review-union proposal
performance. Preserve unfiltered predictions, references and missed references.
Use synthetic_reviews.label_candidates with ambiguous exclusions; mine only
real unmatched proposals from complete analytic references, never fabricate
negative feature rows or label absent/unproposed references as candidates.
If few negatives survive, report the actual count, use train-only class/source
weights and a preregistered broader training proposal profile if justified.
Do not mutate stress.py/detector.py or alter evaluation proposals to manufacture
an improvement. Keep evaluation prevalence natural.
Export shared 13+extra review format and per-partition physical patch caches
with source fingerprints using candidate_patches. Empty cases still belong to
split and full branch evaluation. Stream cases and cap thread use/memory.
Run E0-1 and E1-1/E1-2: declared base/extended tree models, synthetic-only then
optionally downweighted legacy AI reviews for base-feature comparison.
Freeze all models and validation selection BEFORE inspecting sealed test results.
Compare tree-filter predictions to strict and pool baselines with actual
end-to-end TP/FP/FN, induced misses and FP reductions; do not substitute
candidate-label classification scores. Record whether the report's CNN
stop/go gate passes. Bad results are valid; never tune against the test set.
Publish compact model JSON/review/split/summary artifacts to main where useful;
put CTs/patch caches/full reports in a ZIP attachment for the next agent and
return its URL plus exact reproduction commands. Include no real CTs.
""" + "\nFoundation results:\n" + json.dumps(foundations, sort_keys=True),
        phase="corpus", label="synthetic-corpus", schema=SCHEMA, repos=[REPO],
        soft_time_limit_minutes=40,
    )
    log("Corpus result: " + json.dumps(corpus, sort_keys=True))
    if corpus["blockers"]:
        raise RuntimeError("Corpus blocked; inspect recorded result before continuing.")
    log("Preparing appearance model using actual corpus/gate findings")
    attachments = "\n".join(f'ATTACHMENT:"{url}"' for url in corpus["artifact_urls"])
    appearance = await agent(
        COMMON + """
Own patch_learning.py, patch_inference.py, requirements-cnn-training.txt,
requirements-cnn-inference.txt and tests/test_patch_learning.py only, plus
compact ONNX/model reports under labels/research/ using unique names.
Implement the report's optional small 2.5D classifier (<1M parameters, prefer
~50k), training on the exact shared patch caches. CPU training must work;
no GPU assumption. Torch only for training, ONNX Runtime CPU for inference.
Verify pinned dependency release dates and Windows Python 3.13 wheel support.
Use numeric NPZ loading without pickle, provenance/group-safe splits, balanced
train sampling and configurable AI source weight/smoothing. Augment channels
coherently in physical space: CT-only HU noise/blur/jitter, shared view-mask-path
translations; don't rotate independent views into inconsistent anatomy or
alter radius/geometry labels. Classification only, no fabricated regression
targets on pseudo labels. Training normalizers/statistics come only from train.
Threshold uses validation with recall retention; test does not select epochs,
hyperparameters or combined-model weights. Record source/seed/case overlap.
Export ONNX with full preprocessing/channel contract and verify score parity
and empty/batched inference with CPUExecutionProvider. Missing optional deps
must not break base imports/tests. Keep numerical tolerances explicit.
Implement fixed-weight tree/CNN probability combination (validation-selected
only) and reject mixing mismatched case/candidate/order/features metadata.
Do not train a stacking meta-model on in-sample CNN scores; either simple
declared blending or genuine grouped out-of-fold scores.
Run meaningful tiny offline smoke tests including export parity. If the corpus
E1 gate failed, build/test the ready-to-train pipeline but do NOT run/promote a
full CNN experiment merely to hunt a better test score. Report skipped E2/E3
training with its specific gate. If the gate passed, train synthetic-only
then downweighted mixed-pseudo variants where real patch data can be obtained
with matching fingerprints (never invent missing patches), evaluate frozen
models and combined scores on the same held-out data, and measure CPU cost.
Upload model/data handoff ZIPs needed by integration; source goes through main.
""" + "\nFoundations:\n" + json.dumps(foundations, sort_keys=True)
        + "\nCorpus and gate:\n" + json.dumps(corpus, sort_keys=True) + "\n" + attachments,
        phase="appearance", label="patch-cnn", schema=SCHEMA, repos=[REPO],
        soft_time_limit_minutes=35,
    )
    log("Appearance result: " + json.dumps(appearance, sort_keys=True))
    if appearance["blockers"]:
        raise RuntimeError("Appearance preparation blocked; do not mark it complete.")
    log("Integrating all modules and validating final main")
    all_results = {"foundations": foundations, "corpus": corpus, "appearance": appearance}
    all_attachments = "\n".join(
        f'ATTACHMENT:"{url}"' for result in (corpus, appearance) for url in result["artifact_urls"]
    )
    final = await agent(
        COMMON.replace(
            "Do not touch their files, root README,\nrequirements.txt, pyproject.toml, run.py, benchmark.py, learning.py or detector.py.",
            "You are the final integration agent and may wire/refine completed modules and root configuration."
        ) + """
All writing agents finished; integrate their pushed main commits. Own final
integration: research_run.py or minimal existing-CLI wiring, README,
RESEARCH_IMPLEMENTATION.md status/results, pyproject optional module checks,
.github/workflows/checks.yml, research resource measurement tooling/tests.
Do not change default detector logic or required JSON fields. Keep ML scores
in diagnostics/sidecars. Add explicit scores-only (preserves all daughters)
and filter modes for tree/ONNX/combined models, strict/review-union proposal
selection, physical feature/preprocessing validation and model fingerprinting.
Old --candidate-model legacy behaviour and default run command must work.
Provide one-command synthetic reproduction and organizer-reference comparison.
Reject incomplete/pseudo references as a model-promotion basis and explain
case overlap. Heavy 3D/DRL/graph methods are documented prerequisites, not
silently claimed implemented. Every report action/E0-E4 has done/gated/blocked
status with exact evidence. Do not reverse gates to activate a poor model.

Run Ruff, full mypy including new modules, full base pytest, actual optional
tree/CNN tests with dependencies installed, export score parity and a network-
denied integration test. Use CPU/thread caps; measure wall-clock and peak
process-tree RSS for representative synthetic and supplied real cases using
portable optional psutil tooling. Do not call Linux timings native Windows
measurements. Verify new optional Python3.13 Windows wheels and add a bounded
optional Windows CI smoke job; don't inflate base dependencies with torch.
Compare current-main baseline to research variants, retaining no-filter outputs;
source hashes must agree with corpus source or differences explicitly evaluated.
Run preserved synthetic failure/regression tests; fix actual code bugs, not
tests or thresholds to flatter scores. Current teammate same_trunk may differ
from old e939017 baseline: document which detector is evaluated.

Commit small functional increments to main and push without force. Update
final source/weights/manifests/reports and a compact attachment result bundle.
Do not wait indefinitely on expert references; deliver all ready tools and
clearly list conditional experiments not run. Return final commit, exact test
results, comparison/gate verdicts, remaining limitations and attachment URLs.
""" + "\nRecorded stage outputs:\n" + json.dumps(all_results, sort_keys=True)
        + "\n" + all_attachments,
        phase="integration", label="integration", schema=SCHEMA, repos=[REPO],
        soft_time_limit_minutes=45,
    )
    log("Final result: " + json.dumps(final, sort_keys=True))
    if final["blockers"]:
        raise RuntimeError("Integration reports unresolved blockers; parent must address them.")

asyncio.run(main())
