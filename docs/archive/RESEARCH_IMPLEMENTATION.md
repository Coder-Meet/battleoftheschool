> Historical snapshot. Its status, thresholds and task instructions describe an earlier revision. For current operation use [the workflow](../../PRODUCTION_WORKFLOW.md), [final handoff](../../FINAL_HANDOFF.md), and [the current plan](../../CURRENT_E2E_REVIEW.md).

# Branchseed research implementation and evidence gates

**Status: optional research implementation integrated; production promotion gated.**
The original evidence review was made on 12 September 2026 against
`ff91eed3cbe5f7241c511b25cc69f2bfac042410`. The final integration is on `main`,
starting from `9aeb32e055cada051b2ec0509ab8d98e574b8963`; see the
[current-source results](../../labels/research/current-source-v1/RESULTS.md) and
[final status matrix](#11-final-integration-status).
The specification and acceptance criteria below remain the research protocol;
implemented modules do not imply that their models passed those criteria.
Production `detector.py`, `run.py`, existing labels and the submission schema
were not changed by this integration.

The supplied 35-page *Research task: ML for direct abdominal-aorta daughters*
report was read in full. Its SHA-256 is
`938de7513542732a68f68f38ab63717690dce106c6d26c00af6d0006bc068f31`.
The report's private challenge documents, screenshots, and earlier conversation
references were not independently supplied with this assignment. Challenge
scoring weights and eligibility rules quoted there therefore need organizer
confirmation. Public evidence below was checked through primary papers, official
dataset pages, repositories, and API documentation; no patient data was downloaded.

The three additional user-supplied papers now have a
[separate implementation audit and development comparison](../../ADDITIONAL_PAPERS.md).
Their contact-growth, border-cleaning and PCA-direction adaptations are opt-in;
none passed the first synthetic promotion check.

## 1. Decision and current application

Keep the classical proposal pipeline as the backbone and **measure its recall
before adding rejection**. Compare a small tree classifier with the existing
logistic candidate model first. Add the three contracted context features next;
consider a tiny 2.5D patch classifier only if the simpler model leaves a measurable
problem. Geometric connectivity and instance rules remain responsible for
direct-versus-indirect origin and shared-trunk handling. A candidate classifier
cannot recover a daughter that was never proposed.

Keep strict classical detection as the submission fallback. A learned filter
becomes eligible for integration only after the recall, false-positive, geometry,
runtime, artifact, and provenance gates below pass. Uncertain benefit means keep
the model in research scoring. No experiment here authorizes changing production
`detector.py` or its defaults.

### Current source, rather than a new architecture

| Existing component | Verified behavior and implication |
| --- | --- |
| `detector.py`: `detect`, `DetectorConfig`, `detect_pool` | Physical-space vessel evidence, wall contacts, tracing, and branch resolution already exist. Strict, review, and pooled proposals are distinct experiments. `detect_pool` keeps strict detections not represented in the review result; a broader profile is not a new production default. |
| `same_trunk`, `shares_prefix` | Proximal paths sharing lumen within a physical tolerance can represent one opening. This is an existing geometric rule, not proof that all nearby ostia or daughter-of-daughter errors are solved. New broadening must test both over-merging and duplicate survival. |
| `learning.py`: `FEATURE_NAMES`, `CandidateModel`, `load_reviews` | Existing 13-value candidate vectors, NumPy/SciPy inference, JSON weights, review validation, and disjoint case splits are available. Preserve their interfaces. Existing review loading does not itself implement the proposed extended-feature/group protocol. |
| `synthetic.py`, `synthetic_reviews.py` | Analytic synthetic cases and candidate review generation provide machinery for complete-reference experiments, including unproposed references and ambiguous candidates. Generator realism is a separate question. |
| `evaluate.py`, `score_references.py` | One-to-one ostium matching, discovery counts, and matched geometry errors are available. Reference completeness and provenance must be supplied by the experiment; the evaluator cannot infer them. |
| `compare_e2e.py` | Runs plain/filtered detections and reports per-split results. Its current timing is detection time, and memory is process high-water RSS so far. Windows memory is explicitly unavailable (`null`); it is not the Windows end-to-end timing harness required below. Its generic “test” wording does not establish independent expert evidence. |
| `run.py`, README | Existing offline Windows entry point and JSON output remain the submission contract. Four CPU cores, 8 GB RAM, Python 3.13.3, no GPU/network, and approximately 60 seconds/case constrain deployment. |

The supplied case collection has 25 CT/aorta-mask pairs. At the inspected
revision, `labels/reviews.json` has **237 candidate rows from 23 case IDs**, all
with `labeller="claude"`; the existing split lists all 25 cases. Candidate rows
and nonempty proposal pools are not a complete set of true daughters. Preserve
cases with zero proposals in case-level evaluation and preserve prior exposure
for every case, including cases with no review rows. No manual/AI relabelling is
part of this work.

## 2. Corrections to the report

| Report claim or recommendation | Audit and implementation consequence |
| --- | --- |
| Strict results of 79 TP, 2 FP, 15 FN imply a high-recall proposal stage | Recall is `79 / 94 = 0.8404`; precision is `79 / 81 = 0.9753`. These reported historical counts do not establish adequate proposal recall or current-main performance. Re-run E0-1 with source/config hashes. Filtering cannot fix those 15 misses. |
| Small CNN/ONNX overhead should be below 10 seconds; CPU organ segmentation can take 1.6 seconds | These are an engineering target and an external study result, respectively. Neither is measured Branchseed performance. COBRA downsamples organ volumes to `96 × 192 × 192`; resolving tiny daughter origins is a different task. Measure extraction, model loading, inference, I/O, total time, and peak memory on the target-like Windows machine. |
| ROC curves explain calibration | ROC/PR curves measure discrimination. Use reliability diagrams with bin counts, Brier score/log loss, and uncertainty for calibration assessment. Those scores also mix discrimination and uncertainty; a lower Brier score alone does not prove improved calibration. Never describe pseudo-label calibration as clinical calibration. |
| Tune thresholds on five organizer cases and use them as final held-out results | Choose development or final evaluation before opening expert labels. Threshold tuning, model selection, or training makes them development cases. Even without expert-label exposure, prior image inspection, pseudo-training, or tuning prevents claiming fully independent testing. |
| Pseudo-labelled candidate geometry supports localization evaluation | Geometry copied from detector proposals is circular as a localization target. Binary AI verdicts are noisy too. Use analytic geometry for synthetic machinery; use independently annotated expert geometry for real localization. |
| Rotate patches by 90 degrees, translate ±2 mm | Apply one documented physical transform coherently to CT, parent, vesselness, path, all three views, and any targets. Independent channel/view rotations create contradictory anatomy. Cached central planes cannot generally reconstruct a translated 3D intersection: resample a source volume/cube or disable that augmentation. |
| Keep low-probability proposals in output JSON with a flag | Preserve all proposals, scores, and rejection reasons in research artifacts. Do not add noneligible proposals or undocumented flags to submission JSON. The integration owner must confirm the output schema before any such change. |
| Feed CNN score directly into a boosted tree | The agreed tree inputs are base 13 or extended 16 features. A CNN score is not one of those 16. Stacking requires a separately versioned adapter and group-out-of-fold scores; it is not an implicit extension to the shared contract. |
| Export every model to ONNX | Trees use the contracted pure NumPy/JSON runtime. ONNX is optional for the patch model only. Torch, sklearn, model downloads, and ONNX must not become base `run.py` requirements. |
| VESSEL12 has 20 training scans with vessel labels | Official pages describe three example scans with sparse vessel/non-vessel point annotations and 20 test scans; dense reference vessel segmentations are not downloadable. This is not a densely labelled vessel pretraining corpus. |
| No public dataset matches the task | This search did not verify a ready-to-use dataset with complete variable-count direct-daughter instances and all requested geometry. That is not proof of global absence. AVT and AortaSeg24 deserve a label/access audit before dismissing external data. |
| Drop real candidates whose “synthetic analogue” is ambiguous | An analogue is not a reproducible real annotation. Record ambiguity using an explicit source-specific rule and exclude only from supervised loss as documented; retain the case/candidate and report coverage. Never silently change existing verdicts. |

## 3. Primary evidence and relevance

These studies support design choices or supply potential resources. Their
reported metrics use different cohorts, labels, hardware, and endpoints; none is
a Branchseed accuracy or runtime estimate. “Not verified” means not established
by this audit, not that an artifact does not exist.

| Primary source | Verified task, supervision, and result where relevant | Use here; transfer or availability limit |
| --- | --- | --- |
| [Zheng et al., 2012][zheng] | C-arm CT aorta-part segmentation and eight fixed valve landmarks, including two coronary ostia; discriminative learning from expert annotations. Abstract reports about 1.1 seconds/volume. | Supports coarse-to-fine landmark localization. Fixed cardiac/root anatomy and its compute result do not establish variable abdominal daughter discovery. Reusable training corpus/weights and matching runtime were not verified. |
| [Tahoces et al., 2019][tahoces] | CTA aortic root and supra-aortic/visceral branch landmarks; 33 cases for parameter adjustment, 30 independently selected evaluation cases. Reports branch recall 91.8% and precision 98.8%. | Especially relevant classical landmark/connectivity baseline. Borrow evaluation and geometric reasoning, not its accuracy. Public compatible complete labels and portable implementation were not verified. |
| [Han et al., 2016][han] | Bayesian coronary tracking with active search for branches and seemingly disconnected segments, evaluated with Rotterdam coronary CTA. Open-access article is CC BY. | Candidate tracing/gap recovery inspiration for E4-1. Coronary lumen paths and origin definitions differ; a gap bridge needs explicit false-connection controls. No Branchseed CPU claim. |
| [Schaap et al., 2009][schaap] | Standardized coronary centerline extraction evaluation and cardiac CTA reference database, including point correspondence and overlap/distance concepts. | Useful geometry evaluation framework, not a substitute for branch-level ostium matching or complete abdominal references. Access restrictions below. |
| [Sensors coronary-ostia DRL, 2021][drl-coronary] | Sequential landmark localization using orthogonal 2.5D patch states. | Supports a compact patch representation. Its landmark labels, reward/action policy, and fixed coronary targets are different from variable-count discovery; do not implement RL from candidate verdicts. Training artifacts and target-CPU performance not verified. |
| [TBAD landmark DRL][drl-tbad] | 396 CTA scans in type-B dissection cohorts, nine manually annotated aortic landmarks; internal-test median errors of 2.5 mm (cluster) and 2.7 mm (single agent). Median processing time is 1 second for a single agent's prediction. | Requires a landmark dataset and task-specific policy. That timing is not a full multi-landmark episode. Dissection/fixed-target localization is not evidence of completeness for normal, accessory, or small direct daughters. Cohort access/reuse permission not verified. |
| [COBRA, 2022][cobra] | Small 3D CNN for liver/kidney/spleen/pancreas segmentation; 361 training cases, downsampling to `96 × 192 × 192`, ONNX graph optimization; reports 1.6 seconds/image on CPU. | Evidence that optimized CPU neural inference can be feasible. Coarse organ labels, spatial resolution, and hardware scope prevent copying latency or accuracy to this application. |
| [Context-aware cascaded U-Net, 2023][cacu] | CTA AAA lumen and intraluminal thrombus segmentation in 70 patients; dense, multi-class supervision. | Useful understanding of mural thrombus/context. Does not supply direct-daughter instances; whole-volume training and label requirements exceed candidate reviews. Reusable dataset licence not verified. |
| [de Bruijne et al., 2003][asm] | CTA aneurysm segmentation using statistical shape and learned boundary appearance from true/false profiles. | A precedent for local learned evidence with geometric constraints. Aneurysm boundary labels and shape priors are not branch-origin labels; no ready Branchseed model verified. |
| [clDice, CVPR 2021][cldice-paper]; [official code][cldice-code] | Topology-preserving tubular segmentation loss using overlap and soft skeletons; official implementation is MIT. | Future dense-segmentation loss, not candidate binary loss or proof of direct connectivity. Requires vessel masks; skeleton continuity alone cannot distinguish a nearby vein or indirect branch. |
| [Topology-aware uncertainty, NeurIPS 2023][topology-uncertainty] | Segmentation-derived structures and graph-based joint uncertainty inference with ground-truth supervision. | Possible future structural confidence research. It is neither validation of `evidence_score` nor a drop-in candidate probability calibrator. Needs a segmentation/graph-target adapter. |
| [Retinal multi-branch dynamic convolutions][retinal]; [PASC-Net][pasc]; [GA-TAN][gatan] | Vessel segmentation/topology methods. The first evaluates retinal images; GA-TAN evaluates OCTA Retina3D and MRA TopCow with 13 Circle-of-Willis classes. PASC-Net proposes shape-adaptive convolutions and hierarchical topology constraints. | Architecture inspiration only. Dense labels, anatomy/modality shift, custom operators, and CPU export need separate evidence. Do not infer abdominal branch discovery from vessel Dice or retinal connectivity. |
| [AVT, 2022][avt-paper] | 56 CTAs and semi-automatic binary aortic-tree masks; some branches explicitly absent from masks because of acquisition quality/resolution. | Higher-priority external geometry/pretraining audit; incomplete masks cannot establish complete small-branch recall. |
| [AortaSeg24, 2025][aortaseg-paper] | 100 CTA volumes with 23 named branches/zones; 50 training, 10 hidden validation, 40 hidden test during the challenge; DSC and normalized surface distance evaluation. | Promising branch-semantic labels, but named classes are not arbitrary instances or explicit ostia/short paths. Access and completeness audit required. |
| [AortaExplorer paper][aortaexplorer-paper]; [official code][aortaexplorer-code] | Aorta segmentation, centerlines/landmarks, and measurements; repository is MIT, uses TotalSegmentator and requires its `heartchambers_highres` licence. | Development/reference pipeline only. MIT code does not license upstream weights/data; complete direct-daughter labels and target-runtime feasibility are not established. |

The report's [vessel-breakage review][breakage-review] is secondary evidence for
failure modes, not an independent validation cohort. Remaining background theses,
aggregator pages, and alternative architectures in its bibliography are not a
validated dependency list. No execution, model reproduction, external patient
download, or licence agreement acceptance was performed for these studies.

## 4. Dataset audit and admission rules

### Most useful sources to investigate

| Dataset and primary access source | Modality, size, resolution, and annotation granularity | Licence/access and practical availability | Permitted conclusion and next gate |
| --- | --- | --- | --- |
| [AVT article][avt-paper], [Figshare record][avt-data], [metadata API][avt-api] | 56 CTAs: KiTS 20, RIDER 18, Dongyang 18. Binary aortic-tree masks, not per-daughter instance IDs. Article reports slice thickness ranges 0.5–5, 0.625–2.5, and 2–3 mm respectively. Celiac/SMA and other branches are missing in some segmentations; not every case is complete. | Figshare lists CC BY 4.0, public downloads enabled, three data ZIPs totaling about 6.31 GB. Record explicitly says underlying collections' appropriate licences apply. Downloads/archives were not inspected. Check KiTS/RIDER/Dongyang provenance and derived-weight/redistribution terms separately. | Best immediate audit for parent/branch geometry and thick-slice stress. After permission, map binary connected trees to parent and branch candidates, then obtain independent instance/ostium review. Unlabelled branches must be unknown, not negative. No complete-recall claim from these masks alone. |
| [AortaSeg24 dataset][aortaseg-data], [access instructions][aortaseg-access], [paper][aortaseg-paper] | Official data page describes 50 CTA images resampled to isotropic 1 mm, axial dimensions 389–516 pixels. Paper explains 100 total cases as 50 train + 10 hidden validation + 40 hidden test. The 23 labels include celiac, SMA, left/right renal and iliac arteries plus aortic zones. | Verified account, signed DocuSign agreement, challenge membership, dataset access request, and approval are required; official page says approval can take up to 24 hours. Not accessed here. Public pages do not establish permission for this separate competition, redistribution, or all 100 downloadable labels. | The 50-vs-100 descriptions refer to release versus total challenge scope; do not claim 100 downloadable expert-labelled cases. Request current release/terms and ontology. Fixed renal/celiac classes do not establish accessory renal, lumbar, IMA, or arbitrary direct-daughter completeness. |
| [TotalSegmentator CT dataset v2.0.1][totalseg-data], [official project][totalseg-code] | 1,228 routine CTs with 117 semantic structures across varied scanners/protocols; dataset record is about 23.6 GB. Native spacing is heterogeneous, not a verified fixed isotropic specification. Broad organ/vessel masks, not full Branchseed instances. | Dataset record declares CC BY 4.0. Repository lists open `total` task under Apache-2.0 and separate licensed tasks; data, code, and weights are distinct assets. Public links exist; no archive downloaded or optional task provisioned. | Development-only context/pretraining candidate after exact class audit. Whole-aorta/organ masks do not certify daughter absence, and model outputs remain pseudo-labels. Do not bundle TotalSegmentator into the detector. |
| [VESSEL12 details][vessel12], [rules][vessel12-rules] | Lung CT with/without contrast, healthy/diseased cases; slice spacing at most 1 mm, mostly near-isotropic. Three example scans with vessel/non-vessel annotation points and 20 test scans. Dense reference segmentations are explicitly not downloadable. | Official rules restrict use to challenge participation. Public example/test instructions do not grant unrelated training rights. No patient data downloaded. | Not ready for dense pretraining or abdominal origin supervision. Seek separate permission if needed; otherwise use only evaluation ideas. Sparse consensus points cannot measure missed branches. |
| [Rotterdam coronary centerline evaluation][schaap], [official access site][rotterdam] | Cardiac CTA centerlines/radii rather than abdominal direct-daughter instances; 32 datasets, eight training and 24 evaluation, with four selected vessels per case. This is selected-vessel coverage, not all branches. | Published access process requires registration and a signed confidentiality agreement. Current download operation and transfer/competition permissions were not verified. | Centerline/radius methodology may transfer; anatomy, motion, contrast, and branching semantics do not. Do not infer complete abdominal labels or open reuse from a public paper. |

**Other report-mentioned sources:** [LiTS primary paper][lits] describes 131
training and 70 test abdominal CT volumes labelled for liver and tumors, not
aortic daughters. Its paper's CC BY-NC-ND notice does not establish the dataset's
licence; mirrors report other terms, so obtain the official dataset terms before
use. AortaExplorer and the TBAD/AAA study cohorts require author/release-level
access and annotation verification; this audit does not establish them as
downloadable, reusable branch datasets. Prefer AVT/AortaSeg24 investigation over
unrelated organ or retinal data.

### Admission manifest, before any future external-data experiment

The corpus owner must record dataset/release/source URL, retrieved licence text
and date, permitted use and redistribution/weights terms, access approval,
patient/group mapping, modality/contrast, native spacing/origin/direction,
field-of-view, annotation ontology and completeness, reviewer provenance, hashes,
and all transforms. Do not use missing semantic classes as confirmed negatives.
Do not re-host scans in Git or attachments.

Keep permissions for **code**, **weights**, **images**, and **annotations**
separate. An accessible download is not permission to use it in this challenge.
The permitted use and label audit, not physical availability alone, determine
admission. Existing organizer data retains its original scope and exposure
history; this document authorizes no new annotation work.

## 5. Shared implementation contracts

### 5.1 Candidate extraction: `candidate_patches.py`

The owner must provide this exact public contract:

```python
EXTRA_FEATURE_NAMES = [
    "ostium_local_contrast",
    "patch_vesselness_std",
    "parent_wall_curvature",
]

def extract_candidate_data(
    image: sitk.Image,
    mask: sitk.Image,
    result: detector.Detection,
    size: int = 32,
    spacing_mm: float = 1.0,
):
    ...
```

Return `CandidateData` in **`result.branches` order**, with
`patches: float32[N, 12, size, size]` and
`extra_features: float64[N, 3]`. Zero candidates must yield shape-correct empty
arrays. All numeric values must be finite; never replace absent extended
features with zeros.

The 12 channels are three orthogonal views, each ordered **CT, parent,
vesselness, path**. Use a recorded, fixed view ordering (axial, coronal, sagittal)
and record the actual physical axis and pixel-axis mappings. Path is required
in this shared contract, despite being optional in the PDF.

Default sampling is 32 × 32 at 1 mm, centered on the candidate ostium: nominally
32 mm field width, with 31 mm between outer pixel centers. Record the even-size
center convention explicitly. Physical coordinates must honor image
origin/spacing/direction and the detector crop/resampling transform; never
confuse SimpleITK `(x,y,z)` with NumPy `(z,y,x)` or assume identity direction.
Use linear interpolation for continuous channels, nearest-neighbor for binary
masks/path, explicit outside-volume padding, and an explicit path-rasterization
rule in physical space.

The extractor owner defines and versions actual preprocessing and feature
formulas; integration must use those exact definitions, not a second
implementation. Metadata must specify:

- CT HU window and scaling, vesselness normalization, binary channel values,
  view centers/orientations, padding, and handling of invalid/empty masks.
  The PDF's 100–400 HU window is a candidate setting, not an established choice;
  clipping must not erase weak-contrast distinctions without an ablation.
- `ostium_local_contrast`: the physical foreground/background neighborhoods and
  contrast units; no unspecified “local” statistic.
- `patch_vesselness_std`: the channel/view aggregation, normalization, and
  population/sample convention.
- `parent_wall_curvature`: signed/unsigned definition, physical scale/units,
  smoothing and boundary/degeneracy handling.

**Gate P:** rotated/anisotropic image tests must recover the same physical
locations; channel co-registration, path ownership, branch ordering, empty
results, finite features, crop-edge padding, and nonmutation of `Detection` must
be demonstrated. Feature/patch metadata must identify extractor and detector
hashes. Gate P now has the physical extractor and regression evidence in §11;
this protocol does not choose alternate implementation-specific
formulas on behalf of the extractor owner.

### 5.2 Review and corpus contract

Preserve top-level `schema_version=1`, `scope="candidate_reviews_only"`, and
`feature_names=learning.FEATURE_NAMES`. `row["features"]` remains the **same
13 finite numeric values in the same order**:

```text
radius_mm, mean_vesselness, evidence_score, path_length_mm,
seed_distance_mm, tortuosity, path_hu_relative, bone_distance_mm,
parent_angle_degrees, arc_position, native_spacing_mm,
connector_gap, candidate_volume_mm3
```

Research rows add `row["extra_features"]`, mapping exactly the three extra names
to finite floats; `row["group_id"]="synthetic:<seed>"` for synthetic cases; and
`row["family"]`. Optional `row["patch_index"]` points into the corresponding
partition's patch array. Real groups must represent the patient/source identity,
not candidate ID, scan filename, or a synthetic-looking surrogate.

Existing base-only reviews can remain usable by the base model. For extended
training, fail clearly or exclude and count rows lacking valid extra features;
do not fabricate zeros. Re-extraction must reproduce and match the recorded
candidate fingerprint and source transforms before attaching numeric features.
It must not overwrite old features/verdicts or silently attach changed proposals.

Publish the corpus as:

```text
reviews.json
split.json
patches-train.npz           patches-train.json
patches-validation.npz      patches-validation.json
patches-test.npz            patches-test.json
```

Each NPZ contains numeric array `"patches"` and is loaded with
`allow_pickle=False`. Each matching JSON contains ordered `"records"`.
`patch_index` is relative to that partition's NPZ, never a global index. Validate
row counts, unique keys, order, bounds, shapes, dtype, finite values, and
fingerprint agreement across reviews and sidecars. Keep dataset, generator,
detector, extractor, config, split, and file hashes; record proposal profile.
Unproposed-reference misses must remain in the case-level manifest even though
they have no patch. Keep ambiguous candidates and their exclusion reason;
candidate-only training never removes them from discovery denominators.

**Gate D:** reject case/group overlap, reused anatomy under a new seed/name,
cross-partition duplicate fingerprints, mismatched hashes, object NPZ arrays,
nonfinite values, stale candidate ordering, and out-of-bounds patch indexes.
Preserve immutable source labels separately from derived training targets and
weights. No existing labels are changed by this plan.

### 5.3 Tabular learning and inference

`tabular_learning.py` must expose `TreeModel.load(Path)`, `save(Path)`,
`scores(matrix)`, `feature_names`, `threshold`, `split`, and `metadata`.
Loading and scoring use **pure NumPy and JSON**, without sklearn or pickle.
`train_trees.py` owns optional sklearn training and accepts review files, the
existing split JSON's train/validation/test lists, base/extended selection,
source weights, and case/group leakage checks.

Compare the unchanged logistic model with a small tree model on **identical
proposal pools and groups**. The PDF's depth 3–5 and 100–300 trees are a bounded
search suggestion, not a mandated architecture or XGBoost dependency. Prefer a
sklearn estimator whose learned tree structure and output transformation have
documented, reproducible export semantics. A simpler supported tree/forest is
acceptable if boosting export cannot pass parity. Fit transformations and class
balancing on training data only.

The JSON artifact must record model/schema version, ordered feature names,
tree/ensemble/output semantics, threshold and its selection rule, split/groups,
source counts/weights, excluded-row counts, random seed, hyperparameters,
training/library versions, and input/source hashes. Validate tree indices,
acyclicity/reachable leaves, feature dimensions, class order, finite thresholds
and values, and rejection of malformed models. No arbitrary executable payloads.

**Gate T:** round-trip and native-estimator/export score parity, including exact
split-boundary values, empty batches, feature-order mistakes, and malformed
artifacts; no sklearn import during inference. State numeric tolerance and
require threshold decisions to agree, especially near the threshold. Grouped
validation must beat meaningful false-positive counts without violating the
recall gate. Feature importance is descriptive; grouped permutation/ablation
is preferable to treating split importance as causal.

### 5.4 Optional patch model and CPU export

`patch_learning.py` is optional training; `patch_inference.py` is optional CPU
ONNX inference with a sidecar. The integration owner controls opt-in wiring.
Base `run.py` must work with neither Torch nor ONNX installed.

Start with binary classification of **candidate direct/scoring-eligible daughter
versus other**. The PDF's small design is a research starting point: 3–5
convolutional blocks, 3 × 3 kernels, 32–64 filters, optional pooling, global
average pooling, and a small output head, aiming below one million parameters.
Compare a smaller version before adding capacity. Architecture size is not
proof of target-device latency. Per-view towers are a later ablation and still
consume the contracted 12-channel data.

Use binary cross-entropy with explicitly recorded source/class weights.
Synthetic training samples must include true daughters, adjacent/touching veins,
calcification, crop caps, indirect branches, and wall-parallel/low-contrast path
failures. Sample/weight per case so a large proposal pool does not dominate.
Evaluate on the original, unsampled candidate distribution.

Pseudo-label mixing is a separate E2-2 ablation. The PDF's weight 0.3 versus
synthetic 1.0 and label smoothing to 0.9/0.1 are **hypotheses**; compare against
weight zero and unsmoothed targets on grouped validation. Preserve raw verdicts,
labeller, confidence/ambiguity rule, and derived target/weight separately.
Confident AI scores are not expert confidence. Do not use synthetic analogues
to silently discard difficult real examples. For tree algorithms that require
hard labels, do not pretend soft-target smoothing was implemented.

Augmentation is training-only: HU jitter/noise affects CT, not binary masks;
blur/thick-slice simulation uses a physical acquisition model. Spatial transforms
must obey Gate P and transform vector/regression targets too. For 90-degree
rotations, transform every channel coherently and permute/flip orthogonal views
as required by a single 3D coordinate transform. Do not independently rotate
planes or swap anatomical axes without metadata/target updates. With only
cached central planes, disable unsupported translations/rotations rather than
inventing missing image content.

Export a frozen eval-mode model and record ONNX opset, input/output names, shape
and dtype, class order, logits versus probabilities, preprocessing version,
extractor/detector hashes, threshold, split, training versions, weight/ONNX hashes,
and SHA-256 of the sidecar in the run manifest. Use the documented
[`InferenceSession` API][ort-api] with `CPUExecutionProvider`; set a bounded
[`SessionOptions` thread budget][ort-threads]. Batch candidates within measured
memory limits instead of spawning four-thread sessions per candidate.

**Gate N:** native-model versus ONNX parity for real-shaped and empty/boundary
batches, consistent threshold decisions, malformed-sidecar rejection, no network
requests, and CPU-only provider verification. Static INT8
[quantization][ort-quant] is a later measured ablation: use training-only
calibration inputs, recheck quality/parity, and confirm supported operators and
actual speedup. Quantization calibration is not probability calibration.

Optional versions must be pinned, published at least seven days before selection,
and have verified Python 3.13.3/Windows x64 support. This document adds no packages.
Offline wheel/model provisioning is an integration prerequisite; a Linux virtual
environment is not a Windows bundle.

### 5.5 Connectivity, instance geometry, and combined scores

Keep independent geometry fields: `instance_id`, `parent_instance_id`,
`ostium_xyz_mm`, `seed_xyz_mm`, positive `radius_mm`, and unit `direction_xyz`.
Directness means an opening into the supplied parent, not eventual ancestry in
the same vessel tree. The judge confirmed common trunks count once and a vessel
returning through a second aortic opening counts twice. Overlapping-opening and
vein scoring allowances still require the official evaluator; see the
[updated rules](../../SUBMISSION_AUDIT.md#latest-judge-clarifications).
Indirect branches and disconnected nearby vessels do not acquire an aortic
opening through a classifier decision.
Use geometry/evidence together for gaps: mere proximity does not prove a lumen
connection, while a conservative one-voxel connectivity test may miss a true
partial-volume origin.

Classifiers may rank or filter existing candidates. They must not silently move
ostia, change radii, invent an anatomical branch count, or replace `same_trunk`
with class labels. Preserve pre- and post-filter predictions and every score/
decision in research artifacts.

For E3-1, compare strict classical, tree-only, CNN-only, and a **predeclared
combined decision rule** on the same candidate IDs. A validation-selected joint
threshold rule can combine separately logged scores without changing tree input
features. Feeding a CNN score into a tree is a separate pending extension: train
the tree using group-out-of-fold CNN scores, version the feature/adapter schema,
and keep the final test untouched. Do not train a stack on in-sample CNN scores.

## 6. Deferred approaches and the data adapter they need

| Approach | Why candidate reviews are insufficient | Adapter/labels and evidence required before reconsidering |
| --- | --- | --- |
| Whole 3D segmentation, cascaded/attention U-Nets | Binary candidate verdicts do not identify unproposed vessel voxels. Whole-aorta masks or organ Dice do not establish small direct branches, ostia, or instance hierarchy. | Dense daughter/lumen masks with coverage/unknown-region masks, parent identity, branch hierarchy, independent ostia and proximal centerlines; patient splits, legal reuse, training compute, and measured offline CPU/RAM/export viability. Evaluate branch discovery after mask-to-instance conversion, not Dice alone. |
| clDice, dynamic convolutions, topology-aware dense networks | These change dense prediction objectives/operators; candidate classification has no dense target or skeleton to supervise. A continuous wrong vein can score well on connectivity. | Same dense adapter, topology/instance ground truth, artery-versus-vein checks, operator support and target-runtime profiling. Require improvement in direct-daughter errors over a simpler dense baseline. |
| Graph learning/graph uncertainty | An unordered candidate list has no reliable labelled edge set or graph target. A model could learn detector errors or join nearby unrelated vessels. | Nodes keyed to physical candidates/centerlines, edges with geometric evidence, ground-truth connectivity, direct/indirect/vein/unknown labels, parent-child instance IDs, unmatched reference nodes, and graph-extraction version. Evaluate recovery and false connections separately; keep geometric fallback. |
| DRL landmark navigation | Fixed coronary/TBAD landmarks do not specify how to enumerate a variable number of daughters or stop safely. Candidate labels do not supervise navigation away from candidates. | Independent target coordinates and existence/count annotations, state/action coordinate frame, reset/start distributions, trajectories or simulator/reward definition, stopping/visited-target/duplicate rules, and training budget. Test missed targets, false stopping and full episode CPU cost against classical proposals. |
| Ostium/seed/radius/direction regression heads | Pseudo geometry copied from the detector rewards reproducing the detector. Even valid binary labels do not specify offset, lumen center, radius, or direction. | Expert ostium/seed/centerline/radius/direction with units, measurement protocol and uncertainty. Synthetic regression can validate machinery only. The PDF's Smooth L1 offset/radius and cosine direction losses with small regression weight are later ablations; require correct physical/vector transforms and independent real geometry before deployment. |
| Dense wall-scanning patch proposals | Current patch extraction depends on an existing branch and its path channel. A classifier is not automatically a proposal generator. | Explicit wall sampling/coverage and negative targets, a defined path-channel treatment when no path exists, proposal deduplication and geometry generation. Measure all unproposed references and additional scan cost. This is beyond the shared candidate contract. |

A future source adapter must emit original image geometry, parent mask,
dataset/patient/group IDs, acquisition and licence provenance, annotation source,
`complete/partial/unknown` coverage, explicit unknown regions, and reference
daughter instances with physical geometry and optional dense masks/centerlines.
Map dataset semantic labels to instances with a recorded mapping and independent
review; a named renal class cannot silently encode several accessory renal
origins. Keep references separate from detector outputs and record raw-to-working
transforms and content hashes. The adapter then feeds the existing physical
evaluator and candidate corpus; it does not overwrite review schema version 1.
This adapter and all methods in this section are **pending, data-gated research**.

## 7. Evaluation protocol and acceptance gates

### Three different estimands

1. **Proposal recall:** evaluate unfiltered strict/review/pool candidates against
   all eligible reference daughters in fully annotated regions. Record
   unproposed misses and false/duplicate proposals.
2. **Conditional classification:** evaluate candidate labels/scores given that
   the proposal exists. Report per-source confusion counts, precision/recall/F1,
   PR/ROC curves when both classes exist, abstentions/exclusions, and retained
   true proposals. This is not discovery recall.
3. **End-to-end discovery:** evaluate each final prediction against the same
   full case references. All unproposed misses remain false negatives. A
   duplicate competing for one reference is not a second true positive.

Use one-to-one matching as in `evaluate.py`, with predeclared ostium tolerance:
3 mm as the primary local benchmark, and 2/5 mm sensitivity analyses. Report
matched ostium and seed distance, absolute/relative radius error, and direction
angle separately, with count, mean/median and tail (e.g. p95) where sample size
allows. Seed distance to a supplied seed is different from distance to a
reference centerline or confirmation of lumen membership. Do not claim the
latter without those annotations.

The PDF's 5 mm seed and 30-degree direction criteria are proposed geometry
acceptance measures, not verified organizer discovery rules. Do not silently
change ostium matching to incorporate them. Publish an additional geometry-
qualified result only with its exact definition. Report unmatched branches as
well as conditional localization: dropping hard matches can improve mean error
while worsening discovery.

### Split, exposure, and label provenance

- Group all candidates, augmentations, reconstructions, and repeated scans of
  the same patient together. Synthetic `synthetic:<seed>` groups represent
  anatomy; reusing an anatomy/template under a different case ID must not cross
  partitions. Shared broad parameter ranges can support in-distribution tests;
  duplicate realizations cannot. Add separate held-family/parameter-range stress
  tests for extrapolation and report their difference.
- Preserve existing case split lists and add group validation; do not silently
  reshuffle a historic test into training. Use train-group cross-validation and
  validation-only threshold/model selection. Lock the final test until selection
  is complete; do not repeatedly choose models on it.
- Maintain an exposure ledger: patient/case/group, source, image/hash, prior
  inspection, detector tuning, pseudo-labelling, training/calibration usage,
  expert-label availability/version, and dates. Organizer labels can overlap
  all these categories. A new label source does not create a new patient.
- An organizer case not used in training but visually inspected is still
  exposed. An overlap case evaluated after freezing is a **retrospective expert
  audit on exposed images**, with explicit history. Only genuinely untouched
  patients justify an independent test claim. If five cases are used for
  fine-tuning/thresholds, report development results and seek new final cases.
- Keep pseudo agreement, analytic synthetic accuracy, expert exposed-case audit,
  and independent expert test in separate tables. Incomplete references support
  only appropriately scoped evaluation, not full precision/recall from treating
  unknown structures as negatives.

### Operating point, uncertainty, and failure strata

Choose thresholds on validation for a **predeclared allowed true-positive loss**,
not maximum training F1. Until the organizer/team agrees a nonzero loss budget,
the conservative integration gate is **zero additional observed eligible misses
on the locked evaluation cases**, alongside useful false-positive reduction.
This finite-sample gate is not a guarantee about unseen patients. Do not tune
after seeing test results to make the gate pass. If a model removes no meaningful
false positives, or its only effect is pruning true daughters, retain classical.

Report pooled counts and per-case metrics; bootstrap complete cases/groups,
paired across methods (e.g. 2,000 resamples, fixed seed), for F1 and changes in
recall/false positives. Never bootstrap individual patches as independent
patients. State the interval method and handling of no-positive cases; keep
undefined metrics explicit rather than turning them into perfect scores.
With about five real cases, intervals and tails are unstable: show every case,
counts and annotation uncertainty, and avoid strong generalization claims.

Include zero-daughter controls and zero-proposal cases. Define strata before
comparison: thick slices (PDF suggestion: native z-spacing at least 3 mm),
low contrast, calcification, mural thrombus, curved/wall-parallel branches,
dense nearby origins, crop ends, adjacent veins, short stubs, and indirect
daughter branches. Record true-positive losses, misses, duplicates, false
connections, localization and runtime per stratum. Dense synthetic data can
expose pipeline failures but cannot establish their real prevalence.

For [probability calibration][calibration], fit only with training-group
out-of-fold predictions or a separate validation calibration subset. Record
source/domain, bin populations, reliability curves and proper scoring rules.
If data are too sparse, leave scores labelled uncalibrated. Never recalibrate
on the final expert test, and do not claim calibration of true disease/branch
probabilities from noisy AI labels.

### Compute and deployment gate C

Run a fresh process per case to measure total peak memory rather than cumulative
RSS from a multi-case process. On Windows, use a supported measurement harness;
do not call the current `resource`-based helper a Windows benchmark. Record CPU
model, OS/architecture, core/thread limits across SimpleITK/BLAS/ONNX, RAM,
Python/package versions, candidate counts, source/config/model hashes, and
input provenance. Measure cold start and warm repeats, disk load, extraction,
inference, resolution/postprocessing, JSON writing, total wall time, and peak
working set; include largest/highest-candidate stress cases.

The provisional gate is approximately **60 seconds total** and below **8 GB
physical RAM with headroom**, plus valid offline output on four CPU cores.
Clarify whether the organizer enforces max, percentile, or mean time and whether
I/O counts. The PDF's under-10-second added-model budget is subordinate to the
total budget. A fast score call with slow patch extraction fails this gate.
No such benchmark was run for this document. Offline install/import/inference
must succeed with pinned wheels and model files and no downloads; explicitly
requested optional models must fail clearly on invalid artifacts, while the
default classical entry point remains usable without optional dependencies.

## 8. Experiments E0-1 through E4-1

**This table defines the original protocol; executed/gated status is in §11.** Each run
must publish a compact manifest, split/exposure ledger, per-case results and
reference-completeness counts, predictions before/after filtering, unproposed
misses, model/threshold/feature configuration, timings/memory, and hashes.
Store large patches/scans outside Git; version compact JSON and model artifacts
only where permitted. Gates P/D/T/N/C above are prerequisites, not passed badges.

| ID and hypothesis | Concrete owners/contracts and artifacts | Acceptance and stop/go decision |
| --- | --- | --- |
| **E0-1 — measure the current strict baseline before filtering** | Corpus/evaluation owners: extended analytic synthetic cases, grouped seed/family split and complete references; current `detect`/`evaluate.py` with exact source/config hash. Record proposal precision/recall/F1, localization, per-case runtime/memory and strata. Include review/pool as separately labelled comparators, not strict substitutes. | Establish a reproducible baseline, not the archived 79/2/15 counts. If strict misses are substantial in multiple strata, prioritize targeted classical proposal work before a filter. A filter can proceed as conditional research, but cannot be sold as fixing proposal recall. |
| **E1-1 — a tree improves candidate rejection over logistic** | `train_trees.py` + `TreeModel`, initially base 13 features, Gate D/T. Run synthetic-only first; test existing AI reviews only as a separately weighted ablation, preserving source and case/group separation. Save JSON model, native/export parity, split, source weights, and validation operating point. | Compare identical pools against classical and existing logistic. Require meaningful false-positive reduction and the predeclared TP-retention gate, grouped uncertainty and per-stratum counts. No useful gain, recall loss, leakage, or export mismatch: keep strict/logistic as appropriate; no promotion. |
| **E1-2 — three physical context features add useful signal** | `candidate_patches.py` → `extra_features` → extended 16-feature `TreeModel`; same rows/groups/proposals as E1-1. Gate P/D/T, feature definitions and source hashes, missing/excluded-row counts, base/extended ablation and grouped feature importance. | Preserve base 13 exactly. Reject fabricated zeros, mismatched populations, or gains disappearing on grouped validation/held strata. Additional extraction cost must fit C. Small or unstable benefit: retain the base feature model. |
| **E2-1 — tiny 2.5D classifier adds information beyond tabular** | Corpus partition NPZ/ordered JSON, `patch_learning.py` synthetic-only, `patch_inference.py` CPU ONNX. Gate P/D/N; coherent preprocessing/augmentation and training/export metadata. Compare with E1-2 and a smaller CNN. | Require conditional and end-to-end gain without unacceptable retained-TP loss, plus extraction-inclusive target-CPU measurement. Overfitting to synthetic textures, no meaningful gain, malformed artifacts or excessive compute: research-only, no integration. |
| **E2-2 — downweighted real pseudo patches help transfer** | Same contract as E2-1, immutable AI verdict provenance, source weights/ambiguity/smoothing ablations, case-group splits and exposure ledger. Weight zero is the synthetic-only control. No new AI/manual labelling is authorized. | Improvement must survive held synthetic strata and any genuinely held expert evidence. Pseudo agreement alone is not success. Degradation, source shortcuts, or leakage: reject mixing. Without expert data, report only synthetic metrics and pseudo agreement; real transfer remains unmeasured. |
| **E3-1 — combine filters without losing discovery** | Evaluation/integration owners: strict classical, tree-only, CNN-only, predeclared combined-score rule on identical candidate IDs. Same complete synthetic test groups; frozen thresholds; plain/final predictions, misses and geometry. CNN-in-tree stacking needs the separate adapter/OOF gate in §5.5. | Require false-positive reduction at the locked TP-loss budget, acceptable geometry/duplicate behavior, paired uncertainty and C. Retain all unproposed misses in denominators. Reject combined complexity that adds no reliable benefit or trades away eligible daughters. |
| **E3-2 — audit organizer-labelled cases after pipeline freeze** | Evaluation owner: independent reference JSON/optional centerlines, completeness declaration, group/hash overlap audit and exposure ledger. Score strict and frozen research choices with no retuning. Report exposed versus untouched cases separately and all geometry/counts. | If previously inspected/pseudo-trained, call it an exposed-case expert audit, not an independent test. If completeness or eligible-origin definitions are absent, scope metrics or block the claim. About five cases support limited evidence; any subsequent tuning consumes them as development. Integration still needs organizer acceptance of eligibility and C. |
| **E4-1 — broaden only known failure strata** | Classical/evaluation owners, after E0 diagnosis: research profile/config changes, separate hashes, strict-versus-broadened proposal pools, complete references and connectivity/duplicate controls. Possible levers: physical vesselness scales, local threshold/trace-gap handling, crop/short-stub treatment. Production detector remains unchanged. | Require a stratum-specific gain in unfiltered proposal recall with bounded extra false/duplicate/indirect/vein connections and C. Global loosening, merging independent ostia, or overwhelming the filter fails. Freeze a new pool before retraining; old candidate labels/patches cannot be attached by row index. Production adoption requires explicit integration approval. |

## 9. Recommendation-to-deliverable map

This map covers the PDF's implementation and phased recommendations, including
those deferred or corrected rather than implemented literally.

| Recommendation | Deliverable/owner and decision point |
| --- | --- |
| Preserve tracing/vesselness; learn local acceptance instead of named branch counts | Strict E0-1 baseline, candidate-only E1/E2, geometry rules in §5.5; integration owner keeps production unchanged. |
| Generate dozens/hundreds of diverse complete synthetic cases, balance hard negatives | Corpus contract §5.2 and Gate D; case/seed grouping, family/parameter stress tests, complete-reference manifests, original-distribution evaluation. Generator count alone is not diversity. |
| Add local contrast, patch statistics, wall curvature | Exact three-name extractor contract and Gate P; E1-2 base/extended ablation. |
| Boosted trees/random forest rather than only logistic | Optional sklearn training and NumPy/JSON export in §5.3; E1-1 includes logistic control and export parity. No mandatory XGBoost/LightGBM runtime. |
| Tiny axial/coronal/sagittal CNN, alternate per-view towers | 12-channel Gate P/N and E2-1; tower variation only after the smaller baseline. |
| Synthetic + lower-weight AI examples, smoothing, early stopping | E1-1/E2-2 separate source ablations; immutable labels and grouped validation. Stop/selection criteria never inspect final test labels. |
| HU window, noise, blur, translations and right-angle rotations | Extractor sidecar and coherent physical augmentations in §5.1/5.4; unsupported cached-plane transforms disabled. |
| Optional offset/radius/direction heads and weighted multitask loss | Deferred expert-geometry adapter in §6; synthetic-only machinery tests are not real localization evidence. |
| CNN score plus features, thresholding, proposal-only visualization | Separate-score E3-1 or explicitly versioned group-OOF stack; diagnostic artifacts preserve rejected proposals, submission schema stays unchanged. |
| Freeze/export, optimize/quantize, CPU batching, under-10-second overhead | Tree JSON or optional patch ONNX, Gate T/N/C; quantization ablation and full Windows/offline timing, no borrowed paper latency. |
| Calibrate scores and report discovery/localization/instance quality | Three estimands, reliability protocol, one-to-one/tolerance sensitivity, geometry denominators and uncertainty in §7. Scoring weights in PDF need organizer confirmation. |
| Broaden weak classical strata and fix missed branches | E0-1 → E4-1 before relying on candidate filtering; separately hashed profiles and new fingerprints, explicit integration gate. |
| Use the five expert cases to tune, then report | Choose development or frozen audit before labels; E3-2 records prior image exposure and never calls tuned cases independent. |
| Pursue segmentation/DRL/graphs and external datasets later | Data/permissions admission in §4 and model-specific adapter gates in §6. No patient download, labelling or dependency installation performed by this document. |

## 10. Phased handoff, organizer questions, and completion checklist

**Before expert references:** corpus/evaluation owners run E0-1 and identify
proposal ceilings; tabular/extractor owners deliver Gate P/D/T and E1-1/E1-2.
Only then spend training/export effort on E2 if false-positive failures warrant
it. Request external-data terms and label ontology in parallel, without assuming
access. Keep no-model execution and geometry rules intact.

**When references arrive:** establish completeness and prior exposure before
choosing E3-2 versus development use. Freeze pipeline, data and threshold hashes
for an audit. If tuning is necessary, mark cases consumed and ask for new final
cases. Do not “reset” provenance because expert labels are new.

**Before submission:** integration owner verifies all necessary contracts and
test artifacts, runs E3/C offline on target-like Windows, and records the chosen
configuration/fallback. Unpassed gates keep learned modules optional/research-only.
E4 is a new proposal experiment requiring re-extraction and explicit adoption.
Training can use separate hardware; no GPU/network is allowed at inference.

### Concrete organizer questions

1. Are references **complete for every eligible direct daughter** in the supplied
   crop, including accessory renal, lumbar, inferior mesenteric and other unnamed
   branches? Can they provide an eligibility/unknown-region mask and reasons for
   excluded structures?
2. The minimum origin size is confirmed as 2 mm diameter. How is
   it measured? The 5 mm visible-length rule is confirmed. How are partial-volume
   gaps or branches abutting a crop boundary handled?
3. Does the parent mask represent lumen, wall, or thrombus-inclusive outer aorta?
   Should an apparent origin separated by mural thrombus or plaque be considered
   connected? How should errors in the supplied mask be handled?
4. Common trunks count once; a second aortic opening of a returning vessel
   counts separately. How does the evaluator implement the stated allowance
   for one or two overlapping openings and for some vein predictions?
5. How are ostium center, seed distance, lumen radius and proximal direction
   defined? What physical coordinate convention, tolerances, direction sign,
   and reference-centerline support are supplied? Are seed/radius/direction
   scored independently of discovery matching?
6. Presentation is now reported as 5%, with discovery/count emphasized over
   geometric measurements. What is the revised complete score formula and
   official ostium matching tolerance? How are duplicates, extra/unknown
   structures, negative cases and absent predictions penalized?
7. Which reference cases overlap the 25 already inspected or pseudo-trained
   cases? Can untouched patients be reserved, and are references intended for
   development or a single frozen evaluation? Who annotated/adjudicated them?
8. Are external images, pretrained weights and derived annotations permitted?
   Will the organizers approve specific AVT/AortaSeg24 terms for this task?
   Is any additional expert annotation authorized and available?
9. What exact Windows version, CPU architecture/model, thread accounting,
   memory headroom and timeout rule apply? Does the approximately 60-second
   budget include startup, image loading and output writing?
10. Must all dependencies/models be provided as offline wheels/files, and may
    optional CPU ONNX be included? Are confidence/proposal-only fields allowed
    in submission JSON, or must they stay in separate diagnostics?

### Checks performed for the original specification

At the original documentation-only revision, no E0–E4 experiment, neural
training, target-device benchmark, or clinical evaluation had been run.
These historical contract checks do not validate the subsequently integrated
modules; their current verification is recorded in the final status below:

```bash
.venv313/bin/ruff check detector.py learning.py synthetic.py synthetic_reviews.py evaluate.py compare_e2e.py score_references.py
.venv313/bin/mypy detector.py learning.py synthetic.py synthetic_reviews.py evaluate.py compare_e2e.py score_references.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv313/bin/python -m pytest -q tests/test_evaluate.py tests/test_learning.py tests/test_synthetic_reviews.py tests/test_model_artifacts.py tests/test_training_recall.py
```

Ruff passed; mypy reported no issues in seven source files; targeted tests were
**25 passed in 4.22 seconds**. No Python source/tests were edited. Markdown has
no Ruff/mypy target. Separate document checks passed for 34 defined/used primary
links, eight ordered experiment rows, seven consistent tables, balanced code
fences, the required final sentence, and ten recorded source/data hashes.
Web retrieval verified the cited evidence, not uninterrupted availability of
every legacy download endpoint; Rotterdam download operation remains unverified.

### Reproducible audit provenance

These are hashes of the source/data schema inspected, **not experiment
results**. Concurrent work on main can change them; every actual experiment
must recapture its own revision, dirty-file state, source/config, data,
generator, extractor, split, model and preprocessing hashes.

```text
17495723df18c79e4302edc19bd949bb16f615b902403a88c68d6e74fa200063  detector.py
d7acdaa6f78ea9f3b6703d0b457880de37804ee0fc06ff6aeb4c9bfe7fe82c1e  learning.py
923e9d4aefdcf57150aa288d526e03eedd3b4c350e36c3eda2f7204bbe44bc41  synthetic.py
9189ac56e6def4b4090e6cee3cf768cbebd1af3417994524c6dfd78a8751074d  synthetic_reviews.py
8436e03091ab2b794b549a37ea11580dd70df84de5e49cd787ab4b191dd57ee6  evaluate.py
bcb407443fdfff39d190a6f32c60bab88daab26863b763c7865a594d65177edd  compare_e2e.py
483d216d6a3a9993a0ca8f6c2d614f7a85702e51ee4894437369e72487b22e38  score_references.py
9fbc1653eb0c34200855521e2c5039041983a05d30fc7d884d55f11c5769bf94  run.py
319138ab8e6778f4378d5938901e934d5997dc303adc01f54e3ed7be901a7b5f  labels/reviews.json
8709b34e8ade6eb55aba0e222f06bb707076dd08b95cd7d046f541ac9d611989  labels/split.json
```

### Primary links

[zheng]: https://comaniciu.net/Papers/TranscatheterAorticValveImplantation_TMI12.pdf
[tahoces]: https://minerva.usc.gal/rest/api/core/bitstreams/3378c44a-0108-45db-b3c4-5f90ca48860a/content
[han]: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0156837
[schaap]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3843509/
[drl-coronary]: https://www.mdpi.com/1424-8220/21/18/6187
[drl-tbad]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12956048/
[cobra]: https://arxiv.org/abs/2207.10446
[cacu]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10625464/
[asm]: https://image.diku.dk/marleen/papers/deBruijne_SPIE03.pdf
[cldice-paper]: https://arxiv.org/abs/2003.07311
[cldice-code]: https://github.com/jocpae/clDice
[topology-uncertainty]: https://proceedings.neurips.cc/paper_files/paper/2023/file/19ded4cfc36a7feb7fce975393d378fd-Paper-Conference.pdf
[retinal]: https://pubmed.ncbi.nlm.nih.gov/41336308/
[pasc]: https://arxiv.org/abs/2507.04008
[gatan]: https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2026.1812113/full
[avt-paper]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8760499/
[avt-data]: https://figshare.com/articles/dataset/Aortic_Vessel_Tree_AVT_CTA_Datasets_and_Segmentations/14806362
[avt-api]: https://api.figshare.com/v2/articles/14806362
[aortaseg-paper]: https://arxiv.org/html/2502.05330
[aortaseg-data]: https://aortaseg24.grand-challenge.org/dataset/
[aortaseg-access]: https://aortaseg24.grand-challenge.org/dataset-access-information/
[totalseg-data]: https://zenodo.org/records/10047292
[totalseg-code]: https://github.com/wasserth/TotalSegmentator
[vessel12]: https://vessel12.grand-challenge.org/Details/
[vessel12-rules]: https://vessel12.grand-challenge.org/Details/#rules
[rotterdam]: https://coronary.bigr.nl/centerlines/
[lits]: https://doi.org/10.1016/j.media.2022.102680
[aortaexplorer-paper]: https://pmc.ncbi.nlm.nih.gov/articles/PMC13132940/
[aortaexplorer-code]: https://github.com/RasmusRPaulsen/AortaExplorer
[breakage-review]: https://pmc.ncbi.nlm.nih.gov/articles/PMC13467413/
[ort-api]: https://onnxruntime.ai/docs/api/python/api_summary.html
[ort-threads]: https://onnxruntime.ai/docs/performance/tune-performance/threading.html
[ort-quant]: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
[calibration]: https://scikit-learn.org/stable/modules/calibration.html

## 11. Final integration status

The executable entry points and failure semantics are documented in
[README](../../README.md#optional-ml-research). `research_run.py` defaults to
scores-only, preserves all proposals and the standard JSON fields, and writes
scores/model hashes to sidecars. Explicit filtering uses the frozen threshold.
If scoring fails, all candidates are marked unscored with the error; scores-only
still writes the unfiltered prediction, while filter mode withholds its filtered
output. Both return failure status. No zeros or silent candidate deletion
substitute for unsupported physical features.

New tree training binds source-hashed candidate records to detector/extractor/
feature sources and the physical preprocessing contract. CNN/blend inference
checks ordered candidate identities, model/sidecar hashes, extraction contracts
and compatible splits. Original archived models remain immutable and use their
archived source; changing a source hash in a sidecar is not a migration.

The current detector hash is
`9f96f3eb3c06f5e06d5b7db79b199be70e742130fc3885243d25d066adc1c92e`.
It preserves the separately confirmed strict 2 mm origin-diameter rule,
0.7 mm seed-radius setting, existing review override of zero, common-trunk
logic and CT warnings. The prior cohort/CNN detector hash is
`17495723df18c79e4302edc19bd949bb16f615b902403a88c68d6e74fa200063`.
The current replay uses the same already-exposed synthetic seed groups with
fresh source/config exports. Neither archive nor replay is official 2 mm
eligibility validation. Their complete analytic references are retained without
retrospective relabelling.

### Experiment status and evidence

| Experiment | Status | Evidence and remaining gate |
| --- | --- | --- |
| E0-1 | DONE for analytic synthetic research; real accuracy BLOCKED | Current-source unfiltered strict/pool counts, per-family misses and hashed configurations are in the linked results/summary. Existing synthetic failure/regression tests remain unchanged. Complete expert references are required for real proposal recall. |
| E1-1 | DONE synthetic tree/logistic experiments; promotion GATED | Four frozen current-source tree models, train-only weights and validation/test reports. Validation-selected base-feature tree removed four pool FPs on the replayed test with no extra misses (63/0/7); E1 is `go_research_only`. These are source-specific research counts. No pseudo-label tree ablation or clinical calibration is claimed. |
| E1-2 | DONE physical extractor and base/extended ablations; added-feature benefit GATED | Exact three-extra contract, finite physical patch tests and source-bound tree inference. Current base and extended models all scored 63/0/7 on test; extended gradient boosting retained two validation FPs versus one for base, failing its validation gate. No added benefit demonstrated; grouped feature importance and external generalization remain unrun. |
| E2-1 | DONE current and historical retrospective CNN; REJECTED for promotion | Current-source CNN reproduced 59/2/11 versus tree 63/0/7, losing four pool TPs. [Historical CNN results](../../labels/research/patch-cnn-retrospective-v1/RESULTS.md) remain archived separately. Current execution followed unchanged E1 `go_research_only`; native/ONNX parity maximum error 1.19e-7 does not establish model quality. |
| E2-2 | DONE historical mixed ablation; REJECTED for promotion | Historical mixed CNN 58/2/12; 45 exact historical pseudo patches retained with original provenance; mismatches excluded explicitly. All 25 real-case exposures remain visible. No new labelling, broader pseudo sweep, or proof of real transfer. |
| E3-1 | DONE frozen blend experiments and inference integration; REJECTED for promotion | Current blend reproduced 62/0/8, selected CNN weight zero and a stricter tree threshold, losing one pool TP. Both historical blends also failed. No CNN ensemble benefit or activation; group-OOF CNN-in-tree stacking remains unimplemented. |
| E3-2 | TOOLING DONE; expert experiment BLOCKED | `research_validation.py --manifest ... --output ...` ingests references, runs one-to-one/tolerance/geometry/bootstrap comparisons and rejects incomplete/pseudo, contaminated or unknown-history promotion bases. Organizer references have not been supplied; overlapping organizer cases are exposed-case audits. |
| E4-1 | DONE comparison of existing strict versus review union; new tuning GATED | Source/config-hashed unfiltered pools expose the proposal ceiling. No new classical broadening or production tracing changes were authorized. Upstream misses cannot be recovered by a classifier. |

### Every report action: implemented or explicitly gated

| Report action | Status and exact implementation/evidence |
| --- | --- |
| Preserve classical proposals, variable instance count and physical geometry | DONE: unchanged `run.py`/`detector.py`; `research_run.py`, proposal-preservation/error tests. |
| Generate a diverse complete synthetic cohort; sample/balance negatives | DONE: `research_corpus.py` + `research_reproduce.py`; 15 seed groups × 14 families, real negative counts and source weights in reports. No invented negatives; seed reuse is exposed. |
| Add local contrast, vesselness patch statistics and wall curvature | DONE: `candidate_patches.py`; physical/reorientation/boundary/curvature tests; extended model contract rejects unavailable support. |
| Boosted-tree/random-forest models with logistic/keep-all controls | DONE: `train_trees.py`/`tabular_learning.py`, four fixed experiments and native/NumPy probability parity. |
| Small three-view patch model | DONE: `patch_learning.py`/`patch_inference.py`, 34,465-parameter model and CPU ONNX export; alternate towers/3D architecture GATED on a demonstrated benefit. |
| Downweight noisy real labels, smoothing and early selection | DONE historical mixed CNN and train-only source balancing tests. Additional pseudo weights, expert fine-tuning and tree-mixing ablations GATED; prior labels immutable. |
| HU/preprocessing, coherent augmentation, noise and blur | DONE bounded physical extractor and tested allowed transforms. Cached planes cannot support arbitrary physically coherent translations; unsupported transforms remain disabled, not claimed implemented. |
| Optional origin/radius/direction regression and multitask loss | GATED on complete expert geometry, masks and a valid adapter. No trained localization head or real localization improvement is claimed. |
| CNN-plus-tabular inference and thresholded proposal scoring | DONE explicit frozen probability blend, sidecar scores and preserved JSON. Group-OOF stacking and new Explorer score visualization are GATED; CLI does not modify the UI. |
| Export, CPU batching, quantization and under-10-second overhead | DONE NumPy tree/ONNX export, batching/parity, bounded optional Windows CI and portable psutil command. Quantization, native Windows/offline hardware measurements and guaranteed overhead are GATED. |
| Calibration, branch matching, geometry, uncertainty and failure strata | DONE tree calibration support, one-to-one/tolerance sensitivity, full-reference conditional/end-to-end separation, paired group bootstrap. Clinically calibrated probabilities and official weighted challenge scoring are BLOCKED on expert/organizer evidence. |
| Broaden weak strata | DONE existing review-union comparison only; new E4 proposal changes GATED and production untouched. |
| Use approximately five organizer cases | BLOCKED until supplied; choose development versus frozen audit before use, preserve overlap with all 25 prior images. |
| Heavy 3D segmentation, DRL, graph/topology models or external pretraining | GATED on §4 permissions, model-specific dense/trajectory/graph annotations, more independent cases and resource evidence. These methods are documented prerequisites, not implementations. |

### Verification scope

See the [result bundle](../../labels/research/current-source-v1/RESULTS.md) for exact
test counts, source hashes, wheel verification, resource observations and
remaining limitations. Ruff and mypy cover all 27 Python modules, including
optional research modules; optional Torch/scikit-learn/ONNX packages are actually
installed for their checks. A separate base-only environment tests the absence
of those dependencies. Network-denied Linux tests include a socket-denial
assertion and real training/export/inference, rather than an offline claim based
only on import inspection.

`research_resources.py` measures sampled summed process-tree RSS and wall time
with inherited CPU affinity and numerical thread caps. Sampling limitations,
wrapper overhead and unverified GPU/network status are explicit. Linux evidence
is labelled Linux; verified Windows wheels and a manual CI job do not constitute
organizer-hardware measurements. The parent's prior base Windows CI evidence
does not establish native optional-CNN execution.

## Next experiment

Use complete expert references to audit frozen choices and their proposal
misses, preserving case/group/image exposure. If source-matched synthetic gates
reject a classifier, retain strict production and investigate the unfiltered
failure strata before considering more model complexity. No clinical claim can
be made from synthetic accuracy or agreement with candidate-derived labels.

If this were my submission, here is the next experiment I would run and why.
