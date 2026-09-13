# Branchseed — complete presenter briefing

**Team reference for the five-minute talk, live Explorer demo and judge Q&A.**

The submitted choice is **score-before-merge fusion**: two geometric proposal
passes, a bundled logistic filter, and strict-first merging.
This document explains the product, algorithms, experiments, evidence,
limitations and operational details. Read section 1 before rehearsal; use the
remaining sections for preparation and questions.

## 1. The one-minute memory sheet

**Pitch:** “From CT to a map you can inspect: discover aortic branch origins,
verify predictions in 3D and linked CT, and export measurements offline.”

**Problem:** find visible vessels that directly leave a supplied parent aorta.
The number is unknown. Each opening needs its own physical-coordinate instance.

**Our contribution:** a CPU discovery pipeline plus a linked 3D/CT Explorer,
physical measurements, JSON export and a reproducible evaluation framework.

**Rules to remember:** minimum **2 mm origin diameter**; at least **5 mm**
visible continuation; one aortic opening per daughter; a common trunk counts
once; crop caps and daughter-of-daughter vessels are excluded.

**Release:** 1 mm grid; native contrast scale 0.9; strict + review proposals;
bundled 13-feature logistic filter at 0.15; strict-first merging within 3 mm;
four threads and offline inference.

**Numbers to remember:**

- Five reused cases / 19 reference targets / local 3 mm matching:
  **14 TP, 4 FP, 5 FN; precision 0.778; recall 0.737; F1 0.757; count MAE 1.8.**
- Twenty-four synthetic topology cases:
  **47 TP, 11 FP, 2 FN; F1 0.8785.** Includes four negative-control
  detections. Synthetic regression, not patient accuracy.
- All 25 supplied scans completed: **11.60 s mean, 52.67 s maximum,
  1499 MiB maximum sampled process-tree RSS**, under four-core Linux affinity.
- Hidden-test accuracy, clinical validation and native organizer-Windows
  timing are **not available**.

**Demo:** select an origin → inspect 3D → inspect CT → show measurements →
export JSON. Use the wall map/interior tour if time allows.

**Closing:** “We connect automatic branch predictions to the evidence a
reviewer needs to inspect them, with physical measurements and local execution.”

## 2. Problem, anatomy and scope

The aorta is the large parent vessel. An **ostium** is an opening where a
daughter vessel leaves its wall. A **lumen** is the vessel's interior. The
challenge provides a binary mask of the parent lumen; our task starts there.

CTA contrast can make blood bright, but brightness alone does not establish
arterial connectivity. A vein, calcium, a vertebral edge or a partial-volume
artifact can also look bright. The relevant question is whether a supported
vessel path actually leaves the supplied parent.

The task is instance detection. It is not a fixed anatomical checklist:
scan coverage varies, and a case may have zero, a few or many eligible
daughters. Guessing a standard vessel count or naming unseen anatomy would
violate the task.

Toralis Labs' wider context is vascular-surgery software and anatomical
measurement. Branchseed is a challenge prototype that could support inspection
of vessel origins. We have not measured clinical benefit, time saved by
clinicians or effects on surgical decisions.

### Scope boundaries

| Branchseed does | Outside the core task |
|---|---|
| Accept CT plus supplied parent mask | Automatically segment the parent aorta |
| Predict direct, visible origin instances | Guess branches outside image coverage |
| Measure proximal geometry | Reconstruct the complete distal arterial tree |
| Provide interactive evidence and JSON | Assign definitive anatomical vessel names |
| Support local review | Diagnose disease or prescribe treatment |

The Explorer overlaps with ITK-SNAP's visual-inspection role. Its task-specific
contribution is coupling automatic daughter discovery, per-instance
measurements and challenge export to an immediately inspectable workflow.
Do not describe it as a complete replacement for a medical image editor.

## 3. What counts as a daughter

These rules combine the brief with the judge's clarifications relayed by the
team. They are eligibility semantics, not proof that our detector always
implements them perfectly.

| Situation | Intended interpretation | Implementation / presentation caveat |
|---|---|---|
| One lumen directly leaves the parent | One daughter | Must have supported proximal continuation |
| Two separate nearby openings | Two daughters | Spatial proximity alone must not collapse them |
| A common trunk splits downstream | One direct daughter | Shared proximal paths are used to remove duplicates |
| A daughter gives rise to another vessel | The secondary vessel is not another direct daughter | The trace stops at an estimated junction |
| Two overlapping ostia, like a Venn diagram | Technically one opening; the judge said evaluation may allow one or two | We do not have the official evaluator implementing that allowance |
| A visible vessel leaves and re-enters the aorta | Two separate openings may count twice | We score openings within the supplied crop, not the unseen full vessel |
| A flat superior/inferior crop surface | Not an ostium | Reject artificial cap contacts |
| Vessel outside coverage or without 5 mm visible lumen | Do not invent a daughter | Missing evidence remains missing |
| Origin below 2 mm diameter | Ineligible under the confirmed size rule | CT size measurement has partial-volume uncertainty |
| Nearby vein | Connectivity still matters; judge may allow some veins | This is not a validated artery-versus-vein classifier |
| Terminal iliac division | Outside the core task; optional separate evaluation | Do not sell it as solved by the core origin detector |

Thoracic coverage is handled geometrically rather than by a hard-coded list
of named abdominal vessels. Confirm any final organizer-specific coverage
exceptions rather than inferring them from a case ID.

An early bifurcation before 5 mm is an unresolved semantic corner: one trunk
can count once, yet stopping before its junction may leave no valid 5 mm
proximal seed. The implementation rejects too-short proximal segments. If
asked how an official reference resolves that situation, say we have not
received an authoritative placement convention.

## 4. Input/output and physical geometry

### The exact CLI

```bash
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
```

`--diagnostics diagnostics.json` adds inspectable trace/rejection information
without adding research-only keys to the challenge prediction.

### Fields a judge may ask about

| Field | Meaning |
|---|---|
| `case_id` | Input case identifier |
| `parent` | Root object containing `{"instance_id": "aorta"}` |
| `daughters` | Variable-length list; empty is valid if none are accepted |
| `instance_id` | Unique daughter ID, e.g. `branch_001` |
| `parent_instance_id` | Per-daughter link to the supplied `"aorta"` instance |
| `ostium_xyz_mm` | Estimated opening centre on the aortic wall |
| `seed_xyz_mm` | Point 5 mm along the accepted proximal path |
| `radius_mm` | Estimated local lumen radius at the seed |
| `direction_xyz` | Unit vector from ostium toward the seed |

The exported parent link belongs on each daughter as required by the
challenge contract. Do not add UI review votes or classifier features to the
submission JSON.

**Coordinates are physical millimetres**, in the SimpleITK input frame.
Array access often uses z-y-x; SimpleITK point indices use x-y-z. Spacing,
origin and the direction matrix must be respected when converting. Browser
camera coordinates are a display transform, never the submission coordinate
system. “LPS” describes the usual medical physical axes; do not manually flip
values because a rendered camera view looks reversed.

The 5 mm rule uses **path arc length**. The exported direction uses the
normalized **ostium-to-seed chord**, so it need not equal the tangent at the
seed on a curved path. Radius at the seed is distinct from diameter near the
origin. Confusing those two measurements changes eligibility incorrectly.

## 5. The production algorithm, step by step

Implementation: [pipeline.py](pipeline.py) orchestrates [detector.py](detector.py)
and [learning.py](learning.py). The geometric stages below run for both strict
and broader review proposals; section 5.7 explains the final scoring and merge.

### 5.1 Validate and normalize

Check CT/mask dimensions and physical geometry, extract the parent region,
crop around it and resample into a 1 mm working grid. Intensity and mask
resampling have different roles: preserve interpretable CT values while
retaining a discrete parent label.

Estimate blood intensity inside the parent and background around it.
Scan-relative contrast reduces reliance on one fixed global brightness
threshold. The native-contrast setting retains sensitivity to original CT
support; both fusion passes use scale 0.9. Poor contrast cannot be repaired simply
by normalization.

**Why:** physically defined distances and vessel scales must remain comparable
across different voxel spacings.

**Limitation:** interpolation does not create missing anatomical detail.
Coarse scans still carry coarse information.

### 5.2 Enhance lumen evidence

Apply Sato tubularity at physical scales **0.8, 1.5 and 2.5 mm**, combining
tubular response with contrast support. Distance transforms supply a local
radius estimate and distances from the parent. Skeleton/junction analysis
supports proximal-path termination.

**Why:** elongated bright structures are better vessel proposals than isolated
bright voxels.

**Limitation:** tubularity is a shape cue, not an artery classifier. Adjacent
vessels and bright artifacts can satisfy parts of the same evidence.

### 5.3 Propose wall contacts

Search a shell approximately **1.5–3.5 mm outside the parent**. Label
supported contact components, reject weak or inappropriate contacts and rank
root candidates by support, local radius and tubularity. Separated roots can
be retained within one merged contact component.

**Why:** searching near the supplied parent constrains the problem to direct
origins and reduces work.

**Limitation:** a missing wall-contact proposal cannot be recovered by a
classifier applied after proposal generation. Fusion adds broader automatic
review proposals, then filters both passes before merging them.

### 5.4 Trace a supported proximal path

A local minimum-cost path solver follows supported lumen voxels. The base
cost is:

```text
cost = 1 / (0.5 + local_radius) + 0.7 * (1 - vesselness)
```

Higher-radius, more tubular evidence is cheaper; unsupported voxels and the
parent interior are excluded from the outside path search. Endpoint quality
balances cumulative cost against outward progress. A wall-origin check must
then connect the traced candidate to the parent.

**Why:** a path provides evidence of continuation, unlike a single bright
contact point.

**Limitation:** strict endpoint and connection checks can reject a curved
or wall-parallel vessel even when some true lumen is visible. A strict
endpoint currently needs substantial distance from the wall, not merely
5 mm of arc length along it.

### 5.5 Stop, measure and apply eligibility

Limit the proximal trace to **10 mm**, stop at an estimated downstream
junction and reject it if fewer than **5 mm** remain. Interpolate the seed
at 5 mm arc length. Reject near-wall paths with less than **3.5 mm** net
ostium-to-seed displacement. Normalize the direction chord.

Estimate seed radius with a local cross-section when resolvable, otherwise
use the distance-based estimate. The strict seed-radius floor is **0.7 mm**;
very wide regions above **8 mm radius** are rejected. These are detector
heuristics, not the organizer's 2 mm diameter definition.

The origin-size gate samples an early proximal cross-section, approximately
2 mm along the path, using a local contrast level. It estimates a diameter
and adds a native-spacing allowance. Reject when the estimated upper diameter
is still below the **2 mm** cutoff. An unresolved estimate is flagged rather
than automatically discarded.

**Critical caveat:** this is an approximate CT size policy with a one-voxel
allowance. It is not a calibrated statistical confidence bound or proof that
every accepted origin is at least 2 mm.

### 5.6 Resolve instances and export

Compare nearby origins and proximal paths to remove same-opening and
common-trunk duplicates. Keep separated eligible wall origins. Sort the
accepted instances deterministically, assign IDs and export the challenge
fields.

Diagnostics retain rejection reasons, path evidence, warnings and an
uncalibrated heuristic evidence score. **A score of 0.8 does not mean an
80% probability of a true branch.**

### 5.7 Score before merging

The production workflow runs strict and broader review proposals at native
contrast scale 0.9. Each pass keeps the geometric tracing and measurement
framework, while the review policy permits additional candidates. A bundled
13-feature logistic classifier scores each pass independently; candidates
below 0.15 are removed. The model is hash-checked and loaded locally.

Strict survivors are retained first. A review survivor is added only when its
origin is at least 3 mm from every retained origin. Instances receive stable
IDs before export. Filtering before this merge avoids letting a low-scoring
review representation displace a surviving strict candidate.

The 3 mm merge radius is separate from the 3 mm evaluation tolerance and the
2 mm origin-diameter eligibility rule. Distance merging may collapse genuinely
close openings; broader proposals also admit false positives. Neither a
logistic score nor the geometric evidence score is a calibrated clinical
probability. This workflow requires no manual review at inference.

## 6. Why the selected default is fusion

| Configuration / evaluation | TP / FP / FN | Precision | Recall | F1 | Count MAE | Status |
|---|---:|---:|---:|---:|---:|---|
| Selected fusion | 14 / 4 / 5 | 0.778 | 0.737 | 0.757 | 1.8 | Submitted default |
| Fixed strict | 8 / 3 / 11 | 0.727 | 0.421 | 0.533 | 2.0 | Historical baseline |
| Leave-one-case-out selection composite | 8 / 9 / 11 | 0.471 | 0.421 | 0.444 | 3.2 | Selection-procedure estimate |
| Retrospective RF winner | 9 / 1 / 10 | 0.900 | 0.474 | 0.621 | 1.8 | Development only |
| Guarded 1 mm connection recovery | 9 / 3 / 10 | 0.750 | 0.474 | 0.581 | 2.2 | Opt-in experiment |

All rows use the same five reused reference cases and local 3 mm matching.
They are not independently validated deployments.

The team selected fusion for its higher measured reference F1 and lower
count error. The standalone package reproduces all 25 full-checkout outputs.
That verifies the packaging, not generalization: the synthetic comparison
finds eleven fusion false positives versus zero for strict.

The later guarded recovery follows bounded parent-connected support around
curved connections while preserving strict detections. It gains one local
match, passes the 24-case topology comparison and showed no lost baseline
matches or added false positives across 80 procedural synthetic comparisons.
However, count MAE worsens. A newly correct branch can be found in a case
where the detector already overcounts, increasing count error even while
F1 improves.

The team initially accepted strict, then explicitly selected fusion after
reviewing its higher development score. That does
**not** prove fusion is optimal or that a score above 0.7 is overfitting.
The problem is selecting repeatedly on reused data without independent
confirmation, not crossing a particular numeric threshold.

Experimental CLI from the current Git checkout, **not the submitted default**:

```bash
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz \
  --output experimental.json --diagnostics experimental-diagnostics.json \
  --threads 4 --pipeline strict --recover-connected-origins
```

## 7. Shipped ML and historical research

| Approach | What it adds | Strength | Drawback / disposition |
|---|---|---|---|
| Logistic candidate filter | A small weighted classifier over 13 features | Shipped in fusion; fast and inspectable | Cannot recover absent proposals; label provenance matters |
| Random forest | Nonlinear feature interactions | Strong retrospective development score | Reused targets and relaxed proposals; not promoted |
| Gradient-boosted trees | Sequential nonlinear candidate decisions | Flexible with compact feature inputs | Threshold/model selection can overfit small case sets |
| Physical CT patches | Candidate-centred intensity/geometry context | Preserves physical scale across cases | Extraction cost and candidate coverage constrain utility |
| Small patch CNN | A 34,465-parameter local visual classifier | Can learn texture/shape cues beyond scalar features | Some experiments removed true branches; not promoted |
| Tree/CNN or score blends | Combine different evidence streams | Potential complementary errors | Extra tunable weights, dependencies and latency |
| Finer/coarser grids | Change resolution and resource tradeoffs | May recover specific small or curved paths | Some challengers produced major topology false positives |
| Guarded connected recovery | Recover selected curved wall connections | Modest local F1 gain with strict matches retained | Worse local count error; opt-in |
| Heavy 3D segmentation / reinforcement learning | Broader learned discovery | Future research possibility | Deferred; sparse annotations and laptop budget do not justify it now |

Features include physical radius/length, tubularity, contrast relative to blood,
bone distance, parent angle, longitudinal position and connection support.
The shipped feature contract is in `learning.py`; model provenance is in
`models/production-v1/`. Most research tooling was removed in the repository
cleanup; the experiments remain in Git history, not in the judge download.

Research used **210 procedural cases** split into **140 train / 42 validation /
28 test** cases. These cases were inspected, so the partition name “test”
does not establish a sealed independent experiment. Physical patches and
hard negatives support development; they are not substitutes for diverse
expert-labelled patients.

Model/source hashes, lineage checks and scores-only runners help prevent
silently loading incompatible weights or filtering production without
recording it. Runtime stays offline after optional dependencies and artifacts
have been prepared.

**The central lesson:** proposal recall places an upper limit on every
downstream classifier. The next useful improvement may be better candidates,
not a larger classifier.

## 8. The three additional papers

These are adaptations of ideas, not exact reproductions of the authors'
pipelines. Their scope and results are summarized below; the full write-up is in git history.

| Source | Idea adapted here | Result / limitation |
|---|---|---|
| [Danilov et al., 2016](https://www.mdpi.com/2079-3197/4/3/35) | Near-aorta border cleanup using outward support layers | Did not beat production in the recorded synthetic comparison |
| [Riffaud et al., 2022](https://link.springer.com/article/10.1007/s11517-022-02603-2) | Origin-anchored PCA direction | Direction estimation does not solve missing origin proposals; not promoted |
| [Tahoces et al., online 2019 / 2020](https://link.springer.com/article/10.1007/s11517-019-02110-x) | Parent-connected contact growth | Broader support can also admit false contacts; not promoted |

Their input assumptions, target vessels and evaluation datasets differ from
this challenge. Never quote a paper's accuracy as Branchseed accuracy or imply
that including an idea guarantees a gain. We evaluated the adaptations and
kept them optional when results did not support replacement.

## 9. How evaluation works

### Reference provenance

The released package has **19 AI-assisted targets** across subjects 019–023.
The package says the annotations require review and may omit eligible
branches. The team separately confirmed that the judge approved them for
scoring. Preserve both facts.

These are not exhaustive clinical truth, and they are not unused patients:
the team had already inspected these development images. A prediction absent
from the package is counted as a local false positive, even if an expert
might later identify an unannotated eligible vessel.

### Matching and metrics

Our local scorer makes maximum-cardinality **one-to-one** ostium matches within
a distance tolerance. One prediction cannot claim two targets. Duplicate
predictions can therefore become false positives.

- **Precision = TP / (TP + FP):** fraction of predicted instances matched.
- **Recall = TP / (TP + FN):** fraction of reference targets matched.
- **F1 = 2 TP / (2 TP + FP + FN):** balances precision and recall.
- **Count MAE:** average absolute difference between predicted and reference
  daughter count per case. It does not check whether the right vessels were found.
- **Geometry errors:** measured only after matching, with missing reference
  measurements masked out.

Do not say “77.8% accuracy.” Fusion precision is 0.778; recall and F1 are
different, and “accuracy” would need a defined negative population.

### Fusion counts by case

| Case | Reference targets | Fusion predictions | Absolute count error |
|---|---:|---:|---:|
| subject019 | 3 | 2 | 1 |
| subject020 | 4 | 2 | 2 |
| subject021 | 3 | 4 | 1 |
| subject022 | 6 | 9 | 3 |
| subject023 | 3 | 1 | 2 |
| Total / mean | 19 | 18 | MAE 1.8 |

The near-correct total hides case-level errors: subject022 overcounts by
three while subjects020 and 023 each undercount by two. No case has an exact
count. Do not imply equal performance across all patients from the aggregate F1.

### Sensitivity to tolerance

| Local ostium tolerance | Fusion TP / FP / FN | Fusion F1 |
|---|---:|---:|
| 2 mm | 8 / 10 / 11 | 0.432 |
| 3 mm | 14 / 4 / 5 | 0.757 |
| 5 mm | 14 / 4 / 5 | 0.757 |

The organizer has not supplied the official evaluator. Local 3 mm matching
is our declared analysis setting, not a claim about the hidden scorer.
The broader tolerance did not recover the remaining unmatched references.

### Geometry: read the sample size

At 3 mm, across the fourteen fusion matches:

| Measurement | Mean error | Number evaluated |
|---|---:|---:|
| Ostium location | 1.658 mm | 14 |
| Seed location | 1.284 mm | 14 |
| Direction angle | 14.718° | 14 |
| Seed radius | 0.204 mm | **2** |

Only three of the 19 reference radii are known, and only two are matched by
fusion. A radius headline based on two points would overstate evidence.
Matched-error summaries also exclude the five missed targets.

### What leave-one-case-out did and did not fix

The frozen selector validated 280 variants; 180 met its clean eligibility
rules. For each held-out case, it selected using only the other four. The
five chosen configurations differed, and their pooled held-out F1 was 0.444.
This is evidence about a **selection procedure**, not a single model file.

Case-separated selection limits direct fold leakage. It cannot undo earlier
human inspection of the data or make five reused cases representative of
all scanners and anatomy. The RF all-five score remains retrospective.

### Synthetic versus patient evidence

The frozen topology cohort has 24 procedural cases and analytic references.
Fusion produced 47 TP / 11 FP / 2 FN, F1 0.8785, with four negative-control
detections. The strict baseline had 46 / 0 / 3, F1 0.9684. Seven fusion
false positives occur in a touching-vein/calcification case and four in a
negative control. These cases exercise known geometric failure modes. Real CT adds
contrast variation, disease, acquisition effects and anatomy outside the
generator's assumptions. Synthetic success does not resolve low real-case recall.

### Challenge weighting

The written brief lists branch discovery 45%, ostium localization 25%,
daughter quality 15%, compute 10% and reproducibility 5%. The team's later
judge discussion emphasized daughter counts and described presentation as
5%. Keep the written rubric and verbal clarification distinguishable.
Without the official evaluator, **do not calculate or claim an official
weighted score**.

## 10. Failure modes and how to discuss them

| Failure | Mechanism / symptom | Honest response and next action |
|---|---|---|
| Weak contrast | True lumen fails support/tubularity | Inspect CT; expand diverse labelled data before relaxing globally |
| Thin vessel / coarse slice | Partial-volume averaging removes a contact | Native evidence and size allowance help, but cannot create resolution |
| Curved wall connection | Strict origin connector rejects visible curved support | Guarded recovery is available; needs unused-case validation |
| Wall-parallel path | Enough arc length but insufficient outward clearance | Candidate/endpoint design is a recall bottleneck |
| Bone/calcium or nearby vein | Bright elongated structure imitates a branch | Connectivity and geometry reduce, but do not eliminate, false positives |
| Mask boundary error | Wrong wall position creates/misses contact | Parent segmentation is an input assumption; report mask quality |
| Common trunk / early bifurcation | Multiple roots or too-short pre-junction path | Shared-path resolution and junction stopping have corner cases |
| Broad or touching structures | Contact components merge | Separate roots help; increasingly permissive roots can overcount |
| Unlisted true branch | Local evaluation calls it FP | Request expert adjudication; do not relabel it to improve a score |
| Small reference sample | Case-specific changes look globally strong | Freeze decisions and evaluate previously unused patients |

Do not frame all false positives as missing annotations. Some are detector
errors; the current reference package cannot distinguish all of them.
Likewise, do not dismiss missed reference targets simply because labels are
AI-assisted.

## 11. Explorer behavior, reviews and exports

The browser provides a case library, parent surface, colored predicted
instances, direction markers, linked CT planes, physical measurements,
wall map and interior tour. A selection connects the same instance across
these views.

The surface and tour are navigation aids. The wall map uses an approximate
parent-coordinate representation. CT evidence is still needed to assess
whether a predicted path follows real lumen.

Human review votes are stored in that browser and exported separately.
Confirm/reject actions do not retrain the model or silently rewrite the
raw challenge prediction. The viewer's geometric evidence score is heuristic
and uncalibrated.

The product screenshots and 40-second showcase come from a preserved earlier
recording. Their on-screen branch counts can differ from today's fusion
outputs. Explain them as feature demonstrations; use current JSON and frozen
receipts for accuracy claims. For example, the recording's subject001 count
is not the final batch's subject001 count.

## 12. Five-minute talk and live demo

### Speaker ownership

| Time | Owner | Main job | Handoff |
|---|---|---|---|
| 0:00–1:15 | Speaker 1 | Explain direct origins and one-opening semantics | “Now we turn that definition into a physical-space detector.” |
| 1:15–2:30 | Speaker 2 | Explain support, tracing, measurements and fusion | “Every prediction can then be inspected in the Explorer.” |
| 2:30–3:45 | Speaker 3 | Run the live 75-second Explorer sequence | “Those views help inspect a prediction; here is the measured evidence.” |
| 3:45–5:00 | Speaker 4 | Show results, runtime, limitations and next step | Close on inspectable evidence and local execution |

The eight-slide deck and script are in the
[current fusion presentation kit](https://github.com/Coder-Meet/battleoftheschool/releases/tag/branchseed-submission-fusion-2026-09-13).
There is no video playback in the live deck. The separate showcase goes to
Devpost/Drive.

### The 75-second Explorer sequence

1. **0–15 s:** open the prepared case, select a daughter and rotate the 3D
   parent. “The detector proposes a physical origin; we can inspect it.”
2. **15–35 s:** open CT evidence. Point to the wall, selected origin and
   proximal lumen across the linked planes. Explain what would make the
   candidate uncertain rather than calling every displayed point correct.
3. **35–50 s:** show the seed, radius and direction panel. Explain that
   5 mm is along the path and coordinates are in physical millimetres.
4. **50–65 s:** show the wall map; briefly enter the interior tour if it
   is already rehearsed and responsive.
5. **65–75 s:** export the prediction JSON and return to the deck.

Preload the first case. Pick a second case in rehearsal so you can show that
counts vary. Do not claim the first display load is fresh inference:
case analysis may be cached. Use the recorded batch receipt to discuss runtime.

### Fallback

If the browser or projector fails, stay with the deck and its static CT/
geometry evidence. Explain the CLI/output contract from the stored JSON.
Do not spend the talk reinstalling dependencies. The video remains a
separate optional upload asset, not a planned live-talk fallback.

## 13. Judge questions and concise answers

**What does F1 0.757 mean?**\
Fusion matches 14 of 19 local targets, with four unmatched predictions and
five misses. Remaining misses can arise through proposal/connection
limitations. Synthetic topology checks do not capture all real CT variation.
The next step is expert-reviewed misses and unused-case validation.

**Isn't 77.8% your accuracy?**\
It is precision: 14 of 18 predictions matched locally. Recall is 14 of 19,
or 73.7%; F1 is 0.757. The incomplete references further limit interpretation.

**Why show synthetic F1 0.8785?**\
It is the separately labelled synthetic topology F1. It tests known
geometric behaviors; it is not patient or clinical accuracy.

**Why submit fusion instead of strict or the random forest?**\
Fusion has the highest measured reference result of these candidates and
lower count error than strict. It also has more synthetic false positives.
The team selected that tradeoff; hidden-case superiority is not established.

**Why not use the recovery at 0.581?**\
It improved local matching while worsening count MAE. The judge emphasized
counts, so we kept it opt-in pending independent evaluation.

**Does deterministic mean you cannot overfit?**\
No. Hand-picked thresholds, topology rules and repeated parameter selection
can also overfit. We record exposure and freeze evaluation decisions.

**What is new here if you use existing libraries?**\
The task-specific integration: direct-origin policy, physical measurements,
candidate/evaluation tooling and a connected 3D/CT verification workflow.
Sato filtering and minimum-cost paths are established techniques, and we
credit their libraries and relevant research.

**Is the product an AI model?**\
The submitted detector combines classical image processing and geometry
with a small learned logistic candidate filter. It is not a large 3D neural
network. AI-assisted development is a separate fact we also disclose.

**Can it handle a new scan without clicking seed points?**\
The CLI accepts a new CT and compatible parent mask and runs the fixed
detector. Manual point placement is not required. Detection quality still
needs validation on new cases.

**Can it segment the aorta from raw CT?**\
No; the parent mask is supplied by the challenge.

**Can you tell artery from vein?**\
Not reliably as a validated classification task. The pipeline checks
parent connectivity, contrast and geometry. The judge's allowance for some
veins does not establish artery/vein discrimination.

**What happens at a common trunk?**\
We count its one aortic opening, compare proximal paths to avoid duplicates
and stop at the estimated downstream junction.

**Why use physical coordinates?**\
Voxel size and orientation vary. Millimetres preserve meaningful positions,
lengths and radii in the original input frame.

**Is radius accurate?**\
It is an estimate. Only two fusion matched references have measured radii;
we cannot make a broad accuracy claim from that sample.

**Does the 2 mm check guarantee eligibility?**\
No. It is an approximate early-proximal diameter measurement with a native
voxel allowance and unresolved-size warnings.

**What does the 0–1 evidence score mean?**\
It combines geometry/support heuristics. It is not a calibrated probability.

**Does clicking Confirm update the detector?**\
No. Browser review state is separate from the raw prediction and from any
later optional model-training workflow.

**Is it offline?**\
Inference and the built Explorer run locally after dependencies/data are
installed. Offline installation itself needs predownloaded compatible wheels;
we include Windows x64/Python 3.13 wheels in the technical bundle.

**Is it proven on the organizer's laptop?**\
Not yet. The final resource receipt is four-core Linux with 1499 MiB maximum
sampled RSS; native organizer-Windows timing is still required.

**Why not a large 3D network?**\
The task supplies little trusted annotation and has a CPU/offline budget.
A heavier network would add data and validation requirements without
established benefit here.

**Are all exported instances correct?**\
No such claim is supported. That is an output count across 25 completed
scans. Only the five-case reference comparison has local matching results.

**Why does the video show a different count?**\
It is a preserved earlier interface recording. The release JSON and frozen
fusion receipts are the current source for algorithm results.

**Did AI help build this?**\
Yes. Devin and any additional tools actually used by the team should be
credited. The methods, tests, limitations and artifact provenance are
available for inspection.

**Would you trust it clinically?**\
It is a research prototype, not clinically validated. Its current role is
exploring candidate detections and their evidence.

## 14. Laptop preparation and commands

Use [DEMO_GUIDE.md](DEMO_GUIDE.md) as the canonical operational reference.
Download the fusion judge application and current presentation before the
event. The judge ZIP contains inference source, model and offline wheels.
Its evidence ZIP contains predictions and static checks. Prepare the Explorer
separately from current main; it is not included in the minimal judge ZIP.

### Offline Windows x64 / Python 3.13

From the extracted **`branchseed-judge-fusion` directory**, with Python 3.13.3
x64 installed, use its **`wheelhouse` directory**:

```powershell
python --version
python -m venv .venv313
.\.venv313\Scripts\python.exe -m pip install --no-index --find-links wheelhouse -r requirements.txt
$env:OPENBLAS_NUM_THREADS="4"
$env:OMP_NUM_THREADS="4"
$env:MKL_NUM_THREADS="4"
$env:ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="4"
.\.venv313\Scripts\python.exe run.py --image "C:\path\to\image.nii" --aorta-mask "C:\path\to\mask.nii" --output prediction.json
```

Replace the input paths with the actual matching files. For the website,
follow the current-checkout setup in `DEMO_GUIDE.md`, build with Node once,
and run `explorer.py`. Its loopback address is local, not a public Devpost link.

For a new pair:

```powershell
.\.venv313\Scripts\python.exe run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
```

Do not replace the submitted fusion defaults with an experiment immediately
before judging. Do not install optional Torch/ONNX research dependencies just
to run the release.

For the laptop offline check, disconnect its network and run the CLI on a
real compatible CT/mask pair, then launch the Explorer, inspect CT and export
JSON. This is a team verification step, not a claim that a native Windows
measurement has already been performed.

Before presenting: complete that check, open the deck, preload a CT case,
verify JSON export, confirm projector readability, disable disruptive
notifications and agree who advances slides. Measure the actual laptop and
record it honestly; a Linux receipt is not a substitute.

## 15. Claims: use these words

| Say | Avoid |
|---|---|
| “Local reference precision 0.778, recall 0.737, F1 0.757” | “78% accurate” |
| “Synthetic topology F1 0.8785 across 24 procedural cases” | “88% accurate on patients” |
| “All 25 supplied scans completed” | “All branches were detected” |
| “1.658 mm mean origin error across fourteen matched branches” | “Every origin is within 1.658 mm” |
| “Estimated radius; two evaluated fusion matches” | “Clinically accurate radius measurement” |
| “Fusion is our selected development operating point” | “We proved this is the best possible algorithm” |
| “Two proposal passes plus a logistic filter” | “The shipped model uses an ensemble of all our algorithms” |
| “Runs offline after setup” | “Works without installing dependencies or data” |
| “Four-core Linux resource measurements” | “Validated on the organizer's Windows laptop” |
| “Prototype for visual verification” | “Ready for autonomous clinical decisions” |

## 16. Reading map and source of truth

| Question | Source |
|---|---|
| What exactly should we submit? | [DEVPOST_SUBMISSION.md](DEVPOST_SUBMISSION.md) |
| How do we run it? | [DEMO_GUIDE.md](DEMO_GUIDE.md) |
| Where are the downloadable artifacts? | [Current submission kit](https://github.com/Coder-Meet/battleoftheschool/releases/tag/branchseed-submission-fusion-2026-09-13) |
| How was the current default chosen? | [PRODUCTION_WORKFLOW.md](PRODUCTION_WORKFLOW.md) |
| Where is the earlier research and strict-release history? | Git history before the 2026-09-13 cleanup commit |
| What are the exact score and image inputs? | [docs/media/metrics.json](docs/media/metrics.json) |
| Where is the current validation receipt? | [docs/fusion-restored-validation.json](docs/fusion-restored-validation.json) |
| What did the fusion batch measure? | [Judge-package verification receipt](presentation/evidence/judge-fusion-verification.json) |

If an older document, slide or recording conflicts with the accepted fusion
configuration or these current receipts, explain the version difference and
use the current evidence. Do not silently combine metrics from different
algorithms, cohorts or timing scopes.
