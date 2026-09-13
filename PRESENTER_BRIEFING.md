# Branchseed — complete presenter briefing

**Team reference for the five-minute talk, live Explorer demo and judge Q&A.**

The submitted choice is **deterministic strict**, a fixed classical detector.
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

**Release:** 1 mm grid; native contrast scale 1.2; strict connection policy;
four threads; no trained model required.

**Numbers to remember:**

- Five reused cases / 19 reference targets / local 3 mm matching:
  **8 TP, 3 FP, 11 FN; precision 0.727; recall 0.421; F1 0.533; count MAE 2.0.**
- Twenty-four synthetic topology cases:
  **46 TP, 0 FP, 3 FN; F1 0.968.** Synthetic regression, not patient accuracy.
- All 25 supplied scans completed: **5.32 s mean, 23.21 s maximum,
  1481 MiB maximum sampled process-tree RSS**, under four-core Linux affinity.
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

Implementation: [detector.py](detector.py), especially `normalize`, `enhance`,
`propose`, `_trace` and `resolve`. The longer research history is in
[Steven's review](STEVEN_ALGORITHM_REVIEW.md).

### 5.1 Validate and normalize

Check CT/mask dimensions and physical geometry, extract the parent region,
crop around it and resample into a 1 mm working grid. Intensity and mask
resampling have different roles: preserve interpretable CT values while
retaining a discrete parent label.

Estimate blood intensity inside the parent and background around it.
Scan-relative contrast reduces reliance on one fixed global brightness
threshold. The native-contrast setting retains sensitivity to original CT
support; the release uses scale 1.2. Poor contrast cannot be repaired simply
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
classifier applied after proposal generation. Broad review-union pools are
research alternatives, not the submitted origin policy.

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

## 6. Why the default is strict

| Configuration / evaluation | TP / FP / FN | Precision | Recall | F1 | Count MAE | Status |
|---|---:|---:|---:|---:|---:|---|
| Fixed strict | 8 / 3 / 11 | 0.727 | 0.421 | 0.533 | 2.0 | Submitted default |
| Leave-one-case-out selection composite | 8 / 9 / 11 | 0.471 | 0.421 | 0.444 | 3.2 | Selection-procedure estimate |
| Retrospective RF winner | 9 / 1 / 10 | 0.900 | 0.474 | 0.621 | 1.8 | Development only |
| Guarded 1 mm connection recovery | 9 / 3 / 10 | 0.750 | 0.474 | 0.581 | 2.2 | Opt-in experiment |

All rows use the same five reused reference cases and local 3 mm matching.
They are not four independently validated deployments.

Strict has the strongest completed package of fixed configuration, source
identity, deterministic replay and topology evidence. The RF result used
reused development references and relaxed review-union proposals. It lacks
the evidence needed to promote it as the safer unseen-case choice.

The later guarded recovery follows bounded parent-connected support around
curved connections while preserving strict detections. It gains one local
match, passes the 24-case topology comparison and showed no lost baseline
matches or added false positives across 80 procedural synthetic comparisons.
However, count MAE worsens. A newly correct branch can be found in a case
where the detector already overcounts, increasing count error even while
F1 improves.

The judge emphasized daughter counts, so the team accepted strict. That does
**not** prove strict is optimal or that a score above 0.7 is overfitting.
The problem is selecting repeatedly on reused data without independent
confirmation, not crossing a particular numeric threshold.

Experimental CLI from the current Git checkout, **not the submitted default**:

```bash
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz \
  --output experimental.json --diagnostics experimental-diagnostics.json \
  --threads 4 --recover-connected-origins
```

## 7. Optional ML and research: what exists and why

| Approach | What it adds | Strength | Drawback / disposition |
|---|---|---|---|
| Logistic candidate filter | A small weighted classifier over 13 features | Fast and inspectable | Cannot recover absent proposals; label provenance matters |
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
The detailed feature contract and model metadata live in the research tools.

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
pipelines. See [ADDITIONAL_PAPERS.md](ADDITIONAL_PAPERS.md) for scope and results.

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

Do not say “72.7% accuracy.” The strict precision is 0.727; recall and F1 are
different, and “accuracy” would need a defined negative population.

### Strict counts by case

| Case | Reference targets | Strict predictions | Absolute count error |
|---|---:|---:|---:|
| subject019 | 3 | 0 | 3 |
| subject020 | 4 | 1 | 3 |
| subject021 | 3 | 3 | 0 |
| subject022 | 6 | 7 | 1 |
| subject023 | 3 | 0 | 3 |
| Total / mean | 19 | 11 | MAE 2.0 |

Zero predictions on two reference cases are a real recall weakness.
At 3 mm, matches are concentrated in subjects 021 and 022. Do not pick a
successful case and imply equal performance across all patients.

### Sensitivity to tolerance

| Local ostium tolerance | Strict TP / FP / FN | Strict F1 |
|---|---:|---:|
| 2 mm | 6 / 5 / 13 | 0.400 |
| 3 mm | 8 / 3 / 11 | 0.533 |
| 5 mm | 8 / 3 / 11 | 0.533 |

The organizer has not supplied the official evaluator. Local 3 mm matching
is our declared analysis setting, not a claim about the hidden scorer.
The broader tolerance did not recover the remaining unmatched references.

### Geometry: read the sample size

At 3 mm, across the eight strict matches:

| Measurement | Mean error | Number evaluated |
|---|---:|---:|
| Ostium location | 1.59 mm | 8 |
| Seed location | 1.22 mm | 8 |
| Direction angle | 13.95° | 8 |
| Seed radius | 0.253 mm | **1** |

Only three of the 19 reference radii are known, and only one is matched by
strict. A radius headline based on that one point would overstate evidence.
Matched-error summaries also exclude the 11 missed targets.

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
Strict produced 46 TP / 0 FP / 3 FN, F1 0.968, with zero negative-control
detections. These cases exercise known geometric failure modes. Real CT adds
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
recording. Their on-screen branch counts can differ from today's strict
outputs. Explain them as feature demonstrations; use current JSON and frozen
receipts for accuracy claims. For example, the recording's subject001 count
is not the final batch's subject001 count.

## 12. Five-minute talk and live demo

### Speaker ownership

| Time | Owner | Main job | Handoff |
|---|---|---|---|
| 0:00–1:15 | Speaker 1 | Explain direct origins and one-opening semantics | “Now we turn that definition into a physical-space detector.” |
| 1:15–2:30 | Speaker 2 | Explain support, tracing, measurements and strict choice | “Every prediction can then be inspected in the Explorer.” |
| 2:30–3:45 | Speaker 3 | Run the live 75-second Explorer sequence | “Those views help inspect a prediction; here is the measured evidence.” |
| 3:45–5:00 | Speaker 4 | Show results, runtime, limitations and next step | Close on inspectable evidence and local execution |

The eight-slide deck and script are in the
[presentation kit](https://github.com/Coder-Meet/battleoftheschool/releases/download/branchseed-final-2026-09-13/branchseed-final-presentation.zip).
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

**Why is the real-reference F1 only 0.533?**\
Strict misses 11 of 19 local targets, largely through proposal/connection
limitations. Synthetic topology checks do not capture all real CT variation.
The next step is expert-reviewed misses and unused-case validation.

**Isn't 0.727 your accuracy?**\
It is precision: 8 of 11 predictions matched locally. Recall is 8 of 19,
or 0.421; F1 is 0.533. The incomplete references further limit interpretation.

**Why show 0.968?**\
It is the separately labelled synthetic topology F1. It tests known
geometric behaviors; it is not patient or clinical accuracy.

**Why not submit the random forest at 0.621 F1?**\
That is a retrospective winner on reused cases with relaxed proposals and
insufficient independent promotion evidence. The fixed strict path has the
stronger completed validation package.

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
The submitted detector is classical image processing and geometry. We also
built optional learned candidate filters and used AI-assisted development.
Those facts should not be conflated.

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
It is an estimate. Only one strict matched reference has a measured radius;
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
Not yet. The final resource receipt is four-core Linux with 1481 MiB maximum
sampled RSS; native organizer-Windows timing is still required.

**Why not a large 3D network?**\
The task supplies little trusted annotation and has a CPU/offline budget.
A heavier network would add data and validation requirements without
established benefit here.

**Are all 150 exported instances correct?**\
No such claim is supported. That is an output count across 25 completed
scans. Only the five-case reference comparison has local matching results.

**Why does the video show a different count?**\
It is a preserved earlier interface recording. The release JSON and frozen
strict receipts are the current source for algorithm results.

**Did AI help build this?**\
Yes. Devin and any additional tools actually used by the team should be
credited. The methods, tests, limitations and artifact provenance are
available for inspection.

**Would you trust it clinically?**\
It is a research prototype, not clinically validated. Its current role is
exploring candidate detections and their evidence.

## 14. Laptop preparation and commands

Use [DEMO_GUIDE.md](DEMO_GUIDE.md) as the canonical operational reference.
Download the final strict submission and presentation before the event.
The strict ZIP contains dependencies/wheels, source, built web assets,
predictions and required static checks; supply the organizer scans separately.

### Offline Windows x64 / Python 3.13

From the extracted submission's **`source` directory**, with Python 3.13.3
x64 installed, use the sibling **`windows-wheelhouse` directory**:

```powershell
python --version
python -m venv .venv313
.\.venv313\Scripts\python.exe -m pip install --no-index --find-links ..\windows-wheelhouse -r requirements.txt
$env:OPENBLAS_NUM_THREADS="4"
$env:OMP_NUM_THREADS="4"
$env:MKL_NUM_THREADS="4"
$env:ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="4"
.\.venv313\Scripts\python.exe explorer.py --data-root "C:\path\to\data" --port 8000
```

Replace the data path with the actual scan directory. Open
`http://127.0.0.1:8000` on that laptop. The address is local, not a public
Devpost link. Node is unnecessary for serving the already-built release
website. Use the guide's Node commands only when rebuilding the frontend.

For a new pair:

```powershell
.\.venv313\Scripts\python.exe run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
```

Do not replace the submitted strict defaults with an experiment immediately
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
| “Local reference precision 0.727, recall 0.421, F1 0.533” | “73% accurate” |
| “Synthetic topology F1 0.968 across 24 procedural cases” | “97% accurate on patients” |
| “All 25 supplied scans completed” | “All branches were detected” |
| “1.59 mm mean origin error across eight matched branches” | “Every origin is within 1.59 mm” |
| “Estimated radius; one evaluated strict match” | “Clinically accurate radius measurement” |
| “Fixed strict is our best-supported release choice” | “We proved this is the best possible algorithm” |
| “Optional models are research comparisons” | “The shipped model uses an ensemble of all our algorithms” |
| “Runs offline after setup” | “Works without installing dependencies or data” |
| “Four-core Linux resource measurements” | “Validated on the organizer's Windows laptop” |
| “Prototype for visual verification” | “Ready for autonomous clinical decisions” |

## 16. Reading map and source of truth

| Question | Source |
|---|---|
| What exactly should we submit? | [DEVPOST_SUBMISSION.md](DEVPOST_SUBMISSION.md) |
| How do we run it? | [DEMO_GUIDE.md](DEMO_GUIDE.md) |
| Where are the downloadable artifacts? | [Final release](https://github.com/Coder-Meet/battleoftheschool/releases/tag/branchseed-final-2026-09-13) |
| How was the release choice made? | [FINAL_EVALUATION_RESULTS.md](FINAL_EVALUATION_RESULTS.md) |
| What happened in the last improvement attempt? | [ACCURACY_RECHECK.md](ACCURACY_RECHECK.md) |
| What changed in all algorithms? | [STEVEN_ALGORITHM_REVIEW.md](STEVEN_ALGORITHM_REVIEW.md) |
| What did the papers contribute? | [ADDITIONAL_PAPERS.md](ADDITIONAL_PAPERS.md), [RESEARCH_IMPLEMENTATION.md](RESEARCH_IMPLEMENTATION.md) |
| What is and is not compliant? | [SUBMISSION_AUDIT.md](SUBMISSION_AUDIT.md) |
| What are the exact score and image inputs? | [docs/media/metrics.json](docs/media/metrics.json) |
| Where are replay receipts and model lineage? | [labels/final-eval/selection](labels/final-eval/selection) |
| What did the final batch measure? | [batch report](labels/finalization/batch_report.json), [resource receipt](labels/finalization/final-batch-resource.json) |

If an older document, slide or recording conflicts with the accepted strict
configuration or these current receipts, explain the version difference and
use the current evidence. Do not silently combine metrics from different
algorithms, cohorts or timing scopes.
