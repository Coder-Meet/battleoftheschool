# Devpost submission — ready-to-paste copy

[Download the current fusion submission kit](https://github.com/Coder-Meet/battleoftheschool/releases/download/branchseed-submission-fusion-2026-09-13/branchseed-submission-fusion-kit.zip).
Extract it and open `START_HERE.html`; PPTX, PDF, editable Word, the speaking script and video are included.

## 1. What to submit

The [live event requirements](https://battle-of-the-schools.devpost.com/) list
**September 13, 2026, 11:00 a.m. Eastern** as the deadline. The project must
include its name/tagline, description, public code repository, a public demo
video **up to three minutes**, track selection and all teammates. Team
registration and the school you represent must also be completed.

**The video and live presentation are separate requirements.** Our 40-second
showcase meets the published duration limit. Keep the five-minute presentation
as a live Explorer demonstration. Slides are optional on Devpost and enter the
Best Slide Aesthetics side-quest.

| Field / deliverable | Use this |
|---|---|
| Project name | **Branchseed — Aorta Explorer** |
| Tagline / elevator pitch | Copy the short pitch in section 2 |
| Track | **Toralis Labs Healthcare** |
| Public repository | https://github.com/Coder-Meet/battleoftheschool |
| Project story | Copy section 3 |
| Built with | Copy section 4; distinguish runtime from research tools |
| Demo video | Upload the [40-second showcase](https://github.com/Coder-Meet/battleoftheschool/releases/download/branchseed-final-2026-09-13/branchseed-showcase.mp4) to YouTube/Vimeo, or use a public Drive link if accepted by the form |
| Slides | Use the [current fusion PPTX/PDF and script](https://github.com/Coder-Meet/battleoftheschool/releases/tag/branchseed-submission-fusion-2026-09-13); upload the PDF if the form accepts files |
| Technical deliverables | [Standalone fusion judge download](application/README.md): selected inference code/model and offline Windows wheels, without the website. Do not substitute the older strict ZIP. |
| Cover and gallery | Images and captions in section 5 |
| Team | Add every teammate's real Devpost account; complete your team name and school |

The public requirements allow “YouTube, Vimeo or another public link.”
The signed-in submission editor has not been inspected; it may restrict
which providers its video embed field accepts. If a Drive or MP4 URL does
not embed, use YouTube or Vimeo and place the release link in an additional
links field. A GitHub download link is not necessarily an embeddable player.
Test the final video link while signed out; it must not request access.

The Toralis brief additionally requests source, dependencies, setup/run
commands, predictions for the development set and visual checks for at least
three cases. Use the selected **judge-fusion** application ZIP for inference
and its separate evidence download for matching predictions, visual checks
and verification receipts. The older full ZIP contains strict outputs and is
retained for the historical presentation. Ask the track organizer for any
separate upload location.

## 2. Project name, elevator pitch and short description

### Name

Branchseed — Aorta Explorer

### Elevator pitch

From CT to a map you can inspect: discover aortic branch origins, verify predictions in 3D and linked CT, and export measurements offline.

### Short project description

Branchseed turns a CT scan and a parent-aorta mask into inspectable daughter-vessel
instances. A CPU fusion pipeline proposes direct aortic openings and
estimates each origin, a seed 5 mm along its proximal path, radius and direction.
The Explorer connects those predictions to a 3D reconstruction, linked CT planes,
a wall map and an interior tour. Challenge JSON stays in physical millimetres,
and inference runs locally without a GPU or external service.

### GitHub About description — owner action

GitHub denied this session permission to update repository metadata. A
repository owner can replace the current description with:

> Offline aortic branch discovery from CT and parent masks, with interactive 3D/CT evidence and physical-coordinate JSON. Built for the Toralis Labs challenge.

## 3. Project story — paste into Devpost

### Inspiration

A vessel is more than a bright shape in a CT scan. Where does it actually join
the aorta? Is it one common trunk or two separate openings? Does the proposed
branch continue into a supported lumen, or follow an artifact?

For the Toralis Labs challenge, we wanted an answer that could be inspected.
Our goal was to connect automatic branch discovery to the image evidence
behind each prediction, on the kind of offline CPU laptop the challenge
requires.

### What it does

Branchseed accepts a CT volume and a mask of the parent aorta, discovers a
variable number of candidate direct daughters, and exports an instance for
each accepted prediction:

- its opening at the aortic wall;
- a seed 5 mm along the proximal vessel;
- the estimated local seed radius;
- a unit direction vector and a parent link.

Our Aorta Explorer makes those outputs tangible. Orbit the parent vessel,
select a branch, inspect linked CT planes, see where origins sit on the wall
map, and travel through the reconstructed lumen. Export the same prediction
as machine-readable JSON. The interface keeps the scan, geometry and
measurements connected rather than leaving a reviewer with an isolated score.

### How we built it

The submitted detector combines classical image processing with a small
bundled logistic candidate filter. We
validate physical coordinates, crop around the parent, work on a 1 mm grid,
estimate scan-relative contrast, enhance tubular structures at multiple
scales, propose wall contacts and trace supported proximal paths. Connection,
length, size and shared-path checks help reject crop caps and duplicate
openings. The current origin-diameter policy uses the judge's 2 mm minimum
with an explicit allowance for native-voxel uncertainty.

We generate strict and broader review proposals at native contrast scale
0.9, score both with the bundled 13-feature logistic model at threshold
0.15, then merge survivors within 3 mm while retaining strict proposals
first. Scoring before merging prevents a weak overlapping proposal from
displacing a stronger representation before filtering. These are automatic
proposal passes; no manual review or model download is required at inference.

Python, SimpleITK, NumPy, SciPy and scikit-image power the detector. TypeScript
and Three.js power the Explorer; Vite builds its local assets. The Python
server serves the built website and analysis API. After setup, neither
inference nor the Explorer needs a hosted model, API key, CDN or GPU.

We also built a research pipeline: synthetic vascular phantoms with analytic
references, candidate-feature classifiers, physical CT patches, a small CNN,
model blends and case-separated selection. These experiments informed the
final decision; they are not all enabled in the submission.

### Challenges we ran into

Direct connectivity is harder than finding bright voxels. Thin or curved
vessels can lose contrast through partial volume, nearby structures can look
tubular, and relaxing a connection check can turn a missed branch into several
false positives. One opening must remain one instance even when its trunk
splits downstream.

The reference data also required care. The judge approved 19 AI-assisted
targets across five development cases, but the annotations may omit eligible
branches. We had already inspected these cases. Repeatedly choosing the best
score would not establish performance on new patients, so we tracked exposure,
replayed frozen evidence and kept real-reference agreement separate from
synthetic regression results.

### Accomplishments we're proud of

We built a complete path from new CT/mask inputs to valid physical-coordinate
JSON and an interactive evidence viewer. The standalone fusion batch completed
**all 25 supplied scans**, averaging **11.60 seconds per case**, with a
**52.67-second maximum** and **1499 MiB maximum sampled process-tree RSS**
under four-core Linux affinity. These are development measurements;
organizer-Windows timing remains to be measured.

We also made the performance claims inspectable:

| Evidence | Selected fusion result | What it measures |
|---|---|---|
| Five reused reference cases, 19 targets | **14 TP / 4 FP / 5 FN**; precision **0.778**, recall **0.737**, F1 **0.757**; count MAE **1.8** | Local one-to-one ostium agreement at 3 mm |
| Twenty-four synthetic topology cases | **47 TP / 11 FP / 2 FN**, F1 **0.8785**; four negative-control detections | Procedural topology regression at 3 mm |
| Twenty-five supplied scans | **25 completed**, all outputs identical to the full-checkout replay | Execution and output coverage, not detection accuracy |

The local reference result includes five missed targets. Synthetic F1 is not
real-patient accuracy, and neither result establishes hidden-test or clinical
performance. We would rather show the evidence and its limits than attach an
unsupported “accuracy” percentage to the product.

### What we learned

Proposal coverage and filtering order matter. The selected fusion workflow
raises local reference F1 from the earlier strict baseline's **0.533 to
0.757**, while reducing daughter-count error from **2.0 to 1.8**.
This improvement has a cost: on the same synthetic cohort, strict had zero
false positives while fusion has eleven, including four in negative controls.

The team selected fusion for its higher measured reference result. This is
a development decision, not an independently validated generalization claim.
Repeated selection on five reused cases can overfit even without training
new weights. Complete expert annotations and unused patients are the next
test of whether the improvement transfers.

### What's next for Branchseed

The next step is complete expert review of missed origins and unlisted
predictions, followed by evaluation on previously unused patients with
different contrast and slice thickness. That evidence can support better
candidate generation, curved-wall recovery and origin-size calibration.
We also need measurements on the organizer's actual Windows laptop.

Branchseed is a research prototype for branch discovery and visual
verification. It does not segment the parent automatically, reconstruct the
full distal tree or provide a clinically validated diagnosis.

## 4. Built with and credits

**Runtime/product:** Python, TypeScript, Three.js, Vite, SimpleITK, NumPy,
SciPy, scikit-image, Matplotlib.

**Research/authoring:** scikit-learn, PyTorch, ONNX Runtime, FFmpeg, Pillow.
The shipped learned component is the bundled logistic filter. Tree/CNN models
remain experimental; their training frameworks are excluded from the judge ZIP.

**Credits to include:** Toralis Labs for the challenge and supplied CT/mask
data; the referenced research papers and open-source libraries; local
Manrope/Fraunces fonts and their bundled licences; Devin for AI-assisted
development. Add any other coding tools used by teammates. The showcase
uses actual recorded UI footage and an original synthesized soundtrack.
No published paper's score is represented as our own.

The event rules permit credited libraries, datasets, pretrained models and
APIs. Describe your actual development work and tooling accurately; do not
invent authorship, training data, clinician participation or deployment claims.

## 5. Cover image and gallery order

All images are in [docs/media](docs/media). Full interface screenshots come
from the preserved recording; their visible counts and timings are historical.
The accuracy and runtime graphics are generated from the current stored
evaluation receipts.

| Order | Image | Paste-ready caption |
|---|---|---|
| Cover | `branchseed-cover.png` | From CT to connected anatomy: discover, inspect and export. |
| 1 | `explorer-overview.png` | Select a predicted origin and connect its 3D geometry to physical measurements. Recorded interface view. |
| 2 | `ct-evidence.png` | Inspect linked axial, coronal and sagittal CT evidence for the selected candidate. Recorded interface view. |
| 3 | `wall-map.png` | Organize predicted origins on an aortic wall map. Recorded interface view. |
| 4 | `interior-tour.png` | Explore the reconstructed lumen from within. Recorded interface view. |
| 5 | `evidence-scorecard.png` | Fusion results: five reused reference cases and 24 synthetic topology cases are separate evaluations. Local 3 mm matching; not hidden-test or clinical accuracy. |
| 6 | `runtime.png` | All 25 scans completed under four-core Linux affinity. Windows-laptop timing remains unverified. |
| Optional | `pipeline.png` | A physical-space pipeline from CT and parent mask to direct-origin instances and inspectable JSON. |

## 6. Final submission checklist

- [ ] Paste the title, elevator pitch and project story; preview formatting.
- [ ] Select Toralis Labs Healthcare and add every teammate.
- [ ] Confirm team registration, team name and represented school.
- [ ] Add the public repository and selected fusion technical-download link.
- [ ] Upload the showcase and test its public link while signed out.
- [ ] Add the cover and gallery; retain the scope in metric-image captions.
- [ ] Add the optional presentation PDF/link for the slide side-quest.
- [ ] Credit libraries, the sponsor dataset, papers and actual AI tools.
- [ ] Use **Submit**, then verify the project is entered in this hackathon;
      saving a portfolio draft alone is not proof of entry.
- [ ] Keep the fusion judge ZIP, current deck, prepared Explorer and scans on the presentation laptop.
- [ ] Run the laptop's offline check and preload the Explorer before judging.

Do not enter a local loopback address as a public “try it” URL. This product
runs on the team's machine; the repository and release provide the runnable
demo. Do not imply that this document, a Git push or a release upload has
submitted the Devpost form.

Sources: [event requirements](https://battle-of-the-schools.devpost.com/),
[event rules](https://battle-of-the-schools.devpost.com/rules),
[live Branchseed brief](https://docs.google.com/document/d/1oRb2R9pauvsC-9hDIfr23ojLx90JpCt0jVZjCD5l5Cg/edit).
The live event page and rules disagree on some judging/closing times; use
the organizer's current on-site announcements for those logistics.
