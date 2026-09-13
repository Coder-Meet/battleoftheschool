# Branchseed — presenter cheatsheet

Everything needed to explain the system, define the terms, run the demo and
handle the questions. Figures come from `ROBUSTNESS_PROTOCOL.md`.

---

## The 30-second pitch

> You get a CT scan and a mask of the aorta — nothing else. The branches are
> visible in the image but deliberately unlabelled. We anchor to the aortic wall,
> ask which tube-shaped bright structures touch it, and trace each one outward
> until it either survives 5 mm or forks. Small branches can be affected by
> partial-volume averaging, and contrast varies
> seven-fold between patients — so we set our brightness threshold from each
> scan's own measured contrast instead of a fixed number. That one change took F1
> from 0.84 to 0.93 on 26 synthetic stress cases. About one second per synthetic case, CPU only.

---

## The two ideas everything rests on

**1. Partial volume averaging.**
A CT voxel that is half vessel and half fat reports the *average* of the two. A
small branch can occupy only part of a voxel. **Partial-volume averaging can
make small branches appear dimmer than the aorta.** This is not universal:
resolution, contrast timing, flow and reconstruction also affect the measured
brightness. Our thresholds account for the scan's measured contrast.

**2. The mask is an anchor, not an answer.**
The supplied mask contains only the parent aorta. It tells you where to look, not
what to find. We never segment the whole vascular tree — we only ask "what leaves
*this* wall?" That restriction is why it runs in seconds on a laptop instead of
needing a GPU.

---

## Hounsfield units (the CT brightness scale)

Calibrated physical units, not arbitrary pixel values.

| HU | What it is |
|---|---|
| −1000 | Air |
| −100 | Fat |
| 0 | Water (definition point) |
| +40 | Muscle, soft tissue |
| **82–572** | **Contrast-filled aorta across our 25 cases** |
| ≈600+ | Bone and calcification — our false-positive trap |
| +1000 | Dense cortical bone |

**That 82–572 spread is the whole story.** The original code demanded a branch be
within a fixed 65 HU of the aorta. On a 557 HU scan that silently required a
branch to be 87% as bright as the aorta — three cases returned *zero* branches.

---

## The pipeline (all in `detector.py`, entry point `detect()`)

| # | Stage | What it does | Code |
|---|---|---|---|
| 1 | Crop and standardise | 20 mm box around the mask, resampled to a 1 mm isotropic grid. Everything downstream reasons in mm, never voxel indices. | `prepare_roi()` |
| 2 | Learn "blood" for *this* scan | Measure aorta brightness inside the mask and tissue brightness 8–16 mm outside. Threshold is set from the **measured gap**, so contrast timing stops driving sensitivity. | `support_contrast_fraction` |
| 3 | Score tubularity | Multi-scale Sato filter at 0.8 / 1.5 / 2.5 mm asks "does this look like a cylinder?" Separates arteries from blobs and sheets. | `skimage.filters.sato` |
| 4 | Find wall contacts | Search a thin shell 1.5–3.5 mm outside the mask. Bright tube-like clusters touching the wall become candidate origins. Cropped aorta ends are excluded. | `cap_mask()` |
| 5 | Trace outward | Minimum-cost path (Dijkstra) over a field preferring wide, bright, tube-like voxels. Up to 10 mm or until the vessel forks. Survive 5 mm and you're a branch. | `_trace()`, `MCP_Geometric` |
| 6 | Measure and report | Walk back to the wall for the ostium, 5 mm along for the seed, perpendicular cross-section for radius, ostium→seed for direction. Convert to physical mm. | `wall_origin()`, `cross_section_radius()` |

Every rejected candidate is logged with a named reason, which is what makes the
system debuggable rather than a black box.

---

## Glossary

### Anatomy and imaging

| Term | Meaning |
|---|---|
| **CTA** | CT angiogram — scan timed just after iodine contrast injection so blood is bright. Time it wrong and arteries look like everything else. |
| **Hounsfield unit (HU)** | Calibrated CT brightness. Water 0, air −1000, bone ~1000. |
| **Voxel** | A 3D pixel. Ours are 0.6–1.5 mm — same order as the vessels we're hunting, hence partial volume. |
| **Axial slice** | Horizontal cross-section through the body. A head-to-toe vessel appears as a bright dot. |
| **Lumen** | The open channel inside a vessel that carries blood and lights up with contrast. The mask is of the aortic *lumen*, not the wall. |
| **Ostium** (pl. ostia) | The opening where a branch leaves its parent. **Our primary output** — 25% of the score. |
| **Parent / daughter** | The aorta / an artery coming directly off it. A vessel branching off *another branch* does not count. |
| **Common trunk** | One wall opening that splits a few mm later. Counts as **one** daughter, not two. |
| **Bifurcation** | Where a vessel divides. We stop tracing at the first one. |
| **Calcification** | Calcium in vessel walls. Very bright (600 HU+), a persistent false-positive trap. |
| **Iliac arteries** | Where the aorta finally splits near the pelvis. Explicitly **out of scope**. |
| **MIP** | Maximum intensity projection — collapse a slice stack keeping the brightest voxel per ray. What you see in `viz/`. |
| **NIfTI (.nii)** | The file format. Carries the voxel grid *and* the transform into physical space. |
| **Isotropic / anisotropic** | Voxels equal on all sides, or not. Thick-slice anisotropic scans are a known weak spot. |

### Our vocabulary

| Term | Meaning |
|---|---|
| **Seed** | The point exactly 5 mm along the branch from its ostium, measured *along the traced path*, not in a straight line. |
| **Support** | Our binary "this voxel could plausibly be contrast-filled blood" mask. Everything downstream depends on it. |
| **Shell** | The 1.5–3.5 mm band just outside the aortic wall where we hunt for origins. Close enough to be attached, far enough to be distinguishable. |
| **Vesselness** | A 0–1 tubularity score per voxel from local image curvature. High for cylinders, low for blobs. |
| **MCP** | Minimum cost path — Dijkstra in 3D. Cost falls as a voxel gets wider, brighter and more tube-like, so the cheapest route follows the vessel. |
| **Cap mask** | Suppression of the flat faces where the scan cropped the aorta. Acquisition artefacts, not origins — the spec calls this out. |
| **Evidence score** | Our heuristic 0–1 confidence. **Not a calibrated probability** — say so if asked. |
| **Strict vs review profile** | Two tunings. *Strict* produces the submission. *Review* over-proposes for human labelling; the optional classifier can only **remove** candidates, never invent them. |
| **Stress family** | One of 13 adversarial synthetic scenarios — tortuous parents, touching veins, mural thrombus, imperfect masks, dense branch fields, negative controls. |
| **Negative control** | A synthetic case with *no* eligible branches. Correct output is an empty list. Catches a detector that hallucinates. |
| **One-to-one matching** | How scoring works: each prediction pairs with at most one reference. **A duplicate detection is a false positive.** |

**If you memorise six:** lumen, ostium, daughter, partial volume, vesselness, seed.

---

## Numbers to have ready

### Validation — the pre-registered bake-off

26 procedural stress cases, 64 analytic reference daughters, two frozen seeds.
Variants written down *before* results were seen; selected on F1 at 3 mm tolerance.

| Detector | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (fixed HU offset) | 46 | 0 | 18 | 1.000 | 0.719 | 0.836 |
| **Contrast-relative threshold (selected)** | **56** | **1** | **8** | **0.982** | **0.875** | **0.926** |

Ten branches recovered for the cost of one false positive. Matched-pair geometry:
ostium error 0.84 mm mean / 1.76 mm p95, seed 0.65 mm, radius 0.22 mm,
direction 10.6°.

### Budget

| Constraint | Limit | Ours |
|---|---|---|
| Runtime per case | 60 s average | ≈1 s synthetic; 22.3 s slowest real case |
| Memory | 8 GB | 1.43 GiB peak |
| Hardware | 4 CPU cores, no GPU | CPU only, 4 threads |
| Network | None | No downloads, no CDNs, fonts bundled |

### Required geometry and current implementation defaults

| Quantity | Value |
|---|---|
| Seed distance | 5 mm |
| Trace length | 10 mm (or first bifurcation) |
| Minimum radius | Implementation default 0.7 mm; organizer minimum still pending |
| Working grid | Implementation default 1.0 mm isotropic |
| Dev cases | 25 CT + mask pairs; 19 judge-approved, AI-assisted/non-exhaustive targets on subjects 019–023; no complete expert labels |

---

## Demo order of play

Have the Explorer already running and a terminal open before you start.

**0:00 — Show the problem first.** Open `viz/subject005.png`. Point at the cyan
outline, then the branches leaving it.
> "This cyan outline is everything we're given. Every vessel coming off it is what
> we have to find. The 25-case set has no complete expert branch annotations; five
> reused development cases have 19 non-exhaustive AI-assisted targets."

**0:45 — Run the required command.**
```bash
python run.py --image data/subject005/orig5.nii \
              --aorta-mask data/subject005/mask5.nii \
              --output prediction.json
```
> "That's the exact command in the spec. No manual point placement, no per-case
> tuning. About six seconds on this laptop."

**1:15 — Show the output.** `cat prediction.json`, then walk the four fields on one
daughter. Emphasise **physical millimetres, not voxel indices**.
> "Ostium is where it leaves the wall. Seed is 5 mm along the vessel. Radius from a
> perpendicular cross-section. Direction is a unit vector. All in the scan's own frame."

**2:00 — Open the Explorer.** This is the visual payoff.
```bash
python explorer.py --port 8000      # then http://127.0.0.1:8000
```
Orbit the aorta, click a branch, show the path and arrow — then jump to the
**linked CT slices**.
> "A convincing 3D mesh isn't evidence. Every branch we claim, you can click and
> land on the actual CT voxels it came from."

**3:15 — The validation story.** Your strongest ground. Show the bake-off table.
> "We generate CTs from vessel geometry we define, so the synthetic answer is
> exact, and separately report local one-to-one matching against 19 reused-
> development targets. We fixed the synthetic candidate variants and metric
> before looking at results; every failure remains in the report."

**4:15 — Close on a failure, deliberately.** Name a weak spot, say what's next.
> "Five reused development cases have 19 judge-approved but AI-assisted and
> non-exhaustive targets, not complete expert ground truth. Their metrics are
> local only, so independent real-scan recall is genuinely unknown."

---

## Questions you will get

**How do you know it works without complete expert labels?**
We generate CT volumes from vessel geometry we define, so synthetic ground truth
is analytic and exact—26 adversarial cases across 13 failure families, plus
negative controls that must return empty. We separately report local matching on
the 19 reused-development targets, while explicitly withholding independent
and clinical accuracy claims.

**Why not just train a neural network?**
Twenty-five cases, only 19 non-exhaustive AI-assisted targets on five reused
development cases, four CPU cores, and no GPU are not enough independent expert
evidence to justify a deployable neural model. The optional research classifier
can only filter reviewed candidates, never invent new ones, and remains
non-deployable.

**What's your precision and recall on real patients?**
Local reused-development metrics exist, but independent real-patient accuracy is
**unknown**. We have 19 judge-approved, AI-assisted/non-exhaustive targets across
subjects 019–023 and report local one-to-one matching against them. They are not
complete expert/clinical truth or an independent hidden test, and no official
evaluator or weighted challenge score or native organizer validation is available.
Name this exact set; never present its local metrics as clinical or hidden-test
performance.

**How do you avoid counting the same vessel twice?**
Candidates whose ostia and seeds both land within a few mm are merged before
output. Matters because scoring is one-to-one: a duplicate isn't neutral, it's a
false positive.

**Veins are bright too. Why don't they fool you?**
Four independent filters must agree: brightness in the learned blood band,
tubularity above a floor, physical contact with the aortic wall, and a traceable
lumen surviving 5 mm outward. A vein alongside the aorta usually fails wall-contact
or the trace. Still a real failure mode in low-contrast scans — say that.

**Why is the aorta given but not the branches?**
Segmenting the aorta is solved and explicitly out of scope. The mask is the search
anchor; the challenge is discovering what comes off it, which varies in number and
position between patients.

**What happens with no branches, or a very short aorta?**
Returns an empty list — correct and required. Flat cropped ends of a short segment
are suppressed as artefacts rather than reported as origins.

**Where does it break?**
Answer immediately, don't hedge: **thick-slice acquisitions** (1.5 mm voxels lose
small-vessel signal to partial volume), **dense branch fields** (nearby origins
merge), and **low-contrast scans** where the aorta is barely brighter than muscle.
Four of our eight remaining misses are high-noise small branches.

---

## Lines to hold

- Never call the evidence score a probability. It is a heuristic ranking aid.
- Never quote a synthetic F1 as real-scan accuracy. Name the test set every time.
- Never say "detects all branches". Say what it detects and what it misses.
- Never claim clinical validity. This is a research prototype, not a device.
- If you don't know a number, say "I'd have to check". Bluffing costs more.

This isn't false modesty — it's what separates you from every team claiming a
number they can't defend. A judge who tries to break your claim and finds you
already documented the limit will trust everything else you said.
