# External sources of truth

The 25 supplied scans have no branch annotations. Everything below is a way to
get labelled vascular anatomy *without uploading our scans anywhere*. Our scans
were provided under the challenge terms and are patient data: do not upload
them to any third-party service without written approval from Toralis.

Plan doc question 8 (are external datasets and pretrained weights allowed?)
is still open. Until it is answered, use these sources to **validate**, not to
train the submitted model. Validation-only use is uncontroversial.

## 1. Toralis reference outputs (the real answer key)

The challenge brief promises "a small development subset with example
reference outputs". Nothing has arrived. This is the only source that matches
the hidden-test annotation conventions (ostium placement, minimum size,
common-trunk handling). Chase it first; every other source below is a proxy.

Use: drop the reference JSONs next to the cases and run `benchmark.py`.

## 2. AVT — Aortic Vessel Tree CTA dataset

| | |
|---|---|
| What | 56 CT angiograms with expert binary masks of the aorta **and its branches** as one label |
| Sub-collections | KiTS (abdominal, 20), Rider, Dongyang; ~6.3 GB total |
| Licence | CC BY 4.0 — free for any use, attribution required |
| Access | Direct download from Figshare, no registration |
| Link | https://figshare.com/articles/dataset/Aortic_Vessel_Tree_AVT_CTA_Datasets_and_Segmentations/14806362 |
| Paper | Radl et al., *Data in Brief* 2022, https://www.sciencedirect.com/science/article/pii/S2352340922000130 |

Why it matters: the masks are the whole tree, so subtracting the trunk leaves
the branches, and where a branch touches the trunk is an ostium. A short
script gives ostium, direction and radius for every traced branch on real
anatomy, in the same JSON schema we output.

Caveats: branches are one label, not per-vessel instances, so a common trunk
and its children come out as one blob and need the same junction logic we
already run. Which abdominal branches the annotators traced is not stated in
the paper text; check a KiTS case after download before relying on it. The
KiTS subset is the abdominal one; Rider and Dongyang lean thoracic.

## 3. AortaSeg24 — multi-class aortic branches and zones

| | |
|---|---|
| What | 100 CT angiograms, 1 mm isotropic, with 23 labels: 13 branches and 10 aortic zones |
| Branches labelled | celiac, superior mesenteric, left and right renal, left and right common / external / internal iliac, innominate, left common carotid, left subclavian |
| Licence | Not published; access requires a signed data-use agreement (DocuSign), approval within about a day |
| Access | https://aortaseg24.grand-challenge.org/dataset-access-information/ |
| Paper | https://arxiv.org/abs/2502.05330 |

Why it matters: per-branch instance labels for exactly the four large branches
we must never miss. Ostium = where the branch label touches the aorta label.
Radius and direction follow from the label geometry.

Caveats: no lumbar, gonadal or accessory-renal labels, so it cannot score small
branches. Read the agreement before any training use; validation use is fine.

## 4. TotalSegmentator — run locally, never uploaded

| | |
|---|---|
| What | Pretrained nnU-Net models for 100+ CT structures, CPU capable |
| Free task `total` | Apache-2.0: aorta, inferior vena cava, iliac arteries, kidneys, every vertebra |
| Licensed task `renal_arteries` | celiac trunk, superior mesenteric artery, renal arteries. Flagged `commercial: True`; a free non-commercial licence key is available from the authors. Refuses fast mode, so it runs at full 1.5 mm resolution and is slow on CPU |
| Link | https://github.com/wasserth/TotalSegmentator |

Why it matters, in two ways:

- The free `total` model is a **bone-and-vein oracle** on our own 25 scans. A
  candidate whose path lies inside a vertebra or the vena cava is a false
  positive by construction. Those are direct labels for the classifier and a
  sanity check on human reviews.
- The `renal_arteries` model, if the licence is granted, labels the celiac,
  SMA and renals on all 25 cases, which settles the large-branch review work.

Caveats: development-time only. It is far too heavy for the 4-core / 8 GB /
60 s submission target and its licence forbids bundling. Nothing from it may
ship inside `run.py`.

## 5. What not to do

- No cloud radiology or "upload your DICOM" services. Data terms and privacy.
- No training on any of the above until Toralis answers plan-doc question 8.
- No mixing external labels with human reviews of our scans without recording
  provenance per row; `learning.py` rejects reviews that lack the export schema
  for this reason.

## How each source plugs into the repo

| Source | Converts to | Consumed by |
|---|---|---|
| Toralis references | `reference.json` per case | `evaluate.py`, `benchmark.py` |
| AVT masks | `reference.json` via tree-minus-trunk script (to write) | `benchmark.py` on AVT cases |
| AortaSeg24 labels | `reference.json` via label-contact script (to write) | `benchmark.py` on AortaSeg24 cases |
| TotalSegmentator `total` | bone / vein masks per case | extra classifier features, review sanity checks |
| TotalSegmentator `renal_arteries` | large-branch pseudo-references | review pre-fill, recall check on the four big branches |
