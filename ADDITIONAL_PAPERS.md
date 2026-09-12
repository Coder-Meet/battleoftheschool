# Three additional papers: implemented adaptations and measured limits

Reviewed 12 September 2026 alongside [the ML research plan](RESEARCH_IMPLEMENTATION.md).
The executable experiments are in `paper_methods.py`; none changes the default
`run.py` detector or the preregistered tree/CNN corpus.

## What the papers actually support

| Source | Relevant method | Application here | Important mismatch |
|---|---|---|---|
| [Danilov, Pryamonosov and Yurova, 2016 — Image Segmentation for Cardiovascular Biomedical Applications at Different Scales](https://doi.org/10.3390/computation4030035), §§2.2, 3–4 | Multiscale vesselness, aorta-border cleanup and thinning to a vascular graph. The cleanup removes a near-aorta voxel when it lacks a neighbor in the next outward distance layer. | Optional `border-cleaning` experiment. Existing production code already uses vesselness and skeleton junctions. | This is a broader segmentation/modeling paper, including coronary/cerebral work. Its uniform-grid assumptions, 15-voxel cleanup depth and aorta segmentation are not a direct-daughter challenge specification. |
| [Riffaud et al., 2022 — Automatic branch detection of the arterial system from abdominal aortic segmentation](https://doi.org/10.1007/s11517-022-02603-2), §2.2.1, Definition 5 | Origin-anchored principal-component direction: form rows `v_i - b`, take the largest eigenvector of `A.T @ A`, and orient by `sign(sum(A @ e))`. Local direction uses the proximal path through at most `3r`. | Optional `pca-direction` experiment, with physical arc-uniform samples and a 10 mm maximum. | It matches named arteries on an already segmented vascular tree containing the daughters, principally using lumen **and thrombus**. The supplied challenge mask contains only the parent lumen. Naming rules, expected orientations and branch-count restrictions are incompatible with the challenge. |
| [Tahoces et al., 2020, online 2019 — Automatic detection of anatomical landmarks of the aorta in CTA images](https://doi.org/10.1007/s11517-019-02110-x), §2.1 | Intensity-supported expansion from the aorta; a limited first expansion detects gross leakage; external voxels neighboring the parent form contact areas; a distance-transform maximum selects a contact point. | Optional `contact-growth` experiment for parent-connected proposal generation. | The published pipeline labels a fixed main-vessel set. It explicitly removes downward/posterior visceral branches and small vessels. Those exclusions would discard eligible unusual, lumbar or descending daughters here. |

### Source versions and access

- Danilov: [publisher full text](https://www.mdpi.com/2079-3197/4/3/35).
  The paper says its research code and DICOM data are available **on request**.
- Riffaud: the method equations were checked in the
  [HAL author preprint v1](https://hal.science/hal-03520790v1), submitted
  11 January 2022. Its cover notes a later v2; this audit does not assert that
  every equation or table is unchanged in the publisher's version. The reported
  segmentations come from PRAEVAorta/Nurea; an openly downloadable compatible
  training dataset or standalone licensed implementation was not verified.
- Tahoces: full text was checked in the
  [University of Santiago de Compostela repository](https://minerva.usc.gal/rest/api/core/bitstreams/3378c44a-0108-45db-b3c4-5f90ca48860a/content).
  A portable implementation and distributable expert-annotation cohort were
  not verified. Public access to the article does not establish permission to
  reuse patient data.

Downloaded PDF SHA-256 values, for identifying the exact reviewed texts:

| Text | SHA-256 |
|---|---|
| Danilov | `5dbd26e44f07bef913364f86d76572e5c64e238d37419b6ac49c0a218f722d70` |
| Riffaud HAL v1 | `b8cc1f27bbdaf682070529d74bcbef5e97378a1def63e52a952da35468091f7b` |
| Tahoces institutional copy | `bdc6789cc49413fb329eb85abe0cafc1beef95bd8b5a2b6fa5ee173018671204` |

### Published results are not our results

Riffaud reports 239 segmentations from 102 patients and accuracy for named artery
identification. These are not 239 independent patients or complete unnamed
daughter-discovery labels. The preprint reports that atypical directions and
non-anatomical connections cause errors.

Tahoces uses 33 scans for parameter adjustment and 30 for evaluation, out of
63 total. Its Table 1 totals are 168 TP, 15 FN and 2 FP, giving 91.8% recall and
98.8% precision for the named branches assessed. Its reported runtime below
30 seconds excludes disk reading and uses an Intel i7-4790 with 16 GB RAM.
These figures cannot be used as this project's accuracy or organizer-machine
runtime.

Danilov's segmentation examples and graph reconstructions do not establish
complete direct-daughter recall for this challenge. Segmentation quality,
named-artery identification and ostium-instance discovery measure different
outcomes.

## Exact experimental adaptations

### `border-cleaning`

`clean_wall_layers` performs the descending-layer recurrence described by
Danilov. It preserves the supplied parent and the terminal/outward support,
and retains a near-wall support voxel only if the next layer has a retained
26-neighbor. Radius and skeleton junctions are recomputed after cleanup.

We explicitly choose Chebyshev layers on the existing isotropic working grid.
The experimental depth is `ceil(5 mm / spacing)`, rather than copying 15 voxels.
This is an engineering hypothesis, not a paper-validated threshold.
The parameter is a number of scaled grid steps, **not Euclidean wall clearance
or 5 mm of centerline length**. A long artery running parallel to the wall may
never reach the terminal layer. Tests demonstrate that failure mechanism.

The paper's IDT/CHT parent segmentation is not reimplemented: the parent mask
is supplied. Full distal-tree reconstruction and its EM lipid-droplet methods
do not fit the input/output contract.

### `contact-growth`

The experiment uses the existing scan-relative blood window and an expansion
restricted to 16 mm Euclidean distance outside the supplied parent, matching
the current detector's support budget. Binary propagation can reach only
intensity-supported voxels connected to the parent. Crop-cap exclusion remains
active. Connected parent-adjacent contact areas supply roots at maxima of the
expanded-lumen distance transform.

The existing resolver still checks physical connection, path length, crop
boundaries, radius, tubularity, duplicate openings and downstream bifurcations.
The new roots and support can change the proposal pool; this is not a
candidate-only classifier.

Differences from Tahoces are explicit: its first expansion limit is 20 mm,
followed by a larger second pass excluding wrong ramifications. We use one
bounded pass for the 10 mm proximal task, the existing blood calibration,
and the existing rejection/tracing rules. We do not claim to reproduce its
probability model, two-pass whole-branch segmentation or exact contact
distance field. Parent-centerline projection is not substituted for the
required wall ostium.

### `pca-direction`

The implementation follows Riffaud's anchored matrix and sign convention.
It removes repeated path points, samples uniformly in physical arc length
at at most 0.5 mm intervals, and fits through `min(3r, 10 mm, available path)`.
Unlike an ordinary mean-centered PCA, the fit is anchored at the ostium.
Degenerate eigenvectors or outward orientation fall back to the original
ostium-to-seed direction.

Here `r` is the existing **seed** radius, whereas the paper uses the radius at
the branch origin. Uniform sampling and the 10 mm cap are further adaptations.
We retain the existing ostium, 5 mm seed, proximal path and radius estimate,
and update the direction-dependent parent-angle feature.

No anatomical naming, prescribed left/right angles, inferior-branch exclusions,
fixed artery counts or iliac-presence requirement is added.

## Development comparison and decision

These settings were compared on the already inspected 13-family seed-4001
bundle: **13 synthetic cases, 32 eligible references**, manifest SHA-256
`49ad6220758c7a07d8db836ffeeb83a4ed9a3eb329d0c96dd04a819ae297a00f`.
This old bundle predates the fourteenth wall-parallel family; the unit suite
separately tests the border-cleaning wall-parallel failure mechanism.
It is a development/regression cohort, not an independent or clinical test.

At the local 3 mm one-to-one ostium tolerance:

| Method | TP | FP | FN | F1 | Mean matched direction error |
|---|---:|---:|---:|---:|---:|
| Strict production | 29 | 0 | 3 | 0.9508 | 11.06° |
| Contact growth | 25 | 0 | 7 | 0.8772 | 8.50° |
| Border cleaning | 24 | 1 | 8 | 0.8421 | 12.79° |
| Proximal PCA direction | 29 | 0 | 3 | 0.9508 | 11.63° |

**None passes a promotion gate.** Contact growth loses four true branches.
Border cleaning loses five and introduces a false positive. PCA preserves
the instances but increases mean and 95th-percentile direction error
(20.93° to 23.15° at the 95th percentile).

Contact growth's lower matched direction/ostium means use a smaller,
different set of surviving matches and cannot establish an overall quality
improvement. All variants remain opt-in research code. No settings were
tuned after inspecting these results and no sealed tree/CNN cases were used.

The [frozen machine-readable report](frozen/paper-methods-4001.json) preserves
all predictions, per-case/family metrics,
2/3/5 mm tolerance results, source and input hashes, and runtimes. The small
synthetic cases take roughly one second per method on this Linux VM; that
measurement is not a real-CT benchmark or a Windows 4-core/8-GB certification.

## Reproduce or run an individual experiment

Base dependencies are sufficient; no model download, GPU or network call is
used at inference. Run from an environment installed with `requirements.txt`.

```bash
python paper_methods.py run --image image.nii.gz --aorta-mask aorta_mask.nii.gz \
  --method contact-growth --case-id subject001 --output outputs/contact.json \
  --diagnostics outputs/contact-diagnostics.json
```

Choose `border-cleaning` or `pca-direction` explicitly for the other experiments.
The CLI rejects existing output files. The required production command remains:

```bash
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
```

To run all three methods against strict production on a manifest-backed
analytic bundle:

```bash
python paper_methods.py benchmark --data-root outputs/robust-dev \
  --output outputs/papers-dev-4001.json
python -m ruff check .
python -m mypy paper_methods.py
python -m pytest -q tests/test_paper_methods.py
```

Use the existing archived 13-family manifest to reproduce the table exactly.
To generate the same family selection in a fresh directory:

```bash
python stress.py --output-dir outputs/papers-4001-input --seed 4001 --families \
  tortuous_low_contrast thick_slice_anisotropic touching_vein_and_calcification \
  aneurysm_with_mural_thrombus common_trunk_early_split daughter_of_daughter \
  nearby_pair_with_short_stub cropped_short_segment imperfect_parent_mask \
  dense_branch_field negative_controls_only high_noise_small_branches curved_daughters
```

Verify its manifest/input hashes before comparing; generator or dependency
changes can alter voxelization. Omitting `--families` now includes 14 families,
so totals differ.
Organizer expert references should first score frozen strict predictions;
they must not be used to retrospectively select these variants and then be
described as an untouched test.
