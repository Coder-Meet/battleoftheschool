# Five difficult cases: annotation head start

The **real-case packet contains unreviewed model proposals, not ground truth**.
Nothing is confirmed automatically, including predictions shared by both
variants. They share a detector and can share the same errors. The separate
**synthetic bundle has analytic ground truth** and can be used immediately for
development, with the limitations below.

## Real cases selected before seeing organizer labels

| Case | Why it needs review | Question for the organizer |
|---|---|---|
| subject005 | Many openings; wall-parallel variant folds two production openings into existing paths | Are these independent ostia, shared trunks, or downstream branches? |
| subject017 | Production reports zero; experimental variant proposes a wall-adjacent opening | Is any eligible direct daughter visible? Does the proposed structure connect to the supplied parent? |
| subject018 | Coarse 1.5 mm sampling, low contrast, 16 production proposals | Which small structures are arteries rather than tissue, veins or artifacts? What is the minimum eligible size? |
| subject023 | Another short parent segment with zero production output and an experimental proposal | Is zero correct? Trace possible missed posterior branches and check the crop-cap rule. |
| subject025 | Coarse sampling, many proposals and a variant difference near a mask island | Which mask component is the intended parent? Is the extra opening eligible? |

These are deliberate difficult development cases, not a random sample or an
estimate of performance across patients.

## Review each real case

1. Open `01_blinded_survey.pdf` **before** the proposed-opening PDF. It contains
   every native k slice spanning the parent-mask bounding box plus 20 mm of
   context, with no model markers or mask overlay. Note possible openings
   independently. A survey PDF helps navigate; full-volume tracing in the
   original CT remains necessary.
2. Open `02_proposed_openings.pdf`. Each page shows one unreviewed proposal in
   three acquisition planes, with five consecutive native slices per plane.
   There is no interpolated super-resolution. Yellow marks the proposed ostium,
   magenta the proposed seed and cyan the parent boundary. Points are drawn
   only in their intersecting slice, not projected onto unrelated slices.
3. Open the original CT and supplied parent mask in the Explorer or a local
   medical-image viewer. Use the physical coordinates or continuous indices
   from `proposals.json` to navigate. Confirm an actual lumen connection at the
   opening and trace the branch, rather than accepting a bright spot on a MIP.
4. Fill `review_sheet.csv`: status `confirmed`, `rejected` or `uncertain`;
   reviewer; notes; and independently checked/corrected measurements. Suggested
   rejection reasons: no lumen continuity, vein/tissue, calcification, crop cap,
   daughter-of-daughter, duplicate/common trunk, insufficient visible length.
   Keep closely spaced but independent ostia separate.
5. Add every missed opening to `missed_candidates_to_add` in `proposals.json`,
   with measured ostium, seed, radius, direction, reviewer and notes. A candidate
   confirmation workflow alone cannot measure false negatives.
6. Complete `case_review`: reviewer, eligibility policy, eligible daughter
   count, unresolved issues, and whether the **entire supplied parent** was
   inspected. A zero-proposal packet is not a verified negative case.
7. Have the organizer/expert adjudicate uncertain structures and measurements.
   Only then write a separate challenge-format `reference.json`. Keep the
   original proposals intact and retain who reviewed each reference. The
   reference scorer rejects these unreviewed packets.

Physical coordinates are SimpleITK LPS millimetres. Indices are zero-based
continuous **xyz**, while NumPy arrays use **zyx**. PDF axes are acquisition
**i/j/k**, not guaranteed anatomical axial/coronal/sagittal planes on rotated
scans. The image origin, spacing and full direction matrix are recorded; use
them rather than multiplying indices by spacing alone.

Measure the seed about 5 mm **along** the daughter lumen, not 5 mm radial
clearance from the aorta. Review direction and radius independently; confirming
that a vessel exists does not validate its model-generated measurements.
Check a proximal path up to 10 mm or the first downstream bifurcation. A common
trunk is one direct daughter. Confirm minimum vessel size and crop policy with
the organizer before resolving borderline candidates.

No CT volumes are duplicated in the real review archive; use the five existing
supplied scans. The two frozen variants are proposals only. Their exact duplicate
geometries share a row; any changed geometry or nearby opening remains separate
for the reviewer to resolve. No machine-generated branch count is prefilled as
the reference count.

## Reproduce the real-case packet

From the repository root, with its dependencies installed:

```bash
python prepare_annotations.py \
  --predictions frozen/2a40d10-production frozen/2a40d10-wall-parallel \
  --cases subject005 subject017 subject018 subject023 subject025 \
  --output-dir outputs/hard-real-review
```

The output includes PDFs, editable CSV/JSON worksheets, source hashes and an
integrity manifest. Existing output directories are never overwritten. Keep
both frozen variants unchanged until the independently reviewed labels arrive.

## Five harder synthetic ground-truth cases

Generate one case in each selected family with the fixed, new development seed:

```bash
python stress.py --seed 731927 --output-dir outputs/hard-synthetic-731927 \
  --families tortuous_low_contrast thick_slice_anisotropic \
  touching_vein_and_calcification common_trunk_early_split wall_parallel_descending
python benchmark.py --data-root outputs/hard-synthetic-731927 \
  --variant native-contrast --output outputs/hard-synthetic-baseline.json
```

Each case includes a CT, parent mask, analytic daughter `reference.json`, full
proximal centerline provenance, negative-structure metadata and file hashes.
The labels are determined by geometry **before running the detector**.
The seed is at 5 mm along the analytic centerline. Common trunks appear once;
touching veins, calcification and bone are not reference daughters.

These are **development/training phantoms**, not five real annotated scans.
The existing generator specifies stress-testing provenance; this newly shared
seed must be treated as development data. If used to train or tune anything,
do not reuse its score as independent validation. Keep the existing frozen
stress sets and future organizer references separate. The reference geometry
precedes voxelization/noise, so some small simulated branches can be hard to
resolve in the sampled image. Ground truth is exact for the generator's
intended geometry, not guaranteed observability or clinical realism.

## Candidate labels from analytic references

Export training features from the same strict/review union used by the
candidate labelling workflow:

```bash
python synthetic_reviews.py --data-root outputs/hard-synthetic-731927 \
  --output outputs/hard-synthetic-reviews.json
```

The exporter checks manifest hashes and accepts only analytic synthetic
bundles. A one-to-one ostium match within 3 mm supplies a positive example.
Unmatched proposals within 5 mm of a reference are ambiguous and excluded
from training, including duplicate proposals. Farther unmatched proposals
are negative examples. These local thresholds are not the organizer's scoring
policy. Missed references and all false positives remain in the case report;
ambiguity exclusions apply only to classifier training.

The result uses the `learning.py` candidate-review schema and records
`labeller=analytic_synthetic_geometry`. It can be supplied alongside separate
AI review files to `learning.py split` and `learning.py train`. Keep complete
generation seeds and patients in one partition, and inspect provenance counts
in the training report. Neither source establishes real accuracy. A
positive-only bundle cannot by itself fit the two-class classifier.

Evaluate a trained model end to end, rather than relying on reviewed-candidate
accuracy:

```bash
python benchmark.py --data-root outputs/independent-synthetic \
  --candidate-model outputs/candidate-model.json \
  --output outputs/independent-model-results.json
```

The report preserves unfiltered predictions, model hash, threshold, split
overlap and full TP/FP/FN, including every reference removed by the model.
For a separate experiment with the broader candidate pool, add
`--candidate-pool --variant review`. This does not change the production CLI
or promote the pool to default inference.
