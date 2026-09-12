# Robustness protocol, frozen before variant selection

The five released phantoms remain training/development data. This stress suite
is also procedural, not clinical anatomy or the organizer's hidden evaluation.
No synthetic score establishes clinical accuracy.

## Evaluation audit

The evaluator uses maximum-cardinality, minimum-distance one-to-one ostium
assignment. Its invalid-edge cost is `(min(predictions, references) + 1) * tolerance`;
one extra valid assignment saves more than every valid distance can cost.
Duplicate predictions count as false positives. Unmatched IDs are exposed.
Empty-case precision/recall are undefined, not 100%; negative-control false
positives are reported separately. There is no ignored-prediction exemption:
predictions at ineligible short stubs also count as false positives.

Discovery uses 3 mm tolerance, with 2 and 5 mm sensitivity from the same saved
predictions. Geometric errors (ostium, seed, radius and direction) apply only
to matched pairs and must always accompany TP/FP/FN and recall. These rules
are a local proxy; organizer tolerances, minimum eligible size and instance
quality scoring have not been supplied.

## Dataset freeze

`stress.py` uses NumPy SeedSequence with the fixed family index, independent
of Python's randomized hash seed. Analytic references precede noise and
voxelization and never use detector predictions.

- Development: all 13 families, seed 4001.
- Frozen evaluation: all 13 families, seeds 90817 and 112213 (26 cases).
- Do not inspect frozen detector results until the selected implementation and
  its complete configuration have been committed.
- Run the baseline and selected implementation once each on those frozen cases.
  Disclose failures; do not tune and rerun on this set.
- If later changes are needed, label this set development and freeze a new set.

Families include tortuous/tapered parents, curved daughters, low contrast,
2.5 mm slices, touching veins, calcification, aneurysm/thrombus, common trunks,
downstream daughters, nearby ostia, ineligible stubs, short cropped masks,
mask perturbations, dense branches, negative controls and high-noise small
branches. Negative structures and references have independent provenance.
Mask perturbations deliberately disagree with the anatomical wall reference.
The voxelized lumens are still simplified, with no patient-derived texture or
expert adjudication of visibility.

The exploratory pre-protocol `stress-dev` run used a nondeterministic generator
and is discarded; it is not a baseline for any claimed improvement.

## Alternatives and selection

Start with four defensible approaches on development only:

1. Strict production baseline at 1 mm.
2. Finer 0.75 mm sampling, testing geometric resolution against CPU cost.
3. Relaxed blood contrast threshold, testing partial-volume sensitivity.
4. Loose human-review proposal profile, measuring its recall/false-positive
   tradeoff; it is not silently promoted to submission inference.

An additional focused fix may be derived from development failures, but it must
retain genuine aorta connection and respect the physical geometry contract.
Select using branch-level micro F1 at 3 mm. Prefer the baseline if gains require
more false positives or regress the existing topology/geometry tests. Report all
variants and runtime, not only the winner. Freeze the final choice in a commit
before opening frozen results.

## Reproduction

```bash
python stress.py --output-dir outputs/robust-dev --seed 4001
python benchmark.py --data-root outputs/robust-dev --output outputs/robust-strict.json --variant strict
python benchmark.py --data-root outputs/robust-dev --output outputs/robust-finer.json --variant finer
python benchmark.py --data-root outputs/robust-dev --output outputs/robust-relaxed.json --variant relaxed
python benchmark.py --data-root outputs/robust-dev --output outputs/robust-review.json --variant review
```

Generator directories and benchmark reports refuse overwrite. Reports include
input hashes, source hashes, Git revision and working-tree status. Benchmark
checks input hashes before inference. Hold thread limits at four throughout.
Use fresh per-case processes under Linux `/usr/bin/time -v` for peak resident
memory; Python allocation tracing is not a substitute for native memory and
should not distort inference timing.

## Release checks

Run Python tests, Ruff, mypy, network-denied algorithm/training tests, frontend
tests/lint/build, the five development phantoms, all 25 real scans, and the
required CLI under four-core CPU affinity. Validate every saved challenge JSON.
Real-scan execution measures resources and robustness to inputs, not detection
accuracy without complete daughter annotations. Preserve the presentation's
historical development claims, and add a dated stress-results supplement rather
than replacing them with an unqualified accuracy number.

## Selection record — before frozen inference

Baseline revision: `406ab01`. Development seed 4001 has 32 reference daughters.

| Variant | TP | FP | FN | F1 | Mean detector seconds |
|---|---:|---:|---:|---:|---:|
| Strict baseline | 21 | 0 | 11 | 0.7925 | 1.00 |
| 0.75 mm sampling | 22 | 0 | 10 | 0.8148 | 1.82 |
| Relaxed HU threshold | 28 | 0 | 4 | 0.9333 | 1.14 |
| Loose review proposals (without strict-pool union) | 28 | 2 | 4 | 0.9032 | 1.08 |
| Half-contrast support | 28 | 0 | 4 | 0.9333 | 1.09 |

Selected: half-contrast support at 1 mm. It ties the relaxed HU threshold on
discovery and uses the measured blood/background contrast rather than a
fixed HU decrement. Threshold = the midpoint between background and blood
(subject to the existing 30 HU lower bound). Strict wall connectivity,
tubularity, radius, topology and cap filters remain required. The original
five development phantoms still have 10 TP / 0 FP / 0 FN. The full Python
suite has 69 passed and the expected one network-sandbox skip; frontend
lint/types/build and six review tests passed before selection.

Four development misses remain: two thick-slice small branches and two
high-noise small branches, all rejected for lacking a supported distal path.
The procedural references describe analytic geometry; expert judgement of
5 mm visibility after partial-volume loss is unavailable. They remain misses
in the report, not ignored predictions or removed cases.

Frozen manifest SHA-256 values:

- Development 4001: `49ad6220758c7a07d8db836ffeeb83a4ed9a3eb329d0c96dd04a819ae297a00f`
- Evaluation 90817: `f4ba6252877b940c82578a4ba442a77d04cea1d47a0c5c4bb5a4787e6d8f1a82`
- Evaluation 112213: `fe5f96ae9656e062a5080b801470d42122557a687b672abd69b5087f1b0b7e7e`

Both evaluation datasets were generated before selection. Neither detector's
evaluation predictions were opened before this decision was committed.
