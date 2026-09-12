# Submission audit — 2026-09-12

## Verdict

The implementation and local evidence cover the Branchseed prototype
deliverables. The detector is a measured prototype with known errors, not a
perfect or clinically validated model. Hidden-test performance and final
organizer scoring remain unknown.

Sources: the supplied **Branchseed challenge.pdf** and **Hacker Guide – WIP.pdf**.
The detailed selection and failure record is in [ROBUSTNESS_PROTOCOL.md](ROBUSTNESS_PROTOCOL.md).

## Challenge contract and evidence

| Requirement | Implementation / verification |
|---|---|
| New CT + binary parent mask, no manual points | `run.py`; all 25 supplied pairs processed without per-case edits; invalid/mismatched geometry tests |
| Exact required command | `python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json` |
| Dependency file and one setup command | Pinned `requirements.txt`; Python 3.13.3 setup command in README |
| Physical SimpleITK coordinates | Native origin/direction/spacing retained; rotated, reflected, anisotropic and shifted-geometry tests |
| Unique IDs and aorta parent linkage | JSON validator checks every exported real and frozen prediction |
| Seed about 5 mm along path, radius in mm, unit direction | Proximal path interpolation and local cross-section measurement; geometry regressions and matched-error reporting |
| Variable number of daughters, including none | No fixed anatomical vessel list; negative controls and empty-case outputs |
| Crop caps, downstream daughters and distractors excluded | Explicit topology/connectivity filters; synthetic and stress families; measured failures retained |
| Nearby ostia separate; common trunk one origin | Dedicated topology regressions and procedural families |
| CPU-only, offline | Network syscalls denied during 54 passing inference/training tests; no runtime API or model download |
| Four CPU cores, 8 GB target | 25 fresh CLI processes with four-core affinity, four threads and 8 GiB process limit; maximum 23.35 s elapsed and 1.43 GiB RSS |
| Development predictions | All 25 real JSON outputs plus five analytic training cases in delivered evidence bundles |
| Three visual checks | Subjects 001/002/003: CT, parent outline, predicted ostia and direction arrows; included in presentation and evidence |
| Useful visual interface | Local Explorer: 3D, linked CT, wall map, tour and JSON export; separate candidate-review workflow |
| Five-minute demo | Eight slides, four 75-second speaking slots; 60-second actual Explorer film inside Speaker 3's slot |
| Team handoff | [TEAMMATE_GUIDE.md](TEAMMATE_GUIDE.md), README, review workflow and presentation script |

The brief leaves minimum eligible origin size and final runtime limit to the
organizers. The local 3 mm matching threshold is a disclosed proxy, not an
official tolerance. Terminal iliac division is an optional extension and is
not claimed as a separately validated feature.

## Release checks

- Full Python suite: **77 passed, 1 skipped**. The skip is the test requiring
  an externally enforced network sandbox; it passed in the denied-network run.
- Network-denied algorithm/training subset: **54 passed**; an explicit socket
  attempt returned injected `ENETUNREACH`.
- Python Ruff and mypy: passed.
- Frontend: six tests, lint, build and typecheck passed.
- Review/Explorer subset after merging teammate slab views: **11 passed**.
- Five original analytic training cases: **10 TP / 0 FP / 0 FN**. These are
  development data, not an independent estimate.
- Frozen 26-case stress evaluation: **56 TP / 1 FP / 8 FN**, micro F1 **0.9256**.
- All 25 real-input CLI runs: completed; physical-output JSON validation passed.

Non-failing build warnings: the bundled frontend JavaScript exceeds Vite's
500 kB advisory threshold; scikit-image emits NumPy deprecation warnings.
Neither prevented the tests or build.

The final recorded browser retest is documented in its delivery report.
Earlier presentation validation is in the delivered kit's `VALIDATION.md`.
Review tests use isolated labels; synthetic/test verdicts are never committed
as human ground truth. Confirm/reject labels assess proposed candidates only;
they cannot identify a missed artery.

## Presentation evidence

The existing film and slide metrics are historical evidence from the earlier
detector revision and are explicitly dated in `presentation/README.md`.
[The robustness supplement](presentation/ROBUSTNESS_UPDATE.md) provides the
newer results without rewriting that history or extending the five-minute slot.
The HTML kit works with local assets and no external HTTP access.

The MP4 is a silent 60-second product film intended for live narration within
the deck. It is not a five-minute narrated upload. The brief requests a
five-minute demonstration; any additional video-upload rules must be checked
on Devpost or with organizers.

## Team actions before judging

1. Register the 1–4 person team and represented school using the guide's
   [team registration form](https://forms.gle/8jDCNgBJJcedAVai6).
2. Submit through [Devpost](https://battle-of-the-schools.devpost.com/) before
   **Sunday, September 13, 11:00 AM**, as stated in the venue schedule. Check
   organizer updates because the supplied guide is marked WIP. No submission
   has been made on the team's behalf.
3. Confirm any published minimum origin size, final time limit and upload
   fields. The 60-second average target is provisional.
4. Copy dependencies, built assets, scans, predictions and extracted
   presentation kit to the demo laptop before going offline. A source clone
   alone does not contain generated presentation media or downloaded LFS data.
5. Assign Speaker 1–4 to actual teammates and rehearse the handoffs. Test the
   film on the actual laptop; use offline HTML as the primary format.
   Native PowerPoint video, physical touch and mobile fullscreen remain untested.
6. Be present for judging (scheduled 12:30–2:30 PM) and ready for finalist
   presentations (4:00–4:30 PM). A teammate must attend prize collection.

Independent, complete expert daughter references are the next accuracy step.
Candidate review alone, synthetic ground truth, plausible screenshots and
successful execution cannot establish real-scan precision or recall.
