# Submission audit — 2026-09-12

## Verdict

The implementation and local evidence cover the Branchseed prototype
deliverables. The detector is a measured prototype with known errors, not a
perfect or clinically validated model. Hidden-test performance and final
organizer scoring remain unknown.

Sources: the supplied **Branchseed challenge.pdf** and **Hacker Guide – WIP.pdf**.
The detailed selection and failure record is in [ROBUSTNESS_PROTOCOL.md](ROBUSTNESS_PROTOCOL.md).
The [native-contrast supplement](#native-contrast-release-supplement--09f753b)
records the latest results and the incomplete final browser checks.

## Recheck against the organizer's latest brief

The organizer has confirmed **Windows, four CPU cores, 8 GB RAM, no GPU,
offline evaluation**. Previous CPU/RAM measurements below were made on
**Linux**, not Windows. The README now has Windows setup, the exact required run
command and offline wheel installation. CI includes a native Windows Python
suite and offline package installation; its status must be checked before
claiming a Windows pass. Actual Windows 8 GB runtime is still unmeasured.
Windows x64 wheels do not establish Windows ARM support; CPU architecture
remains to be confirmed.

**Development-set predictions mean our algorithm's outputs for the supplied
development inputs.** They are not ground truth, do not require annotations to
generate, and are separate from the hidden test set. Submit all 25 frozen
production JSONs in `frozen/2a40d10-production/` unless the organizer explicitly
names a smaller required subset. Do not substitute the experimental
wall-parallel outputs, synthetic references or unreviewed proposal worksheets.
The five announced example references are for checking predictions, not a
reason to omit the other supplied cases.

### Current status

| Item | Status and remaining gap |
|---|---|
| Source and pinned dependency file | Present; CPU inference uses `requirements.txt` |
| One setup command, exact `python run.py ...` command | Documented for Windows and Unix; setup precedes offline evaluation |
| One prediction per development case | 25 frozen production JSONs; 151 proposed daughters, including three empty cases |
| At least three visual checks | Subjects001/002/003 regenerated from those same frozen production JSONs and packaged with all 25 outputs |
| Physical coordinates, variable counts and parent/instance IDs | Validated by automated tests and all frozen JSONs |
| Five-minute demo | Existing eight-slide deck, four 75-second roles, film inside that time; native laptop playback still needs checking |
| Clinician-useful display | Explorer exists; Subject018's final 3D/selection/export interaction remains unverified |
| Hidden cases without manual edits | Required CLI is generic; hidden-case accuracy cannot be guaranteed |
| 5 mm daughter path, up to 10 mm or first downstream split | Implemented for accepted traces; strict candidate search can still miss wall-parallel branches, with the opt-in alternative deliberately unpromoted |
| Caps, common trunks, adjacent ostia and daughter-of-daughter | Regression coverage exists; exact real-case eligibility needs expert adjudication |
| Minimum eligible origin size | Not supplied; current minimum-radius setting is not an organizer-confirmed size rule |
| Runtime | Initial target is average ≤60 s/case; final limit and native Windows measurement outstanding |
| Ground truth | Five new hard synthetic development cases have 12 analytic daughters; five real-case packets remain unreviewed |

The refreshed Python suite passed **100 tests with one expected network-sandbox
skip** after adding review-packet and scorer-provenance tests. No detector
threshold was changed for the new synthetic seed. At a local 3 mm matching
tolerance, the new five-case development bundle scored 10 TP / 0 FP / 2 FN:
one thick-slice miss and one wall-parallel miss remain. This is synthetic
development evidence, not real-scan accuracy.

The delivered development submission bundle also includes the source,
prebuilt Explorer assets and an offline Windows x64/Python 3.13 wheelhouse.
Its package dependencies were resolved without a network index; this is not
the same as executing the detector on Windows. Native Windows checks run in
[the CI workflow](https://github.com/Coder-Meet/battleoftheschool/actions/runs/34722306805).

The [presentation release](https://github.com/Coder-Meet/battleoftheschool/releases/tag/branchseed-presentation-2026-09-12)
now hosts both the MP4 and complete offline kit. Publishing these assets does
not submit the team to Devpost.

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
| Development predictions | All 25 real JSON outputs; synthetic training/reference cases are a separate artifact |
| Three visual checks | Subjects 001/002/003: CT, parent outline, predicted ostia and direction arrows; included in presentation and evidence |
| Useful visual interface | Local Explorer: 3D, linked CT, wall map, tour and JSON export; separate candidate-review workflow |
| Five-minute demo | Eight slides, four 75-second speaking slots; 60-second actual Explorer film inside Speaker 3's slot |
| Team handoff | [TEAMMATE_GUIDE.md](TEAMMATE_GUIDE.md), README, review workflow and presentation script |

The brief leaves minimum eligible origin size and final runtime limit to the
organizers. The local 3 mm matching threshold is a disclosed proxy, not an
official tolerance. Terminal iliac division is an optional extension and is
not claimed as a separately validated feature.

## Initial release checks (before the native-contrast update)

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

## Native-contrast release supplement — `09f753b`

The teammate's contrast proposal was compared at three scales before selecting
1.2. New frozen procedural cases recovered **53/63 references with one FP**
(F1 **0.9060**), versus **51/63 with one FP** for the previous midpoint.
Both negative controls stayed empty. The more aggressive scales were rejected
because they introduced development false positives. The primary outcome is
this new frozen comparison; older sets were used only for regression checks.

- Full Python suite: **81 passed, 1 expected network-sandbox skip**.
- Enforced network-denied subset: **58 passed**, including an attempted socket.
- Ruff, mypy, frontend lint, six frontend tests, build and typecheck: passed.
- Five training phantoms: **10 TP / 0 FP / 0 FN**.
- Prior frozen sets, now regression data: **57 TP / 0 FP / 7 FN**.
- All 25 real cases: valid physical JSON, no batch failures; every prediction
  exactly matched the preselected experimental implementation.
- Real batch: mean detector time **5.191 s**, maximum **21.427 s**; total process
  elapsed **135.39 s**; maximum RSS **1,505,824 KiB (1.44 GiB)**.
- Fresh required-CLI tail checks: subject018 **22.74 s / 1,495,688 KiB**;
  subject025 **22.47 s / 1,456,464 KiB**. Four-core affinity and an 8 GiB
  address-space limit were enforced.

### Final browser status

Review history freshness, stale-label counts, empty cases, persistence,
save failures, export validation, and modified keyboard shortcuts passed.
Subjects001/002/003 also passed the recovered 3D/linked-CT regression on the
unchanged frontend. A fresh backend at `09f753b` passed Subject001 and showed
Subject018's 16 candidates and three CT planes.

**The final Subject018 3D/changed-selection/browser-download checks remain
untested:** Chrome stopped responding; the permitted fresh-tab recovery failed.
Earlier investigation observed SwiftShader GPU crashes, but the final stall
did not increment the crash count, so its cause is unproven. The shell/backend
and automated API tests remained operational. This is reported as an
environment blocker, not silently counted as a browser pass. The intended demo
laptop must still complete these checks.

### Remaining teammate dependency

The supplied fix note says the stray-mask-island `cap_mask` change and
`test_a_stray_mask_island_does_not_blank_the_surrounding_wall` already exist.
Neither was present in fetched main during this investigation. The teammate
was asked to push or identify that commit; it was not recreated or marked
verified. Do not present this change as included until it is integrated and tested.
