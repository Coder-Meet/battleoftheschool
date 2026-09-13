# Current presenter cheatsheet

Use [DEMO_GUIDE.md](DEMO_GUIDE.md) for exact setup and [presentation/LIVE_DEMO_CUES.md](presentation/LIVE_DEMO_CUES.md) for the talk.

**Pitch:** Given a CT and a parent-aorta mask, Branchseed looks for supported vessel openings at the wall, traces proximal paths, and returns separate daughter instances with physical landmarks. The Explorer links those predictions to the CT for inspection.

| Question | Current answer |
|---|---|
| What runs? | Strict + review proposals at scale 0.9, score before 3 mm merge |
| Does it use a model? | Bundled synthetic logistic, 13 features, threshold 0.15 |
| What is the minimum size? | 2 mm origin diameter with the existing native-voxel uncertainty policy; seed-radius minimum is separately 0.7 mm |
| Where is the seed? | 5 mm along the estimated proximal path |
| How far is the trace? | Up to 10 mm or an estimated downstream bifurcation |
| Reference result? | 14 TP / 4 FP / 5 FN; F1 0.75676, count MAE 1.8 at local 3 mm tolerance |
| Are those independent test results? | No: 19 judge-approved, AI-assisted/non-exhaustive targets on five reused cases |
| Synthetic result? | Current 24-case fusion: 47/11/2, F1 0.8785, including 4 negative-control FPs. Previous strict: 46/0/3, F1 0.9684 |
| Why not recovery by default? | Its fusion combination is unvalidated; retained only with explicit strict mode |
| Runtime evidence? | Fusion offline Linux 25-case run: 11.60 s average, 52.67 s maximum; sampled process-tree peak 1499 MiB. Four cores; organizer-Windows timing remains open |
| What should run on stage? | Normal Explorer, live at slide 5; the product showcase is separate |

```bash
python explorer.py --data-root data --port 8000
python run.py --image image.nii.gz --aorta-mask mask.nii.gz --output prediction.json
```

Use no review/model/experimental flag for the current fusion demo. Use the current fusion presentation and script from the [submission kit](https://github.com/Coder-Meet/battleoftheschool/releases/tag/branchseed-submission-fusion-2026-09-13). Do not call heuristic scores probabilities, proxy scores official challenge accuracy, unknown radii zero, or unmatched predictions clinically proven false positives. See [PRODUCTION_WORKFLOW.md](PRODUCTION_WORKFLOW.md) for the current evidence and [DEMO_GUIDE.md](DEMO_GUIDE.md) for outstanding submission actions.
