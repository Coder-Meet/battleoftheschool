# Workspace index

**Current operation:** [README.md](README.md), [PRODUCTION_WORKFLOW.md](PRODUCTION_WORKFLOW.md), [DEMO_GUIDE.md](DEMO_GUIDE.md), [FINAL_HANDOFF.md](FINAL_HANDOFF.md).

**Current fusion evidence:** [validation receipt](docs/fusion-restored-validation.json), `outputs/fusion-restored-20260913/`, and the bundled model in `models/production-v1/`.

**Frozen strict evidence and next work:** [FINAL_EVALUATION_RESULTS.md](FINAL_EVALUATION_RESULTS.md), [ACCURACY_RECHECK.md](ACCURACY_RECHECK.md), [CURRENT_E2E_REVIEW.md](CURRENT_E2E_REVIEW.md), [FINAL_EVALUATION_PROTOCOL.md](FINAL_EVALUATION_PROTOCOL.md).

**Development and annotation:** [TEAMMATE_GUIDE.md](TEAMMATE_GUIDE.md), [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md), [EXTERNAL_DATA_SOURCES.md](EXTERNAL_DATA_SOURCES.md). External data access and older annotation procedures require their stated provenance checks.

**Presentation:** [presentation/README.md](presentation/README.md), [presentation/LIVE_DEMO_CUES.md](presentation/LIVE_DEMO_CUES.md), [presentation/SHOWCASE.md](presentation/SHOWCASE.md). The showcase is separate from the live talk.

| Data location | Meaning / treatment |
|---|---|
| `data/` | Original 25 CT/parent-mask pairs; retained |
| `eval/` | Original released annotation package; byte identity preserved |
| `labels/organizer-v1/` | Raw/normalized released references and provenance; retained |
| `labels/reviews.json`, splits, verdicts | Historical AI candidate reviews; retain exact identities and provenance |
| `labels/final-eval/` | Frozen scored predictions, receipts and selection; preserve paths and hashes |
| `labels/finalization/` | Current release and accuracy-recheck evidence; preserve |
| `labels/research/`, `labels/synthetic/`, `frozen/` | Source-bound research artifacts and regression evidence; preserve |
| `outputs/archive/pre-release-2a498a4/` | Earlier local runs; historical, source-specific, not current candidate caches |
| `docs/archive/` | Superseded plans/handoffs and fusion trial documents; historical instructions |

Do not infer that an artifact is disposable because it is old: frozen evidence is needed to reproduce the release decision. Use a new output directory for new experiments. Large CTs and labels are intentionally retained, including the byte-identical release copies required by integrity checks.

The cleanup manifest at [docs/workspace-cleanup-manifest.json](docs/workspace-cleanup-manifest.json) records archived paths and removed cache bytes. No production algorithm, threshold, reference verdict or trained weight changed during cleanup.

Published strict-release communications remain available in [DEVPOST_SUBMISSION.md](DEVPOST_SUBMISSION.md), [PRESENTER_BRIEFING.md](PRESENTER_BRIEFING.md) and [docs/media](docs/media/README.md). Their figures must be refreshed before presenting the restored fusion default. The shared Drive showcase link is retained in the README.
