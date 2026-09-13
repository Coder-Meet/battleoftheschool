# Steven review analytics, frozen 13 September 2026

Read [`STEVEN_LABEL_ANALYTICS.md`](../../../STEVEN_LABEL_ANALYTICS.md) for interpretation.
These are candidate-level AI-review comparisons, not expert-reference accuracy.

- `summary.json`: source/review/model hashes and primary aggregates.
- `additional-summary.json`: provenance, count changes and the explicitly separate
  all-field `1e-8` roundoff sensitivity analysis.
- `feature-statistics.csv`: label-conditioned distributions, including training data.
- `case-counts.csv`: historical labels and current strict/pool counts for all 25 cases.
- `historical-disagreements.csv`: frozen logistic disagreements on historical vectors.
- `current-candidates.csv`: current scores, exact old labels where available and
  nearest-origin navigation aids. Blank labels are unknown, not negative.
- `adjudication-queue.csv`: historical candidates needing later CT/reference review.

The raw inference replay used `e8303e5`. Its source hash in the summary describes
that execution; the later chart-label and missing-directory checks do not change
the recorded detector or model decisions.

The shareable handoff bundle also contains both editable documents, the combined
PDF, all unfiltered prediction JSONs and diagnostics, source charts and the
supplementary table-generation script. No CT volumes are included in that bundle.

To repeat the primary analysis from a prepared repository:

```bash
python review_analytics.py \
  --tree-model labels/research/current-source-v1/models/gradient_boosting-base.json \
  --output-dir outputs/fresh-review-analysis
```

Use the source/review/model hashes to check compatibility before comparing a new
run with this archive. Original test cases have already been inspected.
