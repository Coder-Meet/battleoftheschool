# Steven's candidate-label analytics

**13 September 2026 • frozen models • current detector replay • no model promotion**

## 1. Findings to act on

The original logistic filter removes **9 of the 92 AI-confirmed historical candidates**. The synthetic-augmented logistic filter removes eight. Their decisions are identical on the original 68-row test partition: both retain 20 confirmed and two rejected candidates, and remove one confirmed and 45 rejected candidates.

The augmented model's small overall agreement gain occurs on already exposed training/validation cases. It has not established better real-scan detection accuracy.

The **synthetic tree disagrees substantially with the exact current review-pool labels**: it removes 20 of 67 confirmed candidates and retains 36 of 87 rejected candidates. Both logistic models remove four confirmed and retain six rejected candidates on that same subset. This is another reason to keep the tree optional while awaiting expert references.

**Seven of the original model's nine confirmed removals occur at 1.5 mm native spacing.** This is a useful adjudication priority. It is not evidence that all seven are genuine vessels or that voxel spacing alone caused the decisions.

Current predictions need explicit identity checks before reusing a label. Branch numbers are renumbered after sorting, and review-profile geometry/features can differ from strict geometry/features. Matching `subjectNNN/branch_NNN` alone can silently attach the wrong old verdict.

Recommended next step: review the confirmed-removal queue against CT or incoming complete references before making any filter the default. No labels, weights, detector settings or submission predictions were changed by this analysis.

## 2. What was audited

- **237 labelled rows**, across **23 cases**, with **92 confirmed / 145 rejected**.
- Every stored labeller is **`claude`**. These are Steven's imported AI candidate reviews.
- Original split: **13 train / five validation / five test cases**; **122 / 47 / 68 rows**.
- The augmented model uses the same real-case split plus synthetic training cases.
- All 25 supplied CT/mask cases were replayed with strict and review-union proposals.
- The selected current-source synthetic tree was scored on current vectors after source/preprocessing validation. It was not applied to historical vectors from another extraction version.
- Scores were evaluated at the frozen thresholds: original logistic **0.3763143354**, augmented logistic **0.5**, synthetic tree **0.5630076830**. No threshold search or refit was performed.

The review schema, feature dimensions, finite values, duplicate case/instance keys and complete candidate fingerprints were checked. Original model-report confusion counts were independently reproduced from the stored vectors.

The original test cases are subject005, subject013, subject018, subject021 and subject025. They have since been inspected and are now exposed development cases. Case-ID separation does not independently establish patient identity or eliminate earlier image exposure.

## 3. Historical pseudo-label agreement

“Kept” means the score exceeds that model's frozen threshold. A confirmed removal is a disagreement with the reviewer, not a verified missed artery. A rejected retention is likewise not a verified false-positive artery.

| Frozen model / partition | Rows | Confirmed kept / removed | Rejected kept / removed |
| --- | ---: | ---: | ---: |
| Original / train | 122 | 46 / 4 | 9 / 63 |
| Original / validation | 47 | 17 / 4 | 3 / 23 |
| Original / test | 68 | 20 / 1 | 2 / 45 |
| Original / all | 237 | 83 / 9 | 14 / 131 |
| Augmented / train | 122 | 47 / 3 | 7 / 65 |
| Augmented / validation | 47 | 17 / 4 | 2 / 24 |
| Augmented / test | 68 | 20 / 1 | 2 / 45 |
| Augmented / all | 237 | 84 / 8 | 11 / 134 |

Overall agreement is **90.30% original** and **91.98% augmented**, mixing training and exposed holdout data. Both reproduce **95.59% agreement**, **0.9091 candidate precision**, **0.9524 confirmed retention** and **0.9302 candidate F1** on the original test reviews. These are proposal-conditioned pseudo-label metrics, not whole-case recall.

### Candidate removals that deserve attention

These identifiers refer to the **historical cards**, not necessarily today's identically numbered branch. The review queue provides exact current IDs where available and separately marks nearby origins as unverified navigation aids.

| Historical case / candidates | Original partition | Why review |
| --- | --- | --- |
| subject001: branch_003, branch_006 | Train | Both frozen models remove both confirmed candidates; 0.8 mm spacing. |
| subject016: branch_004, branch_008 | Validation | Both remove both confirmed candidates; 1.5 mm spacing. |
| subject018: branch_024 | Test | Both remove the confirmed candidate; 1.5 mm spacing, large contact component. |
| subject020: branch_001, branch_002 | Validation | Both remove both confirmed candidates; short seed displacement and nearby origins need topology review. |
| subject022: branch_003, branch_010 | Train | Both remove branch_003; augmentation retains branch_010. |

The original model removes **7/29 confirmed candidates at 1.5 mm spacing**, versus **2/63 at finer spacing**. The augmented model removes **6/29** and **2/63** respectively. Multiple candidates belong to the same cases, so these are not independent patient observations.

Also inspect subject005/branch_013 and subject013/branch_002: both are rejected historical test candidates retained by both models. subject024 has 28 rejected historical candidates and no confirmed rows; that does not establish that the scan has no real daughters.

## 4. Feature distributions

The following are descriptive medians over all historical reviews, including training data. They reflect the AI review process and its selected proposals.

| Feature | Confirmed median | Rejected median |
| --- | ---: | ---: |
| Seed radius, mm | 2.878 | 1.279 |
| Mean vesselness | 0.7985 | 0.3070 |
| Evidence score | 0.975 | 0.734 |
| Path length, mm | 10.000 | 10.000 |
| Seed displacement, mm | 4.784 | 4.777 |
| Tortuosity | 1.068 | 1.080 |
| Relative path intensity | 0.9567 | 0.8395 |
| Bone distance, mm | 32.361 | 23.854 |
| Parent angle, degrees | 64.97 | 66.53 |
| Position along parent | 0.678 | 0.503 |
| Native spacing, mm | 0.850 | 1.500 |
| Connector gap | 0.000 | 0.000 |
| Contact volume, mm³ | 203.5 | 241.0 |

Confirmed proposals tend to have larger radii and stronger tubularity, but the distributions overlap. Both classes often reach the 10 mm tracing cap, making capped length alone a poor discriminator.

The original logistic model's largest positive standardized coefficients are mean vesselness (**+0.613**) and evidence score (**+0.537**). Native spacing (**−0.344**), contact volume (**−0.335**) and connector gap (**−0.328**) are its largest negative coefficients. These conditional model coefficients are not causal effects; correlated image properties and case composition can confound them.

Do not derive a new radius, vesselness or spacing rejection threshold from these descriptive tables. In particular, seed radius is not the independent 2 mm origin-diameter eligibility measurement.

## 5. Current proposals and label identity

| Current profile | Total proposals | Exact confirmed | Exact rejected | Changed / unlabelled |
| --- | ---: | ---: | ---: | ---: |
| Strict | 150 | 4 | 5 | 141 |
| Review-union | 304 | 67 | 87 | 150 |

Exact identity compares the entire ostium, seed, direction, radius and 13-feature vector within the same case. JSON whitespace and integer/float formatting are normalized; values are not rounded for the primary analysis.

**121 strict proposals and 117 pool proposals reuse a historical branch number with a different fingerprint.** This does not mean all their geometries changed. Labels were made on review proposals; different profiles, trace measurements, feature changes and numerical precision all affect fingerprints.

A sensitivity check accepts an absolute difference of at most **1e-8 in every stored value**, with one-to-one uniqueness checked within each case/profile. It adds eight strict and 55 pool matches. Thus 55 of the pool's primary unmatched candidates have only tiny numerical differences; they should not be described as newly discovered anatomy. The primary CSV verdicts remain unchanged.

### Agreement on exactly matched current candidates

| Profile / model | Matched rows | Confirmed kept / removed | Rejected kept / removed |
| --- | ---: | ---: | ---: |
| Strict / original | 9 | 4 / 0 | 1 / 4 |
| Strict / augmented | 9 | 4 / 0 | 1 / 4 |
| Strict / synthetic tree | 9 | 4 / 0 | 1 / 4 |
| Pool / original | 154 | 63 / 4 | 6 / 81 |
| Pool / augmented | 154 | 63 / 4 | 6 / 81 |
| Pool / synthetic tree | 154 | 47 / 20 | 36 / 51 |

The roundoff sensitivity subset contains **209 pool candidates: 77 confirmed and 132 rejected**. The original model keeps/removes **71/6 confirmed and 9/123 rejected**; augmentation gives **71/6 and 7/125**; the synthetic tree gives **53/24 and 54/78**. The tree's disagreement persists with this less brittle identity rule.

These are selected subsets, and many real cases were used to fit the logistic models. Their better AI-label agreement does not prove that they generalize better than the tree on independent expert references.

### Counts if each optional filter were enabled

| Proposal input | Unfiltered | Original logistic | Augmented logistic | Synthetic tree |
| --- | ---: | ---: | ---: | ---: |
| Strict | 150 | 98 | 93 | 116 |
| Review-union | 304 | 135 | 128 | 142 |

These are counterfactual counts, not accuracy scores. Saved strict and pool predictions are unfiltered. **82 strict and 203 pool origin-diameter measurements are unresolved**; the historical labels do not establish their 2 mm eligibility.

## 6. Case coverage

Historical labels and current proposals describe different candidate versions. Do not subtract the columns to calculate missed vessels.

| Case | Historical confirmed / rejected | Current strict / pool |
| --- | ---: | ---: |
| subject001 | 6 / 9 | 8 / 18 |
| subject002 | 4 / 0 | 4 / 6 |
| subject003 | 4 / 3 | 4 / 11 |
| subject004 | 4 / 1 | 4 / 6 |
| subject005 | 4 / 18 | 13 / 29 |
| subject006 | 5 / 7 | 4 / 17 |
| subject007 | 3 / 6 | 8 / 13 |
| subject008 | 5 / 4 | 7 / 10 |
| subject009 | 3 / 0 | 4 / 5 |
| subject010 | 5 / 6 | 9 / 15 |
| subject011 | 4 / 3 | 5 / 8 |
| subject012 | 4 / 1 | 5 / 5 |
| subject013 | 4 / 3 | 8 / 9 |
| subject014 | 4 / 1 | 4 / 5 |
| subject015 | 4 / 2 | 3 / 8 |
| subject016 | 7 / 12 | 6 / 20 |
| subject017 | No review rows | 0 / 0 |
| subject018 | 4 / 19 | 16 / 42 |
| subject019 | No review rows | 0 / 0 |
| subject020 | 2 / 1 | 1 / 2 |
| subject021 | 3 / 2 | 3 / 5 |
| subject022 | 7 / 13 | 7 / 16 |
| subject023 | 0 / 1 | 0 / 1 |
| subject024 | 0 / 28 | 11 / 25 |
| subject025 | 6 / 5 | 16 / 28 |

Zero output is not a verified negative-control result. In particular, subject017 and subject019 have no review rows or current proposals, but still require complete references to establish whether daughters were missed.

## 7. What Steven can do with the handoff

The adjudication queue contains **27 confirmed-removal and 45 rejected-retention candidates**, combining disagreements across the historical logistic models and exact current tree scores. It includes historical IDs, original partitions, coordinates, exact current IDs when available, and nearby strict candidates as navigation aids only. It does not create new verdicts.

Start with the nine historical confirmed removals above, then the tree's additional confirmed removals. Also review the all-rejected subject024 cohort and cases with empty proposal pools. Use archived cards whose fingerprints match, or current CT with the recorded coordinates; never apply old branch IDs blindly to regenerated cards.

When organizer references arrive, establish completeness, physical frame and eligibility rules; freeze outputs; then score strict, pool and optional models separately. Only complete references can establish missed-branch recall, correct daughter count and independent geometry errors.

### Reproduction and evidence

Current replay used implementation commit **`e8303e5`**, whose detector matches reviewed source `f43165b`. Source, model, review and CT/mask hashes are in the JSON/CSV evidence. Review timestamps span **12 September 2026, 22:15:50–22:35:03 UTC**. No duplicate case/instance rows were found; repeated geometric fingerprints would have caused the audit to stop.

```text
python review_analytics.py --tree-model labels/research/current-source-v1/models/gradient_boosting-base.json --output-dir outputs/fresh-review-analysis
```

The shareable bundle includes the supplementary table/queue script, raw current diagnostics, original unfiltered predictions, CSVs, charts and both review documents. Invoke the supplementary script by its file path from the repository environment; it reads the adjacent `analytics` directory.

```text
PYTHONPATH=. python outputs/steven-review/enrich_analytics.py
```

That second command is the Linux invocation used for this report; on Windows set `PYTHONPATH` to the repository root in the shell before calling the script.

Local verification: **13 targeted tests passed**, including identity/ambiguity safeguards and existing learning/autolabelling tests. Ruff and full Linux/Windows-targeted mypy checks passed. The original model-report confusion cells were reproduced independently, and all 25 strict outputs match the previously frozen resource-check outputs.

Main report: https://github.com/Coder-Meet/battleoftheschool/blob/main/STEVEN_LABEL_ANALYTICS.md

Algorithm review: https://github.com/Coder-Meet/battleoftheschool/blob/main/STEVEN_ALGORITHM_REVIEW.md
