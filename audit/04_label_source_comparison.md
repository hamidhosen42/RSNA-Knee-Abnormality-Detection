# 04 — Label source comparison: dreaddevelopment vs team consensus (2026-09-27)

Two report-derived label sets are now in play:

| | `dreaddevelopment/rsna-knee-labels` `labels_llm_soft` ("dd") | esfersami50 `nx1-01-consensus-labels` `labels_consensus` ("cons") |
|---|---|---|
| Used by | `ours_leg/` (E11, training v2 running on Kaggle) and the public Raptor CoAtNets | the team's R34 / EffB2 / CoAtNet224 fold-0 models |
| Studies | 4,349 (gold-58 excluded) | the same 4,349 |
| Value style | discrete levels (0.05, 0.2, 0.35, 0.76, 0.8, 0.85, …) | smooth mean of 4 sources (negatives ≈0.2–0.3, positives ≈0.93–0.96) |
| Gold-58 score | **cannot be computed** (no gold rows) | macro 0.920, but the 4 sources were *chosen* on gold-58, so this is optimistic |

Data: `label_compare/per_target_agreement.csv`, `top_disagreements.csv`,
`gold_vs_source_prevalence.csv`, `folds_crosstab.csv`.

Gold-58 was used here only descriptively (prevalence and report-vs-label examples). No choice
was made from it.

## 1. Both sets rank studies similarly, except for Synovitis

| Target | AUC(dd → cons≥0.5) | κ@0.5 | dd pos-rate | cons pos-rate |
|---|---|---|---|---|
| ACL | 0.950 | 0.70 | 0.13 | 0.21 |
| MCL | 0.983 | 0.44 | 0.05 | 0.15 |
| Medial Meniscus | 0.966 | 0.84 | 0.38 | 0.42 |
| Lateral Meniscus | 0.967 | 0.80 | 0.14 | 0.16 |
| Medial OA | 0.979 | 0.69 | 0.25 | 0.37 |
| Lateral OA | 0.969 | 0.61 | 0.15 | 0.27 |
| PF OA | 0.987 | 0.73 | 0.34 | 0.46 |
| Effusion | 0.919 | **0.38** | 0.26 | **0.60** |
| Synovitis | **0.619** | **0.31** | 0.25 | 0.13 |
| Baker's | 0.986 | 0.59 | 0.12 | 0.25 |
| Contusion | 0.891 | 0.55 | 0.15 | 0.17 |
| Fracture | 0.979 | 0.68 | 0.04 | 0.07 |

Macro: AUC 0.933, κ 0.61.

**Effusion.** The difference is a threshold, not the ordering.
- dd gives "small/minimal joint effusion" ≈0.2; the consensus gives it ≈0.85. That flips 1,479
  studies (34%) at 0.5.
- The ranking is still preserved (none < small < moderate), so AUC-wise both can work.
- They do teach a BCE model different targets for small effusions.

**Synovitis.** The two sets define it differently.
- Only 7.4% of reports mention synovitis.
- For silent reports, dd **imputes synovitis from effusion**: mean 0.75 when dd-effusion is
  positive, 0.35 otherwise. The consensus does so only weakly (0.47 vs 0.24).
- Evidence on imputation conflicts, and both pieces are gold-58-based and small (27 synovitis
  positives):
  - stevenleehans (forum 733932) found the Effusion field predicts gold Synovitis better than the
    Synovitis field (0.7115 vs 0.6780).
  - The team's label notebook found effusion-imputation *lowered* its consensus Synovitis gold AUC
    (0.7945 → 0.675) and did not keep it.

## 2. Official labels are not report-derived, and gold-58 is enriched

- Host (via forum 742106): *"The labels weren't generated based on the report but by an
  independent process… Same is true of the test set."*
- Our own check agrees:
  - Gold studies with no mention of effusion are labelled 1.
  - Identical "Small joint effusion" wording is labelled 1 in one study and 0 in another.
- Gold-58 is the annotator's sampling, not disease prevalence (every gold study has ≥1 positive;
  mean 4.14 per study, per forum 733932).
  - For example: Fracture 31%, ACL 41%, Synovitis 47% in gold, vs 4–25% in either report set.
  - **Prevalence therefore cannot decide which label definition is right.**

Consequence: every report-derived label has a ceiling set by what reports omit (synovitis,
contusion, fracture, lateral OA are the weakest in the team's gold benchmark). That ceiling, not
the model, is likely why report-only single models plateau.

## 3. The two fold files are unrelated

- `ours_leg/folds_v1.json`: seed 20260927, stratified on dd≥0.5, sha256 `f4292c4a…`.
- Team `folds_v1.json`: seed 2026.
- Both exclude gold-58 and group by normalised-report sha1.
- Fold-0 overlap is 206 studies (Jaccard 0.13).
- **OOF numbers from the two pipelines are not comparable, and OOF-learned blend weights across
  them are impossible** until the team uses one file.

## 4. Recommendations

1. **Do not change labels in the running `ours_leg` v2 job.** It already changes one variable
   (backbone) on a recipe proven with dd labels (dreaddevelopment's CoAtNet reached ~0.932 solo
   LB with them).
2. **Standardise on `ours_leg/folds_v1.json` (`f4292c4a`) for the whole team.** The running job
   uses it, it is stratified and checksummed, and the team's existing models are fold-0 only
   anyway.
3. **Decide the label source with one controlled ablation (E03b, PROPOSED):**
   - Same `ours_leg` recipe, folds and seed; only labels dd → cons.
   - Compare fold-0 predictions against *both* label sets, plus gold-58 as a diagnostic.
   - Look separately at Synovitis and Effusion, where the sets actually differ.
   - Decide on the same rules as `next_10_experiments.md` §1. The consensus's gold-58 advantage is
     partly selection bias, so it must not be the tiebreaker alone.
4. Treat "impute Synovitis from Effusion" as its own experiment later. The evidence conflicts;
   don't adopt it by default in either direction.
