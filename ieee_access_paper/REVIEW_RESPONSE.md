# Response to the external review (2026-11-03)

**FIXED** = changed in `manuscript/main.tex`, `manuscript/supplement.tex` or the code.
**AUTHOR** = needs data, GPU time or a decision only the authors have. Nothing here was invented to fill a gap.

## 1. Untouched evaluation as the main result

**FIXED (structure).**
- The 30% partition is now called the *development partition* everywhere. It is stated that more than 60 configurations were scored on it and that it informed decisions (§III-D).
- The 70% partition is the *final-test partition* and is described as the only unbiased evaluation. Results open with it: §VI-A, Table 5, with a development and a final-test column.
- All model and final-selection decisions were made before any final-test score was visible (§III-D, §VI-A).
- The abstract, every table caption, the discussion and the conclusion now label development results as exploratory.

**AUTHOR.** Fill the final-test column of Table 5 and the two values in the abstract:
- final system (H976, sub 57497105);
- unchanged parent (C2, 57315983);
- A (57053758), B (57053761) and C3 (57103459);
- the teacher alone (nartaa's notebook, if its private score is shown).

The Kaggle token in `~/.kaggle/access_token` returned HTTP 401. Renew it and I can fetch the scores read-only, or copy them from the Submissions page. If A/B/C3 have no private score, keep the sentence that says so.

## 2. Contradictory description of the 58 reference studies

**FIXED** (§III-B). The section now says:
- gold-58 never entered gradient updates;
- it was used as a diagnostic, in adoption rules (the "≥ 0.9127" and "≤ 0.01 drop" rules are quoted), and by the teacher's authors for selection;
- its scores and intervals are descriptive development results.

The words "never used to choose" are gone.

## 3. What "image teacher" means

**FIXED.**
- The text uses "image-input teacher" throughout.
- New **Table 4 (teacher provenance)** has columns for training images, target labels, external data, gold-58 use, and in-sample vs out-of-fold.
- The Mirror row is traced to its author's write-up (forum 746792 in `discussions/`):
  - 4,349 RSNA studies with blended report-derived labels, plus 2,399 OAI knees;
  - the OAI knees carry expert MOAKS grades for PF OA, lateral OA and synovitis only, and the other cells are masked;
  - gold-58 was excluded from training but used for selection;
  - the inference recipe was tuned with development scores.
- §VI-B and §VII state that the gain may come from image evidence, external expert grades or complementary errors, and that the experiments do not separate these.

**AUTHOR.** Rows R6 and S6fall-B are marked *unverified*. Check them against heliosli's R6 bundle and nartaa's write-up (discussion 749359, not in the offline clone).

**Correction to the previous draft.** It said S6fall-B was trained with OAI. That was never verified. The only confirmed external dependency of S6fall-B is the SKM-TEA anatomy model, and the text now says so.

## 4. Teacher-fold provenance and the pseudocode error

**FIXED.**
- Algorithm 4 now trains fold models on `{i : i ∉ G58, φ(i) ≠ k}`, and every batch is drawn from `i ∉ G58`.
- The EMA line matches the code: `d = min(0.998, (1+t)/(10+t))`.
- S1 is described per component in Table 4 (R6 "reported as out-of-fold, unverified"; S6fall-B in-sample).
- §IV-E states that in-sample targets do not leak hidden-test information but rule out cross-fitting claims.

## 5. Originality and the closely related preprint

**FIXED.**
- Fig. 1 is recoloured: green = reused (LLM labels, consensus, preprocessing, reader), blue = external teachers and ensembles, orange = this work.
- Algorithms 2–3 and §IV-B/C are titled "(reused)". A "What was reused" paragraph opens the paper.
- The contributions are cut to three, around the reviewer's central question, which is now quoted in §I.
- The consensus construction is described as *adopted*, with an explicit statement that it was not compared with the individual sources.
- **The Kooklan preprint (arXiv:2610.40708, 28 Oct 2026) is verified and cited.**
  - §II-D summarises it: report vs image-model agreement with experts 0.892 / 0.896, mixture 0.927, but no private-test gain for a ResNet-50 (0.872 vs 0.877).
  - It states how this work differs: the reader, the teachers, fusion, and a development result that points the other way.
  - §VII says that the final-test scores are needed to reconcile the two.

## 6. Ablations (all AUTHOR, need GPU)

The paper now lists them as limitation (iv) instead of implying they were done. Priority order:

| Experiment | Script to adapt | Notes |
|---|---|---|
| τ = 1 (teacher-only target) | `code/distillation/students_round2_C2_C3.py` | Add an arm with `tw=1.0`. |
| Seeds: A, B, C3 × 3 seeds | same scripts | Change `seed=`. ~6 h per pair of arms on 2×T4. |
| Teacher without external data | `students_round1_C_B2.py` with the self-teacher B or an OAI-free model | Separates "image input" from "OAI". |
| Individual label sources vs consensus | `code/reader/make_labels.py` (`public_sources`) | Only needed if the consensus is claimed as a contribution. It no longer is. |
| Parent vs parent + students | already in Table 8 | Needs the final-test column (item 1). |

## 7. Statistics

**FIXED.**
- §VI-F is retitled to "Resolution of the development yardsticks".
- The 385-study experiment is presented as an *illustration under stated assumptions*. Its dependence on model, perturbation, prevalence, correlation and label noise is spelled out.
- The "run-to-run noise ≈ 1e-4" claim and the sign-test p-value are removed.
- It is stated that the gold-58 CIs describe E11/E03b, not A/B/C3.
- §V explains the bootstrap handling: a resample missing a class contributes no AUC for that finding, and resampling is study-level because no patient IDs exist.
- §V also says that overlapping intervals and equal rounded scores are not read as equivalence.

**PARTLY AUTHOR.** `code/analysis/make_figures.py` now computes paired gold-58 CIs for B−A, C−B and C3−A as soon as `gold_arm{A,B,C,C3}.csv` (outputs of the training kernels `esfersami50/rsna-knee-gpjw-pseudo` and `-gpjw-student*`) are copied into `code/analysis/data/`.

## 8. Overclaims

**FIXED.**

| Old wording | New wording |
|---|---|
| "do not matter" | "no clear benefit was observed in the configurations evaluated" |
| "equal to the teacher" | "the same value as the teacher at three-decimal precision" |
| "consistently improved" | eight ordinal, non-independent development comparisons that need final-test confirmation |
| "pre-registered" | "prospectively logged" (the log is version-controlled, not externally registered) |
| Table 2 called "label reliability" | "inter-source agreement, not accuracy" |
| Worklist prioritisation | a "potential application" with the list of required evaluations |

The fixed weight is now disclosed: w = 0.2 was specified first, then w = 0.3 / 0.4 were tried in four submissions on two parents, and none ranked above the w = 0.2 system.

## 9. Dataset and reference-standard reporting

**FIXED (as far as the data allow).**
- §III-A lists every unavailable item (institutions, period, demographics, scanners, field strength, criteria, patient IDs, repeat or bilateral examinations, label definitions, annotators, selection of the 58) and does not infer them.
- It says that finding names follow the organisers' definitions, which may be narrower than the clinical terms.
- It says that report grouping is not patient separation.
- It adds the 24,371 series and zero decoding errors.
- A CLAIM 2024 checklist table is added (Supplement S3). CLAIM 2024 is verified and cited (*Radiol. Artif. Intell.* 6(4):e240300).

**AUTHOR.**
- Check every CLAIM row against the official item list and add item numbers.
- Add the per-slot missing-series counts.

## 10. Clinical evaluation

**FIXED (text).** §VII lists what clinical use would require:
- per-finding sensitivity and specificity at pre-set thresholds;
- PR-AUC;
- calibration;
- reader review of errors;
- subgroup and scanner analyses on an independent cohort with compatible definitions.

It also explains that rank-fused scores are cohort-dependent and not probabilities, and how a single new examination would be scored. External testing is limited to label-compatible cohorts.

**AUTHOR.** Running any of these needs an independent labelled cohort.

## 11. Reproducibility and numerical consistency

| Issue | Status |
|---|---|
| Missing S1 training script | **AUTHOR**: download the script of Kaggle kernel `hosen42/rsna-knee-gpjw-student5` into `code/distillation/`. |
| bf16 vs fp16 trainer | **FIXED**: §IV-D and `README.md` say every reported reader came from the executed scripts in `code/distillation/` (fp16, 2 × 4 accumulation); `reader/train.py` is the upstream reference. |
| EMA warm-up | **FIXED** in §IV-D, Table 3 and Algorithm 4. |
| OOF 0.872 vs 0.870 | **FIXED**: Table 9 caption = soft-label estimand with the 0.25–0.7 band excluded; Table 10 caption = binarised at 0.5, no band excluded, macro 0.870. |
| Percentile subsampling, transpose/flip only | **FIXED** in §IV-B and Algorithm 2 (`V[:, ::4, ::4]`; no oblique resampling). |
| `<OWNER>` in the repository URL | **FIXED**: `github.com/hamidhosen42/RSNA-Knee-Abnormality-Detection`. **AUTHOR**: tag a release, and cite its version or DOI. |
| Planning budgets as runtime | **FIXED**: removed. **AUTHOR**: measured decode time, model time, peak memory and fallback counts, for a single student and for the 7-student stage (TODOs in §VI-B and Supplement S2). |
| Checkpoints and hashes | **AUTHOR**: the hashes are in `ours_leg/build_h960.py` (`S1_SHA`) and `student_leg.py`. Publish the weights and point to them in the availability section. |

## 12. Restructure and finish

**FIXED.**
- New title (the reviewer's suggestion).
- Abstract: "Of its 4,407 training studies… 58 additionally have expert labels" and the development / final-test split. It is 220 words.
- One central question and three contributions.
- The supplement holds the switch table (S1), the fall-back algorithm (S2) and CLAIM (S3).
- Tables 8 and S1 are labelled ordinal and exploratory.
- The discussion separates "Findings" from "Possible explanations (not tested)".
- The conclusion is restricted to what was shown.
- A "Use of AI Tools" disclosure section is added. Confirm its wording.
- Kaggle profile citations are replaced by exact notebooks and datasets, with version notes. The dreaddevelopment, nartaa, heliosli and hengck23 entries still need their versions and access dates confirmed.

**Kept on purpose.** `\history{}` and `\doi{}` are required macros of the IEEE Access template; IEEE fills them at publication. The DOI is now the template's 2026 placeholder rather than an example number.

**AUTHOR.** Author names, affiliations, ORCIDs, biographies, photos, funding and ethics statements (red TODOs).

## Journal tier

The reviewer says IEEE's January 2026 title list places IEEE Access in **Q2 by JIF**. I did not verify this. If your institution requires Q1, check the classification it uses (JCR / Scopus / SJR, category, year) before submitting. A clinically oriented Q1 venue would also expect the external testing of item 10.

## Files

- `manuscript/main.pdf`: 10 pages. Builds without errors.
- `manuscript/supplement.pdf`: 3 pages.
- `manuscript/ieee_access_overleaf.zip`: now includes `supplement.tex`.
- Previous manuscript versions are in the session scratchpad (temporary): `main_before_review.tex`, `main_v2_before_rewrite.tex`.
