# Strong-pipeline replication: SOFT wins 3/3 seeds (+0.014 AUC), but the 58-study CI crosses zero

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734105
- **Author:** **FHZ982** (@fhz982, Contributor) · _topic author_ · 2026-08-09 22:09 UTC · ▲2
- **Topic votes:** 2 · **Comments/replies:** 0 · **Pinned:** no
- **Last comment:** 2026-08-09 22:09 UTC by FHZ982
- **Cloned:** 2026-10-14 18:23 UTC (topic id 734105, raw JSON `_raw/734105.json`)

---

**Update after a stronger, pre-specified replication.** The original version of this post reported a large advantage for graded report-derived targets in a modest ResNet-18 pipeline. We have now repeated the core HARD-vs-SOFT comparison with a DINOv2 backbone, three paired seeds, ten epochs, checkpoint selection separated from evaluation, and the 58 official studies kept out of training and model selection.

The result is more interesting, and less tidy, than the original headline:

- SOFT beats HARD on the 58 official studies in all three paired seeds.
- The rank-mean ensemble improves by **+0.0143 macro AUC**.
- Its paired 95% bootstrap interval is **[-0.0041, +0.0330]**: it crosses zero and does not fit inside our pre-specified equivalence margin of +/-0.01.
- On an independent 430-study evaluation set scored against the *binarized lexicon target*, HARD wins by **0.0132** with a 95% interval of **[0.0059, 0.0207]** in its favour.
- Our proposed mechanism does not survive its pre-specified check: Spearman correlation between the signal destroyed by binarization and the per-finding SOFT improvement is **-0.161**.

So the updated conclusion is deliberately narrower: **SOFT transfers better to the 58 official labels in this run, consistently across seeds, but 58 studies do not establish superiority at 95%. HARD is better at reproducing the binarized lexicon target. We cannot claim that retained target nuance explains the official-label pattern.**

That is the result. The rest of the post documents exactly how we got there, what changed from the first experiment, and what the leaderboard comparison still needs to resolve.

## 1. The comparison

Only 58 of the 4,407 training studies carry per-condition labels, so most pipelines derive targets for the other 4,349 studies from the `Report` column. The choice tested here is simple:

| arm | affirmed mention | hedged mention | explicitly negated | not mentioned |
|---|---:|---:|---:|---:|
| HARD | 1 | 1 | 0 | 0 |
| SOFT | 1 | 0.3-0.8 | 0.04-0.16 | 0.28 |

Both arms use the same deterministic multilingual lexicon, studies, image preprocessing, architecture, loss weights, optimizer, schedule, augmentations and seeds. Only the target tensor changes.

The stronger replication used:

- DINOv2, 12 blocks, last six trainable (10.7M trainable parameters), feature dimension 768;
- ten epochs for every run;
- paired seeds 11, 22 and 33;
- interleaved order: SOFT-11, HARD-11, SOFT-22, HARD-22, SOFT-33, HARD-33;
- 3,479 studies for training;
- `va_sel`: 440 studies used only to choose the epoch;
- `va_ev`: 430 different studies opened only after weights were frozen;
- 58 official studies held outside training and checkpoint selection.

Duplicate report texts remain grouped. The second holdout split uses an independent salted hash, so `va_sel` and `va_ev` are disjoint without breaking report groups.

## 2. Official 58-study result

The selected checkpoints produced:

| seed | SOFT `va_sel` | HARD `va_sel` | SOFT AUC-58 | HARD AUC-58 | SOFT-HARD |
|---:|---:|---:|---:|---:|---:|
| 11 | 0.7853 | 0.7921 | 0.7943 | 0.7840 | +0.0104 |
| 22 | 0.7781 | 0.7912 | 0.7965 | 0.7881 | +0.0083 |
| 33 | 0.7745 | 0.7897 | 0.7959 | 0.7880 | +0.0079 |

Across seeds:

- SOFT: **0.7956 +/- 0.0009**
- HARD: **0.7867 +/- 0.0019**
- paired difference: **+0.0089 +/- 0.0013**
- signs: **3/3 for SOFT**

Three seeds are a stability check, not a sample size. They share training data and evaluation patients; treating them as independent observations would be pseudoreplication. Even the most favourable exact paired sign test is only p=0.125 one-sided.

For the pre-specified primary comparison we rank-averaged the three members of each arm, because AUC reads order rather than calibration. A 4,000-resample paired bootstrap over the same 58 patients gave:

| ensemble | macro AUC | 95% bootstrap interval |
|---|---:|---:|
| HARD rank-mean | 0.7930 | [0.7474, 0.8368] |
| SOFT rank-mean | 0.8071 | [0.7627, 0.8503] |
| SOFT-HARD | **+0.0143** | **[-0.0041, +0.0330]** |

SOFT wins in 93.6% of bootstrap resamples. That is consistent directional evidence, but it is **not** a 95% confirmation. The interval crosses zero. It also does not fit entirely inside the pre-specified practical-equivalence margin of +/-0.01, so it is not evidence of equivalence either. The correct reading is **inconclusive**.

This distinction matters. "The interval crosses zero" means that the experiment has not resolved the effect; it does not mean that the two target schemes are interchangeable.

## 3. Independent evaluation against the lexicon surrogate

The 430-study `va_ev` set was never used to choose an epoch. Both frozen ensembles were scored against the same binarized lexicon labels:

| ensemble | macro AUC on `va_ev` |
|---|---:|
| HARD rank-mean | 0.8059 |
| SOFT rank-mean | 0.7927 |
| SOFT-HARD | **-0.0132** |

Paired 95% interval: **[-0.0207, -0.0059]**. The interval excludes zero, so HARD has a demonstrated advantage on this endpoint.

The scope is crucial. This endpoint is not a radiologist reading images; it is the binarized output of the report lexicon. It is also exactly the target HARD was trained to reproduce. The result therefore says that HARD is better at reproducing the HARD surrogate. It does **not** overturn the official-label result, and it must not be presented as if it did.

The divergence is still useful. On the surrogate, HARD wins cleanly. On the official 58, SOFT wins all three seeds and the ensemble point estimate. The leaderboard is the next external test because it contains far more official patients, even though Kaggle will return only aggregate macro AUC and no patient-level interval.

## 4. The mechanism check failed

The original experiment suggested a plausible story: SOFT should help most for findings where binarization destroys the most graded target signal. We fixed that hypothesis before inspecting the strong-pipeline result.

For each finding we measured, on the training set, the weighted target signal destroyed by binarization:

`sum(W * abs(Y_soft - I[Y_soft > 0.5])) / sum(W)`

We then correlated it with the per-finding SOFT-HARD improvement on the official 58. Spearman rho was **-0.161** across 12 findings.

That sign is null or opposite to the proposed mechanism. The causal explanation must therefore be withdrawn. The per-finding pattern can be described, but this experiment does not explain it.

| finding | positives | HARD | SOFT | difference |
|---|---:|---:|---:|---:|
| ACL | 24 | 0.784 | 0.866 | +0.081 |
| MCL | 9 | 0.755 | 0.760 | +0.005 |
| Medial Meniscus | 26 | 0.692 | 0.695 | +0.003 |
| Lateral Meniscus | 23 | 0.739 | 0.748 | +0.009 |
| Medial OA | 15 | 0.957 | 0.953 | -0.003 |
| Lateral OA | 11 | 0.769 | 0.725 | -0.044 |
| PF OA | 21 | 0.798 | 0.815 | +0.017 |
| Effusion | 35 | 0.841 | 0.880 | +0.039 |
| Synovitis | 27 | 0.683 | 0.713 | +0.030 |
| Baker's cyst | 12 | 0.916 | 0.909 | -0.006 |
| Contusion | 19 | 0.787 | 0.782 | -0.005 |
| Fracture | 18 | 0.794 | 0.838 | +0.044 |

SOFT is higher on eight findings and lower on four. These findings share one model and the same patients, so the count is descriptive, not a test. With only 9-35 positives per finding, individual intervals are necessarily wide.

## 5. What changed relative to the original experiment

The first experiment used a modest ResNet-18 2.5D pipeline and three target arms:

| arm | macro AUC | seeds |
|---|---:|---|
| HARD | 0.6497 | 0.6494 / 0.6492 / 0.6504 |
| GRADED | 0.6698 | 0.6732 / 0.6623 / 0.6740 |
| SOFT | 0.6999 | 0.6983 / 0.7048 / 0.6966 |

Its rank-mean SOFT-HARD difference was **+0.0559**, with a paired bootstrap interval of **[+0.0256, +0.0870]**. Splitting the change through the GRADED middle arm attributed +0.0236 to grading hedges and +0.0323 to distinguishing silence from explicit negation.

Those numbers remain valid for that pipeline, but the strong replication changes their interpretation:

1. The effect size did not generalize quantitatively: +0.0559 became +0.0143.
2. The 58-study interval in the strong pipeline no longer excludes zero.
3. The pre-specified mechanism check failed, so the earlier decomposition cannot be promoted from an ablation description to a causal explanation of the strong-pipeline result.

The original experiment was useful because it generated a falsifiable hypothesis. The replication did what a replication should do: it narrowed the claim.

## 6. Four preprocessing bugs that still matter

These findings survived the architecture change and are independent of the target comparison.

### 6.1 Slice order

Do not sort `.dcm` files by filename. The filename is a SOP Instance UID, assigned to be unique rather than ordered. Sort by physical position `k = p . n`, where `n` is the cross product of the row and column directions from `ImageOrientationPatient`, and `p` comes from `ImagePositionPatient`.

### 6.2 Physical scale

Resizing without a physical crop can erase 1-3 mm pathology before the first convolution. We crop to a fixed 130 mm field and resize to 336 pixels, about 0.387 mm/px. Check that the requested crop actually fits; otherwise it can silently become a no-op.

### 6.3 Laterality

Five targets are medial/lateral pairs, while `train.csv` has no Laterality column. We derive it from geometry using the image centre in patient space, not the corner in `ImagePositionPatient`. Coronal and axial views need a horizontal normalization; sagittal stacks need their order normalized because the medial-lateral axis is the stack axis.

### 6.4 Encoder input

Stacking 18 slices as channels and averaging the first convolution's pretrained weights destroys the input interface the encoder learned. Feed groups of three adjacent slices through the untouched backbone and aggregate feature vectors afterwards.

## 7. Reproducibility notes

- 4,407 studies, 4,276 unique report texts.
- Duplicate report texts are grouped before splitting.
- Nine languages are present; Turkish case folding needs explicit handling of dotted and dotless `i`.
- The labels come from a deterministic multilingual lexicon, not an LLM.
- Preprocessing dominates wall time, so cached volumes are essential for paired experiments.
- The six selected checkpoints, a manifest and per-seed/rank-mean CSVs were saved by the successful run.

## 8. What remains

The pre-specified next external comparison is **SOFT rank-mean vs HARD rank-mean on the competition leaderboard**. The primary pair must be submitted from inference notebooks built from the exact frozen checkpoints. Per-seed submissions are secondary diagnostics and should not be used to search for a favourable configuration.

Until that comparison exists, the defensible summary is:

> In a strong DINOv2 pipeline, SOFT was better on all three seeds and improved the 58-study rank-mean ensemble by +0.0143 macro AUC, but the paired 95% interval [-0.0041, +0.0330] did not exclude zero or establish practical equivalence. HARD was significantly better on an independent binarized-lexicon surrogate, and the proposed target-nuance mechanism was not supported.

I will update this post again when the paired leaderboard scores are available.

---

## Comments (0, threaded, oldest first)

_No comments._
