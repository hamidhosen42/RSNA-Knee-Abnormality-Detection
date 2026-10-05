# RSNA Knee Abnormality Detection — Deep Competition Analysis (Bangla)

**Competition:** RSNA Knee Abnormality Detection  
**Kaggle:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection  
**Language:** Bengali (technical terms in English)  
**Goal:** Strong private leaderboard performance with reproducible, leakage-safe validation

---

## Executive Summary

এই competition-টা সাধারণ image classification competition না। সবচেয়ে গুরুত্বপূর্ণ bottleneck হচ্ছে:

1. **Weak-label construction**
2. **MRI geometry / preprocessing**
3. **Validation design**
4. **Model diversity**
5. **Final ensemble robustness**

Training set-এর বড় অংশে MRI + radiology report আছে, কিন্তু official 12-label ground truth খুব সীমিত। ফলে report-based pseudo/soft labels এবং image-only final inference—এই দুইয়ের মধ্যে bridge বানানোই মূল challenge।

সবচেয়ে গুরুত্বপূর্ণ strategic conclusion:

> **Public leaderboard score chase করার চেয়ে scanner/site-aware grouped CV, robust label generation, correct physical DICOM ordering, 2.5D slice modeling, target-aware multi-view modeling এবং low-correlation ensemble বেশি গুরুত্বপূর্ণ।**

Public notebook ceiling প্রায় **0.943** verify করা যায়, কিন্তু top notebooks-এর বড় অংশ একই DINO/RadImageNet/Raptor/CoAtNet stack-এর fork/reblend। তাই এগুলোকে independent evidence হিসেবে না দেখে shared inference lineage হিসেবে treat করা উচিত।

---

# 1. Competition Objective

প্রতিটি knee MRI study থেকে 12টি clinically important abnormality-এর probability predict করতে হবে।

### Target labels

1. ACL injury
2. MCL injury
3. Medial Meniscus tear
4. Lateral Meniscus tear
5. Medial OA
6. Lateral OA
7. PF OA
8. Effusion
9. Synovitis
10. Baker's cyst
11. Contusion
12. Fracture

Official labels image review-এর ওপর নির্ভর করে তৈরি হয়েছে। দুইজন subspecialty musculoskeletal radiologist independently label করেছেন, disagreement হলে তৃতীয় radiologist adjudicate করেছেন।

Competition overview:  
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview

Data page:  
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data

---

# 2. Dataset Breakdown

## 2.1 Training Studies

প্রায় **4,407 training studies** আছে।

`train.csv`-এ:

- `StudyInstanceUID`
- `Report`
- 12 target columns

কিন্তু গুরুত্বপূর্ণ বিষয়:

> **মাত্র 58 study-তে official image-reviewed 12-label ground truth আছে।**

বাকি ~4,349 study-তে MRI + radiology report আছে, কিন্তু target labels missing। এটাই competition-এর সবচেয়ে গুরুত্বপূর্ণ structural fact।

## 2.2 Series Metadata

`train_series.csv` / equivalent metadata-তে থাকে:

- `StudyInstanceUID`
- `SeriesInstanceUID`
- `Fluid_Sensitive`
- `Fat_Suppression`
- `Anatomical_Plane`

Anatomical plane সাধারণত:

- Sagittal
- Coronal
- Axial

Series প্রতি slice count variable। Typical sequence length প্রায় 20–45 slices, median প্রায় ~30; কিছু series-এ শতাধিক slice থাকতে পারে।

## 2.3 Test Set

Scoring-এর সময় প্রায় **1,300 hidden studies** ব্যবহার হবে। সবচেয়ে গুরুত্বপূর্ণ:

> **Test set-এ radiology `Report` থাকবে না।**

অর্থাৎ final model অবশ্যই **MRI-only inference** করতে পারবে। Report ব্যবহার করা যাবে training-time privileged information হিসেবে।

---

# 3. Target Variable

প্রতিটি study-এর জন্য 12 binary target। Final output probability।

```text
StudyInstanceUID,
ACL,
MCL,
Medial Meniscus,
Lateral Meniscus,
Medial OA,
Lateral OA,
PF OA,
Effusion,
Synovitis,
Baker's,
Contusion,
Fracture
```

---

# 4. Evaluation Metric

Official metric:

\[
Score = \frac{1}{12}\sum_{i=1}^{12} AUC_i
\]

অর্থাৎ:

> **Macro-average ROC AUC across 12 targets**

Important implication:

- প্রতিটি target equal weight পায়
- Probability calibration ranking-এর তুলনায় কম গুরুত্বপূর্ণ
- Rank averaging strong হতে পারে
- Rare targets-ও total score-এ 1/12 weight পায়

---

# 5. Submission Format

Required file:

```text
submission.csv
```

Expected columns:

```text
StudyInstanceUID
ACL
MCL
Medial Meniscus
Lateral Meniscus
Medial OA
Lateral OA
PF OA
Effusion
Synovitis
Baker's
Contusion
Fracture
```

প্রতিটি target-এর জন্য probability দিতে হবে।

---

# 6. Public vs Private Leaderboard

Official structure:

- Public LB ≈ 30% hidden test
- Private LB ≈ 70% hidden test

এর ফলে public leaderboard overfitting-এর risk অনেক বেশি। Public LB score 0.943 থেকে final private score significantly পরিবর্তিত হতে পারে।

### Strategy

Experiment accept/reject করার primary criteria:

1. Grouped CV
2. Seed stability
3. Target-level stability
4. Scanner/site robustness
5. Model diversity

Public LB শুধু secondary sanity check।

---

# 7. Why This Competition Is Hard

এই competition-এর difficulty মূলত পাঁচ জায়গায়:

## 7.1 Weak labels

4,349 studies-এর জন্য official targets নেই।

## 7.2 Report-image mismatch

Radiology report clinical language। Competition label strict image criteria। Example:

```text
"minimal effusion"
```

report-এ mention থাকলেও competition definition অনুযায়ী positive না-ও হতে পারে।

## 7.3 Multi-series MRI

এক study-তে একাধিক acquisition। Different planes / sequences different pathology capture করে।

## 7.4 Site / scanner shift

MRI appearance scanner, protocol, field strength, resolution, vendor অনুযায়ী change করে।

## 7.5 Public LB noise

Public subset মাত্র 30%।

---

# 8. Report Labels vs Ground Truth

Host clarification অনুযায়ী official labels radiology report থেকে extract করা হয়নি। Labels MRI images দেখে independently তৈরি হয়েছে।

তাই:

```text
Report label != Gold image label
```

এই mismatch expected।

Relevant discussion:  
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826

### Consequence

Simple approach:

```text
Report -> hard 0/1 label
```

suboptimal। Better:

```text
Report
  ↓
uncertainty-aware extraction
  ↓
soft labels
  ↓
confidence weighting
  ↓
MRI model
```

---

# 9. Public Notebook Analysis

## Important Warning

Notebook title-এ score লেখা থাকলেই সেটা verified score নয়। Example: title-এ `0946` থাকতে পারে, কিন্তু actual Kaggle Best Score `0.943`। তাই notebook ranking করার সময় Kaggle public score field verify করা জরুরি।

---

# 10. Public Notebook Ranking

| Rank | Notebook / Approach | Author | Public LB | CV | Main Model / Strategy | Overfit Risk | Reproducible |
|---:|---|---|---:|---|---|---|---|
| 1= | RSNA Knee Abnormality Detection | Aman Atar | 0.943 | Unverified | DINO + A5 + RadImageNet + CoAtNet/Raptor | Med-High | Yes |
| 1= | Phase1 Fork — Speedy Raptors | wenyuan Guo | 0.943 | Unverified | Public high-score stack | Med-High | Yes |
| 1= | Reproducing the 0.943 Public Stack | Arin | 0.943 | Unverified | DINO → A5 → RadImageNet → Raptor → CoAtNet | Medium | Yes |
| 1= | DINOsaur V5 | Roman Tamrazov | 0.943 | Unverified | DINO-heavy ensemble | Med-High | Yes |
| 1= | rsna-knee-d4-public0946 | yamadan96 | 0.943 | Unverified | D4 ensemble | High | Yes |
| 1= | D4 + Fine-Tuned v5 | Indar Karhana | 0.943 | Unverified | D4 + fine-tuned stack | Med-High | Yes |
| 1= | 0943-SOTA-D4-global96 | Sushanth Tiruvaipati | 0.943 | Unverified | Global96/D4 stack | Med-High | Yes |
| 8 | rsna-knee-0942-restructured | maverick_ss_26 | 0.942 | Unverified | Restructured public stack | Medium | Yes |
| 9 | NTejas full pipeline | NTejas | 0.941 | 0.9078 gold OOF blend | RadImageNet/ImageNet + public blend | Medium | Strong Yes |
| 10 | DINO-RadImageNet Rank Ensemble | Tony Li | 0.940 | Unverified | DINO + RadImageNet + Raptor + CoAtNet | Medium | Yes |
| 11 | Take Care Of Your Knee | Anhad Mahajan | 0.936 | Unverified | DINOv2 ensemble | Medium | Yes |
| 12 | DINO + RadImageNet + dual CoAtNet | Anthony Therrien | 0.934 | Unverified | Mixed family ensemble | Medium | Yes |
| 13 | Smart Ensemble | Aman Atar | 0.911 | Unverified | Public ensemble | Medium | Yes |
| 14 | Data Structure, EDA, Baseline | Roman Rozen | 0.894 | — | DINO / physical MRI processing | Low-Med | Yes |
| 15 | Simple baseline | Praneeth Paikray | 0.798 | — | ResNet18 + report labels | Low | Yes |
| 16 | Report-Teacher Anatomy-aware | Blacklions | 0.781 | — | DINOv2 + report teacher | Medium | Yes |

---

# 11. Important Public Notebook Links

### Aman Atar — 0.943
https://www.kaggle.com/code/amanatar/rsna-knee-abnormality-detection

### Arin — Reproducing the 0.943 Public Stack
https://www.kaggle.com/code/skarin/rsna-knee-reproducing-the-0-943-public-stack

### Tony Li — DINO / RadImageNet Rank Ensemble
https://www.kaggle.com/code/tonylica/rsna-knee-dino-radimagenet-rank-ensemble

### yamadan96 — D4
https://www.kaggle.com/code/yamadan96/rsna-knee-d4-public0946

### Competition notebooks
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/code

---

# 12. Reverse Engineering the 0.943 Public Stack

Public stack roughly:

```text
MRI
 │
 ├── DINO branch
 │
 ├── A5 / LabelAttentionPool
 │
 ├── RadImageNet CNN branch
 │
 ├── Raptor branch
 │
 └── CoAtNet branch
        │
        ↓
Target-specific blending
        │
        ↓
D4 / TTA
        │
        ↓
Final 12 probabilities
```

### Why it works

1. Strong pretrained representations
2. Complementary model families
3. Multiple views
4. Target-wise blend
5. TTA
6. Prediction rank diversity

---

# 13. Main Weakness of the 0.943 Public Stack

Top notebooks-এর বড় অংশ independent solution না। Same family / public weights reuse করে। Possible issues:

- Public LB tuning
- Similar errors across models
- Target weight over-optimization
- No independent 5-fold OOF
- Shared model lineage
- Hidden private shift vulnerability

---

# 14. Strongest Reproducible External Resource

GitHub:

https://github.com/NTejas-1/RSNA-Knee-Abnormality-Detection

Reported:

- public ensemble ≈ 0.940
- extended pipeline ≈ 0.941
- Gold OOF rank blend ≈ 0.9078

Gold set results:

| Model | Macro AUC |
|---|---:|
| RadImageNet ResNet50 | 0.8955 |
| ImageNet ResNet50 | 0.8892 |
| 50/50 rank blend | 0.9078 |
| LLM-derived labels | 0.8920 |

Important finding:

> Model diversity matters more than adding many highly correlated models.

Reported prediction correlation between two useful arms ≈ 0.756।

---

# 15. Kaggle Discussions — Important Findings

## 15.1 Official label definitions

Host overview:  
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343

Important because targets clinical semantics-driven।

## 15.2 Report vs MRI label mismatch

Discussion:  
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826

Key insight:

> Report labels and official MRI-derived labels disagree naturally.

## 15.3 DICOM preprocessing warning

Discussion:  
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734105

### Key issues

#### Wrong slice sorting

Never sort by:

```text
filename
SOPInstanceUID
```

Correct approach:

```text
ImageOrientationPatient
+
ImagePositionPatient
```

Physical projection along slice-plane normal দিয়ে order।

#### Fixed physical crop

Plain pixel resize different scanner FOV-তে anatomy distort করতে পারে। Better:

```text
fixed physical mm crop
→ resize
```

Example:

```text
130–140 mm crop
```

#### Laterality normalization

Medial/Lateral labels-এর জন্য left/right normalization useful হতে পারে।

#### 18 slices as 18 channels — avoid

Pretrained ImageNet/DINO input conv modify করলে representation damage হতে পারে। Better:

```text
slice i-1
slice i
slice i+1
```

RGB-like input হিসেবে shared 2.5D encoder।

---

# 16. Slice Geometry Experiment

Controlled experiment reported:

| Setup | OOF |
|---|---:|
| 192px, no crop, 24 equally spaced | 0.7695 |
| 288px, 140mm crop, 12 adjacent | 0.7777 |
| 288px, 140mm crop, center-9 | 0.7815 |

Related discussion:  
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737597

### Important finding

Adjacent / anatomically focused slices > wide-spaced slices। Meaning:

> What the network sees matters more than simply increasing resolution.

---

# 17. Training Trick Experiment

Same discussion reported approximate effects:

| Trick | Effect |
|---|---:|
| EMA | +0.003 |
| Mixup | +0.002 |
| Longer training | +0.001 |
| Asymmetric loss | -0.139 |

Conclusion:

> Fancy losses are not automatically useful।

---

# 18. Bigger Backbone Experiment

DINOv2-Small → DINOv2-Base

Approx:

- 22M → 87M params
- compute ~4x
- OOF gain ~+0.0011

Noise floor around ~0.002।

Discussion:  
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735154

Conclusion:

> Bigger backbone is not currently the highest ROI direction।

---

# 19. Scanner / Metadata Leakage

Important experiment:

Random CV metadata-only:

```text
Macro AUC ≈ 0.6516
```

Scanner-grouped CV:

```text
Macro AUC ≈ 0.5981
```

Series composition only:

```text
≈ 0.5954
```

Discussion:  
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733517

Difference:

```text
~0.053 AUC
```

This strongly suggests site/scanner shortcut risk।

---

# 20. Strong Single Model Evidence

Discussion:  
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304

Reported:

```text
Single model raw ≈ 0.942
With TTA ≈ 0.943
Resolution ≈ 288x288
```

This is strategically important। It suggests:

> 20-model ensemble is not mandatory for 0.943-level public performance।

---

# 21. Real Competition Bottleneck

Discussion:  
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735826

High-ranking participants highlighted:

- label generation quality
- soft labels
- not blindly copying public code
- report threshold mismatch

Conclusion:

> Label quality is likely one of the dominant bottlenecks।

---

# 22. Data Quality Issues

Reported corrupted DICOM files:

https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737163

Production loader should:

```python
try:
    decode_dicom()
except Exception:
    log_bad_file()
    skip()
```

---

# 23. External Resources

## 23.1 MRNet

Paper:  
https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699

Dataset:  
https://stanfordmlgroup.github.io/competitions/mrnet/

Useful for:

- multi-plane knee MRI modeling
- ACL / meniscus classification
- slice aggregation

Competition usage requires license/rule verification।

## 23.2 RadImageNet

GitHub:  
https://github.com/BMEII-AI/RadImageNet

Useful pretrained CNN backbones:

- ResNet50
- DenseNet121
- InceptionV3
- InceptionResNetV2

Strong complementary branch for DINO-based models।

## 23.3 RAD-DINO

Hugging Face:  
https://huggingface.co/microsoft/rad-dino

Important caveat: RAD-DINO mainly chest X-ray domain। So knee MRI-তে guaranteed winner নয়। Still useful as diversity experiment।

## 23.4 BiomedCLIP

Paper:  
https://arxiv.org/abs/2303.00915

Potential use:

- image-text alignment
- train-time report supervision
- auxiliary representation learning

---

# 24. Recommended Validation Strategy

## Primary Strategy

Use:

> **5-fold scanner/site-grouped multi-label cross-validation**

Not:

```text
random StratifiedKFold
```

### Group by

- StudyInstanceUID
- scanner fingerprint
- vendor/model
- acquisition protocol proxy
- field strength if available
- near-duplicate report groups

---

# 25. Recommended Fold Construction

Conceptual:

```text
Group-aware
+
multi-label stratification
+
study-disjoint
+
scanner/site-aware
+
report-hash-aware
```

Persist one fixed split file:

```text
StudyInstanceUID
scanner_group
report_group
fold
is_gold
```

Competition শেষ হওয়া পর্যন্ত folds change করবেন না।

---

# 26. Gold 58 Usage

Do NOT tune heavily on 58 gold cases। Use gold only as:

- sanity check
- external validation
- label-quality audit
- final frozen check

Main model selection:

```text
weak-label grouped OOF
```

Gold-58 repeated tuning = overfitting risk।

---

# 27. Duplicate / Near-Duplicate Detection

## Reports

- normalized text hash
- SimHash
- MinHash
- embedding similarity

## Images

- perceptual hash
- downsampled pixel fingerprint
- DINO embedding nearest-neighbor

## Metadata

- identical acquisition parameters
- repeated StudyUID/SeriesUID
- duplicate slices

---

# 28. OOF Prediction Design

For every experiment store:

```text
StudyInstanceUID
fold
scanner_group
target_1 ... target_12
pred_1 ... pred_12
model_name
seed
```

Calculate:

- macro AUC
- per-target AUC
- fold std
- seed std
- scanner-group AUC
- model correlation

---

# 29. Detecting Public Leaderboard Overfitting

Example:

| Experiment | Group CV | Public LB |
|---|---:|---:|
| Base | 0.880 | 0.940 |
| X | 0.884 | 0.941 |
| Y | 0.879 | 0.944 |

Keep:

```text
X
```

Reject:

```text
Y
```

Reason: Y improves public LB but worsens reliable local validation।

---

# 30. Detecting Train/Test Distribution Shift

Since hidden test unavailable, use proxies।

## 30.1 Scanner/site holdout

Train:

```text
sites A/B/C
```

Validate:

```text
site D
```

## 30.2 Protocol holdout

Group by:

- pixel spacing
- orientation
- field strength
- scanner model
- sequence composition

## 30.3 Embedding clustering

Use pretrained encoder। Cluster studies। Check fold score per cluster।

---

# 31. Detecting Leakage

Automated checks:

| Leakage Type | Test |
|---|---|
| Study leakage | StudyInstanceUID uniqueness across folds |
| Series leakage | group all series by study |
| Report leakage | normalized hash grouping |
| Scanner leakage | random vs scanner-grouped delta |
| Gold leakage | gold excluded from iterative tuning |
| Pseudo-label leakage | OOF teacher generation only |
| Near-duplicate image | pHash / embedding neighbor |
| Metadata shortcut | metadata-only baseline |
| Public LB leakage | no weight search solely from LB |

---

# 32. Seed Sensitivity

Run serious experiments with:

```text
seed 42
seed 3407
seed 2026
```

If gain:

```text
+0.0015
```

but seed variation:

```text
±0.003
```

then improvement not proven।

---

# 33. Experiment Roadmap

| ID | Hypothesis | Change | Model | Validation | Expected CV | Expected LB | Risk | Cost | Decision |
|---|---|---|---|---|---|---|---|---|---|
| E0 | Public baseline reproducible | Reproduce 0.943 + 0.941 pipeline | Public stack | LB + OOF audit | Reference | Reference | Low | Med | Must |
| E1 | Random CV leaks scanner | Build grouped CV | — | Scanner/site 5-fold | More honest | Neutral | Low | Low | Keep |
| E2 | Geometry is limiting | Physical sort + mm crop + adjacent slices | Same backbone | Fixed folds | High | High | Low | Med | Keep if stable |
| E3 | Strong simple single model | RadImageNet / DINO | Single model | Fixed folds | High | Medium-High | Low | Med | Keep |
| E4 | Training recipe gain | LR/EMA/Mixup/etc | Same model | Same folds | Small | Small | Medium | Med | Keep if > noise |
| E5 | Model diversity matters | CoAtNet / ConvNeXt / RAD-DINO | Alt model | Same folds | Medium | Medium | Medium | Med | Keep if low corr |
| E6 | OOF blend improves | Rank blend | Ensemble | OOF | High | High | Low | Low | Keep |
| E7 | Rank averaging beats prob avg | Rank normalize | Ensemble | OOF | Small-Med | Medium | Low | Low | Likely keep |
| E8 | Learned stacking helps | Ridge/logistic stacking | Meta-model | Nested OOF | Medium | Medium | High | Low | Strict |
| E9 | Pseudo-label refinement | High-confidence self-training | Teacher-student | OOF | Uncertain | Uncertain | High | High | Conditional |
| E10 | Diverse final ensemble | DINO + CNN + ROI | Ensemble | Grouped CV | Highest | High | Medium | High | Final |

---

# 34. E0 — Reproduce Baselines

## E0-A

Reproduce 0.943 public inference stack। Purpose:

- inference pipeline
- test loader
- runtime
- dependency compatibility
- public reference

## E0-B

Reproduce NTejas 0.941 training pipeline। Purpose:

- label extraction
- training
- OOF
- rank blending
- independent checkpoints

---

# 35. E1 — Validation Upgrade

Build:

```text
5-fold scanner/site grouped CV
```

with:

- multilabel balance
- StudyInstanceUID grouping
- report template grouping
- optional acquisition fingerprint grouping

This is the most important infrastructure experiment।

---

# 36. E2 — Geometry / Preprocessing Experiments

## E2a
Filename sorting → physical DICOM sorting

## E2b
Plain resize → fixed 130/140mm crop

## E2c
Equal-spaced slices → adjacent 2.5D slices

## E2d
Low-quality / empty slice filtering

## E2e
Left-right orientation normalization

## E2f
Sequence quality scoring

---

# 37. E3 — Strong Single Model

Start with:

### Model A
RadImageNet ResNet50

### Model B
EfficientNet-B0/B2

### Model C
DINOv2-S

Same:

- folds
- preprocessing
- augmentation
- labels
- optimizer

Goal:

```text
find independent strong OOF signal
```

---

# 38. E4 — Hyperparameter Optimization

Only after stable baseline। Search:

- learning rate
- weight decay
- dropout
- EMA
- Mixup
- cosine schedule
- epochs
- augmentation strength
- input size

Do not tune all at once।

---

# 39. E5 — Alternative Model Family

Candidates:

- ConvNeXt
- CoAtNet
- EfficientNet
- RAD-DINO
- RadImageNet CNN
- DINOv2
- ROI CNN

Selection metric:

```text
OOF AUC
+
low correlation with current ensemble
```

---

# 40. Target-Aware Modeling

Group targets:

## Ligament
- ACL
- MCL

## Meniscus
- Medial Meniscus
- Lateral Meniscus

## OA
- Medial OA
- Lateral OA
- PF OA

## Fluid / inflammation
- Effusion
- Synovitis
- Baker's cyst

## Bone injury
- Contusion
- Fracture

Different target families should be allowed to attend to different views।

---

# 41. Recommended Multi-View Architecture

```text
Sagittal encoder ─┐
                  │
Coronal encoder ──┼──> Sequence-aware MIL
                  │           │
Axial encoder ────┘           ↓
                        Target-specific attention
                                │
                                ↓
                            12 logits
```

This is likely more anatomically meaningful than one generic pooled feature।

---

# 42. Most Promising Unexplored Idea #1 — ROI Specialists

Local pathologies:

- meniscus
- MCL
- PF OA

may benefit from targeted crops। Architecture:

```text
Global branch
+
Meniscus / joint-line ROI branch
+
Patellofemoral ROI branch
```

Potential advantage:

- higher local resolution
- lower compute
- more target-specific signal

---

# 43. Most Promising Idea #2 — In-Domain Self-Supervised Pretraining

Use all competition MRI images। Pipeline:

```text
MRI images
   ↓
DINO / MAE self-supervised adaptation
   ↓
weak-label supervised fine-tuning
```

Benefits:

- reduce ImageNet domain gap
- exploit unlabeled image signal
- improve representation without gold labels

---

# 44. Most Promising Idea #3 — Weak Label Uncertainty

Instead of hard labels:

```text
negative
likely negative
uncertain
likely positive
positive
```

Map to soft values:

```text
0.02
0.20
0.50
0.80
0.98
```

or confidence weights। Particularly useful for:

- minimal vs moderate effusion
- low-grade ligament injuries
- degeneration vs tear
- mild OA
- contusion vs non-traumatic marrow edema

---

# 45. Most Promising Idea #4 — Multi-Teacher Labeling

Use:

```text
regex teacher
+
local multilingual LLM A
+
local multilingual LLM B
+
host diagnostic rules
```

If teachers agree:

```text
high confidence
```

If disagree:

```text
low training weight
```

---

# 46. Most Promising Idea #5 — Report as Privileged Information

Train-time only:

```text
MRI embedding
   ↔
Report embedding
```

Possible objectives:

- contrastive learning
- auxiliary report prediction
- teacher distillation

Inference:

```text
MRI only
```

This may improve representation without requiring report at test time।

---

# 47. E6 — OOF Blending

For each model and target:

\[
r_{m,t} = \frac{rank(p_{m,t})}{N}
\]

Final:

\[
p_t = \sum_m w_{m,t}r_{m,t}
\]

Start:

```text
equal weights
```

Then optional target-specific OOF weights।

---

# 48. E7 — Rank Averaging

Because evaluation is AUC:

```text
probability scale is less important
```

Rank normalization reduces calibration mismatch across models। Recommended default ensemble method।

---

# 49. E8 — Stacking

Use only with nested CV। Inputs:

```text
DINO OOF
RadImageNet OOF
CoAtNet OOF
ROI OOF
```

Meta models:

- logistic regression
- ridge
- very small MLP

Risk:

> easy to overfit with noisy labels

Rank averaging should be baseline।

---

# 50. E9 — Pseudo-Label Refinement

Do not aggressively self-train। Candidate:

```text
report teacher = uncertain
image OOF ensemble = very confident
multiple seeds agree
```

Then adjust training target cautiously। Example:

```text
Report label = 0.45
OOF ensemble = 0.97
3 seeds agree
```

Potentially useful। High overfit/noise amplification risk।

---

# 51. Recommended Final Ensemble

```text
                       MRI Study
                           │
              Geometry-safe preprocessing
                           │
          ┌────────────────┼─────────────────┐
          │                │                 │
      DINOv2-S      RadImageNet CNN     ROI Specialist
          │                │                 │
          └──────────── OOF ranks ───────────┘
                           │
                  Target-wise rank blend
                           │
                     Limited D4 TTA
                           │
                           ↓
                    12 probabilities
```

Optional fourth branch:

```text
CoAtNet
```

only if OOF correlation adds useful diversity।

---

# 52. Top 10 Experiments to Run

1. Scanner/site-grouped CV
2. Physical DICOM slice ordering
3. Fixed-mm crop + adjacent slice-window ablation
4. Report-label quality: regex vs LLM vs consensus
5. RadImageNet ResNet50 strong single model
6. DINOv2-S strong single model
7. Target-aware multi-sequence attention
8. Meniscus/MCL/PF ROI specialist
9. DINO + CNN + ROI OOF rank ensemble
10. In-domain self-supervised adaptation

---

# 53. Approaches to Avoid

## Avoid random KFold
Scanner/site leakage risk।

## Avoid Gold-58 score chasing
Sample too small।

## Avoid notebook title score trust
Always verify Kaggle Best Score।

## Avoid blindly forking 0.943 stack
Shared lineage / public overfit risk।

## Avoid huge backbone first
Evidence shows very small gain vs much larger compute।

## Avoid plain filename sorting
Anatomically wrong ordering।

## Avoid 18 slices as channels
Breaks pretrained interface।

## Avoid per-target LB-only blend tuning
Direct public overfitting।

## Avoid aggressive pseudo-label iteration
Can amplify teacher errors।

## Avoid unverified external MRI datasets
Check Kaggle rules / licensing first।

---

# 54. Final Comparison Table

| Rank | Approach/Notebook | Public LB | CV | Model | Key Technique | Overfit Risk | Reproducible? |
|---:|---|---:|---:|---|---|---|---|
| 1 | 0.943 Public Stack | 0.943 | Unverified | DINO+A5+RadImageNet+Raptor+CoAtNet | large rank/blend ensemble | Med-High | Yes |
| 1 | DINOsaur/Public forks | 0.943 | Unverified | DINO-heavy ensemble | public asset reuse | High | Yes |
| 1 | D4 Fine-Tuned Stack | 0.943 | Unverified | mixed ensemble | D4/TTA/target blend | Med-High | Yes |
| 4 | Restructured Stack | 0.942 | Unverified | mixed | refactored stack | Medium | Yes |
| 5 | NTejas End-to-End | 0.941 | 0.9078 gold blend | RadImageNet + ImageNet + public stack | decorrelated rank blend | Medium | Strong Yes |
| 6 | Tony DINO-RadImageNet | 0.940 | Unverified | DINO+CNN+Raptor+CoAtNet | target-wise rank blend | Medium | Yes |
| 7 | Anhad Ensemble | 0.936 | Unverified | DINOv2 | ensemble | Medium | Yes |
| 8 | DINO/RadImageNet/dual CoAtNet | 0.934 | Unverified | mixed | model diversity | Medium | Yes |
| 9 | Roman EDA Baseline | 0.894 | — | DINO | MRI structure/geometry | Low-Med | Yes |
| 10 | Simple ResNet Baseline | 0.798 | — | ResNet18 | report weak labels | Low | Yes |

---

# 55. Best Current Public Approach

By verified public notebook score:

> **0.943 DINO / A5 / RadImageNet / Raptor / CoAtNet stack**

But best development baseline:

> **NTejas 0.941 end-to-end reproducible pipeline**

Reason:

- training code available
- OOF available
- label generation documented
- blend logic reproducible

---

# 56. Best Validation Strategy

Recommended:

```text
5-fold scanner/site-grouped multi-label CV
+
report-hash grouping
+
3 seeds for serious experiments
+
per-target AUC
+
fold variance
+
scanner-group variance
+
OOF model correlation
```

Gold 58:

```text
secondary audit only
```

Public LB:

```text
occasional confirmation only
```

---

# 57. Most Promising Unexplored Ideas

1. Better uncertainty-aware weak labels
2. Multi-teacher label consensus
3. Physical-mm preprocessing
4. ROI specialists
5. In-domain SSL
6. Target-specific sequence routing
7. Cross-plane MIL
8. Train-time report distillation
9. Scanner-invariant training
10. Correlation-aware ensemble selection

---

# 58. Recommended Experiment Decision Rule

For each experiment compare:

```text
Δ grouped CV
Δ per-target AUC
fold std
seed std
scanner holdout
model correlation
runtime
public LB
```

### KEEP

If:

```text
CV improves
and
stable across folds/seeds
and
no leakage
```

### REJECT

If:

```text
Public LB improves
but
Grouped CV worsens
```

### NEEDS MORE EVIDENCE

If:

```text
gain < noise floor
```

---

# 59. Step-by-Step Strategy to Beat Current Public Solutions

## Phase A — Reproduce

1. Reproduce public 0.943 inference stack
2. Reproduce NTejas 0.941 pipeline
3. Verify runtime and loader

## Phase B — Lock Validation

4. Build fixed grouped folds
5. Freeze them permanently

## Phase C — Improve Labels

6. Encode host diagnostic definitions
7. Extract soft labels
8. Use multi-teacher agreement
9. Keep uncertainty weights

## Phase D — Fix MRI Geometry

10. Physical slice sorting
11. Fixed-mm crop
12. 2.5D adjacent slices
13. Sequence quality selection

## Phase E — Strong Independent Model

14. Train RadImageNet ResNet50
15. Train DINOv2-S
16. Compare OOF + correlations

## Phase F — Attack Weak Targets

17. Meniscus specialist
18. MCL specialist
19. PF OA ROI branch
20. fluid-sensitive specialist

## Phase G — Diversity

21. Add only low-correlation models
22. Avoid redundant DINO forks

## Phase H — Ensemble

23. Equal rank blend
24. target-specific OOF weights
25. optional regularized stacking

## Phase I — Sparse Kaggle Submission

26. Submit only CV-backed improvements
27. Never optimize weights directly from public LB

## Phase J — Final Submission

Keep two:

### Submission A
CV-best conservative ensemble

### Submission B
slightly more public-LB-friendly variant

Primary should be:

> **CV-best conservative ensemble**

because private leaderboard is 70%.

---

# 60. Final Strategic Priority

Recommended priority order:

```text
Label quality
>
MRI geometry
>
Validation quality
>
Model diversity
>
Architecture size
>
Public leaderboard tuning
```

The most promising route to outperform the current public 0.943 stack is not merely adding more public DINO checkpoints.

A better route is:

```text
higher-quality soft labels
+
correct physical MRI preprocessing
+
strong independent single models
+
target-aware specialists
+
low-correlation OOF rank ensemble
+
scanner/site-safe validation
```

---

# 61. Final Recommendation

If the goal is a strong private leaderboard result:

### Start with

- NTejas reproducible training pipeline
- Arin / Aman public 0.943 stack as inference reference

### Then focus research on

- better weak labels
- scanner/site-safe CV
- physical MRI geometry
- ROI / target-aware specialists
- decorrelated model families
- OOF rank blending

### Avoid

- random split
- Gold-58 tuning
- public-LB-only optimization
- oversized models without CV gain
- redundant public forks

---

# 62. Key Resources

## Kaggle

Competition:  
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection

Data:  
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data

Code:  
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/code

Discussions:  
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion

Host overview:  
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343

Report-vs-label discussion:  
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826

DICOM preprocessing:  
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734105

Scanner leakage:  
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733517

Single-model discussion:  
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304

Bottleneck discussion:  
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735826

Slice geometry:  
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737597

Backbone scaling:  
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735154

Corrupted DICOM:  
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737163

## GitHub

NTejas:  
https://github.com/NTejas-1/RSNA-Knee-Abnormality-Detection

RadImageNet:  
https://github.com/BMEII-AI/RadImageNet

## Papers / Models

MRNet:  
https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699

MRNet dataset:  
https://stanfordmlgroup.github.io/competitions/mrnet/

RAD-DINO:  
https://huggingface.co/microsoft/rad-dino

BiomedCLIP:  
https://arxiv.org/abs/2303.00915

---

# 63. Bottom Line

**Best verified public score:** ~0.943  
**Best reproducible research baseline:** NTejas ~0.941  
**Best validation:** scanner/site-grouped multi-label 5-fold CV  
**Biggest bottleneck:** weak-label quality + MRI preprocessing  
**Highest-risk mistake:** public LB optimization without trustworthy CV  
**Most promising direction:** better labels + physical MRI geometry + target-aware independent models + OOF rank ensemble

