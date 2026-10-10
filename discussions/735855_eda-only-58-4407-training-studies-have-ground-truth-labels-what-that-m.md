# EDA: Only 58/4407 training studies have ground-truth labels what that means for validation

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735855
- **Author:** **Istiyak Amin Santo** (@istiyakaminsanto, Contributor) · _topic author_ · 2026-08-18 13:15 UTC · ▲1 · LB rank 3209
- **Topic votes:** 1 · **Comments/replies:** 1 · **Pinned:** no
- **Last comment:** 2026-08-19 20:52 UTC by JeremyAM
- **Cloned:** 2026-10-09 09:46 UTC (topic id 735855, raw JSON `_raw/735855.json`)

---

Quick EDA that's worth knowing before you build a validation split.

**Label coverage:** `train.csv` has 4,407 rows, but only **58 (1.3%)** have all 12 label columns filled in. The other 4,349 have `Report` text but empty labels. If you carve a validation set from `train.csv` naively, you'll likely end up with very few (or zero) labeled examples in it — worth explicitly filtering to the 58 labeled rows for any clean eval.

**Label prevalence among the 58 labeled studies:**
| Label | Positive rate |
|---|---|
| Effusion | 60% |
| Synovitis | 47% |
| Medial Meniscus | 45% |
| ACL | 41% |
| Lateral Meniscus | 40% |
| PF OA | 36% |
| Contusion | 33% |
| Fracture | 31% |
| Medial OA | 26% |
| Baker's | 21% |
| Lateral OA | 19% |
| MCL | 16% |

**Reports are genuinely multilingual** confirmed at least English, Spanish, Greek, Turkish, Dutch, Bulgarian, Croatian, and German in a sample. If you're doing any NLP on `Report`, don't assume English.

**Series structure** (`train_series.csv`): median 5 series/study (range 3–14), split across Sagittal/Coronal/Axial planes. `Fluid_Sensitive` and `Fat_Suppression` are perfectly correlated in this dataset  only 6 distinct (plane, sequence-type) combos actually occur, despite the data description hedging that they're not always equivalent.

**DICOM notes:** image dimensions vary a lot even within one study (I've seen 640×640 up to 960×960, plus at least one non-square 640×1280), and transfer syntax is mixed (some uncompressed, some presumably JPEG-family per the data page)  resize and don't assume a fixed input shape.

Curious what others are finding on the label side anyone had luck deriving usable labels from the report text for the unlabeled majority?

---

## Comments (1, threaded, oldest first)

### [1] JeremyAM

**JeremyAM** (@jeremyam, Contributor) · 2026-08-19 20:52 UTC

The issue you mentioned with limited labels reminded me of what I saw in my own synthetic project. My synthetic AHT values covered a believable range, but the feature columns were way too uniform and didn’t reflect real‑world variation. I couldn’t find any meaningful relationships and the correlations were almost zero for everything.
I get what you mean about validation being unreliable when the labeled portion barely has any signal. I didn’t even bother splitting the data because there was no pattern for a model to pick up. The synthetic data in my dataset looked clean, but it wasn’t even useful. I guess your subset of the MRI dataset that was actually labeled is too small for reliable evaluation too.
I didn’t have any extra text fields to pull from, but your idea of trying to get labels from text fields sounded interesting and it wasn't something I had even considered for a case like this.
