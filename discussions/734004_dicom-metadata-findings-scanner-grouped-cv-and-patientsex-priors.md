# DICOM metadata findings: scanner-grouped CV and PatientSex priors

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734004
- **Author:** **morningduck** (@morningduck, Contributor) · _topic author_ · 2026-08-09 13:40 UTC · ▲6 · LB rank 2261
- **Topic votes:** 6 · **Comments/replies:** 0 · **Pinned:** no
- **Last comment:** 2026-08-09 13:40 UTC by morningduck
- **Cloned:** 2026-10-14 18:23 UTC (topic id 734004, raw JSON `_raw/734004.json`)

---

## Scanner-grouped vs. random CV splits

The competition DICOMs carry `Manufacturer` (Siemens / Philips / GE) and
`MagneticFieldStrength` (1.5T / 3T). There are ~13 distinct scanner groups
across the 4,407 training studies:

| Scanner Group   | Studies |
|-----------------|---------|
| Siemens_1.5T    | 1,148   |
| Siemens_3T      | 781     |
| GE_1.5T         | 698     |
| Philips_3T      | 663     |
| Philips_1.5T    | 619     |

A standard random K-fold split places the same scanner in both train and
val folds. To measure the effect, I trained a metadata-only classifier
(no pixels) and evaluated it under both split types:

| Split type      | Macro-AUC |
|-----------------|-----------|
| Random 5-fold   | 0.652     |
| Scanner-grouped | 0.598     |

The gap is ~0.05 from metadata alone. OA targets showed the largest drop
(0.07–0.09), consistent with field-strength-dependent cartilage contrast
differences.

## PatientSex in test DICOM headers

`PatientSex` (tag 0010,0040) is present in the test set DICOM headers.

Training distribution: M=2,076 / F=1,894.

| Target      | M prevalence | F prevalence |
|-------------|-------------|-------------|
| ACL         | ~54%        | ~32%        |
| Medial OA   | ~12%        | ~45%        |

Males show higher rates of traumatic injuries (ACL, MCL, Fracture); females
show higher rates of degenerative changes (OA).

## Other metadata available in test headers

`Manufacturer`, `MagneticFieldStrength`, `SeriesDescription`, and
`ImageOrientationPatient` are all readable via `pydicom` at inference time.

---

## Comments (0, threaded, oldest first)

_No comments._
