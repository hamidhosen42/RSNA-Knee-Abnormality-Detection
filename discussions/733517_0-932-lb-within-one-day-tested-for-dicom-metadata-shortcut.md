# 0.932 LB within one day. Tested for DICOM metadata shortcut

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733517
- **Author:** **Oleksii Zhukov** (@zhukovoleksiy, Master) · _topic author_ · 2026-08-07 07:12 UTC · ▲18 · LB rank 2849
- **Topic votes:** 18 · **Comments/replies:** 2 · **Pinned:** no
- **Last comment:** 2026-08-08 14:05 UTC by agr hmmm
- **Cloned:** 2026-09-26 16:24 UTC (topic id 733517, raw JSON `_raw/733517.json`)

---

**TL;DR: DICOM metadata alone reaches 0.6515 macro AUC under random folds but only 0.598 across unseen scanners. Safe to say no meaningful shortcuts found. Leaderboard scores seem to reflect image reading.**

Public LB scores passed 0.8 and 0.9 within about a day of launch. That seemed strangely high for this label set in such short period, so I tested whether DICOM metadata alone could account for this.

It can't. Full header metadata, no pixels, reaches 0.6515 macro AUC under random folds and 0.5981 under scanner-grouped folds. The 0.053 increment is site memorization and it does not transfer to unseen scanners.

**Method**

Two probes, with the decision threshold written down before running anything:

Probe A: site identifiability. Cluster studies on Manufacturer + ManufacturerModelName + SoftwareVersions + ImagingFrequency + ReceiveCoilName. No labels needed. Result: 265 distinct fingerprints, top 20 covering 45.5% of studies.

Probe B: metadata => targets. HistGradientBoosting on study-level metadata features, targets = report-derived labels, scored under random 5-fold and under GroupKFold on the fingerprint above. The gap is the number of interest.

Features: series composition (plane × Fluid_Sensitive × Fat_Suppression counts, protocol signature string), plus per-study median/min/max of TR, TE, TI, flip angle, echo train length, pixel bandwidth, averages, phase encoding steps, slice thickness, spacing, pixel spacing, matrix size, field strength, ImagingFrequency, and categorical scanner/coil/software identity.

**Results:**

| Label            | Random | Site-grouped |  Drop |
| ---------------- | --------: | -----------: | ----: |
| ACL              |  0.705    |        0.670 | 0.035 |
| MCL              |  0.683    |        0.648 | 0.035 |
| Medial Meniscus  |  0.590    |        0.548 | 0.042 |
| Lateral Meniscus |  0.595    |        0.565 | 0.030 |
| Medial OA        |  0.652    |        0.578 | 0.074 |
| Lateral OA       |  0.637    |        0.563 | 0.074 |
| PF OA            |  0.680    |        0.599 | 0.081 |
| Effusion         |  0.628    |        0.582 | 0.046 |
| Synovitis        |  0.644    |        0.602 | 0.042 |
| Baker's          |  0.765    |        0.717 | 0.048 |
| Contusion        |  0.633    |        0.587 | 0.046 |
| Fracture         |  0.605    |        0.519 | 0.086 |
| **Macro**        |**0.6516** |   **0.5981** | **0.0534** |


Series composition alone (no DICOM reads at all) gives 0.5954. So the entire DICOM header pass adds 0.056 over the four columns already in `train_series.csv`

**Note:**
- Targets are report-derived, not expert annotations. Only 58 studies carry per-condition labels, which is too few to fit or validate against. So part of the 0.053 may be metadata predicting reporting style rather than disease.
- 265 fingerprints is finer-grained than institution. I'm separating individual scanners and software revisions within sites, so the grouped folds are stricter than true site holdout. That makes 0.053 an upper bound for a second, independent reason.


My notebook: https://www.kaggle.com/code/zhukovoleksiy/rsna-metadata-probe

The report extractor used to generate targets is from: https://www.kaggle.com/code/romanrozen/rsna-knee-data-structure-eda-baseline

---

## Comments (2, threaded, oldest first)

### [1] Handudu

**Handudu** (@handudu, Expert) · 2026-08-08 04:01 UTC · ▲2

Useful; at first, I also wondered if there was a shortcut.

### [2] agr hmmm

**agr hmmm** (@agrimrai29, Contributor) · 2026-08-08 14:05 UTC · ▲1 · LB rank 3151

0.932 LB within one day is too good tho.
