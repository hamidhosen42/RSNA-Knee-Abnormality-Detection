# train.csv: PatientSex is documented but not present in the file

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733423
- **Author:** **epicfangs** (@epicfangs, Contributor) · _topic author_ · 2026-08-06 20:26 UTC · ▲2
- **Topic votes:** 2 · **Comments/replies:** 3 · **Pinned:** no
- **Last comment:** 2026-08-07 14:54 UTC by SpeedSci
- **Cloned:** 2026-10-01 19:24 UTC (topic id 733423, raw JSON `_raw/733423.json`)

---

The Data page describes train.csv as containing StudyInstanceUID, PatientSex and Report alongside the twelve labels. The file itself has 14 columns: StudyInstanceUID, Report, and the 12 labels. There is no PatientSex column.

Could the hosts confirm whether PatientSex was intended to be included and was dropped during export, or whether the data description should be updated instead?

Not a blocker either way — just flagging it early so nobody writes ingest code against a column that isn't there.

Thanks for putting together an interesting dataset.

---

## Comments (3, threaded, oldest first)

### [1] Po-Hao "Howard" Chen

**Po-Hao "Howard" Chen** (@javaduke95, Contributor) · _competition host_ · 2026-08-06 22:45 UTC · ▲1

Hello!

Yes we did intend to have PatientSex be available - at one point we indeed built it into the CSV but since patient sex is also in the DICOM header from image files directly, we felt it's redundant and removed the column from CSV.

We will update the data page to reflect this more accurately. Good call out to avoid confusion. 

Thank you so much for the kind words.

#### ↳ [1.1] SpeedSci

**SpeedSci** (@caoyupeng, Expert) · 2026-08-07 14:54 UTC · LB rank 1073

Are you going to modify the training data?

### [2] Fabian Friedland - xBrain

**Fabian Friedland - xBrain** (@fabotelli, Contributor) · 2026-08-07 00:37 UTC · LB rank 172

The literature indicate a possible strong AUC between frequency and MCL.
