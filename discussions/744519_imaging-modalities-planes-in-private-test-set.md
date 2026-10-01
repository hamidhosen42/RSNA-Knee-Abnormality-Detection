# Imaging modalities / planes in private test set

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744519
- **Author:** **Satwik** (@p4rallax, Master) · _topic author_ · 2026-09-30 05:50 UTC · ▲1 · LB rank 1304
- **Topic votes:** 1 · **Comments/replies:** 1 · **Pinned:** no
- **Last comment:** 2026-10-01 13:53 UTC by Rubens Sosa
- **Cloned:** 2026-10-01 19:24 UTC (topic id 744519, raw JSON `_raw/744519.json`)

---

Apologies if this has already been mentioned / answered somewhere else, but do all studies in the test set have all planes / modalities (T2/ FS / PD, etc)? Or are there cases where one/many of these is unavailable in the test set?

---

## Comments (1, threaded, oldest first)

### [1] Rubens Sosa

**Rubens Sosa** (@rubenssosa, Contributor) · 2026-10-01 13:53 UTC · LB rank 4691

@javaduke95 Following up on this topic: can you confirm whether every hidden test scan contains all three planes (sagittal/coronal/axial) and at least one fluid-sensitive series, or should the pipelines be prepared for scans with missing planes/sequences? A yes/no answer regarding whether "all three planes are always present" would suffice
