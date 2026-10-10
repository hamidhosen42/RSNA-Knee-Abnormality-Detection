# Clarification on FrameOfReferenceUID in the released DICOM files

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/736457
- **Author:** **Enai** (@enaimaan, Contributor) · _topic author_ · 2026-08-20 21:17 UTC · LB rank 3044
- **Topic votes:** 0 · **Comments/replies:** 0 · **Pinned:** no
- **Last comment:** 2026-08-20 21:17 UTC by Enai
- **Cloned:** 2026-10-09 09:46 UTC (topic id 736457, raw JSON `_raw/736457.json`)

---

Hello,

While checking the consistency of the released DICOM metadata, I noticed that series grouped under the same study generally have distinct `FrameOfReferenceUID` values.

Could the organizers clarify whether this is:

1. a property of the original acquisitions, or
2. a result of the deidentification/export process?

Also, should spatial position and orientation fields be interpreted only within each individual series, or are their numerical values expressed in a common study-level patient coordinate system?

Thank you.

---

## Comments (0, threaded, oldest first)

_No comments._
