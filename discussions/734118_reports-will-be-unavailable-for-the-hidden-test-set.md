# reports will be unavailable for the hidden test set?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734118
- **Author:** **Nicolas Pantoja** (@nicolaspantoja, Contributor) · _topic author_ · 2026-08-10 01:24 UTC
- **Topic votes:** 0 · **Comments/replies:** 1 · **Pinned:** no
- **Last comment:** 2026-08-10 12:05 UTC by Po-Hao "Howard" Chen
- **Cloned:** 2026-10-14 18:23 UTC (topic id 734118, raw JSON `_raw/734118.json`)

---

hello, 

Could you please confirm whether radiology reports will be unavailable for the hidden test set? Since train.csv includes a Report column but test.csv only includes StudyInstanceUID, why is the competition considered multimodal? Is text intended only for training, while inference must rely solely on MRI images and series metadata?

---

## Comments (1, threaded, oldest first)

### [1] Po-Hao "Howard" Chen

**Po-Hao "Howard" Chen** (@javaduke95, Contributor) · _competition host_ · 2026-08-10 12:05 UTC

Confirmed.  Reports are not available for the hidden test set.
