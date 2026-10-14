# Data downloading problem

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742474
- **Author:** **Meerub Atiq** (@meerubatiq, Contributor) · _topic author_ · 2026-09-22 07:56 UTC · ▲-1
- **Topic votes:** -1 · **Comments/replies:** 1 · **Pinned:** no
- **Last comment:** 2026-09-22 08:41 UTC by Ángel Jacinto Sánchez Ruiz
- **Cloned:** 2026-10-14 18:23 UTC (topic id 742474, raw JSON `_raw/742474.json`)

---

Hi, I'm working on the RSNA Knee Abnormality Detection competition and I'm developing locally on a laptop with about 118 GB of free storage. The full competition data is ~570 GB, so downloading it locally isn't feasible for me.  I'm trying to understand the practical workflow competitors are using. Do you work directly with the competition input inside Kaggle, or do you create a reduced DICOM/pixel cache locally?  In particular, I'm interested in how you handle preprocessing and cache generation without first storing the entire raw dataset locally. Any advice or notebook references would be really helpful.

---

## Comments (1, threaded, oldest first)

### [1] Ángel Jacinto Sánchez Ruiz

**Ángel Jacinto Sánchez Ruiz** (@sacuscreed, Expert) · 2026-09-22 08:41 UTC · LB rank 269

You can download by chunks. I made public the notebook I'm using.
