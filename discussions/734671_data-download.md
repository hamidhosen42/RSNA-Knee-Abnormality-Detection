# Data download

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734671
- **Author:** **OzanM.** (@analyticaobscura, Grandmaster) · _topic author_ · 2026-08-12 11:00 UTC · ▲4 · LB rank 219
- **Topic votes:** 4 · **Comments/replies:** 5 · **Pinned:** no
- **Last comment:** 2026-08-12 14:13 UTC by Enai
- **Cloned:** 2026-10-09 09:46 UTC (topic id 734671, raw JSON `_raw/734671.json`)

---

First of all, hello everyone. The competition data is quite large. Which method did those who downloaded it locally use? I have a few questions about this. Good luck to everyone!

---

## Comments (5, threaded, oldest first)

### [1] Giovanny Rodríguez

**Giovanny Rodríguez** (@giovannyrodrguez, Master) · 2026-08-12 13:12 UTC · LB rank 1880

mount it in read-only mode.

#### ↳ [1.1] OzanM.

**OzanM.** (@analyticaobscura, Grandmaster) · _topic author_ · 2026-08-12 13:42 UTC · ▲1 · LB rank 219

Thank you.

### [2] morningduck

**morningduck** (@morningduck, Contributor) · 2026-08-12 13:38 UTC · LB rank 2264

you can save a decent bit by transforming full DICOM data to uint8 compressed .npz files, there is a community dataset that's compressed like that https://www.kaggle.com/datasets/barun2104/rsna-knee-mri-processed-3d-volumes

#### ↳ [2.1] OzanM.

**OzanM.** (@analyticaobscura, Grandmaster) · _topic author_ · 2026-08-12 13:42 UTC · LB rank 219

Thank you. I did it that way too, but the size was a bit larger. I'll download this.

### [3] Enai

**Enai** (@enaimaan, Contributor) · 2026-08-12 14:13 UTC · ▲1 · LB rank 3044

I suggest you don't download it locally unless you plan to train locally. What I did was to have a script aggregate all non-pixel data (csv) in a Kaggle notebook and download it as a zip file. it is quite small I think, just a couple of GB max. AND for pixel data, download just a couple of studies to help you understand what's going on.
