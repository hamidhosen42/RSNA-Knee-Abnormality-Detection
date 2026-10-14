# EDA: labels, co-occurrence and MRI views (official data only)

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747865
- **Author:** **Eesh saxena** (@eeshsaxena, Contributor) · _topic author_ · 2026-10-09 19:38 UTC · ▲1 · LB rank 286
- **Topic votes:** 1 · **Comments/replies:** 0 · **Pinned:** no
- **Last comment:** 2026-10-09 19:38 UTC by Eesh saxena
- **Cloned:** 2026-10-14 18:23 UTC (topic id 747865, raw JSON `_raw/747865.json`)

---

Hi all,

I put together an EDA notebook for this competition that might help anyone getting started: https://www.kaggle.com/code/eeshsaxena/rsna-knee-eda-labels-and-mri-views

A few things that stood out to me:

Only 58 training studies come with labels. Effusion is the most common finding (about 60%) and MCL the rarest (about 16%). With so few studies the 95% intervals span roughly 20 to 25 percentage points, so local validation will be noisy.
None of the labelled studies is completely normal. A study has about four positive findings on average, and 54 of the 58 studies have a label combination of their own, so there is no small set of typical patterns.
The data is big and varied: 24,371 series and about 819k DICOM files, with a real mix of scanners, field strengths, slice thicknesses and planes.
The data page lists JPEG Lossless and JPEG 2000 transfer syntaxes. The files I sampled were uncompressed, but the default Kaggle image cannot decode the compressed ones without pylibjpeg or gdcm, so it is worth guarding for that in a submission notebook.

It runs on CPU in about two minutes and uses only the official files. If you spot anything wrong or want another plot added, let me know in the comments.

---

## Comments (0, threaded, oldest first)

_No comments._
