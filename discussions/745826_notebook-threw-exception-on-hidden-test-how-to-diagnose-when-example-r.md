# Notebook Threw Exception on hidden test: how to diagnose when example run passes?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745826
- **Author:** **joshi malav** (@joshimalav, Contributor) · _topic author_ · 2026-10-04 16:43 UTC · ▲-1
- **Topic votes:** -1 · **Comments/replies:** 0 · **Pinned:** no
- **Last comment:** 2026-10-04 16:43 UTC by joshi malav
- **Cloned:** 2026-10-09 09:46 UTC (topic id 745826, raw JSON `_raw/745826.json`)

---

My latest submission (Version 4) fails with "Notebook Threw Exception", but the visible saved-notebook logs only show a successful run on the three example test studies. I have not found a traceback from the hidden scoring run.

The submission uses GPU with internet disabled, loads attached weights and offline decoder wheels, predicts every row in test.csv, and validates submission.csv (study IDs/order, all 12 columns, finite probabilities in [0, 1]). Training and diagnostic execution cells were removed from Version 4.

Checks completed separately:
- Inference on 100 training studies: no failures.
- Header audit of 1,868 training series and decoding checks on 68 selected series: no failures, but all inspected series were uncompressed.
- Actual pydicom sample decoding passed for JPEG Lossless SV1 and JPEG 2000 lossless/lossy using pylibjpeg.
- Environment: Python 3.12, pydicom 3.0.2, NumPy 2.0.2, pylibjpeg 2.1.0, pylibjpeg-libjpeg 2.4.0, pylibjpeg-openjpeg 2.6.0.

These checks have not reproduced the hidden-test exception, and I am not claiming a platform-wide issue. Is there a way to view the hidden-run exception or failing stage? Are there documented differences in test-series metadata or DICOM structure that should be checked beyond the examples? Suggestions for a targeted diagnostic would be appreciated.

No competition images, reports, or weights are included in this post.

---

## Comments (0, threaded, oldest first)

_No comments._
