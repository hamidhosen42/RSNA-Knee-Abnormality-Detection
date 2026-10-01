#  Benchmarked the public report-label tables against the 58 annotated studies

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737454
- **Author:** **starkhushi** (@starkhushi, Expert) · _topic author_ · 2026-08-25 15:52 UTC · LB rank 609
- **Topic votes:** 0 · **Comments/replies:** 0 · **Pinned:** no
- **Last comment:** 2026-08-25 15:52 UTC by starkhushi
- **Cloned:** 2026-10-01 19:24 UTC (topic id 737454, raw JSON `_raw/737454.json`)

---

Since only 58/4,407 studies carry radiologist labels, everyone is training on report-derived tables — but I couldn't find a comparison of them, so I measured each against those 58 studies.

Macro AUC vs radiologists: stevenleehans v4 blend 0.893, stevenleehans v2 0.887, pilkwang report_labels_v2 0.870. Averaging two independent tables beats either alone on several labels.

Per-label the spread is huge — ACL ~0.99, Synovitis ~0.79 (only ~16% of reports mention synovitis in any of the nine languages, so it's largely inferred). Worth knowing before you blame your architecture for a weak Synovitis score.

Notebook with the code and per-label table: [https://www.kaggle.com/code/starkhushi/rsna-knee-which-report-labels-should-you-train-on]

Thanks to both dataset authors — all credit for the tables is theirs.

---

## Comments (0, threaded, oldest first)

_No comments._
