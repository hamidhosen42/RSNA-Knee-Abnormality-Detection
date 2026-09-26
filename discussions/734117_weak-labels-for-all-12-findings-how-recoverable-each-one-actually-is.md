# Weak labels for all 12 findings + how recoverable each one actually is

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734117
- **Author:** **Luka Duvanov** (@nekkon, Contributor) · _topic author_ · 2026-08-10 01:10 UTC · ▲1 · LB rank 4183
- **Topic votes:** 1 · **Comments/replies:** 0 · **Pinned:** no
- **Last comment:** 2026-08-10 01:10 UTC by Luka Duvanov
- **Cloned:** 2026-09-26 16:24 UTC (topic id 734117, raw JSON `_raw/734117.json`)

---

Follow-up to the earlier notebook. This one extracts all twelve findings from all 4,407 reports and writes weak_labels.csv as notebook output, so you can attach it directly.

The more useful half is the per-finding measurement on the 58 gold studies. Balanced accuracy ranges from 0.82 (Baker's cyst) to 0.56 (medial meniscus), and the ordering is not random — it tracks what kind of thing the finding is:

Named objects (Baker's cyst, ACL) work: present or absent, one word per language.
Graded severities (effusion) fail: "minimal joint fluid", "trace effusion" — a binary keyword has nowhere to put the adjective. Getting effusion right means learning the annotators' severity threshold, which is a modelling problem, not a vocabulary one.
Unstated inferences (fracture) fail hardest: 0.93 specificity but 0.44 sensitivity, because more than half are described by appearance without the word ever appearing.
Also worth knowing: the extractor finds 2.6 findings per study where the annotators recorded 4.1, and returns nothing at all for 23% of studies. Treat those as unlabelled rather than as all-negative.

Vocabulary corrections are the most useful thing anyone can leave in the comments, especially for Turkish and Greek.

https://www.kaggle.com/code/nekkon/weak-labels-for-all-12-knee-mri-findings

---

## Comments (0, threaded, oldest first)

_No comments._
