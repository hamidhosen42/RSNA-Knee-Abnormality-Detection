# Rule 2.5.a.4 vs 2.8.a.1 — do weights trained on no-redistribution external data have to be published? (yes/no)

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743420
- **Author:** **Richard_61** (@yiliu15, Contributor) · _topic author_ · 2026-09-26 03:15 UTC · ▲3 · LB rank 40
- **Topic votes:** 3 · **Comments/replies:** 1 · **Pinned:** no
- **Last comment:** 2026-09-29 02:15 UTC by Richard_61
- **Cloned:** 2026-10-14 18:23 UTC (topic id 743420, raw JSON `_raw/743420.json`)

---

@javaduke95 A narrow follow-up on the winners' obligations. The same point was raised in [#737950](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737950) and [#735121](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735121) but has no host answer yet.

Rule 2.8.a.1 requires that "the model weights should be provided as a public kaggle dataset so it is both publicly accessible and linked to the inference/submission code." Rule 2.5.a.4 says: "In the event that input data or pretrained models with an incompatible license are used to generate your winning solution, you do not need to grant an open source license in the preceding Section for that data and/or model(s)."

Several free, click-through knee-MRI datasets forbid sharing material derived from them: KneeMRI is CC BY-NC-ND 4.0, the MRNet Research Use Agreement (§6) prohibits derivative works, and the fastMRI Data Sharing Agreement prohibits distributing data files derived from the dataset. Weights trained on them may therefore not be publishable as a public Kaggle dataset.

We understand you cannot interpret the dataset stewards' own terms (as in [#741819](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819)). This question is only about the competition's Rules:

1. When a winner's weights were trained on such data, does Rule 2.5.a.4 also exempt those weights from the Rule 2.8.a.1 requirement to publish them publicly — for example, by delivering them privately to the Sponsor for verification while all code is public?
2. If not, is such a solution ineligible for Leaderboard and Efficiency prizes, so that teams should remove that data before the Final Submission Deadline?

A yes/no on each would settle this for everyone before the deadline. Thank you.

---

## Comments (1, threaded, oldest first)

### [1] Richard_61

**Richard_61** (@yiliu15, Contributor) · _topic author_ · 2026-09-29 02:15 UTC · LB rank 40

@javaduke95 Gentle follow-up on Q1 and Q2 above. Final selections are due by Oct 22 and depend on the answer, so a brief yes/no on each would let every team using no-redistribution external data (KneeMRI, MRNet, fastMRI) choose its submissions correctly. Thank you!
