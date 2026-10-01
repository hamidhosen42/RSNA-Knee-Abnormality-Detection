# Winner's obligations when the submitted weights are a public checkpoint someone else trained

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741212
- **Author:** **Ziad Ahmed** (@ziadahmeddd, Contributor) · _topic author_ · 2026-09-13 22:13 UTC · LB rank 1258
- **Topic votes:** 0 · **Comments/replies:** 0 · **Pinned:** no
- **Last comment:** 2026-09-13 22:13 UTC by Ziad Ahmed
- **Cloned:** 2026-10-01 19:24 UTC (topic id 741212, raw JSON `_raw/741212.json`)

---

Section 2.8.a requires a winner to deliver "training code, inference code, model weights" and to publish the weights as a public Kaggle dataset. Section 2.6 separately permits freely and publicly available pretrained models, and the Code Requirements page states this explicitly.
Those two are clear individually but not in combination. Consider a submission whose predictions come entirely from a publicly available checkpoint that another participant trained and released, used without modification. The participant's own contribution is the inference pipeline preprocessing, windowing, aggregation, runtime engineering not the weights.
1. Can such a submission be eligible for a Leaderboard or Efficiency Prize?
2. If so, what satisfies the training-code obligation pointing to the original author's published training code, or must the winner have trained the weights themselves?
Section 2.8.a.a carves out commercially available software that the Sponsor can procure without undue expense, but says nothing about model weights, which is where the ambiguity sits.
This matters most for the Efficiency Track, where a single strong public checkpoint plus a fast inference pipeline is a natural approach, and asking now avoids discovering the answer at the Nov 5 deadline.

---

## Comments (0, threaded, oldest first)

_No comments._
