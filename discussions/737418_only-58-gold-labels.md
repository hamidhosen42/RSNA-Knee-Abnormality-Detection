# Only 58 Gold Labels. 

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737418
- **Author:** **Jaideep726** (@jaideep726, Contributor) · _topic author_ · 2026-08-25 12:39 UTC · ▲-1 · LB rank 3647
- **Topic votes:** -1 · **Comments/replies:** 3 · **Pinned:** no
- **Last comment:** 2026-09-26 00:25 UTC by Chris Deotte
- **Cloned:** 2026-10-01 19:24 UTC (topic id 737418, raw JSON `_raw/737418.json`)

---

I know maybe this has been discussed before but this is the first problem I encountered starting out as a beginner. how are you guys trying to verify your labelling techniques on the the bigger 4349 reports

---

## Comments (3, threaded, oldest first)

### [1] LaiCxm

**LaiCxm** (@laicxm, Contributor) · 2026-08-26 09:06 UTC · LB rank 908

I've tried numerous combinations of LLMs, but the performance consistently plateaus at around 0.89. I suspect this might be the actual capability ceiling for LLMs. Furthermore, in real-world scenarios, different doctors have varying levels of experience and naturally provide different opinions, meaning the ground truth reports themselves aren't absolutely flawless.

Because of this, I've basically given up on chasing higher metrics. I feel like the ultimate solution might be training multiple models on different labels and relying on trial and error to get better scores. But honestly, this approach feels almost identical to continuous ensembling, which inevitably leads to overfitting.

#### ↳ [1.1] Jaideep726

**Jaideep726** (@jaideep726, Contributor) · _topic author_ · 2026-08-26 11:38 UTC · ▲-1 · LB rank 3647

what is the 0.89 figure exactly?

### [2] Chris Deotte

**Chris Deotte** (@cdeotte, Grandmaster) · 2026-09-26 00:25 UTC · LB rank 93

Great question. We have 3 choices
* use 58 gold labels
* use our 4.2k created labels from reports
* use the public LB
