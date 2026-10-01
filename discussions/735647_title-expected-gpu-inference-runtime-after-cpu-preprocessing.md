# Title: Expected GPU inference runtime after CPU preprocessing?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735647
- **Author:** **Seung Sup Lee** (@seungsuplee, Contributor) · _topic author_ · 2026-08-17 10:47 UTC · ▲3 · LB rank 2635
- **Topic votes:** 3 · **Comments/replies:** 2 · **Pinned:** no
- **Last comment:** 2026-08-17 15:35 UTC by Seung Sup Lee
- **Cloned:** 2026-10-01 19:24 UTC (topic id 735647, raw JSON `_raw/735647.json`)

---

Hi everyone, I’m new to Kaggle competitions, so I’d appreciate some guidance from more experienced participants.

I’ve optimized the CPU preprocessing pipeline for the full dataset and can currently complete that stage in under 1 hour in the Kaggle environment.

Since the total runtime limit is 9 hours, I’m trying to understand how much time I should realistically budget for the remaining GPU inference/model processing stage.

For those who have already tested an end-to-end pipeline, roughly how long does the GPU stage take, and which Kaggle GPU are you using?

I’m not asking for anyone’s solution or model details — I’m simply trying to establish a realistic runtime budget for the competition.

Thanks for any practical experience you can share.

---

## Comments (2, threaded, oldest first)

### [1] Harmain Mughal

**Harmain Mughal** (@harmainmughal11, Contributor) · 2026-08-17 13:52 UTC

Welcome to Kaggle! A safe baseline is 1 hour CPU + 5-6 hours GPU + 2 hours safety buffer. Always keep a 2-hour buffer to prevent your notebook from timing out on the larger hidden test set during submission!

#### ↳ [1.1] Seung Sup Lee

**Seung Sup Lee** (@seungsuplee, Contributor) · _topic author_ · 2026-08-17 15:35 UTC · LB rank 2635

😀thank you😀

Thank you for your advice. I incorporated some of your suggestions into my optimization process and managed to bring the CPU runtime down to 37 minutes and 22 seconds. I really appreciate your input.
