# How to use multilingual reports

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/736711
- **Author:** **doheon114** (@doheon114, Expert) · _topic author_ · 2026-08-22 07:50 UTC · ▲1 · LB rank 2998
- **Topic votes:** 1 · **Comments/replies:** 1 · **Pinned:** no
- **Last comment:** 2026-08-22 08:58 UTC by Komil Parmar
- **Cloned:** 2026-10-01 19:24 UTC (topic id 736711, raw JSON `_raw/736711.json`)

---

Hi, I’ve been thinking a lot about how to make the best use of the available reports.

I guess there are two main ways we could leverage them:

1. Use the reports only to extract pseudo-labels.
2. Use the reports directly for model training, e.g., by aligning image and text features.

How are you planning to make use of the reports in your approach?

---

## Comments (1, threaded, oldest first)

### [1] Komil Parmar

**Komil Parmar** (@komilparmar, Expert) · 2026-08-22 08:58 UTC · LB rank 59

The straight forward way to utilize the reports is definitely by simply using them as pseudo labels, when you do that, there are mainly two things to keep in mind:
1. Weight the gold labels much higher
2. If you are using multiple models, you can use their disagreement as a metric of how accurate those labels are, roughly at least, and use that again to weight those pseudo labels better. 

I don't believe we should be aligning image with the text features for various reasons, and reports are not available at test time any way so its better to stick with image only to keep the local eval clean.
