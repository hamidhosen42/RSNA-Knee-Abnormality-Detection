# [Debug] DINOv3 cannot detect ACL/MCL tears

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747553
- **Author:** **Jin Park** (@jinspark621, Contributor) · _topic author_ · 2026-10-09 00:48 UTC
- **Topic votes:** 0 · **Comments/replies:** 1 · **Pinned:** no
- **Last comment:** 2026-10-09 02:00 UTC by hengck23
- **Cloned:** 2026-10-14 18:23 UTC (topic id 747553, raw JSON `_raw/747553.json`)

---

I'm using DINOv3 frozen embeddings as a backbone for my model. My issue is that the model isn't detecting small localized features like those needed for detecting ACL/MCL tears. I've tried increasing resolution and using patch tokens (not only the CLS token), but it's still having trouble detecting the features. Does anyone know how to debug this?

---

## Comments (1, threaded, oldest first)

### [1] hengck23

**hengck23** (@hengck23, Grandmaster) · 2026-10-09 02:00 UTC · ▲1 · LB rank 728

Try to do segmentation, supervision is pixel. Visualise results at each epoch. See how fast the loss descends and the predicted pixels show up. This is your baseline oracle.

Then go back to your current slot transformer model. Visualise your attention weights or cam map.

Input my above to chatgpt and ask to improve the algorithm debug
