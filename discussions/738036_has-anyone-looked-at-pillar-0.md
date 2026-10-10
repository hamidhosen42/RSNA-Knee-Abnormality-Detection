# Has anyone looked at Pillar-0?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738036
- **Author:** **Fabian Friedland - xBrain** (@fabotelli, Contributor) · _topic author_ · 2026-08-28 20:25 UTC · LB rank 172
- **Topic votes:** 0 · **Comments/replies:** 1 · **Pinned:** no
- **Last comment:** 2026-08-28 22:24 UTC by Fabian Friedland - xBrain
- **Cloned:** 2026-10-09 09:46 UTC (topic id 738036, raw JSON `_raw/738036.json`)

---

Has anyone taken a look at Pillar-0, the new model from UCF?
It seems to specialize in volumetric analysis of MRIs, but I'm having the same problem I had with dinov3 - in that it's not fine-tuned to our speciality.

---

## Comments (1, threaded, oldest first)

### [1] Fabian Friedland - xBrain

**Fabian Friedland - xBrain** (@fabotelli, Contributor) · _topic author_ · 2026-08-28 22:24 UTC · LB rank 172

I fine-tuned YalaLab's Pillar-0 BreastMRI foundation model on this competition's knee MRIs and I'm sharing the weights + full recipe as a public model: pillar0-knee-finetune (https://www.kaggle.com/models/fabotelli/pillar0-knee-finetune) (loader demo notebook attached to the comp).

 TL;DR of the transfer study: frozen-probe macro 0.697 → 0.797 after a 2-epoch full fine-tune on a 58-study expert holdout. It's strong on Effusion (0.965) and Medial OA (0.941) but ~0.10 below task-specific 2.5D CNNs overall — the breast→knee domain gap concentrates in ligament/meniscus fine structure (MCL 0.630). Full per-target table, adapter details, and split-LR recipe in the model card. ECL-2.0, full attribution to the Pillar-0 authors.

Sharing since it might save someone GPU hours...and maybe someone else might have better insight. Pillar looks like it should do better.
