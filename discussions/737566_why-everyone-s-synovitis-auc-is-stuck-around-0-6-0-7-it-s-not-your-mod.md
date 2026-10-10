# Why everyone's Synovitis AUC is stuck around 0.6-0.7 (it's not your model)

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737566
- **Author:** **starkhushi** (@starkhushi, Expert) · _topic author_ · 2026-08-26 04:42 UTC · ▲6 · LB rank 601
- **Topic votes:** 6 · **Comments/replies:** 0 · **Pinned:** no
- **Last comment:** 2026-08-26 04:42 UTC by starkhushi
- **Cloned:** 2026-10-09 09:46 UTC (topic id 737566, raw JSON `_raw/737566.json`)

---

Looking at public notebooks, Synovitis is consistently the worst of the 12 labels — mine included (0.62-0.72 across four different architectures). I dug into why, and I'm now fairly convinced it's the supervision, not the models:

1. Only ~16% of the 4,407 reports mention any synovitis term at all — I searched synovit/sinovit/υμεν/синови/sinovyal across the nine report languages. For the other ~84% of studies, whatever label table you train on has inferred the target rather than read it.

2. The label tables themselves agree with the radiologist-annotated studies at only ~0.79 AUC on Synovitis, versus ~0.99 for ACL. Your model is learning from a teacher that's barely better than a coin flip with a lean.

3. The one thing that moved it for me: an encoder pretrained on medical images (RadImageNet) hit 0.78 on Synovitis where all my ImageNet-pretrained models sat at 0.62-0.72 — suggesting the image signal exists, but natural-image features can't see it.

Practical takeaways: don't burn architecture-tuning time on Synovitis expecting parity with other labels; a medically-pretrained member in an ensemble specifically helps here; and if anyone has a labeling approach that reads Synovitis better from the reports (e.g., inferring it from effusion co-mentions), that might be worth more than any model change.

Effusion↔Synovitis correlate at ~0.5 in the annotated studies, so there's signal to borrow. Curious if anyone has tried it.

---

## Comments (0, threaded, oldest first)

_No comments._
