# Has anyone tried a GRU/LSTM head over slices here?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738495
- **Author:** **Abdullah Wasee** (@abdullahwasee, Contributor) · _topic author_ · 2026-08-31 17:46 UTC · LB rank 2417
- **Topic votes:** 0 · **Comments/replies:** 0 · **Pinned:** no
- **Last comment:** 2026-08-31 17:46 UTC by Abdullah Wasee
- **Cloned:** 2026-10-14 18:23 UTC (topic id 738495, raw JSON `_raw/738495.json`)

---

Every RSNA winner I can find uses a recurrent head to aggregate slices rather than pooling - 2019 hemorrhage (1st place was literally titled "Sequential model wins"), 2022 cervical spine (CNN+LSTM), 2023 abdominal trauma (CNN+GRU, max-pooled over slice logits), 2024 lumbar spine. The 2023 writeup says GRU beat self-attention in their ablations.

I haven't seen it mentioned in this competition's discussions, where the consensus seems to be attention pooling over sequence slots.

My guess at why it might not transfer: those are CT volumes with 96 slices, and most setups here use 3–9 slices per plane. A GRU over 3 steps has little sequence to model.

Has anyone tried it? And if so, did more slices per plane make it worth it, or does the resolution trade-off kill it?

Currently at 0.807 public / 0.751 on the annotated 58, ResNet-34 2.5D with attention pooling, so I'm asking from behind - happy to share my ablation log if useful.

---

## Comments (0, threaded, oldest first)

_No comments._
