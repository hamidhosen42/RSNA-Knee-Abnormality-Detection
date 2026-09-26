# Is the gated KneeCoT dataset permitted as external data?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734109
- **Author:** **Mazzutti** (@profmazzutti, Contributor) · _topic author_ · 2026-08-09 23:34 UTC · ▲10 · LB rank 18
- **Topic votes:** 10 · **Comments/replies:** 2 · **Pinned:** no
- **Last comment:** 2026-08-25 14:43 UTC by Mazzutti
- **Cloned:** 2026-09-26 16:24 UTC (topic id 734109, raw JSON `_raw/734109.json`)

---

The KneeCoT dataset is available through gated Hugging Face access under
CC BY-NC 4.0:

https://huggingface.co/datasets/YiHui0124/KneeCoT

Its dataset card requires a formal ethical-use agreement with the affiliated
hospital and prohibits unauthorized redistribution. We would use it only for
auxiliary training, without publishing images or reports. Potentially required
winner deliverables would include only code and trained weights.

Could the host confirm:

1. Whether KneeCoT satisfies the competition requirement that external data
   be equally and reasonably accessible to participants.
2. Whether using it in this prize-bearing competition is permitted.
3. Whether publishing only trained weights is sufficient for reproducibility.
4. Whether the host knows of any subject/study overlap with the competition
   train or hidden-test datasets.

---

## Comments (2, threaded, oldest first)

### [1] Po-Hao "Howard" Chen

**Po-Hao "Howard" Chen** (@javaduke95, Contributor) · _competition host_ · 2026-08-25 11:00 UTC · ▲3

Thank you for your question and patience. After deliberation, we decided to not allow the use of KneeCoT dataset due to the required data agreement with dataset institution. In our interpretation of the rules, this can make the playing field uneven for participants who can complete the formal agreement to seek approval compared those that cannot.  This is our interpretation of the dataset rules and the discussion threae on HF.  If the situation changes, for instance if the data use policy changes on HF, or if the data organizers communicate their intent differently, we may change our stance. 

For now, the KneeCoT dataset is not allowed in this competition. 

Thank you for your understanding and patience.

#### ↳ [1.1] Mazzutti

**Mazzutti** (@profmazzutti, Contributor) · _topic author_ · 2026-08-25 14:43 UTC · LB rank 18

Thank you for looking into this and providing the clarification. It makes complete sense to ensure a fair playing field for all participants. Appreciate the quick feedback and clear guidance!
