# are public weights trained on OAI knee MRI allowed?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747455
- **Author:** **Anay Chauhan** (@anaychauhan, Contributor) · _topic author_ · 2026-10-08 19:16 UTC · ▲10 · LB rank 603
- **Topic votes:** 10 · **Comments/replies:** 2 · **Pinned:** no
- **Last comment:** 2026-10-13 16:09 UTC by Arihant Jain
- **Cloned:** 2026-10-14 18:23 UTC (topic id 747455, raw JSON `_raw/747455.json`)

---

Question for the host: are public weights trained on OAI knee MRI allowed?
Several public notebooks attach a Kaggle dataset of weights whose description says they were trained with OAI (Osteoarthritis Initiative) knee MRIs as external supervision. OAI is free but requires registration and a data-use agreement. Under the External Data rules (“publicly available and equally accessible… at no cost”), can participants use these OAI-trained public weights, or OAI data directly? A clear ruling would help many teams. Thank you!

---

## Comments (2, threaded, oldest first)

### [1] Civitasmass

**Civitasmass** (@godhand, Master) · 2026-10-09 15:56 UTC · LB rank 233

I think the safest approach is to avoid using this external data,
it offers limited benefits but carries high risks.

### [2] Arihant Jain

**Arihant Jain** (@arihanttjainn, Contributor) · 2026-10-13 16:09 UTC · LB rank 1314

@macruzbar @javaduke95 - could we get a ruling on this before the notebook lock? Many top public notebooks (0.954-0.956) use nartaa/rsna-knee-publication-swa-weights-20261007, whose description says the models used 2,399 baseline OAI knees (NIMH Data Archive, Shared NDA Study 3407) for external supervision. OAI access requires an NDA data-use application, which per earlier guidance is not the same as click-through registration. Specifically:
1) Are public weights trained on OAI data allowed for leaderboard/medal eligibility?
2) Are they allowed for prize eligibility?
A clear answer would help a lot of teams choose their final submissions. Thanks!
