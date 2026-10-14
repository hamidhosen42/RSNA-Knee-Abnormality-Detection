# Labelling?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745730
- **Author:** **OmerZalman** (@omerzalman, Contributor) · _topic author_ · 2026-10-04 01:00 UTC · LB rank 4718
- **Topic votes:** 0 · **Comments/replies:** 1 · **Pinned:** no
- **Last comment:** 2026-10-05 07:50 UTC by Sotarriva Alvarez Isai Roberto
- **Cloned:** 2026-10-14 18:23 UTC (topic id 745730, raw JSON `_raw/745730.json`)

---

How did you guys label your data for the competition?

---

## Comments (1, threaded, oldest first)

### [1] Sotarriva Alvarez Isai Roberto

**Sotarriva Alvarez Isai Roberto** (@irsotarriva, Contributor) · 2026-10-05 07:50 UTC · LB rank 1006

Take the reports give them to an LLM to produce soft labels with different confidence ranges for each category. And ask it to give you the probability of each category.
You should get about 0.9 using those labels, still not leaderboard material but you have not seen the images yet. If you combine both to refine the labels or just use them as soft targets you should be able to go past the soft labels ceiling.
