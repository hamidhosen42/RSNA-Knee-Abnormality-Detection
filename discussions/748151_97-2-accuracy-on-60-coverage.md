# 97.2% accuracy on 60% coverage

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748151
- **Author:** **OmerZalman** (@omerzalman, Contributor) · _topic author_ · 2026-10-10 21:11 UTC · ▲-6 · LB rank 5107
- **Topic votes:** -6 · **Comments/replies:** 2 · **Pinned:** no
- **Last comment:** 2026-10-10 22:28 UTC by OmerZalman
- **Cloned:** 2026-10-14 18:23 UTC (topic id 748151, raw JSON `_raw/748151.json`)

---

would anyone want a dataset that covers around 60% of the rsna labels (pseudolabelling) and I tested it on expert labels to be around 97.2% accuracy (I didnt train a model yet, my last submission is 2 weeks ago)

---

## Comments (2, threaded, oldest first)

### [1] Siem Mebrahtu

**Siem Mebrahtu** (@siemmebrahtu, Contributor) · 2026-10-10 22:23 UTC · LB rank 1776

Ya I think you should publish it...the community would definitely find a 60% coverage pseudolabeled dataset useful  either for pre-training or as features for Level-2 stacking models..but I think you should be careful relying on the standard 'accuracy.' because medical datasets usually have huge class imbalances since most scans are normal the 97% accuracy you have might actually result in a poor leaderboard score if it's missing the minority positive cases and I think you should test the pseudolabels against the competition's official ROC AUC metric first.

#### ↳ [1.1] OmerZalman

**OmerZalman** (@omerzalman, Contributor) · _topic author_ · 2026-10-10 22:28 UTC · LB rank 5107

Ok i will try this
