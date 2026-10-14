# Public/private test split — stratified by site, or entire sites held out?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734681
- **Author:** **Matteo Vitali** (@matteovitali1004, Contributor) · _topic author_ · 2026-08-12 11:50 UTC · ▲7 · LB rank 14
- **Topic votes:** 7 · **Comments/replies:** 3 · **Pinned:** no
- **Last comment:** 2026-08-14 14:08 UTC by PC Jimmmy
- **Cloned:** 2026-10-14 18:23 UTC (topic id 734681, raw JSON `_raw/734681.json`)

---

Could the hosts clarify whether the public and private test sets are stratified by site (each of the 16 sites present in both) or whether entire sites are held out for the private set?

A yes/no on the category is all I'm after — no need to reveal which studies are where. 

Thanks!

---

## Comments (3, threaded, oldest first)

### [1] paul margain

**paul margain** (@paulmargain, Contributor) · 2026-08-13 10:21 UTC

I’m also very interested in this point. Generalization to unseen sites is crucial for real-world applications, and since the MRNet, it has remained one of the major challenges in this field. I therefore hope that the high leaderboard scores are not partly explained by overlap between sites in the different splits. But anyway, thank you to the organizers for this very interesting challenge and unique dataset.

### [2] dk2lone

**dk2lone** (@dk2lone, Contributor) · 2026-08-14 03:57 UTC · ▲-1 · LB rank 2330

Adding a concrete reason this matters beyond curiosity.

The answer changes how folds should be built. If entire sites are held out for the private set, then site-grouped cross-validation is measuring the right thing and the OOF score it costs is worth paying. If both splits are stratified across all 16 sites, that cost buys nothing, and a random-fold OOF is the better guide.

Those two designs pull in opposite directions, so without an answer everyone is choosing one and hoping. A yes/no on the category, as originally asked, would settle it without revealing anything about which studies are where.

Thank you.

### [3] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-08-14 14:08 UTC · ▲1 · LB rank 636

Not my experience that the hosts will identify the site split.   

You might want to look at past RSNA competitions to see what kind of shakeup occurred.  Big shakeups always suggest non random public vs private test splits.  My recollection is that some level of shakeup has occurred in the RSNA's I have been a part of - #1 public leader seldom the private but not by huge drops.  I have been in a few where I moved 1000 places between public and private - do not recall RSNA every being one of those.

And remember - to get a high final private score you need to predict the kaggle truth and not any generalization to the real world :)

EDIT - looked at 6 past competitions.   Shakeup ranged from None to 1039 places for first.  Guess that tells me that non random might happen now and than?
