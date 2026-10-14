# looking for help

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744495
- **Author:** **TheMightiestMan** (@themightiestman, Contributor) · _topic author_ · 2026-09-30 02:16 UTC · ▲1 · LB rank 707
- **Topic votes:** 1 · **Comments/replies:** 5 · **Pinned:** no
- **Last comment:** 2026-10-02 19:22 UTC by Gabriel
- **Cloned:** 2026-10-14 18:23 UTC (topic id 744495, raw JSON `_raw/744495.json`)

---

I’ve joined two similar competitions. In both cases I landed in the silver range on the public LB, but fell out of bronze on the private LB. How can I prevent this? Is the score from my local dataset during training just as important as the public leaderboard score?

---

## Comments (5, threaded, oldest first)

### [1] TheMightiestMan

**TheMightiestMan** (@themightiestman, Contributor) · _topic author_ · 2026-09-30 02:20 UTC · LB rank 707

I usually improve the code and then test on my local data to see whether it runs, then submit to Kaggle to check whether the score improves. If it improves, I keep improving the code on top of that new version. My experience from two competitions tells me that local test scores can't reflect the public leaderboard score, but the private leaderboard has a big gap from the public leaderboard score.

### [2] Chris Deotte

**Chris Deotte** (@cdeotte, Grandmaster) · 2026-09-30 03:26 UTC · ▲2 · LB rank 131

The best way to maintain rank on private LB compared to public LB is to "trust your CV". Set up a local CV scheme which mimics the relationship between Kaggle test and Kaggle train. (i.e. if Kaggle test has different studies than train, them make your kfold validation have different studies than kfold train).

The next technique is more advanced. If there are differences in your LB score versus your CV score, you need to do detective work to understand what is different between test and train. And you then extrapolate this conclusion to what may be different between private and public test. Then you adjust/adapt your local validation and model training to handle/accomdate this.

#### ↳ [2.1] TheMightiestMan

**TheMightiestMan** (@themightiestman, Contributor) · _topic author_ · 2026-09-30 03:42 UTC · ▲1 · LB rank 707

thanks for your help

### [3] SpeedSci

**SpeedSci** (@caoyupeng, Expert) · 2026-09-30 04:56 UTC · ▲1 · LB rank 116

If we don't rely on the public LB, RSNA probably won't shake much

### [4] Gabriel

**Gabriel** (@shawsebastian, Expert) · 2026-10-02 19:22 UTC · ▲1 · LB rank 832

Based on my experience of slipping off the private leaderboard in cell tracking competitions, the first thing to do to avoid this situation is to absolutely not optimize on the basis of public notebook.
