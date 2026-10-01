# The leaderboard shows 3 decimals and I spent 20 submissions inside one of them

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740439
- **Author:** **Lê Quang Cảnh** (@zhincez, Contributor) · _topic author_ · 2026-09-09 17:06 UTC · LB rank 1454
- **Topic votes:** 0 · **Comments/replies:** 2 · **Pinned:** no
- **Last comment:** 2026-09-12 08:48 UTC by Navneet
- **Cloned:** 2026-10-01 19:24 UTC (topic id 740439, raw JSON `_raw/740439.json`)

---

I blended a public notebook with my own CoAtNet ensemble, then spent two weeks tuning the blend. Here is the whole weight sweep with the leaderboard score each one produced. W is the weight on the public source, rank-mean blending.

```
W_pub   LB
0.00   0.934   (pure my own ensemble)
0.15   0.935
0.30   0.936
0.45   0.936
0.50   0.937
0.60   0.937
0.70   0.937
0.75   0.937
0.90   0.936
1.00   0.936   (pure public notebook)
```

Ten points on the axis, four distinct numbers, and a flat plateau from 0.50 to 0.75.

Then I changed the blend operator instead of the weight, all at W = 0.60:

```
rank-mean                 0.937
probability space         0.937
logit space               0.937
geometric mean on rank    0.937
power mean, order 2       0.936
rank clipped to [.05,.95] 0.937
```

Then adaptive weights: per study, per label, twelve separate coefficients, the same thing with the sign flipped, a version with double the span. Every one of them: 0.937.

Counting up, 13 of my submissions read exactly 0.937 and I cannot order any of them.

The part worth sitting with: pure public is 0.936 and pure mine is 0.934, so everything my own ensemble adds on top of a public notebook is worth at most one displayed digit. Two weeks of blend tuning bought me 0.001, and I have no idea whether that 0.001 is real or rounding.

I do not have a clean answer, but two things I would do differently:

Check the display resolution against the effect size before starting an axis. Three decimals means anything under about 0.0005 is invisible, and blend weight moves the score by less than that across most of its range. That axis was never measurable here, no matter how many submissions I threw at it.

Build the local ruler first. Mine ran the wrong way: my 58-study gold set ranked variants in roughly the opposite order to the leaderboard, so I had no way to break the ties either. I kept submitting because the leaderboard was the only number I had, which is exactly backwards.

If anyone has a local validation setup that actually correlates with the board here, I would like to hear how you built it. Especially how you split, given how few studies carry the rarer labels.

---

## Comments (2, threaded, oldest first)

### [1] ParsaAshrafi

**ParsaAshrafi** (@parsaashrafii, Contributor) · 2026-09-12 08:26 UTC · LB rank 1035

Your observation is very reasonable. With a leaderboard displayed to three decimals, changes below roughly 0.0005 can be impossible to distinguish, so a large sweep may mostly be measuring rounding noise.

I would build the local ruler first:

- use study-level, multilabel-stratified splits;
- generate strict OOF predictions for every candidate blend;
- report bootstrap confidence intervals, not only one mean AUC;
- compare per-label deltas, especially for rare findings;
- only spend a submission when the expected gain is larger than the display resolution and remains stable across folds.

For blend tuning, tracking rank correlation between OOF variants and leaderboard results is also more informative than chasing equal displayed scores. If several submissions show the same rounded score, keep the simplest and most reproducible one.

### [2] Navneet

**Navneet** (@navneetbende, Contributor) · 2026-09-12 08:48 UTC · ▲-1

Thank you for the leaderboard showing 3 decimals @zhincez
