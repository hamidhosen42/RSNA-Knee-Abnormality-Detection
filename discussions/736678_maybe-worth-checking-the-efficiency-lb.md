# Maybe worth checking the Efficiency LB?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/736678
- **Author:** **k256.dev** (@k256net, Master) · _topic author_ · 2026-08-22 01:27 UTC · ▲13 · LB rank 1952
- **Topic votes:** 13 · **Comments/replies:** 4 · **Pinned:** no
- **Last comment:** 2026-08-24 13:24 UTC by Berat Kirbiyik
- **Cloned:** 2026-10-09 09:46 UTC (topic id 736678, raw JSON `_raw/736678.json`)

---

As of Aug 22, not many of the public LB top 50 also appear in the Efficiency LB top 50 — about 32%.

A high score with a long runtime might mean two things: with a 9h limit there may be little ensembling headroom left, and the score might have been bought with ensembling in the first place — this could be a competition where ensembling pays, judging from the public notebooks.

So a fast, decent-scoring solution might have more room left than the public LB makes it look.

For reference: I'm currently Efficiency LB 22nd, Public LB 78th  with 0.926. my runtime is about 20 minutes — just preprocessing plus a single model trained with 5-fold. Reading the Efficiency LB is what tells me a better score should be reachable in about the same runtime. Maybe worth a look for you too?

Efficiency LB
https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb

---

## Comments (4, threaded, oldest first)

### [1] Chikuwabu

**Chikuwabu** (@prgckwb, Master) · 2026-08-22 02:16 UTC · ▲1 · LB rank 28

I agree. 

One thing to keep in mind is that the top two Public LB submissions seem to be used for the Efficiency LB as well, so we should be careful not to accidentally get a high score through ensembling.

### [2] Scott Willis

**Scott Willis** (@scottgwillis, Contributor) · 2026-08-22 02:45 UTC · ▲1 · LB rank 3

So I'm surprisingly #1 on the efficiency LB right now.  No ensemble and I'm not too sure how much more I can squeeze out of it, but it's been really fun seeing how far I can push it without hurting efficiency.

#### ↳ [2.1] k256.dev

**k256.dev** (@k256net, Master) · _topic author_ · 2026-08-22 04:03 UTC · LB rank 1952

Honestly, I've come to think the Efficiency LB is the real public leaderboard. Let's both watch out for overfitting to public — good luck to us both!

### [3] Berat Kirbiyik

**Berat Kirbiyik** (@beratkirbiyik, Contributor) · 2026-08-24 13:24 UTC · LB rank 1241

Thanks for opening this up — the Efficiency LB really is the more informative board.

I profiled our inference pipeline and hit a wall I can't get past. Breakdown for ~1300 studies:

header pass          0.5 min
slice ordering      12.0 min   <- the wall
decode + resize      4.1 min
container + import   5.0 min
GPU (5-fold)        10.0 min

The ordering pass reads the geometry tags of every slice in every selected series (~186 files per study) to sort by the projection onto the slice normal, then keeps only 9. Things I tried that didn't work:

Threads: saturates at 64. Going to 128/256/512 changes nothing, so it's not latency-bound.
Reading only every Nth slice and interpolating: slice agreement with the full ordering drops to ~39%, no better than chance for the ones not sampled.
InstanceNumber instead of the projection: 94.5% agreement but zero speedup — the cost is opening the file, not parsing the tag.

You mentioned ~20 min total with preprocessing plus a single 5-fold model. Are you doing something fundamentally different at the ordering step, or skipping it? Genuinely curious whether the geometric sort is worth its cost at all
