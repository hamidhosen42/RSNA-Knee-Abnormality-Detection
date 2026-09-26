# How bad will the private LB shake-up be?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742327
- **Author:** **Raymond Yuen** (@laymond, Contributor) · _topic author_ · 2026-09-21 13:46 UTC · ▲3 · LB rank 76
- **Topic votes:** 3 · **Comments/replies:** 4 · **Pinned:** no
- **Last comment:** 2026-09-22 04:15 UTC by Raymond Yuen
- **Cloned:** 2026-09-26 16:24 UTC (topic id 742327, raw JSON `_raw/742327.json`)

---

Hello there... I'm pretty new to Kaggle. The only comps I've done with a private LB are MAP and S6E8, and in both the private scores barely moved (relatively), so I've never actually seen a serious shake-up. I'm quite curious about how this one will end.

So we're only given 58 gold studies, plus reports that don't exactly line up with the dicoms. My best training label (picked by LB, not by its AUC against the 58) sits at around 0.89 macro AUC against the gold, while my model's local eval against gold is around 0.925 and LB 0.947. So the model already beats the labels it was trained on, which makes local validation pretty hard to trust. My current approach is to just train on everything (eval set just soley on gold) and trust +0.002 level LB changes.

What I see is everyone's clustered up. The silver line and bronze line literally share the same three digits, and near my rank teams squish within a 0.005 range. Up there 0.001 looks like it's worth roughly 7–9 places, and there's a big pile of teams sitting on the exact same score. 

So how much does it usually move? And could this competition be more severe?

Any advice would be great. May the force be with you.
![](_assets/403e09378636654b.png)

---

## Comments (4, threaded, oldest first)

### [1] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-09-21 16:54 UTC · ▲2 · LB rank 636

I have not looked close but I think most of the pile up in bronze and some silver are two shared notebooks that have been tweaked or just copied by lots of folks.  When a huge portion of the metals are based on only a couple of shared notebooks the potential for shakeup is huge.  I once jumped up 1000 positions into a silver in a competition where most of the pre-final metals were the result of one shared notebook that "failed".

#### ↳ [1.1] Raymond Yuen

**Raymond Yuen** (@laymond, Contributor) · _topic author_ · 2026-09-22 04:13 UTC · LB rank 76

wow. a 1000 places jump is insane! I guess the biggest question is whether the shared solutions are robust or people are just sitting on the same lb overfit

### [2] Prateek Grover

**Prateek Grover** (@prateekgroverin, Contributor) · 2026-09-21 22:13 UTC · ▲1 · LB rank 129

It's hard to say, but Top 50, everyone is now over 0.95 - so I believe they all do indeed have separate models outperforming the public stack's 0.941 number. (atleast I do). 

Perhaps in a couple of weeks, when teams start forming ensembles - we'll see atleast the entire Top 10 go over 0.96, and they should remain there with the Private LB too.

#### ↳ [2.1] Raymond Yuen

**Raymond Yuen** (@laymond, Contributor) · _topic author_ · 2026-09-22 04:15 UTC · LB rank 76

Hopefully we’re not overfitting, and hopefully we won’t be🫣. It’s just that with this structure, it feels hard to actually tell whether a model is genuinely better or just happens to line up with the public LB.
