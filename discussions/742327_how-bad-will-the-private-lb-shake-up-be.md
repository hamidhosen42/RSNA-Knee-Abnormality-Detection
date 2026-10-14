# How bad will the private LB shake-up be?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742327
- **Author:** **Raymond Yuen** (@laymond, Contributor) · _topic author_ · 2026-09-21 13:46 UTC · ▲3 · LB rank 38
- **Topic votes:** 3 · **Comments/replies:** 7 · **Pinned:** no
- **Last comment:** 2026-09-29 11:40 UTC by CoreyJamesLevinson
- **Cloned:** 2026-10-14 18:23 UTC (topic id 742327, raw JSON `_raw/742327.json`)

---

Hello there... I'm pretty new to Kaggle. The only comps I've done with a private LB are MAP and S6E8, and in both the private scores barely moved (relatively), so I've never actually seen a serious shake-up. I'm quite curious about how this one will end.

So we're only given 58 gold studies, plus reports that don't exactly line up with the dicoms. My best training label (picked by LB, not by its AUC against the 58) sits at around 0.89 macro AUC against the gold, while my model's local eval against gold is around 0.925 and LB 0.947. So the model already beats the labels it was trained on, which makes local validation pretty hard to trust. My current approach is to just train on everything (eval set just soley on gold) and trust +0.002 level LB changes.

What I see is everyone's clustered up. The silver line and bronze line literally share the same three digits, and near my rank teams squish within a 0.005 range. Up there 0.001 looks like it's worth roughly 7–9 places, and there's a big pile of teams sitting on the exact same score. 

So how much does it usually move? And could this competition be more severe?

Any advice would be great. May the force be with you.
![](_assets/403e09378636654b.png)

---

## Comments (7, threaded, oldest first)

### [1] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-09-21 16:54 UTC · ▲3 · LB rank 1066

I have not looked close but I think most of the pile up in bronze and some silver are two shared notebooks that have been tweaked or just copied by lots of folks.  When a huge portion of the metals are based on only a couple of shared notebooks the potential for shakeup is huge.  I once jumped up 1000 positions into a silver in a competition where most of the pre-final metals were the result of one shared notebook that "failed".

#### ↳ [1.1] Raymond Yuen

**Raymond Yuen** (@laymond, Contributor) · _topic author_ · 2026-09-22 04:13 UTC · LB rank 38

wow. a 1000 places jump is insane! I guess the biggest question is whether the shared solutions are robust or people are just sitting on the same lb overfit

### [2] Prateek Grover

**Prateek Grover** (@prateekgroverin, Contributor) · 2026-09-21 22:13 UTC · ▲1 · LB rank 184

It's hard to say, but Top 50, everyone is now over 0.95 - so I believe they all do indeed have separate models outperforming the public stack's 0.941 number. (atleast I do). 

Perhaps in a couple of weeks, when teams start forming ensembles - we'll see atleast the entire Top 10 go over 0.96, and they should remain there with the Private LB too.

#### ↳ [2.1] Raymond Yuen

**Raymond Yuen** (@laymond, Contributor) · _topic author_ · 2026-09-22 04:15 UTC · LB rank 38

Hopefully we’re not overfitting, and hopefully we won’t be🫣. It’s just that with this structure, it feels hard to actually tell whether a model is genuinely better or just happens to line up with the public LB.

### [3] Georgy Mamarin

**Georgy Mamarin** (@georgymamarin, Master) · 2026-09-28 12:14 UTC

Around the bronze line, usually a lot. At today's 4,500 teams bronze ends at place 450. I looked at the 11 medal competitions since 2022 with 3,000 to 6,750 teams and an ordinary public/private split. Teams just inside the line on the public board (places 338 to 450 here) finished with a medal 24% of the time, and teams just outside it (451 to 562) also 24%, so the public line made little difference.

This one could be worse than that, or milder: across those competitions the rate just inside the line ran from 2% to 51%, and in 10 of the 11 it was under half.

My notebook computes these rates for any rank and team count: [How safe is your medal spot on the public board?](https://www.kaggle.com/code/georgymamarin/how-safe-is-your-medal-spot-on-the-public-board)

### [4] (unknown)

**(unknown)** · 2026-09-28 16:29 UTC

_(no content — deleted or empty)_

### [5] CoreyJamesLevinson

**CoreyJamesLevinson** (@returnofsputnik, Grandmaster) · 2026-09-29 11:40 UTC · ▲2 · LB rank 2811

The public notebook is overfit blend scoring 0.945, and is in the silver range. I suspect if you are at 0.935+ and not using the public notebook then you could shake up into silver
