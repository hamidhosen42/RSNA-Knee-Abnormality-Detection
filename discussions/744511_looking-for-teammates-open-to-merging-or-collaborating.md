# Looking for teammates—open to merging or collaborating.

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744511
- **Author:** **SpeedSci** (@caoyupeng, Expert) · _topic author_ · 2026-09-30 05:13 UTC · ▲2 · LB rank 128
- **Topic votes:** 2 · **Comments/replies:** 6 · **Pinned:** no
- **Last comment:** 2026-10-01 13:00 UTC by ringbearer
- **Cloned:** 2026-10-01 19:24 UTC (topic id 744511, raw JSON `_raw/744511.json`)

---

We have a 5-fold Enet at 0.940, a 5-fold ResNet at 0.940, and a 5-fold DINOv2 at 0.931. We also used a public single-model CoAtNet at 0.928. By ensembling them, we reached 0.947, without using any public notebook models.

We still have some experiments training, and our team's current score ceiling is around 0.948–0.950.

We've done a lot of work on model optimization, model ensembling, and labels, but we're stuck. If your goal is a gold medal and you think our teams can help each other, please comment below.

---

## Comments (6, threaded, oldest first)

### [1] madmax0404

**madmax0404** (@max6296, Expert) · 2026-09-30 08:08 UTC · ▲1

I haven't started this competition, but can I join? I have a lot of spare time in the next month.

#### ↳ [1.1] SpeedSci

**SpeedSci** (@caoyupeng, Expert) · _topic author_ · 2026-09-30 08:55 UTC · LB rank 128

Sorry man,you're strong and I trust you, but right now we're looking to merge with teams that already have results on the board.

### [2] Renmq

**Renmq** (@renmingqian, Expert) · 2026-09-30 08:17 UTC · ▲1 · LB rank 1648

Bro,I have a CoAtNet singel model at 0.936，and I have 8 gpus，my score is a little bit low，I have some experiments now，I think I can get to about 0.938 singel CoAtNet model。

### [3] KalyanG17

**KalyanG17** (@kalyang17, Contributor) · 2026-09-30 08:20 UTC · LB rank 999

Hi, I'd be interested. I'm solo at 0.942.

Quick summary of what I have: my best own model is a 5-fold ResNet50 (RadImageNet init, 2.5D at 288px) that scores 0.924 by itself, trained only on my own LLM report labels. I also have a CoAtNet at 0.935 and a ConvNeXt-T at 0.930 from earlier label experiments, and a bunch of other trained variants I never bothered to submit.

To be honest my models never gave the public stack anything, but they were trained on different labels than yours so they might blend better with your ensemble than with the public stuff. 

For the compute I have a 5070 locally and I barely use my kaggle GPU quota, so I can help with training.

If you're ok with that, ping me and we can move to email/discord.

### [4] Lavin Wins

**Lavin Wins** (@lavinwins, Contributor) · 2026-09-30 16:39 UTC · ▲-4 · LB rank 406

Hey! This is the kind of clean, scratch-built progress that deserves a gold. We love that you avoided the public notebook trap and built a robust, diverse stack (Enet + ResNet + DINOv2 + CoAtNet is a killer foundation).

We are on the exact same trajectory, sitting at 0.943 ceiling (no public notebook!!!), and we’ve been tackling heavy test-time augmentation / advanced pseudo-labeling / custom post-processing.

Mathematically, your architecture mix combined with our stack creates a massive diversity boost—which is practically a cheat code for breaking past 0.950 and locking in gold.

Let’s not waste time competing instead we should connect and build better model fit for the competition...

### [5] ringbearer

**ringbearer** (@mehmetbarankrat, Contributor) · 2026-10-01 13:00 UTC · LB rank 108

hi, solo here. i have a 5 fold coatnet at 0.943, another coatnet variant at 0.940, convnext at 0.937 and resnet at 0.933, plus a radimagenet model . blend is at 0.948, interested to talk ?
