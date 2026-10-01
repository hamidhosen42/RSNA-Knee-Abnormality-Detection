# One fold(No valid setup) model is better than Five fold ensemble models?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740375
- **Author:** **XuKong Ji** (@xukongji, Expert) · _topic author_ · 2026-09-09 11:24 UTC · ▲1 · LB rank 227
- **Topic votes:** 1 · **Comments/replies:** 4 · **Pinned:** no
- **Last comment:** 2026-09-11 08:38 UTC by Salem Ali
- **Cloned:** 2026-10-01 19:24 UTC (topic id 740375, raw JSON `_raw/740375.json`)

---

My Five fold ensemble models score:0.938(Pick highest valid auc)  
One fold score:0.939(Pick last epoch)  
Encoder: ImageNet Pretrained Convnext tiny

---

## Comments (4, threaded, oldest first)

### [1] Dread Development

**Dread Development** (@dreaddevelopment, Expert) · 2026-09-09 22:37 UTC · ▲1 · LB rank 144

My latest submission that I have been working is a single fold, 0.941 at 23 minutes
My prior was a 5 fold at 0.939 scoring in 5 hours

#### ↳ [1.1] XuKong Ji

**XuKong Ji** (@xukongji, Expert) · _topic author_ · 2026-09-10 01:08 UTC · ▲1 · LB rank 227

Appreciate the info. My 5-fold inference runs in ~30 mins, and a single fold is around 20–30 mins. Seems like preprocessing is the main bottleneck in my situation.

##### ↳ ↳ [1.1.1] Salem Ali

**Salem Ali** (@salemali7, Expert) · 2026-09-11 08:38 UTC · ▲1 · LB rank 182

You can speed up the preprocessing by running it in parallel with 4 CPU threads and a queue. I tried it myself, and it cut the time from about an hour to just 15 minutes.

### [2] SpeedSci

**SpeedSci** (@caoyupeng, Expert) · 2026-09-10 18:42 UTC · ▲1 · LB rank 1069

If the single fold uses full training, the results might be similar to 5-fold.
