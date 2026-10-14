# How do you optimize labels?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745214
- **Author:** **SpeedSci** (@caoyupeng, Expert) · _topic author_ · 2026-10-02 17:53 UTC · ▲8 · LB rank 118
- **Topic votes:** 8 · **Comments/replies:** 3 · **Pinned:** no
- **Last comment:** 2026-10-02 18:45 UTC by SpeedSci
- **Cloned:** 2026-10-14 18:23 UTC (topic id 745214, raw JSON `_raw/745214.json`)

---

I've tried using GPT, Claude, and Gemini to clean labels, then used voting to produce my first version.

After that, I used OOF and teacher models based on my trained vision models to further refine the labels. But I've been stuck on this for a long time, and there are some things I don't understand:

My DINOv2 scores 0.917 on Gold58 and 0.942 on LB

But EffNet and ResNet only score ~0.89 on Gold, yet also achieve 0.940 on LB

(I haven't explored CoAtNet much) CoAtNet: Gold 0.9149, LB 0.926

One label version even produced an unusually high Gold of 0.945 on EffNet, but only 0.910 on LB—clearly overfitting

Could top competitors share some insights on how to optimize labels?

---

## Comments (3, threaded, oldest first)

### [1] SpeedSci

**SpeedSci** (@caoyupeng, Expert) · _topic author_ · 2026-10-02 17:57 UTC · LB rank 118

We've optimized a lot on DINOv2, EffNet, and ResNet50. I think the label quality is what's holding us back now. I'd like to ask: how do we achieve label optimization that brings genuinely valuable improvement?

### [2] Tucker Arrants

**Tucker Arrants** (@tuckerarrants, Master) · 2026-10-02 18:11 UTC · ▲1 · LB rank 17

The size of the host provided labels is too small to actually resolve anything. The number of samples in the LB is significantly larger than the host provided labels, so is much safer to test actual label extraction differences against. It still cannot resolve much. The 58 provided labels and LB samples are created by the same radiologists, you just lose the ability to see per-label differences and the accompanying report - but you never really had these to begin with, as 58 is too small a sample size.

I use a simple extraction process with Qwen, no frontier models at all, and most of my models are in the low 0.95 range on the LB. I am doubtful that "improving labels" matters much here, at least in my experiments. Just make sure you don't have obvious extraction failures (I doubt you do if you are using 3 different frontier models).

#### ↳ [2.1] SpeedSci

**SpeedSci** (@caoyupeng, Expert) · _topic author_ · 2026-10-02 18:45 UTC · LB rank 118

Appreciate the reply. For my DINOv2, just changing the labels moved LB from 0.931 to 0.942—Gold went up by about 0.01 too.
