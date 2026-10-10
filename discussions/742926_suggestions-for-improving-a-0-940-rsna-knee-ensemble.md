# Suggestions for improving a 0.940 RSNA Knee ensemble

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742926
- **Author:** **MUTTAHIR** (@muttahir, Contributor) · _topic author_ · 2026-09-24 06:48 UTC · ▲-4 · LB rank 1608
- **Topic votes:** -4 · **Comments/replies:** 8 · **Pinned:** no
- **Last comment:** 2026-09-28 00:13 UTC by Chris Deotte
- **Cloned:** 2026-10-09 09:46 UTC (topic id 742926, raw JSON `_raw/742926.json`)

---

Hi everyone. I’m looking for a controlled next experiment to improve my current public AUC of 0.940 in the RSNA Knee Abnormality Detection competition. My notebook is an attributed reproduction of Mattia Angeli’s public version 27, ‘Bend the Knee to the DINOv3.’ It includes 20 DINO MRI members, five DINOv3/A5 folds, RadImageNet heads, and probability-based TTA pooling with target-specific overrides. For those who improved beyond 0.94, what helped most: better report-derived labels, a genuinely different encoder, or OOF-based ensemble blending? How did you validate changes without overfitting to the public leaderboard? I’ll retain attribution for any code or ideas I use.

---

## Comments (8, threaded, oldest first)

### [1] Lavin Wins

**Lavin Wins** (@lavinwins, Contributor) · 2026-09-24 11:35 UTC · ▲-3 · LB rank 410

Try blending some other model with this to increase the score

#### ↳ [1.1] MUTTAHIR

**MUTTAHIR** (@muttahir, Contributor) · _topic author_ · 2026-09-25 06:32 UTC · LB rank 1608

Thanks, Lavin! I already have a blended ensemble with DINO-family MRI models and RadImageNet heads, so I’ll focus on adding a genuinely complementary model and validate the blend with OOF predictions rather than tuning to the public leaderboard. Did you have a particular backbone or checkpoint that worked well for this dataset?

##### ↳ ↳ [1.1.1] Lavin Wins

**Lavin Wins** (@lavinwins, Contributor) · 2026-09-27 06:22 UTC · LB rank 410

Relying on OOF predictions rather than chasing the public leaderboard is absolutely the right move, especially for preventing overfitting on medical datasets.
Since the current ensemble relies heavily on self-supervised, vision-transformer-based representations via DINO and RadImageNet, the most effective way to gain a genuine performance boost is by introducing a backbone with a completely different structural bias.
A ConvNeXt-V2 (specifically convnextv2_huge or large) or a Swin Transformer V2 initialized with masked autoencoder (MAE) weights would be excellent complementary choices.

### [2] Chris Deotte

**Chris Deotte** (@cdeotte, Grandmaster) · 2026-09-25 23:53 UTC · ▲1 · LB rank 100

All of these things can help
* improve train labels
* change image preprocess
* change the representation we input into our model
* change model architecture
* change backbone
* change data augmentation
* change learning schedule

#### ↳ [2.1] MUTTAHIR

**MUTTAHIR** (@muttahir, Contributor) · _topic author_ · 2026-09-26 06:49 UTC · ▲1 · LB rank 1608

Thanks, Chris—this is a helpful list. I’ll test one change at a time, starting with train-label quality and image preprocessing, then try a different input representation or backbone if validation supports it. I’ll keep the same study-level folds and compare macro AUC/OOF predictions rather than chasing the public leaderboard. For this dataset, which single change would you test first?

#### ↳ [2.2] MUTTAHIR

**MUTTAHIR** (@muttahir, Contributor) · _topic author_ · 2026-09-26 09:12 UTC · ▲1 · LB rank 1608

Thanks, Chris. I tested the report/E11 signals as a small, controlled first step. Only 57 matched gold-labeled studies were available per finding across the 12 targets, so these local results are noisy and are not leaderboard scores. They also do not measure my complete ensemble, whose current public score is 0.940.

Validation results (mean macro AUC):
• Report-only: 0.8692
• E11 OOF-only: 0.8582
• Fixed 50/50 percentile-rank blend: 0.9077
• Nested-CV-tuned rank weights: 0.9026
• Raw 50/50 probability mean: 0.9105
• Train-standardized mean: 0.9086

The weight-tuned rank blend was worse than the fixed blend. Raw averaging was only +0.0028 over percentile ranking on this small sample, so I did not change the 0.940 submission. The rank-weight experiment used nested CV (outer 3-fold × 10 repeats; inner 3-fold × 3 repeats for weight selection); the signal-transform comparison used repeated 3-fold validation × 10. These tests cover only the report/E11 branch; I still need OOF predictions from the full ensemble to evaluate a real addition.

Next I plan to audit report labels, then test one MRI preprocessing or input-representation change at a time, followed by a different backbone or augmentation/schedule if validation supports it. Practical tips: keep all slices/series from each study in one fold; inspect per-target AUC as well as macro AUC; generate full-ensemble OOF predictions before blending; tune weights only inside training folds; and repeat apparent gains across folds/seeds before trusting them. I’ll avoid choosing changes based only on public leaderboard movement.

Which preprocessing or representation change would you try first for this dataset?

### [3] Dread Development

**Dread Development** (@dreaddevelopment, Expert) · 2026-09-27 22:25 UTC · ▲2 · LB rank 205

I have found that having models that disagree with each other outperform models that agree within ensembles.

#### ↳ [3.1] Chris Deotte

**Chris Deotte** (@cdeotte, Grandmaster) · 2026-09-28 00:13 UTC · LB rank 100

Yes, this is the secret to ensembles. It is said well by Paul [here][1] (in Kaggle playground competition)

[1]: https://www.kaggle.com/competitions/playground-series-s6e9/discussion/743860
