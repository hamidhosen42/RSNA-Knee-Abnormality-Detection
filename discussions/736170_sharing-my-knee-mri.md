# Sharing my knee MRI

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/736170
- **Author:** **tennogh** (@tennogh, Master) · _topic author_ · 2026-08-19 18:21 UTC · ▲20 · LB rank 8
- **Topic votes:** 20 · **Comments/replies:** 6 · **Pinned:** no
- **Last comment:** 2026-08-26 22:14 UTC by Harsh_2035
- **Cloned:** 2026-10-14 18:23 UTC (topic id 736170, raw JSON `_raw/736170.json`)

---

After seeing [this discussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733497) by @harsh2035, I figured it would be fun to share [my knee MRI](https://www.kaggle.com/datasets/tennogh/my-knee-mri).

For context, this was taken 6 weeks after an ice climbing accident and confirmed a full ACL rupture. There are other abnormalities such as partial tears and contusion. Because of this accident I was forced to stop mountaineering for 6 months and started competing on Kaggle.

The dataset license is CC0, however the report text is written by a radiologist and I don't claim any rights to it. I have purged the data of any sensitive information and put it into the competition format.

Edit: interestingly my current model correctly predicts the ACL, but it also confidently identifies a fracture, even though there was none.

---

## Comments (6, threaded, oldest first)

### [1] Chikuwabu

**Chikuwabu** (@prgckwb, Master) · 2026-08-20 01:58 UTC · ▲2 · LB rank 28

Thanks for sharing! I ran your MRI through my model:

| Rank | Finding   | Score |
| ---: | --------- | ----: |
|    1 | Contusion | - |
|    2 | Synovitis | - |
|    3 | ACL       | - |

Contusion is a nice hit since you already confirmed it, and ACL also made the top 3.

So for Synovitis, one of two things may be true:

* My model has a high Public LB score, but has learned some competition-specific bias and is weak to OOD cases.
* You actually have some signs of Synovitis — which I hope is not the case.

#### ↳ [1.1] tennogh

**tennogh** (@tennogh, Master) · _topic author_ · 2026-08-21 11:11 UTC · LB rank 8

I'm not completely certain because I don't fully understand the terminology in French vs English, but I believe the report does not mention synovitis, but it does mention effusion. According to [this discussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932), both are often co-occurring and synovitis is under-reported.

### [2] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-08-21 14:22 UTC · LB rank 643

I might have missed it - did you share the "report" - I don't see it in the data set you shared.

Never-mind - looked again and it was right there   :)

#### ↳ [2.1] tennogh

**tennogh** (@tennogh, Master) · _topic author_ · 2026-08-21 14:36 UTC · ▲1 · LB rank 8

The structure might not be very clear, I tried to follow this competition's format. The report (in French) is included in "train.csv", so you can generate your own labels.

##### ↳ ↳ [2.1.1] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-08-21 14:42 UTC · LB rank 643

Thanks - you did good - I did not yet have enough coffee to have a functional brain.

Thanks for the share

### [3] Harsh_2035

**Harsh_2035** (@harsh2035, Expert) · 2026-08-26 22:14 UTC · LB rank 845

Great, and thanks for sharing this dataset @tennogh.

I would share my MRI and data but i have just immigrated to Canada, and i didn't bring that MRI with me, so i won't be able to share the data.
