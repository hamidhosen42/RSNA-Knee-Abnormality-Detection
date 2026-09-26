# Things that helped (and didn't) when approaching RSNA Knee from scratch   

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740162
- **Author:** **Nathan Jacob** (@nathanjacob, Expert) · _topic author_ · 2026-09-08 14:15 UTC · ▲1 · LB rank 460
- **Topic votes:** 1 · **Comments/replies:** 1 · **Pinned:** no
- **Last comment:** 2026-09-08 19:02 UTC by Rasoul Mojtahedzadeh
- **Cloned:** 2026-09-26 16:24 UTC (topic id 740162, raw JSON `_raw/740162.json`)

---

I've been working through this competition over the past few days and wanted to share what I've      
learned, since a lot of it wasn't obvious coming in.                                               
                                                                                                           
**The labeling situation is tricky.** Only 58 out of 4,407 studies have gold labels. The rest need to be 
extracted from radiology reports — and those reports are in 9+ languages. I wrote up the data structure  
and DICOM quirks in [Know Your Knee](https://www.kaggle.com/code/nathanjacob/know-your-knee-eda-dicom-deep-dive-and-baseline) if you're just getting started.                                                                                    
                                                                                                           
**The 4-arm ensemble architecture** that the top solutions use is really clever once you break it down - separate heads for different MRI planes fused with attention. I tried to explain the intuition behind it in [4-Arm Ensemble Explained](https://www.kaggle.com/code/nathanjacob/4-arm-ensemble-explained-rsna-knee-0-937).
                                                                                                           
**The multilingual reports** are an underexplored angle. Running DINOv2 with cross-lingual label
extraction covers a lot more of the dataset than English-only approaches. Wrote about that [here](https://www.kaggle.com/code/nathanjacob/dinov2-baseline-knees-in-9-languages).                    
                                                                                       
Would love to hear what's working for others. Anyone had luck with the efficiency track?

---

## Comments (1, threaded, oldest first)

### [1] Rasoul Mojtahedzadeh

**Rasoul Mojtahedzadeh** (@rasoulmojtahedzadeh, Expert) · 2026-09-08 19:02 UTC · LB rank 789

Nice summary!
What didn't work for me so far:
1. DinoV3 doesn't perform as good as DinoV2 for me, which is strange.
2. I tried hard to use reports as an auxiliary loss, unfortunately it hurts the performance no matter how much I played with its weight in the total loss. It seems that reports are quite noisy, unfortunately.
