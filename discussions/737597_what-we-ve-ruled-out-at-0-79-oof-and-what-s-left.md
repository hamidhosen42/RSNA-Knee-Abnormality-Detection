# What we've ruled out at 0.79 OOF — and what's left

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737597
- **Author:** **Berat Kirbiyik** (@beratkirbiyik, Contributor) · _topic author_ · 2026-08-26 08:57 UTC · ▲4 · LB rank 1241
- **Topic votes:** 4 · **Comments/replies:** 0 · **Pinned:** no
- **Last comment:** 2026-08-26 08:57 UTC by Berat Kirbiyik
- **Cloned:** 2026-10-09 09:46 UTC (topic id 737597, raw JSON `_raw/737597.json`)

---

We're stuck at 0.7815 OOF (0.870 public) and have run a series of controlled ablations, each single-variable, fold-0 unless stated. Sharing the negatives because they seem more useful than the positives.

Setup: EfficientNet-B0, 4 sequence slots (Axial_FS, Sagittal_FS, Coronal_FS, Sagittal_T1), 9 adjacent slices at 288px from a 140mm physical crop, task-specific masked attention over slots, weighted BCE, 5-fold scanner-aware split.

What moved the needle:

192px, no crop, 24 equal-spaced slices     0.7695 OOF
288px, 140mm crop, 12 adjacent             0.7777 OOF
288px, 140mm crop, center-9 of 24          0.7815 OOF

Slice geometry mattered more than resolution: center-9 adjacent beat 9 equal-spaced across 24 by +0.018 on fold 0.

What didn't move it (fold 0, baseline 0.7914):

EMA, decay 0.999          0.7940   +0.003
Mixup, alpha 0.2          0.7937   +0.002
60 epochs, cosine         0.7922   +0.001
Asymmetric loss           0.6526   -0.139

Three different regularization approaches all land within noise of each other. That reads to me like the bottleneck isn't the training regime.

Laterality — tested and ruled out. Right and left knees are mirror images along the medial-lateral axis, so I expected the side-dependent targets to suffer without normalization. They don't:

mean |right - left| AUC gap, side-dependent targets   0.014
same for all other targets                            0.031

If anything the side-dependent targets are less affected. Also worth noting: the Laterality tag is present on 2185/4407 studies, but deriving side from the image centre in patient coordinates resolves 4274/4407 and agrees with the tag 98.4% of the time.

Where the error concentrates:

Baker's           0.855      Lateral Meniscus  0.740
ACL               0.853      PF OA             0.748
Effusion          0.843      Lateral OA        0.751
Fracture          0.832      MCL               0.758

The pattern I see: large, discrete, positionally distinctive findings work. Thin, small, graded ones don't. A Baker's cyst is a fluid sac in a fixed location. A meniscal tear is 1-3mm of signal that has to touch an articular surface on two consecutive slices.

The question: for those who are past 0.90 — is the gap in how the model attends spatially? We pool globally over the encoder output. A 3mm tear is roughly 0.01% of a 288px frame. I'm considering either an ROI crop around the joint line or spatial attention pooling instead of GAP, but I'd rather not spend a week on the wrong one.

Happy to answer questions on any of the above.

---

## Comments (0, threaded, oldest first)

_No comments._
