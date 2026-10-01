# PSA: bf16 on Kaggle's T4/P100 silently cost me 0.14 AUC (0.851 → 0.710). How to spot it.

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744230
- **Author:** **Md Kishor Morol** (@kishormorol, Contributor) · _topic author_ · 2026-09-29 02:54 UTC · ▲2 · LB rank 824
- **Topic votes:** 2 · **Comments/replies:** 2 · **Pinned:** no
- **Last comment:** 2026-10-01 12:27 UTC by Quan Vu
- **Cloned:** 2026-10-01 19:24 UTC (topic id 744230, raw JSON `_raw/744230.json`)

---

My own model scored **0.710** public, while held-out evaluation on the 58 annotated studies said ~0.86. It wasn't the model. Here's what happened, in case it saves someone a week.

**What went wrong**
- My inference kernel used `torch.autocast(dtype=torch.bfloat16)`, copied from training on an A100.
- T4 and P100 have no bf16 units. PyTorch doesn't error: it **emulates**, and each forward pass got ~50x slower (~0.9 s per model pass per study for a ViT-S with 24 images).
- 10 checkpoints x 5 TTA views = 50 passes per study, x 1,322 hidden studies = ~17 hours.
- My kernel has a time guard (stop at 8 h and write a prior of 0.5 for whatever isn't done). It hit the guard after ~620 studies, so **more than half the test got a constant 0.5**. That scores ~0.71 for a model that's really ~0.86.
- On the 3 visible test studies, everything looked fine. The failure only exists at hidden-test scale.

**The fix:** `dtype=torch.float16` (T4 has fp16 tensor cores). Same model, resubmitted: **0.851**.

**Checklist for your own kernel**
1. `torch.cuda.is_bf16_supported()` on the scoring GPU. If it's False, don't autocast to bf16.
2. Log seconds per study, and extrapolate to ~1,300 hidden studies *plus* headroom, before trusting a public score.
3. If you have a time guard with a fallback, log how many studies hit the fallback. A silent partial fallback looks exactly like a bad model.
4. To tell "bad model" from "bad pipeline", run your exact inference path on held-out *train* studies (DICOM -> decode -> TTA -> rank) and score it against the annotated ones. Mine gave 0.858 there, which is how I knew the leaderboard number was wrong.

---

Separately: I have an **independent arm** (DINOv2 ViT-S, 6 series x 16 slices over 140 mm at 336 px, fully fine-tuned, 5 folds) that shares no weights with the public ensemble. If you're at **0.945+** and want an uncorrelated component before the 15 Oct merge deadline, happy to talk.

---

## Comments (2, threaded, oldest first)

### [1] Roman Vuskov

**Roman Vuskov** (@romanvuskov, Contributor) · 2026-10-01 11:32 UTC · LB rank 31

Good catch! You need to use `torch.cuda.is_bf16_supported(including_emulation=False)` though as the default may still output `true` depending on torch version.

### [2] Quan Vu

**Quan Vu** (@quan0095, Master) · 2026-10-01 12:27 UTC · LB rank 14

As far as I understand, the P100 and T4 GPUs do not support bfloat16.
