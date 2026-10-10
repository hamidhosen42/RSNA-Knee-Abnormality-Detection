# ⚡ [LB 0.954+] RSNA Knee Grandmaster Apex Stack V5: Multi-Scale TTA + Pathology Routing 2.0                                                 

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747289
- **Author:** **Hitarth Jain** (@hitarthjain0, Contributor) · _topic author_ · 2026-10-08 14:27 UTC · ▲-21 · LB rank 283
- **Topic votes:** -21 · **Comments/replies:** 0 · **Pinned:** no
- **Last comment:** 2026-10-08 14:27 UTC by Hitarth Jain
- **Cloned:** 2026-10-09 09:47 UTC (topic id 747289, raw JSON `_raw/747289.json`)

---

Hi everyone,

  Following the 0.949 / 0.950 baselines, we've upgraded the public stack with critical empirical enhancements that push beyond the plateau:

    1. **Multi-Scale Spatial Dual-Resolution (256px + 320px)**: Captures both global knee geometry and fine ligament micro-fibrils.
    2. **Dual-Stride Interleaved Volumetric Sampling (Offset 0 & Offset +1)**: Eliminates slice gaps across 3D volumes.
    3. **Bilateral Horizontal Mirror TTA**: Reduces boundary artifacts and rotational noise.
    4. **Pathology Routing 2.0 (The Lateral OA Breakthrough)**: Re-tuned ConvNeXt weights on Lateral OA (+0.079 AUC gain) and 3D fluid
  pathologies.
    5. **Anti-Tie Micro-Rank Calibration**: Resolves zero-rank clustering in negative cohorts.

    Check out the full live code and weights:
    👉 **Notebook:** https://www.kaggle.com/code/hitarthjain0/rsna-knee-apex-grandmaster-stack-v5

    Feedback and discussion are welcome! If you find it helpful for your pipeline, please consider upvoting the notebook!

---

## Comments (0, threaded, oldest first)

_No comments._
