# 8 public LLM label sets vs gold-58: 11 of 12 findings show no clear difference

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747540
- **Author:** **karttikjangid05** (@karttikjangid05, Contributor) · _topic author_ · 2026-10-08 23:45 UTC · ▲1 · LB rank 271
- **Topic votes:** 1 · **Comments/replies:** 1 · **Pinned:** no
- **Last comment:** 2026-10-09 00:29 UTC by hengck23
- **Cloned:** 2026-10-14 18:23 UTC (topic id 747540, raw JSON `_raw/747540.json`)

---

Many of us train on LLM labels read from the reports, and there are now several public sets. I scored
  eight of them on the 58 gold studies, with a 95% bootstrap CI for every source and finding
  (class-stratified, 2,000 resamples).

  Result: for 11 of 12 findings, the best and the median set are not clearly different. MCL has 9
  positives, too few to rank. On gold-74 (with JEV's 16 extra studies) all 12 findings show no clear

  A plain mean of six CC0 sets has a CI that overlaps the best single set on all 12 findings. Macro AUC on
  gold-58: consensus 0.8887, steven_v4 0.8927, steven_v2 0.8873.

  | Finding | Gold-58 positives | Verdict | Best source (AUC, 95% CI) | Consensus (AUC, 95% CI) |
  |---|---|---|---|---|
  | ACL | 24 | no clear difference | pilkwang 0.997 (0.988 to 1.000) | 0.993 (0.974 to 1.000) |
  | MCL | 9 | too few positives to rank sources | yunus 0.984 (0.955 to 1.000) | 0.986 (0.955 to 1.000) |
  | Medial Meniscus | 26 | no clear difference | steven_v2 0.954 (0.901 to 0.993) | 0.945 (0.885 to 0.989)
  |
  | Lateral Meniscus | 23 | no clear difference | steven_v4 0.879 (0.766 to 0.973) | 0.876 (0.759 to 0.971)
  |
  | Medial OA | 15 | no clear difference | steven_v4 0.932 (0.830 to 0.995) | 0.928 (0.811 to 0.997) |
  | Lateral OA | 11 | no clear difference | yunus 0.835 (0.662 to 0.960) | 0.818 (0.613 to 0.971) |
  | PF OA | 21 | no clear difference | steven_v4 0.902 (0.808 to 0.977) | 0.883 (0.777 to 0.972) |
  | Effusion | 35 | no clear difference | steven_v4 0.877 (0.771 to 0.955) | 0.873 (0.771 to 0.952) |
  | Synovitis | 27 | no clear difference | steven_v2 0.790 (0.670 to 0.898) | 0.751 (0.623 to 0.872) |
  | Baker's | 12 | no clear difference | steven_v2 0.946 (0.868 to 1.000) | 0.932 (0.835 to 0.998) |
  | Contusion | 19 | no clear difference | steven_v4 0.860 (0.755 to 0.942) | 0.855 (0.750 to 0.945) |
  | Fracture | 18 | no clear difference | qwen 0.876 (0.774 to 0.965) | 0.824 (0.679 to 0.942) |

  "Clear winner" would mean the best source's lower bound is above the median source's upper bound. The
  rule was set before the first gold run, with no per-finding tuning on gold.

  Limits: 58 studies is a small ruler. steven_v2's Synovitis rule was informed by gold-58, so treat
  Synovitis with care. This measures agreement with gold, not leaderboard impact.

  Consensus file (CC0) and all metrics:
  https://www.kaggle.com/datasets/karttikjangid05/rsna-knee-public-llm-label-sets-compared
, 
  Notebook: https://www.kaggle.com/code/karttikjangid05/rsna-knee-which-llm-labels-can-you-trust

  Thanks to every label author and to JEV for the 74-study set and for showing that per-finding blending on
  gold overfits.

---

## Comments (1, threaded, oldest first)

### [1] hengck23

**hengck23** (@hengck23, Grandmaster) · 2026-10-09 00:29 UTC · LB rank 718

Google for papers that can predict OOD performance based on in-distribution data, or predict performance based on limited data in N-shot learning settings, etc ...
