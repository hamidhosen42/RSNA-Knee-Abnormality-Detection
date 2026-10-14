# Tips for improving AUC above 0.90?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745576
- **Author:** **Rachit Yadav** (@rachiteagles, Contributor) · _topic author_ · 2026-10-03 09:41 UTC · ▲1 · LB rank 1645
- **Topic votes:** 1 · **Comments/replies:** 2 · **Pinned:** no
- **Last comment:** 2026-10-04 11:00 UTC by ZavodRobotov
- **Cloned:** 2026-10-14 18:23 UTC (topic id 745576, raw JSON `_raw/745576.json`)

---

Hi everyone, I'm training DINOv2 models using labels derived from the reports and aiming for a score above 0.90. Ignoring uncertain labels hasn't helped in my local tests.

What changes helped you most: better labels, MRI slice selection, or training settings? Any useful public notebooks or discussions would be appreciated. Thanks!

---

## Comments (2, threaded, oldest first)

### [1] Sotarriva Alvarez Isai Roberto

**Sotarriva Alvarez Isai Roberto** (@irsotarriva, Contributor) · 2026-10-04 06:47 UTC · ▲2 · LB rank 975

Labels from the reports cap at around 0.9. To get past that the biggest improvements for me came from:
 a) ensembles of models
 b) data augmentation (flipping, zoom, etc).

If you check the code section there are many posts about how to get to the 0.943 score that is so common on the leaderboard, you could start by checking what they did.

Hope it helps you. Regards

### [2] ZavodRobotov

**ZavodRobotov** (@zavodrobotov, Expert) · 2026-10-04 11:00 UTC

There is small train dataset, or big and complex dataset? How many data sources? It there domain shift presented in test and train?
