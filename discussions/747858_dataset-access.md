# Dataset Access

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747858
- **Author:** **Yogesh Krishna** (@yogeshkrishna1709, Contributor) · _topic author_ · 2026-10-09 19:19 UTC
- **Topic votes:** 0 · **Comments/replies:** 1 · **Pinned:** no
- **Last comment:** 2026-10-11 06:24 UTC by Mingquan Zhuang
- **Cloned:** 2026-10-14 18:23 UTC (topic id 747858, raw JSON `_raw/747858.json`)

---

I am new to Kaggle. Can someone help me out in understanding how to access this data in Kaggle notebook and also externally like VS code IDE on local system.  How to access its individual files? Will this data be available in Kaggle to access even after the competition ends?

---

## Comments (1, threaded, oldest first)

### [1] Mingquan Zhuang

**Mingquan Zhuang** (@mingquanzhuang, Contributor) · 2026-10-11 06:24 UTC · LB rank 3192

In a Kaggle Notebook, attach the competition data under Add Input and check the mount instead of assuming one path:

```python
from pathlib import Path
for root in [Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
             Path('/kaggle/input/rsna-knee-abnormality-detection')]:
    if (root / 'train.csv').exists():
        print(root)
```

I encountered both mount layouts while adapting a fork. The Data tab documents individual DICOMs as `train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm`; `train_series.csv` maps series to studies. The Data Explorer lets you inspect individual files. For VS Code on your own machine, the Data tab also has a Download all files button, but the full dataset is about 569.76 GB, so check disk space first.

I could not verify any guarantee that competition files will remain downloadable after the deadline. For long-term access, check the competition rules and ask the organizers. Retain only data you are allowed to keep under those rules.
