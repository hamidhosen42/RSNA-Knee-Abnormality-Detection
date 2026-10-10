# Has anyone tried YOLO / ROI cropping for the knee joint?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/736999
- **Author:** **Mehmet Özer** (@ozermehmet, Contributor) · _topic author_ · 2026-08-22 16:44 UTC · ▲3 · LB rank 408
- **Topic votes:** 3 · **Comments/replies:** 0 · **Pinned:** no
- **Last comment:** 2026-08-22 16:44 UTC by Mehmet Özer
- **Cloned:** 2026-10-09 09:46 UTC (topic id 736999, raw JSON `_raw/736999.json`)

---

I'm wondering if a simple detector -> crop pipeline could help here, especially for subtle meniscus/ligament findings.

Something along the lines of:

```python
bbox = detector(image)
roi = crop(image, bbox, margin=0.15)
prediction = model(roi)
```

Has anyone tried this? Curious whether the extra localization step gives a meaningful gain over using the full FOV, or whether things like edema/effusion are better handled with the full image.

---

## Comments (0, threaded, oldest first)

_No comments._
