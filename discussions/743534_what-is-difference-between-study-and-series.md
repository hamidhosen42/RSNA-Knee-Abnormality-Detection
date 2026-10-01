# What is difference between study and series?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743534
- **Author:** **Yogesh Krishna** (@yogeshkrishna1709, Contributor) · _topic author_ · 2026-09-26 06:50 UTC
- **Topic votes:** 0 · **Comments/replies:** 3 · **Pinned:** no
- **Last comment:** 2026-09-28 16:25 UTC
- **Cloned:** 2026-10-01 19:24 UTC (topic id 743534, raw JSON `_raw/743534.json`)

---

What do these terms in problem statement actually mean? 
In a session, multiple MRIs taken, are these of same individual or different individuals? Because neither taking multiple MRIs of same individuals make sense nor does different individuals MRIs taken in same session/study.
Also does 1 study mean a single MRI only or multiple MRIs? (Ofcourse, represented as several DICOM series/files)

---

## Comments (3, threaded, oldest first)

### [1] Wisdom Aduah

**Wisdom Aduah** (@anaphase21, Contributor) · 2026-09-26 08:42 UTC · ▲2 · LB rank 3347

I will offer my layman understanding of this. All the series within a study represents a single person. So why multiple series? Well, each series tries to provide different evidence, or makes it easier to identify different abnormalities.

Within a study, three different views of the knee are taken: side view (sagittal), front view (coronal) and bottom view (axial). Each of these planes provides the reviewer a different angle to look for an abnormality. This is because an abnormality may only be visible when looking at it from one particular view (which you can't tell beforehand when performing the MRI scan). 

Each of these three planes may also have different properties such as fat suppression and fluid sensitivity. A normal MRI will make fat appear bright, which can make it sometimes difficult to differentiate fat from fluids. A fat suppressed series only allows fluids to look bright while suppressing fats from appearing bright. This makes it easier to identify abnormalities related to fluids such as effusion and synovitis. See the hosts's post [here](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343) for more on this.

The slices in a series may also vary by thickness. Series with many slices will typically have thinner slices. If you consider all these variations (view, fat suppression, fluid sensitivity, and thickness), it's not difficult to see why a study can have up to 14 series. A sagittal series may have fat suppression or not. The same sagittal series may have different slice thickness as well.

In order to make a judgment about a knee abnormality, the reviewer will need to see all these series (different angles, different fat and fluid configurations) in order to have a holistic understanding of the knee's condition.

#### ↳ [1.1] Yogesh Krishna

**Yogesh Krishna** (@yogeshkrishna1709, Contributor) · _topic author_ · 2026-09-26 09:32 UTC

Thanks man great explaination

### [2] (unknown)

**(unknown)** · 2026-09-28 16:25 UTC

_(no content — deleted or empty)_
