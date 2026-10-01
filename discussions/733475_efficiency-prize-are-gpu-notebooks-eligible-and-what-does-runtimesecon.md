# Efficiency Prize: are GPU notebooks eligible, and what does RuntimeSeconds cover?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733475
- **Author:** **epicfangs** (@epicfangs, Contributor) · _topic author_ · 2026-08-07 02:36 UTC
- **Topic votes:** 0 · **Comments/replies:** 2 · **Pinned:** no
- **Last comment:** 2026-08-07 16:56 UTC
- **Cloned:** 2026-10-01 19:24 UTC (topic id 733475, raw JSON `_raw/733475.json`)

---

Two questions about the Efficiency Prize, both of which change how a submission would be built.

1. Are GPU notebooks eligible for the Efficiency Prize, or is that track CPU-only? The Code Requirements allow either a CPU or a GPU notebook up to 9 hours for the competition generally, but the Efficiency Prize Evaluation section doesn't say whether the second track narrows that.

2. What does RuntimeSeconds actually cover? The Overview describes it as "the number of seconds it takes for the submission to be evaluated". Is that the full wall time of the rerun notebook — including package installation, model loading and reading the test DICOMs — or only an inner scoring portion?

Both matter quite a bit here because the DICOM read and decode path is CPU-bound, so whether a GPU is available at all, and whether setup time is counted, point at fairly different designs.

Thanks!

---

## Comments (2, threaded, oldest first)

### [1] Ryan Holbrook

**Ryan Holbrook** (@ryanholbrook, Staff) · 2026-08-07 10:22 UTC

1. GPU notebooks are allowed.
2. RuntimeSeconds is the time from when execution of the submitted notebook begins to the time when execution ends. If your notebook is installing packages, loading models, and reading DICOMs, then all of that is included in RuntimeSeconds.

#### ↳ [1.1] (unknown)

**(unknown)** · 2026-08-07 16:56 UTC

_(no content — deleted or empty)_
