# DOES Ensemble allowed in RSNA?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740624
- **Author:** **CavdadHun - VBECZ** (@viktorbecz, Expert) · _topic author_ · 2026-09-10 14:27 UTC · ▲-6 · LB rank 1200
- **Topic votes:** -6 · **Comments/replies:** 1 · **Pinned:** no
- **Last comment:** 2026-09-26 00:18 UTC by Chris Deotte
- **Cloned:** 2026-09-26 16:24 UTC (topic id 740624, raw JSON `_raw/740624.json`)

---

Guys, I would really appreciate some help....
When I have a single pipeline, producing a submission, and submit for scoring ,that works.. scored on LB..

HOWEVER.....
If  I try to ensemble - using multiple submissions to ensemble them into a single prediction submission - same output format, same saving etc, but it constantly getting a scoring error on LB..

I would really appreciate some help, or at least a standpoint if only me a noob here - which is ok - or not working in this competition, unlike in other competitions where it was working just fine...

thank you..

---

## Comments (1, threaded, oldest first)

### [1] Chris Deotte

**Chris Deotte** (@cdeotte, Grandmaster) · 2026-09-26 00:18 UTC · LB rank 93

Ensemble submissions are allowed. If you infer multiple models in a single notebook, be careful to free memory and avoid OOM errors.
