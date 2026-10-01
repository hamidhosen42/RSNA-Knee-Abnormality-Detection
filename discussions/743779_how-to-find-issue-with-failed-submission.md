# How to find issue with failed submission

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743779
- **Author:** **Michael Egle** (@michaelegle, Contributor) · _topic author_ · 2026-09-27 01:46 UTC · ▲1 · LB rank 4357
- **Topic votes:** 1 · **Comments/replies:** 2 · **Pinned:** no
- **Last comment:** 2026-09-27 22:26 UTC by Chris Deotte
- **Cloned:** 2026-10-01 19:24 UTC (topic id 743779, raw JSON `_raw/743779.json`)

---

Newcomer. I've had no trouble with my submission running when I hit save and commit, but the last few times I've tried to submit, it's run for about the same time as it took for my initial notebook to run (if not longer) and then says that there was an unhandled exception in the notebook. Is there a way to see what the issue was?

---

## Comments (2, threaded, oldest first)

### [1] Enai

**Enai** (@enaimaan, Contributor) · 2026-09-27 16:20 UTC · ▲1 · LB rank 3323

Had a few similar issues in the past. Since there is no way to debug using logs when scoring, you will perhaps need to spend a few submissions to debug it. 
What helped me: first check what error message you get, is it "Notebook Threw Exception", or something else? Then, if you already have a successful submission, check what is actually different in terms of code vs the new failed one, this can help you reduce search space. otherwise, just comment out functional sections of code, and submit, if it works, then the commented out  section is the culprit, if it doesn't then the issue is somewhere else.

### [2] Chris Deotte

**Chris Deotte** (@cdeotte, Grandmaster) · 2026-09-27 22:26 UTC · ▲1 · LB rank 100

This is difficult to diagnosis. One trick is to infer 1300 train studies in a Kaggle commit notebook and see if that completes successfully because that is more similar to actual Kaggle submit then just commiting the few test examples.

The two common errors I have seen are NAN values in predictions and malformed DICOM files. So maybe try catching and dealing with both of these errors.
