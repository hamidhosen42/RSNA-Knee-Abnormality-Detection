# Corrupted DICOM Pixel Data Found in 2 Training Studies (ValueError Traceback)

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738708
- **Author:** **Tahmid Tawsif** (@tahmidtawsif, Contributor) · _topic author_ · 2026-09-01 11:48 UTC · ▲1 · LB rank 3403
- **Topic votes:** 1 · **Comments/replies:** 1 · **Pinned:** no
- **Last comment:** 2026-09-01 18:14 UTC by PC Jimmmy
- **Cloned:** 2026-10-09 09:46 UTC (topic id 738708, raw JSON `_raw/738708.json`)

---

Hi Everyone,

While building and decoding the volume cache for the training set using pydicom, I encountered pipeline crashes caused by corrupted pixel buffer sizes in 2 specific DICOM studies.

 Problem Description
The pydicom decoder fails when calling dcm.pixel_array due to a byte mismatch between the metadata header (Group 0028) and the actual pixel payload. The actual byte count in the buffer is exactly half of what is expected.

Affected Study UIDs & Errors:
Study UID: 1.2.826.0.1.3680043.8.498.34685905030370793639196564723935583035

ValueError: The number of bytes of pixel data is less than expected (409600 vs 819200 bytes)

Study UID: 1.2.826.0.1.3680043.8.498.37833587429731221455928642963031995680

ValueError: The number of bytes of pixel data is less than expected (173056 vs 346112 bytes)

Hopefully, the host team can inspect or patch these two DICOM volumes if necessary.

---

## Comments (1, threaded, oldest first)

### [1] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-09-01 18:14 UTC · LB rank 641

See earlier [discussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737163l)

I tried to use MedGemma to look at the DICOMs - I needed to error trap and ended up with the same 6 in that discussion not working on that LLM.

PS - it was a pretty useless effort.  After lots of tweaks to the code it took 6 days to run on local version of the model and the results were worse than the regrex code shared in lots of notebooks.  It's always possible I goofed on the label making.  I was surprised that it produced around 1500 to 2000 different text outputs per label - I had hoped it would be cleaner.
