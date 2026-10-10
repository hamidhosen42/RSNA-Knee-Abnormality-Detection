# Six data traps in RSNA Knee: slice order, missing laterality, silent reports (EDA)

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741310
- **Author:** **gchauhan** (@gchauhan, Contributor) · _topic author_ · 2026-09-14 15:31 UTC · ▲1 · LB rank 1696
- **Topic votes:** 1 · **Comments/replies:** 0 · **Pinned:** no
- **Last comment:** 2026-09-14 15:31 UTC by gchauhan
- **Cloned:** 2026-10-09 09:46 UTC (topic id 741310, raw JSON `_raw/741310.json`)

---

I published an EDA that tries to answer one question: what does this data force a model to do? 
https://www.kaggle.com/code/gchauhan/eda-rsna-knee-what-the-data-tells-us

**What stood out**

- Training targets come from reports; the score comes from radiologists reading images. On the 58 expert-labelled studies, reports are silent on about half of the expert-positive Synovitis cases. Fracture is in 7% of reports but 31% of expert labels. Treating "not mentioned" as "absent" teaches the model the reporting habit, not the finding.
- InstanceNumber runs against slice position in over a third of series. If you stack slices by instance number, neighbouring-slice context is scrambled for those series. Sort by position along the slice normal.
- Laterality is missing in about half of studies. Work out the side from geometry before mirroring right knees to left. If you flip for augmentation, medial/lateral labels (menisci, OA) have to swap too.
- The CSV hides the contrast. "non-FS" mixes T1, PD and T2, and most repeated series types are a second contrast rather than a retake. Keying slots only on plane × FS merges different images.
- A study is a variable set of 3–14 series. The most common combination covers only 40% of studies, so a missing-series mask isn't optional.
- Slices are ~10× farther apart than pixels (~0.3 mm in-plane vs ~3.5 mm between slices). That favours 2.5D, meaning a 2D encoder plus a depth model, over 3D from scratch, which is also what past RSNA spine/PE winners found.

The last section maps each finding to a modelling consequence and to what winners of past RSNA imaging competitions did.

Report labels are from @pilkwang's LLM-read report labels (https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels). Thanks!

Question for others: how are you handling "not mentioned" in the report labels: as negative, as a prior, or masked? And has anyone measured the effect of fixing slice order on local CV?

---

## Comments (0, threaded, oldest first)

_No comments._
