# Clarification on Manual Annotation of Unlabeled Training MRIs

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740718
- **Author:** **oshiko0450** (@oshiko0450, Contributor) · _topic author_ · 2026-09-10 21:40 UTC · LB rank 1397
- **Topic votes:** 0 · **Comments/replies:** 2 · **Pinned:** no
- **Last comment:** 2026-09-12 08:23 UTC by ParsaAshrafi
- **Cloned:** 2026-09-26 16:24 UTC (topic id 740718, raw JSON `_raw/740718.json`)

---

Hi,

I would like to clarify one point about manual annotation of the training data.

I understand from previous discussions that:

* the 58 provided labels were assigned from the MRI images independently of the clinical reports;
* when the report and image-based label disagree, the image-based label should be considered authoritative;
* the remaining training reports may be used to derive weak labels.

The Competition Rules also state that submissions may not use information from hand labeling or human prediction of the validation dataset or test data records.

Would it be permitted for a participant to manually review some of the unlabeled MRI studies in the provided training set and create additional image-based labels, using the published competition label definitions, and then use those labels for model training?

If this is permitted, would participant-created labels for the provided training images be considered External Data under the competition rules, or would they be considered derived from the provided competition data?

Thank you for the clarification.

---

## Comments (2, threaded, oldest first)

### [1] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-09-11 03:17 UTC · LB rank 639

I believe the rules state that you cannot hand label test images - that's not applicable to this competition as we cannot see the test images.  It was applicable in competitions in the past when the test images were available for download/review.

If you want to manually examine and label the train images - go for it.  Modification of train data has NEVER been considered as "external data".

### [2] ParsaAshrafi

**ParsaAshrafi** (@parsaashrafii, Contributor) · 2026-09-12 08:23 UTC · LB rank 1035

Based on Section 3.4(b), the restriction applies to hand-labeling or human prediction of validation/test records. It does not prohibit manually reviewing the provided training MRIs. Since those images are part of the Competition Data, participant-created labels for the training set would generally be derived labels rather than External Data.

I would still recommend documenting the labeling protocol and asking the host for explicit confirmation before relying on it, especially if the labels are used in a final submission.
