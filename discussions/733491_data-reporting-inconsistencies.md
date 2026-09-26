# Data/Reporting Inconsistencies

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733491
- **Author:** **avg-HU** (@patrickmatulich, Contributor) · _topic author_ · 2026-08-07 05:03 UTC · ▲8 · LB rank 717
- **Topic votes:** 8 · **Comments/replies:** 3 · **Pinned:** no
- **Last comment:** 2026-08-27 11:39 UTC by telingc
- **Cloned:** 2026-09-26 16:24 UTC (topic id 733491, raw JSON `_raw/733491.json`)

---

The decision criteria for the competition is binary for the 12 areas. However, for scenarios like the meniscus in which it is common to have degenerative interstitial signal this is counted as 0 and 1 in the sample set. Can you clarify the grading of these findings? Is it supposed to be 0 = no tear and 1 = tear? If so, then the example dataset has errors. Secondly, I noticed in the reports there are some inconsistencies such as "Minimal reactive bone marrow edema present…related to adjacent meniscal tear…No acute fracture or bone bruise" which is somewhat contradictory. Especially since in that same dictation it says that the medial compartment cartilage is intact. 

I don't know of a scenario in which a patient has "bone marrow edema" in the setting a meniscal tear and normal cartilage in which you could factually say "no contusion"

Another report describes chondral lesion within the lateral facet of the patella which should represent chondrosis of the patellofemoral compartment, but the impression lists "lateral compartment chondromalacia" which is also factually wrong

"Lateral meniscus and medial meniscus are normal. Loss of integrity of the medial meniscus in the trunk section and changes in signal intensity consistent with horizontal tear in the posterior horn are observed. "

---

## Comments (3, threaded, oldest first)

### [1] (unknown)

**(unknown)** · 2026-08-07 05:34 UTC

_(no content — deleted or empty)_

### [2] Po-Hao "Howard" Chen

**Po-Hao "Howard" Chen** (@javaduke95, Contributor) · _competition host_ · 2026-08-07 11:51 UTC · ▲4

Hello! This is an important question and useful observation.

The radiology reports in the training set are deidentified versions of the original contributor reports. They are provided as clinical text and may contain ambiguous wording, internal inconsistencies, or findings that do not map one-to-one to the challenge’s binary targets. This design reflects a common real-world setting in medical imaging where large-scale model development often relies on routinely generated clinical reports.

The sample labels were assigned from image review using the challenge annotation rubric by two independent readers with a third adjudicating disagreements. 

For meniscal tears, the target is a definite tear; intrasubstance degenerative signals that do not reach the articular surface are negative.  The same process was used for the testing set.

Similarly, marrow edema, cartilage findings, and terminology in the narrative or impression do not by themselves determine the contusion, osteoarthritis, or other challenge labels. Those labels reflect the standardized image-review criteria.  (See the [Overview discussion thread](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343) near the bottom where these are defined)

A report may therefore use language that appears inconsistent with the corresponding image-based label because the original report and the image-based labels were generated through different processes.

### [3] telingc

**telingc** (@telingc, Contributor) · 2026-08-27 11:39 UTC · LB rank 2761

good question, tks tks.👍
