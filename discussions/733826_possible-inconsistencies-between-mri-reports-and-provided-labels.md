# Possible inconsistencies between MRI reports and provided labels

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826
- **Author:** **Cho Royou** (@ngyzly, Expert) · _topic author_ · 2026-08-08 13:08 UTC · ▲27 · LB rank 1383
- **Topic votes:** 27 · **Comments/replies:** 2 · **Pinned:** no
- **Last comment:** 2026-08-08 14:02 UTC by Po-Hao "Howard" Chen
- **Cloned:** 2026-10-14 18:23 UTC (topic id 733826, raw JSON `_raw/733826.json`)

---

Hello,

I am auditing the relationship between the 12 provided binary labels and the corresponding MRI reports. I performed a preliminary report-only review on 20 multilingual studies selected from the 58 studies with complete labels.

For each study, I manually classified ACL injury, MCL injury, medial and lateral meniscus tears, medial/lateral/patellofemoral osteoarthritis, joint effusion, synovitis, Baker's cyst, bone contusion, and acute fracture.

I used a strict textual rule:

- Positive when the report explicitly affirms the finding
- Negative when the report explicitly negates or does not mention it
- Uncertain statements were not treated as definite positives

Across 240 decisions, the report-only labels matched the provided labels in 198 cases:

- Overall agreement: **82.5%**
- Positive predictive agreement: **68/93 = 73.1%**
- Positive recall: **68/85 = 80.0%**
- TP: 68, FP: 25, FN: 17, TN: 130

Some discrepancies appear difficult to explain as ordinary interpretation errors.

### Examples

| StudyInstanceUID | Report evidence | Provided label |
|---|---|---|
| 1.2.826.0.1.3680043.8.498.11287937729196958426538087439102017580 | Impression explicitly states “Synovitis of left knee and massive joint effusion.” | Effusion = 1, but Synovitis = 0 |
| 1.2.826.0.1.3680043.8.498.54977581323931277817074741182670080450 | “Leve derrame articular con mínima sinovitis” (“Mild joint effusion with minimal synovitis”). | Effusion = 0 and Synovitis = 0 |
| 1.2.826.0.1.3680043.8.498.26790702379506190936834447203448882465 | “There is a mild knee effusion.” | Effusion = 0 |
| 1.2.826.0.1.3680043.8.498.73926443729786165628848843707532839995 | The German report explicitly states “Baker-Zyste.” | Baker's cyst = 0 |
| 1.2.826.0.1.3680043.8.498.90283565381042081768587894596970552767 | “OA femorotibial medial y lateral. OA patelofemoral. Derrame con sinovitis.” | Medial OA = 0 and Lateral OA = 0, while PF OA, Effusion, and Synovitis = 1 |
| 1.2.826.0.1.3680043.8.498.28925345859498351203477642741908452608 | The Turkish report states “Lateral menisküs normal” (“The lateral meniscus is normal”). | Lateral meniscus tear = 1 |
| 1.2.826.0.1.3680043.8.498.12448079646359892252441208258836556945 | The report describes patellar chondropathy but contains no apparent mention of synovitis. | PF OA = 1 and Synovitis = 1 |
| 1.2.826.0.1.3680043.8.498.75187434248356774277526985329346125190 | The Bulgarian report explicitly describes an osteochondral fracture. | Acute fracture = 0 |

The last example may reflect a deliberate distinction between an osteochondral injury and the competition's definition of acute fracture, rather than an incorrect label.

Similar ambiguity may affect:

- Whether chondropathy should be considered osteoarthritis
- Whether bone marrow edema should be considered bone contusion
- Whether osteochondral, subchondral, or insufficiency fractures count as acute fractures
- Whether minimal synovitis should receive a positive synovitis label
- Whether suprapatellar or intra-articular fluid should always count as joint effusion

### Questions for the organizers

1. Were the labels assigned independently from the MRI images, rather than extracted from the reports?
2. If image interpretation and report text disagree, should the image-derived label be considered authoritative?
3. Could you provide the exact positive-label definitions for osteoarthritis versus chondropathy, bone contusion versus nonspecific bone marrow edema, acute fracture versus osteochondral/subchondral/insufficiency fractures, and joint effusion versus synovitis?
4. Do negative labels mean that the finding was confirmed absent, or can they also mean that it was not annotated?
5. Are multiple reports or bilateral examinations sometimes associated with one StudyInstanceUID? If so, how were the study-level labels consolidated?
6. Should the examples above be considered expected report-image disagreement, or are any of them possible annotation issues?

Clarification would be particularly helpful for participants using the reports to train auxiliary models or generate weak labels.

Thank you.

---

## Comments (2, threaded, oldest first)

### [1] Handudu

**Handudu** (@handudu, Expert) · 2026-08-08 13:40 UTC

Thank you; this is crucial for the competition.

### [2] Po-Hao "Howard" Chen

**Po-Hao "Howard" Chen** (@javaduke95, Contributor) · _competition host_ · 2026-08-08 14:02 UTC · ▲5

Hello - You have encountered a major design choice in this challenge, and what we believe makes this challenge instructive, real-world, and difficult (although we are not blind to the leaderboard 😄 - it is *incredible* how well a community can do when collaborating, better than any one individual team).

A related discussion is [here](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733491).  It outlines the design principles that led to perceived inconsistencies.

Below, I will focus on specific questions.

**Were the labels assigned independently from the MRI images, rather than extracted from the reports?**

Yes

**If image interpretation and report text disagree, should the image-derived label be considered authoritative?**

Yes.  Note that only a small sample of provided data contains both.  It is intended to help participants surface this conclusion.

**Could you provide the exact positive-label definitions for osteoarthritis versus chondropathy, bone contusion versus nonspecific bone marrow edema, acute fracture versus osteochondral/subchondral/insufficiency fractures, and joint effusion versus synovitis?**

Please refer to the pinned [Overview discussion post](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343).  At the bottom, textual and image-based explanations of each abnormality is provided.

**Do negative labels mean that the finding was confirmed absent, or can they also mean that it was not annotated?**

It means the finding was annotated as absent using methods outlined in [Overview discussion post](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343) under **Label Description** section.

**Are multiple reports or bilateral examinations sometimes associated with one StudyInstanceUID? If so, how were the study-level labels consolidated?**

Yes. In clinical practice, both knees may occasionally be scanned under one StudyInstanceUID. For the challenge, each bilateral study or bilateral report was individually reviewed, and the released report text or DICOM metadata was adjusted as needed to provide sufficient information for participants to disambiguate the relevant study/studies.

**Should the examples above be considered expected report-image disagreement, or are any of them possible annotation issues?**

Discrepancies are plausible and expected because clinical reports typically involve one signing radiologist who created it for clinical care, and the image-based labels uses multiple readers with stricter image-based thresholds.

I hope this helps, and good luck on the challenge!
