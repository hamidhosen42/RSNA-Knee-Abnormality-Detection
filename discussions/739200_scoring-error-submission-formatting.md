# Scoring error - submission formatting

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/739200
- **Author:** **CavdadHun - VBECZ** (@viktorbecz, Expert) · _topic author_ · 2026-09-03 04:17 UTC · ▲1 · LB rank 1198
- **Topic votes:** 1 · **Comments/replies:** 4 · **Pinned:** no
- **Last comment:** 2026-09-08 06:24 UTC by CavdadHun - VBECZ
- **Cloned:** 2026-10-14 18:23 UTC (topic id 739200, raw JSON `_raw/739200.json`)

---

Hi All..
In the overview it says: 
"Submission File
For each row in the test set, you must predict a confidence score for each of the twelve target labels. The file should contain a header and have the following format:

StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
<uid_1>,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
<uid_2>,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
..."

Generally it is pretty straightforward, however, looking into the code sections, I see that someone using 1 decimal place, someone uses8 or more decimal places in the prediction confidence score.

So my question is that: Can be the number of decimal places  used in the submission.csv for a single confidence score can cause a scoring error? Should I limit the scores to two digits or else?

---

## Comments (4, threaded, oldest first)

### [1] Gallant Ngetich 

**Gallant Ngetich ** (@gallantngetich, Contributor) · 2026-09-03 04:27 UTC

Limit to two digits

#### ↳ [1.1] CavdadHun - VBECZ

**CavdadHun - VBECZ** (@viktorbecz, Expert) · _topic author_ · 2026-09-07 03:55 UTC · LB rank 1198

did but that does not help.. example... any ideas?

### [2] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-09-07 14:40 UTC · LB rank 639

Just looked at my file and 0.1300851 is an example of the format used.  Don't think that is your issue - with auc as metric its possible that integers could work but have not tried it :).

Have you tried predicting on more images by use of the train data rather than test.  Not sure how easy that is with your code.  Had some errors on this competition but format not one of them.  In past I have used clip to make sure the values are between 0.0 and 1.0.

#### ↳ [2.1] CavdadHun - VBECZ

**CavdadHun - VBECZ** (@viktorbecz, Expert) · _topic author_ · 2026-09-08 06:24 UTC · LB rank 1198

it was not an issue since now.. I have an ensemble pipeline, but submission building and saving uses the same method, but still getting scoring error... using previous runs' CSV then ensemble them should not be an issue either, as many others does that..
