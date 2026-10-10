# How the ground truth labels are labelled ?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733864
- **Author:** **Dennis** (@dennisfong, Expert) · _topic author_ · 2026-08-08 17:18 UTC · ▲3 · LB rank 2214
- **Topic votes:** 3 · **Comments/replies:** 1 · **Pinned:** no
- **Last comment:** 2026-08-08 19:58 UTC by Tom Aindow
- **Cloned:** 2026-10-09 09:46 UTC (topic id 733864, raw JSON `_raw/733864.json`)

---

For StudyInstanceUID: 1.2.826.0.1.3680043.8.498.73926443729786165628848843707532839995

The ground truth shows Baker's as 0, which means it does not present in the MRI.

The translated (from Google translate) report is:
T2-hyperintense linear radial lesion of the pars intermedia of the medial meniscus with incomplete separation of a small flap (3/7, 5/109-110). The initially extruded pars intermedia of the medial meniscus shows mucoid fraying in the body. Further subtle horizontal T2 hyperintensity is seen in the posterior horn of the medial meniscus (3/7, 4/20). Focal T2-hyperintense signal is present in the medial tibial plateau (7/22). Fluid signal is present around the MCL. The ACL, PCL, MCL, and LCL are intact and show normal signal intensity. Small cystic changes are present at the ACL insertion sites. There is a small joint effusion. A Baker's cyst is present, measuring up to 2 x 1.5 x CC (3.5 cm).

The last sentence shows "A Baker's cyst is present, measuring up to 2 x 1.5 x CC (3.5 cm)." From the label description provided by @javaduke95 (https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343), "Baker (popliteal) cyst: A moderate or large fluid collection in the characteristic location behind the knee." How to define "moderate or large"?

I know that different radiologists may have different perspectives towards the images. I wonder if the one who wrote the report found the Baker's cyst, but the one gave the ground truth label does not found it. I am confused of which radiologist I should follow.

The ground truth is given from reading the report or the MRI images?

I am not a radiologist. I am pleased to have someone to share the insight on finding the Baker's cyst?

Thanks.

---

## Comments (1, threaded, oldest first)

### [1] Tom Aindow

**Tom Aindow** (@taindow, Master) · 2026-08-08 19:58 UTC · ▲1 · LB rank 81

See the response from competition hosts here: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826.
