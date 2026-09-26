# Clarification on Training Data

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733836
- **Author:** **agr hmmm** (@agrimrai29, Contributor) · _topic author_ · 2026-08-08 14:43 UTC · ▲3 · LB rank 3158
- **Topic votes:** 3 · **Comments/replies:** 4 · **Pinned:** no
- **Last comment:** 2026-08-08 23:39 UTC by PC Jimmmy
- **Cloned:** 2026-09-26 16:24 UTC (topic id 733836, raw JSON `_raw/733836.json`)

---

Hi,

I’m trying to understand the intended training setup. From train.csv, I see only 58 studies with the 12 ground truth labels, while the remaining 4.3k studies have MRI data and reports but no labels. 
Since the test set appears to provide MRI data without reports (as per [this discussion ](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733592))
what is the intended way to use the large unlabeled training set? 

With only 58 labeled studies, this doesn't seem like a conventional supervised CV problem, while an NLP-based approach doesn't transfer directly if test reports aren't available. What is the intended learning setup here?

Am i expected to use self-supervised learning, report-derived weak labels/pseudo-labeling, or another semi-supervised approach to train the MRI model?

Thanks

---

## Comments (4, threaded, oldest first)

### [1] k256.dev

**k256.dev** (@k256net, Master) · 2026-08-08 15:07 UTC · ▲1 · LB rank 1952

The dataset description mentions that part of the training data can be labeled
based on the reports — in other words, as long as you have the reports, labeling
them one way or another is possible. You could even do it by hand, although the
reports are written in multiple languages, so in most cases that would be hard.
As discussed in other threads, using a local LLM is one option here.

I think report-derived weak labels are the starting line. Most of the public
baseline notebooks seem to start from there.

#### ↳ [1.1] agr hmmm

**agr hmmm** (@agrimrai29, Contributor) · _topic author_ · 2026-08-08 20:15 UTC · ▲2 · LB rank 3158

Gotcha. Thanks
Not the best dataset for a compitition in my opinion. Could be wrong. Wdyt. I think let's get the proper data (all label or all report based) and when see how different people come up with different architectures and models and best one wins.

##### ↳ ↳ [1.1.1] Tucker Arrants

**Tucker Arrants** (@tuckerarrants, Master) · 2026-08-08 20:41 UTC · ▲4 · LB rank 19

This is my current understanding. There's no proper ground truth "label" here. A radiologist reads an MRI and writes a report, but scans are hard to read and two radiologists can disagree. That happens in every medical imaging competition, you're just not given the report, so you take the label as truth when it isn't.

What's different is the host told us how the labels were made: two readers against a published rubric, a third adjudicating, same for the test set. So there is a procedure and it disagrees with the reports in a known direction: the readers use stricter thresholds than a radiologist writing for clinical care, so the reports over-call.

The reports are arguably a richer dataset than usual, since a report says more than a binary label. The reason you'd want more than the ~50 provided labels is to calibrate your extractor against the competition's annotators - you're trying to match your label generator to theirs, which is hard with so few.

### [2] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-08-08 23:39 UTC · ▲2 · LB rank 642

You got labels that are ground truths (call them Gold).  The rest of the images need you to create the labels by correctly reading the reports.

I think there have been a number of kaggle competitions over the past years with a similar situation - some GOLD and than rocks you need to pan to find the gold.  My guess - the best panners will be the winners.

And always remember - to be a winner you need to predict the Kaggle truth which is sometimes harder than the real truth :)  Medical competitions are the place I first started using the words "Kaggle Truth".  Reading these images is one heck of a hard job, I would never want to do it.   When humans make visual inspections there is always lots of variation.  We need to use the images without labels but must remember that the host tells us the labelers did NOT use the reports.
