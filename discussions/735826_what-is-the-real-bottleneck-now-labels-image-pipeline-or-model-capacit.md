# What is the real bottleneck now: labels, image pipeline, or model capacity?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735826
- **Author:** **Oscar Yáñez Feijóo** (@oscaryezfeijo, Contributor) · _topic author_ · 2026-08-18 09:38 UTC · ▲13 · LB rank 2335
- **Topic votes:** 13 · **Comments/replies:** 11 · **Pinned:** no
- **Last comment:** 2026-08-29 12:15 UTC
- **Cloned:** 2026-10-01 19:24 UTC (topic id 735826, raw JSON `_raw/735826.json`)

---

I’ve been following the recent discussions and there seems to be an interesting pattern:

- Single DINO-based models can already reach around **0.91–0.92+**.
- Increasing encoder size does not necessarily improve CV.
- Several people report meaningful gains from **crop geometry, resolution and slice selection**.
- At the same time, most of the training labels are derived from reports, and label quality may itself impose a ceiling.

This makes me wonder: **what do you currently think is the main bottleneck in this competition?**

A. Report-derived label quality

B. MRI preprocessing / physical crop / slice selection

C. Model architecture or backbone capacity

D. Validation strategy

E. Something else entirely

Without revealing solution-specific details, I’d be interested to hear which category has produced the largest improvements in your own experiments.

For me, the emerging lesson is that improving what the model *sees* and what we *teach it* may be more important than simply using a larger backbone.

---

## Comments (11, threaded, oldest first)

### [1] Seung Sup Lee

**Seung Sup Lee** (@seungsuplee, Contributor) · 2026-08-18 10:26 UTC · ▲1 · LB rank 2631

Interesting question. I’m new to this competition, but I’ve spent the last few days approaching it mainly as a systems/optimization problem rather than assuming that a larger model will automatically give a better result.

So far, my experiments seem to support B, with A potentially being just as important.

I first froze my CPU preprocessing pipeline as a control. It preprocesses all 4,407 studies into a fixed (4407, 6, 9, 224, 224) uint8 cache. I then kept that exact cache and label pipeline unchanged while testing the GPU side.

With a pretrained ResNet34 pipeline, my current control OOF BCE is about 0.664. I then replaced the representation/model side with pretrained DINOv2-Small, including diagnosis-specific slot attention, while keeping the same CPU data and labels.

Surprisingly, simply moving to DINOv2 did not improve the result. My best 5-fold DINO candidate produced approximately 0.674 OOF BCE (rank correlation ~0.391), actually worse than the ResNet34 control on BCE. Increasing or changing model-side complexity therefore did not solve the problem by itself.

That result changed where I’m spending my compute.

My next controlled experiment keeps the successful ResNet34 GPU pipeline fixed and changes only the image preprocessing: physical slice ordering using DICOM orientation/position, removal of stack-end regions, anatomically better slice grouping, and fixed-mm physical cropping before resize. This should let me measure the contribution of the image pipeline independently from the backbone.

After that, I want to isolate label quality in the same way. Most of my current training targets are report-derived soft labels, so I suspect there is also a point where better image representation cannot compensate for noisy supervision.

One practical lesson from these experiments has been that compute cost and model size are poor proxies for useful information. A more expensive model can simply process the same imperfect input more expensively.

At this stage my working hypothesis is:

image geometry / slice selection + label quality > backbone size.

I don’t consider that conclusion proven yet, but the controlled experiments so far are pushing me strongly in that direction.

### [2] Tom Aindow

**Tom Aindow** (@taindow, Master) · 2026-08-18 13:20 UTC · ▲3 · LB rank 81

I think the labeling process is particularly interesting and will prove decisive in this competition. It's one thing to extract labels from the report, but another to faithfully reproduce the labels derived from the image. If you look over the gold set it's not hard to find instances where the image label clearly flouts the rules of diagnosis laid out in the introduction to the competition (e.g. the report says "small effusion" and the effusion label is set to 1, even though only a "moderate or large" is the heuristic. It's not surprising, as there is an element of subjectivity in such a judgement, and no doubt it's similar for many of the heuristics used: close to the decision boundary, there will often be disagreement between the report label and the image label; radiologists will just disagree - it's the nature of the beast.

One immediate thing this suggests: asking an LLM for it's confidence in a diagnosis might be misleading if what you're getting is an estimate of it's confidence in applying heuristics to reports. It may be very confident that effusion is not present after reading "small effusion", since the rule is easily applied and the words are clear; whereas in reality the process that generated the word "small" has a lot of noise and may just have easily produced the word "moderate". I will be very interested to read how everyone generated their labels at the end of this competition.

#### ↳ [2.1] Neelkant Newra

**Neelkant Newra** (@newra008, Contributor) · 2026-08-19 18:10 UTC · ▲1

@taindow  I'm actually using this approach already:

- Prompt Benchmark: Created a gold standard prompt using the 58 labeled samples and prescriptions, achieving 79% accuracy.
- Dataset Auto-Labeling: Used that prompt to derive labels for the remaining unlabeled data. It was quite compute-heavy, taking about 14 hours of GPU time , but the entire batch completed successfully without any failures.
- Now next task is on image

### [3] Chikuwabu

**Chikuwabu** (@prgckwb, Master) · 2026-08-18 13:33 UTC · ▲4 · LB rank 28

The quality of the soft labels and not being overly influenced by the public code are important.

### [4] Tucker Arrants

**Tucker Arrants** (@tuckerarrants, Master) · 2026-08-18 18:26 UTC · ▲4 · LB rank 19

Check to see how your label extractor performs relative to the 58 provided labels. Then check how your image model performs on them. If your image model is not beating the label extractor "teacher", there is likely modeling headroom. If your image model is beating the label extractor by a large margin, the quality of the labels might be holding the image model back. There's a lot of cool things to try with the label extraction, but even the simple regex extraction shown in the public notebooks can take you pretty far.

### [5] k256.dev

**k256.dev** (@k256net, Master) · 2026-08-19 09:44 UTC · ▲1 · LB rank 1946

I think the biggest bottleneck is A, although I also believe there is something in E.

We have confirmed that some of the labeled data contains cases where the label contradicts the content of the report. This suggests that even the labeled data itself may not be 100% reliable.

It is extremely difficult to know whether a given label represents the “true diagnosis” or a “false diagnosis.” The same issue may also exist in both the Public and Private Leaderboards.

In other words, some cases that are not actually correct may be treated as ground truth.

For example:
1. Filter to reports whose effusion sentence contains a small-size qualifier
   ("small", "minimal", "trace", or the equivalent in the other languages).
   You will find this group contains **both** `Effusion = 0` and `Effusion = 1`.
2. In my sample, two of these reports used the **same English template and the
   same sentence**, and got opposite labels.
3. Then look for a report stating a **moderate** effusion. At least one of those
   is labeled `Effusion = 0`, so it is not a simple
   "small = negative, moderate = positive" threshold either.

#### ↳ [5.1] (unknown)

**(unknown)** · 2026-08-19 11:08 UTC

_(no content — deleted or empty)_

#### ↳ [5.2] Enai

**Enai** (@enaimaan, Contributor) · 2026-08-24 16:25 UTC · LB rank 3040

```
We have confirmed that some of the labeled data contains cases where the label contradicts the content of the report. This suggests that even the labeled data itself may not be 100% reliable.
```

I think in this case we should prefere gold label not report content. I believe I saw it somewhere  by the Hosts.

#### ↳ [5.3] (unknown)

**(unknown)** · 2026-08-29 12:15 UTC

_(no content — deleted or empty)_

### [6] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-08-20 04:13 UTC · LB rank 636

IMO - the entire competition is the label making.

After you get some decent labels - than matching the radiologists who created the test labels - this will create the shakeup.  According to comments from the hosts those folks did not use the reports as a starting point.

I don't think we have enough reports to require a model size that's huge but I am still struggling with getting labels I like.
Since we are creating our own labels the time honored tradition of trusting you CV is pretty suspect.

### [7] (unknown)

**(unknown)** · 2026-08-29 12:15 UTC

_(no content — deleted or empty)_
