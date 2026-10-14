# External data behind an institutional data use agreement - does the KneeCoT ruling generalize?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737950
- **Author:** **Ercan Gurvit** (@ercangurvit, Contributor) · _topic author_ · 2026-08-28 10:15 UTC · ▲5 · LB rank 3542
- **Topic votes:** 5 · **Comments/replies:** 10 · **Pinned:** no
- **Last comment:** 2026-10-09 14:34 UTC
- **Cloned:** 2026-10-14 18:23 UTC (topic id 737950, raw JSON `_raw/737950.json`)

---

Thanks to the host for the clear ruling on KneeCoT (topic 734109). The stated reasoning was that a dataset requiring a formal data agreement with the providing institution can make the playing field uneven.

Could the host confirm whether that interpretation generalizes to other knee MRI datasets that are free of charge but gated behind an institutional agreement - for example OAI (NIH/NIMH Data Archive), which requires a signed Data Use Certification from an institution holding an active Federal Wide Assurance, or MRNet (Stanford), which requires a Research Use Agreement.

A related question, since it bears on the same decision: winners' obligations require delivering model weights as a public Kaggle dataset, and sharing the final model publicly for open distribution and validation. If a pretrained model were derived from data under such an agreement, would publishing those weights satisfy that obligation - or does that requirement independently rule out pretraining on agreement-gated data, regardless of the license?

Asking before investing compute, so the answer is on record for everyone.

---

## Comments (10, threaded, oldest first)

### [1] Doublej37

**Doublej37** (@doublej37, Expert) · 2026-09-01 16:59 UTC · LB rank 3582

A related edge case I would appreciate clarification on:

Suppose a pretrained model checkpoint is publicly downloadable, free of charge, and equally accessible to every competition participant, but the authors originally pretrained it partly on a private clinical cohort that competition participants themselves cannot access.

Participants would use only the public pretrained checkpoint, not the underlying private data.

Would that public checkpoint be considered an acceptable external model under the competition’s “reasonably accessible to all” rule?

I am specifically trying to distinguish this case from directly using an external dataset that requires a DUA or institutional access. The resource actually used by competitors would be the freely available pretrained model, while some of its historical pretraining data would not itself be reproducible or accessible.

A host ruling would be helpful before deciding whether such a model can be included in a final solution.

#### ↳ [1.1] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-09-01 17:52 UTC · LB rank 2508

Lets take your thought a tiny bit further.  I don't have access to any of the data used to train ChatGPT, Gemini Pro, or Claude.
For that matter Dino V2 training data is hidden from me.

What do you think the host would say about the use of these models?

Think the answer given in another post - its our job to understand the restrictions of any data or model used.

#### ↳ [1.2] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-09-02 01:28 UTC · LB rank 2508

I myself not worried about this particular rule.  It only comes into play if you are a winner.  

I see no way in heck I am going to make it to the top 10 - this (IMO) is a really hard set of data and matching what the radiologists reported based on only 58 gold images is a shake up in the making.

I got  7 day ban from discussion a few months ago for suggesting to someone pretty new that the probability of winning was approaching 0.0.

#### ↳ [1.3] junicus

**junicus** (@junicus, Expert) · 2026-09-03 05:56 UTC · LB rank 2386

@javaduke95 Could you please provide an official host ruling on comment #3519419? Specifically, is a freely downloadable checkpoint trained on inaccessible private clinical data allowed, and can it satisfy winner-delivery requirements when its repository only declares Apache-2.0 in the README but provides no LICENSE file or checkpoint-specific terms?

We need clarification before using MRI-CORE

### [2] Tucker Arrants

**Tucker Arrants** (@tuckerarrants, Master) · 2026-09-01 18:15 UTC · ▲2 · LB rank 4

NC licenses are allowed from the host's perspective,[ as mentioned here](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965#3517696). ND licenses are a problem. Most external datasets that are relevant here are ND, and training a model on images, then releasing those weights as required by Kaggle winning solutions, would be a derivative and hence prohibited.

MRNet, KneeMRI, etc. all carry a ND license and I would be surpised if they are allowed for model training.

#### ↳ [2.1] Cody_Null

**Cody_Null** (@cody11null, Grandmaster) · 2026-09-01 20:04 UTC · ▲3 · LB rank 100

Yeah I haven’t tried any external data for this reason, would be curious if top teams are using it

### [3] SpeedSci

**SpeedSci** (@caoyupeng, Expert) · 2026-09-03 12:25 UTC · LB rank 1027

Not sure if external data is permitted—but if I do training or label cleaning locally, it's unlikely anyone would be able to tell. Definitely something to think about.

#### ↳ [3.1] Steve421471

**Steve421471** (@steve421471, Contributor) · 2026-09-05 16:03 UTC · ▲1 · LB rank 172

Sure, but the competition rules state that your published results have to be replicable. This isn't possible if you don't disclose all of your training data. I suppose it doesn't matter if you're not trying to win, but in that case, why bother?

### [4] Fernando Faria

**Fernando Faria** (@fernandoadefaria, Contributor) · 2026-09-06 03:34 UTC · LB rank 245

Following up on the KneeCoT ruling above, and on your note in "Use of Commercially Hosted LLMs" distinguishing "straightforward registration or click-through" access from "institution-specific approvals, negotiated written agreements, IRB approval, lengthy credentialing" -- could you confirm which side of that line these fall on:

- MRNet (Stanford AIMI): a Research Use Agreement accepted via click-through on the AIMI portal
- fastMRI (NYU, raw k-space/DICOM release): requires emailing NYU and waiting on an approved data-sharing agreement
- OAI (NIH-funded, accessed via a federal data archive): requires a signed Data Use Certification
- SKM-TEA (Stanford AIMI): same access pattern as MRNet

A ruling per access-pattern (click-through vs. emailed/signed agreement) rather than per dataset would help everyone here, since most of the open questions on this board reduce to the same two patterns. Thanks for the KneeCoT ruling above -- trying to apply it correctly before committing pretraining compute.

### [5] (unknown)

**(unknown)** · 2026-10-09 14:34 UTC

_(no content — deleted or empty)_
