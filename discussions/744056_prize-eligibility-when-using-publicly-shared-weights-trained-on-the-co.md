# Prize eligibility when using publicly shared weights trained on the competition data

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744056
- **Author:** **Shalini S** (@shals14, Contributor) · _topic author_ · 2026-09-28 13:16 UTC · LB rank 2959
- **Topic votes:** 0 · **Comments/replies:** 4 · **Pinned:** no
- **Last comment:** 2026-10-05 08:03 UTC by Anh-Vu Mai-Nguyen
- **Cloned:** 2026-10-09 09:46 UTC (topic id 744056, raw JSON `_raw/744056.json`)

---

Could the hosts clarify how the rules apply to a practice that's become common in
this competition?

Several public Kaggle datasets contain model weights trained on this competition's
data by other participants, and a number of public notebooks blend them.

The rules appear to permit using them:

  - 2.6.b: "The use of external data and models is acceptable unless specifically
    prohibited by the Host."
  - 3.6.b: code shared publicly on the competition forum or notebooks is "deemed to
    have licensed the shared code under an Open Source Initiative-approved license"
    for the benefit of all competitors.

What I can't resolve is how that sits with Winner's Obligations. Section 2.8 requires
a prize winner to deliver "training code, inference code, model weights" for the
winning submission. A participant whose submission includes another team's publicly
shared checkpoints has no training code for that portion. Section 3.14.a also warrants
that the submission is "your own original work."

Two questions:

1. Is a submission incorporating publicly shared weights that were trained on the
   competition data by other participants eligible for a prize?

2. If so, what satisfies the 2.8 training-code requirement for the third-party
   portion — citing the source dataset, or is the winner expected to be able to
   reproduce that training?

I'd guess the answer matters to a fair number of teams, so thanks in advance for
putting it on the record.

---

## Comments (4, threaded, oldest first)

### [1] Lutfiya

**Lutfiya** (@lutfiya, Contributor) · 2026-09-28 16:12 UTC · ▲-2 · LB rank 1567

Adding a closely related case, since it changes how a solution is built.

Knowledge distillation: a team runs publicly shared weights (trained on this competition's data by other participants, released in public Kaggle datasets and notebooks under Apache 2.0) only offline, to produce soft labels on the training set. It then trains its own smaller model on those soft labels. The submitted notebook contains only the team's own model.

1. Is that allowed for prize eligibility?
2. For Winner's Obligations (2.8), is it enough to deliver our distillation training code, the teacher inference code with links to the public teacher weights, and our own weights? Or would the teachers' original training code also be required?
3. Related and still open in https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/739905 : which winner licence applies to such weights, CC BY-NC 4.0 or an OSI-approved licence that does not limit commercial use?

Thank you.

### [2] Rubens Sosa

**Rubens Sosa** (@rubenssosa, Contributor) · 2026-10-01 13:54 UTC · LB rank 4475

@javaduke95 Could you clarify whether a prize-eligible submission may use publicly released Kaggle weights that another participant trained on this competition's data (e.g., by ensembling them or distilling them into our own model)? If so, does posting a link to that public dataset satisfy the weight submission requirement in Section 2.8 for that component?

### [3] Lutfiya

**Lutfiya** (@lutfiya, Contributor) · 2026-10-02 23:23 UTC · ▲-1 · LB rank 1567

Correction to my comment above: it described distillation into a smaller model of our own. Our submissions do not do that, so here is the accurate case.

Our model is a fine-tuned public checkpoint: the Repair-v1 e12 CoAtNet weights in mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2 (CC0-1.0; my earlier comment said Apache 2.0, the dataset page states CC0-1.0), which another participant trained on this competition's training data. We fine-tune those weights on the training studies, with soft labels from a public notebook run offline on the training set plus labels extracted from the reports. The submitted notebook runs only our fine-tuned weights, through that dataset's public inference code, adapted to our reduced input.

For Winner's Obligations (2.8): is it enough to release our fine-tuning code, our weights, and links to the public base weights and inference code? Or is the training code of the base weights also required? We do not have it.

The licence question for such weights stays as asked in /discussion/739905. Thank you.

### [4] Anh-Vu Mai-Nguyen

**Anh-Vu Mai-Nguyen** (@mainguyenanhvu, Contributor) · 2026-10-05 08:03 UTC · LB rank 171

+1, a host answer here would decide which final submissions teams can safely select before the deadline. One point not yet covered above: for a submission that ensembles publicly released checkpoints trained on the competition data by other participants (used unmodified, linked as public datasets), would providing our own training code, the full inference notebook and links to those public datasets satisfy Section 2.8 for the third-party components? And is the answer the same for the Leaderboard prizes and the Efficiency prizes? A yes/no would be very helpful. Thank you.
