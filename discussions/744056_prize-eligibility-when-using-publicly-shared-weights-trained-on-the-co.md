# Prize eligibility when using publicly shared weights trained on the competition data

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744056
- **Author:** **Shalini S** (@shals14, Contributor) · _topic author_ · 2026-09-28 13:16 UTC · LB rank 2180
- **Topic votes:** 0 · **Comments/replies:** 2 · **Pinned:** no
- **Last comment:** 2026-10-01 13:54 UTC by Rubens Sosa
- **Cloned:** 2026-10-01 19:24 UTC (topic id 744056, raw JSON `_raw/744056.json`)

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

## Comments (2, threaded, oldest first)

### [1] Lutfiya

**Lutfiya** (@lutfiya, Contributor) · 2026-09-28 16:12 UTC · ▲-1 · LB rank 2271

Adding a closely related case, since it changes how a solution is built.

Knowledge distillation: a team runs publicly shared weights (trained on this competition's data by other participants, released in public Kaggle datasets and notebooks under Apache 2.0) only offline, to produce soft labels on the training set. It then trains its own smaller model on those soft labels. The submitted notebook contains only the team's own model.

1. Is that allowed for prize eligibility?
2. For Winner's Obligations (2.8), is it enough to deliver our distillation training code, the teacher inference code with links to the public teacher weights, and our own weights? Or would the teachers' original training code also be required?
3. Related and still open in https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/739905 : which winner licence applies to such weights, CC BY-NC 4.0 or an OSI-approved licence that does not limit commercial use?

Thank you.

### [2] Rubens Sosa

**Rubens Sosa** (@rubenssosa, Contributor) · 2026-10-01 13:54 UTC · LB rank 4722

@javaduke95 Could you clarify whether a prize-eligible submission may use publicly released Kaggle weights that another participant trained on this competition's data (e.g., by ensembling them or distilling them into our own model)? If so, does posting a link to that public dataset satisfy the weight submission requirement in Section 2.8 for that component?
