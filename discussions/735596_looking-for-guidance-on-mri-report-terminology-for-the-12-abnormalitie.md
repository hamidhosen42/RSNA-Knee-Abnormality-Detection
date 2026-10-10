# Looking for guidance on MRI report terminology for the 12 abnormalities

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735596
- **Author:** **Kaustubh Ratna** (@kaustubhratna, Contributor) · _topic author_ · 2026-08-17 05:08 UTC · ▲1 · LB rank 3642
- **Topic votes:** 1 · **Comments/replies:** 6 · **Pinned:** no
- **Last comment:** 2026-08-25 09:53 UTC by Enai
- **Cloned:** 2026-10-09 09:46 UTC (topic id 735596, raw JSON `_raw/735596.json`)

---

Hi everyone,

I'm trying to better understand the terminology used in the MRI reports for the 12 target abnormalities in this competition.

For those with radiology/MRI experience, could you share some common keywords or phrases that are useful for identifying each of these?

ACL
MCL
Medial/Lateral Meniscus
Medial/Lateral/PF OA
Effusion
Synovitis
Baker's cyst
Contusion
Fracture

I'm particularly interested in:

Common synonyms/alternative wording
Phrases that strongly indicate a finding
Phrases that indicate the finding is absent
Wording that can be misleading or ambiguous
Important distinctions between similar findings (e.g. degeneration vs tear)

Even a few examples or general guidance would be really helpful.

Thanks!

---

## Comments (6, threaded, oldest first)

### [1] Jose Luis Sabogal

**Jose Luis Sabogal** (@joseluissabogal, Contributor) · 2026-08-19 13:49 UTC · ▲2 · LB rank 2894

Visit this site https://www.radiologymasterclass.co.uk/tutorials/mri/mri_system surely it will be useful

### [2] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-08-20 04:20 UTC · ▲1 · LB rank 643

Recommendation (its still on my to do list).

Translate all reports to English.  (got that done - it takes a long time :) 
Generate an EDA of the words.

My label maker is building columns for me of the evidence used for each target.  I am hoping I can see some of the phrasing differences, etc.  in the eventual EDA.

#### ↳ [2.1] Jose Luis Sabogal

**Jose Luis Sabogal** (@joseluissabogal, Contributor) · 2026-08-20 16:24 UTC · ▲1 · LB rank 2894

Be careful because some of the reports in different languages  do not exactly match the classification criteria of radiology associations for the different types of knee abnormalities

##### ↳ ↳ [2.1.1] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-08-20 18:40 UTC · ▲1 · LB rank 643

For sure an issue I have seen in every Medical competition over the 9 years I been on kaggle is what I think of as measurement error.  Most of them I would have always said, the winner is predicting the kaggle truth and not necessarily the real world truth.

The business I spent 45 years in used mostly humans to make classification judgments.  I spent more hours of my life than I care to think about as we attempted to improve those judgments.  It was mission impossible - the only real fix was to replace or augment those decisions with machines/vision systems, etc.

To expect that 4700 different radiology folks from all across the world to be in agreement with the limited folks who classified our 58 train and the full test would never cross my mind for sure.  My assumption and hopefully yours is that both random and systemic variation to the reports exist.

The power of what we are doing - helping make those judgments consistent is really a great thing.  Its one of the reasons that you will find me every day hollering at the folks on Facebook who want to not allow data centers to exist in their town.

##### ↳ ↳ [2.1.2] Jose Luis Sabogal

**Jose Luis Sabogal** (@joseluissabogal, Contributor) · 2026-08-20 18:58 UTC · ▲1 · LB rank 2894

I completely agree with you. This is one of the ways to continue improving the quality of care and, given the competition we face, to seek greater accuracy in medical diagnoses related to knee injuries.

##### ↳ ↳ [2.1.3] Enai

**Enai** (@enaimaan, Contributor) · 2026-08-25 09:53 UTC · LB rank 3044

Thank you for sharing this rich experience!
But I am even wondering if the folks who classified the 58 gold studies are the same folks who annotated the private/public test studies. Do we have any proof from the Hosts?
