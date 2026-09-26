# 58 labelled studies out of 4,407 — the rest of the supervision is in the reports

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734106
- **Author:** **Luka Duvanov** (@nekkon, Contributor) · _topic author_ · 2026-08-09 22:26 UTC · ▲7 · LB rank 4183
- **Topic votes:** 7 · **Comments/replies:** 3 · **Pinned:** no
- **Last comment:** 2026-08-12 18:03 UTC by Luka Duvanov
- **Cloned:** 2026-09-26 16:24 UTC (topic id 734106, raw JSON `_raw/734106.json`)

---

Ground facts I wanted before starting, in case they save someone else the time:

Exactly 58 studies are labelled, and all twelve columns are filled for the same 58. There is no partial labelling to exploit.
The reports are in seven languages — English, Turkish, Spanish, German, Greek (178 of them, in Greek script), Dutch, French — plus 428 a stopword detector cannot place. Turkish negates after the term (efüzyon izlenmedi), so a left-only negation window silently inverts the second-largest language in the corpus.
Fluid_Sensitive and Fat_Suppression are the same column, identical on all 24,371 series. Encode one, drop the other.
Every study has all three planes and both contrast types — no fallback path needed for a fixed-shape input.
A plain multilingual keyword matcher, vocabulary not tuned on the 58, gets 0.86 on Baker's cyst and 0.66 on effusion. That is the floor to beat.
The 58 are not a random sample: ACL prevalence is 41% there against ~20% in the corpus. Do not set priors from them.
https://www.kaggle.com/code/nekkon/knee-mri-58-labels-for-4-407-studies

---

## Comments (3, threaded, oldest first)

### [1] Navneet

**Navneet** (@navneetbende, Contributor) · 2026-08-10 00:57 UTC

Thank you for the supervision @nekkon

### [2] Malav D Modi

**Malav D Modi** (@malavdmodi, Contributor) · 2026-08-12 04:07 UTC · ▲1

I need to ask that my approach i thought was to convert the report to identify labels for rest of report but it cannot happen due to large computation hours of report conversion into english and then into labels. What else approach i can use to convert my report into labels given in the csv file ?

#### ↳ [2.1] Luka Duvanov

**Luka Duvanov** (@nekkon, Contributor) · _topic author_ · 2026-08-12 18:03 UTC · ▲2 · LB rank 4183

You don't need to translate the reports. Translate the vocabulary.

The cost you're hitting is O(corpus): 4,407 reports through a seq2seq model.
What you need is O(vocabulary): roughly 40-60 terms per finding times 7
languages. That's a few hundred strings, translated once, and after that it's
plain string matching over the corpus — seconds on CPU, no GPU hours.

Radiology reports are formulaic, which is what makes this work. "Effusion"
shows up as effusion / efüzyon / derrame / Erguss / συλλογή / effusie /
épanchement plus a few morphological variants each. Build that table per
finding, then match.

Two things that will cost you more than the model choice if you skip them:

1. Negation direction is language-specific. Turkish negates after the term —
   "efüzyon izlenmedi" means NO effusion. A left-only negation window (the
   default in every English NegEx port) silently inverts the second-largest
   language in this corpus. Right-side window for Turkish, left for the
   Germanic and Romance ones.

2. Detect the language first, then apply that language's lexicon and negation
   rule. Stopword-frequency ID is enough and costs nothing. About 428 reports
   won't resolve — those are worth looking at rather than guessing at.

For calibration: a matcher built this way, with the vocabulary written without
looking at the 58, gets 0.86 on Baker's cyst and 0.66 on effusion. That's the
floor, and it runs in minutes.

If you do want a model, skip translation and go multilingual directly —
sentence embeddings from LaBSE or multilingual-e5 are one forward pass per
sentence and stay comparable across all seven languages. But don't fit
thresholds on the 58: ACL prevalence there is 41% against ~20% in the corpus,
so anything calibrated on them will be biased
