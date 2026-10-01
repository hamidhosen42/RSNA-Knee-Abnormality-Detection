# train.csv has 4,407 studies and 58 labels — the other 4,349 have reports

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734055
- **Author:** **maximo lorenzo y losada** (@maximolorenzoylosada, Contributor) · _topic author_ · 2026-08-09 17:09 UTC · ▲3 · LB rank 2604
- **Topic votes:** 3 · **Comments/replies:** 3 · **Pinned:** no
- **Last comment:** 2026-08-15 08:36 UTC by maximo lorenzo y losada
- **Cloned:** 2026-10-01 19:24 UTC (topic id 734055, raw JSON `_raw/734055.json`)

---

Counting what is actually in train.csv turned up something I had not expected, so flagging it early.

Of 4,407 studies, 58 carry labels. The other 4,349 have every label column empty.

It is all-or-nothing per study: the 58 labelled studies have all twelve findings filled, and the rest have none. There is no partial labelling.

Every one of the 4,407 studies has a Report, including all 4,349 unlabelled ones. So the diagnostic information is present for the whole training set, just as prose rather than in the label columns.

Two parsing traps in the same file that cost me a wrong answer first time round:

1. The file is 58,556 lines but only 4,407 rows. Reports contain newlines, so wc -l overstates the dataset by about 13x. Use a real CSV reader.
2. Empty is not zero. The unlabelled cells are empty strings, not 0. Filling them with 0 invents roughly 4,349 unasserted negatives per class — my first pass did exactly that and produced a nonsense prevalence of 0.5% per finding.

Prevalence within the 58 labelled studies, which is emphatically not population prevalence — it looks like a deliberately enriched annotation set:

Effusion 60.3%, Synovitis 46.6%, Medial Meniscus 44.8%, ACL 41.4%, Lateral Meniscus 39.7%, PF OA 36.2%, Contusion 32.8%, Fracture 31.0%, Medial OA 25.9%, Baker's 20.7%, Lateral OA 19.0%, MCL 15.5%

Mean 4.1 findings per labelled study, max 9 — heavily multi-label and co-occurring.

One more thing worth knowing before you write a text pipeline: the reports are not all in one language. A crude keyword heuristic over all 4,407 gives roughly 51% English-leaning, 16% Spanish-leaning, 34% ambiguous — and the very first report in the file is Spanish ("Técnica: RMN de la rodilla. Resultados: Rotura de menisco interno..."). That is a keyword count and not language ID, so treat the split as indicative, but assuming English throughout is clearly unsafe.

With 58 labelled studies across 12 correlated findings there is not much room to train or validate a twelve-class imaging model directly — a held-out split of 58 gives single-digit positives for the rarer findings.

Notebook with all of it computed in-cell:
https://www.kaggle.com/code/maximolorenzoylosada/4407-studies-and-58-labels

Deliberately just a count, no method. If I have misparsed something I would like to be told — particularly if the empty cells are meant to be read as negatives rather than unknowns, since that changes the problem completely.

---

## Comments (3, threaded, oldest first)

### [1] maximo lorenzo y losada

**maximo lorenzo y losada** (@maximolorenzoylosada, Contributor) · _topic author_ · 2026-08-09 18:11 UTC · ▲-1 · LB rank 2604

Correction to my own post above, and credit where it is due.

I said the reports were roughly 51% English-leaning, 16% Spanish-leaning and 34% ambiguous. That was a keyword heuristic testing only English and Spanish, so it was structurally incapable of identifying anything else, and the "ambiguous" third was hiding real languages rather than being genuinely undecidable.

What tipped me off was the notebook "RSNA Knee: read the report, then the knee" by prvsiyan, which mentions Greek reports and a codepoint problem with mu. I had not considered Greek at all.

So I redid it by Unicode script rather than keywords. Over all 4,407 reports, and a report can contain more than one script:

Latin 4,301 (97.6%)
Greek 321 (7.3%)
Cyrillic 220 (5.0%)

The Greek is genuine prose, not stray symbols: 321 of 321 contain three or more consecutive Greek letters. One opening, for illustration:

MRI ΑΡΙΣΤΕΡΟΥ ΓΟΝΑΤΟΣ ΤΕΧΝΙΚΗ Η εξέταση έγινε με ακολουθίες παλμών

So this is at least a multi-script corpus: Latin covering English and Spanish at minimum, plus Greek and Cyrillic. Roughly 12% of reports contain non-Latin script.

Why this matters more than a tidier statistic: a keyword or regex pipeline built on English or Spanish terms does not fail loudly on those reports. It extracts nothing and returns a confident negative for every finding. Same shape as the empty-versus-zero trap in my original post, a silent wrong answer rather than an error.

On the mu point in that notebook, worth checking whether your text uses U+03BC GREEK SMALL LETTER MU or U+00B5 MICRO SIGN. They render identically and do not match each other.

One caveat on my own numbers: this is a script check, not language identification. Latin covers English, Spanish and anything else in that alphabet, so treat 97.6% as an upper bound on what I have actually accounted for rather than a solved question.

#### ↳ [1.1] OpPrime

**OpPrime** (@farshidamira, Contributor) · 2026-08-14 14:37 UTC · LB rank 151

Hi, I spent a bit of time on this topic also till I found this post: [link to LLM work to label studies](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932)

##### ↳ ↳ [1.1.1] maximo lorenzo y losada

**maximo lorenzo y losada** (@maximolorenzoylosada, Contributor) · _topic author_ · 2026-08-15 08:36 UTC · LB rank 2604

Thanks for the pointer — that post is the most useful thing in this forum, and it let me test something my post above implied and never checked.

I warned that a text pipeline built on Latin-script terms fails silently on the Greek and Cyrillic reports: it extracts nothing and returns a confident negative rather than an error. stevenleehans' labeller has an explicit "the report does not address this" answer, published as 0.5, which makes that worry measurable for the first time. If it were quietly failing on the non-Latin reports, its refusals should pile up there.

They don't. It goes the other way, and I think that is worth having on the record.

Setup: joined llm_labels_full.csv to train.csv, 4,407 of 4,407 studies matched. Both source numbers reproduce first — 13,438 of 52,884 cells sit at exactly 0.5, which is the 25.4% reported, and the script counts come back at Greek 321 and Cyrillic 220, for 541 studies containing non-Latin script (12.3%).

Refusal rate on Latin-only reports: 25.9% of cells. On reports containing Greek or Cyrillic: 21.9%. Every study carries all twelve finding cells, so the finding mix is identical across the two groups by construction and the rates compare directly.

That is a ratio of 0.845 and a gap of 4.0 points. Shuffling the script label across the 4,407 studies 2,000 times gives a null sd of 0.96 points, and no shuffle reached the observed gap, so p < 0.0005.

Split out: Cyrillic 17.5%, Greek 24.9%.

It is not a length artefact, and length runs against the result rather than for it. The non-Latin reports are slightly shorter — median 932 characters against 988 — and a shorter report gives a labeller less to work with, not more.

Here is the part I would not want taken the wrong way. A lower refusal rate is not evidence of a correct read. A labeller that reads Greek fluently and one that hallucinates fluently produce the identical signature here: fewer refusals. Refusal rate measures willingness to answer, not accuracy.

And the accuracy question is not currently answerable. Of the 58 gold studies, 6 contain non-Latin script. That is the entire ruler available for it.

So the claim I would actually make: the specific failure mode I flagged — a pipeline going quiet on non-Latin reports — does not appear in this label set, and to that extent my post above was warning about a hazard this approach avoids. Whether the answers it gives on those 541 studies are right is a separate question, and 6 gold studies cannot settle it.

One thing in the per-finding split may matter more than the headline. Ten of the twelve findings are refused less often on non-Latin reports, but two move the other way: Synovitis +8.5 points (82.7% to 91.1%) and Baker's +7.8 (47.3% to 55.1%). Those are already the two most-refused columns, and section 4 of that post turns on how differently they behave when the report is silent. If the silence rate itself shifts by script, the silence ratio that section leans on is being measured across a mixed population — worth a look, since that section is the part of the post I would least want resting on an artefact.

To reproduce: classify a report as containing a script if it has three or more consecutive letters of it (the same rule as my correction above), then take the fraction of the twelve label cells equal to exactly 0.5 within each group. It is about ten lines.

A correction to my own numbers while I am here: my Latin count comes out 4,266 here against the 4,301 I posted on 08-09, a 35-study difference down to the exact codepoint range used for "Latin". Greek and Cyrillic are unchanged, and nothing above depends on the Latin count.
