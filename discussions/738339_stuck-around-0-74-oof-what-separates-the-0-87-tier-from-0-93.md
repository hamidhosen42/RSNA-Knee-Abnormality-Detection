# Stuck around 0.74 OOF - what separates the 0.87 tier from 0.93+?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738339
- **Author:** **Abdullah Wasee** (@abdullahwasee, Contributor) · _topic author_ · 2026-08-31 10:57 UTC · ▲1 · LB rank 2417
- **Topic votes:** 1 · **Comments/replies:** 3 · **Pinned:** no
- **Last comment:** 2026-09-09 08:44 UTC by Nathan Jacob
- **Cloned:** 2026-10-09 09:46 UTC (topic id 738339, raw JSON `_raw/738339.json`)

---

Hi - first Kaggle competition, and I've hit a wall I can't reason my way past. Sharing what I've measured in case the gap is obvious to someone who's been here longer.

Where I am. 0.739 macro AUC on the 58 annotated studies, 5-fold. Report extraction at 0.725 against the same 58.

Pipeline. Rule-based multilingual extraction (clause-scoped negation in both directions, severity grading), then a 2.5D ResNet-34 with gated attention pooling over 4 sequence slots × 3 adjacent-slice triplets at 256px, 140 mm physical crop.

Things that measurably helped me, in case they're useful to other beginners:

Sorting slices by ImagePositionPatient projected onto the slice normal rather than filename. Filenames are SOP Instance UIDs and are effectively random - this was my single biggest bug.
Grading mild findings lower. I found reports saying "geringer Gelenkerguss" (slight effusion) annotated as negative, so scoring every mention at 0.85 was systematically wrong. Effusion went 0.628 → 0.726, ACL 0.775 → 0.855.
Sharpening the soft labels: BCE against targets clustered near 0.5 barely trains. GOLD went 0.653 → 0.725.

Where I'm stuck. Medial Meniscus 0.59 and MCL 0.60 - the thin structures. Everything large and diffuse is fine (Medial OA 0.89, Effusion 0.88).

What I'd love guidance on:

Reading public notebooks, the consensus seems to be 336px with 3 slices rather than more slices at lower resolution. Is that mostly resolution, or mostly that adjacent triplets beat spread-out slices? I've seen both claimed.
How much of the ResNet → DINOv2 gap is real versus fold noise? One post measured DINOv2-S → B as a null (+0.0011), which makes me unsure whether the backbone matters at all past a point.
Public solutions seem to plateau near 0.87 while the top is 0.952. Is that gap ensembling and scale, or is there a class of technique the published notebooks aren't using?

Happy to share my full ablation log if useful. Thanks for reading - the discussions here have already saved me several wasted weeks.

---

## Comments (3, threaded, oldest first)

### [1] Rasoul Mojtahedzadeh

**Rasoul Mojtahedzadeh** (@rasoulmojtahedzadeh, Expert) · 2026-08-31 11:17 UTC · ▲2 · LB rank 791

Use Steven's knee report dataset to see if it improves your LB score:

https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels

#### ↳ [1.1] Abdullah Wasee

**Abdullah Wasee** (@abdullahwasee, Contributor) · _topic author_ · 2026-08-31 17:47 UTC · ▲1 · LB rank 2417

Thanks, that made a bigger difference than I expected.

### [2] Nathan Jacob

**Nathan Jacob** (@nathanjacob, Expert) · 2026-09-09 08:44 UTC · LB rank 464

Your pipeline is actually solid - ResNet-34, gated attention pooling, 4 sequence slots, clause-scoped negation in extraction. The
fact that you caught the DICOM sorting bug (SOP Instance UID vs ImagePositionPatient) tells me you're debugging at the right level.
                                                                                                                                   
I think the gap from 0.74 → 0.87+ is mostly a label quality ceiling, not an architecture problem. Here's why:                    
  
**The label extraction bottleneck**                                                                                              
                                                                                                                                   
With only 58 gold-labeled studies out of 4,407, you're almost entirely dependent on your rule-based extraction for training signal. The thread on best single-model scores (735304) has a revealing pattern: every jump from 0.87 → 0.93+ came from better label extraction (mostly LLM-based), not from architecture changes. One competitor got 0.949 single-model just by switching to Gemma 4 for local label extraction.                                                                                              

The consensus emerging is that a "single prompt LLM extraction has not been beaten by more sophisticated extraction procedures" -and the image model ("student") consistently outperforms the label extractor ("teacher"), which means better labels → the model learns what the labels can't explicitly capture.                                                                                 
                                                                                                                                   
**Resolution might also be a factor**
                                      
Your 256px at 140mm crop gives you 0.547 mm/px effective resolution. Some of the 0.93+ models are running at 336-384px (0.383-0.416 mm/px at the same physical crop). For subtle findings like early OA or small meniscal tears, that resolution difference could matter — especially in the coronal plane where compartmental OA grading needs to resolve joint space narrowing. 
                                                                                                                                   
**What I'd try in your position:**                                                                                               
                                      
1. Before touching architecture: swap your rule-based extraction for Steven's public labels dataset or run a local LLM extraction pass. This is probably the single biggest lever.                                                                                
2. If you want to stay rule-based: make sure your extraction handles severity grading for effusion/synovitis (these are the hardest labels - reports often say "mild effusion" without a clear positive/negative boundary) and multilingual reports (9-12 languages in this dataset).
3. Try bumping to 336px at the same 140mm physical crop — relatively cheap experiment that might add a point or two.             
                                                                                                                                   
The good news: your pipeline fundamentals (attention pooling, physical crop, proper slice ordering) are in the right ballpark.   
The bad news: below 0.87, it's almost certainly a data problem, not a model problem.
