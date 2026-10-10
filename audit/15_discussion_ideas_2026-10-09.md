# 15 — Forum review and new experiment ideas (2026-10-09)

**Scope and method.** Read-only. Nothing was run, pushed or submitted on Kaggle.

- **Every topic created or commented on since 2026-10-01 was read in full.** That is 50 topics, listed in the appendix.
- **Older high-signal topics were also read:** 735304, 735826, 737597, 735154, 734105, 733517, 743416, 741819, 737696, 743148, 742050, 742926, 742327, 741310, 737566, 736635, 740375, 735639, 734681, 738096, 733313, 744230, 735854, 743779, 740094, 738708, 740474, 737454, 734117, 738339, 733826 and 733491. All host comments across the whole forum were extracted from `_raw/*.json`.
- **Ten key topics were fetched again** through the discussions API into the scratchpad (not into the archive). No newer comments were served.
- **A few comments are counted in the listing but not served by the API.** In 743416 the listing counts 30 comments but 27 are served; the missing ones include Satwik on 10-09 08:06. In 735304 the last comment (hikaggler, 10-09 09:09) is not served. In 747660 the listing names a last commenter (Yan Zhou, 09:23) but 0 comments are served. Their content is **unavailable**.
- **Notation.** Citations are `topic id, author (LB rank as shown in the archive), date`. **HOST** marks an official host or admin statement.
- **Evidence labels:**
  - **verified** = we reproduced it, or it is visible in code or a scored notebook;
  - **LB-probe** = the author's own public-LB submissions (unaudited);
  - **CV/gold claim** = the author's offline numbers;
  - **claim** = no numbers or code.

---

## 0. Executive summary

1. **No new host ruling of any kind since 2026-09-21.**
   - The last host comment anywhere in the forum is the OAI answer in 741819.
   - The direct questions about OAI-trained public weights are still unanswered: 747455 (10-08), 747660 (10-09), 743416 (10-08, which asks for a per-dataset yes/no).
   - What did change is the **evidence from the data stewards**. The NDA help desk says OAI is "open access for everyone", has no country restriction, needs no institutional e-mail, and that "approval is granted upon request" (743416, Tucker Arrants, 10-08).
   - YYama (rank 10, who asked the question) now reads OAI as "reasonable to consider … permissible".
   - On the other side, a participant in China could not register at all (746792, xun1993, 10-08).
   - Net assessment: the OAI rule risk drops from "HIGH" (audit/14) to **"unresolved, medium"**. It is still not safe for a final selection without a host answer.
2. **The 0.950–0.955 tier is made of own-trained models with image-informed labels, and nobody there uses the public stack.**
   - ConvNeXt-T 2.5D single model **0.953**, DINOv2-S 0.951, both together 0.955. Both use pseudo-labels and no external data (735304, colum2131, rank 63, 10-09).
   - 5-fold ResNet50 at 224 px: **0.954** (CoolinLai, rank 21).
   - A single model at 0.955 (Tanuki_boosting, rank 43).
   - Single-fold CoAtNet at 224 px reaching 0.950 by mixing out-of-fold predictions into the labels (Archit Konde, rank 80).
   - CoAtNet at 288 px reaching 0.949 with multi-teacher pseudo-labels at a 50/50 mix (Raymond Yuen, rank 38–43).
   - Tucker Arrants (rank 17–19): "most of my models are in the low 0.95 range", using a plain Qwen extraction (745214, 10-02).
   - Nobody at 0.96 or above has disclosed anything. YYama (rank 10) declined.
3. **The same architecture family as our reader scores 0.953 at colum2131 and 0.929 in our hands** (the goodpjw2008 reader alone, 3 folds).
   - The disclosed difference is the labels: pseudo-labels, plus an LLM that read **report + MRI images**.
   - colum2131 gives label AUC on gold-58: Effusion 0.835 → about 0.92, Synovitis 0.80 → about 0.85. The image model trained on them reaches Effusion 0.986 and Synovitis 0.859 (745949, 10-05).
   - Our reader's gold Synovitis is 0.68–0.70, *below* its own label tables (0.75–0.79). That is a gap the model could still close.
4. **Text-only label extraction is saturated.**
   - Eight public LLM label sets cannot be told apart on gold-58 for 11 of 12 findings (747540, 10-08).
   - Frontier LLMs all land between 86.4 % and 89.2 % on report-to-gold extraction (747658, Kirderf, 10-09).
   - Further text-label work is low value. Image-informed labels (pseudo-labels or a vision LLM) are where the remaining label signal is.
5. **More of the same family does not move our LB.** E16 (3 → 5 reader folds), E18 (7 checkpoints) and E19 (a fifth Raptor view) were all neutral at 0.944/0.945. Small reader-OOF gains will be invisible in the blend. Only a **much stronger or much more decorrelated** component can move it.
6. **Cheap inference levers with forum evidence** (useful mainly to build a better *teacher*):
   - more eval windows (dreaddevelopment: 42 → 62 → 78 windows gave 0.924 → 0.927 → 0.932; nartaa: 94 → 24 windows lost 0.006);
   - an anatomical left/right mirror test-time augmentation (TTA) (nartaa: +0.001);
   - a series-alternative TTA that is specific to our reader (new; see R4).

   All three can be validated on our fold 3/4 out-of-fold (OOF) predictions with **CPU only**.
7. **Retraining noise on the LB is about 0.003–0.004.** nartaa's fresh retrains of the same recipe lost 0.948 → 0.944 and 0.949 → 0.946 (746792). Any pre-registered LB margin should be at least about 0.003, ideally with two seeds.
8. **Is 0.964 realistic?** Not for us in 13 days on about 20 GPU-hours a week. My estimate is under 5 %, and the reasons are in §5. A realistic rule-clean target is **0.948–0.952**. If OAI is allowed, that becomes **0.950–0.953**.
9. **Top 5 ideas**, detailed in §6:
   - **(1)** a team merge before 10-15 with an own-model team (no GPU);
   - **(2)** a pseudo-label (out-of-fold soft-bootstrap) retrain of the reader as a clean A/B, about 6.5 GPU-hours;
   - **(3)** vision-LLM (report + key slices) labels for Effusion and Synovitis (API cost, no GPU for labelling);
   - **(4)** a CPU-only reader inference sweep (windows, anatomical mirror, series TTA) plus complete 5-fold OOF;
   - **(5)** runtime hygiene (E20 fp16 plus guards), which every addition to the 0.945 stack depends on.

---

## 1. Host clarifications (a)

### 1.1 All substantive host statements (complete list from `_raw`)

| Topic | Date | Who | Statement (paraphrased; quotes are verbatim) | Implication for us |
|---|---|---|---|---|
| 741819 | 09-21 | **HOST** Po-Hao Chen | OAI "does not violate the … rules if it is generally accessible to all participants without an escalated (i.e. institutional) review and sign-off or access to a legal team, and is free of charge." The host does not control OAI's own terms. | The deciding test is about **accessibility**, not the licence. |
| 734109 | 08-25 | **HOST** | **KneeCoT is not allowed** "due to the required data agreement with dataset institution". | A dataset that is easy to download but has an institutional agreement attached is banned. |
| 733965 | 08-27 | **HOST** | LLM APIs may read the reports to make labels. LLM-derived labels and embeddings are OK. Non-commercial licences are not excluded by themselves. Straightforward registration or a click-through data-use agreement is "generally" fine. "Institution-specific approvals, negotiated written agreements, IRB approval, lengthy credentialing, or other substantial administrative steps" may make a dataset not allowed. | Report-to-LLM labelling is explicitly fine. **Sending images** to an LLM API is not addressed explicitly (see R3). |
| 733826 | 08-08 | **HOST** | Gold labels were assigned from **images**, independently of the reports. When they disagree, the **image label is authoritative**. A negative means "annotated as absent". Bilateral studies were reviewed one by one, and the report text or DICOM metadata was adjusted so they can be told apart. | Report labels have a hard ceiling. Pseudo-labels from images are the way past it. |
| 733491 | 08-07 | **HOST** | The same image-review process was used for the test set. Report wording (marrow edema, cartilage terms) does not by itself decide the contusion or OA labels. | Same as above. |
| 733343 | 08-06 (post) | **HOST** | Label rubric: "on the fence" → negative. Effusion is positive only if moderate or large. OA means ≥1 cm of loss deeper than 50 %. Synovitis means inflammation or thickening of the synovial lining. | Use these definitions in any vision-LLM prompt (R3). |
| 733592, 734118 | 08-07/10 | **HOST** | There are no reports for the test set. | — |
| 733475 | 08-07 | **ADMIN** Ryan Holbrook | GPU notebooks are allowed. RuntimeSeconds covers the whole notebook run. | — |
| 741916, 737163 | 08-26, 09-18 | **ADMIN/HOST** | Corrupted DICOMs are acknowledged. A fix was attempted, and the dataset update "may have failed". | Keep the per-file guards. |

**Still open, with no host answer as of 2026-10-09 09:47 UTC:**
- 743416: per-dataset yes/no for MRNet, SKM-TEA, fastMRI, OAI, KneeMRI and KneeXNet.
- 747455 and 747660: are OAI-trained public weights allowed?
- 744056: prize eligibility when using public weights that someone else trained on the competition data.
- 739905: which winner licence applies, CC BY-NC 4.0 or OSI?
- 745283 and 735121: RadImageNet, which is CC-BY-NC-SA on Kaggle but MIT on Hugging Face.
- 733313: DINOv3 (Meta gated access; some applicants were refused).
- 744519: does every hidden test study have all planes?
- 733343 #13: how bilateral studies are labelled.
- 740094: what happens to single-class columns in the private set.

### 1.2 OAI and the nartaa 0.949 weights: the evidence as of 10-09

| For "allowed" | Against / still unclear |
|---|---|
| The NDA help desk e-mail (743416, Tucker Arrants, 10-08): "OAI is open access for everyone"; "the country of concern does not apply to the OAI permission group"; "approval is granted upon request"; "an institutional email is not currently required". | **There is no host ruling.** The KneeCoT precedent and the 733965 wording ("lengthy credentialing … substantial administrative steps") still apply. |
| The OAI/UMass e-mail (746792 [1.1.3], Satwik, dated 08-21): competition use falls under the data use agreement, and trained weights may be shared. | The same UMass e-mail advises "contact your institution's IRB to verify". That is close to the host's "institutional review" wording, although it is framed as advice. |
| Tucker Arrants, who in September called OAI a "no" (741819), registered successfully with a fresh e-mail (746792, 10-08). | xun1993 (China, rank 99) tried five e-mail addresses and was blocked at login.gov with error LG22 (746792, 10-08). So access is **not demonstrably equal for everyone**. |
| YYama (rank 10): "it seems reasonable to consider OAI permissible" (743416, 10-08). | The download is hard: NDA tools, about 450 GB, packages that need building (746792, Cody_Null / hengck23 / Quan Vu). |
| hengck23 (GM): "i think OAI changes the competition!" Several top-50 users downloaded OAI on 10-08 (minhtu rank 26, Quan Vu, Cody_Null, hengck23). | The nartaa weights are licensed **"Other"** (research/educational model use). That matters for winners' release obligations (prize only). |

**Conclusion for us.**
- Update audit/14's "HIGH" to **"unresolved, medium"**. The NDA statement undercuts the "institutional sign-off" reading, but the China access failure and the missing host answer remain.
- Build an OAI-based candidate so it is *ready*. **Select it as a final only if the host answers yes before 10-22.**
- If OAI is later disallowed, a selected submission that used it risks the whole team, not just that one submission (DeepFake precedent, 743416 [6.1.1]).
- Distilling from nartaa's model also counts as using OAI-derived information.

---

## 2. Techniques with a claimed effect (b)

### 2.1 Labels and pseudo-labels (the main lever for the 0.950+ tier)

| Technique | Source | Claimed effect | Evidence type | Relevance to us |
|---|---|---|---|---|
| Pseudo-labels from teacher models (several weak-label sources → teachers → pseudo-labels → final model). Blend 0.5 teacher + 0.5 original; **do not replace**. | 735304 [15] and 743148, Raymond Yuen (38–43), 09-15…09-25 | CoAtNet 288 full-train: 0.938 → 0.94 → **0.949**, "still label". Teachers help different targets. A model with local 0.930 scored worse than one with 0.926. | LB-probe | **High.** This is R2. |
| Mix **out-of-fold** predictions into the extracted labels, pushing the mix *heavier* toward the model ("50/50 already jumped"). In-fold predictions "just parrot the labels". Rounding to 0/1 hurt. | 735304 [17], Archit Konde (80), 09-17…09-21 | Single-fold CoAtNet 224: **0.950** LB, gold OOF 0.930. Biggest effect on the report-silent Synovitis cases. | LB-probe and gold claim | **High.** This is R2. |
| Pseudo-labelled single models: ConvNeXt-T 2.5D **0.953**, DINOv2-S 2.5D 0.951, ensemble 0.955. No external data. | 735304 [29], colum2131 (63), 10-09 | as stated | LB-probe | **High.** Same backbone as our reader (0.929). |
| An LLM that reads **report + MRI** for Effusion and Synovitis, with soft probabilities (present > equivocal > not mentioned > absent). | 745949, colum2131, 10-05 | Label AUC on gold-58: Effusion 0.835 → about 0.91–0.92, Synovitis 0.799 → about 0.84–0.86. Image model trained on those pseudo-labels: Effusion **0.986**, Synovitis **0.859**. | gold claim (58 studies) | **High.** This is R3. |
| "Show ChatGPT the report and the image and ask it to label the ROI." Active labelling: spend the expensive annotator only on uncertain cases. | 745861, hengck23 (GM), 10-04…10-06 | none | claim | Supports R3. Budget control. |
| Pseudo-labels make single models match ensembles ("two types of single models"). | 735304 [21], Chris Deotte (GM), 09-25 | none | claim | Context. |
| OOF pseudo-labels correlate well with LB. | 735304 [12], tennogh (8), 09-03 | none | claim | Supports R2. |
| **Negative.** 50/50 soft bootstrapping lifted CV and gold but **not** LB. | 735304 [17.1.8], Nicolai Karcher, 09-27 | 0 on LB | LB-probe | Risk for R2. |
| **Negative.** OOF-teacher 50/50 with the *same* ResNet as teacher and student: CV up, LB not. | 743148, Myo Min Htet, 09-25 | 0 on LB | LB-probe | Risk for R2. The suspected cause is the same encoder on both sides. |
| **Negative.** Masking or replacing noisy labels using the model's strongest predictions: no effect. | 735304 [17.1.3], Tom Aindow, 09-17 | 0 | claim | — |
| Labels moved DINOv2 0.931 → 0.942 (gold +0.01). One label version reached gold 0.945 on EffNet but only 0.910 on LB, i.e. it overfit gold. | 745214, SpeedSci (116–121), 10-02 | as stated | LB-probe | A warning against tuning on gold. |
| Single-prompt Qwen extraction. "I am doubtful that improving labels matters much here". 58 gold labels are too few to resolve anything. | 745214 and 735304 [5.1.x], Tucker Arrants (17–19), 08-31…10-02 | most models in the "low 0.95 range" | claim | Text-label work is low value. |
| 8 public LLM label sets compared on gold-58 with 95 % CIs: no clear winner on 11 of 12 findings (MCL has too few positives). Macro: consensus 0.8887, steven_v4 0.8927, steven_v2 0.8873. | 747540, karttikjangid05, 10-08 | — | verified-style (public notebook) | Supports dropping text-label work. A CC0 consensus file is available. |
| A frontier-LLM report-extraction benchmark: every model between 86.4 and 89.2 %. | 747658, Kirderf (GM), 10-09 | — | benchmark | Same conclusion. |
| Adding JEV labels to the label blend. | 746792, nartaa, 10-07 | 0.940 → **0.942** (with a blend change) | LB-probe | Low; folded into R8. |
| SOFT beats HARD targets on gold (+0.0143, CI [−0.004, +0.033]); HARD wins on the lexicon surrogate. | 734105, FHZ982, 08-09 | — | gold claim | Keep soft targets (our reader already does). |
| "Not addressed" is a label too: 25 % of cells are silent, and silence means something different for each finding. | 733932, stevenleehans, 08-09 | regex 0.8136 vs LLM 0.8780 on gold | gold claim | — |
| Synovitis: only about 16 % of reports mention it. A RadImageNet encoder reached 0.78 on Synovitis where ImageNet encoders got 0.62–0.72. Effusion and Synovitis correlate at about 0.5 in gold. | 737566, starkhushi, 08-26 | — | claim | Motivates R3 and R9. |
| Report auxiliary loss hurts. Generating report text from images was negative. | 740162 (Rasoul Mojtahedzadeh, 09-08); 743148 (Myo Min Htet, Sanjib Biswas, 09-25) | negative | claim | Do not pursue. |

### 2.2 Input geometry, windows, slice order

| Technique | Source | Claimed effect | Evidence | Relevance |
|---|---|---|---|---|
| **Slice-order fix:** filenames are SOP UIDs, so filename order is random (median \|Spearman ρ\| 0.164 against IPP over 60 series). Sort by IPP along the slice normal. | 745759, willyp74, 10-04 | One fold, same seed: CV 0.7695 → **0.7971 (+0.0276)**. A re-seed gave 0.7960, so seed noise is about 0.001. | CV claim | The reader already sorts by IPP·normal (verified in `preprocess.py`). The Tri stack's DINO stage orders by IPP, and A5 orders by InstanceNumber, the way it was trained. **No free fix is available to us**, and changing a pretrained component's order at inference would create a mismatch. |
| Filename order matches anatomy only about 5 % of the time; InstanceNumber runs against position in more than ⅓ of series. | 735154 (stevenleehans); 741310 (gchauhan, 09-14) | — | claim | Same. |
| **Eval window count** for MIL CoAtNet. | 737696, dreaddevelopment, 09-05…09-23 | Same weights: 42 windows 0.924 → 62 windows 0.927 → 78 windows 0.932. 48 → 72 windows "moved the board" (the number is truncated). Spreading 44 slices over 2–98 % instead of 6–94 % lost 0.009. | LB-probe | Supports R4 (reader k sweep). |
| Window-coverage curve on weak-label OOF: k16 0.8867, k24 0.8909, k31 0.8932, k62 0.8948. It saturates at about 31; denser slices cost −0.0045 in their pipeline. | 737696, Ziad Ahmed, 09-19/23 | as stated | CV claim | Expect diminishing returns. |
| 94 → 24 windows on an older model. | 746792, nartaa | 0.940 → 0.934 | LB-probe | Same. |
| (ours) E03b leg: k_eval 24 → 42. | audit/07 E03b-k | +0.0058 pooled OOF, both folds | **verified** (ours) | Same mechanism; never LB-tested. |
| Native 320 crop of a 384-trained model, no resize. | 746792 | 0.948 → 0.948. Crops of 304/288 → 0.947/0.946. | LB-probe | Low value for the reader. |
| Centre-9 adjacent slices beat 9 equally spaced (geometry matters more than resolution). | 737597, Berat Kirbiyik, 08-26 | +0.018 OOF on fold 0 | CV claim | The reader already uses 3 adjacent slices per window. |
| Resolution is not a driver: 224/288 is the "sweet spot"; 384 → 288 "barely had any effect". | 735304 (Tucker, tennogh, Raymond Yuen); 743148 | — | claim | Do not upscale the reader. |
| Field of view: 140–160 mm, and "different models do well with different FOVs". 100 mm cuts anatomy off; 140 "helped a ton". | 735304 [12.1.x], Parag (33), Cody_Null (GM), 09-11…09-30 | — | claim | The reader uses about 141 mm, which is fine. |
| Depth (slice position) conditioning plus laterality canonicalisation. | 746611, Phil Gear (rank ~2000), 10-07 | CV 0.8208 → 0.8782 | CV claim, promotional, unverified | The reader already has slice-position embeddings, so laterality canonicalisation is relevant (R7). |
| Laterality "ruled out": the medial/lateral AUC gap is no larger than for other targets. The side derived from geometry agrees with the tag 98.4 % of the time. | 737597, 08-26 | — | CV claim | Weakens R7. |
| DICOM metadata shortcut: random folds 0.6515 vs site-grouped 0.5981. No meaningful shortcut. | 733517, Oleksii Zhukov, 08-08 | — | CV claim | — |

### 2.3 TTA, routing, blending

| Technique | Source | Claimed effect | Evidence | Relevance |
|---|---|---|---|---|
| **Anatomical left/right mirror TTA:** sagittal → reverse the slice channels; coronal/axial → flip the width; average probabilities 50/50. | 746792, nartaa, 10-07 | 0.948 → **0.949** | LB-probe; code verified in audit/13 | R4 (reader version: reverse sagittal slice order and flip coronal/axial width). |
| A width flip on *every* plane (not anatomical). | 746792 | 0.940 → 0.939 (older checkpoint) | LB-probe | Use the anatomical version only. |
| "Multi-scale 256 + 320 TTA, dual-stride sampling, mirror TTA, pathology routing 2.0 (+0.079 Lateral OA), anti-tie calibration" → "LB 0.954+". | 747289, Hitarth Jain, 10-08 (**−21 votes**) | 0.954+ | **claim, contradicted.** Per audit/14 this notebook is an Apex fork that scores 0.950. | Ignore. |
| Per-target reader routing (Apex: reader weight 0.03–0.35 per finding). | audit/13 | 0.949 → 0.950 | verified notebook, LB noise | The weights are gold-derived. Do not tune them further. |
| Blend tweaks (0.55/0.15 vs 0.60/0.10, per-label maps) are minute-scale changes that overfit the public LB. About 200 forks sit at 0.941. | 742050, starkhushi / Mattia Angeli, 09-19 | — | claim | Supports our "no LB-tuned weights" rule. |
| "Models that disagree with each other outperform models that agree". | 742926, Dread Development / Chris Deotte, 09-27/28 | — | claim | Diversity (R1, R5). |
| An ensemble of own families: 5-fold EffNet 0.940 + 5-fold ResNet 0.940 + 5-fold DINOv2 0.931 + a public CoAtNet 0.928 → **0.947**, later 0.949. | 744511, SpeedSci, 09-30/10-01 | +0.007 over the best member | LB-probe | Shows blending own models works. A merge target (R1). |
| Two single models 0.953 + 0.951 → 0.955. | 735304, colum2131 | +0.002 | LB-probe | Size of the ensemble gain at the top. |

### 2.4 Architecture, backbone, training schedule

| Technique | Source | Claimed effect | Evidence | Relevance |
|---|---|---|---|---|
| DINOv2-S → B: +0.0011 against a 0.0020 noise floor. Encoder scale is a null. | 735154, stevenleehans, 08-17 | — | CV claim | Our E17 (ConvNeXt-small < tiny) agrees. Do not scale up. |
| DINOv2 Base converges faster but ends at the same score as Small. | 738096, Komil Parmar, 08-29 | — | gold claim | Same. |
| DINOv3 is no better than DINOv2 (V2 0.775 vs V3 0.763). Frozen DINOv3 cannot detect ACL/MCL tears. | 733313 (PC Jimmmy, 08-07); 740162; 747553 (Jin Park, 10-09) | negative | claim | Avoid. Access is also gated and the rule is unclear. |
| Fine-tuning a vision LLM: Qwen 3.5-2B with LoRA on the language model plus the vision encoder, 384 input. | 735304 [19], NguyenThanhNhan (GM, 61), 09-21/23 | single fold **0.950**, about 3.5 h | LB-probe | Interesting diversity, but too GPU-heavy for us. |
| Small models: "start with 224/288 and a ResNet34 or EffNet-B0, simple pooling, no attention → 0.94+". CV threshold about 0.003. Ablate one thing at a time. | 740610 [1], Tucker Arrants, 09-10 | 0.94+ | claim (top-20 author) | R5. |
| Single-fold small ResNet: **0.949**, about 5 min scoring, Gemma-4 labels (~0.89 on gold), about 900 models trained. | 735304 [8.x], 740474, Scott Willis (29), 09-05…09-19 | — | LB-probe | R5 and efficiency. |
| 25 → 50 epochs: 0.899 → 0.903. 2×2 spatial region tokens: +0.000003. | 740610, Will, 09-10 | — | LB / CV claim | Our reader's validation plateaus at epochs 8–12 (ours, verified), so longer training is not indicated. |
| EMA / mixup / 60 epochs within noise; **asymmetric loss −0.139**. | 737597, 08-26 | — | CV claim | Avoid ASL. |
| SWA (top-3 epoch average) vs epoch 14: 0.948 vs 0.947 (gold preferred epoch 14). Over six pairs SWA averaged out to nothing (dreaddevelopment). | 746792; 737696 | ±0.001 | LB-probe | Neutral. |
| **Retrain variance on LB:** BN-replay retrain 0.948 → 0.944; fresh crop-trained retrain 0.949 → 0.946. | 746792 | −0.003 to −0.004 | LB-probe | **Calibrates the noise of every training decision we make.** |
| A medically pretrained encoder (RadImageNet) helps Synovitis. | 737566 | Synovitis 0.78 vs 0.62–0.72 | claim | Minor. Its licence is unclear (745283). |

---

## 3. What top-LB people reveal (c)

Ranks are as displayed in the archive. They are approximate and change daily; for example, Quan Vu showed 14 on 10-01 and 42 on 10-08.

| User (rank) | Score | Disclosed approach | Source |
|---|---|---|---|
| tennogh (8) | 0.942 raw / 0.943 with TTA at 288 (early Sept; current score unknown) | Own labels "not measurably better than the public ones"; "a lot of incremental improvements"; OOF pseudo-labels correlate with LB; resolution is not a big driver. | 735304 [12], 09-03 |
| YYama (10) | ~0.96 (implied by rank) | **No external data.** Teamed up. Declined to share details. | 743416 [8.2.x], 10-07 |
| Quan Vu (14 → 42) | — | "Reaching 0.959 would already be a stretch" (09-27). Only +0.001–0.002 gains; "last hope … ensembling". Downloaded OAI on 10-08. | 735767, 746792 |
| Tucker Arrants (17–19) | "low 0.95 range" | 5-fold ResNet/EffNet at 224/288 with simple pooling. Single-prompt Qwen labels. CV on report labels with a 0.003 threshold. Gold-58 used *in training* (1.3 % of loss mass). Cautious about OAI. | 735304, 740610, 745214 |
| CoolinLai (21) | **0.954** | 5-fold ResNet50 at 224 px. | 735304 [24], 09-28 / 10-02 |
| minhtu.mt.mt (26) | — | Downloaded OAI, including the "extra 24M data". | 746792, 10-08 |
| Scott Willis (29) | 0.949 single fold (later a 5-fold ensemble) | Small ResNet, fast (about 5 min), Gemma-4 labels. "Spent a lot of time … working around the low quality labels". | 735304, 740474 |
| Raymond Yuen (38–43) | 0.949 | CoAtNet at 288 trained on all data. Multi-teacher pseudo-labels at 50/50. API labels under $5. "The model already beats the labels it was trained on." | 735304, 743148, 742327 |
| Tanuki_boosting (43) | **0.955** | Single model, 5 folds. Looking for teammates with 0.950+ single models. | 735304 [27], 10-06 |
| NguyenThanhNhan (61, GM) | 0.950 single fold | Qwen 3.5-2B vision-language model, LoRA, 384. Argues external datasets are "readily accessible". | 735304 [19], 743416 [11] |
| colum2131 (63) | **0.953 / 0.951 / 0.955** | ConvNeXt-T 2.5D and DINOv2-S 2.5D trained on pseudo-labels, with an LLM reading report + MRI. No external data "yet". | 735304 [29], 745949 |
| Musa Tur Farazi (79) | 0.952 | Single model at 224 px. | 735304 [28] |
| Archit Konde (80) | 0.950 single fold | 2.5D CoAtNet at 224. OOF predictions mixed into the labels (heavier toward the model). | 735304 [17] |
| SpeedSci (116–121) | 0.949 | Own EffNet/ResNet/DINOv2 5-fold models plus a public CoAtNet. Labels from GPT/Claude/Gemini voting, then OOF-teacher refinement. "Stuck"; says "we need most is a good label". | 744511, 735304, 745214 |
| Cà Rốt Production (118–137) | claims "0.95+" single fold, single model (ResNet) | No details; asks for help. | 747613, 10-09 |

**Synthesis of the 0.950–0.955 tier:**
- Simple 2.5D CNN or ViT (ResNet50, EffNet, ConvNeXt-T, CoAtNet) at 224–288 px.
- Soft labels.
- One of: pseudo-labels or OOF label mixing, or an LLM that sees images.
- 1–5 folds.
- No public-stack components.

**About the top 1–9 (0.958–0.964), nothing is disclosed.**
- Candidates for the extra +0.005–0.01 are: ensembles of 2–4 such families (+0.002–0.005); OAI-type external supervision (nartaa: about +0.006–0.008, not compute-matched); and heavy public-LB iteration (about 300 submissions per team are possible).
- At least some top-50 users began downloading OAI on 10-08.

---

## 4. Known pitfalls (d) and our status

| Pitfall | Evidence | Our status / action |
|---|---|---|
| **bf16 on T4/P100** is emulated about 50× slower without any error, so a time guard writes 0.5 for half of the test. | 744230, Md Kishor Morol, 09-29: 0.851 → **0.710**, fixed back to 0.851 with fp16. Check `torch.cuda.is_bf16_supported(including_emulation=False)` (Roman Vuskov, rank 31, 10-01). | The Tri stack's A5 stage runs bf16 → **E20 (fp16) is committed but not submitted**. The reader's `train.py` and `predict()` use bf16; our E16 already switched to fp16 + GradScaler. Any new training kernel must stay on fp16. |
| **Hidden-test exceptions** while the 3 visible studies pass. | 745984 (10-05…10-09: fails 1–2 h into scoring; zhuo wamg had 4 of 5 submissions fail); 745826 (10-04); 742122; 743779 (Chris Deotte: run inference on about 1,300 train studies in a commit; the common causes are NaN predictions and malformed DICOMs); 734062. | Keep per-study try/except, a NaN check and a time guard. Before any new final, run a **CPU commit on 1,300 train studies** (no GPU quota). |
| Corrupted DICOMs: PixelData holds half the bytes the header claims. | 738708 and 737163 (2 training studies). The host's fix "may have failed" (741916). | The reader's `load_series` skips unreadable slices. Keep it. |
| Missing codecs (JPEG Lossless / JPEG 2000), P100 incompatibility, mount paths `/kaggle/input/competitions/<slug>/` vs `/kaggle/input/datasets/<owner>/<ds>/`. | 735854, 746124 [1] | The reader ships pylibjpeg wheels, and the Tri stack resolves paths dynamically. Keep T4. |
| Kaggle-side `ERRORED_MOUNTING_DATASET`. | 746542, 10-07 (Quan Vu: the resubmission did not use up an attempt) | Resubmit; it is not our bug. |
| **Slice order** by filename or InstanceNumber. | 745759 (+0.028 CV), 734105, 735154, 741310, 738339 | The reader is correct (IPP·normal). Do **not** re-order the inputs of pretrained Tri components at inference. |
| **Laterality.** The tag is missing in about half of studies. Derive it from geometry, and if you flip for augmentation, swap the medial and lateral labels. An anatomical mirror is not the same as an all-plane flip. | 741310, 737597, 734105 §6.3, 746792 | Relevant to R4 (mirror TTA) and R7. |
| Bilateral studies (both knees in one study; 7 such training studies). | 735639, 733826 (**HOST**: disambiguated via the report or metadata), 733343 #13 (unanswered) | The reader's centre crop at 154 mm can miss a knee. The effect is minor; no action. |
| Do all planes exist in the test set? Unanswered. | 744519 | Keep missing-slot masks (the reader has them). |
| A checkpoint's own normalisation contract (RAD-DINO mean/std); a silent backbone fallback. | 735154 | Assert backbone and normalisation for any new model. |
| **Overfitting gold-58.** | 745214 (gold 0.945 → LB 0.910); 746792 (epoch 14 better on gold, worse on LB); 747540; 735304 (Tucker) | Gold stays a diagnostic only (our rule). |
| **LB retrain noise ±0.003–0.004**, and 3-decimal rounding. | 746792; our medvax rerun 0.943 vs 0.944 | Pre-register margins of at least 0.003. Use 2 seeds for important A/B tests. |
| **Runtime.** The Tri stack takes 6.9–8.7 h wall time, close to the 9 h limit. | audit/07 E18–E20 | Any addition (mirror TTA, more windows, partner models) needs E20's headroom or must replace something. |
| "Rule-clean" is not fully clean: the Tri stack's A5 uses **DINOv3-derived** weights (gated by Meta; not ruled on, 733313), and its RadImageNet licence is ambiguous (745283). | 733313, 745283, 742926 | This mostly matters for prize eligibility. Under the host's accessibility test, DINOv3's gated approval is a small residual risk. Note it; no action unless the host rules. |

---

## 5. Where we stand and what the gap is made of

**Our assets.**
- Tri-Specialist stack plus the goodpjw2008 reader: **0.945**, from E18 (7 reader checkpoints) and E19 (a fifth Raptor view). Both were neutral.
- Our own legs: EffNetV2-S Raptor recipe 0.896–0.899 alone, and ResNet34 E13.
- The reader alone: 0.929 (the author's 3 folds). Our fold-3/4 reader: validation 0.8836 / 0.8933, gold 0.901 / 0.899.
- The PNG cache of all 24,371 train series (384 px, the author's preprocessing).
- 4 public LLM label tables, plus the team consensus labels and karttikjangid05's new CC0 consensus.

**Where the LB sits** (from forum ranks, approximate):

| Score | Approximate rank |
|---|---|
| 0.964 | #1 |
| about 0.958 | top 10 (YYama) |
| 0.954–0.955 | rank about 20–60 |
| 0.949–0.950 | rank about 110–135 |
| 0.945 | rank about 160–200 (Pand at 0.945 was rank 159 on 10-07) |

With about 4,500 teams, the silver line (top 5 %) is around rank 225. **0.945 is in silver today, but the line is rising.**

**The 0.945 → 0.964 gap (0.019 of macro AUC, i.e. about 0.23 of summed per-finding AUC). Honest decomposition, all estimates:**

| Component | Estimated share | Basis |
|---|---|---|
| Stronger own single models through image-informed labels (pseudo-labels / vision LLM) | +0.005–0.010 | colum2131 0.953 vs our reader 0.929 (same family); Archit and Raymond at 0.949–0.950; Synovitis and Effusion are the largest per-finding gaps (reader gold Synovitis 0.68–0.70 vs 0.859 reported). |
| An ensemble of 2–4 decorrelated strong own families | +0.002–0.005 | colum2131 +0.002; SpeedSci +0.007 from weaker members. |
| External supervision (OAI masked PF OA / Lateral OA / Synovitis) | +0.000–0.008 | nartaa 0.940 → 0.948 (not compute-matched). Unknown whether the top uses it. |
| TTA and inference details | +0.001–0.003 | mirror +0.001; window count +0.003–0.008 on MIL models |
| Public-LB iteration and noise | ±0.002–0.004 | retrain noise; 3-decimal rounding |

**Is beating 0.964 realistic?**
- **No**, with the resources and 13 days left. My estimate is under 5 %.
- It would need at least one own model at about 0.955 plus a second decorrelated family plus probably OAI. Our best own-trainable model is 0.929 alone, and retraining it costs about 5–6.5 GPU-hours per model.
- What *is* realistic: **0.948–0.952 rule-clean** if pseudo-labels or vision-LLM labels lift the reader to about 0.940–0.945 alone, and/or a merge brings an independent strong family. If OAI is allowed: **0.950–0.953** via nartaa's weights blended with our stack.
- A clean, CV-validated own model also protects against private-LB shake-up, which tends to hit public-stack forks (742327).

---

## 6. Ranked experiment ideas: rule-clean

Ranking is by expected LB gain per GPU-hour, weighted by the strength of the evidence. All are **PROPOSED**. GPU numbers are Kaggle T4 quota-hours. One session on T4×2 trains 2 models in parallel; the reader takes about 5.0 h per fold model, measured in E16.

### R1. Team merge before 2026-10-15 with an own-model team (no GPU)
- **Hypothesis.** Our 0.945 is a public-stack blend. A partner with strong, independently trained models adds the decorrelation that the reader (+0.001) could not.
- **Evidence.**
  - SpeedSci's own 0.940/0.940/0.931 models plus a public CoAtNet → 0.947–0.949 (744511).
  - colum2131: 0.953 + 0.951 → 0.955.
  - Decorrelation is "the secret" (742926).
  - Our E16/E18/E19 show that more of the same family adds nothing.
- **Candidates seen in the forum.** Their data sources need checking.
  - Pand: 0.945, 5-fold bag with full OOF (745091, 744511).
  - SpeedSci: 0.949, own EffNet/ResNet/DINOv2.
  - ringbearer: 0.948 blend.
  - KalyanG17: 0.942, own labels, spare GPU hours.
  - Md Kishor Morol: an independent DINOv2-S arm (744230).
  - Nikita Ageev: CoAtNet-2 0.937 with own Claude labels and an RTX 4090 (743374).
  - Christoffer Thimsen: a team at 0.940 with own label tables (746060).
  - kmn: single CoAtNet 0.940.
- **Expected gain.** +0.002–0.006 if the partner's family is at 0.94 or above and decorrelated.
- **GPU.** 0 training. Inference has to fit within 9 h *together with* the Tri stack, so E20 is a prerequisite, or the stack is swapped for the leaner 0.943 base.
- **CPU-only alternative.** N/A. This is an organisational action.
- **Risks.**
  - The merged submission count must stay within Kaggle's cap, so check both teams' counts.
  - A partner trained on OAI/KneeCoT or other unclear data contaminates our team, so require a written data-source list.
  - Runtime.
  - Trust.
- **Fit with our assets.** Good. A rank blend at a pre-registered weight (the partner gives OOF on our folds if possible, otherwise w = 0.30 fixed).
- **Decision rule.** Merge only if the partner's own model scores at least 0.940 alone and uses only documented rule-clean data.

### R2. Pseudo-label (out-of-fold soft bootstrap) retrain of the reader: a clean A/B
- **Hypothesis.** Targets t = 0.5·mean(4 LLM tables) + 0.5·reader-OOF move the reader from about 0.929 toward the 0.94–0.95 that the same family reaches with pseudo-labels.
- **Evidence.**
  - Raymond Yuen (0.938 → 0.949, 50/50).
  - Archit Konde (0.950; OOF only, heavier on the model).
  - colum2131 (0.953 ConvNeXt-T).
  - tennogh, Chris Deotte.
  - Negatives: Nicolai Karcher, and Myo Min Htet (same encoder as teacher and student).
- **Steps.**
  1. **Complete 5-fold OOF.**
     - Run the author's fold 0–2 checkpoints on their own validation folds (the GroupKFold split is the same as ours; E16 reproduced the fold sizes 870/870/870/870/869).
     - The author's dataset ships **no OOF files** (checked through the Kaggle API: only the 3 checkpoints plus code and wheels).
     - Our fold-3/4 OOF already exist (`ours_leg/results/gpjw_f34_v1/oof_fold{3,4}.csv`).
     - Use R4's best inference config if it is ready.
  2. **Build the targets** on all 4,349 non-gold studies. Pre-register α = 0.5.
  3. **One T4×2 session, two models in parallel, `--fold -1`:**
     - **A** = control: full-data, original labels.
     - **B** = full-data, pseudo-label targets.
     - Everything else identical: fp16, 14 epochs, seed 0.
  4. **Leg-alone LB for A and B**, one submission each.
- **Expected gain.** Reader alone +0.005–0.015 (a wide range). In the 0.945 blend, +0.001–0.004 at w = 0.30; more if B is at 0.940 or above alone and the weight is raised to a pre-registered 0.45.
- **GPU.** OOF inference about 0.3 h (or 0 on CPU). The A/B session is about 6–6.5 h. A second seed of the winner, if wanted, is another about 6.5 h.
- **CPU-only alternatives.**
  - Step 1 can run on Kaggle CPU in 2–3 shards, about 1–1.5 h each.
  - Training on the user's Mac (MPS) is possible in principle: 735154 trained a fold on an M4 Pro in 67 min for a smaller DINOv2-S/224 cache. But our cache is 53.6 GB of PNG, MPS is non-deterministic, and they saw a −0.0026 offset between local and Kaggle results. Feasible only if the Mac has enough RAM and disk; not verified.
- **Risks.**
  - Self-distillation with the same encoder may not transfer to LB (two negative reports).
  - A full-data model has no OOF, so the decision is LB plus gold diagnostic.
  - LB retrain noise of about 0.003.
- **Fit.** Excellent: the author's code, our cache, our fold-3/4 OOF.
- **Pre-registered rule.** Adopt B only if LB(B) − LB(A) ≥ +0.003 and gold-58 does not drop by more than 0.01 (diagnostic only).

### R3. Vision-LLM labels (report + 2–4 key slices) for Effusion and Synovitis (possibly Baker's and Contusion)
- **Hypothesis.** Text cannot recover image-only findings. Giving an LLM the relevant slices lifts label quality exactly where reports are silent or graded.
- **Evidence.**
  - colum2131: label AUC Effusion 0.835 → about 0.92 and Synovitis 0.80 → about 0.85 on gold; an image model on them reached 0.986 / 0.859 (745949).
  - hengck23's "show ChatGPT report + image" (745861).
  - The host's rubric: effusion must be moderate or large; synovitis is lining thickening.
  - colum2131: 38 % of "effusion absent" reports were gold-positive (745949).
- **Expected gain.** Label AUC on those 2–4 columns +0.04–0.08. LB +0.002–0.006 *if* the image model learns it (Synovitis alone is 1/12 of the macro).
- **GPU.** 0 for labelling. Cashing it in needs training: run it as arm C in a later R2-style session (about 6.5 h), or merge it into B's targets if the budget forces it (which loses attribution).
- **Cost.** API only. About 4,349 studies × (report plus 3 images at about 1–1.5k tokens each) ≈ 20–30 M input tokens. That is roughly $25–90 depending on the model, an order-of-magnitude estimate. Pilot on 300 studies first.
- **CPU-only alternative.** This *is* the GPU-free route. Slices come from the existing PNG cache: mid-sagittal FS through the suprapatellar recess, and axial FS at the patellofemoral level.
- **Risks.**
  - **Rules:** the host explicitly allowed LLM APIs for *reports* (733965). Images are not addressed explicitly. Use a zero-retention API setting, and consider asking the host.
  - Prompt overfitting to gold-58: write the prompt from the host rubric before looking at gold, and score gold once.
  - API cost.
  - The vision LLM over-calls synovitis on non-contrast MRI (colum2131's own caveat).
- **Fit.** Good: the cache already holds canonicalised slices.
- **Pre-registered rule.** Gold AUC is diagnostic only. The adoption decision is a leg-alone LB of at least +0.003 over arm A.

### R4. CPU-only reader inference sweep plus complete OOF (better teacher, no training)
- **Hypothesis.** Cheap inference changes improve the reader's OOF and therefore the R2 teacher:
  - **(a)** k_eval 12 → 16 / 24 windows per series (at most 64 slices);
  - **(b)** anatomical mirror TTA: reverse the sagittal slice order, flip the coronal/axial width, keep the labels;
  - **(c)** a new "series-alternative" TTA.
- **Why (b) is in-distribution.** The reader's canonical frame is patient LPS without laterality canonicalisation, so a mirrored knee looks like a contralateral knee, and the training data contains both sides.
- **Why (c) is in-distribution.** In training `StudyDataset` picks a **random** candidate series per slot, but inference always takes `cands[0]`. Averaging over the alternative series, which are often a second contrast (741310), is therefore in-distribution.
- **Evidence.** dreaddevelopment and Ziad Ahmed (window count; saturates at about 31 windows), nartaa (mirror +0.001; 94 → 24 windows −0.006), our E03b-k (+0.0058 OOF), Apex (reader k = 16).
- **Expected gain.** Reader OOF +0.002–0.006. In the blend LB +0.000–0.002, which is likely invisible at 3 decimals (as E16/E18 were). The value is mostly the better teacher for R2.
- **GPU.** About 1–1.5 h for every config on folds 3/4 plus folds 0–2 OOF, or **0 on Kaggle CPU**: about 3–4 s per study per config, so about 2 h per config per 1,740 studies, in parallel CPU sessions.
- **CPU-only alternative.** Yes, that is the default.
- **Risks.**
  - The reader was trained with 8 windows per series; 16–24 tokens per series may dilute attention (Ziad: "more tokens competing in the same softmax").
  - Mirror doubles the reader's runtime inside the 9 h stack.
- **Fit.** Excellent: the checkpoints, the cache and `infer.py` exist. Truth labels can be rebuilt locally from the 4 label CSVs (verified in E17).
- **Pre-registered rule.** Adopt a config only if pooled OOF on folds 3+4 improves by at least 0.003 and both folds are positive.

### R5. Runtime hygiene: E20 (A5 in fp16) plus a dry run on 1,300 train studies (CPU)
- **Hypothesis.** The 0.945 stack sits at 6.9–8.7 h with an emulated-bf16 stage. Every addition (R1 partner, R2 reader, mirror TTA, an OAI candidate) needs headroom and protection from hidden-test failures.
- **Evidence.** 744230 (bf16 emulation), 743779 (dry run on 1,300 studies), 745984 / 745826 (hidden-test exceptions), our E18/E19 runtimes.
- **Expected gain.** 0 on LB directly. It protects against −0.1-class failures and unlocks R1–R4.
- **GPU.** E20 is already committed. Submitting it costs one slot (the hidden-test rerun is the scoring compute). A CPU commit run over 1,300 train studies uses CPU quota only; verify that GPU-only stages have CPU paths, or run just the reader and new components.
- **Risks.** fp16 overflow in A5: assert finite outputs.
- **Fit.** Ready (`submission_e20/`).

### R6. A second, cheap own family for diversity: a simple 2.5D CNN at 224 px, mean pooling, on pseudo-label targets
- **Hypothesis.** The 0.950–0.955 tier uses plain ResNet50 / EffNet / ConvNeXt at 224–288 px with simple pooling. A second family trained on R2/R3 targets, using the reader's cache and dataset code but a different backbone, head and resolution, adds decorrelation that our EffNetV2-S Raptor leg (0.896–0.899) did not.
- **Evidence.** Tucker (740610: 0.94+ with ResNet34 / EffNet-B0 and simple pooling); CoolinLai (ResNet50 224, 5-fold, 0.954); Scott Willis (small ResNet 0.949); Cody_Null (EffNet-B0 0.94+ with 0.88 labels plus 5-fold TTA, 743148).
- **Expected gain.** +0.001–0.004 in the blend if it reaches about 0.935 or more alone.
- **GPU.** About 4–6 h per full-data model. The loader is probably the bottleneck (PNG decode), so it is not much cheaper than the reader. Running in parallel with an R2 arm saves a session.
- **CPU-only alternative.** The Mac (MPS) is the most realistic here, since EffNet-B0 at 224 is about 10× lighter than ConvNeXt-T at 256. The 53.6 GB cache would have to be downloaded.
- **Risks.** It costs GPU we may not have. Our previous own legs were weak, and the cause was never isolated beyond "same corpus, labels and head as Raptor".
- **Fit.** Medium: reuse `knee.py` with `timm` backbone `resnet50` / `efficientnet_b0` and `depth=0` (no transformer).

### R7. Laterality canonicalisation in reader training (mirror left knees into a right-knee frame)
- **Hypothesis.** The reader has slice-position embeddings. Without canonicalisation, position 0.2 means medial for one side and lateral for the other, which hurts the medial/lateral pairs (Lateral Meniscus and Lateral OA are the reader's weak findings: gold 0.85–0.87).
- **Evidence.** 746611 (CV claim, unverified); nartaa mirror (+0.001); 741310 / 734105 (method). **Counter-evidence:** 737597 (laterality ruled out in a model without position embeddings).
- **Expected gain.** +0.000–0.004 on the reader alone.
- **GPU.** A full retrain (about 6.5 h), so it is only worth doing as a later arm. Side is derived from IPP geometry, which matches the tag 98.4 % of the time (737597).
- **CPU alternative.** None for training. R4(b) is the inference-only analogue.
- **Risk.** Low evidence, and it costs a full arm.

### R8. Add more public label sources to the soft-target mean
- **Sources.** JEV labels (riadmohamed42), steven_v4, karttikjangid05's CC0 consensus.
- **Evidence.** nartaa +0.002 (confounded); our E03b +0.003 on the weak leg. Against: 747540 and 747658 show the label sets are statistically indistinguishable.
- **Expected gain.** +0.000–0.003.
- **GPU.** Needs a retrain. Fold it into R2's teacher mean instead of running a separate arm.
- **Risk.** Low value.

### R9. Synovitis structural prior (post-processing, CPU only)
- **Hypothesis.** Experts call synovitis together with effusion. Synovitis_final = rank-mean(Synovitis, Effusion) at a fixed 0.5 weight may beat the reader's weak Synovitis head (gold 0.68–0.70, below the label tables).
- **Evidence.** 737566 (Effusion/Synovitis correlation about 0.5 in gold); colum2131 (31 % of report-silent synovitis cases are gold-positive); the reader's teacher beats the reader on Synovitis.
- **Expected gain.** ±0.002.
- **GPU.** 0. One pre-registered LB probe.
- **Risks.** It is a per-target change decided on the LB. Our rule forbids *fitting* per-target weights on the LB; a single fixed, pre-registered transform is borderline. Cannot be validated offline: the report Synovitis labels are themselves partly imputed from effusion, so the comparison is biased.
- **Recommendation.** Low priority. Only as a one-shot probe if submission slots are spare.

### R10. Pre-registered weight change for an improved reader (follow-up to R2/R3)
- If the new reader is at least 0.940 alone, test reader weight 0.45 against the current 0.30 within the Tri blend, in two submissions.
- Adopt 0.45 only if the gain is at least 0.002 *and* consistent with the leg-alone gap.
- **GPU.** 0 beyond the inference already planned.

### Not recommended (evidence says low or negative value)
- Bigger backbones (735154 null; our E17 small < tiny; 738096).
- DINOv3 (733313, 740162, 747553; also gated).
- Asymmetric loss (737597: −0.139).
- Longer schedules for the reader (our validation plateaus at epochs 8–12).
- More text-only label engineering (747540, 747658, Tucker).
- Image → report generation (743148).
- Report auxiliary loss (740162).
- Multi-scale 256 + 320 TTA (747289 is unverified, and the reader was trained at a single scale).
- LB-tuned per-target weights (742050).
- Training on gold-58 (conflicts with our guard; at most +1.3 % of loss mass per Tucker).
- heliosli's 0.951 "blend-gold" fork (adds SKM-TEA and KneeXNet weights, more external data with unanswered status, *plus* OAI).

---

## 7. Ideas only if OAI is ruled allowed

| # | Idea | Expected | GPU | Risk |
|---|---|---|---|---|
| O1 | **nartaa 0.949 CoAtNet (accuracy model, mirror TTA, 94 windows) + our Tri stack + reader**, as a pre-registered rank blend: Tri stack 0.5 / nartaa 0.5 for the main candidate, reader as is. Apex (nartaa + reader) already scores 0.950. | **0.950–0.953**. Tri is CoAtNet-heavy too, so the correlation is moderate. | 0 training. Inference: nartaa about 26 min submit-to-score alone. Fits only with E20 headroom. | Licence "Other" (a prize issue); a gold-58 / LB-selected recipe. |
| O2 | Fix the failed blend-gold fork (pin `heliosli/rsna-knee-compact-v1/9`, audit/14). | 0.951 | 0 | **Adds SKM-TEA and KneeXNet weights**, whose status is unanswered (743416). Not recommended even if OAI is cleared. |
| O3 | Use nartaa's model as an *extra teacher* in R2, only for PF OA, Lateral OA and Synovitis (the OAI-supervised findings). | +0.001–0.004 on top of R2 | same as R2 | nartaa was trained on all RSNA studies, so its train predictions are in-fold and partly parrot the labels. The benefit is limited to the OAI-informed findings. |
| O4 | Download OAI ourselves and add masked MOAKS supervision. | +0.005 (nartaa's ladder) | High, plus about 450 GB of download and label mapping | Not feasible in 13 days on our budget. **Skip.** |

If OAI is allowed, it is allowed for everyone, and top teams likely already use it. **O1 is catch-up, not an edge.**

---

## 8. Suggested schedule (needs user approval; the user has said no GPU without asking)

| When | Action | GPU quota |
|---|---|---|
| 10-09 → 10-11 | **R1** outreach (merge deadline 10-15). **R4** CPU kernels: full reader OOF on folds 0–2, plus the k / mirror / series-TTA sweep on folds 3–4. **R3** pilot on 300 studies (API) with a rubric-based prompt; score gold once. | 0 |
| 10-11 → 10-12 | Submit **R5** (E20) if not done, to confirm runtime headroom. Build the R2 targets (using R4's best teacher config). | 0 (submission slot only) |
| 10-12 → 10-13 | **R2 A/B session** (2 models in parallel, full data). | about 6.5 h |
| 10-13 → 10-14 | Leg-alone LB for A and B (2 submissions). If B wins, plan the blend probes (R10). | 0 |
| 10-15 → 10-18 | Second session: arm C (R3 labels) **or** a second seed of B **or** R6. Whichever has the stronger evidence by then. | about 6.5 h |
| 10-18 → 10-21 | Final blends with pre-registered weights. 1,300-study dry run. Choose finals. Decide on OAI only if the host has answered. | 0 |
| 10-22 | Final selection: 1 best clean + 1 robust clean (fewest LB-tuned constants); or 1 clean + 1 OAI, **only** with a host "yes". | — |

Total: about 13 GPU-hours plus CPU kernels, which is within the ~20 h/week across two accounts.

---

## Appendix A. Topics read (created or commented on since 2026-10-01; all 50)

733343, 730709, 747676, 747660, 735304, 743416, 747658, 746792, 745984, 747613, 747553, 747540, 747455, 745861, 745798, 746611, 747289, 740610, 739905, 746124, 746542, 745091, 746443, 745820, 745823, 744519, 746060, 737333, 742053, 745979, 745949, 744056, 745730, 735767, 733932, 744511, 745826, 743374, 745759, 745441, 745576, 745666, 735121, 733652, 745283, 744495, 745214, 744961, 733313, 744230.

The older topics are listed at the top. All HOST/ADMIN comments in the archive (174 topics) were extracted. **The most recent host comment of any kind is 741819, 2026-09-21.**

## Appendix B. Facts verified locally for this report

- **goodpjw2008 reader, `ours_leg/gpjw/src/`:**
  - It sorts slices by IPP·normal with a canonical LPS orientation and **no** laterality canonicalisation.
  - It uses slice-position embeddings.
  - Training picks a random candidate series per slot; inference picks `cands[0]`.
  - Training uses k = 8 windows per series, evaluation k = 12, at 256 px from a 0.92 FOV crop of the 384 cache.
  - `train.py` uses bf16 autocast; our E16 used fp16 + GradScaler.
- **Public reader dataset file list** (Kaggle API, read-only): `cnxt_v0_fold{0,1,2}.pt`, the code and the wheels. **There are no OOF files.**
- **Our reader fold logs** (`ours_leg/results/gpjw_f34_v1/`):
  - Validation plateaus at epochs 8–12: fold 3 0.8814 → 0.8842 → 0.8836; fold 4 0.8911 → 0.8935 → 0.8933.
  - About 1,095 s per training epoch plus about 370 s per evaluation on a T4; 5.0 h per fold.
  - Gold Synovitis 0.679 / 0.700, Lateral Meniscus 0.851 / 0.860, PF OA 0.813 / 0.848.
- **Tri-Specialist notebook (`submission_tri945/`):**
  - The DINO stage orders slices by IPP, with a filename fallback.
  - The A5 stage orders slices by InstanceNumber (as trained).
  - Rad-dual5 uses the legacy `dominant_axis` / `corner_x` rules.
  - None of these should be changed at inference.
