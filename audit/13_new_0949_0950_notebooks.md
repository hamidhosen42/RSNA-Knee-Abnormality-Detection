# 13 — New public 0.949 / 0.950 notebooks vs our tri945 base (analysed 2026-10-08)

Every source was pulled at its **scored** version, resolved through `GetKernelViewModel.bestSubmissionScore`. Nothing was run, pushed or submitted.

Saved to `public_notebooks/`, with rows appended to `_index.csv`:
- `0.950_45v_sujanmajhisuzan__rsna-knee-apex-grandmaster-stack.ipynb`
- `0.949_29v_nartaa__rsna-knee-0949-anatomical-mirror.ipynb`

Velciraptor and haideptry were recorded as `removed_duplicate` and not saved. The reasons are in the duplicate notes under the table.

| # | Notebook | Scored version (scriptVersionId) | Current | Votes | LB |
|---|---|---|---|---|---|
| A | sujanmajhisuzan / rsna-knee-apex-grandmaster-stack | v1 (356192954) | v1 | 45 | **0.950** |
| V | matterhorn3838 / rsna-knee-v2-velciraptor-dinosaur-speed | v6 (356180772) | v6 | 55 | 0.949 |
| N | nartaa / rsna-knee-0949-anatomical-mirror | v2 (356175397) | v2 | 29 | 0.949 |
| H | haideptry / rsna-knee-v2-vs-apex-comparative-study | v3 (356214541) | v3 | 2 | 0.950 |
| — | base: sujanmajhisuzan / rsna-knee-tri-specialist-superstack | view model says **v1** (355737240) is best | v2 | 102 | 0.945 |

**About our tri945 copy.** `submission_tri945/rsna-knee-tri945.ipynb` is byte-identical to Tri **v2** (40/40 cells). The view model lists **v1** as the best-scoring version, also 0.945. v1 differs only in cells 17, 19 and 21, which are meniscus-bag DINO-cache capture plumbing. Its blend cell (cell 38) is identical to v2's.

**Duplicate notes:**
- **V and N:** the code cells are byte-identical. V's cell 1 has sha256 `bcd864cb…`, which is exactly the "source_sha256" of nartaa's scored submission 56906959 in the dataset's `checkpoint-manifest.json`.
  - V drops N's three markdown cells, including the credits and the OAI notice, and does not credit nartaa.
  - V's kernel is a fork of `mattiaangeli/bend-the-knee-to-speedy-raptors-the-original`, with the content replaced.
  - N is the original (dataset published 19:26 UTC, N updated 20:28, V 21:02 on 2026-10-07). I kept N even though V has more votes. That breaks our votes tie-break.
- **H:** code cells 3–9 are identical to A's. H adds markdown, a matplotlib cell and a CSV sanity check.
  - The matplotlib cell plots **made-up "estimated" per-finding AUCs**, which are not measurements.
  - H confirms the A/V relationship but adds no technical content.

## Bottom line

1. **None of the new notebooks is a modification of tri945.**
   - They drop the whole 0.943 community stack: DINOv2 ×20, A5, RadImageNet heads, the 4 Raptor views, the CoAtNet readers and the meniscus bag.
   - They also drop all 16 datasets, the 2 kernels and the model source that tri945 mounts.
   - Instead, N and V run **one CoAtNet-RMLP-2 checkpoint** (nartaa). A runs the same single model plus the public goodpjw2008 2.5D ConvNeXt reader.
   - **Hash-identical cells vs tri945:** N 0/5, V 0/2, A 2/10 (only the reader's `preprocess.py` and `knee.py`).
2. **The 0.945 → 0.949 jump comes from external training data: the Osteoarthritis Initiative (OAI).**
   - The checkpoint `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` was trained on 4,349 RSNA report-labelled studies **plus 2,399 OAI knees**, with masked external supervision for PF OA, Lateral OA and Synovitis.
   - The author's own LB ladder (forum topic 746792):
     - best **non-OAI** single model: **0.942**
     - with OAI: 0.945
     - 384 full-field OAI recipe: 0.948
     - plus anatomical-mirror TTA: 0.949
   - A's further +0.001 (0.950) is the public reader blended in with per-target rank weights. That is the same mechanism that took the community stack from 0.943 to 0.944/0.945.
3. **Rule status of OAI is unresolved, and leans against.** Details are in section 4. Under our rigor rules we should **not** build on these checkpoints.
4. **Without OAI, the best known public number on this lineage is 0.942** (nartaa's own non-OAI single model). Our rule-clean E18/E19 at 0.945 is still at or above every rule-clean public result we know of.
5. **Reusable legally:** two small, single-variable inference ideas. Neither is expected to move public LB by more than about 0.001. See the last section.

## Comparison table

| | tri945 (= Tri v2) / our E18, E19 | N nartaa 0.949 (= V) | A Apex 0.950 (= H) |
|---|---|---|---|
| Base | 0.943 community stack (DINOv2×20 → A5 → RadImageNet heads → 4 Raptor views + CoAtNet readers) + meniscus bag + goodpjw reader | **No stack.** One self-contained 95 kB script (dd Raptor lineage, nartaa fastread DICOM reader embedded as base64) | N's script verbatim (cell 4 = N + 2 lines saving `_sota_0949.csv`) + goodpjw reader |
| Cells identical to tri945 | 40/40 | 0/5 (V: 0/2) | 2/10 (`preprocess.py`, `knee.py`) |
| Models | ~40 checkpoints across families; reader 3 ckpts (E18: 7) | **1 checkpoint**: CoAtNet-RMLP-2 (`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`), 3-epoch weight average (14/11/12), seed 42 | N's checkpoint + goodpjw ConvNeXt-T reader, 3 ckpts (dataset v1) |
| New data sources vs tri945 | — | `nartaa/rsna-knee-publication-swa-weights-20261007` v1 | same + `goodpjw2008/rsna-knee-2-5d-convnext-reader` (already in tri945) |
| Training data | competition only (as far as is published) | RSNA 4,349 + **OAI 2,399 knees (external, NDA controlled access)** | same as N + reader (competition-only, public LLM labels) |
| Inference geometry | stack-specific | 96 slices (26/22/18/12/18), native 384 cache, **centre crop [32:352] → 320 px, no resize**, K_EVAL 94 windows | same; reader `k` 12 → **16** windows per series |
| TTA | maxspan-v5 "reverse" view = `windows.flip(1)` on all planes (w 0.10) | **anatomical L/R mirror**: sagittal → reverse the 3 slice channels; coronal/axial → width flip; done on uint8 before normalisation; 0.5·(p+q) on probabilities | same (CoAtNet only) |
| Precision | A5 bf16 on T4 (E20 = fp16 fix) | fp16 autocast, `isfinite` assert | fp16 (both legs) |
| Blend | stack → rank blend with reader, per-target w 0.03–0.55 (default 0.30) | none (w = 1.0, per-column rank pct) | per-target rank blend, reader w 0.03–0.35 |
| How weights were chosen | tri author: gold-58 per-class numbers (comments) | recipe chosen on gold-58 **and** on public LB (crop 320 vs 304/288, SWA vs epoch 14, mirror) | hand-set from gold-58 per-class numbers plus a misattributed comment (see A/d) |
| Fallback | full contract + reader try/except | per-study exception → row stays 0.5; failed prep → zero volume; no global time guard | reader try/except restores the N csv |
| Runtime | 6.9–8.7 h wall (ours) | 26.1 min submit-to-score, as reported by the author (includes the queue) | about 26 min + reader (3 ckpts, k 16); H's "36–40 min" is a claim, not a measurement |
| Rule risk | low (all public) | **HIGH**: OAI external data, model-use licence "Other" | **HIGH** (inherits N) |

## Per notebook

### N — nartaa "0949 Anatomical Mirror" (v2); V is byte-identical code

**a) Scored version.** v2. The scored source equals the private original `nartaa/rsna-knee-original384-s42-mirror320-infer` v1 (sha256 `bcd864cb…`).

**b) Diff vs tri945.** There is no shared code. Cells 0–2 are markdown: provenance, reproduction notes and the OAI acknowledgement.

Cell 3 verifies the checkpoint sha256 `7e5315da…` before anything runs.

Cell 4 is the single-model script:
- **Model:** dd's `RaptorClassifier` (CoAtNet backbone with per-finding attention-MIL head), loaded `strict=True`. `ARMS` holds one arm at w=1.0.
- **Volume:** 5 slots of 26/22/18/12/18 = 96 slices. Each series is sampled over 2–98 % of its slices, cropped to 140 mm and stored as native 384 px uint8, with a 2–98 percentile window per series.
- **Fast reader:** the embedded `_fastread.py` decodes the base64 blob, which I inspected. It is a DICOM fast-read that claims byte-identity with `build_study_v2`, with no network or IO beyond `/kaggle/input`.
- **Windows:** `K_EVAL=94`, i.e. every window centre. Each window is centre-cropped to 320 px, with no interpolation.
- **TTA:** anatomical mirror (`_mirror_tri`). The slot of each window's centre slice picks the axis. Applied on uint8, before the per-channel ImageNet LUT.
- **Precision:** fp16 autocast with a finite-logit assert. One consumer thread per GPU. Preprocessing runs in a fork pool of `max(4, cpu−8)` processes.
- **Output:** per-column percentile rank.
- **No hard-coded test information:** no UID-like strings; reads `test.csv` and `test_series.csv`.
- **Stale comments:** many inline comments (the "Three arms…" text, the 0.924/42-window docstring, "+0.071 Lateral Meniscus") describe older dd-era versions. The author says so in cell 0.

**c) New data source.** `nartaa/rsna-knee-publication-swa-weights-20261007` v1:
- **Status:** public (HTTP 200), 586 MB, created 2026-10-07, 215 downloads.
- **Licence:** "Other (specified in description)". The description gives research/educational model-use terms, "including the RSNA Knee competition". There is explicitly **no CC0 or open licence**, and OAI data stays governed by NDA terms.
- **Files:**
  - `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` (accuracy model: submission 56906959, 0.949)
  - `raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt` (efficiency model: 224 crop, submission 56707773, 0.945; used by `nartaa/rsna-knee-0945-efficient-224crop`)
  - `checkpoint-manifest.json`
  - `README.md`
- **Training, as reported by the author:**
  - Data: 4,349 RSNA studies with weak labels plus 2,399 baseline OAI knees, 6,748 rows in total. OAI cells exist only for PF OA, Lateral OA and Synovitis; other cells are masked.
  - Weak labels: own report-derived labels blended with stevenleehans, pilkwang and **riadmohamed42 JEV** labels.
  - Schedule: 16 epochs, backbone lr 4.01e-5, head lr 1.95e-3, wd 0.0588, 12 windows in training, batch 1 with gradient accumulation over 16 studies.
  - Gold-58 was excluded from training but **reused for selection**, as the author states.
- **Reported metrics:** public LB only, plus the ladder in topic 746792. No CV is reported.

**d) Weight and recipe choice.** A single model, so there are no blend weights and no per-target overrides. The recipe was selected on gold-58 and on many public-LB probes:
- crop 320 beat 304 and 288 (0.948 vs 0.947 / 0.946)
- the 3-epoch average beat epoch 14 (0.948 vs 0.947), although gold preferred epoch 14
- the mirror took 0.948 to 0.949
- a fresh crop-trained model lost (0.946)

So the inference recipe is effectively **LB-selected**, with steps of about 0.001.

**e) Risks.**
- **Rules:** OAI external data (section 4). Two private assets exist (`nartaa/rsna-knee-oai-original384-s42-pair` and the original notebook) but are **not** mounted.
- **Runtime:**
  - There is no global time guard. That is acceptable at about 26 minutes.
  - A failed study silently stays at 0.5. Rank-ties hurt AUC, but the run does not crash.
  - CUDA is required (`RuntimeError` otherwise).
  - No bf16: fp16 is T4-safe.
  - The image is pinned to `2757e0c7…` (the same digest as tri945). The author recorded torch 2.11 / timm 1.0.29. V ran on torch 2.10 and still scored 0.949.

**f) Gain driver.** OAI supervision plus the 384 full-field recipe (0.942 → 0.948), then the mirror (+0.001).

### A — sujanmajhisuzan "Apex Grandmaster Stack" (v1)

**a) Scored version.** v1, the only version.

**b) Diff.** Cells 0–2 are N's markdown, copied verbatim, so the notebook still says "0.949 public AUC". The code cells:
- **Cell 3:** N's checkpoint check, plus a fallback for Python versions without `hashlib.file_digest`.
- **Cell 4:** N's script, plus `sub.to_csv('_sota_0949.csv')`.
- **Cells 5–8:** the goodpjw reader source. `preprocess.py` and `knee.py` are hash-equal to tri945 cells 35 and 36. `infer.py` differs only in `run(..., k=12)` → **`k=16`** windows per series; the reader was trained with k=8 and evaluated by its author at k=12.
- **Cell 9:** the blend. It backs up N's CSV, runs the reader (3 checkpoints) in a subprocess on the GPU with the most free memory, and applies a per-target rank blend: `(1−w)·rank(N) + w·rank(reader)`, then re-ranks.
  - The weights are: Baker's 0.35, Contusion 0.30, Medial OA 0.25, ACL 0.25, Medial Meniscus 0.20, MCL 0.20, Lateral Meniscus / Fracture / Lateral OA / PF OA 0.08, Effusion 0.04, Synovitis 0.03.
  - All 12 keys match the label columns, so the default of 0.15 is never used.
  - On any exception, cell 9 restores N's CSV.
  - There is no timm wheel; the container's timm is used.
- **Log check.** The output log of v1 shows the blend actually ran on the 3 visible studies: `[Apex Fusion SUCCESS] … 3 checkpoints`. Whether it also ran on the hidden set is unknowable. However, 0.950 ≠ 0.949 suggests it did.

**c) Data sources.** The only one not already in tri945 is nartaa's weights (above). The goodpjw reader v1 is public, Apache-2.0, and holds 3 checkpoints (folds 0–2).

**d) How the weights were chosen.** They are not stated to be LB-tuned. They are hand-set, with the same structure as tri945's cell 38, scaled down roughly ×0.6. The justifications cite two kinds of per-finding numbers:
- **Reader AUCs** (Baker's 0.976, Contusion 0.966, Medial OA 0.966, ACL 0.965). These are goodpjw2008's gold-58 per-class figures, the same numbers as in tri945's comments. So the weights are **gold-58-derived**.
- **"SOTA CoAtNet gained +0.071 / +0.057 / +0.048"** for Lateral Meniscus, Fracture and Lateral OA. These come from a stale comment in N's script about an old dd-era corpus-expansion test on the 58-study gate. They are **misattributed**, not a comparison of N vs the reader.

Compared with tri945's routing:
| Finding | tri945 | A |
|---|---|---|
| Lateral Meniscus | 0.05 | 0.08 |
| Fracture | 0.15 | 0.08 |
| Lateral OA | 0.15 | 0.08 |
| PF OA | 0.10 | 0.08 |
| Effusion | 0.05 | 0.04 |

The ordering is otherwise unchanged.

**e) Risks.**
- **Rules:** inherits N's OAI risk.
- **Runtime:**
  - The reader fallback is sound.
  - The reader depends on the container's timm, because no wheel is mounted.
  - k=16 is untested at the reader's eval geometry, though harmless here.
  - fp16 only.

**f) Gain driver.** About 0.949 from N (OAI). The +0.001 comes from blending the public reader, the same effect that goodpjw2008's reader gave the community stack (0.943 → 0.944). At 3 decimals this is within noise.

### V — matterhorn3838 (v6)

V is identical code to N. Its only difference from N is the missing credits and OAI notice. It adds no new information.

### H — haideptry (v3)

H confirms the analysis: V is a clone of N, and A = N + reader. Its "estimated AUC" plot and its "0.950 ~ 0.952+ guaranteed" claim are illustrative, not measured.

## 4. Rule risk: OAI (applies to N, V, A and H)

**What the host said** (topic 741819, 2026-09-21): OAI "does not violate the … rules **if it is generally accessible to all participants without an escalated (i.e. institutional) review and sign-off** … and is free of charge."

**Why OAI may fail that test:**
- NDA access to OAI needs a Data Use Certification. Several users read this as requiring an institutional sign-off (topics 737950, 743416).
- KneeCoT was banned on the same "institutional agreement" reasoning (topic 734109).

**What is still open:**
- Topic 743416 asks for a per-dataset yes/no and lists OAI as "Unclear". It had 21 comments by 2026-10-08 and **still no host answer**.
- In topic 746792, users warned nartaa. Nartaa replied that OAI's own terms allow it. Another user says NDA signup needs only a login.gov account (email) and posted an email from UMass confirming permission from the data side. As that thread notes, this does not settle Kaggle Rule 2.6(a) ("publicly available and equally accessible to all Participants at no cost").

**Further problems for prize eligibility:**
- The weights carry a restrictive "Other" model-use licence, which may conflict with the winners' obligation to release models openly (raised in 737950 and 743420).
- Gold-58 was reused for selection.
- Top-10 solutions are audited; a precedent of removal for disallowed external data is cited in 743416 (DeepFake Detection Challenge).

**Our position under the rigor rules:** treat the nartaa checkpoints as **rule-blocked** until the host rules on OAI explicitly. Do not average them into E18/E19, and do not train on OAI.

## 5. What explains 0.945 → 0.949 / 0.950

| Step | Evidence | Size |
|---|---|---|
| Switch base: community stack + reader (0.945) → nartaa single CoAtNet | nartaa's ladder | non-OAI model 0.942 < 0.945, so the switch alone does **not** explain the gain |
| **OAI external supervision** (masked labels: PF OA, Lateral OA, Synovitis) | 0.940 → 0.945, then the 384 full-field OAI recipe 0.948 | **+0.006 to +0.008, the main driver.** Not compute-matched (4,336 → 6,736 optimizer updates), as the author notes. |
| Native 320 crop (no resize) | 0.948 → 0.948 | 0 |
| Anatomical-mirror TTA | 0.948 → 0.949 | +0.001 (one probe) |
| A: + goodpjw reader, per-target rank weights, k 16 | 0.949 → 0.950 | +0.001 (one probe, noise level) |

**Most plausible single cause: training on 2,399 OAI knees.** Everything else is at most 0.001.

## What we can reuse legally, as a single-variable change on our E18/E19 base

These are PROPOSED: none has been run. Each changes one variable, with the parent E19 (or E18) otherwise verbatim.

**1. E21 (PROPOSED) — make the existing `maxspan-v5-reverse` Raptor view a true anatomical mirror.**
- **What changes:** in tri945 cell 30, `run_shared_maxspan` currently scores `windows.flip(1)` for every window at w=0.10. That reverses slice order on coronal and axial slots, which is not a mirror. It is also applied after the ImageNet normalisation, so the R and B channel statistics get swapped.
- **Change:** replace it with N's rule, on the un-normalised windows or by re-normalising after the flip:
  - sagittal windows (centre slice in a sagittal slot of the capacity-aware layout) → `flip(1)`
  - coronal and axial windows → `flip(-1)`
- **Unchanged:** weights and all else.
- **Cost:** zero extra forwards, so no runtime risk.
- **Evidence:** N's mirror gave +0.001 on a weight-1.0 single model. The all-plane width flip was −0.001 on an older model.
- **Expectation:** at w=0.10 inside a stack, very likely invisible at 3 decimals.
- **Success metric:** public LB ≥ the parent, and no `raptor_study_failed` events. Gold-58 is a diagnostic only.
- **Failure modes:**
  - The slot index is wrong when capacities vary per study. The layout is capacity-aware with up to 96 slices, not fixed 18/14/12/8/12, so the slot of each centre must be computed from that study's `capacities`/`quotas`.
  - The run fails a downstream check. Cell 30 has no output-fingerprint gate on the Raptor views; its only fingerprint gate is DINOv2 20/20. So the risk is silent wrong values, not a crash. Compare the mirrored and plain view on the 3 visible studies: the correlation should be high (above 0.9).

**2. E22 (PROPOSED, low priority) — reader `k` 12 → 16 windows per series** on E18's 7-checkpoint reader (cell 37 `infer.py`). This is A's only reader change.
- **Cost:** about +33 % reader time on a run that already takes 6.9–8.7 h wall time. Do it only together with, or after, the E20 runtime fix.
- **Evidence:** none in isolation, because A changed the base at the same time.
- **Success metric:** LB ≥ the parent. An offline check is possible: our f3/f4 val folds at k=12 vs k=16 (val AUC Δ ≥ +0.003 on both folds, per our rule) before spending a submission.

**3. Training-side, not an E18 inference change: add riadmohamed42's public JEV labels to the reader's label blend.**
- This is a label-only ablation in the E16/E17 retrain line.
- nartaa credits it ("Blend + JEV labels" 0.940 → 0.942). That gain is confounded with "Blend", so it is weak evidence.
- Check the licence of the JEV output before use.

**4. Do NOT reuse:**
- **nartaa's checkpoints**, in any role (OAI, section 4). If the host later explicitly permits OAI, the single-variable test would be E19 + N's 384 checkpoint as one extra rank-blend leg with a pre-registered weight. Our team would still have to accept the licence terms.
- **A's per-target weights.** They are gold-58-derived and fitted to a different base, and our rules forbid tuning on gold.
- **N's native-320 crop.** It is specific to N's 384-px corpus. Our Raptor views use dd's 336→384 corpora, where it does not transfer.
- **Raising `K_EVAL`.** Our Raptor views already score all centres (62 of 64, 42 of 44).
