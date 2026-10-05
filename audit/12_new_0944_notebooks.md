# 12 — Three new public 0.944 notebooks (analysed 2026-10-05)

Sources were pulled at the **scored** version, resolved through `GetKernelViewModel.bestSubmissionScore`, and saved in `public_notebooks/`. Rows were appended to `_index.csv`. Nothing was run or pushed on Kaggle.

| # | Notebook (file in `public_notebooks/`) | Scored version | Current version |
|---|---|---|---|
| G | `0.944_82v_goodpjw2008__rsna-knee-stack-2-5d-convnext-mil-lb-0-944.ipynb` | **v2** (scriptVersionId 355300588) | v4 |
| M | `0.944_5v_medvax__rsna-knee-independent-2-5d-reader-trial.ipynb` | v1 (355354994) | v1 |
| B | `0.944_1v_bliverrigpdoge__rsna-knee-raptor-student-blend-lb-0-944.ipynb` | v1 (355268991) | v1 |

Notes on G's versions:
- v1 and v2 are byte-identical sources.
- v3 and v4 change only markdown, plus an unused `img_size` argument for ViT backbones (cells 35–36 in v4). Predictions are identical.

## Bottom line

- **G and M: the leg is PUBLIC and mountable.**
  - Leg: 2.5D ConvNeXt-Tiny with a study transformer, at 0.30 rank weight.
  - Weights: `goodpjw2008/rsna-knee-2-5d-convnext-reader` (v1, Apache-2.0, HTTP 200, 380 MB). It holds 3 fold checkpoints plus inference code **and** the training code (`train.py`, `make_labels.py`, `oof.py`).
  - M is a third party's reproduction with the same three checkpoints (sha256-pinned), and it also scored 0.944. So the public weights are verified to give 0.944 on top of our exact anchor.
- **B: the leg is PRIVATE.**
  - Leg: a CoAtNet-RMLP-2 "Raptor student" at 0.45 rank weight.
  - Its code and weights live in `bliverrigpdoge/rsna-knee-leg-r9all`, which returns HTTP 403 and shows as `private-data-source` in the view model.
  - We cannot mount it, and its training and inference code are not public. Only its starting checkpoint is public (dd `raptor-knee-finespacing/raptor_ft_coatnet_v9_full.pt`, CC0).

## Comparison table

| | G goodpjw2008 | M medvax | B bliverrigpdoge | Ours E11 / E03b |
|---|---|---|---|---|
| Public base | D4 public part = Jiwei v11 + Fracture flip (`_RAD_EXCLUDE=("Baker's",)`), identical to our E02 anchor (0.943) | same as G | RSNA_Versia_5 v18 code = Jiwei v11 verbatim, **without** the Fracture flip (`("Baker's",'Fracture')`) | E02 anchor |
| Base reuse | all 23 code cells hash-equal to D4-lite cells 2–29 (G cells 5–32; cell 28 = D4 cell 25) | same (M cells 5–32) | all 23 code cells hash-equal to Jiwei v11 (B cells 2–46, even indices); markdown headers rewritten | — |
| Leg architecture | ConvNeXt-Tiny (timm, ImageNet) on 3-slice windows → tokens + slot/position embeddings → 2-layer transformer (dim 384) → 12 per-finding attention pools | same weights | CoAtNet-RMLP-2 (Raptor architecture), fine-tuned from dd v9 "Fine Spacing" | EffNetV2-S in dd Raptor MIL recipe |
| Leg input | **Own DICOM canonicalisation** from the competition files. Details below the table. | same | 80 slices per study over 2–98 % of each series, 140 mm crop at 336 px, every window scored (cell 0 claim only) | dd corpus stacks, 44×336, 140 mm, k_eval 24 |
| Labels | Mean of 4 public LLM tables. Gold-58 excluded. Details below the table. | same | dd soft labels (cell 0) | E11: dd soft; E03b: team consensus (4 gold-selected sources) |
| Folds / seeds | 5-fold GroupKFold by normalised report text, seed 0. **3 of 5 folds** shipped. 1 seed. | 3 | unknown | 2 folds (0+1), 1 seed |
| Reported CV | 3-fold OOF vs report labels >0.5: **0.888** (2,610 studies); gold-58: **0.911** (cell 1 figure, per class) | — | none | E11 OOF 0.872 / gold 0.900; E03b gold 0.883 |
| Leg alone LB | **0.929** (3 folds; cell 0, dataset description) | — | **0.942** (cell 0) | 0.896 (E11a), 0.899 (E03b) |
| Blend | rank-pct, one global w=0.30, re-ranked (cell 38) | identical formula, hard-coded (cell 38) | rank-pct per finding, w=0.45, no re-rank (cells 48–49) | rank w=0.30 → **0.935** |
| How w was chosen | "chosen before submitting, not LB-tuned" (cells 0, 38) | copied from G | `leg_weight_for_score()`: step function of the **leg's own public LB** (≥0.93 → 0.45) | pre-registered 0.30 |
| TTA | none (centre crop, `fov=0.92`) | none | unknown | — |
| Fallback | try/except, keeps stack on failure | **none**: asserts are fatal | stash/restore; time-gated | — |
| Training code public | **yes** (in the dataset) | n/a | **no** | ours |
| Weights mountable by us | **YES** | **YES** (same dataset) | **NO** (403) | — |

Details for the two longest cells:
- **G's leg input:** 0.4 mm/px; 384 px canvas (153.6 mm FOV); canonical in-plane orientation from IOP; slices sorted along the normal; ≤64 slices per series; per-series 1–99.5 % window. There are 6 slots (3 planes × fat-sat yes/no), and the largest series wins each slot. Each series contributes 12 windows of 3 adjacent slices. Input is **256 px** at fov 0.92, about 141 mm, about 0.55 mm/px.
- **G's labels:** the 4 public tables are stevenleehans `llm_labels_v4_blend`, pilkwang `report_labels_v2`, lixin73 `gpt56sol` and laymond qwen3-8b. They are combined as a mean, clipped to [0.01, 0.99], and used as soft BCE targets. Gold-58 is never trained on.

## Per notebook

### G — goodpjw2008 "Stack + 2.5D ConvNeXt MIL" (v2)

**a) Base.**
- Cells 3–32 ("Part A") are, by the author's account, skarin's reproduction of the 0.943 stack.
- By code hash, they are exactly D4-lite cells 2–29, i.e. Jiwei v11 with the Fracture flip. This is the same code as our E02 anchor (0.943).
- Every data source matches the anchor's set, plus the reader dataset.

**b) Own leg** (cells 0–1 describe it; cells 35–37 write `preprocess.py`, `knee.py` and `infer.py`; the dataset adds `train.py`, `make_labels.py` and `oof.py`).
- Training recipe in `train.py`:
  - 14 epochs at 256 px, bs 8, k=8 windows per series in training and 12 at eval.
  - AdamW (encoder 1e-4, head 5e-4, wd 0.02), 1-epoch warmup then cosine.
  - bf16 autocast, grad-clip 3, EMA 0.998 (the EMA weights are saved).
  - Augmentation: rotation ±12°, scale ±12 %, shift ±6 %, gamma/gain jitter shared per series.
- It was trained off-Kaggle: `/mnt/ssd`, an 18-process pool and bf16 imply an Ampere-class GPU.
- Inference: the mean of sigmoid outputs over the 3 checkpoints. The model reads raw test DICOMs directly, so there is no cache dependency.

**c) Sources.**
- The reader comes from `goodpjw2008/rsna-knee-2-5d-convnext-reader` (public, Apache-2.0, 692 downloads).
- Everything else is the public stack's sources, which our anchor already mounts.
- `pylibjpeg*` wheels are inside the dataset. timm comes from the stack's wheel.

**d) Blend.**
- Formula: `rank_pct(0.7·rank_pct(stack) + 0.3·rank_pct(reader))`, global across findings, with no per-target weights and no TTA.
- The weight is claimed to be fixed before submission. The author also submitted the reader alone, which scored 0.929.

**e) Risks and runtime.**
- The author reports 6.5–8 h of the 9 h budget for the full hidden run. The stack dominates; the reader alone took under 90 min with 5 checkpoints.
- Rules:
  - The labels are derived from `train.csv` reports and public label datasets, which is fine.
  - Gold-58 is used only as a check, but the per-class gold AUCs are published. Selection on gold cannot be ruled out, though the label sources were not gold-selected.
- The leg was not LB-tuned, but the stack's constants were (the author warns about this in cell 0).

### M — medvax "Independent 2.5D Reader Trial" (v1)

**a, b, c)**
- M is a self-declared **attributed reproduction** of G (cell 0): "MedvAx did not train these weights".
- It uses the same base cells 5–32 and the same three checkpoints, sha256-pinned in `READER_HASHES`, cell 38.
- Its data sources are identical to G's. No new model is involved.
- The changes are hardening only:
  - `weights_only=True` (cell 37);
  - asserts on DICOM coverage per study (cell 37);
  - asserts on the stack's internal state: `_DINOV2_MATCHED_MEMBERS==20` and `V18_CALIBRATOR_APPLIED` (cell 38);
  - a receipt JSON.

**d)** The blend formula is identical to G's: 0.70/0.30 rank, re-ranked.

**e)**
- Higher submission-error risk: there is no try/except, so any reader failure fails the whole submission instead of falling back to 0.943.
- Value to us: an **independent confirmation that the public checkpoints reproduce 0.944**. That matters because G's text says its own 0.944 came from a "private run". G v2 itself also scored 0.944.

### B — bliverrigpdoge "Raptor student blend" (v1)

**a) Base.**
- Cell 0 claims the base is RSNA_Versia_5. The code cells are identical to Versia-5 v18 (scored 0.943), which in turn is Jiwei v11 code verbatim.
- So B keeps `_RAD_EXCLUDE=("Baker's",'Fracture')` (cell 42) and does **not** include the D4 Fracture flip that our anchor and G use.

**b) Own leg.**
- Everything known comes from markdown cell 0. The code (`leg_main.py`) is inside the private dataset.
- Architecture: CoAtNet-RMLP-2, fine-tuned from dd's public Fine Spacing Raptor checkpoint (`raptor_ft_coatnet_v9_full.pt`).
- Input: 80 slices per study over 2–98 % depth, 140 mm crop at 336 px, all windows scored.
- Labels: dd soft labels.
- Leg alone: LB 0.942.
- Unknown: number of folds and seeds, CV/OOF, gold numbers, and whether it is distilled from Raptor (the word "student" suggests it).
- Note that v9 is not among the stack's Raptor arms (v5, v10, v8), so this is a new member of the *same* CoAtNet/Raptor family.

**c) Sources.**
- The leg comes from `bliverrigpdoge/rsna-knee-leg-r9all` (cell 49 `LEG_DATASET`), which is PRIVATE (403).
- The author has no public datasets and no other public kernels.
- Everything else is Versia-5's public set (DINOv2-base model attached, unused by the leg).

**d) Blend.**
- Cell 48: per-finding `(1−w)·rank01(public) + w·rank01(ours)` with w=0.45 and no final re-rank.
- w was picked by `leg_weight_for_score(leg's own public LB)`. That is LB-informed, though not a grid search on the blend LB.
- Studies the leg did not score keep their pipeline rank, provided coverage is ≥ 50 %.

**e) Risks and runtime.**
- The leg starts only if elapsed time is < 7.75 h and is killed at 8.5 h (cell 49). On a slow hidden run it silently falls back to Jiwei v11, i.e. 0.943.
- A partial-coverage blend mixes blended and unblended ranks.
- Weight selection depends on the LB.
- The leg cannot be audited: there is no code and no data.

## f) Why their legs add +0.001 while ours lost 0.008

1. **Leg strength relative to the anchor.**
   - The legs sit at −0.014 (G, 0.929) and −0.001 (B, 0.942) from the anchor. Ours sits at −0.044 (0.899).
   - At w=0.30, a rank blend with a leg 0.044 weaker mostly injects noise. That is why our E11 @0.30 scored 0.935.
   - Empirically, a +0.001 gain needs a leg within about 0.015 of the stack, or a much smaller weight.
2. **Diversity (G).**
   - Every part of G's leg is independent of the stack:
     - preprocessing: its own DICOM canonicalisation (0.4 mm/px, 154 mm FOV, IOP-based orientation, all six plane × fat-sat slots including non-FS series);
     - backbone family: ConvNeXt;
     - head: a study-level transformer;
     - labels: 4 LLM tables, mean, not dd.
   - Ours used dd's corpus and stack geometry (44×336 stacks, 140 mm), dd's Raptor recipe and MIL head, and (in E11) dd labels. That is the same lineage as the stack's Raptor arm, which already carries most of the stack's weight. E11v concluded that ours adds little diversity.
3. **Raw quality.**
   - G's gold-58 is 0.911 (ours 0.883–0.900), from 3 folds with EMA (ours 2 folds with SWA).
   - G's labels are individually *weaker* than our consensus sources (gold 0.82–0.89 per source). So labels do not explain the gap; model and input do.
   - Plausible contributors:
     - EMA;
     - 12 windows from every slot (up to 72 tokens) with cross-series attention;
     - fat-sat-aware slots;
     - a ConvNeXt-T at 256 px over about 141 mm, a coarser view that is cheaper and adds a different scale.
4. **B is the opposite route.**
   - B adds a same-family model (another Raptor CoAtNet), but one that is almost as strong as the whole stack.
   - So low diversity is tolerable when the leg is at stack level. Ours was neither diverse nor strong.

## What we can reuse legally

1. **Mount G's reader as-is.**
   - The dataset `goodpjw2008/rsna-knee-2-5d-convnext-reader` is public, Apache-2.0 and available to all participants.
   - Anchor (E02 code = G Part A) + reader @0.30 is exactly G/M, verified at 0.944 twice. This would become the new reproducible anchor.
   - The pin is dataset v1, and the three checkpoint sha256 values are in M's cell 38.
   - Credit goodpjw2008 if the notebook is published.
2. **Retrain G's recipe ourselves.**
   - Use their Apache-2.0 `preprocess.py`, `knee.py`, `train.py` and `make_labels.py`.
   - Candidate single-variable experiments:
     - (a) folds 3–4 to complete 5 folds, with the identical recipe and labels;
     - (b) the same recipe with our consensus labels;
     - (c) a second seed.
   - Caveats:
     - Kaggle T4s have no native bf16, so switch to fp16 with GradScaler (that is itself a variable; keep it for every arm).
     - Their DICOM cache step uses 18 processes; plan the CPU and disk budget.
     - Their folds (GroupKFold, seed 0, by report text) differ from our `folds_v1`, so OOF is not comparable across the two fold files.
3. **dd's Fine Spacing checkpoint** (`dreaddevelopment/raptor-knee-finespacing/raptor_ft_coatnet_v9_full.pt`, CC0, public) is B's starting point and is not used by the stack.
   - Two options: add it inference-only as an extra Raptor view, or fine-tune it as B did.
   - Any weight for it must be pre-registered. Do not copy B's LB-derived 0.45.
4. **Not reusable:** B's leg (private dataset, private code) and the code-free claims in B's cell 0.
5. **Base choice:** G and M sit on our exact anchor code (with the Fracture flip); B does not. Keep the E02 base.

Pre-registration reminder (project rigor rules): change one variable per submission, use gold-58 only as a guard, and do not tune weights on the LB. "Anchor + G reader @0.30" is a pure reproduction, so it needs no tuning.
