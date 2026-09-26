# Next 10 experiments (all PROPOSED — none has been run)

## 0. Gaps

**Our solution.** There is none yet.
- The project folder is empty, so there is no code, labels, OOF, cache or checkpoints.
- Local Python is blocked (Xcode license) and there is no GPU.
- Everything below therefore runs as Kaggle notebooks. That is an outward action on your account,
  so I will not push or submit without your go-ahead.

**The public ecosystem.** These gaps are opportunities for us.
1. **One ecosystem.**
   - 57 of the 59 ≥0.941 notebooks re-weight the same five model families.
   - At the final blend, ~60–100% of each column is CoAtNet (`coatnet_rmlp_2_rw_384`).
   - No public top notebook loads an EfficientNet or ConvNeXt model.
2. **The only step above 0.943 has the same shape.** D4 adds a 45%-weight independent CNN leg
   (EfficientNetV2-S). Its weights are private, so we have to build our own.
3. **Inconsistent preprocessing inside the public stack.**
   - Four slice-ordering rules and four laterality policies.
   - The dominant CoAtNet branch has **no laterality normalization**, which matters for medial vs
     lateral targets.
   - All crops are image-centred.
4. **Label monoculture.** Public members train on report-derived labels, and the attached
   `pilkwang/rsna-knee-llm-labels` is the common one. Label sources disagree up to 2× on Fracture
   prevalence. Nobody public uses an uncertainty mask.
5. **No clean OOF for the anchor.**
   - Raptor checkpoints are full fits.
   - CoAt readers were gold-58-selected.
   - The DINO fold split is unpublished.

   So the anchor↔ours weight cannot be learned on OOF. It must be pre-registered.

## 1. Decision rules (the protocol every experiment below uses)

**Labels.**
- Consensus report labels from E03 with a positive / negative / uncertain state.
- Uncertain targets are masked out of both the loss and the metric.

**Gold-58.**
- Excluded from training and from fold assignment.
- Scored once per experiment as a diagnostic with bootstrap CIs.
- Never used to choose hyperparameters or prompts.

**Folds.**
- 5 folds over the 4,349 non-gold studies.
- Grouped by study, and additionally by normalized-report hash (near-duplicate reports cannot cross
  folds).
- Stratified on a multi-label signature.
- Frozen in `folds_v1.csv` before any training.

**Primary metric.** OOF macro AUC over 12 targets, against consensus labels, on non-masked entries.

**Secondary metrics.**
- Per-target AUC.
- Fold std.
- Seed std (when ≥2 seeds).
- Gold-58 macro and per-target (with CI).
- Rank correlation vs reference models.

**Acceptance of a change.** All of the following must hold:
- ΔOOF-macro ≥ +0.003 with a paired study-level bootstrap 90% CI excluding 0, **or** Δ positive in
  ≥4/5 folds when run 5-fold.
- No target drops by more than 0.01 OOF.
- Gold-58 does not drop beyond its CI.

Anything smaller is recorded as "no evidence of change".

**LB.** Public LB is a sanity check. It never selects between candidates that OOF cannot separate.

**One variable per experiment.** Each row of `07_experiments.csv` names its single changed variable.

**Compute.** About 26 days and roughly 30 GPU-h/week means about 110 GPU-h total. Single-fold
screening first; 5-fold only for survivors.

## 2. The first five

### E01 — Pin and reproduce the strongest *reproducible* public anchor (Jiwei v11, 0.943)
- **Why:** every later comparison needs a fixed, working reference. D4 Lite cannot be the anchor,
  because 45% of it is private.
- **Evidence:** Jiwei v11 is the best-scoring version (current = best), is fully public, and is
  line-identical to two other 0.943/0.942 notebooks.
- **Exact experiment:**
  - Fork Jiwei v11 unchanged with all 19 data sources pinned to the same version IDs
    (`d4/d4_dependencies.csv` lists them).
  - Commit, then submit once.
  - Record runtime, peak memory (from its `diagnostics/*.json`), model count, TTA passes and the
    `submission.csv` SHA256.
- **Success:** LB = 0.943, the receipts show 0 fallback studies and all 4 CoAt readers, and it
  completes under 9 h.
- **Could fail:** dataset version drift, pinned wheels (timm 1.0.22, opencv 4.12.0.88), Kaggle
  image changes.

### E02 — D4 private-leg removal test (the only runnable D4 ablation)
- **Why:** it answers "what produced 0.943 → 0.945?" with the only data available.
- **Evidence:** D4 Lite differs from E01 only by the Fracture token plus cells 30–32. A plain fork
  crashes in cell 31 without the private data.
- **Exact experiment:** fork D4 Lite, delete cells 30–32, submit once. This equals
  "Jiwei v11 + Fracture flip".
- **Reading the result:**
  - E02 − E01 = Fracture-flip effect.
  - 0.945 − E02 = private-leg effect.
  - Both are LB-only, labelled as diagnostics, and never used for selection.
- **Could fail:** a displayed difference of 0.000/0.001 is inside rounding, and the result may be
  uninformative. That is acceptable at 1 submission.

### E03 — Label-source audit and consensus v1 (CPU only, no training)
- **Why:** label noise is the single biggest limiter. Sources disagree 2× on Fracture, and
  report-label vs gold disagreement drove the public Fracture debate.
- **Evidence:** the per-source Fracture numbers in `01_competition_analysis.md`; the member
  holdout-AUC ceiling of about 0.84.
- **Exact experiment:**
  - Download the 5 public label sets and write a transparent rule-based extractor v0
    (negation/uncertainty cues) from the report text in `train.csv` (column names to be confirmed
    from the file).
  - Per target, per source: agreement with gold-58 (sensitivity, specificity, κ; Wilson CIs) and
    pairwise inter-source κ on the 4,349 studies.
  - Build a consensus with states positive, negative and uncertain, plus a soft confidence.
  - List every report-vs-gold disagreement for manual review.
- **Success:**
  - The consensus is not worse than the best single source on any target beyond CI on gold-58
    (measured **once**).
  - The uncertain fraction is reported per target.
- **Could fail:** gold-58 is too small to separate sources, in which case we fall back to
  majority-vote plus uncertainty and document it.

### E04 — DICOM geometry, laterality and slot audit (CPU only)
- **Why:** the public stack uses 4 ordering and 4 laterality rules. We need one correct rule set
  before training anything.
- **Evidence:** `d4/d4_lite_full_architecture.md` §5.
- **Exact experiment:** header pass over all train and visible test series. Measure:
  - ordering agreement (IPP·normal vs InstanceNumber vs filename);
  - IOP-derived plane vs `Anatomical_Plane`;
  - `Laterality` tag vs geometric side;
  - fat-sat/fluid flags vs header regex;
  - spacing, thickness, matrix, vendor and field-strength distributions;
  - missing-slot rates.

  Then render the exact model-input grid per slot for 30 random studies and all 58 gold studies.
- **Success:** a written preprocessing spec with measured error rates for each rule.
- **Could fail:** header reads may be slow on Kaggle CPU. We sample and extrapolate if needed.

### E05 — Controlled backbone screen: DINOv2-S vs EfficientNetV2-S (single fold)
- **Why:** we need an independent model that is decorrelated from the public stack. The only
  public evidence above 0.943 (D4) points to an EfficientNetV2-S-class CNN.
- **Evidence:** D4's private leg (architecture from its log). There is no public EfficientNet
  anywhere in the ≥0.941 set.
- **Exact experiment:** two runs, identical except the backbone.
  - Shared settings: E03 consensus labels (masked BCE), E04 preprocessing, 5 slots (Sag FS, Sag
    non-FS, Cor FS, Cor non-FS, Ax FS), 2.5D triplets, 288 px, 140 mm crop, attention-MIL over
    slices then per-target slot attention, fold 0, seed 0, no TTA.
  - Backbones: `facebook/dinov2-small` vs `tf_efficientnetv2_s.in21k_ft_in1k`.
- **Metrics:** fold-0 OOF macro AUC (about 870 studies), per-target AUC, gold-58 diagnostic, and
  Spearman between the two models' fold-0 predictions.
- **Decision:** EfficientNetV2-S advances to 5-fold if its OOF macro is ≥ DINOv2-S − 0.005 (it
  does not need to win) and its correlation to DINOv2-S is clearly below the DINOv2-S seed-to-seed
  level.
- **Could fail:** one fold is noisy (CI about ±0.01), in which case we add a second fold before
  deciding. The preprocessing cache may exceed the 20 GB `/kaggle/working` limit; plan a 224–288 px
  uint8 cache split into Kaggle datasets.

## 3. The next five

| ID | Hypothesis (one variable) | Why / evidence | Success metric | Could fail because |
|---|---|---|---|---|
| E06 | Laterality normalization ON vs OFF (winner of E05) | CoAtNet branch has none; medial/lateral targets depend on side | Δ OOF on {Med/Lat Meniscus, Med/Lat OA, MCL} ≥ +0.003, others ≥ −0.002 | CNNs may learn side from context; tag/geometry errors (measured in E04) |
| E07 | Uncertain-masked + confidence-weighted BCE vs "silence = negative" BCE | label noise; uncertain states from E03 | Δ OOF-macro ≥ +0.003 on the *same* masked evaluation set | masking removes hard cases; less data for rare targets |
| E08 | Crop 130 mm vs 150 mm at 288 px | public uses 130/140 without reported ablation; small structures vs context trade-off | Δ OOF-macro ≥ +0.003 | image-centred crop may cut anatomy regardless |
| E09 | 288 px vs 384 px | D4's private leg uses 384; the cost is ~1.8× | Δ OOF ≥ +0.003 **per unit runtime** is worth it | gain may not survive the runtime budget |
| E10 | 5-fold × 2 seeds of the E05–E09 winner, then an anchor blend | the only way to get a trustworthy independent leg | OOF macro, fold/seed std, gold-58; blend: pre-registered rank weights w ∈ {0.30, 0.45} for ours vs E01 anchor, at most 2 submissions; keep the lower w unless the higher w is ≥ the lower on LB **and** gold-58 | the anchor's gold-58 is biased (gold-selected components); LB noise; runtime of anchor + our leg under 9 h |

## 4. Your E-D4-1…6: feasibility map

| ID | Request | Feasible? | How |
|---|---|---|---|
| E-D4-1 | D4 Lite alone | observed only | LB 0.945; no OOF, predictions or weights available |
| E-D4-2 | Existing 0.942 anchor alone | yes (use 0.943) | E01 |
| E-D4-3 | 50/50 rank blend of D4 Lite and anchor | **no** | we cannot obtain D4 Lite's hidden-test predictions or its private weights |
| E-D4-4 | OOF-optimized constrained blend | **no, for D4**; yes for our own models | the anchor has no clean OOF; D4 has none at all |
| E-D4-5 | D4 + a genuinely non-DINO model | **reframed** | E05→E10: our own independent CNN blended with the E01 anchor, which is D4's recipe with components we control |
| E-D4-6 | Remove each Blend constituent | only "minus private leg" | E02 (S_dist/S_coat/S_cnxl are private) |

## 5. Explicitly not planned

- Averaging multiple public ≥0.941 notebooks (same weights, no diversity).
- Per-target weights tuned on public LB.
- Reproducing D4 Blend.
- Any use of the private D4 datasets.
