# Public ≥0.941 notebook lineage (RSNA Knee, audited 2026-09-26)

Scope: every public notebook whose `bestPublicScore ≥ 0.941` in the competition's score-sorted
code listing (Kaggle internal list API, 400 notebooks scanned). There are **59**: 1×0.946,
1×0.945, 9×0.943, 9×0.942, 39×0.941. The per-notebook inventory is in
`02_public_0941_plus_inventory.csv`, and the raw sources and metadata are in `raw/`.

Method:
- Pulled every notebook's current source.
- Normalized code lines (comments and whitespace stripped) and computed line-set containment for
  all pairs.
- Grepped component signatures (checkpoint names, manifests, blend constants).
- Compared attached data sources (Kaggle source IDs).
- Read authors' markdown for declared upstreams.

For the ten notebooks whose best score came from an older version, I also pulled that exact version
(`kernels/pull?versionLabel=vN`, saved in `raw/notebooks_best_versions/`) and re-checked it. Two
family assignments changed: `evgendvorkin/rsna-versia-5` and `luonmayman/knee-probe42-cpu-smoke`.
The CSV column `family_at_best_version` is authoritative.

The tenth, `mattiaangeli/bend-the-knee-to-the-dinosaurs`, was first missed because Kaggle's view
model reports v32 as "current" while the pull returns v38. Its scoring v32 stays in F3, but it
already uses the 0.942-era inner weights (A5 .52, Rad .55).

## 1. Headline

**The 59 notebooks are one model ecosystem, not 59 solutions.**
- 56–58 of the 59 attach the same core weight datasets.
- 55 of 59 embed a byte-identical fitted calibrator.
- 44 of 59 fall into 12 near-exact duplicate groups (≤0.5% of normalized lines differ).
- The score steps 0.941 → 0.942 → 0.943 come from re-weighting and adding readers inside that
  one ecosystem.
- The only ≥0.941 notebooks carrying genuinely new model weights are D4 Lite/Blend, and those
  weights are private.

## 2. Effective independent model families (weight level)

| # | Family | Weights | Architecture | Used by | Notes |
|---|---|---|---|---|---|
| M1 | DINOv2-S slot ensemble | `pilkwang/rsna-knee-weights` | DINOv2-small, 6 slots, 336 px, 12 slices, last 6 blocks tuned | 56/59 | **1 config × 5 folds × 4 seeds**, so seed/fold diversity only |
| M2 | "A5" = DINOv3-S/16 slot-token ViT | `mattiaangeli/knee-mri-fold-weights` | ViT-S/16, 16-channel input, slot token | 56/59 (as "DINOv3" in Tony-rank) | same files re-hosted in `tonylica/...dinov3-0917-repro-assets` |
| M3 | RadImageNet ResNet-50 + attention heads | `marwanmath/...`, `prvsiyan/...v52`, `antoinegg1/...`, `sofiaanjenje/e11,e13` | frozen R50, 224 px, 5-fold heads ×2 | 56–59 | the only CNN encoder in the public stack; frozen |
| M4 | CoAtNet MIL ("Raptor") | `dreaddevelopment/raptor-knee-*` | `coatnet_rmlp_2_rw_384` + attention MIL | 58/59 | v5/v8/v10 (+ finespacing v9, widedense, arms-x in a few) |
| M5 | CoAt readers | `mattiaangeli/rsna-knee-coat*` | CoAtNet variants (residual-gated, D4 depth-zone, Global96, Repair-v1) | resgated 58/59; D4 21; G96 15 | **same backbone family as M4**; selected with gold-58 (see §5) |
| M6 | Private fleet (D4 only) | 3 × [Private Datasource] | EfficientNetV2-S (+ CoAtNet, ConvNeXt-L in Blend) | 2/59 | inaccessible |
| m | minor: band32 DINOv2 leg, meniscus bag, finespacing | `ricecakar/...band32`, `renta0426/...meniscus-bag` | DINOv2 / Raptor variants | 1–5 | same families as M1/M4 |

At the final blend of the 0.943 recipe, M4+M5 (CoAtNet) carry about 60–100% of the nominal rank
share per target; M1+M2 carry about 0–19% (see `d4/d4_targetwise_analysis.csv`).

**No public ≥0.941 notebook loads an EfficientNet or ConvNeXt model**, and no public notebook
fine-tunes a CNN end-to-end other than the CoAtNet family.

## 3. Code families (lineage clusters)

| Family | n | Best | Defining recipe | Members |
|---|---|---|---|---|
| F3 BTKD / Dinosaurs (0.941) | 37 | 0.941 | DINO20 + A5(.45) + Rad(.50/.15) + calibrator + Raptor4 (62/42 windows) + resgated CoAt (.40) + per-target outer CoAt weights. Some members, e.g. mattiaangeli v32, already use A5 .52 / Rad .55 without the D4 reader and still score 0.941 | mattiaangeli/bend-the-knee-to-the-dinosaurs (v32) and its forks; maverickss26 0941-restructured forks (ricecakar, valentinonguyen control, beraterolelk, kolyaflexcu, luonmayman probe22 ×2 and probe42-cpu-smoke v1, tsuyu122, nobuwlnter, mtitoadhip, ricardopaizinho, …); renta0426 meniscus-specialist; prvsiyan bee's knees (v27); elshahed/starkhushi p22 |
| F2 0.942 recipe | 8 | 0.942 | F3 + A5 .52, Rad .55/.20, 94 windows, + D4 depth-zone CoAt (2-member rank mix) | maverickss26/rsna-knee-0942-restructured v7; sushanthtiruvaipati rad-fracture-oof; valentinonguyen radfracture942 / dinov9-routing; kolyaflexcu speedy947; nobuwlnter e1; kjy666 (v3); **evgendvorkin versia-5 v16** (scoring version; line-identical to the maverick 0.942 code) |
| F1b 3-way CoAt | 2 | 0.943 | F2 + Global96 reader | maverickss26/restructured-version-3; hyakumanben2025/…-speedy-raptors |
| F1a 4-way CoAt (0.943) | 7 | 0.943 | F2 + Global96 + Repair-v1, rank of mean probabilities | mattiaangeli speedy-raptors-the-original v39; jiweiliu fast-2xt4 v11; haideptry; romantamrazov dinosaur-v5 (v15); aastikrajan15; gwey1025; hyakumanben2025 v2. (evgendvorkin's *current* v17 is a Jiwei copy, but it has not produced a better score) |
| F4 Tony-rank bundle | 3 | 0.942 | same public weights re-hosted in one bundle; DINOv2 .55 / "DINOv3"(=A5) .45; Raptor v9+v10; up to 4 CoAt readers | andreluizpedroso tony-rank (v29); akhyar2612; valentinonguyen knee-sub-tonyrank |
| F0 D4 | 2 | 0.946 | F1a (Jiwei v11) + Fracture-in-Rad + private leg at 0.45 | pjmathematician d4-lite, d4-blend |

Declared upstreams (from markdown) match the code containment:
- Jiwei declares maverickss26 0941-restructured and Mattia speedy-raptors v34.
- sushanth declares maverick 0942-restructured.
- Most F3 forks declare renta0426 / anvithpothula / mattiaangeli.

Kaggle's API does not expose a fork-parent field, so lineage here is established from code and
declared links.

### Near-exact duplicate groups (mutual containment ≥ 0.995)
- [8] mattiaangeli bend-the-knee-to-the-dinosaurs, hyakumanben2025 live-repro-v2, yamadan96 bk1/ar2/fh3/subg, valentinonguyen ar2combo, rafanikitas — all 0.941
- [7] beraterolelk ×2, kolyaflexcu plateau-base/raptor-v9, rushmanthnalluri, luonmayman probe22 ×2 — all 0.941
- [4] starkhushi why-public-forks-stop-at-0-941, elshahed stark-p22 ×3 — 0.941
- [4] mattiaangeli speedy-raptors-the-original, aastikrajan15, hyakumanben2025 v2, gwey1025 — 0.943
- [4] maverickss26 0942-restructured, sushanthtiruvaipati, valentinonguyen radfracture942, kolyaflexcu speedy947 — 0.942
- [3] jiweiliu fast-2xt4, haideptry …-d4-0943, evgendvorkin versia-5 (current) — 0.943/0.943/0.942
- [3] renta0426 meniscus-specialist, vaihero rentafork/rentanorev — 0.941
- [3] ricecakar control/dino2pf, valentinonguyen control — 0.941
- [2] tsuyu122 audit, nobuwlnter fast-2xt4-original — 0.941
- [2] ayodejiibrahimlateef a01, apilkhadka a01 — 0.941
- [2] valentinonguyen band32w08/w15 — 0.941
- [2] pjmathematician d4-lite/d4-blend — 0.945/0.946

## 4. The named notebooks you asked about

| Notebook | Family | Independent? |
|---|---|---|
| rsna-knee-0942-restructured (maverickss26 v7) | F2 anchor | No: public weights, re-weighted |
| RSNA Baseline by evgendvorkin | The ≥0.941 entry is `rsna-versia-5`. Its 0.942 came from v16, which is line-identical to the maverick 0.942 family (F2). Current v17 is line-identical to Jiwei v11 (F1a) | No |
| RSNA Knee Fast 2xT4 Inference (Jiwei v11) | F1a | No. It is the **strongest fully public and reproducible** notebook (0.943; best = current version) |
| The bee's knees — final RSNA push (prvsiyan, best v27) | F3 | No: same weight datasets |
| historical rsna-knee-tony-rank-ensemble | F4 | No: re-hosted identical files; different blend weights |

## 5. Things every ≥0.941 notebook shares that matter for private-LB risk

1. **Gold-58 has already been used for selection.**
   - CoAt reader manifests carry `gold58_rank_ensemble_auc` as the selection field.
   - One dataset is named "Global96 **Gold** Top3".
   - The resgated bundle includes `coat_resgated_gold_e04.pt` and gold-58 portability JSONs.

   A gold-58 score of the public stack is therefore **optimistically biased** for those components.
2. **Per-target outer weights were chosen on the public LB.** maverick's own comment on
   Lateral Meniscus = 1.00: "on the strength of a public split … the likeliest place to give back
   points privately."
3. **The fitted calibrator (N = 4349 = 4407 − 58) is identical in 55/59.** Its targets are
   report-derived, so every notebook inherits the same label-noise-trained correction.
4. **Report-label AUC ≠ LB AUC.** DINO members' `holdout` AUC is 0.83–0.86 against report labels,
   while the blends score about 0.94 on LB. Report-label OOF is useful for *ranking* candidates, but
   its absolute level is not comparable to LB.
5. **No OOF is available for most public components.**
   - Raptor checkpoints are "full" fits.
   - CoAt readers are gold-selected.
   - The DINO fold split is not published in the manifest.

   **Consequence:** OOF-learned blend weights between the public anchor and our models are not
   possible on clean OOF. Any anchor weight must be a small number of pre-registered values.

## 6. Implication for ensembling

Averaging several ≥0.941 public notebooks re-weights the same five model families. The data give no
reason to expect meaningful diversity from it, and it adds public-LB overfitting risk (weights
tuned on public splits). **Count all 57 non-D4 notebooks as one effective family (plus M1–M5
internally).** The only demonstrated route above 0.943 in public is D4's: blend the 0.943 stack with
an independently trained non-CoAtNet/non-DINO model at a large rank weight.

## 7. What could not be established

- Hidden-test runtimes: not exposed. Only the commit-run times on 3 visible studies are in the CSV.
- Notebook licenses: not exposed by the API.
- Fork-parent pointers: not exposed; lineage inferred from code and declared links.
- Near-duplicate groups (§3) and the containment matrix use *current* versions. Best-scoring
  versions were checked separately (`raw/notebooks_best_versions/`).
- Internals of the CoAt reader runtimes: they are `.py` files inside datasets, not in the notebooks.
