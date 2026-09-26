# D4 Lite — full architecture (from source, not from the title)

Source: `pjmathematician/rsna-knee-d4-lite`, version 1 (the only version; best score 0.945 was
achieved by v1, scriptVersionId 352859223). Pulled 2026-09-26 via Kaggle API. Code copy:
`audit/raw/notebooks/pjmathematician__rsna-knee-d4-lite.py`. Commit-run log:
`audit/raw/logs/pjmathematician__rsna-knee-d4-lite.commit_run.log`.

Everything below is read from code or from the commit-run log. Items marked **UNKNOWN** are
inside private datasets or inside dataset-hosted runtimes that the notebook only calls.

## 0. What "D4" refers to in the code

The string "D4" in the code refers to an existing public component,
`mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2` ("CoAtNet D4 Depth-Zone SWA3"). Its
serving receipt has `slice_depth_arm: d4_zones` and `head: rank8_swa2_plus_d4_three_zone_expanded_fsx`.
That component was already in the public 0.942 recipe (Jiwei changelog v7) and the 0.943 recipe.
**D4 Lite did not introduce it.** Whether the notebook title refers to it is not stated anywhere
in the notebook, so I don't assume it.

## 1. Top-level structure

```
cells 2-29  = jiweiliu/rsna-knee-fast-2xt4-inference v11 (public 0.943), with ONE token changed
cell  30    = stash that public output as _public_probe22.csv
cell  31    = run a PRIVATE "student fleet" model (arm S_dist) from private datasets
cell  32    = final = rank( 0.55 * rank(public) + 0.45 * rank(private) )
```

Code-level evidence:
- 100% of Jiwei v11's normalized code lines are contained in D4 Lite. 96.7% of D4 Lite's lines are
  in Jiwei v11. The remainder is cells 30–32. See `diff_jiwei_v11_vs_d4_lite.txt`.
- The only change inside cells 2–29 is at line 3003 of the diff:
  `_RAD_EXCLUDE = ("Baker's", 'Fracture')` → `("Baker's",)`.
  Jiwei's rendered versions v7 through v11 all carry the Fracture exclusion.
- The commit run printed the same `submission.csv` SHA256 (`8f0c5e7b…`) **before** the private leg
  for D4 Lite, D4 Blend and Jiwei v11. That run used only 3 visible test studies, so this does
  **not** prove the outputs are equal on the hidden test (the Fracture flip can change hidden
  rankings).

## 2. Public part (stages 1–5), inherited from Jiwei v11

### Stage 1 — DINOv2-S slot ensemble (20 members)
| Item | Value |
|---|---|
| Weights | `pilkwang/rsna-knee-weights`: 20 members = 1 config × 5 folds × 4 seeds (`base-s336x12`, `-seed2/3/4`) |
| Backbone | `facebook/dinov2-small` (HF), last 6 blocks fine-tuned, frozen 6-block prefix shared at inference |
| Series selection | 6 slots `SAG_FLUID_FS, COR_FLUID_FS, AX_FLUID_FS, SAG_FLUID_NOFS, COR_T1, SAG_T1`. Plane from `test_series.csv`; fat-sat/weighting from header regex + ScanOptions; ties go to the series with the most slices |
| Slice ordering | projection of IPP on slice normal (cross of IOP rows/cols); per-file InstanceNumber fallback; if any key missing → natural filename sort |
| Laterality | DICOM `Laterality` tag, else geometric image-centre x (median, ±20 mm dead zone); R knees flipped (cor/ax: width axis; sag: slice order reversed) |
| Pixels | band 0.2–0.8 of stack, 12 slices, 130 mm centred square crop (image centre, not anatomy), 1–99 percentile, bilinear → 336 px |
| Model input | 3 contiguous slices as RGB per slot; features = CLS + mean patch; SlotHead = per-target query attention over 6 slots |
| TTA / pooling | 10 overlapping 3-slice windows; per-target window pooling: max (Fracture, Contusion, Med/Lat Meniscus, Baker's), top-2 mean (ACL, MCL), mean (others). No jitter in retained output |
| Member fusion | each member → percentile rank; equal rank mean over 20 members; re-rank |
| Member quality (manifest) | per-fold `holdout` AUC 0.828–0.860; `annot` 0.736–0.916 (per fold, small-n) |

### Stage 2 — "A5" = DINOv3-S/16 slot-token ViT, 5 folds
| Item | Value |
|---|---|
| Weights | `mattiaangeli/knee-mri-fold-weights` `m_f0..4.pt` (94,000,039 B each; byte-size-identical files re-hosted in `tonylica/...dinov3-0917-repro-assets`, whose notebook calls them "DINOv3-small fold checkpoints") |
| Series selection | 6 slots (Sag/Cor/Ax × FS/non-FS) from `test_series.csv` `Fat_Suppression`; first matching row |
| Slice ordering | **InstanceNumber only**; unreadable files dropped |
| Laterality | **only non-sagittal**, flipped if IPP[0] of the *first* file < 0; no tag use; sagittal never normalized |
| Pixels | band 0.12–0.88, 16 slices, 130 mm crop, per-slice 1–99 percentile, INTER_AREA → 336 px |
| Precision | bf16 autocast on T4 |
| Fusion | per-fold rank01, mean over folds; `0.48·rank(DINO) + 0.52·rank(A5)` for studies with ≥1 acquired A5 slot (base only otherwise) |

### Stage 3 — RadImageNet ResNet-50 heads + calibrator
| Item | Value |
|---|---|
| Encoder | `marwanmath/resnet-50-radimagenet-marwan` ResNet50.pt, frozen, GAP 2048-d per slice |
| Heads | E10 reference heads (5 folds, `prvsiyan/…v52…`), E13 heads (5 folds, `sofiaanjenje/rsna-knee-e13-train`) |
| Rules | **legacy**: dominant-axis IPP ordering, corner-x laterality (±5 mm), zero-fill for failed slices, slot fallback |
| Layouts | E10: 3 FS planes × 8 slices, 224 px, **no crop**. E13: SAG_FS, COR_FS, AX_FS, SAG_NOFS, 130 mm. E11 layout: 3 non-FS planes + SAG_FS, 130 mm |
| Blend | `Rad = rank(0.5·rank(E10) + 0.5·rank(E13))`; for labels not in `_RAD_EXCLUDE`: `cand = 0.45·rank(parent) + 0.55·Rad`; then `branch = rank(0.8·rank(cand) + 0.2·rank(E13 heads on E11 layout))` for all 12 labels |
| Calibrator | embedded zlib/base64 JSON: sparse linear map 88 features → 12 outputs. Features = 3 rank blocks + 2 diffs + mean (72), 4 group means, 12 protocol counts. Applied as `0.6·branch + 0.4·rank(cal)` to 7 gated labels (ACL, Medial OA, Lateral OA, PF OA, Effusion, Baker's, Contusion) |
| Calibrator provenance | input means are exactly `0.5 + 1/(2N)` with **N = 4349.0 = 4407 − 58**, so it was fit on the non-gold training studies. Targets must therefore be report-derived labels. Identical payload in 55 of 59 ≥0.941 notebooks |
| **D4 change** | `_RAD_EXCLUDE` no longer contains Fracture, so Fracture now receives the 0.55 Rad blend |

### Stage 4 — Raptor CoAtNet MIL (4 views, 3 checkpoints)
| Item | Value |
|---|---|
| Backbone | `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` + attention-MIL head per label (`dreaddevelopment` datasets) |
| Views (prob weight) | maxspan-v5 (0.60), native384dense-v10 (0.10), maxspan-v5-reverse (0.10), native384-v8 (0.20) |
| Series selection | slots (Sag fluid ×18, Sag non-fluid ×14, Cor fluid ×12, Cor non-fluid ×8, Ax ×12) keyed on `test_series.csv` **Fluid_Sensitive** (not fat-sat) |
| Slice ordering | IPP·normal; per-file InstanceNumber fallback (mixed keys possible within a series) |
| Laterality | **none** |
| Pixels | modality LUT, MONOCHROME1 inversion, per-series 2–98 percentile, 140 mm crop, INTER_AREA → 336/384, bilinear to 384 |
| Sampling | "capacity-aware dense" up to 96 unique slices → 94 triplet windows (parent release: 62/42) |
| "reverse" view | same windows with the triplet channel order flipped |
| Fusion | weighted **probability** mean of 4 views → rank |

### Stage 5 — four CoAt "readers" (subprocess children; runtimes live inside datasets)
| Reader | Dataset | Checkpoints | Notes |
|---|---|---|---|
| residual-gated top3 | `mattiaangeli/rsna-knee-coat-resgated-ep10-top3` | epochs 4, 6, 8 | manifest field `gold58_rank_ensemble_auc` = selection |
| D4 depth-zone SWA3 | `mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2` | parent rank-8 SWA2 [15,16] + adapter SWA [11,9,5] | 384 px, canonical sagittal, input prep patched to dense 96-slot stack |
| Global96 "Gold Top3" | `mattiaangeli/rsna-knee-coatnet-global96-top3` | epochs 16, 23, 18 | name indicates gold-set top-3 selection |
| Repair-v1 top3 | inside the D4 dataset | epochs 12, 7, 11 | — |

- Family fusion: **rank of the mean of the 4 readers' probabilities** (25% each).
- `C = 0.6·rank(Raptor) + 0.4·rank(CoAt family)` with no inner re-rank.
- Internal preprocessing of the readers is **UNKNOWN**: it lives in dataset `.py` files not included
  in the notebook.

### Outer fusion (public part)
`P_t = rank( (1 − w_t)·rank(T_t) + w_t·C_t )`, where T is the transformer stack (stages 1–3) and
w_t = 0.60 default, ACL 0.75, Medial Meniscus 0.80, **Lateral Meniscus 1.00**, Lateral OA 0.75,
Fracture 0.75. For Lateral Meniscus the DINO, A5 and Rad stages are discarded entirely.
maverick's own note says these per-target weights were chosen "on the strength of a public split".

## 3. Private part (cells 30–32) — D4's actual addition

| Item | Value (source) |
|---|---|
| Assets | `/kaggle/input/datasets/pjmathematician/rsna-knee-eff6-assets` (8 `.pt`) and `…/rsna-knee-ens14-assets` (43 `.pt`, plus `src/infer_kernel.py`) — mount paths printed in the commit log; Kaggle lists 3 data sources as **[Private Datasource]** |
| Arm used by Lite | `S_dist`: `tf_efficientnetv2_s.in21k_ft_in1k`, px=384, slots=5, T=96, 4 checkpoints `dist2_s0, dist2_s1, f13_s0, f13_s1` (log) |
| Inference settings | `RSNA_MAX_WIN=36`, `RSNA_WORKERS=4`, 2 GPUs |
| Checkpoint combination | **UNKNOWN** (inside private `infer_kernel.py`) |
| Training data, labels, folds | **UNKNOWN** |
| Blend | one global weight: `final = rank(0.55·rank(P) + 0.45·rank(F))`; a second "EFF/core" leg has weight 0 in Lite |
| Author's claims (unverified) | "Our legs measured 0.75–0.89 [rank correlation] against the public ones offline (Sec 8.147)"; "870 studies cannot support 12 weights". 870 ≈ 4349/5, i.e. consistent with one fold of a 5-fold split of the non-gold studies (inference) |
| Failure mode | soft-fail to public output only if the fleet CSV is missing or constant. But `_EFF6 / 'arms'` is evaluated unconditionally, so **a fork without the private data raises TypeError in cell 31** |

## 4. Nominal per-target rank share in the final D4 Lite submission

Computed analytically from the blend constants (`d4_targetwise_analysis.csv`). This ignores the
non-linearity of the intermediate re-ranks and A5 eligibility, so treat it as an approximation.

| Target | DINO+A5 | RadImageNet (+cal) | CoAtNet (Raptor+CoAt) | Private |
|---|---|---|---|---|
| ACL | 0.030 | 0.108 | 0.412 | 0.45 |
| MCL | 0.079 | 0.141 | 0.330 | 0.45 |
| Medial Meniscus | 0.040 | 0.070 | 0.440 | 0.45 |
| Lateral Meniscus | 0.000 | 0.000 | 0.550 | 0.45 |
| Medial OA | 0.048 | 0.172 | 0.330 | 0.45 |
| Lateral OA | 0.030 | 0.108 | 0.412 | 0.45 |
| PF OA | 0.048 | 0.172 | 0.330 | 0.45 |
| Effusion | 0.048 | 0.172 | 0.330 | 0.45 |
| Synovitis | 0.079 | 0.141 | 0.330 | 0.45 |
| Baker's | 0.106 | 0.114 | 0.330 | 0.45 |
| Contusion | 0.048 | 0.172 | 0.330 | 0.45 |
| Fracture | 0.050 | 0.088 | 0.412 | 0.45 |

Reading: the "public DINO ecosystem" is, at the final blend, mostly a **CoAtNet** ecosystem. Every
Raptor and CoAt checkpoint uses `coatnet_rmlp_2_rw_384`. The DINO transformer stack is at most
about 11% of any column in D4 Lite and at most about 19% in Jiwei v11.

## 5. Preprocessing inconsistencies found (inputs to our own data audit)

1. **Four different slice-ordering rules** across branches:
   - DINO: geometric normal, with filename fallback.
   - A5: InstanceNumber only.
   - Rad: dominant-axis IPP.
   - Raptor: normal with **per-file** InstanceNumber fallback, so keys can be mixed inside one series.
2. **Four different laterality policies**:
   - DINO: tag, else centre-x ±20 mm, including sagittal.
   - A5: non-sagittal only, IPP[0] of the first file.
   - Rad: corner-x ±5 mm.
   - Raptor: none.

   Medial/lateral targets depend on laterality, and the dominant CoAtNet branch does not normalize it.
3. **Different series-selection keys**: header regex (DINO), `Fat_Suppression` (A5), `Fluid_Sensitive`
   (Raptor), and regex with legacy fallback (Rad).
4. **Crops are image-centred**, not anatomy-centred: 130 mm (DINO/A5/Rad E13/E11), 140 mm (Raptor),
   none (Rad E10).
5. **bf16 autocast on T4** in A5. T4 has no native bf16. The results are numerically fine but the
   choice is unusual.

These are observations about the public code, not verified defects. Whether they cost AUC needs our
own ablations (see `../next_10_experiments.md`).
