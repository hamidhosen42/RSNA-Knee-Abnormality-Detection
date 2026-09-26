# D4 Blend — exact formula (and its siblings)

Notation:
- `R(x)` = column-wise percentile rank (pandas `rank(method='average', pct=True)`) over the test studies.
- `R01(x)` = `(rank−1)/(n−1)`.
- All operations are per target column t.
- Weights are the literal constants in code. Line references are to
  `audit/raw/notebooks/pjmathematician__rsna-knee-d4-blend.py`.

## 1. The only difference between D4 Lite and D4 Blend

```diff
- _run(_mkleg(..., _EFF6, ['S_dist']),                    '/kaggle/tmp/ours_fleet', 36, 4, 'S_dist')
+ _run(_mkleg(..., _EFF6, ['S_dist','S_coat','S_cnxl']),  '/kaggle/tmp/ours_fleet', 36, 8, 'S_dist,S_coat,S_cnxl')
```

Same public part, same outer weight (0.45), same MAX_WIN (36). Blend only adds two private arms.

## 2. Final equation

```
BLEND_t = R( 0.55 · R(P_t) + 0.45 · R(F_t^blend) )
LITE_t  = R( 0.55 · R(P_t) + 0.45 · R(F_t^lite)  )
```

- `F^lite` = private `infer_kernel.py` output for arm `S_dist` (4 checkpoints).
- `F^blend` = the same script over arms `S_dist` (4 ckpt), `S_coat` (2 ckpt), `S_cnxl` (2 ckpt).
  The log prints `weight=1.0` for each arm. **How arms and checkpoints are combined (probability vs
  rank, per-arm vs per-checkpoint weighting) is UNKNOWN**, because `infer_kernel.py` is private.
- A second leg `EFF/core` has weight `_OV_W2 = 0.0` in both Lite and Blend. The log shows
  "EFF leg invalid → folding its weight into public".

So **D4 Blend uses rank averaging at the outer level, with one global (not target-specific) weight,
and no clipping or calibration** after the private leg. All target-specific logic sits inside P.

## 3. The public part P (identical in Lite and Blend)

```
DINO_t   = R( mean_{m=1..20} R(dino_m,t) )                       # 20 members, equal weights
A5_t     = mean_{f=0..4} R01(a5_f,t)                              # 5 folds
T1_t     = R( 0.48·R(DINO_t) + 0.52·R(A5_t) )   if study has an A5 slot, else R(DINO_t)
RAD_t    = R( 0.5·R(E10_t) + 0.5·R(E13_t) )                       # each = mean of 5 fold-head probabilities
T2_t     = 0.45·R(T1_t) + 0.55·RAD_t    for t ≠ Baker's            # D4: Fracture INCLUDED; Jiwei: Fracture excluded
         = T1_t                          for t = Baker's
T3_t     = R( 0.8·R(T2_t) + 0.2·R(E13onE11_t) )                   # all 12 targets
CAL      = R( ((X − μ)/σ) · Wᵀ + b )    X = 88 features [R(T1),RAD,R(E13onE11),RAD−R(T1),R(E13onE11)−R(T1),mean3, 4 group means, 12 protocol counts]
T_t      = R( 0.6·T3_t + 0.4·CAL_t )     for t ∈ {ACL, Medial OA, Lateral OA, PF OA, Effusion, Baker's, Contusion}
         = R( T3_t )                     otherwise

RAPTOR_t = R( 0.60·p_maxspan_v5 + 0.10·p_native384dense_v10 + 0.10·p_maxspan_v5_reverse + 0.20·p_native384_v8 )   # probability space
COAT_t   = R( 0.25·p_resgated + 0.25·p_d4depthzone + 0.25·p_global96 + 0.25·p_repairv1 )                           # probability space
C_t      = 0.60·R(RAPTOR_t) + 0.40·COAT_t                         # no inner re-rank

P_t      = R( (1 − w_t)·R(T_t) + w_t·C_t )
w_t      = 0.60 default; ACL 0.75; Medial Meniscus 0.80; Lateral Meniscus 1.00; Lateral OA 0.75; Fracture 0.75
```

Fallbacks (none fired in the commit run):
- If a CoAt reader fails, the survivors share the 0.40 equally.
- If none survive, the pipeline keeps Raptor only.
- If raw probabilities are missing, it uses a rank mix.

## 4. Classification of each operator

| Operator | Present? |
|---|---|
| Probability averaging | yes: Raptor views, CoAt readers, fold heads, private arms (UNKNOWN) |
| Logit averaging | no (DINO `TTA_POOL='prob'`) |
| Rank averaging | yes: DINO members, A5 folds, every inter-stage blend, the outer private blend |
| Geometric averaging | no |
| Target-specific weights | yes, inside P only: w_t, `_RAD_EXCLUDE`, calibrator gate, DINO window pooling (max/top-2/mean per target) |
| Conditional rules | yes: A5 applied only to studies with an acquired A5 slot; CoAt reader fail-open; EFF-leg validity check |
| Clipping / calibration | 88-feature linear calibrator on 7 targets; Raptor probabilities clipped to [0,1]; no post-hoc calibration of the final ranks |

## 5. Sibling notebooks by the same author (current versions)

| Notebook | Public score | Outer equation |
|---|---|---|
| d4-lite | 0.945 | `R(0.55·R(P) + 0.45·R(F[S_dist]))` |
| d4-blend | 0.946 | `R(0.55·R(P) + 0.45·R(F[S_dist,S_coat,S_cnxl]))` |
| d4-w55 | not in ≥0.941 list | `R(0.45·R(P) + 0.55·R(F[S_dist,S_coat,S_cnxl]))` |
| d4-blend-b | not in ≥0.941 list | `R(0.60·R(P') + 0.18·R(F[3 arms, MAX_WIN 72]) + 0.22·R(Core[A_coat,B_eff2,C_cnxl,Z_pf_spec], MAX_WIN 54))`. P' re-excludes Fracture from Rad. Core runs only if >2.6 h remain. It fails loudly instead of softly |

I don't know whether w55 and blend-b were scored; they do not appear among the 59 notebooks with
best score ≥0.941.

The per-target nominal shares are in `d4_targetwise_analysis.csv`, and the machine-readable
equation is in `d4_blend_equation.csv`.
