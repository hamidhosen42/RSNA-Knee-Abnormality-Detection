# D4 Lite (0.945) vs D4 Blend (0.946)

## What differs

One line (cell 31). Blend adds two private arms to the fleet leg:

| | Lite | Blend |
|---|---|---|
| Private arms | `S_dist` (EfficientNetV2-S, 4 ckpt) | `S_dist` + `S_coat` (CoAtNet-rmlp-2-rw-384, 2 ckpt) + `S_cnxl` (ConvNeXt-L, 2 ckpt) |
| Private checkpoints | 4 | 8 |
| Outer weight of private leg | 0.45 | 0.45 |
| MAX_WIN | 36 | 36 |
| Fleet time on 3 visible studies | 33 s total, 10.97 s/study (log) | 95 s total, 31.60 s/study (log) |
| Public part | identical | identical |
| Private data sources attached | same 3 | same 3 |
| Public LB | 0.945 | 0.946 |

## Is the +0.001 real?

Unknown, and not demonstrable from public information.
- **Rounding alone:** Blend − Lite ∈ (0.000, 0.002).
- **LB split noise:** adds on top of that.
- **No OOF, seeds or fold data exist for either private leg.**
- **Only 3 visible test studies**, so output correlation between Lite and Blend cannot be measured.

## Does Blend add real diversity?

Partly, and it cannot be verified.
- `S_cnxl` (ConvNeXt-L) is a genuinely different family from everything public.
- `S_coat` uses **the same backbone as every public Raptor and CoAt checkpoint**
  (`coatnet_rmlp_2_rw_384`), which already holds 33–55% of the public part's nominal rank share.
  It is the arm least likely to add diversity against the public stack.

## Runtime cost

Blend's fleet took about 2.9× Lite's on the same 3 studies. Those per-study times include model
load and warm-up and **cannot be extrapolated**: 31.6 s/study × ~1300 studies would exceed 9 h, yet
Blend completed its hidden-test run, so the true hidden per-study time must be far lower.

Hidden-test runtimes are not exposed by the API.

## Which is safer for the private LB?

For a leaderboard entry, on the available evidence: **D4 Lite**.
- It is the simpler object: 4 checkpoints of one architecture.
- Blend's extra +0.001 is inside rounding/noise.
- Blend's extra arms include a CoAtNet that duplicates the public stack's dominant family.

Neither is usable by *us* as a component: the private leg cannot be attached to our notebooks.

## Which is better for an efficiency track?

**Lite.** It produces the same public-LB class of result with about a third of the private-leg
compute, based on the only timing we have (3 studies).
