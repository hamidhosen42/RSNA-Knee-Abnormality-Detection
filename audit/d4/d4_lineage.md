# D4 lineage and classification

## Lineage chain (code-level evidence)

```
renta0426 / anvithpothula early bases (0.93x)
  └─ mattiaangeli "Bend the knee to the dinosaurs" (BTKD, 0.941): DINO20 + A5 + Rad(+calibrator) + Raptor4 + resgated CoAt
       └─ maverickss26 rsna-knee-0941-restructured → 0.942 recipe: A5 .52, Rad .55/.20, 94 windows, + D4 depth-zone CoAt
            │     (maverickss26/rsna-knee-0942-restructured v7; Jiwei v7; beraterolelk speedy-raptors-coatnet-d4-sota v5)
            ├─ sushanthtiruvaipati/rsna-knee-0942-rad-fracture-oof (0.942): ONE token: Fracture re-admitted to Rad
            └─ mattiaangeli speedy-raptors v34 (0.943): + Global96 (+ Repair-v1), rank-of-mean-probabilities
                 └─ jiweiliu/rsna-knee-fast-2xt4-inference v9 → v11 (0.943)
                      ├─ haideptry …-coatnet-d4-0943 (0.943)   [100% line-identical]
                      ├─ evgendvorkin/rsna-versia-5 v17        [100% line-identical; its 0.942 came from v16 = maverick 0.942 code]
                      └─ pjmathematician/rsna-knee-d4-lite (0.945) = Jiwei v11 + sushanth's token + private leg
                           ├─ rsna-knee-d4-blend (0.946)   = Lite + 2 private arms
                           ├─ rsna-knee-d4-w55             = Blend with private weight 0.55
                           └─ rsna-knee-d4-blend-b         = Blend + private "ens14 core" leg, Fracture re-excluded
```

Separately, the "OUR LEG 1/3–3/3" wrapper around the private leg is a template that appeared
earlier (2026-09-14) in `valentinonguyen/knee-sub-band32w08/w15`:
- There it blends a **public** band32 leg (`ricecakar/rsna-knee-band32-fast3`) at weight 0.08,
  scoring 0.941, the same as its base.
- The blend cell is near-verbatim, including the comments "Sec 8.139b" and "870 studies cannot
  support 12 weights".

This shows shared code lineage. It does not by itself prove team membership.

## Comparison against every family you named

| Family | Relation to D4 |
|---|---|
| rsna-knee-0942-restructured (maverick) | D4 public part = its 0.943 successor; 91% of maverick's lines are in D4 |
| RSNA Knee Fast 2xT4 Inference (Jiwei) | **Direct parent.** 100% of Jiwei v11 lines are in D4; D4 differs by 1 token + 3 appended cells |
| The bee's knees (prvsiyan, 0.941, best v27) | Older BTKD-lineage branch; shares only 37–51% of lines, but loads the same weight datasets |
| Tony rank ensemble (andreluizpedroso, 0.942, best v29) | Repackage: `tonylica/...dinov3-0917-repro-assets` re-hosts the *same* files (identical sizes) — A5 = "DINOv3", Raptor, RadImageNet, resgated CoAt |
| Pilkwang baseline / descendants | Source of the 20 DINOv2-S members used by D4 (1 config × 5 folds × 4 seeds) |
| Max-Span Dense Corpus | Raptor v5 checkpoint, used by D4 at 0.60 + 0.10 (reverse view) inside Raptor |
| Native-384 / Native-384 dense | Raptor v8 (0.20) and v10 (0.10), used by D4 |
| Shared knee MRI fold weights | `mattiaangeli/knee-mri-fold-weights` = A5 = DINOv3-S/16, used by D4 at 0.52 in T1 |
| DINOv2 checkpoints | `metaresearch/dinov2 small/1` + pilkwang members, used by D4 |

Reused checkpoints: **all public D4 checkpoints are reused**. The only non-reused ones are the
private fleet checkpoints.

## Classification (evidence-based)

**As a leaderboard submission: B, partially independent.**
- 55% of its final rank is a byte-level copy of the public 0.943 pipeline, plus one public idea.
- 45% comes from a separately trained EfficientNetV2-S model. It is architecturally different from
  every public component (none of the 59 notebooks has a timm EfficientNet), but its training data,
  labels and correlation cannot be verified.

**As something we can build on: D, near-duplicate/repackage.**
- The only part we can access or reproduce is Jiwei v11 + the Fracture token.
- The genuinely new information lives in private datasets.

**D4 Blend relative to D4 Lite: C, mostly same-family.**
- It adds a ConvNeXt-L (different family) and a CoAtNet-rmlp-2 (same family as the public
  Raptor/CoAt stack) to the same private leg.
- The +0.001 is inside rounding.

## Answers to the 10 questions

1. **What exactly is D4 Lite?**
   `rank(0.55·rank(Jiwei v11 with Fracture re-admitted to the RadImageNet blend) + 0.45·rank(private EfficientNetV2-S "S_dist" leg, 4 checkpoints))`.
2. **What exactly is D4 Blend?**
   The same, with the private leg expanded to EfficientNetV2-S + CoAtNet-rmlp-2-rw-384 + ConvNeXt-L
   (8 checkpoints). How the arms are combined inside the leg is unknown (private script).
3. **How are they related?** Identical except one line in cell 31.
4. **Are they related to the shared 0.941–0.942 DINO family?**
   Yes. The public 55% is that family's 0.943 descendant, and 56–58 of the 59 notebooks with public
   score ≥0.941 load the same weight datasets. At the final blend the public part is about 60–100%
   CoAtNet (Raptor + CoAt readers) by nominal rank share, not DINO.
5. **Which checkpoints are reused?**
   - All public ones: 20 DINOv2-S members, 5 A5/DINOv3 folds, RadImageNet R50 + 10 heads,
     3 Raptor CoAtNets (4 views), and the 4 CoAt readers (3 + SWA + 3 + 3 checkpoints).
   - New: 4 private checkpoints (Lite) or 8 (Blend).
6. **Which component actually produces the gain?**
   Not established. The bounds and prior evidence make the private leg the most plausible source.
   The Fracture flip was LB-neutral in the 0.942 family, and its report-label and gold-58 evidence
   disagree. A single-submission ablation is proposed (see `d4_component_ablation.csv`).
7. **Does Blend add real model diversity?**
   ConvNeXt-L plausibly does. The CoAtNet arm duplicates the public stack's dominant family. The net
   +0.001 is not distinguishable from noise.
8. **Which is safer for private LB?** Lite, on simplicity; the evidence cannot separate them.
9. **Which is better for the efficiency track?** Lite (about ⅓ of the private-leg compute on the only
   timing available).
10. **Next controlled experiment?**
    - For the attribution question: the one-submission ablation "Jiwei v11 + Fracture flip".
    - For our own solution: a fully independent, OOF-validated EfficientNetV2-S-class CNN that we
      train ourselves. D4 is indirect evidence that an independent CNN leg blended at a large rank
      weight with the public 0.943 stack is where the remaining public headroom is.

    See `../next_10_experiments.md`.
