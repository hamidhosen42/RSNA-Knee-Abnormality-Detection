# RSNA Knee — audit workspace (2026-09-26)

Scope of this pass: Phase A (public ≥0.941 audit and lineage), Phase B (local project audit), the
D4 special audit, and the plan for the first experiments. **No model has been trained, no notebook
has been pushed, and no submission has been made.** Every experiment is PROPOSED.

## Files

| File | What |
|---|---|
| `01_competition_analysis.md` | verified vs unverified competition facts; environment state |
| `02_public_0941_plus_inventory.csv` | all 59 notebooks with best public score ≥0.941: versions, best-scoring version, data sources, components, blend knobs, family, nearest code neighbour. Also serves as Phase A `public_notebook_inventory.csv` |
| `03_public_notebook_lineage.md` | lineage map, duplicate groups, effective model families. Also serves as Phase A `public_notebook_lineage.md` |
| `next_10_experiments.md` | gaps, decision rules, first five and next five experiments, E-D4 feasibility |
| `07_experiments.csv` | experiment table (all PROPOSED, metric columns empty) |
| `d4/d4_lite_full_architecture.md` | D4 Lite end to end, from code |
| `d4/d4_0942_to_0945_diff.md` | 20-point differential audit 0.942 → 0.943 → Lite → Blend |
| `d4/d4_blend_exact_formula.md`, `d4/d4_blend_equation.csv` | exact blend equations |
| `d4/d4_lite_vs_blend.md` | the one-line difference and what it implies |
| `d4/d4_lineage.md` | lineage, A–D classification, answers to the 10 questions |
| `d4/d4_dependencies.csv` | every data/model/notebook dependency with accessibility |
| `d4/d4_component_ablation.csv` | what is observed, runnable, or not runnable |
| `d4/d4_prediction_correlation.csv` | why correlations are unavailable (visible test = 3 studies) |
| `d4/d4_targetwise_analysis.csv` | per-target nominal rank share of each component (Jiwei, Lite, Blend) |
| `d4/diff_jiwei_v11_vs_d4_lite.txt` | raw diff |
| `raw/notebooks/` | extracted code of all 59 notebooks (current versions) plus the two unscored D4 siblings; API metadata |
| `raw/notebooks_best_versions/` | exact best-scoring versions of the 10 notebooks whose latest version is not their best |
| `raw/api/` | listing JSON, view models, containment matrix, decoded calibrator, gold-58 IDs, DINO manifest |
| `raw/logs/` | commit-run logs of D4 Lite, D4 Blend, Jiwei v11 |
| `raw/scripts/` | `lineage.rb`, `inventory.rb`, `targetwise.rb` (re-runnable with Ruby 2.6) |

Not yet produced (need data or experiments): `04_data_label_audit.md` (from E03/E04),
`05_validation_protocol.md` (summarized in `next_10_experiments.md` §1), `06_experiment_plan.md`
(the same document), `08_error_analysis.md`, `09_ensemble_analysis.md`,
`10_private_lb_risk_report.md`, `train.ipynb`, `inference.ipynb`.

Note: `../public_notebooks/` (full `.ipynb` downloads named `{score}_{votes}v_{author}__{slug}`)
was created by a separate session today. This audit did not modify it.

## How the data was obtained

The Kaggle token in `~/.kaggle/access_token` works as `Authorization: Bearer …` for:
- `www.kaggle.com/api/v1/kernels/{list,pull,output}`;
- `www.kaggle.com/api/i/kernels.KernelsService/{ListKernels,ListKernelVersions}` (these give
  `bestPublicScore`);
- `kernels.LegacyKernelsService/GetKernelViewModel` (gives the best-scoring version number).

Rendered old versions come from each version's `renderedOutputUrl`. The token is never written to
any file here.
