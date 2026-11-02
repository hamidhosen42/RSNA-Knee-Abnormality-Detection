# IEEE Access paper: manuscript and code

*Report-Supervised Distillation for Multi-Label Knee MRI Classification: An Evaluation Under Limited Expert Annotation*

| Folder | Contents |
|---|---|
| [manuscript/](manuscript/) | `main.tex` in the official IEEE Access template, `supplement.tex` (Supplementary Material), `references.bib`, figures, the compiled `main.pdf` and `supplement.pdf`, and `ieee_access_overleaf.zip`. The build steps are in [manuscript/BUILD.md](manuscript/BUILD.md). |
| [REVIEW_RESPONSE.md](REVIEW_RESPONSE.md) | Point-by-point response to the external review, and what still needs the authors. |
| [code/](code/) | The code behind every method and result in the paper. |

The files in `code/` are **copies**. The originals stay where they were (`ours_leg/` and `our_submission_notebook/training/`), so nothing that refers to them breaks. If you change a method, change the original as well, or say which copy is the reference.

## Code by paper section

| Paper | File | What it does |
|---|---|---|
| Alg. 1, soft report targets and folds | `code/reader/make_labels.py` | Averages the four LLM label tables (Eq. 1) and builds report-grouped GroupKFold(5). gold-58 gets fold −1. |
| Alg. 2, canonical preprocessing | `code/reader/preprocess.py` | DICOM → uint8 volume. Slices are ordered along the normal and re-oriented to LPS, at 0.4 mm and a 153.6 mm FOV, windowed at percentiles 1/99.5, ≤ 64 slices. |
| Alg. 3, 2.5D multi-slot reader | `code/reader/knee.py` | `StudyDataset` (window sampling) and `KneeNet` (ConvNeXt, slot/position embeddings, 2-layer transformer, per-finding attention pooling). |
| §IV-D, optimisation (Table 3) | `code/distillation/*.py` (executed) and `code/reader/train.py` (upstream reference) | **Every reader in the paper was trained by the executed scripts in `code/distillation/`** (fp16 + GradScaler, micro-batch 2 × 4 accumulation). `reader/train.py` is the upstream trainer (bf16, direct batches); it is kept as reference only. Both use the EMA warm-up `min(0.998, (1+t)/(10+t))`. |
| Inference | `code/reader/infer.py`, `code/reader/infer_fam.py` | DICOM → predictions with a probability mean over checkpoints. `infer_fam.py` is the family-balanced variant. |
| Alg. 4, self-distillation (arm B) | `code/distillation/self_distillation_B.py`, `code/reader/oof.py` | The Kaggle script that was run. It trains 5 OOF readers, builds 0.5·report + 0.5·OOF targets, and trains arms A and B. |
| Alg. 4, image-teacher students | `code/distillation/students_round1_C_B2.py`, `students_round2_C2_C3.py`, `students_round3_S75_T75s1.py` | The scripts that were run for the arms in Table 5. `build_targets_C` implements Eq. 5. |
| Image teacher on training studies | `code/distillation/teacher_mirror_on_train.py` | Runs the published Mirror CoAtNet on the 4,407 training studies, which produces the teacher probabilities t. |
| Ensemble-teacher gate | `code/distillation/gate_ens_teacher.py` | Sanity checks (coverage, spread, correlation, AUC vs report labels) before a teacher file is used. |
| Eq. 5 + Supplementary Alg. S1, rank fusion and fall-back | `code/fusion/final_system_h976.ipynb`, `code/fusion/student_leg.py`, `code/fusion/r6_switches.py` | The final fused system, as run. `student_leg.py` holds the hardening patches (checkpoint lookup, time guard, rank imputation). `r6_switches.py` holds the BK/FK switches (Supplementary Table S1). |
| §VI-E, EfficientNetV2-S negative results (Tables 9–10) | `code/efficientnet_baseline/` | `train_ours_leg.py` (E11/E03b), `infer_kernel.py`, `folds.py` + `folds_v1.json`, `verify_corpus_rebuild.py` (the DICOM-vs-corpus check), and `e13/` (the ResNet-34 multi-view model). |
| §VI-F, yardstick illustration, Figs. 8–9 | `code/analysis/make_figures.py`, `code/analysis/data/*.npz`, `code/analysis/stats.json` | Bootstrap CIs on gold-58, the 385-study illustration, per-finding OOF, and both plots. It runs standalone (numpy + matplotlib). If `gold_arm{A,B,C,C3}.csv` from the training outputs are placed in `data/`, it also writes paired gold-58 CIs for B−A, C−B and C3−A. |

| §III-E, label analysis, Figs. 2–5 | `code/analysis/make_data_figures.py`, `code/analysis/data/labels_consensus.parquet`, `code/analysis/data/label_benchmark.csv`, `code/analysis/data_stats.json` | Label composition, co-occurrence, positives per study, and report-labeller vs expert AUC. The two data files are copies of the team's label pipeline output (`team_shared/esfersami50/nx1-01-consensus-labels/`). Run with numpy, pandas, pyarrow and matplotlib. |
| Optional example-input figure (MRI slices) | `code/analysis/make_input_figure.py` | Renders the 6 slots × 5 evaluation windows of one **training** study with the paper's own preprocessing. It needs the competition DICOMs (run it on Kaggle) plus torch/timm (imported by `knee.py`). It is not in the paper yet: first check that the dataset terms allow publishing images. |

Figures 1, 6 and 7 (overview, data use, architecture) are TikZ sources in `manuscript/figures/*.tex`. Compile each with `pdflatex`.

Reproduce the statistics and figures from the repository root:

```bash
uv run --no-project --with numpy --with matplotlib python -I ieee_access_paper/code/analysis/make_figures.py
```

## Known gaps before public release

- **S1 students (E75t, E75s).** These are the two ensemble-teacher students in Table 5 and the final system. They were trained by the Kaggle kernel `hosen42/rsna-knee-gpjw-student5`, whose script is **not** in this repository. Download it from Kaggle and add it as `code/distillation/students_round4_E75t_E75s.py`. The `gpjwstudent4` script in `our_submission_notebook/training/` uses a different teacher (Mirror/M224/R7) and is not the one that produced the checkpoints in the paper.
- **Training paths.** The scripts expect the Kaggle input layout (`/kaggle/input/...`) or the paths at the top of each file. They were not refactored for a local run.
- **Credit.** The reader architecture and its training code (`knee.py`, `preprocess.py`, `train.py`, `infer.py`, `make_labels.py`) follow the public code of goodpjw2008, which the paper cites. Keep that attribution and the original licence when you release the code.
- **Trained weights** are not in the repository (Kaggle datasets `hosen42/rsna-knee-student-readers` and the student5 output). The paper's Data Availability section should say where they can be obtained.
