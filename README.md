# RSNA Knee Abnormality Detection (Kaggle 2026): team Hack2Publish

This is the working repository of team **Hack2Publish** for the [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) code competition.

## Team: Hack2Publish

- **Md. Hamid Hosen**: [hosen42](https://www.kaggle.com/hosen42)
- [esfersami50](https://www.kaggle.com/esfersami50)
- [foysalemonshanto](https://www.kaggle.com/foysalemonshanto)
- [kishormorol](https://www.kaggle.com/kishormorol)

## Competition

- **Task:** predict 12 findings per knee MRI study: ACL, MCL, Medial / Lateral Meniscus, Medial / Lateral / PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture.
- **Metric:** macro ROC AUC.
- **Leaderboard split:** public LB = 30% of about 1,284 hidden studies, private LB = the other 70%. Scores are shown truncated to 3 decimals.
- **Format:** code-only, 2×T4 GPU, at most 9 h per run, 5 submissions per day.
- **Final deadline:** 2026-10-22 23:59 UTC.

## Result

| | |
|---|---|
| Best public score | **0.958**. 23 of our submissions reached it, the first on 2026-10-16. |
| Team-best submission | **H976**, sub `57497105` (#1 of our submissions by unrounded public score) |
| Public rank at the end | 300 of 6,402, silver zone. Silver ended at rank 320, and 1,624 teams were tied at 0.958 (snapshot 2026-10-22 19:21 UTC). |
| Recommended final pair | **H976 `57497105` + C2 v2 `57315983`**. The selection itself is made in the Kaggle UI. See [Final selection](#final-selection). |
| Private score / final rank | *not recorded here yet: fill in from the Kaggle leaderboard* |

### Public score timeline

| Date | Public | What |
|---|---|---|
| 09-26 | 0.943 | Public pipeline anchor (Jiwei v11 + Fracture token, E02) |
| 09-27 | 0.935 | Our own D4-style EfficientNetV2-S leg at 0.30 (E11). It lowered the score, so it was stopped. |
| 10-05 | 0.944 | Anchor 0.944: the E02 anchor + the public goodpjw 2.5D ConvNeXt reader |
| 10-07 | 0.945 | Tri-specialist notebook + our 7-checkpoint reader (E18 / E19 / E20) |
| 10-11 | 0.946 | E21: the best rule-clean submission (no OAI / fastMRI-trained weights) |
| 10-11 | 0.954 | Public Mirror + compact-S6 notebook (velciraptor fork) |
| 10-12 | 0.955 | H955C: 0.954 notebook × 0.8 + Mirror-student reader C × 0.2 (rank blend) |
| 10-13 | 0.957 | H957: R7 + Raptor notebook × 0.8 + 3 Mirror students × 0.2. It beat every public notebook at the time. |
| 10-16 | 0.958 | Fork of heliosli's public "blend-mine" notebook |
| 10-17 → 10-22 | 0.958 | Team best moved C2 → H964 → H961BK → H969 → **H976**. All of these stayed at 0.958; the ordering is by unrounded score. |

## Final solution (H976)

H976 = **C2 + V2 + BK + 7 students**, in [our_submission_notebook/submission_h976/](our_submission_notebook/submission_h976/):

1. **C2 parent** (foysal's `rsna-knee-sub-c2` v2):
   - heliosli's public 0.958 blend: Raptor CoAtNet 384 + 224, a compact R6 ConvNeXt with flip TTA, and per-finding Raptor weights;
   - nartaa's S6fall-B model, added at 1/3 rank weight.
2. **V2:** heliosli's later per-finding Raptor weights. Lateral OA 0.15, PF OA 0.65, Synovitis 0.85, Baker's 0.30.
3. **BK:** the R6 Baker's state stacker is turned off, so Baker's comes from the plain R6 fold mean.
4. **Our students:** ConvNeXt students distilled from the teacher models, rank-blended as 0.8 parent / 0.2 students.
   - 5 Mirror-teacher students plus 2 S1 students (teacher = R6-OOF + S6fall-B), combined as an equal 7-checkpoint probability mean.
   - Every checkpoint is pinned by sha256.
   - On held-out gold-58, adding S1 lifts the student leg from 0.9260 to 0.9279.
5. **Hardening:**
   - The student stage has time guards.
   - If anything in the student stage fails, the notebook restores the parent's `submission.csv` byte for byte.
   - A strict final cell validates the file.

### What we measured on the public LB

The changes below are all small: the scores stayed inside the 0.958 band, about 1e-4 apart. They are read from the unrounded order of our submissions, and run-to-run noise is about 1e-4.

| Helped | Hurt or neutral |
|---|---|
| S6fall-B at 1/3 (C2) | hengck23 MaViT leg (0.957) |
| Students at 0.2 (3 of 3 pairs) | Student weight 0.3 (2×) |
| V2, only when students are present (3 of 3 pairs) | V2 without students |
| BK, Baker's stacker off (2 of 2 pairs) | Student anatomical mirror TTA (2×) |
| FK, Effusion stacker off (2 of 2 pairs) | heliosli Part B reader (2×) |
| S1 students (2 of 2 pairs) | Raptor-224 anatomical mirror |

### Last day (2026-10-22)

- **Scored:** H981 = H974 + S1, a V2-free stack. It finished at public 0.958, #2 in our order.
- **Failed:** H980 (H976 + FK), H983 and H984 (family-balanced student mean) and an exact duplicate of H980. All four got "unhandled error" on the hidden rerun.
  - Every part of H980 had passed the hidden rerun in other combinations, and H981 was submitted 3 s after H980 and succeeded. So the cause looks like infrastructure, not code, but this was not verified.
- **Never ran:** H982. Its Kaggle smoke run was cancelled in the queue 4 times, most likely because the GPU quota of `hosen42` had run out.

## Final selection

An 8-agent validation (evidence → Monte Carlo over candidate pairs → adversarial critics → judge) recommended **H976 + C2 v2** instead of the top two by public score (H976 + H981).

- **Why not H976 + H981:** the two differ only in V2 and FK. Their private scores were modelled as about 97% correlated, so the second pick adds almost nothing.
- **What C2 adds:** C2 carries none of the changes we adopted from public reads (students, S1, BK, FK, V2).
  - The pair can therefore not end below C2 on private.
  - C2 is also V2-free, so it hedges heliosli's public-tuned weights.
- **Robustness:** C2 was in the best pair in 20 of 22 sensitivity cases, and neither critic refuted the choice.
- **Size of the effect:** the expected gain is only about +1e-4. The point of the pair is protection against a private drop.

## Repository layout

| Folder | Contents |
|---|---|
| [our_submission_notebook/](our_submission_notebook/) | Every notebook we pushed, trained or submitted. Each `submission_*` folder holds the `.ipynb` plus its `kernel-metadata.json`; `training/` holds the teacher, student and cache kernels. The README there has the full submission table; `_index.csv` has one row per folder. |
| [candidate_notebook/](candidate_notebook/) | Built and tested notebooks that were not submitted: H965, H966, H972, H973, H977, H978, H982 / H982b, and the `ens_teacher_pipeline`. |
| [our_base_notebook/](our_base_notebook/) | Frozen copies of the base notebooks: C2, H960, the heliosli fork, H958B. |
| [ours_leg/](ours_leg/) | Our code: notebook builders (`build_h960.py` for the whole C2 family, `build_h957.py`, `build_h958.py`, ...), the student leg (`student_leg.py`, `infer_fam.py`, `infer_tta.py`), R6 switches (`r6_switches.py`), training code (`train_ours_leg.py`, `e13/`, `gpjw/`), tests (`tests/`) and Kaggle tools (`kaggle/`). |
| [audit/](audit/) | Research notes: the competition analysis, an inventory and lineage of 59 public notebooks scoring ≥ 0.941, the D4 audit, the experiment table (`07_experiments.csv`) and evidence files. |
| [public_notebooks/](public_notebooks/) | Downloaded public notebooks, named `{score}_{votes}v_{author}__{slug}.ipynb`. |
| [discussions/](discussions/) | An offline clone of the competition forum, one `.md` per topic. Re-sync it with `_tools/clone_discussions.py`. |
| [team_shared/](team_shared/) | Teammates' kernels and checkpoints (esfersami50). |
| `students_upload/` | Student checkpoints staged for a Kaggle dataset upload. It is gitignored. |

## Rebuilding and testing a C2-family notebook

```bash
# example: H976 = C2 + V2 + BK + students + S1 (output goes to candidate_notebook/submission_<name>/)
python -I ours_leg/build_h960.py --parent c2 --students --robust --v2 --bk --s1 --name h976 --owner hosen42

# structure checks (stdlib only)
python -I ours_leg/tests/test_c2_family_build.py candidate_notebook/submission_h976

# the hardened student leg on fake data (no GPU)
uv run --no-project --with numpy --with pandas --with scipy python -I ours_leg/tests/test_h960r.py <notebook.ipynb> <scratch dir>
```

Other builder flags:

| Flag | Effect |
|---|---|
| `--fk` | turns off the R6 fluid stacker |
| `--fam` | family-balanced student mean |
| `--partb` | adds heliosli's Part B reader |
| `--student-tta` | student mirror TTA |
| `--mavit W` | adds the MaViT leg at weight W |
| `--student-weight W` | changes the students' rank weight |

Old notebooks rebuild byte for byte.

Kaggle helpers in `ours_leg/kaggle/`:

| Script | What it does |
|---|---|
| `smoke_check.py` | stage markers and `submission.csv` shape of a finished run |
| `wait_kernels.py` | waits for kernel runs to finish |
| `wait_subs.py` | waits for submissions to be scored |
| `fam_check.py` | checks the family-balanced mean against earlier runs' reader outputs |

All of them call the Kaggle API read-only with `~/.kaggle/access_token`.

## Working rules we followed

- One variable per experiment, with every change tested against a scored parent.
- Experiment labels stay PROPOSED until measured.
- gold-58 was used only as a guard, never for tuning.
- No weights tuned by us on the public LB.
- Every notebook was smoke-tested on Kaggle before submission.
- **External data:** most 0.949+ public weights (nartaa's publication-SWA Raptor, Mirror) were trained with OAI knees. The host allowed OAI only without institutional sign-off, and each team is responsible for its own compliance. Our best rule-clean submissions are E21 (0.946) and C3 (0.954, OAI-free but uses R6).
- **Secrets:** Kaggle tokens are never committed (`.gitignore`: `token*.txt`, `*token*.txt`, `access_token`, `kaggle.json`, `.env`). The git history was checked on 2026-10-29 and holds no tokens.

## Copyright

Copyright © 2026 Team Hack2Publish: Md. Hamid Hosen (hosen42), esfersami50, foysalemonshanto and kishormorol. All rights reserved.

This covers the team's own work in this repository: our code in `ours_leg/`, the notebooks we built in `our_submission_notebook/` and `candidate_notebook/`, our trained student checkpoints, and our research notes in `audit/`.

Third-party material keeps its original copyright and licence and is stored here only for reference:
- the public notebooks in `public_notebooks/` and `our_base_notebook/`;
- the forum posts in `discussions/`;
- the components of other authors embedded in our notebooks: heliosli's blend and R6 bundle, nartaa's weights, the Raptor models, goodpjw's reader and hengck23's MaViT.

The competition data belongs to RSNA and may be used only under the competition rules.
