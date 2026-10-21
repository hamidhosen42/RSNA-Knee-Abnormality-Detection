# Our own notebooks (team Hack2Publish)

This folder holds only notebooks we pushed, trained or submitted ourselves. Drafts that are built and tested but not pushed yet live in `../candidate_notebook/`, and move here once pushed. Public notebooks, including verbatim forks of them that teammates submitted, stay in `../public_notebooks/`.

- **`submission_*/`**: one folder per submission notebook, holding the `.ipynb` plus the `kernel-metadata.json` that was pushed.
- **`training/`**: our training, teacher and cache kernels, each with its script and metadata. Examples:
  - `gpjwstudent*`: the Mirror-student readers;
  - `mirrorteacher`, `raptorteacher_m224`, `r7teacher_*`: teachers on the train split;
  - `gpjwcache_*`: DICOM caches;
  - `training/colab/`: the Colab notebook, prepared but never used.
- **`_index.csv`**: one row per folder, giving:
  - the Kaggle kernel and the account that ran it;
  - the submission ids and public score;
  - the date and the submission description.

Every builder in `../ours_leg/` takes its paths from `ours_leg/paths.py`. Path arguments follow one rule:
- an absolute path is used as is;
- `our_submission_notebook/...` and `public_notebooks/...` are relative to the repo;
- a bare `submission_x/...` is relative to this folder.

For `--out`, a bare folder name goes into this folder, and any other relative path is relative to the current directory.

A builder never replaces an archived file with different content unless you set `OVERWRITE=1` (or pass `--overwrite` to build_h958). This matters because several builders evolved after their pushes. After adding a folder, run `python ours_leg/update_our_index.py`. For example:

```
python ours_leg/build_h958.py --flip-r7 --out submission_h958b_r   # the patched (R1-R6) H958b, as a new folder
python ours_leg/build_h958.py --owner foysalemonshanto --flip-r7 --out /abs/scratch/submission_h958b   # another account: build at push time
```

## C2-family experiments (10-18 / 10-19, hosen42; each smoke-tested on Kaggle before submitting)

Built with `ours_leg/build_h960.py --parent c2 ...`. Tests are in `ours_leg/tests/`: `test_c2_family_build`, `test_h960r`, `test_v2_blend`, `test_mavit_leg`, `test_student_tta`, `test_partb`, and `tta/`.

| Folder | Change vs its parent | Sub | Public |
|---|---|---|---|
| `submission_h961` | C2 + our 5 students 0.8/0.2 | 57369206 | 0.958 |
| `submission_h963` | C2 + heliosli V2 per-finding Raptor weights (LB-chosen by heliosli) | 57369271 | 0.958 |
| `submission_h964` | C2 + V2 + students | 57369477 | 0.958, **team best** (beat C2, H961 and H963) |
| `submission_h967` | H964 + hengck23 MaViT leg after the final cell, 0.9/0.1 rank | 57406594 | **0.957** (MaViT hurt) |
| `submission_h968` | H964 + anatomical L/R mirror TTA for the students (2 GPUs, guarded) | 57409126 | 0.958 (order vs H964 unknown) |
| `submission_h969` | H964 + foysal's BK (Baker's from the R6 plain fold mean, state stacker off) = H961BK + V2 | 57447421 | pending |
| `submission_h970` | H969 + the student mirror TTA | 57447461 | pending |

foysal's H961BK (57417566, H961 + BK) has been team best since 10-20. His H961S3 (students 0.3) and H961SB (S3 + BK) scored below it, so student weight 0.3 hurts.
Reading rule, fixed before the results: if the team best moves to H970, keep the TTA; if it moves to H969, drop the TTA; if it stays on H961BK, V2 on top of BK has not been shown.

Built but not submitted: `../candidate_notebook/submission_h965` (H964 + heliosli Part B, effect expected about 0) and `submission_h966` (students at 0.3; rejected as LB tuning).

## Latest submissions (2026-10-17, archived 2026-10-18)

| Folder | Kernel (archived version) | Sub | Public | Note |
|---|---|---|---|---|
| `submission_c2` | foysalemonshanto/rsna-knee-sub-c2 v2 | 57315983 | 0.958 | **team best** (rank 254, silver, 10-18): the 0.958 notebook + nartaa S6fall-B at 1/3 rank. A copy is in `../our_base_notebook/`. |
| `submission_c2_noflip` | foysalemonshanto/rsna-knee-sub-c2-noflip v2 | 57315976 | 0.958 | C2 with R6 flip off |
| `submission_c2_syn0` | foysalemonshanto/rsna-knee-sub-c2-syn0 v4 (= v3) | 57320843 | 0.958 | no description |
| `submission_c3` | foysalemonshanto/rsna-knee-sub-c3 v3 | 57307392 | 0.954 | OAI-free. Scored version unverified: v1 lacks the 10-17 review fixes; v2 and v3 have identical code. |
| `submission_h960` | kishormorol/rsna-knee-h960 v1 | 57281541 | 0.958 | 0.958 notebook + 5 students. Team best until C2. A copy is in `../our_base_notebook/`. |
| `submission_h958m`, `submission_h960rm` | kishormorol v1 | 57281543, 57292960 | 0.958 | 224 mirror: no gain over 0.958 |

## Best of our own notebooks (2026-10-16)

| Notebook | Public | Note |
|---|---|---|
| `submission_h958b` | 0.957 | R7 + Raptor 384/224 (mirror) + R7 flip + 5 Mirror students; it was the team best on 10-15. This is the exact notebook foysal submitted (before the R1–R6 patches). A copy is in `../our_base_notebook/`. |
| `submission_h957x5w40`, `submission_h957x5`, `submission_h957`, `submission_h958`, `submission_h958bw40` | 0.957 | |
| `submission_e21` | 0.946 | rule-clean (no OAI / fastMRI weights) |

## Team submissions of public notebooks (not stored here)

On 2026-10-16, foysalemonshanto (57252333) and kishormorol (57252502) submitted forks of `heliosli/blend-mine-in-private-lb-luck-to-all-upvoters` and scored **0.958**. 57252502 is the team best now, at rank 282 of 6,098, inside a 0.958 block of 435 teams. The notebook itself is `../public_notebooks/0.958_157v_heliosli__blend-mine-in-private-lb-luck-to-all-upvoters.ipynb`.
