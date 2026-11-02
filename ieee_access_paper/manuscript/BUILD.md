# IEEE Access manuscript

| File | Contents |
|---|---|
| `main.tex` | The manuscript, in the official IEEE Access template (`ieeeaccess.cls`, 2023 release) |
| `ieeeaccess.cls`, `spotcolor.sty`, `IEEEtran.cls`, `IEEEtran.bst`, `t1*` font files, `logo.png`, `notaglinelogo.png`, `bullet.png` | Official IEEE Access template files, unchanged. The IEEE site is behind Cloudflare, so they were taken from public GitHub copies. The same `ieeeaccess.cls` (sha256 `67b73c4d…`) appears in three independent repositories |
| `references.bib` | Bibliography (IEEEtran style) |
| `figures/` | `overview.pdf` (from `overview.tex`), plus `leg_alone.pdf` and `yardstick_noise.pdf` (from `../code/analysis/make_figures.py`) |
| `main.pdf` | The compiled manuscript |

## Build

Fig. 1 is `figures/overview.pdf`, compiled from `figures/overview.tex` (`pdflatex overview`), because the Access class clashes with TikZ. The class needs **pdfLaTeX**: it loads its Type 1 Formata/Times fonts with `\pdfmapfile` and its Pantone colour with `\pdfobj`. XeLaTeX and tectonic fail.

```bash
# from the repository root
uv run --no-project --with numpy --with matplotlib python -I ieee_access_paper/code/analysis/make_figures.py   # figures
cd ieee_access_paper/manuscript && pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex supplement && pdflatex supplement          # Supplementary Material
(cd figures && pdflatex overview)                    # Fig. 1, only after editing overview.tex
```

On **Overleaf**, upload `ieee_access_overleaf.zip` and set the compiler to pdfLaTeX.

## Before submission

The open items (final-test scores, provenance checks, ablations, author details) are listed in [../REVIEW_RESPONSE.md](../REVIEW_RESPONSE.md). Every `\TODO{}` prints in red in both PDFs.

## Where the numbers come from

| Numbers | Source |
|---|---|
| Development-partition scores and decisions | `audit/07_experiments.csv`, `our_submission_notebook/_index.csv`, `our_submission_notebook/README.md` and the root `README.md` |
| Label agreement (Table II) | `audit/04_label_source_comparison.md` |
| gold-58 counts, bootstrap CIs, 385-study resolution | `../code/analysis/stats.json`, recomputed from the stored predictions by the script |
| Student gold-58 values | `audit/07_experiments.csv` and the `--s1` docstring in `ours_leg/build_h960.py` |
