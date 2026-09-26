# 01 — Competition facts: verified vs unverified (2026-09-26)

| Fact | Status | Evidence |
|---|---|---|
| Competition: RSNA Knee Abnormality Detection, `rsna-knee-abnormality-detection` | VERIFIED | Kaggle API competitions/list |
| Metric: ROC AUC | VERIFIED (API field "Roc Auc Score") | API |
| Macro average over the 12 columns | CONSISTENT, not read from rules page | user statement; every public pipeline ranks per column; one notebook comment calls it "macro AUC" |
| 12 targets: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture | VERIFIED | schema assertions against `sample_submission.csv` in public code; `targets` in `pilkwang/rsna-knee-weights/manifest.json` |
| Train studies: 4,407 | VERIFIED | commit-run logs compute `len(train.csv)` = 4407 |
| Train series ≈ 24,371 | UNVERIFIED | `train_series.csv` download returned HTTP 429 (rate-limited) |
| Studies with official labels: 58 | VERIFIED indirectly (3 independent sources) | (a) 58 unique IDs embedded in `kjy666/bend-the-knee-to-speedy-raptors` (saved: `raw/api/gold58_ids.json`); (b) the shared calibrator was fit on N = 4349 = 4407 − 58 studies; (c) sushanth's notebook reports "official labels in train.csv: 58 studies, 18 Fracture positives" |
| Visible test during interactive/commit runs: 3 studies | VERIFIED | commit-run logs ("3 test studies") |
| Hidden test size | UNKNOWN | D4 author comment "~1300" (unverified). The log line "1322 test studies" is `0.3 × 4407`, a cache heuristic, **not** the test size |
| Submission via notebooks only | VERIFIED | API `isKernelsSubmissionsOnly: true` |
| Deadline | VERIFIED: 2026-10-22 23:59 UTC (26 days from today) | API |
| Max daily submissions | VERIFIED: 5 | API |
| Teams / prize | VERIFIED: 4,386 teams, 77,000 USD | API |
| You have entered | VERIFIED (`userHasEntered: true`) | API |
| ≤ 9 h runtime, internet off | CONSISTENT, rules page not fetched | all 59 audited notebooks have internet disabled; public code budgets `9.0 h − elapsed` and `TIME_BUDGET = 8 h` |
| External data/models allowed | NOT ASSESSED | the ecosystem uses DINOv2 (Kaggle Models) and RadImageNet ResNet-50 (third-party dataset). **Check the rules page and the RadImageNet license before we depend on either** |

## Label sources publicly available (for Phase E)

From `sushanthtiruvaipati/rsna-knee-0942-rad-fracture-oof` markdown. Counts are as that author
reports them; not yet downloaded by us.

| Dataset | Studies | Fracture positives |
|---|---|---|
| `pilkwang/rsna-knee-llm-labels` (attached to 56/59 top notebooks) | 4406 | 314 |
| `stevenleehans/rsna-knee-llm-report-labels` | 4407 | 336 |
| `lixin73/rsna-knee-llm-report-labels-sol56` | 4407 | 300 |
| `dreaddevelopment/rsna-knee-labels` | 4349 | 159 |
| `yunusgmsoy/…-4-source-merged` (full slug not given) | 4407 | 311 |
| official `train.csv` | 58 | 18 |

The Fracture positive rate differs about 2× between sources (159 vs 336). Report-label choice is a
first-order variable.

## Scale facts that shape validation

- Public DINO members score 0.83–0.86 `holdout` AUC against report labels, while the blends score
  about 0.94 on LB. Report-label noise caps measurable OOF AUC, so **OOF deltas are informative,
  OOF levels are not comparable to LB**.
- Gold-58: several targets have fewer than about 20 positives (Fracture has 18). Per-target gold AUC
  CIs will be wide, roughly ±0.05–0.10. It is a sanity guard, not a selector.
- Gold-58 has been used to select public CoAt checkpoints (see `03_public_notebook_lineage.md` §5),
  so the public stack's gold-58 score is optimistically biased.

## Our working environment (as found)

| Item | State | Consequence |
|---|---|---|
| Local project folder | empty apart from `.DS_Store` and an empty `token.txt`. `.DS_Store` references deleted folders `Rsna knee 0936 to 0950/` and `v1/` | no baseline to protect; nothing to reproduce locally |
| Local Python | `/usr/bin/python3` blocked: "You have not agreed to the Xcode license agreements". A working Python 3.12 exists at `../Hyperspectral Object Detection Challenge 2026/.venv/bin/python3`, but it belongs to another project | for local analysis either accept the Xcode license (`sudo xcodebuild -license accept`) or create a dedicated venv for this project; I did not install anything into the other project's venv |
| GPU | none locally | all training/inference must run on Kaggle (typically ~30 GPU-h/week per account; check yours) |
| Kaggle auth | `~/.kaggle/access_token` works as a Bearer token against the REST and internal APIs; the installed `kaggle` CLI 1.7.4.5 cannot use it | the audit used `curl` + `jq` + `ruby` |
| Competition data download | small CSVs returned HTTP 429 | retry later or use Kaggle notebooks (the data is mounted there) |
