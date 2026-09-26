# Metric edge cases: single-class label columns, and corrupted DICOMs in the test set

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740094
- **Author:** **Piterson465** (@piterson465, Contributor) · _topic author_ · 2026-09-08 08:13 UTC · LB rank 2115
- **Topic votes:** 0 · **Comments/replies:** 1 · **Pinned:** no
- **Last comment:** 2026-09-12 08:25 UTC by ParsaAshrafi
- **Cloned:** 2026-09-26 16:24 UTC (topic id 740094, raw JSON `_raw/740094.json`)

---

Two narrow questions about how the evaluation behaves in edge cases. Both change
how a submission is built, so I would rather ask than guess.

1. Single-class label columns.
   The metric is the unweighted mean of 12 per-label ROC AUCs. If one of the 12
   label columns happens to contain only one class in the private test set, what
   does the scorer do with it: skip that column and average over the remaining
   ones, substitute 0.5, or does the submission error out? With 12 labels and the
   rarer findings this seems worth pinning down, because "skip" and "substitute
   0.5" give materially different macro scores for the same predictions.

2. Corrupted DICOMs in the test set.
   Two threads report training studies whose PixelData is truncated relative to
   the Group 0028 header, so pydicom raises ValueError on .pixel_array
   (discussion/738708 and discussion/737163). Are the test studies known to be
   free of this, or should submissions be written to survive it? A single
   unhandled file would turn a working submission into a zero, so if the test set
   is not guaranteed clean it would help everyone to say so explicitly.

Thanks!

---

## Comments (1, threaded, oldest first)

### [1] ParsaAshrafi

**ParsaAshrafi** (@parsaashrafii, Contributor) · 2026-09-12 08:25 UTC · LB rank 1035

Good questions. I would handle both cases defensively in the submission:

1. For a single-class label, do not assume that the scorer skips the column or substitutes 0.5. In local validation, `roc_auc_score` is undefined for a single class, so the exact competition behavior should be confirmed by the host.

2. For corrupted DICOMs, wrap `pixel_array` in a guarded loader, log the failure, and use a deterministic fallback such as a neighboring valid slice or a zero-filled image. This prevents one malformed study from crashing the entire submission.

Until the host confirms the scorer behavior and test-data guarantees, treating both cases defensively is the safest approach.
