# RSNA Knee — submission 56338136 fails during hidden rerun; V20 CPU run succeeds

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742122
- **Author:** **Danilo Pedroso** (@danilopedroso, Contributor) · _topic author_ · 2026-09-20 00:35 UTC · ▲-1 · LB rank 3765
- **Topic votes:** -1 · **Comments/replies:** 0 · **Pinned:** no
- **Last comment:** 2026-09-20 00:35 UTC by Danilo Pedroso
- **Cloned:** 2026-10-01 19:24 UTC (topic id 742122, raw JSON `_raw/742122.json`)

---

Hello Kaggle Support,

I am danilopedroso. Please advise on the failed hidden rerun for:

Competition: rsna-knee-abnormality-detection
Notebook: danilopedroso/rsna-knee-30-main-hybrid-v2-v1-lateral-meniscus
Resource: 130569806
Source Version: 20
Submission ID: 56338136

At 2026-09-19 00:29:02 UTC (API observation time), the official API returned COMPLETE with an error description and no Public Score:

“Your notebook hit an unhandled error while rerunning your code.”

This indicates processing ended with an error, not successful scoring. No exception class, traceback, failure stage, or reliable hidden-rerun duration was available in that response.

The expanded error view was not available for inspection. The error message quoted above was obtained from the official API.

The private V20 notebook uses CPU for both models, with no accelerator and GPU, TPU, and Internet disabled, and two public weight inputs. Its visible execution completed and its submission.csv was validated. This does not establish that the hidden rerun succeeded.

Our investigation passed 37 distinct local tests. Structural scenarios up to 1,000 studies passed with stubs, which do not reproduce the full pipeline’s resource usage. We did not reproduce OOM. The internal deadline and output rollback behaved as designed. V2 preprocessing rejects a synthetic case with differently sized slices; we have no evidence this occurred in hidden data or belongs to the inputs our code must support. No speculative patch was applied.

Could you advise, within the competition’s disclosure restrictions:

A. Is any additional error category or sanitized diagnostic guidance shareable for this submission?

B. Is there an indication of a violation of a published environment, resource, input, or format requirement that can be explained without revealing hidden-set characteristics?

C. If the exception class, stage, or duration cannot be disclosed, what official diagnostic procedure is recommended?

D. Is there public documentation of guarantees about image dimensions across slices within a series, relevant to preprocessing? We seek only the general public contract, not whether a condition occurs in hidden data.

The cause remains unknown. We are not asserting a platform bug or requesting hidden data, labels, or individual records. No sensitive files or private source code are attached. We have not repeated the submission.

Thank you.

---

## Comments (0, threaded, oldest first)

_No comments._
