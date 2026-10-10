# Public notebook landscape, 2026-10-09 (top of LB = 0.951)

Checked with Kaggle's view model (data sources of each notebook's scored version).

| Notebook | LB | OAI-linked sources | Other sources |
|---|---|---|---|
| heliosli/rsna-knee-blend-gold v1 | 0.951 | nartaa publication-swa-weights (RSNA+OAI), heliosli orthofoundation (bundles SKM-TEA / KneeXNet auxiliary weights) | heliosli/rsna-knee-compact-v1 |
| evgendvorkin/rsna-versia-5 v19 | 0.950 | nartaa | goodpjw2008 reader |
| romantamrazov/rsna-knee-dinosaur-v5 v30 | 0.950 | nartaa, heliosli orthofoundation | heliosli compact-v1 |
| sujanmajhisuzan apex-grandmaster v1 (+ forks hitarthjain0 v5, rian55, sameerk2004, rabari9999, kozykappa, matterhorn3838 d4/velciraptor) | 0.950 | nartaa | goodpjw2008 reader (+ some CoAt resgated) |

**Every public notebook at 0.946 or above depends on nartaa's RSNA+OAI checkpoint** (licence "Other", research/educational use). The 0.951 notebook also adds heliosli's orthofoundation bundle with SKM-TEA and KneeXNet weights.

**Rule status:**
- KneeCoT is prohibited.
- OAI is allowed only if it can be obtained without institutional review or sign-off (host, topic 741819). The OAI DUC process requires such a sign-off.
- In topic 743416 the host has not given a per-dataset yes/no.

So these notebooks carry a HIGH disqualification risk. The rule-clean public ceiling is about 0.945 (Tri-Specialist and our E18/E19).

## Why the team's fork of blend-gold failed (submission 56996760)

The fork mounted the latest `heliosli/rsna-knee-compact-v1` (v10, bundle_id 209a32c8b50a6b1c). Notebook cell 3 asserts manifest bundle_id == a5e76db599f5bb66 and an S5 file hash of c7ebab1eb953f1a6, and that combination is **v9**. The assert failed already in the commit run, so the submission errored.

Fix: pin `heliosli/rsna-knee-compact-v1/9`, plus the other two sources at their scored versions. A new commit run is needed, which needs a GPU.
