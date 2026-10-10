# blend-gold (heliosli, public LB 0.951) — fixed fork

Code = heliosli/rsna-knee-blend-gold **v1** verbatim. The team's first fork failed (submission 56996760) because it
mounted heliosli/rsna-knee-compact-v1 **v10** while the notebook asserts the v9 bundle (bundle_id a5e76db599f5bb66,
S5 hash c7ebab1eb953f1a6). This copy pins compact-v1/9, nartaa publication-swa-weights/1, orthofoundation/1 and the
original docker image.

RULE RISK: uses nartaa's RSNA+OAI checkpoint and heliosli's orthofoundation bundle (SKM-TEA / KneeXNet auxiliary
weights). See audit/14_public_0951_landscape.md. Submitted at the user's explicit request (2026-10-09).
