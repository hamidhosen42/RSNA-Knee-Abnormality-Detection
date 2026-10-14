# Is the Osteoarthritis Initiative (OAI) permitted as external data?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819
- **Author:** **John Doe** (@osamurai, Expert) · _topic author_ · 2026-09-17 17:03 UTC · ▲2 · LB rank 117
- **Topic votes:** 2 · **Comments/replies:** 5 · **Pinned:** no
- **Last comment:** 2026-09-21 21:48 UTC by OmerZalman
- **Cloned:** 2026-10-14 18:23 UTC (topic id 741819, raw JSON `_raw/741819.json`)

---

@javaduke95 A narrow follow-up, naming one dataset rather than a category.

In https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734109
the host ruled that KneeCoT is not permitted "due to the required data
agreement with dataset institution". The broader question about free
click-through research datasets
(https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733652)
is still open, and
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737950
asks about OAI but has no answer yet.

Could you rule specifically on the Osteoarthritis Initiative (OAI)? It is free
of charge and is distributed through the NIMH Data Archive (NDA), which
requires registering and agreeing to data use terms before download.

Two points would settle it:

1. Is OAI permitted as External Data under Section 2.6?
2. If NDA access to OAI requires an institutional sign-off (for example, a
   Data Use Certification signed by the participant's institution), does the
   KneeCoT reasoning apply to OAI as well?

A yes or no on OAI would be enough. Thank you.

---

## Comments (5, threaded, oldest first)

### [1] Po-Hao "Howard" Chen

**Po-Hao "Howard" Chen** (@javaduke95, Contributor) · _competition host_ · 2026-09-21 15:34 UTC · ▲1

Using OAI as an external dataset does not violate the RSNA Knee Abnormality Detection challenge rules if it is generally accessible to all participants without an escalated (i.e. institutional) review and sign-off or access to a legal team, and is free of charge.

We do not own or control the OAI dataset and cannot comment on whether this challenge complies OAI data-use terms from the dataset stewards' perspective.

#### ↳ [1.1] Optimo

**Optimo** (@optimo, Master) · 2026-09-21 20:08 UTC · ▲1 · LB rank 53

It's difficult to be less clear :)

##### ↳ ↳ [1.1.1] OmerZalman

**OmerZalman** (@omerzalman, Contributor) · 2026-09-21 20:36 UTC

does this mean yes or no lol?

##### ↳ ↳ [1.1.2] Tucker Arrants

**Tucker Arrants** (@tuckerarrants, Master) · 2026-09-21 20:56 UTC · ▲6 · LB rank 19

It is a no, as it requires institutional sign off. 

Virtually all external datasets that are remotely useful here are not allowed. They are either not “freely and generally accessible to all” or they **explicitly** carry a no derivatives clause/license.

Trying to find labeled knee MRI datasets to use here is a waste of time. I would love to be proven wrong, but on applying the host own criteria and reading the usage requirements of the data providers, none pass both.

##### ↳ ↳ [1.1.3] OmerZalman

**OmerZalman** (@omerzalman, Contributor) · 2026-09-21 21:48 UTC

what datasets did you use?
