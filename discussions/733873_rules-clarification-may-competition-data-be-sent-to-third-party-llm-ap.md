# Rules clarification: may Competition Data be sent to third-party LLM APIs?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733873
- **Author:** **FHZ982** (@fhz982, Contributor) · _topic author_ · 2026-08-08 19:04 UTC · ▲3
- **Topic votes:** 3 · **Comments/replies:** 1 · **Pinned:** no
- **Last comment:** 2026-08-09 09:59 UTC by Po-Hao "Howard" Chen
- **Cloned:** 2026-09-26 16:24 UTC (topic id 733873, raw JSON `_raw/733873.json`)

---

Only 58 of the 4,407 training studies carry per-condition labels, so a lot of us are deriving labels from the Report text, and most of those pipelines involve a large language model. I would like to get a clear answer on where the line is, because two clauses seem to point in different directions and I would rather ask than guess.

Section 2.6 (External Data and Tools) treats external models as acceptable unless the Host prohibits them, and gives a commercial large language model as an example of a tool that meets the Reasonableness Standard. That reads as though using a hosted LLM is expected.

Section 2.4.b (Data Security) says participants agree not to "transmit, duplicate, publish, redistribute or otherwise provide or make available the Competition Data to any party not participating in the Competition." The RSNA MIRA Dataset Research Use Agreement, incorporated by reference in 2.4.a, is stricter still: §4 prohibits distributing or reproducing any portion of a dataset without prior written permission, and §5 prohibits sharing access.

Sending a report to a hosted model is, on a literal reading of 2.4.b, providing Competition Data to a party that is not participating. On the reading suggested by 2.6, it is ordinary tool use.

Four specific questions:

Does passing Report text to a third-party hosted LLM API count as transmitting Competition Data to a non-participating party under 2.4.b?
If the answer depends on the provider's terms, does an enterprise or API configuration with zero data retention and no training on submitted data change it?
Are locally-run open-weights models (nothing leaves the participant's machine) unambiguously acceptable? That is the conservative option, and it would be useful to have it confirmed.
Does the answer differ for the DICOM images versus the report text? The reports are free-text clinical documents, which feels like a different category from pixel data.

If the answer is that hosted APIs are not permitted, an early and visible statement would help a lot of people — several public notebooks already describe LLM-based label derivation, and I doubt anyone has read this as a restriction.

Thanks.

---

## Comments (1, threaded, oldest first)

### [1] Po-Hao "Howard" Chen

**Po-Hao "Howard" Chen** (@javaduke95, Contributor) · _competition host_ · 2026-08-09 09:59 UTC

This point is worth settling clearly.  Thank you for pointing it out. 

[This post](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965) was just added and hopefully adds clarity.
