# Please clarify which external knee MRI datasets are allowed

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416
- **Author:** **YYama** (@yosukeyama, Master) · _topic author_ · 2026-09-26 01:52 UTC · ▲15 · LB rank 52
- **Topic votes:** 15 · **Comments/replies:** 2 · **Pinned:** no
- **Last comment:** 2026-09-26 05:50 UTC by YYama
- **Cloned:** 2026-09-26 16:24 UTC (topic id 743416, raw JSON `_raw/743416.json`)

---

Thank you to the hosts for organizing this challenge and for taking the time to answer questions throughout the competition. I appreciate that assessing third-party datasets individually is difficult, especially when their access requirements and licenses differ.

Questions about external knee MRI data are spread across several threads. Without a clear answer for commonly used datasets, teams may train under different assumptions about what is permitted, which makes it difficult to ensure a level playing field.

I have read the [original question](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733652), the [registration-gated dataset question](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735497), and the [institutional-agreement follow-up](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737950). There are also separate questions about [pretrained weights](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735121) and [research-use agreements](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738111).

The host has [explicitly ruled that KneeCoT is not allowed](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734109) because its formal institutional agreement may create unequal access. The host has also [explained](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965) that non-commercial terms alone do not exclude a dataset, while straightforward registration and substantial institutional approval may be treated differently. It remains unclear to me how these principles apply to the following resources:

| Resource | Relevant access or licensing issue |
| --- | --- |
| [Stanford MRNet](https://aimi.stanford.edu/datasets/mrnet-knee-mris) | Research-use agreement |
| [Stanford SKM-TEA](https://aimi.stanford.edu/datasets/skm-tea-knee-mri) | Registration and data-use terms |
| [NYU fastMRI](https://fastmri.med.nyu.edu/) and [fastMRI+](https://github.com/microsoft/fastmri-plus) | Data-sharing agreement for images; separately released annotations |
| [Osteoarthritis Initiative (OAI)](https://nda.nih.gov/oai) | Data Use Certification and institutional requirements |
| [KneeMRI](https://zenodo.org/records/14789903) | CC BY-NC-ND 4.0 |
| [KneeXNet-2.5D](https://github.com/pitthexai/Knee_MRI_Segmentation_2.5D) | Public OAI-derived annotated images and pretrained weights; no explicit repository license that I could identify |

Could the hosts give a **yes or no for each resource** under this competition's external-data rules? If the answer differs between training on the images and fine-tuning publicly released weights, please state that separately. It would also help to know whether using each resource is compatible with the winner's requirement to release the final model. I understand that teams remain responsible for complying with the original providers' terms.

Because the same question keeps appearing in different threads, could the hosts **pin the answer in the Discussion** so all teams can find and follow the same guidance?

@javaduke95

---

## Comments (2, threaded, oldest first)

### [1] Tucker Arrants

**Tucker Arrants** (@tuckerarrants, Master) · 2026-09-26 03:25 UTC · ▲2 · LB rank 19

This is quite pressing and could really throw off the leaderboard. I could be wrong, but it's only the winning solutions (top 10) that would be audited thoroughly enough that training on external data sources that are not allowed would even get "caught" and disqualified. Basically everyone else could probably get away with it. This is a problem as it can totally reshape the final non top 10 rankings. I think external data, even small amounts of it, can substantially increase score in this competition so the impact of this could be extreme.

#### ↳ [1.1] YYama

**YYama** (@yosukeyama, Master) · _topic author_ · 2026-09-26 05:50 UTC · ▲3 · LB rank 52

I couldn't agree more. I worry that detailed review may naturally focus on prize-eligible teams, while the effects of unclear external-data rules could extend across the entire leaderboard.

What particularly concerns me is that relevant external knee MRI data could allow even a fairly simple approach to achieve a dramatic improvement in score. If some teams use those datasets while others refrain because the rules are unclear, the competition becomes substantially unequal. I would appreciate explicit guidance from the hosts on which datasets are allowed.
