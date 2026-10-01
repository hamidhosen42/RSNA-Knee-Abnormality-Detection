# Please clarify which external knee MRI datasets are allowed

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416
- **Author:** **YYama** (@yosukeyama, Master) · _topic author_ · 2026-09-26 01:52 UTC · ▲55 · LB rank 12
- **Topic votes:** 55 · **Comments/replies:** 12 · **Pinned:** no
- **Last comment:** 2026-10-01 07:56 UTC by Optimo
- **Cloned:** 2026-10-01 19:24 UTC (topic id 743416, raw JSON `_raw/743416.json`)

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

## Comments (12, threaded, oldest first)

### [1] Tucker Arrants

**Tucker Arrants** (@tuckerarrants, Master) · 2026-09-26 03:25 UTC · ▲5 · LB rank 34

This is quite pressing and could really throw off the leaderboard. I could be wrong, but it's only the winning solutions (top 10) that would be audited thoroughly enough that training on external data sources that are not allowed would even get "caught" and disqualified. Basically everyone else could probably get away with it. This is a problem as it can totally reshape the final non top 10 rankings. I think external data, even small amounts of it, can substantially increase score in this competition so the impact of this could be extreme.

#### ↳ [1.1] YYama

**YYama** (@yosukeyama, Master) · _topic author_ · 2026-09-26 05:50 UTC · ▲5 · LB rank 12

I couldn't agree more. I worry that detailed review may naturally focus on prize-eligible teams, while the effects of unclear external-data rules could extend across the entire leaderboard.

What particularly concerns me is that relevant external knee MRI data could allow even a fairly simple approach to achieve a dramatic improvement in score. If some teams use those datasets while others refrain because the rules are unclear, the competition becomes substantially unequal. I would appreciate explicit guidance from the hosts on which datasets are allowed.

### [2] Cemal Atas

**Cemal Atas** (@cemalata, Contributor) · 2026-09-27 11:19 UTC · ▲2 · LB rank 22

This is really important. I have a similar question for the hosts. Can we use publicly released pretrained models (MIT license and all) that were trained on OAI-derived data?

#### ↳ [2.1] YYama

**YYama** (@yosukeyama, Master) · _topic author_ · 2026-09-28 11:22 UTC · ▲1 · LB rank 12

I agree. Clear guidance on this point is important to ensure a level playing field for all participants.

### [3] madmax0404

**madmax0404** (@max6296, Expert) · 2026-09-30 08:03 UTC · ▲3

4 days and still no answer. is that a yes?

### [4] Pratham Lahoti

**Pratham Lahoti** (@prathamlahoti2, Contributor) · 2026-09-30 17:25 UTC · ▲2 · LB rank 551

++ This is really important, please provide clarification

### [5] Ali Ozan Memetoglu

**Ali Ozan Memetoglu** (@aliozanmemetoglu, Expert) · 2026-09-30 19:08 UTC · ▲3 · LB rank 226

MRNet’s user agreement says we’re not allowed to modify, reverse engineer, decompile, or create derivative works from the dataset, so that might be an issue too.

### [6] Eugene Khvedchenya

**Eugene Khvedchenya** (@bloodaxe, Grandmaster) · 2026-09-30 19:10 UTC · ▲5 · LB rank 466

Based on all my past kaggle experience, the most 'safe' action of host is to ignore the dataset clearance questions competely. 
Since to do it "the right way" requires involvement of legal department and I imagine it's tedious process.  I would be happy to proven wrong and get clear answer. But before that the best course of action is to use common sense: public registration is ok, non-commerial/research license is ok; gated dataset _requiring_ academia email - probably less ok, but if top 10 solutions will end up using it - also ok ;)

#### ↳ [6.1] pompom555

**pompom555** (@pompom555, Expert) · 2026-10-01 04:46 UTC · LB rank 103

@bloodaxe 

Are you aware of any past instances in similar competitions where the same external dataset was used in a gold medalist's write-up and didn't result in a ban?

I'm gonna also do my analysis once I scraped write-ups in the previous competitions. I'll post the result here.

### [7] Tucker Arrants

**Tucker Arrants** (@tuckerarrants, Master) · 2026-09-30 22:59 UTC · ▲2 · LB rank 34

I also think this is only going to get worse, largely due to the widespread adoption/use of LLM agents. They insist on finding and using external data, basically ignoring the fine print, to improve CV/LB score. I guess it's their reward training - you tell a LLM to "improve the score" and that is what it will try to do. 

If people are minimally "in the loop" or fully out of it, it is hard to catch. I've had to paste verbatim the host comments and the content of the Rules tab multiple times for LLMs to finally accept that most of these sources do not strictly comply with usage requirements, either from the host's perspective, Kaggle's perspective, or the data owner's perspective...

Not sure what the solution is but the host has no reason to check solutions other than the prize eligible ones, the data providers themselves are unlikely to respond, so what to do? "Cheat" and hope you don't place in top 10 but otherwise secure a medal? Play it safe and wonder if the people that placed above you weren't as conservative?

#### ↳ [7.1] pompom555

**pompom555** (@pompom555, Expert) · 2026-10-01 04:40 UTC · ▲1 · LB rank 103

We're now at the point where we're basically "jumping at shadows." Whether some users get banned could depend on the host, while some cheaters might still manage to get ahead of the pack. We're literally just rolling the dice.

### [8] Optimo

**Optimo** (@optimo, Master) · 2026-10-01 07:56 UTC · LB rank 57

Have you seen improvements by using external data? I haven't tried yet but this discussion makes me want to give it a try...
