# Use of Commercially Hosted LLMs

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965
- **Author:** **Po-Hao "Howard" Chen** (@javaduke95, Contributor) · _competition host_ · 2026-08-09 09:52 UTC · ▲20
- **Topic votes:** 20 · **Comments/replies:** 13 · **Pinned:** yes
- **Last comment:** 2026-09-23 20:31 UTC by Danny Daniel
- **Cloned:** 2026-09-26 16:24 UTC (topic id 733965, raw JSON `_raw/733965.json`)

---

Use of commercially hosted LLMs and other external inference services is permitted, provided that the service and method of use otherwise comply with the Competition Rules, including requirements that external data, models, software, and associated tools be reasonably accessible to all participants and of minimal cost. 

In other words, for purposes of this competition, submitting Competition Data, including report text, to an external LLM or API for inference or other computational processing (for example, extracting labels from reports) will not, by itself, be considered prohibited **PRIVATE SHARING** of Competition Data outside the Team.

Please refer to the [RSNA Knee Abnormality Detection Competition Rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules).  This is in accordance to Section 2.6.b, **EXTERNAL DATA AND TOOLS**.

The **PRIVATE SHARING** restriction is intended to prohibit sharing Competition Data, code, or competition-specific work product with other participants, teams, or third parties for collaboration or competitive use outside the registered Team.

Participants remain responsible for ensuring that any external service complies with all applicable Competition Rules and the service's terms of use, including the requirements for reasonable accessibility and minimal cost.

The Competition Host reserves the right to determine whether a particular service, model, or configuration is reasonably accessible, is prohibitively costly, or otherwise creates an unfair competitive advantage.

---

## Comments (13, threaded, oldest first)

### [1] Nicolai Karcher

**Nicolai Karcher** (@pejophylo, Contributor) · 2026-08-09 10:23 UTC · ▲4 · LB rank 784

"Participants remain responsible for ensuring that any external service complies with all applicable Competition Rules and the service's terms of use, including the requirements for reasonable accessibility and minimal cost.

The Competition Host reserves the right to determine whether a particular service, model, or configuration is reasonably accessible, is prohibitively costly, or otherwise creates an unfair competitive advantage."



Could you give us your assessment regarding the use of common public MRI datasets for this challenge? Many of them only allow research (and no commercial) use, and getting a price to me could be classified as commercial use (I'm not a lawyer though so I don't know).

For example, [this dataset](https://stanford.redivis.com/datasets/4a2c-4cpkzrn2c?access&requirement=474) specifically states: "Permission is granted to view and use the Dataset without charge for personal, **non-commercial research purposes only**. Any commercial use, sale, or **other monetization is prohibited**."

I understand these restrictions are describing external datasets, so we should probably get in contact with those databases. But I was wondering what your independent assessment would be. Thanks!

#### ↳ [1.1] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-08-18 15:07 UTC · LB rank 635

That's a great suggestion - contact the source and ask if they are ok with you using the data in a kaggle competition.  Not sure if they would reply, but also past experience suggest kaggle reluctant to reply to this type of question during the competition.

### [2] Ragheb haddara

**Ragheb haddara** (@l4r3nn, Contributor) · 2026-08-10 21:45 UTC · ▲1 · LB rank 4224

The rules about commercially hosted LLMs are important here. Since knee MRI interpretation requires domain expertise, it's tempting to use an LLM to help with reasoning. But the competition rules likely restrict this to ensure the solutions are based on the image data, not external reasoning. Check the rules page carefully before incorporating any LLM-based features.

### [3] UGUR OZER

**UGUR OZER** (@aeonairisk, Contributor) · 2026-08-14 21:30 UTC · ▲1 · LB rank 2572

To make the dataset question above concrete: the Osteoarthritis Initiative (OAI) is free of charge and open to any researcher after a click-through data use agreement, but the agreement limits use to non-commercial research. The same access pattern covers MRNet, fastMRI+ and SKM-TEA, so a ruling on the category would settle it for essentially every public knee-MRI resource. Given the prize money, could the host confirm whether external datasets under research-only agreements are permitted, and if so whether the winners' open-source obligation is affected? Teams need this before committing pretraining work ahead of the entry deadline. Thank you.

#### ↳ [3.1] Po-Hao "Howard" Chen

**Po-Hao "Howard" Chen** (@javaduke95, Contributor) · _competition host_ · 2026-08-27 22:14 UTC · ▲6

From the competition’s perspective, use of these datasets would be considered non-commercial; receipt of competition prize money does not, by itself, make their use a commercial sale or commercial exploitation. These datasets would not be excluded by the competition rules solely because their licenses restrict use to non-commercial purposes.

A separate consideration is whether an external dataset is reasonably accessible to all competitors. Datasets that can be accessed through a straightforward registration or click-through data-use agreement would generally satisfy this requirement. By contrast, datasets requiring institution-specific approvals, negotiated written agreements, IRB approval, lengthy credentialing, or other substantial administrative steps may present a meaningful accessibility barrier, particularly given the remaining competition timeline. Such requirements could therefore affect whether a dataset is considered allowable under the competition’s external-data rules, independent of whether its use is characterized as commercial or non-commercial.

Our job (Host) is tough here because RSNA and Kaggle do not own or control these external datasets and cannot interpret or waive their individual data-use agreements, expedite requests and so on, on behalf of the dataset owners. Teams remain responsible for ensuring that their intended use, including participation in a prize-bearing competition and any required release of code or models, is permitted under the applicable dataset license or agreement.

Therefore, to the extent the question concerns the *competition rules*, readily accessible datasets with non-commercial restrictions are not prohibited on that basis alone. Dataset-specific licensing obligations remain the responsibility of each team, and datasets involving substantial approval or access hurdles may need to be considered separately under the competition’s equal-accessibility requirements.

Sorry for the verbose response. I hope it's at least a little helpful...

### [4] Oscar Yáñez Feijóo

**Oscar Yáñez Feijóo** (@oscaryezfeijo, Contributor) · 2026-08-18 09:26 UTC · LB rank 2335

Thanks for clarifying this. Allowing hosted LLMs opens some interesting possibilities for extracting structured labels or features from the reports. Would using LLM-derived labels or embeddings from the report text be fully acceptable as model inputs at both training and inference time? Thanks in advance

#### ↳ [4.1] Po-Hao "Howard" Chen

**Po-Hao "Howard" Chen** (@javaduke95, Contributor) · _competition host_ · 2026-08-27 22:05 UTC

"Would using LLM-derived labels or embeddings from the report text be fully acceptable as model inputs at both training and inference time?" 

Yes. Just make sure you read some of the other discussion threads about what the model is provided as input at "inference time."

### [5] Mayobanex Santana

**Mayobanex Santana** (@mayobanexsantana, Contributor) · 2026-08-21 21:48 UTC · ▲1

Ok, by the sentence: *"(for example, extracting labels from reports) will not, by itself, be considered prohibited PRIVATE SHARING of Competition Data outside the Team."* i can conclude that i can use LLM API (i.e. openAI) to read the reports, which are actually in diifferent languages, to generate the labels, so i don't have to struggle?
I know that the answer may be abvious but i just want to make 100% sure i can save me so much trouble and focus on model not in getting the target labels without been kicked out from the competence.
Thanks!

#### ↳ [5.1] Po-Hao "Howard" Chen

**Po-Hao "Howard" Chen** (@javaduke95, Contributor) · _competition host_ · 2026-08-27 21:59 UTC · ▲1

You can use LLM API, such as those from OpenAI, to read the reports to generate the labels.

### [6] Sebastian Garcia

**Sebastian Garcia** (@sebgarciavillalobos, Contributor) · 2026-09-05 06:31 UTC

@javaduke95 Thanks for all the clarifications.

Would it be allowed to publish a Kaggle Dataset like the one below if it contains the original radiology report text from the competition data, or fragments/excerpts of that text?

https://www.kaggle.com/datasets/laymond/rsna-knee-abnormality-qwen3-8b-weak-labels

I’m specifically wondering whether including either the full report text or extracted portions of it in a public dataset would be considered redistribution of Competition Data under the competition rules.

### [7] Navneet

**Navneet** (@navneetbende, Contributor) · 2026-09-05 07:20 UTC · ▲-1

Thank you for the Use of Commercially Hosted LLMs info @javaduke95

### [8] Manish

**Manish** (@manishpatil115, Contributor) · 2026-09-06 16:59 UTC · ▲-1

This is a very welcome clarification from the organizers. Allowing the use of commercially hosted LLMs (like OpenAI, Anthropic, or Google APIs) for tasks like text parsing, label extraction, or metadata enrichment really opens up creative pipeline designs especially for handling the report text efficiently.It’s great to see the rules explicitly state that API inference doesn't count as prohibited private sharing, provided the tools remain lowcost and accessible to everyone. It gives teams the green light to leverage state-of-the-art text processing without the fear of accidental disqualification. Thanks for highlighting this section!

### [9] Danny Daniel

**Danny Daniel** (@danny514, Contributor) · 2026-09-23 20:31 UTC · ▲-1

Thanks for the clarification! Just one quick question: if LLM-generated labels/embeddings from the reports are allowed, would it also be okay to use an external LLM to extract structured features like findings, laterality, or severity and combine those with MRI features?

And would this be allowed at inference time only if those same features can be generated from the information available at inference?
