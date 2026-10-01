# Rules clarification: external knee-MRI datasets, and using an LLM API to derive labels from the reports

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733652
- **Author:** **Fernando Faria** (@fernandoadefaria, Contributor) · _topic author_ · 2026-08-07 18:25 UTC · ▲21 · LB rank 1307
- **Topic votes:** 21 · **Comments/replies:** 7 · **Pinned:** no
- **Last comment:** 2026-08-20 09:27 UTC by Mazzutti
- **Cloned:** 2026-10-01 19:24 UTC (topic id 733652, raw JSON `_raw/733652.json`)

---

1. Do external knee-MRI datasets with free research-use agreements count as "publicly available and equally accessible at no cost"?

The rules permit external data that is freely and publicly available to all participants. The main public knee-MRI resources — MRNet (Stanford), fastMRI+ (NYU), the Osteoarthritis Initiative, and SKM-TEA — are all free of charge, but each requires registering and accepting a research-use agreement before download. None of them charge, and none restrict access to a particular institution, but the click-through step means they aren't anonymously downloadable.

Does that satisfy the "equally accessible at no cost" condition? A yes/no on the category would be more useful than a ruling on any single dataset, since the same access pattern covers essentially every public MSK-MRI resource.

Related: does the answer change for pretrained weights released under a research-use agreement rather than a permissive licence, given the winners' obligation to open-source code and weights under CC-BY-NC 4.0?

2. Is it permitted to send the training reports to a commercial LLM API in order to derive labels from them?

The competition explicitly suggests deriving labels for the unlabeled studies from the report text, and the reports are in nine languages, so a multilingual LLM is the natural tool. But the data-security rule says competition data must not be provided to anyone who has not accepted the rules, and it isn't obvious to me whether sending report text to a third-party API for inference counts as "providing data" or simply as using a tool.

Concretely, is either of these permitted?

(a) Sending report text to a hosted LLM API (OpenAI, Anthropic, Google, etc.) during development, with the derived labels used purely as a training artifact — no report text and no API access at submission time.
(b) Same, but with an open-weights model run locally or in a Kaggle notebook, so the text never leaves the participant's own environment.
I assume (b) is fine and am mainly asking about (a). If (a) is not permitted, that's very much worth everyone knowing before they build on it — and if it is, saying so publicly puts teams without local GPU capacity on the same footing as those with it.

Thanks. Also thanks for the label-criteria post, which was unusually detailed and useful.

---

## Comments (7, threaded, oldest first)

### [1] Oleksii Zhukov

**Oleksii Zhukov** (@zhukovoleksiy, Master) · 2026-08-07 20:22 UTC · ▲1 · LB rank 2850

@fernandoadefaria Good questions, and I'd like hosts answers to both.  

1. Same question here, I don't think a participant can settle it. The winners' obligation angle you raise makes it two questions: whether click-through datasets are usable at all, and whether using them complicates the CC-BY-NC grant. The rules do carve out incompatible licences on input data and pretrained models, which may cover it, but that's the host's call.

2. My reading is that (a) is not permitted. In competition rules, Section 4.b "Data Security" says participants agree not to "transmit, duplicate, publish, redistribute or otherwise provide or make available the Competition Data to any party not participating in the Competition." Obviously commercial API provider is such a party. That reading doesn't depend on what happens at submission time, since the transmission of reports data itself is the issue and violation of competition rules.

I agree (b) looks fine and that was my initial plan to generate labels: with open weights run locally or in a Kaggle notebook the text never leaves your own environment, and internet-off only applies to the submission notebook, so offline label generation during development isn't restricted. Small multilingual models are great for this task.

But this is my reading of the clause, not a ruling, and I think it's a large enough decision that a host answer would be worth having in writing. I agree it matters for fairness either way. If (a) is allowed it advantages teams who can pay for API calls; if it isn't, teams should know before they build on it to not get DQ.
@javaduke95

### [2] NNMax

**NNMax** (@ashok205, Expert) · 2026-08-08 09:32 UTC · ▲1

> The use of external data and models is acceptable unless specifically prohibited by the Host. Because of the potential costs or restrictions (e.g., “geo restrictions”) associated with obtaining rights to use external data or certain software and associated tools, their use must be “reasonably accessible to all” and of “minimal cost”

LLMs are indeed allowed as far as I can think of. You can get more info on the **6. EXTERNAL DATA AND TOOLS** section in the competition rules page



> In the event that input data or pretrained models with an incompatible license are used to generate your winning solution, you do not need to grant an open source license in the preceding Section for that data and/or model(s). 

Public datasets are always allowed in most of the kaggle competitions regardless of their license types unless the rules explicitly says so. If it doesn't allow commercial use but only research use, then you must mention that clearly in your writeup as the competition involves prize money which accounts to solutions being used for commercial purpose.

#### ↳ [2.1] Po-Hao "Howard" Chen

**Po-Hao "Howard" Chen** (@javaduke95, Contributor) · _competition host_ · 2026-08-09 09:57 UTC · ▲3

This point is worth settling clearly.  Thank you for pointing it out. 

[This post](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965) was just added and hopefully adds clarity.

### [3] k256.dev

**k256.dev** (@k256net, Master) · 2026-08-08 11:54 UTC · ▲2 · LB rank 1946

I believe that sending the report text to commercial LLMs (OpenAI, Anthropic, Google, etc.) is not permitted under the rules. For that reason, I believe labeling has to be done either with locally hosted LLMs (such as Qwen) or by human annotation.

However, there is already a problem: label datasets derived using GPT have been publicly shared on the platform. Other public datasets may likewise have been labeled with commercial LLMs. If using such datasets is to be considered a rule violation, I don't see how the competition hosts or Kaggle staff could realistically make that determination for any given dataset. I would appreciate urgent, official guidance on how this will be handled.

#### ↳ [3.1] k256.dev

**k256.dev** (@k256net, Master) · 2026-08-09 10:11 UTC · ▲1 · LB rank 1946

The host just made an announcement to everyone.

For more details, please see https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965

### [4] Nicolai Karcher

**Nicolai Karcher** (@pejophylo, Contributor) · 2026-08-09 08:25 UTC · LB rank 785

any verdict on any of this? :)

### [5] Mazzutti

**Mazzutti** (@profmazzutti, Contributor) · 2026-08-20 09:27 UTC · LB rank 18

I pulled the actual rule text, and I think this thread (and [#735497](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735497), [#734109](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734109), [#735121](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735121)) may be asking about the wrong branch of 2.6(a).

2.6(a) is a disjunction: External Data must be "either publicly available and equally accessible ... at no cost to the other Participants, **or** satisfies the Reasonableness criteria as outlined in Section 2.6.b."

And 2.6.b states that "the use of external data and models is acceptable **unless specifically prohibited by the Host**," with the Reasonableness Standard aimed at excessive cost and geo restrictions.

MRNet, fastMRI/fastMRI+, OAI, SKM-TEA and KneeCoT are free, not geo-restricted, and open to any researcher after a click-through agreement. On a plain reading they clear 2.6.b even if one thinks the click-through defeats the first branch
of 2.6(a).

So the question I'd ask the Host is narrower:

**1. Has the Host specifically prohibited any public knee-MRI corpus under 2.6.b?** If not, pretraining on that category is permitted and these four threads are settled.

**2. Section 2.8(a)(1) requires winners to publish model weights as a public Kaggle dataset, and 2.8(a)(2) requires representing an "unrestricted right to grant" the winners licence.** Can weights whose training data carried a non-commercial or no-derivatives licence (CC BY-NC, CC BY-NC-ND) satisfy that representation? 2.6(a) explicitly says permission to use External Data "does not limit your other obligations ... including Section 2.8."

Question 2 is the one that actually gates decisions: permission to train and eligibility to win may differ, and teams are committing pretraining compute now against an October 22 deadline.
