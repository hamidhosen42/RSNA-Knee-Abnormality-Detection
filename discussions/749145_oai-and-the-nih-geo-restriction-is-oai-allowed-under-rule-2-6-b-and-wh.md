# OAI and the NIH geo restriction: is OAI allowed under Rule 2.6.b, and what happens to teams and public notebooks that use it?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749145
- **Author:** **Mobassir** (@mobassir, Grandmaster) · _topic author_ · 2026-10-14 03:08 UTC · ▲12 · LB rank 1061
- **Topic votes:** 13 · **Comments/replies:** 4 · **Pinned:** no
- **Last comment:** 2026-10-14 07:36 UTC by Wildata
- **Cloned:** 2026-10-14 18:23 UTC (topic id 749145, raw JSON `_raw/749145.json`)

---

Hi @javaduke95, @michelleriopel and @shosys (competition hosts), and @ryanholbrook and @macruzbar (Kaggle team),

With about one week left before the final submission deadline (October 22), many teams still do not know whether OAI (the Osteoarthritis Initiative, distributed through the NIMH Data Archive, NDA) is allowed. The question has been asked several times without an answer:
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416#3534883
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748276
A large part of the leaderboard depends on it, so I would like to ask for a clear, pinned ruling.

What the rules say

Rule 2.6.a: External Data must be "publicly available and equally accessible to use by all Participants of the Competition for purposes of the competition at no cost to the other Participants", or meet the Reasonableness criteria of Rule 2.6.b.

Rule 2.6.b: "Because of the potential costs or restrictions (e.g., 'geo restrictions') associated with obtaining rights to use external data ... their use must be 'reasonably accessible to all' and of 'minimal cost'."

The eligibility rules exclude only residents of Crimea, DNR, LNR, Cuba, Iran and North Korea. Participants from China, Russia and Venezuela are eligible.

KneeCoT was ruled out because its access procedure could create unequal access:
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734109

On September 21 the host wrote that using OAI "does not violate the ... rules if it is generally accessible to all participants without an escalated (i.e. institutional) review and sign-off or access to a legal team, and is free of charge":
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819#3526495

The geo restriction

NIH notice NOT-OD-25-083 restricts access to NIH controlled-access data repositories from "countries of concern", effective April 4, 2025: China (including Hong Kong and Macau), Russia, Iran, North Korea, Cuba and Venezuela.
https://grants.nih.gov/grants/guide/notice-files/NOT-OD-25-083.html

NDA's own notice, as quoted in topic 748276, states that NIH prohibits access to "NDA data by researchers and institutions located in countries of concern, including China".
https://nda.nih.gov/nda/enhancing-dar-security-measures

Participants in China report that they cannot register: Login.gov rejects them with error LG22, even after trying several email addresses:
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792#3535178
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792#3536422
One participant reported a successful download from a China IP:
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792#3536289

A participant posted an NDA help-desk reply saying that OAI is open access for everyone and that the countries-of-concern rule does not apply:
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416#3535226
This contradicts the reports above, and participants cannot resolve the conflict themselves.

So OAI is free and needs no institutional sign-off, but for participants in at least one eligible country it appears not to be accessible in practice. This looks like exactly the "geo restrictions" case named in Rule 2.6.b.

Why this affects much of the leaderboard

The top-scoring public notebooks use a checkpoint trained partly on OAI (the public 0.949 model, released under Apache 2.0), and many teams submit those notebooks or build on them. A ruling on OAI therefore affects not only the teams that downloaded OAI themselves, but potentially a large share of the leaderboard.

Questions (a yes or no for each would be very helpful)

1. Is OAI allowed as external training data in this competition?

2. Are public checkpoints trained partly on OAI allowed in other teams' submissions, including teams that never accessed OAI themselves?

3. If OAI is not allowed: will every submission that used OAI or an OAI-trained checkpoint be removed from the leaderboard, including the many teams that submitted public notebooks built on it? Or is external-data compliance checked only for prize-eligible teams, during winner verification?

4. Can a team that finishes in the medal zone but outside the prize ranks be removed from the private leaderboard, or lose its medal, for having used OAI? Does the answer differ between gold, silver and bronze?

5. Would a ruling against OAI also cover derived artifacts: labels or pseudo-labels produced by OAI-trained models, and models fine-tuned from OAI-trained weights?

6. Could the answer be posted and pinned before the final submission deadline on October 22, so that every team selects its final submissions under the same rule?

Thank you for organizing this competition. A clear answer would help every team, whichever way it goes.

---

## Comments (4, threaded, oldest first)

### [1] madmax0404

**madmax0404** (@max6296, Expert) · 2026-10-14 03:22 UTC · ▲4 · LB rank 15

It's been discussed a lot for a while and they never gave us an answer. I don't think they will now.

#### ↳ [1.1] Satwik

**Satwik** (@p4rallax, Master) · 2026-10-14 03:25 UTC · ▲2 · LB rank 76

I agree. The hosts also made a new discussion post today regarding a survey and also replied in another discussion about the survey, while not replying to the dozens of open questions regarding external data use, so we probably are not going to get any answers till the end.

##### ↳ ↳ [1.1.1] Cody_Null

**Cody_Null** (@cody11null, Grandmaster) · 2026-10-14 03:57 UTC · ▲1 · LB rank 100

I haven't really used the OAI data myself, as it made very little difference in my pipeline, but I can imagine how confusing the situation is for the hosts. Lots of people are reporting issues, and from a fairness standpoint, everyone should have equal access to the data.

The problem, *in my opinion*, is that the website is a perfect example of government efficiency: somehow costing millions to build and maintain while still making a simple task impressively convoluted. Truly amazing how much taxpayer money it takes to make Walmart brand craigslist for healthcare data.

### [2] Wildata

**Wildata** (@wildata, Contributor) · 2026-10-14 07:36 UTC · ▲1 · LB rank 3724

This is game-breaking for many teams. I hope organizers address it cleanly asap.

The implicit assumption would be this is legal, until confirmed otherwise.
