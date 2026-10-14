# Are you actually looking at the data?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/736301
- **Author:** **Chikuwabu** (@prgckwb, Master) · _topic author_ · 2026-08-20 08:43 UTC · ▲17 · LB rank 221
- **Topic votes:** 17 · **Comments/replies:** 7 · **Pinned:** no
- **Last comment:** 2026-10-13 21:10 UTC by Po-Hao "Howard" Chen
- **Cloned:** 2026-10-14 18:23 UTC (topic id 736301, raw JSON `_raw/736301.json`)

---

![](_assets/f84e293beed11dbe.webp)

The image above is a CT scan. This image contains a gorilla. According to this article, even highly trained radiologists often miss the gorilla.

Before blindly training models with Coding Agent, it might be a good idea to check if there is a gorilla in the knee MRI.

https://www.npr.org/sections/health-shots/2013/02/11/171409656/why-even-radiologists-can-miss-a-gorilla-hiding-in-plain-sight

---

## Comments (7, threaded, oldest first)

### [1] Tom Aindow

**Tom Aindow** (@taindow, Master) · 2026-08-20 11:31 UTC · ▲5 · LB rank 164

I already insert random images of gorillas and other primates as part of my augmentation process. I thought this was common practice?

### [2] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-08-20 15:38 UTC · ▲2 · LB rank 2507

I have found one of the bad parts of using LLM's to help has been a huge decline in my looking at the images.

I use to spend most of the first couple of weeks of a competition doing mostly EDA's and looking at hundreds or even thousands of the images.  This is the 2nd competition would I seem to have completely skipped that vital step.

#### ↳ [2.1] Дворкин Евгений Владимирович

**Дворкин Евгений Владимирович** (@evgendvorkin, Master) · 2026-10-13 15:43 UTC · LB rank 674

Hello. Previously, I couldn't understand at all, I worked with public notebooks from DEEPSEEK. This is the first competition where I work in ChatGPT and I have a separate chat with EDA in my project and I'm just chatting to understand what's what, I started late, there are 9 days left!
Chat for research, chat for public work, a separate chat for all discussions, reading the discussions helped to get the second bronze)
I've never used coding agents, but reading the discussions, I realize that I might get lazy in the future))

### [3] Syed Asad Ali

**Syed Asad Ali** (@syed3000, Expert) · 2026-10-13 18:23 UTC · ▲1 · LB rank 1423

Can confirm. Adding a primate penalty to my loss function boosted local CV by +0.03. Huge shakeup on private LB incoming.

### [4] Phil Gear

**Phil Gear** (@philgear, Contributor) · 2026-10-13 19:15 UTC · LB rank 1040

i thought it was the mouth of a shark! An open pomegranate! Finally, I found the gorilla. 🤭

### [5] Syed Asad Ali

**Syed Asad Ali** (@syed3000, Expert) · 2026-10-13 19:23 UTC · LB rank 1423

funny thing here is that a lot of viewers can't still find the gorilla on top right in the image, they focus more on the center and think oh wow its a gorilla 😄

### [6] Po-Hao "Howard" Chen

**Po-Hao "Howard" Chen** (@javaduke95, Contributor) · _competition host_ · 2026-10-13 21:10 UTC

Oy - I too missed the gorilla in the chest CT until I read the text beneath the image.

I love this discussion thread.  We are actually [launching a survey about participant use of AI ](https://forms.gle/mRFnAkyXL1o9MuVY8) in this challenge.

TLDR is it'll take <10 mins to complete.  We would really appreciate a little of your time to share how you used AI in this challenge.  Do you use it to learn about MRIs, to code, to build a model, and so on.  It is collected anonymously.  

From the RSNA organizers' vantage point, we see you guys once a year, so the change from year to year has been quite interesting - so the survey is out of genuine curiosity.

Now for the typical challenge organizer response: I can neither confirm nor deny the presence of gorillas in the Knee Abnormality Challenge.
