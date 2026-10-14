# Title: Building an open analytics companion for this competition — what would you want in it?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742770
- **Author:** **RamónLópez** (@lopeznomar, Contributor) · _topic author_ · 2026-09-23 10:04 UTC · ▲-3
- **Topic votes:** -3 · **Comments/replies:** 1 · **Pinned:** no
- **Last comment:** 2026-09-25 23:57 UTC by Chris Deotte
- **Cloned:** 2026-10-14 18:23 UTC (topic id 742770, raw JSON `_raw/742770.json`)

---

Hi everyone,

Quick intro to what I'm working on, and a genuine ask for your input.

The problem this competition puts in front of us A si![]ngle knee MRI can carry a dozen different findings at once — ACL and MCL injuries, meniscus tears, osteoarthritis, effusion, synovitis, Baker's cyst, contusion, fracture. We're scoring on macro-averaged AUC across all of them, plus a separate efficiency track. That combination is unusual: a model can look strong on the aggregate number and still be quietly weak on two or three findings that happen to be rare. And the efficiency track means the fastest path to a better score isn't always a bigger model.

The leaderboard tells you where you are. It never tells you why.

What I'm building In the last Kaggle competition I entered, I built a free open-source analytics tool for replays. It ended up being used by people from 51 countries in the final weeks, and the biggest lesson was simple: competitors don't want another score, they want to understand their own model.

So I'm building the same idea for this one — a free, no-sign-up, open-source companion where you upload your predictions and immediately see:

Per-finding performance, so you know which of the 12 targets is dragging your macro AUC down
Your hardest cases: where the model is confidently wrong
Threshold and calibration behaviour per target
Efficiency trade-offs: accuracy gained versus runtime spent
Run-to-run comparison, so you can tell a real improvement from noise
Everything runs in your browser. Nothing gets uploaded or stored.

The ask It isn't finished, and that's on purpose. I'd rather build what you actually need than guess.

What would genuinely help you in the next few weeks? A view, a metric, a plot, a sanity check you keep doing by hand? Drop it in the comments and I'll work through the list publicly — I shipped one new feature per day in the last competition and plan to do the same here.

Where to reach me

Right here in the comments — I'll work through every suggestion publicly.
On X: @lopeznomar — I post daily build updates there.
By email: lopeznomar@gmail.com
The app itself is in the works — as soon as it's live, this thread gets the link, and the best ideas from this conversation become the first features.                                                                                                             
Good luck to everyone. 🦵

Independent open-source project. Not affiliated with, endorsed by, or sponsored by Kaggle or RSNA.

(https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F812296%2F1d1a6e3cd2b4892a9bb835c0ec7e326d%2Frsna-knee-ops-lab-intro-clean.png?generation=1790157292685080&alt=media)

---

## Comments (1, threaded, oldest first)

### [1] Chris Deotte

**Chris Deotte** (@cdeotte, Grandmaster) · 2026-09-25 23:57 UTC · LB rank 93

Great idea. In today's era of humans collaborating with AI, it is important to use visualization tools to understand what the AI are doing. Then we can review the visualization and make suggestions to the AI.
