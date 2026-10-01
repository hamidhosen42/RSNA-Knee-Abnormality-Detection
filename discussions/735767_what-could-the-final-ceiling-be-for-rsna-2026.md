# What Could the Final Ceiling Be for RSNA 2026?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735767
- **Author:** **Tony Li** (@tonylica, Master) · _topic author_ · 2026-08-18 00:05 UTC · ▲16 · LB rank 883
- **Topic votes:** 16 · **Comments/replies:** 12 · **Pinned:** no
- **Last comment:** 2026-09-30 10:24 UTC by Quan Vu
- **Cloned:** 2026-10-01 19:24 UTC (topic id 735767, raw JSON `_raw/735767.json`)

---

Really happy 😄 to see Kaggle competitions becoming more and more popular, with so many new teams joining.

For comparison:

* **RSNA 2025:** finished with **1,149 teams**
* **RSNA 2026:** only **12 days in**, and we already have **1,832 teams**

The leaderboard is also moving incredibly fast. The current **#1 public LB is already around 0.951**, while publicly shared solutions are already reaching around **0.92**.

So I’m curious: **where do you think the final ceiling will be?**

If BirdCLEF+ 2026 is any indication of how much solutions can improve during a competition, could we eventually see something like:

* **#1 final/Public LB:** ~**0.97**
* **Best publicly shared solution:** ~**0.95**

Of course, medical imaging competitions can behave very differently, and CV/generalization on the private set may change the picture a lot.

But with this many strong teams joining this early, one thing seems certain: **this is going to be a really tough competition. 😄**

Curious what others think the end-game scores might look like.

---

## Comments (12, threaded, oldest first)

### [1] Tim Krige

**Tim Krige** (@timkrige, Expert) · 2026-08-18 11:34 UTC · ▲2 · LB rank 1569

@tonylica, I take it these are then your predictions? ~0.97 public?

I am far more curious about private. With the current gold -> public leaderboard overfit chance, and the tendency to overfit public leaderboards that has become possible due to agents and autosubmissions poling the leaderboard so fast, I think the public leaderboard will lie.

My guess: private LB >0.94 would be gold. Winner 0.96.

### [2] Feroz Ahmed DS

**Feroz Ahmed DS** (@ferozahmedds, Contributor) · 2026-08-18 12:17 UTC

With so many strong teams joining this early, I think the final scores could improve significantly. A 0.97+ public LB seems possible, but the private LB and generalization will probably be the real challenge. Excited to see how far the leaderboard moves!

### [3] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-08-18 14:31 UTC · ▲6 · LB rank 1093

I would expect a huge shakeup and lots of over fit models.  Since we need to create the labels, trusting your cv as a golden rule not very helpful. 

It's pretty easy to spot what looks like conflicting results in the 58 gold labels we have received.  So we are trying to build models that predict the kaggle truth - often an elusive monster to grab hold of.

### [4] Tom

**Tom** (@tom99763, Master) · 2026-08-21 00:57 UTC · ▲1 · LB rank 1333

0.975 is the threshold of gold medal

#### ↳ [4.1] MengYe

**MengYe** (@konohayui, Expert) · 2026-08-21 02:13 UTC · ▲2 · LB rank 395

welcome back

##### ↳ ↳ [4.1.1] Tom

**Tom** (@tom99763, Master) · 2026-08-21 03:29 UTC · ▲1 · LB rank 1333

Thanks it's very tough to me

### [5] Chris Deotte

**Chris Deotte** (@cdeotte, Grandmaster) · 2026-09-26 00:27 UTC · ▲2 · LB rank 101

Hard to predict. Gold threshold is probably 0.96 and first place is probably 0.975.

#### ↳ [5.1] Lavin Wins

**Lavin Wins** (@lavinwins, Contributor) · 2026-09-26 00:51 UTC · ▲2 · LB rank 413

Yeah I think the same , cause there are alot of good teams participating just like you lol!

#### ↳ [5.2] Quan Vu

**Quan Vu** (@quan0095, Master) · 2026-09-27 23:44 UTC · ▲2 · LB rank 14

Both my agent and I have completely run out of ideas to test. Reaching 0.959 would already be a stretch, so hitting 0.975 is a massive surprise for me 😱

##### ↳ ↳ [5.2.1] Chris Deotte

**Chris Deotte** (@cdeotte, Grandmaster) · 2026-09-28 00:12 UTC · ▲1 · LB rank 101

I agree it sounds crazy, but Kagglers do amazing things in the last 2 weeks. They always do more than I expect, so I predict aggressively 😀

##### ↳ ↳ [5.2.2] Quan Vu

**Quan Vu** (@quan0095, Master) · 2026-09-28 00:20 UTC · ▲2 · LB rank 14

I hope that solution won't come from an autonomous research agent 🫠

##### ↳ ↳ [5.2.3] Quan Vu

**Quan Vu** (@quan0095, Master) · 2026-09-30 10:24 UTC · ▲1 · LB rank 14

At this stage, I am only seeing marginal gains—with score increases of just about 0.001 to 0.002, which likely fall within the margin of noise. All experiments aimed at securing a major improvement have failed. My last remaining hope lie in ensembling ^^
