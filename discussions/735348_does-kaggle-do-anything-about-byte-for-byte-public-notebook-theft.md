# Does Kaggle Do Anything About Byte-for-Byte Public Notebook Theft?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735348
- **Author:** **Mattia Angeli** (@mattiaangeli, Master) · _topic author_ · 2026-08-15 08:01 UTC · ▲15 · LB rank 102
- **Topic votes:** 15 · **Comments/replies:** 14 · **Pinned:** no
- **Last comment:** 2026-09-23 10:05 UTC by Lavin Wins
- **Cloned:** 2026-09-26 16:24 UTC (topic id 735348, raw JSON `_raw/735348.json`)

---

This has happened twice now: I publish an update, go offline, and a few hours later some accounts like @nikitagajbhiye30 republishes it under another title with the credits removed and comments disabled.

Copied notebook:

https://www.kaggle.com/code/nikitagajbhiye30/rsna-knee-00

My original:

https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-dinov3-ensembled

"RSNA Knee 00" v5 is not merely similar to my notebook. All 25 non-empty executable cells are byte-for-byte identical. The only meaningful change was deleting my attribution and community-credit markdown. All three generated CSV files are byte-identical too.

This is not a new solution or an independent fork. It is my exact published notebook with the credits stripped.

I already reported both incidents.

---

## Comments (14, threaded, oldest first)

### [1] Mahog

**Mahog** (@mahoganybuttstrings, Expert) · 2026-08-15 09:04 UTC · ▲3

I have reported a lot of notebook duplicates/near-duplicates in the past and all of them got deleted afterwards. Maybe Kaggle will take action soon, idk :V

#### ↳ [1.1] Ravi Ramakrishnan

**Ravi Ramakrishnan** (@ravi20076, Grandmaster) · 2026-08-15 09:18 UTC · ▲4 · LB rank 292

@mahoganybuttstrings I think Kaggle's new fork = vote policy is easily hackable and such cases illustrate how people hack the voting rules for vested interests. I think the rules for non-competition tiers should be altered (and perhaps medals and tiers too) to prevent such duplications.

#### ↳ [1.2] Mattia Angeli

**Mattia Angeli** (@mattiaangeli, Master) · _topic author_ · 2026-08-15 10:08 UTC · ▲2 · LB rank 102

Yes, the notebooks usually do get removed eventually. The frustrating part is that the same users seem able to repeat the behavior over and over without any meaningful consequence.

### [2] Mattia Angeli

**Mattia Angeli** (@mattiaangeli, Master) · _topic author_ · 2026-08-15 10:19 UTC · ▲1 · LB rank 102

Byte-identical copies are the easiest cases to prove. The real elephant in the room is people changing one line or a single parameter—usually ensemble weights—and republishing the notebook as original work. Forking and
  improving someone’s work is fine when it is substantial and clearly credited; disguising a trivial edit as a new solution is the real problem.

### [3] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-08-15 13:16 UTC · ▲4 · LB rank 641

Been here lots of years - I would guess that almost every rule change or competition structure change has been aimed at Kaggle trying to stop the dog's rear end folks from cheating or hacking the system.

They have made a ton of changes over my 9 years of activity on the site.  Like most rules in life the changes almost always hurt the good users and only slow down the dogs.  But they do seem to keep working at it.

Keep reporting every single time.  Who knows, maybe they can use some AI to identify the repeat offenders.

#### ↳ [3.1] Mattia Angeli

**Mattia Angeli** (@mattiaangeli, Master) · _topic author_ · 2026-08-16 08:34 UTC · ▲1 · LB rank 102

While offline I asked Codex to scan the public notebooks with similar scores.

The result was an ordered description: notebooks that are byte-identical, notebooks that added moot cells to hide their theft (like the now silver-medal https://www.kaggle.com/code/salemali7/rsna-knee-90-reports-llm-30-epochs), and those that made substantial changes.

It took minutes. The AI might be already here.

##### ↳ ↳ [3.1.1] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-08-16 15:59 UTC · ▲3 · LB rank 641

For sure Kaggle has had that capability for a long time.

Some of the copy and paste notebooks have always seemed to just be noobs who don't understand what they are really doing.  Another part are the folks who know and abuse.  

For sure, if you don't want your work stolen - than don't share it.

IMO - the change to a fork is a vote was not a good idea - seems to me that made abuse easier by those who are attempting to get some glory with no work as part of a team effort.

### [4] Cody_Null

**Cody_Null** (@cody11null, Grandmaster) · 2026-08-16 20:24 UTC · ▲3

Yeah I have been commenting on this for some time, the problem has only gotten worse because now those who farmed discussions just moved to other tiers. https://www.kaggle.com/competitions/birdclef-2026/discussion/704413 https://www.kaggle.com/discussions/product-feedback/701499

#### ↳ [4.1] Mattia Angeli

**Mattia Angeli** (@mattiaangeli, Master) · _topic author_ · 2026-08-17 10:25 UTC · LB rank 102

I was in BirdCLEF 2026 too. The last few weeks there went completely crazy - a hundred times worse than what we are seeing here now.

##### ↳ ↳ [4.1.1] Cody_Null

**Cody_Null** (@cody11null, Grandmaster) · 2026-08-17 17:44 UTC · ▲3

Yeah the santa competition was the worst I have ever seen. People were updating public notebooks in the last few hours (because you cant post a new one) and kaggle did nothing about it from what I can tell. Some people farmed to notebooks M / GM notebooks teir with that competition alone 😂

##### ↳ ↳ [4.1.2] Ravi Ramakrishnan

**Ravi Ramakrishnan** (@ravi20076, Grandmaster) · 2026-08-17 18:07 UTC · ▲1 · LB rank 292

Kaggle was better managed in the years gone by. Things have changed for the worse since the past year, including competitions. 
Hope things improve from here - we can hope for better management and moderation from our end @cody11null

### [5] (unknown)

**(unknown)** · 2026-08-17 20:49 UTC

_(no content — deleted or empty)_

### [6] Danny Daniel

**Danny Daniel** (@danny514, Contributor) · 2026-09-23 00:41 UTC

Has anyone here dealt with a situation where another public Kaggle notebook was essentially an exact copy of their work, rather than just using a similar approach?

I’m curious about what Kaggle actually considers sufficient evidence when reporting something like this. For example, if the executable cells are identical, the outputs/files are identical, and the main difference is that the original author’s attribution and comments have been removed, is that something Kaggle can verify from the notebook history or metadata?

Also, what would be the best evidence to preserve before reporting? Would screenshots, the original notebook’s publication/update timestamps, downloaded notebook files, revision history, or hashes of the generated outputs be useful?

One thing I’m also wondering about is whether Kaggle has any mechanism for notifying the original author about the outcome of a report. If the copied notebook remains public while the report is being reviewed, is there anything else the original author should do besides submitting the report?

And for future notebooks, is there a recommended way to include attribution or licensing information so that it’s clear what can and cannot be reused?

### [7] Lavin Wins

**Lavin Wins** (@lavinwins, Contributor) · 2026-09-23 10:05 UTC · LB rank 279

Bro love your work.
Btw had some topic to discuss if you are intersted then please let me know .
