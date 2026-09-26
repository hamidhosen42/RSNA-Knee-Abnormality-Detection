# Your notebook hit an unhandled error while rerunning your code. 

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734062
- **Author:** **Saanvi Belede Naga** (@saanvibeledenaga, Contributor) · _topic author_ · 2026-08-09 17:30 UTC · ▲-2 · LB rank 3809
- **Topic votes:** -2 · **Comments/replies:** 3 · **Pinned:** no
- **Last comment:** 2026-08-10 14:40 UTC by PC Jimmmy
- **Cloned:** 2026-09-26 16:24 UTC (topic id 734062, raw JSON `_raw/734062.json`)

---

I encountered an error in one of the cells of my notebook and forgot to remove it before submitting. Though the notebook had an error one of my submissions got used up. I think the competition hosts should look into this and only consider submission counts for valid notebooks only. 
If I were to submit a csv file and it throws an error while scoring, it doesn't count as a valid submission and the daily submission count doesn't get affected(speaking from prev kaggle comp experiences).

---

## Comments (3, threaded, oldest first)

### [1] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-08-09 19:31 UTC · LB rank 642

This is a common discussion point over lots of years.   

Sadly folks can gain information probing the leader board and using an error to stop the submission attempt.  Without those submissions being counted you could run probes all day long.

Like most things in life - cheaters make it harder for the rest of us.

### [2] SUDHISH KUMAR

**SUDHISH KUMAR** (@sudhishkumar021, Contributor) · 2026-08-10 08:35 UTC · LB rank 3451

Each time I am submitting I am encountering same error you mentioned above although my log is not showing any error, could you please tell me what could be the reason?

#### ↳ [2.1] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-08-10 14:40 UTC · LB rank 642

What is the specific error message.

Most common errors often deal with memory or shape issues.  The notebook works OK with 3 fake test files but than fails with the full load.   Modify your prediction code so it predicts on a large number of the train files.  If you have timout errors than you would want to predict on 1300 train.  Doing that exposes the code to those common issues.

If you cannot figure it out - make the notebook and any datasets public and share a link here.  There will be some folks who will help - trouble shooting code without seeing it seldom works.
