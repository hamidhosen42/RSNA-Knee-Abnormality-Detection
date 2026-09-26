# what is u label auc on gold?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738172
- **Author:** **Myo Min Htet(wnp)🇲🇲** (@myominhtet, Expert) · _topic author_ · 2026-08-30 07:23 UTC · ▲2 · LB rank 1130
- **Topic votes:** 2 · **Comments/replies:** 1 · **Pinned:** no
- **Last comment:** 2026-08-30 07:34 UTC by Komil Parmar
- **Cloned:** 2026-09-26 16:24 UTC (topic id 738172, raw JSON `_raw/738172.json`)

---

my labels auc on gold is 0.9083 and lb 0.924.
I wonder is my fold auc is enough coz i am hitting the wall, differnet training appraoch cant push further. And can i get more like pushing LB with current label auc
.
what is ur label auc and lb result?

---

## Comments (1, threaded, oldest first)

### [1] Komil Parmar

**Komil Parmar** (@komilparmar, Expert) · 2026-08-30 07:34 UTC · ▲2 · LB rank 59

Actually, there are already some discussions where many participants have shared their local Gold and LB scores. 
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/736635
You should check them.

Personally, I would say that the calibration (tradeoff) you are seeing is spot on. Matches the pattern everyone is getting at the moment. And hitting the wall is fine. You will break it soon enough. Keep experimenting.
