# Why ~200 public forks sit at exactly 0.941 

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742050
- **Author:** **starkhushi** (@starkhushi, Expert) · _topic author_ · 2026-09-19 07:49 UTC · ▲2 · LB rank 607
- **Topic votes:** 2 · **Comments/replies:** 6 · **Pinned:** no
- **Last comment:** 2026-09-20 14:31 UTC by starkhushi
- **Cloned:** 2026-09-26 16:24 UTC (topic id 742050, raw JSON `_raw/742050.json`)

---

Around 200 teams share a public score of exactly 0.941, and most run a fork of @mattiaangeli's *Bend the Knee to the Dinosaurs* with the 0.941 recipe from @maverickss26 / @renta0426. I diffed the forks against the base and found three things worth knowing if you're in that block.

**1. The base notebook documents its own experiments.** Its `.ipynb` metadata holds an `rsna_optimization` entry per version, naming the single predictive change each version made. Pull the notebook with `kaggle kernels pull mattiaangeli/bend-the-knee-to-the-dinosaurs -m` and read the metadata of the JSON file. It's the closest thing to an ablation log this competition has.

**2. The 0.941 recipe is three changes, not four.** Against v27 (0.940) it changes the per-label outer CoAt weights, re-ranks the inner CoAt/Raptor hybrid, sets the Raptor arms to 0.55/0.15, and re-ranks `_ke_cr`. That last one does nothing: `_raptor.csv` is already a percentile rank at that point.

**3. Mattia's own stack keeps the Raptor arms at 0.60/0.10.** His ledger shows the per-label map and the inner re-rank adopted, but not the 0.55/0.15 arms. Every 0.55/0.15 build I could find (maverickss26, renta0426, mine, a public copy of mine) sits at 0.941, and @jiweiliu's speed fork independently reverted the same two weights.

Every number here is a real submission:

| configuration | public |
|---|---|
| v27, `model_sources`/`kernel_sources` missing (all 20 DINOv2 members silently dropped) | 0.937 |
| v27, complete sources | 0.940 |
| + per-label map, inner re-rank, arms 0.55/0.15 | 0.941 |
| + arms back to 0.60/0.10 | [SCORE] |

A runnable version with the full source checklist is here: https://www.kaggle.com/code/starkhushi/rsna-knee-why-public-forks-stop-at-0-941

One caveat: the per-label map was tuned on the public split, and `Lateral Meniscus = 1.00` discards three of the four stages for that column. Keep a flat-0.60 version as your second final pick.

---

## Comments (6, threaded, oldest first)

### [1] Mattia Angeli

**Mattia Angeli** (@mattiaangeli, Master) · 2026-09-19 09:37 UTC · ▲4 · LB rank 101

The bigger point is that basically **no one besides [@DreadDevelopment](https://www.kaggle.com/dreaddevelopment) and me has released genuinely new models recently**.

Building a new model, prototyping it, training it and running proper ablations takes hours or days. Changing `0.55/0.15` to `0.60/0.10`, re-ranking, or tweaking per-label blend weights takes minutes.

Repeat those minute-scale changes across dozens or hundreds of submissions and you are simply overfitting the public LB. A `.940 → .941` move from another blend tweak is not strong evidence of a better model.

That is also why I would be very cautious about reading too much into whether 0.55/0.15 or 0.60/0.10 is “better.” Without new models or an independent validation signal, repeatedly searching those blend weights is basically another hyperparameter search on the public split.

Some of this is probably encouraged by the incentive to reach the top of the public leaderboard and make popular notebooks, but that is different from demonstrating better generalization on the hidden test.

#### ↳ [1.1] (unknown)

**(unknown)** · 2026-09-19 09:48 UTC · ▲-2

_(no content — deleted or empty)_

### [2] Dread Development

**Dread Development** (@dreaddevelopment, Expert) · 2026-09-19 13:41 UTC · ▲4 · LB rank 144

So I’m going to agree with @mattiaangeli on this, covered it very well.
Personally I have been working hard on developing my next iteration, which happens to be much slower that a few tweaks & a submit.

My current score & best time (0.944 & 13 minutes) has been a 2 week long training process. 
In the last two weeks I’ve trained 40 different models, attacking this from every possible angle with a goal of catching Scott.
The weights I had made public as well as the datasets, I get on average 50 notifications a day (0.924 Raptor)
So that bulk section of teams sitting at 0.941 are sitting there due to the mass forking and minor tweaking not actual tuning nor training nor anything that’s not already being done.

This makes this competition especially slow & difficult for people like me, I personally can only really use the discussions with others to build on new ideas due to a large percentage of public code/notebooks just being forks of my earlier work.

I did audit the competition a week or so ago & in total, 10,000 downloads over my datasets with roughly 3,000 forks. Just about everyone gives credit so I am appreciative of that! 
Just takes teams like my team longer to produce genuine quality work.
(I have been non stop working this for weeks & just in the past 2 or so days broke my own best of 0.941 & scoring time of 14 minutes)

#### ↳ [2.1] Scott Willis

**Scott Willis** (@scottgwillis, Contributor) · 2026-09-19 18:11 UTC · ▲1 · LB rank 3

Getting that down to 13 minutes is really impressive.  When I tried using CoAtNet (assuming that's what you're still using), my best submission time was around 25 minutes.

##### ↳ ↳ [2.1.1] Dread Development

**Dread Development** (@dreaddevelopment, Expert) · 2026-09-20 01:17 UTC · LB rank 144

Thank you sir! & Yes, still CoAtNet, paired with a ConvNeXt now.
The thing that really surprised me was how little of our runtime is actually the model, with as little as 2 minutes of diff between a single arm & a dual arm run.

#### ↳ [2.2] starkhushi

**starkhushi** (@starkhushi, Expert) · _topic author_ · 2026-09-20 14:31 UTC · ▲-2 · LB rank 607

Absolutely — that’s the distinction I was trying to highlight. The consistency of the 0.941 score shows how much of the public leaderboard comes from the same recipe and minor tweaks.

Respect for making your datasets and weights public despite the massive number of forks. And congrats on 0.944 in 13 minutes — that kind of improvement clearly comes from much deeper experimentation.
