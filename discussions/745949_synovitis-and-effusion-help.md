# Synovitis and Effusion Help

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745949
- **Author:** **OmerZalman** (@omerzalman, Contributor) · _topic author_ · 2026-10-05 12:49 UTC · ▲4 · LB rank 4704
- **Topic votes:** 4 · **Comments/replies:** 4 · **Pinned:** no
- **Last comment:** 2026-10-05 14:38 UTC by colum2131
- **Cloned:** 2026-10-14 18:23 UTC (topic id 745949, raw JSON `_raw/745949.json`)

---

My worst labelling issues have been with synovitis and effusion. I tested few pipelines against the expert labels and have been getting around 0.7 AUC/75% accuracy no matter how hard I try. Could anyone spare some advice on how to increase AUC and accuracy?

---

## Comments (4, threaded, oldest first)

### [1] colum2131

**colum2131** (@columbia2131, Master) · 2026-10-05 14:09 UTC · ▲2 · LB rank 63

I joined this competition about a week ago and I'm still struggling with the same two labels. Here are my results on the 58 expert-labelled studies (accuracy uses a 50% confidence threshold):

| | Effusion AUC / Acc | Synovitis AUC / Acc |
|---|---|---|
| LLM, reports only | 0.835 / 0.707 | 0.799 / 0.741 |
| LLM, reports + MRI | ~0.91-0.92 / 0.845 | ~0.84-0.86 / 0.69-0.72 |
| Image model trained on my pseudo-labels | 0.986 / 0.931 | 0.859 / 0.776 |

An AUC of around 0.7 with ~75% accuracy makes me wonder whether your pipeline outputs hard 0/1 labels. Since AUC measures ranking, asking for probabilities might help, e.g. ranking roughly "explicitly present" > "equivocal" > "not mentioned" > "explicitly absent". This could improve AUC without necessarily changing accuracy.

In my extraction, ~73% of the training reports say nothing about synovitis. My prompt treats silence as weak evidence of absence, while using effusion, trauma, degeneration, and age as context rather than simply defaulting to a negative label.

The report negatives also don't always agree with the expert labels. Among the 58 studies:

- Synovitis not mentioned: 11/36 (31%) were positive according to the experts.
- Effusion reported as absent/normal: 10/26 (38%) were positive.
- Effusion reported as equivocal/small: 14/18 (78%) were positive.

The expert panel calls fluid positive more often than the reports do. Adding images helped substantially with effusion: 0.835 -> ~0.92 for the LLM, and 0.986 for the image model trained on the pseudo-labels. Synovitis improved much less. My guess is that synovium and fluid are difficult to distinguish on non-contrast MRI, so the image input may have led the LLM to predict synovitis more often than the experts did.

#### ↳ [1.1] OmerZalman

**OmerZalman** (@omerzalman, Contributor) · _topic author_ · 2026-10-05 14:26 UTC · LB rank 4704

Thank you so much, also, im assuming that you are using pseudolabelling to train a better model. How would you trust the synovitis labels when most models usually get around 70-80% accuracy on the synovitis labels?

##### ↳ ↳ [1.1.1] colum2131

**colum2131** (@columbia2131, Master) · 2026-10-05 14:38 UTC · LB rank 63

Yes, it's pseudo-labelling. To be honest, I don't fully trust the synovitis labels either, and it's my weakest label as well.

I train on soft probabilities rather than 0/1, so uncertain cases stay around 0.3-0.6 instead of being forced to one side. Since the metric is AUC, ranking matters more than accuracy at 0.5, and labels with ~75% accuracy can still rank reasonably well.

For effusion, the noise largely averages out and the image model ends up better than its labels. Synovitis doesn't benefit as much, so I'm still looking for ways to reduce the noise in the synovitis labels and make them more accurate.

#### ↳ [1.2] OmerZalman

**OmerZalman** (@omerzalman, Contributor) · _topic author_ · 2026-10-05 14:37 UTC · LB rank 4704

Also, How did you get effusion accuracy so high?
