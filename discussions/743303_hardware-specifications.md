# Hardware Specifications

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743303
- **Author:** **Tanvir Anjum Apurbo** (@tanviranjumapurbo, Contributor) · _topic author_ · 2026-09-25 15:32 UTC
- **Topic votes:** 0 · **Comments/replies:** 10 · **Pinned:** no
- **Last comment:** 2026-09-25 23:45 UTC by Chris Deotte
- **Cloned:** 2026-10-14 18:23 UTC (topic id 743303, raw JSON `_raw/743303.json`)

---

May I know the hardware specifications of you guys are using? I am wondering I will be able to train this huge dataset or not.

---

## Comments (10, threaded, oldest first)

### [1] pompom555

**pompom555** (@pompom555, Expert) · 2026-09-25 15:39 UTC · LB rank 80

RTX 3090 works for me

#### ↳ [1.1] Tanvir Anjum Apurbo

**Tanvir Anjum Apurbo** (@tanviranjumapurbo, Contributor) · _topic author_ · 2026-09-25 19:39 UTC

24gb?

### [2] Cà Rốt Production

**Cà Rốt Production** (@crtproduction, Contributor) · 2026-09-25 15:58 UTC · LB rank 642

Colab A100 for me , my team has consumed more than 4000 Compute Units, wasted but have a deep understanding into the datasets , love the competition

#### ↳ [2.1] Tanvir Anjum Apurbo

**Tanvir Anjum Apurbo** (@tanviranjumapurbo, Contributor) · _topic author_ · 2026-09-25 19:39 UTC

how much it cost?

#### ↳ [2.2] Wisdom Aduah

**Wisdom Aduah** (@anaphase21, Contributor) · 2026-09-25 19:50 UTC · LB rank 2997

Could you please tell how you managed to get the dataset into colab? I have colab pro+ subscription, but for almost a month now, I have not been able to use it because I simply do not have enough disk space on colab. I got the data into my google drive but reading from drive during training is EXTREMELY SLOW.

##### ↳ ↳ [2.2.1] Tucker Arrants

**Tucker Arrants** (@tuckerarrants, Master) · 2026-09-25 20:31 UTC · ▲1 · LB rank 19

You preprocess the cache on Kaggle, which has the data mounted automatically, push it to a private dataset, and load it on Colab with the Kaggle API. My cache is 50GB at 384px and takes about 5m to load into Colab disk.

##### ↳ ↳ [2.2.2] Wisdom Aduah

**Wisdom Aduah** (@anaphase21, Contributor) · 2026-09-25 20:50 UTC · LB rank 2997

Just to be sure I understand you: what you did was to preprocess the dicom files on a kaggle notebook. But "Push to a private dataset" is where I am lost. Do you mean to commit the notebook and create a dataset from the outputs?

##### ↳ ↳ [2.2.3] Tucker Arrants

**Tucker Arrants** (@tuckerarrants, Master) · 2026-09-25 21:07 UTC · ▲1 · LB rank 19

You can do that if you want or you can create the dataset directly in the Kaggle notebook, again using the Kaggle APIs. Same outcome, first skips the commit + download step, it just uploads directly from the notebook when you run it. Ask a LLM, it'll know what to do but you certainly should not be reading any DICOMs from a Google Drive folder.

##### ↳ ↳ [2.2.4] Wisdom Aduah

**Wisdom Aduah** (@anaphase21, Contributor) · 2026-09-25 21:24 UTC · ▲1 · LB rank 2997

That was really helpful. Thanks.

### [3] Chris Deotte

**Chris Deotte** (@cdeotte, Grandmaster) · 2026-09-25 23:45 UTC · ▲1 · LB rank 92

There is still 1 month remaining, so that gives lots of time to succeed even if you only use 1 GPU. Currently I am participating with one L40S 48GB GPU. I queue a sequence of experiments on a single L40S 48GB GPU. Each experiment takes a few hours, so we get dozens of experiments each day.

In about a week, my experiments have found some strong single models and a LB 0.945 ensemble (without using public notebook models)! (Note I am not using public notebook models. If we use public notebook models, the search is quicker because we just search for something to improve public notebooks and don't start from scratch).
