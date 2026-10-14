# Using Dino3

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733313
- **Author:** **Ryuhki Kimura** (@kimura0415, Expert) · _topic author_ · 2026-08-06 12:50 UTC · ▲9 · LB rank 1017
- **Topic votes:** 9 · **Comments/replies:** 17 · **Pinned:** no
- **Last comment:** 2026-10-01 13:56 UTC by Rubens Sosa
- **Cloned:** 2026-10-14 18:23 UTC (topic id 733313, raw JSON `_raw/733313.json`)

---

From what I can tell, DINOv3 doesn’t seem to be registered. Am I correct in assuming that we can’t use it?

---

## Comments (17, threaded, oldest first)

### [1] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-08-06 15:15 UTC · ▲2 · LB rank 1080

Dino v3 has been used in other competitions this year - so I think its good !

#### ↳ [1.1] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-08-07 02:28 UTC · ▲1 · LB rank 1080

The [attached](https://www.kaggle.com/competitions/csiro-biomass/writeups/41st-place-solution-self-prompt-tuning-with-di) solution shows a write-up from a person who used Dino V3 - it was NOT one of the models in the models tab however.  You might find code in that competition on how folks handled getting v3.

I am using it now for this competition, but training local and including the files in my dataset so I have no need to get to the model on kaggle.

I mention the solution as he had nice things to say about me :)

##### ↳ ↳ [1.1.1] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-08-07 04:55 UTC · ▲1 · LB rank 1080

My V2 model scored 0.775 while pretty much the same V3 scored 0.763 - I was expecting some gain, for sure not a small loss.

##### ↳ ↳ [1.1.2] Ryuhki Kimura

**Ryuhki Kimura** (@kimura0415, Expert) · _topic author_ · 2026-08-07 09:49 UTC · ▲2 · LB rank 1017

Thank you for your valuable feedback. I’m also going to experiment with DINOv2 and DINOv3, and I’ll share the results once they’re available.

##### ↳ ↳ [1.1.3] Ryuhki Kimura

**Ryuhki Kimura** (@kimura0415, Expert) · _topic author_ · 2026-08-07 09:50 UTC · ▲1 · LB rank 1017

I didn’t expect the performance to drop with DINOv3. This is a very puzzling result.

##### ↳ ↳ [1.1.4] OverfitOracle

**OverfitOracle** (@alisalmanrana, Master) · 2026-08-07 21:54 UTC · LB rank 1017

quite surprised that the performance to drop with DINOv3.

##### ↳ ↳ [1.1.5] Dung Duong (Tony) Huynh

**Dung Duong (Tony) Huynh** (@dungduongtonyhuynh, Contributor) · 2026-08-09 15:50 UTC

Hi, did you guys train Dino on LLM-generated labels?

##### ↳ ↳ [1.1.6] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-08-10 14:56 UTC · LB rank 1080

My v2 vs v3 test model had hard coded labeler.  I also attempted to use V3 in the plant competition that I referenced in the attached solution.  V3 did not perform as well there, but the training on v2 did sound like it had a lot more plant based input so the result made sense.   Gemini Pro tells me that V3 had a lot more medical images than v2 and that v3 should be the best choice.

I do plan to repeat my evaluation when I get a model that is closer to 0.90.

##### ↳ ↳ [1.1.7] Tim Krige

**Tim Krige** (@timkrige, Expert) · 2026-08-12 10:54 UTC · ▲2 · LB rank 1558

I had similar experience with the CSIRO Biomass competition... My hypothesis is that the gram anchoring that makes DinoV3 so powerful is only helpful with massive datasets and long training, which we do not have. The pretrained weights are also better from v3, but if finetuning is needed, these are lost.

##### ↳ ↳ [1.1.8] Komil Parmar

**Komil Parmar** (@komilparmar, Expert) · 2026-08-13 14:05 UTC · ▲1 · LB rank 51

So good and nostalgic to see CSIRO people here 😄. @pcjimmmy @timkrige 
Yes, as mentioned above, Dinov3 isn't allowed on its own, but during CSIRO the competition hosts made it explicitly clear that we can use Dinov3. Unless we get such a confirmation again during this competition, I don't believe we should be using it.

Also, the pattern seen for Dinov3 vs Dinov2 was very interesting in CSIRO. It was complicated actually. It wasn't that Dinov3 is clearly performing worse than Dinov2, or vice versa but Dinov3 actually generalized much more than Dinov2 especially when it came to private LB. 

I won't mention all the details but I highly recommend checking the solution writeups of CSIRO, especially for better leveraging these Dino models.

##### ↳ ↳ [1.1.9] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-08-13 16:37 UTC · ▲2 · LB rank 1080

[Komil Parmar](https://www.kaggle.com/komilparmar)


Not sure I ever thanked you for your kind comments in your write-up in CSIRO.  Thanks!!!

I do see that someone got a v3 loaded into the Models - not sure that means any approval but I am still a long way from trying V3 again.  Maybe in another month I will have labels I like and a need to improve the model that trains on them.

### [2] Tim Krige

**Tim Krige** (@timkrige, Expert) · 2026-08-13 19:08 UTC · ▲3 · LB rank 1558

@javaduke95 I think a verdict on this would be great. It was offered for CSIRo Biomass (https://www.kaggle.com/competitions/csiro-biomass/discussion/617603#3377219) as the DinoV3 license is not a trivial yes or no here. 

The linked thread goes into the details, specifically here: https://www.kaggle.com/competitions/csiro-biomass/discussion/617603#3377219

As a disclaimer, I would like to use DinoV3.

#### ↳ [2.1] kumachl

**kumachl** (@kumachl, Contributor) · 2026-08-16 20:08 UTC · LB rank 3374

I got rejected while applying for dinov3 soooo

##### ↳ ↳ [2.1.1] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-08-16 20:50 UTC · LB rank 1080

So it might not be available to the world?   


Did you try using the one that is currently in the Models tab?

##### ↳ ↳ [2.1.2] kumachl

**kumachl** (@kumachl, Contributor) · 2026-08-16 22:33 UTC · LB rank 3374

ngl this works but its still slick bc license is form meta

### [3] phongtintruongg

**phongtintruongg** (@phongtintruongg, Contributor) · 2026-09-13 07:57 UTC · LB rank 1521

@javaduke95 Following up since this thread never got an official ruling:
several public notebooks now scoring 0.941-0.942 (e.g. "Bend the Knee to the Dinosaurs", "rsna-knee-0941-restructured") depend on a community-uploaded Kaggle dataset (mattiaangeli/knee-mri-fold-weights) containing 5 fold-checkpoints derived from the official Meta DINOv3 ViT-S/16 (LVD-1689M) backbone, relicensed CC0 by the uploader.

### [4] Rubens Sosa

**Rubens Sosa** (@rubenssosa, Contributor) · 2026-10-01 13:56 UTC · LB rank 4719

@javaduke95 Are the official Meta DINOv3 weights (which are closed, with access requiring approval that some participants report having been denied) permitted under the accessibility criteria you described in 733965? Also, are third-party re-uploads of checkpoints derived from DINOv3 acceptable?
