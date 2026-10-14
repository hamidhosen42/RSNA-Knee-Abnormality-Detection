# The core to reach top score without any ensemble

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747613
- **Author:** **Cà Rốt Production** (@crtproduction, Contributor) · _topic author_ · 2026-10-09 05:05 UTC · ▲-7 · LB rank 49
- **Topic votes:** -7 · **Comments/replies:** 4 · **Pinned:** no
- **Last comment:** 2026-10-12 04:24 UTC by Cà Rốt Production
- **Cloned:** 2026-10-14 18:23 UTC (topic id 747613, raw JSON `_raw/747613.json`)

---

Our team reached 0,95+ with single fold single model, I just want to discuss what is the limitation of the label, it can be improved over the period over time, but running and experiment may lead to the shortage of time when the competition is coming to the end. Hopefully someone with the top score can suggest me "The AUC peak when it come to the label only", and "the main model that u trained on" mine is RESNET, I would be really appreciate with even small suggestion. (just like the best single model like that) 
By the way, would really like to ask if any suggestion about the training process to utilize all the sources of the data to have the best score that it can reach (the config of some specific models that can utilize it peak learning rates (Resnet, DINO, in what size, what crop, the slices, the window , escpecially the cache ? ...). I saw 0,96+ is crazy, those seem to be having the techniques or the data that no one could have which is distinctive, however, I can see no reply about the OAI pretrained models rules, but I have never experienced that before even believing it will be helpful to improve the score. Up to date, the limitation is about the GPU limitations, the time, and the frustrating effort to try any available way to improve a model score sometimes it just about, improving a tiny adjustment can help use improve a tiny step. Wanting to hear all the great contestants opinions.

---

## Comments (4, threaded, oldest first)

### [1] hengck23

**hengck23** (@hengck23, Grandmaster) · 2026-10-09 10:38 UTC · ▲-1 · LB rank 455

there are some classes that has more error than others. and these classes have external data with eternal label

#### ↳ [1.1] Cà Rốt Production

**Cà Rốt Production** (@crtproduction, Contributor) · _topic author_ · 2026-10-11 15:08 UTC · ▲-1 · LB rank 49

nah dont even try external dataset

##### ↳ ↳ [1.1.1] SpeedSci

**SpeedSci** (@caoyupeng, Expert) · 2026-10-11 19:21 UTC · LB rank 1026

Can you tell me why?

##### ↳ ↳ [1.1.2] Cà Rốt Production

**Cà Rốt Production** (@crtproduction, Contributor) · _topic author_ · 2026-10-12 04:24 UTC · ▲1 · LB rank 49

it will be risky, many top - scorer wont try it, and the rules are not visible to see the "permission of the OAI or external dataset, so it is niche", and trying to finetuning MRI pretrained model will only benefit if u are trying to improve the label, and the blending model in my opinion, I wont even try it because the data blending leakage, the time to train the model, the time to evaluate the config , there are various disadvantages, besides, there are many other great ways to improve the score.
