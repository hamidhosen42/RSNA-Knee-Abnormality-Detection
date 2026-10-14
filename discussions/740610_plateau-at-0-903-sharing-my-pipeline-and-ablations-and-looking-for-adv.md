# Plateau at 0.903: sharing my pipeline and ablations, and looking for advice

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740610
- **Author:** **Will** (@pareru806, Contributor) · _topic author_ · 2026-09-10 12:48 UTC · ▲18 · LB rank 2621
- **Topic votes:** 18 · **Comments/replies:** 16 · **Pinned:** no
- **Last comment:** 2026-10-08 14:19 UTC by Arun
- **Cloned:** 2026-10-14 18:23 UTC (topic id 740610, raw JSON `_raw/740610.json`)

---

Hi Kagglers,

My current best public score is **0.903**. I have tried several backbones and approaches to report-derived supervision, but improvements of even 0.001 are becoming difficult. Reading the [discussion about compact models reaching 0.93–0.94+ at 224/288 resolution](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304) has made me reconsider where to spend my next experiments.

I would appreciate advice on what to audit or ablate next. I am sharing my setup and some unsuccessful experiments so others can see what I have tried.

## Current setup

- **Labels:** report-derived soft targets for the 4,407 training studies, with masks/weights for missing, uncertain or ambiguous supervision.
- **Images:** physical slice ordering using image position projected onto the slice normal; 130-mm crops cached at 336×336; six sequence slots; 12 selected slices per available slot, mainly within the central 20–80% of the stack. Selected slices are grouped into three-channel inputs.
- **Model:** mainly DINOv2-S/14, hierarchical aggregation within and across series, and twelve learned label queries.
- **Training:** 50-epoch schedules for the DINO members, using BCE-only or BCE plus a pairwise ranking loss.
- **Validation:** five-fold CV against fixed report-derived validation targets and masks. Promising pilots are extended to the remaining folds. The 58 expert-labeled studies have already informed development, so I treat them as a development panel rather than an independent test set.

The current composition has **four model recipes, each trained on five folds: 20 checkpoints total**. Fold predictions are rank-averaged within each recipe, and the resulting family outputs are rank-normalized for composition. Each label receives three equal-weight family contributions; I have not tuned those weights.

| Role | Backbone | Model input | Epochs | Loss | Training labels |
| --- | --- | ---: | ---: | --- | --- |
| Generalist | DINOv2-S/14 | 336 | 50 | BCE | All 12 |
| Specialist | DINOv2-S/14 | 336 | 50 | BCE | ACL, MCL, medial/lateral meniscus |
| Second DINO | DINOv2-S/14 | 336 | 50 | BCE + pairwise ranking | All 12 |
| CoAtNet | coatnet_rmlp_2_rw_384 | 384 | 25 | BCE | All 12 |

The CoAtNet input is resized from the same 336-pixel cache. The specialist replaces the generalist on its four columns. After rank normalization:

```text
Four specialist-owned columns: (specialist + CoAtNet + second DINO) / 3
Other eight columns:            (generalist + CoAtNet + second DINO) / 3
```

## Diagnostics on the expert-labeled development panel

These are OOF predictions on the 58 expert-labeled studies: each study uses its held-out checkpoint from each recipe before composition. They are not predictions from all five checkpoints on every gold study. There are only 9–35 positives per finding, so the intervals are wide.

“Report target AUC” means the report-derived target scored against the expert labels on this panel. The difference from image-model AUC is descriptive; I do not think its sign alone identifies whether supervision or representation is the bottleneck.

| Finding | Positives | Image-model AUC (95% CI) | Report target AUC |
| --- | ---: | --- | ---: |
| MCL | 9 | 0.980 [0.932, 1.000] | 0.921 |
| Baker's | 12 | 0.958 [0.892, 1.000] | 0.902 |
| ACL | 24 | 0.958 [0.893, 1.000] | 0.944 |
| Medial OA | 15 | 0.950 [0.889, 0.992] | 0.891 |
| Medial Meniscus | 26 | 0.947 [0.888, 0.990] | 0.927 |
| Contusion | 19 | 0.915 [0.826, 0.981] | 0.821 |
| Effusion | 35 | 0.901 [0.804, 0.971] | 0.696 |
| Fracture | 18 | 0.875 [0.756, 0.969] | 0.814 |
| Lateral Meniscus | 23 | 0.858 [0.746, 0.949] | 0.894 |
| PF OA | 21 | 0.824 [0.696, 0.924] | 0.819 |
| Lateral OA | 11 | 0.782 [0.575, 0.954] | 0.829 |
| Synovitis | 27 | 0.693 [0.544, 0.824] | 0.676 |

Two observations are motivating my next experiments:

- **Synovitis supervision is sparse.** My extractor classifies 3,610 of 4,407 reports (about 82%) as `not_mentioned`. Overall, 3,745 Synovitis targets (about 85%) are masked, including uncertain cases. Separately, my report audit found that 12 of 27 expert-positive studies had no explicit Synovitis statement. This limits what direct extraction can recover from those reports; it does not establish the image model's ceiling.
- **Lateral findings remain difficult.** The image-model AUC is below the report-target AUC for Lateral OA and Lateral Meniscus. For Lateral OA, 10 of 11 expert-positive studies mention the finding in their reports. This motivates looking at image construction and training, but does not rule out problems with target quality or disease definitions.

## Experiments that did not provide a useful further gain

Longer schedules and the specialist update have helped; by increasing the training epochs from 25 to 50, the AUC score raised from 0.899 to 0.903 on the public LB. The experiments below did not provide a convincing further gain in the recipes I tested.

- **Alternative public labels:** I tried [Steven's public LLM labels](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels). Better target AUC on gold58 did not translate into a convincing image-model improvement in my tested recipe. I do not regard target-quality differences as predicted model gains.
- **Explicit 2×2 spatial region tokens:** I appended pooled regional features to the label-query attention. At 50 epochs, the paired composition gain over two folds was **+0.000003**, with 95% CI **[−0.00086,+0.00084]**. This recipe did not help; I have not tested anatomical segmentation supervision.
- **Earlier DINOv3-B trials:** These were unsuccessful under an older pipeline. However, the configurations differed in preprocessing, supervision and training settings, so I do not regard them as a controlled conclusion about DINOv3 or about freezing versus fine-tuning.

## Questions I would appreciate help with

1. **Study construction:** For strong single-model results, what mattered most: series selection, physical crop, slice coverage or sampling? What would you check first in the images actually reaching the model? In 2.5D inputs, have you compared consecutive source slices with neighbors in a sparsely sampled stack?

2. **Weak supervision:** How do you handle report-silent and uncertain findings? Have masking, weighting or objective changes helped more than further extraction work?

3. **Compact CNNs:** For competitors doing well with ResNet, EfficientNet or ConvNeXt, which broad training choices mattered most? I am running a matched ConvNeXt-Tiny experiment before spending compute on larger inputs.

4. **Validation:** How do you decide whether a small gain against noisy report-derived labels is worth pursuing while limiting dependence on the public leaderboard?

Thanks to everyone sharing experiments. The recent threads on [fold count](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740375) and compact models prompted this post. Even high-level observations about what to audit or test next would be helpful.

If any of this is useful to you, I'd genuinely enjoy comparing notes — especially with anyone working on MRI preprocessing, weak supervision, or a modelling approach different from mine. 

**I'm also looking for teammates!** If anyone would like to join my team — or would have me join yours — please let me know. I'd bring this pipeline and the full ablation history behind it, including the experiments that didn't work. Everything here was trained on my own GPU.

---

## Comments (16, threaded, oldest first)

### [1] Tucker Arrants

**Tucker Arrants** (@tuckerarrants, Master) · 2026-09-10 17:19 UTC · ▲19 · LB rank 17

Some thoughts: take the best scoring single model public notebook (ignore the blends) and reproduce it locally. Use public LLM labels if you don't have your own, especially those that have documented extraction processes. Ideally you use your own so you know exactly where they succeed and where they fail. 

Start with 224px / 288px and a simple CNN encoder, like a ResNet34 or EffNetB0. No attention mechanisms, just use simple pooling operations. You can score in the 0.94+ range with this alone and it allows for efficient experiments at the lower resolutions.

Validate against the report extracted labels, not the provided 58 labels and not against the LB - neither are large enough sample sizes to resolve 0.002-0.003 differences. You want to stack these small improvements. Set a CV threshold for what is a genuine improvement and what is noise. Mine is around 0.003. You can get this tighter running multiple seeds if you need. When you get a real improvement, submit to LB to establish a CV - LB correlation.

This is your baseline. Then ablate through the standard image augmentation stacks. Many of these are generic and many are MRI specific, which you can get from previous competitions. Change one thing at a time so you can isolate what helped and what did not.

Then, you can try different backbones and resolutions, to see if the additional training time is really worth it.

After all of that, you can start trying more "experimental" mechanisms like attention and/or newer image encoders. I have yet to try any DINO inits and based on some preliminary experiments, I have a low prior for it matching standard ImageNet init backbones, but I could be wrong.

Kaggle public code section has been sullied with AI slop as people medal farm. Be cautious of this. There are only a few useful notebooks out there ([like this one](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model)). I generally assume a contrarian position about public notebooks - they are likely overly complex and not properly "built up" from simple baselines, using a clean CV protocol. Use them as their own baselines, and prune them to see what is actually useful and what is uneeded. "Prime" your own LLM saying something like "this code is probably over engineered, how can I simplify it?"

Last point - spend a lot of time just thinking about the problem. Have brainstorm sessions with your LLM of choice, without building anything. Stream of conciousness type chats. These "vibe chatting" sessions are where I get most of my ideas - LLMs may not be particularly creative in the context of Kaggle competitions, but they are great for organizing thoughts and finding ideas in that structure, that you might have otherwise not noticed yourself.

#### ↳ [1.1] Tom Aindow

**Tom Aindow** (@taindow, Master) · 2026-09-10 23:21 UTC · ▲5 · LB rank 130

Excellent advice thank you for taking the time to share. I can also second that a simple baseline will take you far if done carefully OP (also have 0.94+ at 224px with small simple model). Follow this post and you'll get there!

Just reflecting on your comments. It's a shame how polluted the kernel space has become on Kaggle. It used to be such a great resource to learn from one another. So many great and novel ideas. Now it's just overflowing with AI slop and needlessly complex brute forcing lb attempts. So much compute and so much noise. It's such a shame.

##### ↳ ↳ [1.1.1] Cody_Null

**Cody_Null** (@cody11null, Grandmaster) · 2026-09-10 23:43 UTC · ▲3 · LB rank 850

The super ensemble ultra pro max. 

The way the system is set up just incentivizes tons of copy paste leaderboard overfit ensembles. When in all actuality, unique training pipelines like that of the one by @dreaddevelopment shared what’s really the most valuable. It is very much the same with discussions, where there was a time where it was once extremely useful and people started taking advantage of the ranking and point system too much, devaluing the discussions. Glad to see in the last 2 competitions there are still people like @tuckerarrants who are willing to share. Hoping to return the favor once I get a little more time! Currently coding between diaper changes and midnight feedings hahaha

##### ↳ ↳ [1.1.2] Tucker Arrants

**Tucker Arrants** (@tuckerarrants, Master) · 2026-09-10 23:46 UTC · ▲6 · LB rank 17

Happy to help. 

And yes I agree. When I was last very active on Kaggle, around 2019 to 2020, the public code section was invaluable. GMs were sharing custom training loops, unique augmentation ideas, TTA, etc. You could learn so much just synthesizing data and the notebook medals were actually **earned**. Putting the ideas together with your own code was also a learning experience.

Now you can spawn up a notebook in a prompt and as long as it has a good public score, it gets forked and auto liked. I genuinely think the only solution is removing all non-competition tiers. But even that might not matter as people can always resort to bragging about their "notebooks forked" count instead of "gold medal notebooks". But alas, it is what it is. 

@dreaddevelopment's notebook was a breath of fresh air. Of course it was absorbed into the "super ensemble ultra pro max" notebooks but that was inevitable.

##### ↳ ↳ [1.1.3] Cody_Null

**Cody_Null** (@cody11null, Grandmaster) · 2026-09-10 23:50 UTC · ▲3 · LB rank 850

Yep totally with you, missing the days where you could sort by most votes and find actually useful code at every position

##### ↳ ↳ [1.1.4] Dread Development

**Dread Development** (@dreaddevelopment, Expert) · 2026-09-11 00:12 UTC · ▲4 · LB rank 802

So this is part of the reason I have not dropped my faster or newer weights yet, my team member sean & I were discussing useful code & datasets/other discussions & I couldn't at this point find any without our work wrapped into it.

##### ↳ ↳ [1.1.5] MengYe

**MengYe** (@konohayui, Expert) · 2026-09-11 20:38 UTC · ▲3 · LB rank 263

i miss the old days, i start kaggle back in high school, and i literally learned machine/deep learning from others notebooks.

##### ↳ ↳ [1.1.6] Dread Development

**Dread Development** (@dreaddevelopment, Expert) · 2026-09-11 22:20 UTC · ▲3 · LB rank 802

I have only recently begun kaggle comps (I had no idea this was a thing) & in the little over a month of competing, I have noticed a lot of people will publish code/notebooks just to snag a medal.. whats even worse is when you read through it & none of it or very little is theirs, makes it hard to tell whom you're actually competing with.
Also super new with medical models entirely so this is a learning experience for me. Currently juggling this competition, UMUD, Biohub, Solar filament, ARC 2 & 3 & kaggriculture.

#### ↳ [1.2] zhenglei_001

**zhenglei_001** (@zhenglei001, Contributor) · 2026-09-12 10:50 UTC · LB rank 453

My dino model, single fold can reache 0.932. Based on your opinions, I tried resnet50 with the same lables, preproccesed datas , only change the pooling way and head . the pooling is avg and the head is mlp.but the cv only can reache 0.86.I am really curious about the resnet whether  it is efficient like what you said. Maybe I still  don't find the reason why the resnet is worse than dino

##### ↳ ↳ [1.2.1] Tucker Arrants

**Tucker Arrants** (@tuckerarrants, Master) · 2026-09-12 13:44 UTC · ▲1 · LB rank 17

DINO is not a model, it's a way of pretraining that basically teaches a model to match crops of itself. It works on any architecture. DINO v1 even released a ResNet50. I am guessing you are using DINOv2 which is ViT-only. So comparing DINOv2 to a ResNet changes the backbone, the objective, and the pretraining data all at once.

Seems like you also compared a LB result with a CV result. 0.86 CV could score 0.95 on the public leaderboard - everyone has different labels so I can't tell you what a CV of 0.86 means. Use the same fold split and find the CV between your models to compare them.

"DINO crop matching pretraining objective creates a useful prior for MRI classication" is a very different claim from "ViTs are better at MRI classifciation than standard CNNs" and using DINOv2 mixes both at the same time.

##### ↳ ↳ [1.2.2] (unknown)

**(unknown)** · 2026-09-13 02:46 UTC · ▲1

_(no content — deleted or empty)_

##### ↳ ↳ [1.2.3] zhenglei_001

**zhenglei_001** (@zhenglei001, Contributor) · 2026-09-13 02:49 UTC · LB rank 453

Hi Tucker, thanks for the detailed reply — this is exactly the kind of clarification I needed.

    You're right on all three points:

        DINO ≠ DINOv2: I was sloppy. I meant DINOv2 (ViT-S/14) specifically, and I now see the comparison against ResNet changes three things at once: backbone (ViT vs CNN), pretraining objective (DINOv2 self-supervised vs ImageNet supervised), and pretraining data (LVD-142M vs ImageNet-1k). Any of those alone could explain the gap.

        LB vs CV: You're completely right, and this is the more embarrassing mistake. The 0.932 is my public LB score, while the 0.86 is a local CV on my own fold — I shouldn't have compared them at all. 0.86 CV could easily correspond to a very different LB number, exactly as you said. I'll only compare models on the same fold split from now on.

        "DINO prior for MRI" vs "ViT > CNN": Yeah, I was conflating them. Since I've only tried DINOv2-ViT against supervised-ResNet, I can't attribute the gap to either factor.

    My plan to disentangle this:

        Run ResNet50 with DINOv1 self-supervised pretraining (Meta released that checkpoint) on the same fold split, same preprocessing, same labels, evaluated purely on CV.

        Also run a supervised ViT (e.g. vit_small_patch16_224.augreg_in1k from timm) under the identical setup.

    Between those two I should be able to say something meaningful about whether the DINO objective or the ViT backbone is doing the work.

    Quick question if you don't mind: have you personally seen the DINOv1-ResNet50 checkpoint work well on medical imaging? I'm wondering whether the older DINOv1 objective is strong enough to matter here, or whether it's really the DINOv2 improvements (larger curated data, better augmentation recipe) that make the difference.

##### ↳ ↳ [1.2.4] Tucker Arrants

**Tucker Arrants** (@tuckerarrants, Master) · 2026-09-14 02:03 UTC · ▲3 · LB rank 17

The last medical image project I worked on was before DINO pretraining protocol existed, so I am not an expert. Don't take my word for it - try the experiment you just outlined and change nothing but the init, ImageNet versus DINO (v1, v2, or v3, or all the above if you really want to be thorough).

If you say "well maybe DINOv1 isn't good here but DINOv2 and DINOv3 are", which is a valid question as they are all different pretraining protocols, look for sets of models that have both ImageNet and the newer DINOs. I think ResNet only has DINOv1, but I believe ConvNext has a DINOv3 version as do other backbones. Ask your LLM to search the timm library or HuggingFace for those exact pairs where literally nothing changes but the init. 

That is about as clean and direct a comparison as you can get. Hope that helps.

#### ↳ [1.3] Arun

**Arun** (@oldwine357, Expert) · 2026-10-08 14:19 UTC · LB rank 532

Great insights, @tuckerarrants! I've trained EfficientNet models up to 0.91 and I'm curious which window size and batch size you used. In my runs, increasing the window from 6 to 12 improved the score, and 288px gave the best results. I'm currently training ResNet34, and it's also landing around 0.91.

### [2] Chris Deotte

**Chris Deotte** (@cdeotte, Grandmaster) · 2026-09-26 00:21 UTC · LB rank 134

Read public notebooks and discussions. There are many great ideas shared.

### [3] Will

**Will** (@pareru806, Contributor) · _topic author_ · 2026-10-02 05:25 UTC · ▲1 · LB rank 2621

Thanks to everyone’s advice, I’ve improved my public LB score to 0.935.
The suggestions helped me rethink my input pipeline, training setup, and ensembling strategy. Working through these experiments taught me a lot, and I really appreciate everyone who took the time to share their experience.
Thank you all for helping me move past that plateau!
