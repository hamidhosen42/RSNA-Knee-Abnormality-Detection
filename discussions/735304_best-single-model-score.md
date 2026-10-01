# Best single-model score

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304
- **Author:** **roy214** (@phamgianguyen, Contributor) · _topic author_ · 2026-08-14 23:38 UTC · ▲66 · LB rank 2795
- **Topic votes:** 66 · **Comments/replies:** 112 · **Pinned:** no
- **Last comment:** 2026-09-30 19:18 UTC by Ace
- **Cloned:** 2026-10-01 19:24 UTC (topic id 735304, raw JSON `_raw/735304.json`)

---

What is the best score you’ve achieved with a single model?

I’m trying to train a strong single model first before moving on to ensembling, and I’m curious how far a single model can go.

I’ve noticed that most of the public solutions use ensembles of 20+ models and achieve around 0.91. For those scoring around 0.92–0.94 on the leaderboard, are you also using ensembles, or is it possible to reach that range with a single model?

For reference, my current best score is 0.887 with DINOv2.

---

## Comments (112, threaded, oldest first)

### [1] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-08-15 01:32 UTC · ▲9 · LB rank 1072

In my option - 9 days into a competition is a very silly time to be wasting efforts refining ensemble models.

### [2] Chikuwabu

**Chikuwabu** (@prgckwb, Master) · 2026-08-15 03:07 UTC · ▲6 · LB rank 26

A single model actually works much better than you'd expect.

#### ↳ [2.1] Devguru Tiwari

**Devguru Tiwari** (@dev123123456, Contributor) · 2026-09-06 08:11 UTC · LB rank 1825

Could you tell me the labels you are using. Since, i dont have any personal labels and using public available labels. Also, any hint on how to improve methodology will be helpful. Thank you in adv man.

### [3] Tom Aindow

**Tom Aindow** (@taindow, Master) · 2026-08-15 06:46 UTC · ▲5 · LB rank 98

Single model 0.915, DinoV2 based. 

Highly doubt the top results are ensembles, look at the [efficiency lb](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb), you can see many high scoring submissions also have decent run times, i.e. they're not running 20 different models.

Public notebooks just ensemble early on to try and get a high leaderboard score and exposure/kudos on their notebooks. Waste of time IMO.

#### ↳ [3.1] Berat Kirbiyik

**Berat Kirbiyik** (@beratkirbiyik, Contributor) · 2026-08-16 12:51 UTC · LB rank 1582

Single model 0.92 is impressive. Would you mind sharing the input geometry you landed on — resolution, physical crop in mm, and slices per series?

We're at 0.826 with 192px / 9 slices / 4 sequence slots and trying to work out whether our ceiling is the input pipeline or the model itself.

##### ↳ ↳ [3.1.1] Tim Krige

**Tim Krige** (@timkrige, Expert) · 2026-08-16 13:27 UTC · ▲2 · LB rank 1548

Sharing is quite revealing on my methods. What I can strongly suggest - look at the images your model sees. Does it look detailed enough? Do you think 192px is sufficient to see mm level details? 

I find playing with the pipeline is crucial.  If I can it see the details, how could a model?

Do research on the labels. Learn what the model needs to see. Llms are very wrong a lot of the time here without careful guidance.

##### ↳ ↳ [3.1.2] Tom Aindow

**Tom Aindow** (@taindow, Master) · 2026-08-17 20:55 UTC · ▲3 · LB rank 98

Sure, I'm using 392x with 150mm center crop (0.383 mm/px) and I randomly sample a bag of 32 slices per study across all studies during training. I'm still not quite sure what does and doesn't work here. I suspect that labels may actually be one of the most important parts.

### [4] Tim Krige

**Tim Krige** (@timkrige, Expert) · 2026-08-15 14:37 UTC · ▲3 · LB rank 1548

I am on a single model. OOF AUC = LB AUC within noise, 0.92.

### [5] Tucker Arrants

**Tucker Arrants** (@tuckerarrants, Master) · 2026-08-16 15:23 UTC · ▲6 · LB rank 34

Single model 0.934 @ 224px 
 
Update: 0.943 with some tweaks, still @ 224px

#### ↳ [5.1] Chikuwabu

**Chikuwabu** (@prgckwb, Master) · 2026-08-17 00:32 UTC · LB rank 26

Is this a single fold, or N-fold ensemble?

##### ↳ ↳ [5.1.1] Tucker Arrants

**Tucker Arrants** (@tuckerarrants, Master) · 2026-08-17 01:52 UTC · ▲3 · LB rank 34

5 fold ensemble

##### ↳ ↳ [5.1.2] (unknown)

**(unknown)** · 2026-08-17 21:00 UTC

_(no content — deleted or empty)_

##### ↳ ↳ [5.1.3] Tom Aindow

**Tom Aindow** (@taindow, Master) · 2026-08-17 21:00 UTC · ▲1 · LB rank 98

Really impressive result at 224x, are you willing to give any insight into your approach, either labeling or modelling? My current thinking is that labeling can be a significant source of error, since many labels in the gold set do not agree with the reports when the diagnosis rules are applied strictly (e.g. small effusions being labeled as 1).

##### ↳ ↳ [5.1.4] Tucker Arrants

**Tucker Arrants** (@tuckerarrants, Master) · 2026-08-17 21:33 UTC · LB rank 34

My label extraction is very crude - haven’t spent much time on it. But yes, I agree, I think it’s important here.

The model architecture is standard for these types of competitions and relatively small, currently taking around 25m for preprocessing and inference via submission scoring, including TTA.

##### ↳ ↳ [5.1.5] Cody_Null

**Cody_Null** (@cody11null, Grandmaster) · 2026-08-30 23:06 UTC · ▲1 · LB rank 2170

Are you guys only using the 58 to validate labels and then using the LB as your CV effectively? Or are you trusting the created labels to compute CV? Curious what some are getting for their CV and label scores compared to gold 58.

##### ↳ ↳ [5.1.6] Tucker Arrants

**Tucker Arrants** (@tuckerarrants, Master) · 2026-08-31 00:07 UTC · ▲3 · LB rank 34

I only validate against the report generated labels, never the provided labels. The sample size of the gold labels is just too small to resolve anything.

I've only used the LB to check correlation between it and my report generated labels CV, which is, so far, strongly correlated. If the correlation breaks, might just treat it like BirdCLEF and use LB for CV.

My CV is around 0.87 - 0.9 depending on the generated labels used, all leading to the same ~0.94 leaderboard score.

##### ↳ ↳ [5.1.7] Cody_Null

**Cody_Null** (@cody11null, Grandmaster) · 2026-08-31 01:02 UTC · ▲1 · LB rank 2170

Great! That’s about what my plan was too! Have you tried comparing the score of your generated labels to the 58 gold? Same issues exist with small sample size and the domain issue of their being natural disagreement between experts. But curious if you’ve managed to score much differently than I have. I’m at about .9 AUC. And another method at .92 that I worry is probably just a bit too good to be true.

##### ↳ ↳ [5.1.8] Tucker Arrants

**Tucker Arrants** (@tuckerarrants, Master) · 2026-08-31 01:10 UTC · ▲2 · LB rank 34

Yes I did a while back, the label extractor against gold is around yours, a bit lower I believe. Maybe 0.89. I've read the 58 reports of the gold labels, and my extraction is not missing anything - the reports genuinely do not contain the information required to generate the corresponding label, which is that domain issue you just mentioned. The radiologist writting the report either missed or simply did not care to report that particular label. 

Spent the last week or so getting "better" labels and none of it translated to the LB. Single prompt LLM extraction has not been beaten, outside of noise, by any more sophisticated extraction procedures that I have tried.

##### ↳ ↳ [5.1.9] Cody_Null

**Cody_Null** (@cody11null, Grandmaster) · 2026-08-31 01:15 UTC · ▲1 · LB rank 2170

I get the same result. Any time I try more than that my resulting curves definitely look like overfitting more than actual better labels. I assume the .89 you have also gets better results than the public .89? For me I did fair with public labels but using a different model got .9 and the cv is much much better.

##### ↳ ↳ [5.1.10] Tucker Arrants

**Tucker Arrants** (@tuckerarrants, Master) · 2026-08-31 01:25 UTC · ▲1 · LB rank 34

Exactly, I try not to look at the gold reports too much to avoid overfitting prompts to them. I actually haven't tried the public labels at all, as I am not totally sure how they were generated. My best current LB model is from my first pass at extraction, with thousands of verified extraction failures. I correct them, no change in LB. Image model "student" seems to consistently outperform the label extractor "teacher". Where it struggles are on the labels that are just poorly reported in general.

##### ↳ ↳ [5.1.11] Cody_Null

**Cody_Null** (@cody11null, Grandmaster) · 2026-08-31 01:28 UTC · ▲1 · LB rank 2170

Maybe I am not as far off as I thought then. Thanks so much for sharing, bringing back the kaggle spirit!

##### ↳ ↳ [5.1.12] Tucker Arrants

**Tucker Arrants** (@tuckerarrants, Master) · 2026-08-31 01:40 UTC · ▲1 · LB rank 34

Of course. I could be wrong, but I lean more "extracted labels are probably fine" than "extracted labels are the main lever to pull here". I'm not the best with this whole LLM thing, but based on all the work I've done with labels so far, it has not made a difference. The fundamental challenge is that report->label vs. image->label gap, which no amount of extraction tweaking can fix.

##### ↳ ↳ [5.1.13] Cody_Null

**Cody_Null** (@cody11null, Grandmaster) · 2026-08-31 01:58 UTC · ▲1 · LB rank 2170

I am personally much better with LLMs than image models and I think LLMs would be the main lever if there wasnt such a grey area there in the middle. The actual difference between a .92 and a .89 cv on those 58 gold samples is really explained by just small disagreement between only a few samples which would be natural in the field. So I think here it could be a difference but THE difference would probably be more about how you actually handle labels in modeling. At least thats my take on it, going to be fun to see how it all goes in the end!

##### ↳ ↳ [5.1.14] tennogh

**tennogh** (@tennogh, Master) · 2026-08-31 07:05 UTC · ▲1 · LB rank 5

It does seem like in many cases the reports just don't contain the information and clever "fixes" end up overfitting to the 58 set. Some of the disagreements are also interesting, could be different diagnosis policies or missing context (e.g. previous MRI/x-ray). Maybe I'll find a radiologist to look at some examples.

##### ↳ ↳ [5.1.15] Salem Ali

**Salem Ali** (@salemali7, Expert) · 2026-08-31 22:26 UTC · ▲1 · LB rank 268

I completely agree with your point. i came to the exact same conclusion: due to the inherent noise in the data and the fact that radiologists simply don't mention everything in their reports, the report->label mapping can never truly equal the image->label reality. Just like in your experience, i found that no matter how much I tried to refine and tweak the labels, it had no meaningful impact on the LB.
​Since the image model (Student) manages to bypass these textual gaps and detect the true visual features, have you tried using the model's high-confidence predictions to relabel the noisy data and train iteratively? or do you avoid this to prevent a confirmation bias

##### ↳ ↳ [5.1.16] Tucker Arrants

**Tucker Arrants** (@tuckerarrants, Master) · 2026-09-01 00:33 UTC · ▲1 · LB rank 34

Yes, its a good idea and worth trying. But you certainly can't validate against labels you repair or your CV will improve by definition. Would be like pseudo labeling then validating against your pseudo labels.

##### ↳ ↳ [5.1.17] (unknown)

**(unknown)** · 2026-09-05 10:56 UTC

_(no content — deleted or empty)_

##### ↳ ↳ [5.1.18] Tucker Arrants

**Tucker Arrants** (@tuckerarrants, Master) · 2026-09-05 18:39 UTC · ▲1 · LB rank 34

I use them in training, but not for validation. They make up around 1.3% of loss mass so not going to make much difference with or without, but they are not useful for CV so might as well train with them so they aren't completely wasted.

##### ↳ ↳ [5.1.19] (unknown)

**(unknown)** · 2026-09-06 06:54 UTC

_(no content — deleted or empty)_

#### ↳ [5.2] (unknown)

**(unknown)** · 2026-08-22 16:55 UTC

_(no content — deleted or empty)_

#### ↳ [5.3] Charles Savas

**Charles Savas** (@charlessavas, Contributor) · 2026-09-04 20:53 UTC · ▲-6 · LB rank 357

Hi Tucker — we're the team at 0.943 (charlessavas). Your posts on label handling match what we've measured: we probed public per-target AUC by column ablation, priced the report-to-image label ceiling per target, and built a one-hour harness that A/B tests any external model against our 8-signal ensemble.

Our lineup is DINOv2-MIL-heavy; yours sounds highly decorrelated from it. Our one external arm (correlation 0.83) was worth +0.004 public across four submissions.

Proposal: compare model and validation evidence first. If the families look independent, merge through Kaggle; then measure prediction correlation inside the team before changing either pipeline. A combination plausibly gives us a path through the current 0.949 top-10 cutoff. We'd bring the probe map, the external-integration harness, and a documented do-not-retry log of about 40 dead ends. Interested?

### [6] Aic

**Aic** (@gunianxue, Expert) · 2026-08-17 09:09 UTC · LB rank 2083

I’d like to ask how long it usually takes to train a model using DINOv2. Also, roughly how much computational resources are required for this competition?

#### ↳ [6.1] Tim Krige

**Tim Krige** (@timkrige, Expert) · 2026-08-17 09:44 UTC · ▲2 · LB rank 1548

This is a hard question to answer. Dino has a variety of sizes, small to huge. Within a size there are layers, with larger sized having more layers. In some competitions there have been amazing results from the pretrained dino weights and only finetuned heads. Others require unfreezing a few of the last layers, and still other competitions benefit from a full backbone unfreeze. The more of the backbone you unfreeze the more VRAM and GPU speed  you need. I am finding strong correlations between number of layers unfrozen and score, but I only have limited vram so I am typically training the smallest dino's.

I am on a 3070 and a 5070ti for most of my training. Models take a day or so to train 5 folds. 

A good technique on a tiny model is better than just throwing data at a large model and hoping compute can solve it, typically. You would also learn more like this.

I hope this helps.

### [7] k256.dev

**k256.dev** (@k256net, Master) · 2026-08-20 16:49 UTC · ▲4 · LB rank 2267

single(5-fold) model 0.926 @ 336px

LLM output labels only

#### ↳ [7.1] (unknown)

**(unknown)** · 2026-08-22 09:22 UTC

_(no content — deleted or empty)_

### [8] Scott Willis

**Scott Willis** (@scottgwillis, Contributor) · 2026-08-20 22:27 UTC · ▲6 · LB rank 3

I'm at 0.938 with a single model, single fold.

I'm also pretty new to all of this, so I'm going to assume that I'm missing a lot that I could be doing to help out my model.

#### ↳ [8.1] Tom Aindow

**Tom Aindow** (@taindow, Master) · 2026-08-22 07:24 UTC · LB rank 98

Really impressive result, congratulations! Would you be willing to share any details about how you are approaching the problem? It seems you are also top of the efficiency leaderboard, so perhaps you are doing something quite different to most !

##### ↳ ↳ [8.1.1] Scott Willis

**Scott Willis** (@scottgwillis, Contributor) · 2026-08-22 16:55 UTC · ▲1 · LB rank 3

I just tried to start with as small of a model as possible.  My whole goal was to start aiming for the efficiency LB and then go up from there if needed.  I also spent a lot of time trying to figure out how to work around the low quality labels I have.  I did break down and submit a 5 fold ensemble though, so my current 0.947 is no longer a single fold.

I'm not too confident my score is going to hold on the private LB, though.  Should be interesting to see.

##### ↳ ↳ [8.1.2] Berat Kirbiyik

**Berat Kirbiyik** (@beratkirbiyik, Contributor) · 2026-09-01 07:11 UTC · ▲1 · LB rank 1582

Congrats on holding #1 on the efficiency board — the "start as small as possible" framing is the opposite of what most of us drifted into, and clearly it worked.

The part I'd love to hear more about is the labels. You said you spent a lot of time working around the low-quality labels. Were you working from one of the public LLM-extracted sets, or did you build your own extraction? And when you say "working around" — is that a labelling-side fix (better extraction, soft targets, per-class confidence) or a training-side one (loss weighting, sample reweighting, masking uncertain cells)?

Asking because we're stuck at 0.924 and every architectural lever we've tried has come back inside noise: larger backbone, cross-slot attention, top-k pooling, EMA, mixup, longer schedules, slice-count sweeps. Seven ablations, all within 0.008 of each other. That pattern usually means the ceiling is upstream of the model, and your comment is the first thing I've read that points squarely at it.

Happy to share our timing profile in return — we got a CoAtNet-384 run down from 65 minutes to 17 by pipelining preprocessing against GPU, if that's useful to anyone.

##### ↳ ↳ [8.1.3] Devguru Tiwari

**Devguru Tiwari** (@dev123123456, Contributor) · 2026-09-05 17:11 UTC · LB rank 1825

please help me what to do to reduce time

##### ↳ ↳ [8.1.4] Scott Willis

**Scott Willis** (@scottgwillis, Contributor) · 2026-09-05 17:40 UTC · ▲8 · LB rank 3

I'm not using public labels.  I used Gemma 4 locally to extract labels from the reports.  I ran through it multiple times until it was "good enough" and they ended up being around .89.   I haven't tried public labels at all, like most people have said, I haven't really found a huge difference in the label quality right now.  My current LB best (0.949) is a single fold, single model and scores around .930 on gold.  I'm basically out of ideas right without sacrificing effeciency.

Scoring usually takes around 5 minutes, if that's helpful.

##### ↳ ↳ [8.1.5] Cody_Null

**Cody_Null** (@cody11null, Grandmaster) · 2026-09-05 18:11 UTC · ▲1 · LB rank 2170

Single fold, single model, 5 minute sub 0.949 is unreal haha I cant even process the images that fast right now

##### ↳ ↳ [8.1.6] (unknown)

**(unknown)** · 2026-09-09 13:28 UTC

_(no content — deleted or empty)_

##### ↳ ↳ [8.1.7] Scott Willis

**Scott Willis** (@scottgwillis, Contributor) · 2026-09-09 14:54 UTC · ▲5 · LB rank 3

I've really only ever spent time on efficentnet and resnet.  I spent a little time on others but at least this point I want to see how much I can get out of them.

#### ↳ [8.2] huytngo

**huytngo** (@huytngo, Contributor) · 2026-09-09 16:56 UTC · LB rank 46

> I'm at 0.938 with a single model, single fold.
> 
> I'm also pretty new to all of this, so I'm going to assume that I'm missing a lot that I could be doing to help out my model.

Bro you are a beast man, appreciate the advice on this thread.

##### ↳ ↳ [8.2.1] MJ121212

**MJ121212** (@mj121212, Contributor) · 2026-09-15 10:37 UTC · LB rank 1212

what base model did you use for this and where did you get/assemble your labels from ?

##### ↳ ↳ [8.2.2] Scott Willis

**Scott Willis** (@scottgwillis, Contributor) · 2026-09-15 16:53 UTC · LB rank 3

I'm using a smaller resnet.  Honestly at this point I'd be shocked if I'm not overfitting and my score holds up on the private LB.

##### ↳ ↳ [8.2.3] Optimo

**Optimo** (@optimo, Master) · 2026-09-16 08:41 UTC · ▲3 · LB rank 57

@scottgwillis What makes you worry about overfitting the public LB ? You do have a lot of subs, but a single fold single model running in 5 min and reaching such high scores seems like a solid solution instead of a worrying one. Are you relying a lot on LB to make changes to your approach ? Do you have a good CV/LB correlation ? Did you tweak any parameter on LB alone ?

#### ↳ [8.3] Lavin Wins

**Lavin Wins** (@lavinwins, Contributor) · 2026-09-16 15:17 UTC · ▲-9 · LB rank 410

@scottgwillis  , Bro you are at the top , can you please give me some advice related how to get started , like I used public baseline at recieved 0.94 so what next step should I take or what was your strategy?

### [9] diet1236364

**diet1236364** (@kosuke123, Contributor) · 2026-08-22 09:23 UTC · LB rank 52

single(5-fold) model 0.929 @ 224px

LLM output labels only

#### ↳ [9.1] Saicharan Ramineni

**Saicharan Ramineni** (@godlydonuts, Contributor) · 2026-09-02 23:36 UTC · LB rank 1132

What LLM did you use?

##### ↳ ↳ [9.1.1] diet1236364

**diet1236364** (@kosuke123, Contributor) · 2026-09-03 15:07 UTC · ▲-1 · LB rank 52

im using "astra" lol

### [10] Yann Majewski

**Yann Majewski** (@yannmajewski, Expert) · 2026-09-01 10:06 UTC · ▲6 · LB rank 83

0.936lb, single fold, small resnet model, 224x224

#### ↳ [10.1] Salem Ali

**Salem Ali** (@salemali7, Expert) · 2026-09-01 14:48 UTC · ▲1 · LB rank 268

Really impressive results with such a small resnet! I’m curious about your approach to handling label noise. Would you be willing to share some details?

##### ↳ ↳ [10.1.1] Yann Majewski

**Yann Majewski** (@yannmajewski, Expert) · 2026-09-02 17:04 UTC · ▲2 · LB rank 83

I did a little bit of label work, combined my labels with some public dataset and it increased lb score by 0.015

Then some pre training on the encoder and training longer with more regularization helped to reach 0.936!

Its a little hard to properly evaluate what really helped, when I made too many changes in between runs (e.g. changing the encoder architecture), I stopped seeing a correlation between validation and lb…

##### ↳ ↳ [10.1.2] Salem Ali

**Salem Ali** (@salemali7, Expert) · 2026-09-02 19:25 UTC · ▲1 · LB rank 268

I’ve found the same thing in my experiments data quality and training strategy can matter more than just making the model more complex. 0.936 with one fold ,a small ResNet is impressive,
congrats!

##### ↳ ↳ [10.1.3] (unknown)

**(unknown)** · 2026-09-26 13:31 UTC

_(no content — deleted or empty)_

#### ↳ [10.2] (unknown)

**(unknown)** · 2026-09-26 13:32 UTC

_(no content — deleted or empty)_

### [11] Jack

**Jack** (@jackvd, Master) · 2026-09-03 00:21 UTC · ▲3 · LB rank 1194

0.930 in 46 minutes, hoping a label improvement experiment pushes this further

### [12] tennogh

**tennogh** (@tennogh, Master) · 2026-09-03 15:13 UTC · ▲2 · LB rank 5

So far 0.942 raw, 0.943 with TTA at 288x288. OOF pseudo-labels have been pretty well correlated with LB.

#### ↳ [12.1] Komil Parmar

**Komil Parmar** (@komilparmar, Expert) · 2026-09-03 16:54 UTC · LB rank 50

That's really high! At just 288x288 resolution. If resolution was just 288x288 to get such high scores, I wonder what are the core drivers of such a high score then. If you would be willing to share any hints... Great job regardless!

##### ↳ ↳ [12.1.1] tennogh

**tennogh** (@tennogh, Master) · 2026-09-03 17:00 UTC · ▲2 · LB rank 5

Just a lot of incremental improvements. I have my own labels but they are not measurably better than the public ones. I'm not sure resolution is such a big driver.

##### ↳ ↳ [12.1.2] Tucker Arrants

**Tucker Arrants** (@tuckerarrants, Master) · 2026-09-03 17:09 UTC · ▲3 · LB rank 34

Agreed with resolution not being that important. I'm sure its backbone dependent and I've only tried ResNets and EffNets, but 224 / 288 is the sweet spot so far.

##### ↳ ↳ [12.1.3] Komil Parmar

**Komil Parmar** (@komilparmar, Expert) · 2026-09-03 17:16 UTC · ▲1 · LB rank 50

Wow, that's really interesting and worring at the same time. I gotta be finding the missing peice staying low-res. Thankyou @tennogh @tuckerarrants

##### ↳ ↳ [12.1.4] Shuolin Liu

**Shuolin Liu** (@lsl000ud, Master) · 2026-09-04 03:32 UTC

Thanks for sharing! I'm planning to start training with public labels. Do you think it's better to start with hard (0/1) or soft labels?

##### ↳ ↳ [12.1.5] Tucker Arrants

**Tucker Arrants** (@tuckerarrants, Master) · 2026-09-04 03:35 UTC · LB rank 34

Start with soft but worth trying both so you can measure the value of the soft labels. The delta between the two is itself insightful.

##### ↳ ↳ [12.1.6] Cody_Null

**Cody_Null** (@cody11null, Grandmaster) · 2026-09-10 17:54 UTC · LB rank 2170

Have you found any magic slices/windows for 224 res? I am trying it now with 0.446 mm/px on 64 slices 62 windows but my model scores struggle. Cant make out if my data is the problem or I have just busted something in my pipeline hahaha

##### ↳ ↳ [12.1.7] Parag

**Parag** (@paragkale, Master) · 2026-09-11 00:06 UTC · ▲1 · LB rank 6

At 100mm Field of View you may be losing some useful information. My sweet spot is 140mm to 160mm FOV. 

UPDATE: I was wrong about the 140-160 comment. Different models do well with different FOVs.

##### ↳ ↳ [12.1.8] Cody_Null

**Cody_Null** (@cody11null, Grandmaster) · 2026-09-11 03:03 UTC · LB rank 2170

Ill have a think about that one! Thanks!

##### ↳ ↳ [12.1.9] Cody_Null

**Cody_Null** (@cody11null, Grandmaster) · 2026-09-30 16:49 UTC · LB rank 2170

The switch to 140 helped a ton and I tried 160 and it was similar but not better. Would be curious what you found now? I was previously using 100 and it was obviously cutting things off. I also do tissue centering even tho it makes no difference on scores. I am just so confused about my scores as I am sitting at 0.9 val and 0.92-0.93 gold and best single model is still at 0.931. Many are reporting this massive 0.02 positive jump from cv to lb and it makes me wonder if I am missing something big

##### ↳ ↳ [12.1.10] Nicolai Karcher

**Nicolai Karcher** (@pejophylo, Contributor) · 2026-09-30 17:05 UTC · LB rank 1067

I validate OOF on extracted labels, and this is where I see the jump between local CV and LB (which makes sense, since the extracted labels are noisy, but the LB GT is not). Since you see 0.9 val (I assume on ex. labels) and LB of 0.931, don't you observe the same jump?

##### ↳ ↳ [12.1.11] Cody_Null

**Cody_Null** (@cody11null, Grandmaster) · 2026-09-30 17:31 UTC · ▲1 · LB rank 2170

Correct, I get a small jump between val. Gold has been weridly correlated and close to my LB despite only 58 sample (means nothing because of this but still odd). I think it may be because my extracted labels is like .91 as I found a few tricks to get better. As of now I think I have modeling work to do

##### ↳ ↳ [12.1.12] Nicolai Karcher

**Nicolai Karcher** (@pejophylo, Contributor) · 2026-09-30 19:06 UTC · LB rank 1067

0.91 is better than anything I ever achieved - did the label improvement translate to LB imrpvement? if so, care to give some hints on what you did to achieve these label improvements?

### [13] (unknown)

**(unknown)** · 2026-09-04 22:42 UTC · ▲-3

_(no content — deleted or empty)_

#### ↳ [13.1] SpeedSci

**SpeedSci** (@caoyupeng, Expert) · 2026-09-05 08:15 UTC · LB rank 129

I think different architectures can still complement each other, so I’m not fully convinced that ensembling is just averaging the same label noise. Right now, the most promising direction to me feels like: good labels + a strong single model + a few truly complementary models blended together.

##### ↳ ↳ [13.1.1] (unknown)

**(unknown)** · 2026-09-05 22:06 UTC · ▲-1

_(no content — deleted or empty)_

#### ↳ [13.2] SpeedSci

**SpeedSci** (@caoyupeng, Expert) · 2026-09-05 08:20 UTC · LB rank 129

I’m a bit confused about something and wanted to ask: in my experiments, v10 labels + DINOv2 outperformed v9 labels + DINOv2 — but when I switched to ConvNeXt, v9 labels + ConvNeXt actually did better than v10 labels + ConvNeXt.

It seems like different architectures have different "ability" to benefit from label improvements. Has anyone else run into something similar?

##### ↳ ↳ [13.2.1] (unknown)

**(unknown)** · 2026-09-05 21:55 UTC

_(no content — deleted or empty)_

### [14] Less

**Less** (@jiaoyouzhang, Master) · 2026-09-14 07:41 UTC · ▲4 · LB rank 119

Single fold, CoAtNet, 384*384, LB 0.937, but fusion is ineffective.

### [15] Raymond Yuen

**Raymond Yuen** (@laymond, Contributor) · 2026-09-15 16:41 UTC · ▲4 · LB rank 38

0.938 Lb. full train CoAtNet with 288*288. For now labels are still my greatest lever, especially pseudo labeling and blending different labels.

Update: Now at 0.94 lb. still label improvements

Update: Now 0.949. still label :D

#### ↳ [15.1] (unknown)

**(unknown)** · 2026-09-23 02:07 UTC

_(no content — deleted or empty)_

#### ↳ [15.2] Baquannn

**Baquannn** (@phngbqun, Contributor) · 2026-09-23 10:29 UTC · LB rank 1581

Really impressive improvement! Would you mind sharing a bit more about your label strategy? In particular, how are you generating the pseudo labels and how do you blend them with the original/other labels? Are the pseudo labels coming from OOF predictions, external models, or another source? Thanks for sharing!

##### ↳ ↳ [15.2.1] Raymond Yuen

**Raymond Yuen** (@laymond, Contributor) · 2026-09-23 10:38 UTC · ▲2 · LB rank 38

Thanks! I’m mainly using multiple weak-label sources (public, api, local) to train teacher models, then using the teacher models to generate pseudo-labels and eventually training a final model on them. This way the model gets more information from the images. Different teachers seem to help different targets.

##### ↳ ↳ [15.2.2] Raymond Yuen

**Raymond Yuen** (@laymond, Contributor) · 2026-09-23 10:45 UTC · ▲2 · LB rank 38

Btw from my experiments the difference between expensive and cheap models are minimal. The total API cost of the labels I'm currently using (including experimentation) is less than $5

### [16] kmn

**kmn** (@kmatsu01, Expert) · 2026-09-16 15:28 UTC · LB rank 2211

I'm at 0.930 with a single model (CoAtNet, 1h22min), and I haven't tried ensembles yet.

### [17] Archit Konde

**Archit Konde** (@architkonde, Contributor) · 2026-09-16 23:32 UTC · ▲7 · LB rank 43

Got a single fold CoAtNet at 224px to 0.950 right now.

#### ↳ [17.1] huytngo

**huytngo** (@huytngo, Contributor) · 2026-09-17 01:20 UTC · ▲1 · LB rank 46

Nice! do you mind sharing a bit of your thought process behind getting that single fold? anyways congrats!

##### ↳ ↳ [17.1.1] Archit Konde

**Archit Konde** (@architkonde, Contributor) · 2026-09-17 05:25 UTC · ▲4 · LB rank 43

Thank you! The main thing was one check. On the 58 provided labels, my OOF predictions were already closer to the radiologists than the extracted labels the model was trained on. Bigger models weren't helping either, so I figured the labels were the bottleneck and started working on that. The model is pretty standard: 2.5D CoAtNet at 224 (a few neighbouring slices at a time).

##### ↳ ↳ [17.1.2] Salem Ali

**Salem Ali** (@salemali7, Expert) · 2026-09-17 09:04 UTC · ▲1 · LB rank 268

That’s interesting! I got a mean AUC of 0.9208 on the 58 provided labels.Synovitis was my lowest-scoring class at 0.7969, followed by PF OA at 0.8533.
what did you get on the 58 labels? And what were your Synovitis and PF OA scores? Which classes were the main bottlenecks for you?

##### ↳ ↳ [17.1.3] Tom Aindow

**Tom Aindow** (@taindow, Master) · 2026-09-17 21:12 UTC · ▲2 · LB rank 98

Congratulations of the great model performance! 0.950 single fold is amazing.

Seems like the labels are really the key here, then. I've tried lots though and haven't had much success. I thought that perhaps using my models strongest predictions to highlight noisy labels might help, whether by masking them in training or by replacing them outright, but neither really have any noticeable effect. 

But then again, I suppose feeding a model back it's own predictions isn't exactly telling it anything new, it just makes it easier to reach the same point. 

I'm scratching my head trying to think about how to improve these labels further. Maybe multiple models with different strengths can teach each other when to listen to the teacher and when to ignore it? 

If you had any gentle hints to spare it would be much appreciated !

##### ↳ ↳ [17.1.4] Archit Konde

**Archit Konde** (@architkonde, Contributor) · 2026-09-18 17:25 UTC · ▲1 · LB rank 43

Single model (out of fold) is 0.930 on the 58 for me. Synovitis 0.797 and Lateral OA 0.816 are my weakest, PF OA 0.874. Our Synovitis numbers are nearly identical, which is interesting. I wouldn't read too much into the per-class values though, some classes only have 9-12 positives across the 58 studies.

##### ↳ ↳ [17.1.5] Archit Konde

**Archit Konde** (@architkonde, Contributor) · 2026-09-18 17:26 UTC · ▲2 · LB rank 43

Thank you! Of course, happy to help.

Masking the noisy labels did nothing for me either. Replacing them with the model's predictions came out worse than mixing the two, so I ended up keeping the extracted label and mixing the prediction into it rather than choosing between them. Why that helps, I'm honestly not sure. My guess is it's less about new information and more about how confident the label is, but that's a guess.

Rounding the labels to 0/1 also hurt; keeping them as fractions was better. And I found it useful to look at calibration on the 58 rather than only AUC. Mine happened to line up already (labels around 0.3 were positive roughly 30% of the time), which is what made me stop poking at the extraction.

Have you found anything that moves the diffuse classes? Synovitis and Lateral OA are where I'm stuck, and nothing I've tried on the model side seems to touch them.

##### ↳ ↳ [17.1.6] Tom Aindow

**Tom Aindow** (@taindow, Master) · 2026-09-19 16:29 UTC · ▲1 · LB rank 98

Interesting I've also tried blending prediction and labels together but didn't have much luck, something to come back to perhaps.

Synovitis I have tried to attack with little luck. It's clearly under reported I think, based on the golden data, so I suspect there are lots of false negatives in my labels. It's there, present, but they don't say anything about it. I tried to filter out these false positives by basically masking all negative synovitis labels except the ones where it was explicitly declared absent (not many), or was negative alongside lots of other negatives (I.e. the whole scan basically came back normal). This didn't do anything to my result though. I suspect masking false negatives isn't enough: the model needs more positives.

##### ↳ ↳ [17.1.7] Archit Konde

**Archit Konde** (@architkonde, Contributor) · 2026-09-21 03:00 UTC · ▲1 · LB rank 43

Yeah, I see the same thing on the gold set. Synovitis is the one the reports just don't mention, so most of the negatives are really silence. I tried something like your mask, dropping the low-confidence cells, and it did nothing for me either. What I did instead was look at where my image model was already beating the report labels, and it was mostly those silent cases, so I figured the model knew more than the labels did and used its predictions to fill them in instead of cutting anything out.

That one moved right away, but only because the predictions were properly out of fold. In fold ones just parrot the labels back. And I ended up pushing the blend a lot further toward the model than felt sane. 50/50 already jumped, going heavier on the model added a bit more.

Were yours out of fold? And how heavy did you go on the prediction side?

##### ↳ ↳ [17.1.8] Nicolai Karcher

**Nicolai Karcher** (@pejophylo, Contributor) · 2026-09-27 09:03 UTC · LB rank 1067

thanks both for this discussion, a lot of good information there.

@architkonde I tried the 'soft bootstrapping' between labels and model predictions (50/50) and retrained. I see a clear lift on both gold and on (original, non-bootstrapped) labels in local CV but for some reason doesn't translate to LB at all. Am really at a loss for why that might be.

To clarify: historically, without touching labels in this way, I saw an extremely strong correaltion between local AUC on extracted labels and LB score.

#### ↳ [17.2] (unknown)

**(unknown)** · 2026-09-17 07:30 UTC

_(no content — deleted or empty)_

### [18] Shiv Satyam

**Shiv Satyam** (@shivsatyam, Expert) · 2026-09-20 14:42 UTC · ▲1 · LB rank 2281

I got 91.0 using a Resnet34 backbone coupled with a mundane linear head.

### [19] NguyenThanhNhan

**NguyenThanhNhan** (@andy2709, Grandmaster) · 2026-09-21 09:01 UTC · ▲21 · LB rank 56

I got bored with the usual vision backbones from previous years' winning solutions and decided to finetune Qwen 3.5 with 384x384 inputs.

Single fold, 0.950, ~3.5 hours.

#### ↳ [19.1] Baquannn

**Baquannn** (@phngbqun, Contributor) · 2026-09-23 10:26 UTC · LB rank 1581

Nice result! Could you share which exact Qwen 3.5 variant you used (4B/9B/27B/etc.)? And did you finetune the full VLM or just the vision part?

##### ↳ ↳ [19.1.1] NguyenThanhNhan

**NguyenThanhNhan** (@andy2709, Grandmaster) · 2026-09-23 15:36 UTC · ▲6 · LB rank 56

I used qwen 3.5 2b. I attached lora modules to the language model and jointly fine-tuned them with the vision encoder.

##### ↳ ↳ [19.1.2] SpeedSci

**SpeedSci** (@caoyupeng, Expert) · 2026-09-23 17:50 UTC · LB rank 129

Thanks for sharing.I almost went with 9B, which would've required renting much more expensive machines.

##### ↳ ↳ [19.1.3] Lavin Wins

**Lavin Wins** (@lavinwins, Contributor) · 2026-09-29 15:45 UTC · LB rank 410

Great Work!!!

#### ↳ [19.2] Dhiyanesh Babu C V

**Dhiyanesh Babu C V** (@dhiyaneshbabu, Contributor) · 2026-09-26 13:37 UTC · LB rank 2559

You did on kaggle gpu ? did ypu use all the dcm files ?

##### ↳ ↳ [19.2.1] Ace

**Ace** (@shikhageekachad, Expert) · 2026-09-30 19:18 UTC · LB rank 661

kaggle Gpu is not sufficient, otherwise you have to do a lot checkpointing  and the kernel will eventually die. 
You cn use npz files for training

### [20] (unknown)

**(unknown)** · 2026-09-23 07:06 UTC

_(no content — deleted or empty)_

### [21] Chris Deotte

**Chris Deotte** (@cdeotte, Grandmaster) · 2026-09-25 23:55 UTC · ▲11 · LB rank 100

There are two types of single models. Those using original report extracted labels and those using pseudo labels. If we use pseudo labels, then every single model can match our best ensemble CV score and reporting that hides the fact that it isn't actually "single model".

#### ↳ [21.1] Quan Vu

**Quan Vu** (@quan0095, Master) · 2026-09-27 15:25 UTC · ▲7 · LB rank 14

Ever since I started participating in Kaggle competitions, I've wondered what the concept of a 'single model' actually means. To this day, I'm still not entirely clear on what qualifies as one. It feels like the more I learn, the less intelligent I feel—everything just gets more and more confusing 😂

### [22] OmerZalman

**OmerZalman** (@omerzalman, Contributor) · 2026-09-26 04:23 UTC

I was thinking a CNN would work best

### [23] Navneet

**Navneet** (@navneetbende, Contributor) · 2026-09-26 05:34 UTC · ▲-2

Thank you for the single-model score @phamgianguyen

### [24] CoolinLai

**CoolinLai** (@coolinlai, Expert) · 2026-09-28 07:03 UTC · ▲6 · LB rank 13

A five-fold ResNet with a resolution of 224px     . 0.954

#### ↳ [24.1] Musa Tur Farazi

**Musa Tur Farazi** (@musaturfarazi, Contributor) · 2026-09-28 09:12 UTC · LB rank 175

What was the score on gold ? Also did you focus on better labeling ?

#### ↳ [24.2] Cody_Null

**Cody_Null** (@cody11null, Grandmaster) · 2026-09-28 12:55 UTC · LB rank 2170

Are you using resnet 18?
