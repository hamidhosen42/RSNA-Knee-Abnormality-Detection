# RSNA Raptor Weights

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737696
- **Author:** **Dread Development** (@dreaddevelopment, Expert) · _topic author_ · 2026-08-26 23:40 UTC · ▲19 · LB rank 144
- **Topic votes:** 19 · **Comments/replies:** 20 · **Pinned:** no
- **Last comment:** 2026-09-25 23:56 UTC by Chris Deotte
- **Cloned:** 2026-10-01 19:24 UTC (topic id 737696, raw JSON `_raw/737696.json`)

---

Just want to thank everyone who has upvoted the Raptor dataset & the other relevant notebooks! I didn't expect 1,900 downloads in just under a week!

I do plan on releasing more weights asap, I just have not had time to consolidate as of yet but do keep an eye out as the initial release was only my 0.924 set & I do have that same set trained & scoring 0.936-0.938 with one un-blended model.
Still trying to ensure I am as efficient as can possibly be!
Again, thank you all !!

---

## Comments (20, threaded, oldest first)

### [1] Cody_Null

**Cody_Null** (@cody11null, Grandmaster) · 2026-08-27 20:10 UTC · ▲3

Is this saying you have a 5 fold raptor sub with .936-.938 or is that a larger ensemble? Thanks for sharing in any case!

#### ↳ [1.1] Dread Development

**Dread Development** (@dreaddevelopment, Expert) · _topic author_ · 2026-08-27 22:21 UTC · ▲4 · LB rank 144

Great question Cody! So neither & that may not of been what you were expecting.
its a single model, not a 5 + - fold, one CoAtNet 384 run over the full labelled set with a small fully labelled holdout to pick the checkpoint.
A couple of the subs average the best three epochs of that same run, but there is no fold ensembeling in it.

I will release updated weights as soon as possible, I am just really crunched for time at the moment as I have another competition wrapping up (OpenAI Security) & I would rather not finish sub 20 on the public board so I have diverted all of my attention there until end of that comp.
I hope this answered your question!!!

##### ↳ ↳ [1.1.1] Cody_Null

**Cody_Null** (@cody11null, Grandmaster) · 2026-08-28 05:57 UTC · ▲1

Very cool! Thanks for sharing, interested in seeing your next update!

##### ↳ ↳ [1.1.2] nguyen214

**nguyen214** (@nguyen214, Contributor) · 2026-08-29 16:09 UTC · ▲1 · LB rank 2474

May I ask about the inference time of your pipeline? I think your leaderboard score is quite strong, but I didn’t see your name on the efficiency leaderboard, so I was curious about the runtime.

##### ↳ ↳ [1.1.3] Dread Development

**Dread Development** (@dreaddevelopment, Expert) · _topic author_ · 2026-08-29 18:18 UTC · LB rank 144

Sure, its around 9 hours for the full test set.
Almost all of that is the window count rather than models. We score 94 windows per study, so our CoAtNet ends up costing about as much as the entire 5 fold dino half we blend it with.
we did run a trimmed version a few days back, 24 windows on one T4 scored 0.02 and 42 windows got 0.930, so the cheap end is only about 0.01 behind.
Just havent selected one of those for the effiecienc side.

##### ↳ ↳ [1.1.4] Dread Development

**Dread Development** (@dreaddevelopment, Expert) · _topic author_ · 2026-09-06 19:05 UTC · ▲1 · LB rank 144

Wanted to update you on the timing - I have dropped scoring time to under 2 hours now & am continuing to work both the timing & accuracy in parallel.

Update 9/7/2026 at 11:51 EST - Time down from 2 hours to 29 minutes

### [2] Jack

**Jack** (@jackvd, Master) · 2026-09-04 14:51 UTC · ▲2 · LB rank 793

A couple questions..
1. Is your 0.936-0.938 model trained on the same labels as the 0.924?
2. Is your 0.936-0.938 model trained on the same datasets as the 0.924?
3. Is your 0.936-0.938 model using the same inference pipeline as the 0.924 or did you make improvements to it?
4. Does your SWA checkpoint from the 0.924 model set score better or worse than best epoch model? I'd answer this myself but I'm out of submissions from testing a bunch of other things lol

I've been testing a bunch of things with your previous SWA checkpoint and have it scoring well and quite efficiently. Your work has been insightful and a good starting point for this competition. Although I'd happily mess around with whatever new weights you release, don't feel pressured to release more weights, especially if they're your best or close to your best. This is a competition after all 😅

#### ↳ [2.1] Dread Development

**Dread Development** (@dreaddevelopment, Expert) · _topic author_ · 2026-09-04 23:23 UTC · ▲4 · LB rank 144

1. Labels - yes, every checkpoint published  is trained on the same label set with the only difference being the corpus geometry & window count.

2. Dataset - Same comp data, nothing external with the only change in preprocessing. The 0.924 set is 64 slices,140mm crop, span 0.06-0.94, 336px. Later ones widen the span, go to 80-96 slices or move to 384 px.

3. Pipeline - Same architecture and same windowing code

4. SWA - slightly worse. On our 58 gold labeled studies the final checkpoint reads 0.9167 and the SWA one 0.9150. Across six SWA/non-SWA pairs it averages out to nothing, and on the board an SWA arm scored 0.937 where its non-SWA twin scored 0.938.

I have not been able to make nearly as much progress in the last two weeks due to the OpenAI  security comp consuming all of my attention, however as that has concluded I am back to focusing on RSNA so I expect to gain headway this weekend.
I do ... want very much so to release more raptor weights & I do have quite a few to release even as of right now, I just dont want a repeat of my last drop.
I had made a few datasets public as well as the weights & I watched the entire board shift.. once I can asses where I stand I will release them, just being careful in doing so.

I am also trying to get back on track for Biohub, Solar filament, UMUD & another small lang model thats wrapping up so I am still diving time between several comps though these are no where near the headache the OpenAI  Security comp was so expect the release sooner (maybe sunday)😀

##### ↳ ↳ [2.1.1] Jack

**Jack** (@jackvd, Master) · 2026-09-05 00:00 UTC · ▲1 · LB rank 793

Thanks for the response. I've got the solo swa checkpoint from https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense scoring 0.930 - you're telling me this was worse than 0.924? Or am I confused about your datasets and notebooks?

##### ↳ ↳ [2.1.2] Dread Development

**Dread Development** (@dreaddevelopment, Expert) · _topic author_ · 2026-09-05 00:44 UTC · LB rank 144

You're not confused about the datasets, 0.924 isnt a property of the checkpoint but rather how many windows it was scored with. That number was the corpus at 42 windows per study.
The identical weights at 62 windows scored 0.927 on our board so window count alone was worth +0.003 & full coverage is higher still. A solo SWA checkpoint reading 0.930 is consistent with that - You're just scoring more of each study than my early run did.
I have another checkpoint public already that pushes those exact weights to 0.932 If im not mistaken.

I am scoring a few subs right now and expect them back soon, I've made some tweaks & am not entirely sure but very optimistic about them.

* I have 7 in flight scoring right now

##### ↳ ↳ [2.1.3] Dread Development

**Dread Development** (@dreaddevelopment, Expert) · _topic author_ · 2026-09-05 00:46 UTC · ▲1 · LB rank 144

This is the solo SWA that scored 0.932 - https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-finespacing

### [3] Ziad Ahmed

**Ziad Ahmed** (@ziadahmeddd, Contributor) · 2026-09-15 15:24 UTC · LB rank 1256

Thanks for publishing these, and for the geometry table on the dataset pages - it's what makes them usable without burning submissions to guess. We run maxspan-v5 and native384dense-v10 as a two-arm blend over a shared decode.

A question about the one page that omits the geometry. Nine of the ten Raptor datasets state slices / span / crop / stored px; raptor-knee-finespacing says only "the Fine Spacing preprocessing of the corpus". Given your note above that the later corpora "go to 80-96 slices", I'd rather not feed v9 a volume it never saw:

1. What slice count and span does the Fine Spacing corpus use?

2. Was the 0.932 scored at your 94 windows per study, or at a lower count?

The reason for (2) is your own fullspan result: widening the span at a fixed 44 slices cost 0.009, with the conclusion that widening only pays if slices are added to keep the sampling dense. If Fine Spacing is a 96-slice corpus, then scoring it on a 64-slot volume at 62 windows is that same density mismatch in the same direction - so the 0.932 wouldn't be available at 62 windows, and the checkpoint swap and the window count would be two different levers rather than one.

Also, if you happen to remember: "42 windows got 0.930" for the trimmed run - was that on the Fine Spacing corpus or the 64-slice one?

Happy to report back what we measure either way.

#### ↳ [3.1] Dread Development

**Dread Development** (@dreaddevelopment, Expert) · _topic author_ · 2026-09-15 16:05 UTC · LB rank 144

Good catch. That page went up without the geometry line and I've just fixed it, so it's on the dataset now along with a note about window counts.
Fine Spacing is 80 slices per study across 2 to 98 percent of each series, a 140 mm crop, stored at 336 pixels. The slots are 22 sagittal fluid, 18 sagittal, 15 coronal fluid, 10 coronal, 15 axial. Full window coverage is 78.
The 0.932 was scored at 78, so no, it isn't available at 62, and your reasoning about why is right. Same effect as the full span result: 44 slices over 2 to 98 percent scored 0.917 against 0.926 for 44 over 6 to 94, so spreading the same slices thinner cost 0.009. Putting 64 slices over the wide span brought it to 0.928, and 80 slices to 0.932. Density is the lever rather than span, and 1.20 percent of the stack per slice is the finest I've measured and also the best. Scoring an 80 slice volume at 62 windows leaves about a fifth of it unread, and at that point the checkpoint swap and the coverage change stop being separable, which is exactly what you said.
On the 42 window 0.930, I think two numbers from this thread have got stuck together. My 42 window results top out at 0.926. The 0.924 is the 64 slice corpus at 42 windows, and the identical weights read 0.927 at 62. The 0.930 is Jack's, further up this thread, from running the widedense SWA checkpoint with more coverage than my original run used.
Please do report back what you get, I'd be interested either way.

##### ↳ ↳ [3.1.1] Ziad Ahmed

**Ziad Ahmed** (@ziadahmeddd, Contributor) · 2026-09-19 08:11 UTC · ▲-1 · LB rank 1256

Reporting back as promised. Your density curve doesn't transfer to our pipeline, and I think the reason is instructive.
We tested it at fixed everything-else on one fold: 8 slices/slot → 10 slices/slot, which on our 0.12–0.88 band moves us from 1.58% to 1.27% of the stack per slice into the range you measured as best. It came back −0.0045 (plateau −0.0046) against a 0.003 noise threshold, so a real, small negative. We also tried tightening the crop 140 → 110 mm at the denser sampling: +0.0005, inside noise, unresolved.
The difference is probably structural rather than a contradiction. You sample 80 slices across whole series and score every window position, so extra slices buy genuinely new coverage. We sample 6 per-series slots into a masked attention head that pools 48 tokens, so extra slices mostly dilute the softmax rather than adding field of view the head already sees the whole span, just coarsely. Density is a coverage lever for you and a pooling-composition change for us.
Two other one-variable results from the same week, in case they're useful: swapping RadImageNet for ImageNet ResNet-50 at identical architecture and correct per-corpus normalisation cost us −0.063, and replacing our attention head with masked mean pooling cost −0.014. So for this pipeline the medical pretraining and the learned pooling are both doing real work.
Thanks again for publishing the geometry it saved us from spending a submission finding out that 0.932 wasn't available at 62 windows.

##### ↳ ↳ [3.1.2] Dread Development

**Dread Development** (@dreaddevelopment, Expert) · _topic author_ · 2026-09-23 01:20 UTC · LB rank 144

Thats a clean result & I think your read on it is right.

Worth knowing from this side, the lever has a ceiling for us too.
Going from 48-72 eval windows , a 50% increase moved the board 0.944 & cost 8 minutes of runtime. So it isnt that density pays for me & not for you, its that it stops paying & in your setup it stop paying earlier then goes slightly negative.

Coverage lever versus pooling compositon  change is exactly the distinction. If the head already sees the whole span, extra slices arent new information, they're more tokens competing in the same softmax & past some point thats a cost rather than a gain.

One that does go the other way for you is RadImageNet, that was a negative in my pipeline.

And your masked mean result matches something I saw from the other direction, changing the pooling at inference on weights that were trained with attention cost about 0.02. The heads weighting is not a formality.

thank you greatly for reporting back properly with the noise threshold attached. rare <3

##### ↳ ↳ [3.1.3] Ziad Ahmed

**Ziad Ahmed** (@ziadahmeddd, Contributor) · 2026-09-23 10:53 UTC · ▲-1 · LB rank 1256

That framing helps a lot "more tokens competing in the same softmax" predicts the shape we got, not just the sign. Our coverage curve over 4,407 studies, weak-label macro, shipped heads, nothing retrained:

k=16  0.8867     k=40  0.8934  (+0.0002 vs k=31)
k=24  0.8909     k=48  0.8939  (+0.0007)
k=31  0.8932     k=56  0.8940  (+0.0008)
k=62  0.8948  (+0.0016)

So it climbs to ~31 and then flattens for the next 30 windows. Noise floor on this ruler is ~0.002, so everything from 40 to 62 is one flat line with a slight upward drift at the end. That matches your 48→72 reading the density lever isn't differently-signed between us, it's just that we're further along the same saturating curve. Caveat on my side: those checkpoints were trained on these studies, so the absolute level is inflated; the shape is what I'd trust, not the number.

On RadImageNet going the other way I think I can offer a mechanism, because it was negative for us too until one change. Fold-0, out-of-fold, n=1102:

ImageNet ResNet-34,   3 adjacent slices as channels   0.8152
RadImageNet R50,      3 adjacent slices as channels   0.8339
RadImageNet R50,      centre slice repeated x3        0.8542

The trunk swap alone was worth +0.019. The channel contract was worth another +0.020 on its own. RadImageNet is pretrained on single-channel medical images replicated across RGB, so a 2.5D stack of adjacent slices is out-of-distribution for its first conv in a way ImageNet weights don't care about ImageNet was trained on real colour. If your pipeline fed it a slice triplet (which any Raptor-geometry setup naturally does), that's likely most of your negative. Costs one line to test.

Same family as your pooling point, now that I look at it: both are cases where the weights carry an input contract that isn't in the config, and violating it reads as "that component doesn't work here."

One disagreement worth flagging, since it's your shipped config: the reversed arm measures negative on our ruler. v5 forward 0.8943 vs reversed 0.8873, and adding both reversals takes the blend from 0.8948 to 0.8918. Your stack weights maxspan-v5-reverse at 0.10, so either our ruler disagrees with the board, or it earns its place through decorrelation inside a bigger ensemble than the two arms I'm testing it in. I'd believe the second just recording that I couldn't reproduce it as a standalone gain.
One more, unrelated but cheap: coatnet_rmlp_2 accepts img_size=256 with the 384 checkpoint and zero missing tensors, strict=True passes, 2.9× faster. It also loses 0.029 AUC. The rmlp bias generalises architecturally and not functionally  the tensors fit, the features don't survive.

### [4] Lavin Wins

**Lavin Wins** (@lavinwins, Contributor) · 2026-09-22 11:42 UTC · ▲1 · LB rank 278

Bro just want to say , the work you are doing is absolutely amazing. Keep doing it.
Btw just a question like are you just using the kaggle T4 only or you are also using some external GPUs ?

#### ↳ [4.1] Dread Development

**Dread Development** (@dreaddevelopment, Expert) · _topic author_ · 2026-09-23 01:12 UTC · LB rank 144

Thank you! I truly appreciate it! Ehhh so I learned real quick 45 hours (I pay for the sub) is not enough & when I bleed it, It essentially stops all my progress until reset as I am doing entirely to many competitions right now. I do use my local GPU as well as rented instances & a few VPS (so I dont wait 2hrs on data to transfer)

I was previously already training models so I had the general pipeline in place & have learned a lot since starting.

##### ↳ ↳ [4.1.1] Lavin Wins

**Lavin Wins** (@lavinwins, Contributor) · 2026-09-23 09:18 UTC · LB rank 278

Ohh, nice! Great job, anyways. 🙌

If you don’t mind, we can talk later about something else. I actually have an idea that I think we could work on together.

### [5] Chris Deotte

**Chris Deotte** (@cdeotte, Grandmaster) · 2026-09-25 23:56 UTC · LB rank 92

@dreaddevelopment Thanks for all your contributions. I am currently using your labels. And I have read your notebooks and discussions for ideas.
