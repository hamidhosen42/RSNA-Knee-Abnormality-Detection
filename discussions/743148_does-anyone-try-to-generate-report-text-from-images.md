# Does anyone try to generate report text from images?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743148
- **Author:** **Myo Min Htet(wnp)🇲🇲** (@myominhtet, Expert) · _topic author_ · 2026-09-25 02:51 UTC · ▲4 · LB rank 1128
- **Topic votes:** 4 · **Comments/replies:** 23 · **Pinned:** no
- **Last comment:** 2026-09-26 02:49 UTC by zhenglei_001
- **Cloned:** 2026-09-26 16:24 UTC (topic id 743148, raw JSON `_raw/743148.json`)

---

so the problem for me  is labelling part,  LLM extracted labels got 0.9083 label auc  on 58 gold and 0.935 lb on single fold, i could not reach further, so  the idea comes to my mind is to  generate report text label from image using qwen model. have someone tried that? does that works? coz i have limited GPU compute.

---

## Comments (23, threaded, oldest first)

### [1] Cody_Null

**Cody_Null** (@cody11null, Grandmaster) · 2026-09-25 05:15 UTC

If that extraction score comes from single shot and not a weighted blend then I think thats the highest match with the data that I have heard on gold. Labels shouldnt be your setback. There are a number of teams who have reported taking effnet b0 to 0.94+ on LB with 5 fold ensemble + TTA with 0.88 AUC on extraction labels. I obviously dont know anything about your pipeline but I would say your inference may give you more in TTA + 5 fold. And there is probably a bit of headroom in modeling or unique tricks from that point that I dont think have gone public in detail like pseudo labels? But it sounds like you are at the edge of top public knowledge. I can save you the GPU hours and say confidently that it shouldnt be your labels.

#### ↳ [1.1] Sanjib Biswas

**Sanjib Biswas** (@aidiagnosis, Contributor) · 2026-09-25 07:10 UTC · ▲-1 · LB rank 2223

That's a genuinely useful calibration. My single-shot extraction (LLM + regex, no image predictions blended in) reads 0.8843 on the 58, so I am right where you'd expect and I'll stop treating labels as the bottleneck.

On the effnet-b0 teams reaching 0.94+ - do you know what input geometry those runs use? Specifically slices cached per series, how many 2.5D windows per study, and whether they train per-study across series or per-series with the predictions pooled afterwards. I am at 5 plane/fat-sat slots × 12 windows at 448 px with attention pooling over both windows and series, and I'm starting to suspect that's more machinery than the problem wants.

##### ↳ ↳ [1.1.1] Raymond Yuen

**Raymond Yuen** (@laymond, Contributor) · 2026-09-25 07:25 UTC · LB rank 76

448px is a bit too expensive. I began with 384px and downgraded to 288px, and it barely had any effect on my results. I think many top LBs even use 224px

##### ↳ ↳ [1.1.2] Cà Rốt Production

**Cà Rốt Production** (@crtproduction, Contributor) · 2026-09-25 07:29 UTC · LB rank 641

lets be real here , 0,94+ effnet-b0 should have the great label , it will not come from TTA and folds, their labels should be really close to the test, LLM + regex 0.8843 on the 58 still be low, even the gold labels are the noise also, I would recommend 384 x 384 12 slices 140mm , 448 will be so slow but 16 windows randomly/60 will be standard. Even that, I still dont know where is the real bottleneck beside the labels, it could be better

##### ↳ ↳ [1.1.3] Cà Rốt Production

**Cà Rốt Production** (@crtproduction, Contributor) · 2026-09-25 07:36 UTC · LB rank 641

i saw u doing a lot, Raymond Yuen can you share just a bit about your process of the labelling process, I got stood at 0,916 label (evaluating on gold AUC) , my best single fold effV2 still be in 0,930 only, I tried a lot of things to try to improve the label, once I want to try to combine finetuning models on the picture to be the teacher model then combine and improve the label, will that be a good choice? calibrating and improving the insights into this datasets , labels still be too hard for me. Agreeing on a term that resolution wont affect, so I really cant see the real bottleneck is

##### ↳ ↳ [1.1.4] Raymond Yuen

**Raymond Yuen** (@laymond, Contributor) · 2026-09-25 07:42 UTC · ▲1 · LB rank 76

yes. using teacher models to generate pseudo labels was my main leverage.

##### ↳ ↳ [1.1.5] Myo Min Htet(wnp)🇲🇲

**Myo Min Htet(wnp)🇲🇲** (@myominhtet, Expert) · _topic author_ · 2026-09-25 08:06 UTC · LB rank 1128

did u try on image teacher labels, i have tried that but the training val and oof auc works out  and it did not transfer on LB.
i used the same resnet model on 5 fold and made oof prediction and combine with original labels . retrianing with teacher model labels +  oringal label looks good but did not push lb

##### ↳ ↳ [1.1.6] Raymond Yuen

**Raymond Yuen** (@laymond, Contributor) · 2026-09-25 08:09 UTC · LB rank 76

i didnt use image teachers to completely replace the labels. i instead blended them with 0.5+0.5 ratio. the eval set in this competition is very noisy, i had a 0.930 local model that scored worse than a 0.926 local.

##### ↳ ↳ [1.1.7] Myo Min Htet(wnp)🇲🇲

**Myo Min Htet(wnp)🇲🇲** (@myominhtet, Expert) · _topic author_ · 2026-09-25 08:15 UTC · LB rank 1128

Same with u 50/50 i dont replace labels but may be coz i use the same encoder for both teacher and student

##### ↳ ↳ [1.1.8] Myo Min Htet(wnp)🇲🇲

**Myo Min Htet(wnp)🇲🇲** (@myominhtet, Expert) · _topic author_ · 2026-09-25 08:21 UTC · LB rank 1128

did u use multiple teacher ensembling for oof pseudo labelling, or single model

##### ↳ ↳ [1.1.9] Raymond Yuen

**Raymond Yuen** (@laymond, Contributor) · 2026-09-25 08:25 UTC · LB rank 76

both works for me. depends on your label source and whether if you have disagreeing models

#### ↳ [1.2] Myo Min Htet(wnp)🇲🇲

**Myo Min Htet(wnp)🇲🇲** (@myominhtet, Expert) · _topic author_ · 2026-09-25 07:39 UTC · LB rank 1128

i have been working and focusing on my own work. i use simple resnet  model, feeding better quality data to the model with 24 sclices. there are some low qulaity and no info sclices i remove them, and  heavy augmentation and regularization helps the model a lot. 5 fold ensemble got 0.94, i dont know why i am not breaking the wall, one thing i suspect is i used patience=8 with 50 epoch and got best val at 28

##### ↳ ↳ [1.2.1] Myo Min Htet(wnp)🇲🇲

**Myo Min Htet(wnp)🇲🇲** (@myominhtet, Expert) · _topic author_ · 2026-09-25 07:41 UTC · LB rank 1128

224 resolution, i mean image size and different models doesnt matter . for me it is all about adding aug and regulariztion coz the model easy to get overfit

##### ↳ ↳ [1.2.2] Cà Rốt Production

**Cà Rốt Production** (@crtproduction, Contributor) · 2026-09-25 07:42 UTC · LB rank 641

24 slices are crazy , 50 epochs are hard work also , perhaps the model should have learnt a lot so best val at 28, should find the bottleneck to break the wall, right ?

##### ↳ ↳ [1.2.3] Myo Min Htet(wnp)🇲🇲

**Myo Min Htet(wnp)🇲🇲** (@myominhtet, Expert) · _topic author_ · 2026-09-25 08:01 UTC · LB rank 1128

yes, i am stuck

##### ↳ ↳ [1.2.4] zhenglei_001

**zhenglei_001** (@zhenglei001, Contributor) · 2026-09-26 02:49 UTC · LB rank 620

the same with you , I am stuck at 0.940 for single fold. enesemble only improve 0.02.

### [2] Sanjib Biswas

**Sanjib Biswas** (@aidiagnosis, Contributor) · 2026-09-25 07:01 UTC · ▲-1 · LB rank 2223

I went down a version of this road and it didn't pay, so hopefully this saves you the GPU time.

The short answer: the text round-trip can't add information, only lose it. At inference you have pixels and nothing else. Image -> generated report -> extracted labels is image -> labels with a lossy text bottleneck inserted in the middle. Whatever the VLM can see, a classifier trained directly on the same pixels can also see, without the generation and re-extraction steps each contributing their own errors.

The one version that isn't circular is using image -> text as pretraining - train the encoder to match report embeddings, then fine-tune a label head on top. I tested the premise of that and it failed: a linear probe on BGE-M3 embeddings of the reports (PCA to 128 dims) scored 0.8423 on 58 gold, against 0.8843 for LLM+regex extraction on the same studies. The embedding carried less retrievable label signal than twelve extracted numbers - much worse on MCL (0.78 vs 0.95) and lateral meniscus (0.78 vs 0.89), marginally better on contusion and fracture, all within the ±0.02 that 58 studies carry. I dropped the idea on that result.

Two things I measured that might be more useful to you:

Swapping the LLM is worth nothing. On all 58 gold: qwen2.5:14b rev4 0.8619, qwen3:14b 0.8572, qwen3:30b-a3b 0.8677. A 0.010 spread on 58 studies is noise. Watch out for reasoning models - qwen3's <think> block collides with JSON-constrained decoding and silently eats answers (12/58 and 52/58 unparsed until we turned thinking off).

Extraction is closer to its ceiling than it looks. I audited every gold positive, report score missed and asked whether the report contains any term for that finding, negation ignored. Extraction caught 207 of 240 positives, and 22 of the 33 misses were reports that never name the finding at all - synovitis, fracture and effusion miss only on silent reports. No reader recovers text that was never written, and neither will a VLM.

Given your labels already read 0.9083 on gold, I'd guess extraction isn't where your remaining headroom is either.

Reciprocating with a question, since your 0.935 on a single fold is well ahead of where I am: how many slices per series are you caching, and how many 2.5D windows do you feed per series? I measured mine corpus at median 30 slices per series (p90 39, p99 160), and going from 24 to 32 cached slices - 69% to 84% of all slices retained - was worth more to me than anything I did to the labels.

### [3] Navneet

**Navneet** (@navneetbende, Contributor) · 2026-09-25 08:42 UTC · ▲-1

Please infrom us if you got the report @myominhtet

#### ↳ [3.1] Myo Min Htet(wnp)🇲🇲

**Myo Min Htet(wnp)🇲🇲** (@myominhtet, Expert) · _topic author_ · 2026-09-25 11:19 UTC · LB rank 1128

it is negative and i have limited compute so not fully trained just fold0 and few epochs

### [4] Yann Majewski

**Yann Majewski** (@yannmajewski, Expert) · 2026-09-25 19:47 UTC · ▲1 · LB rank 70

I've tried multiple sets of labels scoring ranging from 0.86 to 0.895 on gold labels, and my models all gave very similar results on the LB score. Using only 58 labels to evaluate the quality of your dataset is not a reliable indicator of how well they will perform. I suggest you try using public datasets for these labels, as they tend to work pretty well!

#### ↳ [4.1] Myo Min Htet(wnp)🇲🇲

**Myo Min Htet(wnp)🇲🇲** (@myominhtet, Expert) · _topic author_ · 2026-09-26 00:23 UTC · LB rank 1128

nice point, i did use 58 gold to measure the label quality, but in  trianing i used silver labels only for both train and val set, 58 golds are excluded

### [5] Chris Deotte

**Chris Deotte** (@cdeotte, Grandmaster) · 2026-09-25 23:51 UTC · ▲1 · LB rank 92

Interesting suggestion. This is similar to using zero shot model to predict labels that we can then use to train supervised learning on other models. (In other words, have GPT-6-Astra or other strong multi-modality models like Qwen series look at images and give us a label zero shot)

#### ↳ [5.1] Myo Min Htet(wnp)🇲🇲

**Myo Min Htet(wnp)🇲🇲** (@myominhtet, Expert) · _topic author_ · 2026-09-26 00:19 UTC · ▲1 · LB rank 1128

yes like the radiologist look at the images  not report text, u already know some report text are against labels
