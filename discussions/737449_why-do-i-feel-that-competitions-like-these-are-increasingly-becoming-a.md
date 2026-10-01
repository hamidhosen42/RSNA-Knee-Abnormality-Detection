# Why do I feel that competitions like these are increasingly becoming accessible only to people with premium GPUs or Colab Pro subscriptions?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737449
- **Author:** **Dnyanesh** (@dnyaneshbharambe, Expert) · _topic author_ · 2026-08-25 15:42 UTC · ▲1
- **Topic votes:** 1 · **Comments/replies:** 12 · **Pinned:** no
- **Last comment:** 2026-08-27 18:15 UTC
- **Cloned:** 2026-10-01 19:24 UTC (topic id 737449, raw JSON `_raw/737449.json`)

---

Free‑tier users often face long notebook queues, while competitions like these demand heavy compute resources, which makes participation feel skewed toward those with premium GPUs or Colab Pro accounts. What strategies or alternative resources can help free‑tier participants stay competitive despite these limitations?

---

## Comments (12, threaded, oldest first)

### [1] Komil Parmar

**Komil Parmar** (@komilparmar, Expert) · 2026-08-25 15:55 UTC · ▲3 · LB rank 59

The compute requirements of this competition is still very much affordable. More resources if anything would help do the experiments faster, but there is nothing in this competition like the more you train the more you score stuff. 

If you want to realize this, try ARC agi.

#### ↳ [1.1] Dnyanesh

**Dnyanesh** (@dnyaneshbharambe, Expert) · _topic author_ · 2026-08-25 16:47 UTC

ya sure , will give it a try

##### ↳ ↳ [1.1.1] Ravi Ramakrishnan

**Ravi Ramakrishnan** (@ravi20076, Grandmaster) · 2026-08-25 18:54 UTC · ▲1 · LB rank 292

@dnyaneshbharambe don't do that, ARC needs too much GPU

##### ↳ ↳ [1.1.2] Komil Parmar

**Komil Parmar** (@komilparmar, Expert) · 2026-08-25 19:32 UTC · LB rank 59

@dnyaneshbharambe I think you didn't understand it. I said if you want to realize how cheap are the compute requirements of this competition, try ARC AGI.

The last year winning team's solution model was trained on H100 for ~900 GPU hours, and the rate of H100 at the moment is around $4 per hour per gpu. Let alone be the cost of experimentations and ablations.

##### ↳ ↳ [1.1.3] (unknown)

**(unknown)** · 2026-08-27 18:15 UTC

_(no content — deleted or empty)_

### [2] starkhushi

**starkhushi** (@starkhushi, Expert) · 2026-08-25 16:29 UTC · ▲1 · LB rank 609

Feel this too — same queues, same 30h wall, same P100 crash (switch to T4). But my biggest gains cost zero GPU: rank-mean ensembling already-trained models (+0.015 LB) and picking a better label source (a 2-min CPU check). Also: cache decoded DICOMs in RAM (3x more experiments per quota), validate ideas locally before training them, run at off-peak hours, and there's 20 TPU-hrs/week nobody touches. Money buys the ceiling, but discipline covers most of the gap. Still agree free quotas should be higher for Featured comps.

#### ↳ [2.1] Dnyanesh

**Dnyanesh** (@dnyaneshbharambe, Expert) · _topic author_ · 2026-08-25 16:47 UTC · ▲1

ya discipline covers most of the gap i agree on this

### [3] John Taylor (AI)

**John Taylor (AI)** (@johntaylorai, Contributor) · 2026-08-25 17:27 UTC · ▲1 · LB rank 792

Most of these totally doable, I do on cpu first preprocessing, labelers, folds, and ablations. Once it's reduced you can use the quota where need. I personally will keep with cpu all the way through. In this one the biggest levers look to be cpu friendly.

### [4] DhawalModi

**DhawalModi** (@daveconnor, Contributor) · 2026-08-25 18:17 UTC · ▲2 · LB rank 24

Having good hardware does give you an edge, that is the truth. But by trying to reduce the "problem" into a reduce-able search space  of cheap experiments and being efficient with your data and training you can get ahead a lot. Here's some "free" GPU resources that you can use to train models:

https://modal.com/ (gives $30 of compute if you link a CC)}
https://www.beam.cloud/ (gives $30 of compute refreshed monthly if you link a CC)

(I am in no way affiliated to the above orgs, just something i use for my experiments)

### [5] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-08-25 19:21 UTC · LB rank 643

Love hearing this comment from someone who has been on the site for only a year :)
Been here 9 years - pretty sure a version of this post shows up every year multiple times.

I would guess its getting pretty hard for kaggle to find small data set problems.   When I look back over the competitions I spent lots of time working - pretty large percentage of them (in the first 5-7 years) can be solved with a handy open sourced model or one of the LLM's in an hour or two.  So almost a certain expectation that the compute will get more expensive each year, hosts are not going to go thru the effort and cost if the data set can be resolved in Claude by an intern.

I do lots of my work local and for the first 7 or so years I could do a machine upgrade / replacement and keep up with the compute.  My daughter just returned a machine I let her use for a few years that was my first kaggle build 9 years ago.  Its a wimp.  But the cost to build or upgrade has gone crazy - I am now happy if I can add a new graphic card once a year rather than the complete build of a new system.

Kaggle has done a decent job over the years of providing better compute power.  If memory serves 9 years ago it was 2 core CPU with 8GB of RAM?   We can hope that they can keep up in the future, as they do realize the need to balance the compute with the data to keep the site going strong.

My experience suggests you can be a lot less efficent and maybe even on the sloppy side if you have a lot of local compute.  While my team mate in a recent competition was a marvel at maximizing the use of kaggle resources.

If fact, I am finding it a bit like the use of Claude/ChatGPT etc for coding.  I am writing much higher level code than last year but I am a whole lot dumber doing it.  I am skipping most of the steps that use to inform my brain.    

My belief, your going to be better off long term operating within the kaggle limits than you would be if you had Elon's Nashville facility available for your use, assuming your object is to learn.

#### ↳ [5.1] (unknown)

**(unknown)** · 2026-08-26 12:11 UTC

_(no content — deleted or empty)_

### [6] Dana

**Dana** (@de3xie, Contributor) · 2026-08-25 19:31 UTC · LB rank 4144

kinda makes me wonder if having less compute actually makes you better at this lol. like you have to think more about what’s worth testing instead of just running everything
