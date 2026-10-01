# A lot of submissions are just forks of someone's solution

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737635
- **Author:** **LavanBth99** (@lavanbth99, Contributor) · _topic author_ · 2026-08-26 14:15 UTC · ▲4 · LB rank 2487
- **Topic votes:** 4 · **Comments/replies:** 7 · **Pinned:** no
- **Last comment:** 2026-09-23 01:03 UTC by Dread Development
- **Cloned:** 2026-10-01 19:24 UTC (topic id 737635, raw JSON `_raw/737635.json`)

---

As someone who is looking to make progress with each submission attempt, I see that a lot of participants have just 'copied'/forked someone else's solution. As someone who is approaching this competition as a puzzle where each submission reveals something new or a new piece to the board, it doesn't feel correct by any means to 'lift' someones original work and label it as your own during submission. 

I am curious as to how the host would tackle this specifically? Like I can right now just go and ensemble models with slight tweaks and make a submission attempt and probably score a higher leaderboard (overfit or not) but that doesnt signify the originality of the work.

---

## Comments (7, threaded, oldest first)

### [1] Dread Development

**Dread Development** (@dreaddevelopment, Expert) · 2026-08-26 23:47 UTC · LB rank 145

Hello! So, I initially released one dataset about a week ago that I myself trained scoring a 0.924 with that one model.
It did get a lot of traction and people asked if I planned on releasing the weights, so I did & I do plan on updating that notebook sometime this week with a stronger base.

My goal was 1 model not 19 packed together for the sole purpose of scoring high on that base while maintaining efficiency, needless to say a week later 150 forks & 1,900 downloads of those weights & I have watched the board sky rocket.
To be as transparent as one could be, I will update that notebook & I do see the same thing happening again, however anyone straight copying or mixing in 7 other models will only climb so high before hitting a wall.

You as well as everyone else, are more than welcome to fork or download my open weights & build on it which is what I recommend vs a straight copy for the score.
I have tried my best to make my data as easy as possible to follow, however if you do run into any issues please feel free to reach out !

#### ↳ [1.1] FGPC

**FGPC** (@projdev, Expert) · 2026-08-27 02:44 UTC · ▲1

hello, thanks for your notebook. Is it possible for you to  publish also the source code on how you generate those datasets "Knee MRI preprocessed slice stacks"?

##### ↳ ↳ [1.1.1] Dread Development

**Dread Development** (@dreaddevelopment, Expert) · 2026-08-27 03:49 UTC · LB rank 145

The preprocessing code isnt't published right now, but the dataset descriptions cover the substance: one series per plane & fluid sensitivity slot, slices sampled evenly across most of each series span, center cropped to a fixed physical size in mm using DICOM pixel spacing, & stacked in a fixed slot order per study.
I am planning on releasing the full pipeline, preprocessing included.
I will update the notebooks soon!

##### ↳ ↳ [1.1.2] FGPC

**FGPC** (@projdev, Expert) · 2026-09-22 02:04 UTC · ▲1

Hi, do you still have a plan to publish the source code of your MRI preprocessed slice stacks?

##### ↳ ↳ [1.1.3] Dread Development

**Dread Development** (@dreaddevelopment, Expert) · 2026-09-23 01:03 UTC · LB rank 145

Hey! So not right now however most of it is already on the datasets page!

### [2] Komil Parmar

**Komil Parmar** (@komilparmar, Expert) · 2026-08-27 07:12 UTC · ▲1 · LB rank 59

I am unable to understand what concerns you here. 

You should rather be focused on learning (just like you mentioned you are doing), rather than keeping a track of ranks and scores like this.

Even if you don't work at all, at the end of the competition on the very final day if you simply copy the then best scoring public notebook, you should still be able to get into the top 500 or more. 

The goal is to surpass the best available public notebook, showcase your own work. And that's where the real LB starts. People who understands Kaggle well, will credit you correctly.

People who simply copied without even having any originality, self conducted work, would fail to answer when asked anything about this competition, you won't, and that's what you should be proud of.

### [3] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-08-27 17:25 UTC · ▲2 · LB rank 641

Kaggle like most of the world often has competing goals and objectives.  They will tweak and modify the rules over time to stay aligned with their goals.

For sure, the host's don't get much benefit from 2000 copy and paste notebooks with very minor changes.  I am sure they would rather see a wide variety of approaches.

Those of us on kaggle also have a wide range of goals.  If your only goal is to get a high kaggle ranking to fill your CV, than copy and paste at the last minute might work.

If your goal is too learn a skill set that might serve you for many years, copy and paste teaches you nothing.

It's your path to walk - kaggle might occasionally over time change some of the rules of that journey.
