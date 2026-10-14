# 9h runtime question

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740474
- **Author:** **bdbotond** (@bdbotond, Contributor) · _topic author_ · 2026-09-09 19:01 UTC · ▲1 · LB rank 4167
- **Topic votes:** 1 · **Comments/replies:** 13 · **Pinned:** no
- **Last comment:** 2026-09-20 08:33 UTC by Lavin Wins
- **Cloned:** 2026-10-14 18:23 UTC (topic id 740474, raw JSON `_raw/740474.json`)

---

Not clear for me that the 9h evaluation runtime, only counting on the test part, or it has to include the training time of the model as well?

---

## Comments (13, threaded, oldest first)

### [1] Dread Development

**Dread Development** (@dreaddevelopment, Expert) · 2026-09-09 22:34 UTC · LB rank 144

Hey! So that 9 Hour grading/scoring time is not a floor/ceiling at all & it can be improved! There are several methods to increase the time, I have scored/graded between 23 minutes & 9 hours with every combo in between.
My best as of 9/9/2026 6:32 EST is 29.5 minutes placing my team at 24th in efficiency, I have another in now that is a 23 minute run that should reflect once the host updates the list today & dont hold me to it but I believe the current leader is down to 5 minutes so thats the time to beat!!

#### ↳ [1.1] Scott Willis

**Scott Willis** (@scottgwillis, Contributor) · 2026-09-10 00:36 UTC · ▲2 · LB rank 3

Interesting to see what someone else's timing is and where they are at the the efficiency leaderboard!  My best is around 5 minutes, probably closer to 6.  I'm not too sure it's worth trying to lower it even more at this point.

##### ↳ ↳ [1.1.1] Cody_Null

**Cody_Null** (@cody11null, Grandmaster) · 2026-09-17 14:09 UTC

Just to make sure I understand this is the time from the notebook submission until you have a score on the leaderboard right? I only check just because that time is so impressive that it could be mistaken for a save and run all notebook time 😂

##### ↳ ↳ [1.1.2] Tom Aindow

**Tom Aindow** (@taindow, Master) · 2026-09-17 14:24 UTC · ▲1 · LB rank 81

Just incredible that you can hit 5 or 6 minutes at inference! You must be sub sampling the dicoms right? I cannot even read them all in that time let alone generate predictions :D can't wait to see your final solution ..

##### ↳ ↳ [1.1.3] Lavin Wins

**Lavin Wins** (@lavinwins, Contributor) · 2026-09-17 14:53 UTC · LB rank 278

That's really impressive , my bare minimum is 5-6 hours , and your is just 5 mins . It shows the level between us.
Damn!! I need to work harder

##### ↳ ↳ [1.1.4] Scott Willis

**Scott Willis** (@scottgwillis, Contributor) · 2026-09-17 16:25 UTC · ▲1 · LB rank 3

Yep, that's the time from hitting submit to getting a score.  I spent so much time trying to figure out how to get read times lower.  It's still probably 60 or 70% of my submission time

##### ↳ ↳ [1.1.5] Cody_Null

**Cody_Null** (@cody11null, Grandmaster) · 2026-09-17 17:55 UTC

Given that and the LB I really hope to see you hold strong in private leaderboard. Hoping to learn lots from your solution!

##### ↳ ↳ [1.1.6] Lavin Wins

**Lavin Wins** (@lavinwins, Contributor) · 2026-09-19 11:41 UTC · ▲-1 · LB rank 278

since , you are no. 1 in this competition . So there is some difference in the working method ofc.
Can you give some advice tho.

##### ↳ ↳ [1.1.7] Scott Willis

**Scott Willis** (@scottgwillis, Contributor) · 2026-09-19 18:24 UTC · ▲4 · LB rank 3

I'll be honest, I just tried things.  I used a small model that I can train fairly quick so I could just keep iterating as fast as I could.  This isn't something I'm experienced with.  It was a lot of reading papers, chatting with LLMs about what I didn't know, and then reading more papers to make sure the information was correct.  My goal wasn't necessarily to win and even though I'm 1st currently.  I'm expecting things to change quite a bit with the private leaderboard.

I've trained around 900 models at this point.  Considering I'm someone who doesn't work in machine learning, or even in tech, so I've just been trying whatever idea comes into my head.  It's also the reason I don't feel confident my score will hold up.

##### ↳ ↳ [1.1.8] Jack

**Jack** (@jackvd, Master) · 2026-09-19 19:01 UTC · ▲1 · LB rank 789

900 models is insane 🤣 I respect it. Be more confident in yourself. If you have some sort of validation set you're confident in, you'll be fine on the private leaderboard, even if your score drops somewhat. I can't image there will be that much shake-up in this competition anyways. 

Your training times must be incredibly fast for 900 models or you've been training/experimenting 24/7 lol

##### ↳ ↳ [1.1.9] Cody_Null

**Cody_Null** (@cody11null, Grandmaster) · 2026-09-20 01:46 UTC

I agree with Jack, if you are also getting good CV on your out of fold scores and your gold score then you should be fine. Sure it might shift a bit, but the trend should be good.

##### ↳ ↳ [1.1.10] Lavin Wins

**Lavin Wins** (@lavinwins, Contributor) · 2026-09-20 08:33 UTC · LB rank 278

That's really helpful insight. One more question like which particular model are you training?

### [2] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-09-10 02:57 UTC · LB rank 635

Your 9 hours does not need to include anything but the generation of the submission file by predicting on the hidden test file.  You can load your model by way of a dataset or use a kaggle notebook with the trained models.

If using kaggle to train I believe the current time restriction is 12 hours for a single run.  But you can break the kaggle training into multiple notebooks if needed.

I do all my training on local computer - my current attempt is pretty slow and taking almost 24 hours to train.
