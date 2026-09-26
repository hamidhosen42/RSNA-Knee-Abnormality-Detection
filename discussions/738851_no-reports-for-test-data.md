# No Reports for Test Data?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738851
- **Author:** **Ahmed Ali** (@alia22, Contributor) · _topic author_ · 2026-09-01 23:16 UTC
- **Topic votes:** 0 · **Comments/replies:** 4 · **Pinned:** no
- **Last comment:** 2026-09-02 05:22 UTC by Komil Parmar
- **Cloned:** 2026-09-26 16:24 UTC (topic id 738851, raw JSON `_raw/738851.json`)

---

Hi all,
I noticed that the test data provided does not include the "Report" field, and the dataset description says "The Report field will not be provided at the testing stage." How exactly are we supposed to use the reports if they aren't in the test data? Wouldn't that mean that any model that is trained on the reports would not be usable on the test data? Is there something I'm missing here, or are the reports completely useless for the competition?

---

## Comments (4, threaded, oldest first)

### [1] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-09-02 01:21 UTC · ▲2 · LB rank 641

You identified the problem.  You need to build a model that uses the 12 categories as labels based on examination of only the dicom images.

To get the training data you have to convert the reports to labels between 0.0 and 1.0 for each of the 12 classes.  Not an easy task.  The best set of labels probably wins the competition.

You want to learn how to make labels - dig in.   

You only want to learn the prediction model than a number of folks have shared a set of pretty decent labels.   

Learning to make labels is a skill set you need in the real world.   Not one that LLM's seem to have mastered yet.

#### ↳ [1.1] Ahmed Ali

**Ahmed Ali** (@alia22, Contributor) · _topic author_ · 2026-09-02 01:29 UTC

Thanks for the clarification, I didn't realize that most of the training data isn't labeled. So if I understand correctly, I use the labeled training data to train a model on the reports to create labels for the rest of the data, then use those labels to train a different model on the images? Interesting.

##### ↳ ↳ [1.1.1] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-09-02 01:33 UTC · ▲1 · LB rank 641

You got it!

Yes - only 58 of the images are labeled out of the 4700.   

I been here for almost 9 years - have not participated in every competition but I think this is the first were we were not handed the labels.  Back in the day when I was working for real I had to do a lot of label creation - sadly I sucked at it.   You can find the shared labels searching in the discussion posts.

#### ↳ [1.2] Komil Parmar

**Komil Parmar** (@komilparmar, Expert) · 2026-09-02 05:22 UTC · ▲1 · LB rank 59

@pcjimmmy nicely shared. I started to read works on Pseudo Labelling which is probably the single best investment anyone can make in this competition. And I was stunned by how many papers are there just regarding pseudoabelling. I also suggested the paperswithcode site to add that as a separate category and **they added it**. 🤯

The problem though which one will realize while applying those methods is that all of those papers are regarding pseudo labelling on unlabeled data. That isn't the situation here. We already have near perfect labels for those samples achieved using llms. And if you replace them or even update them via pseudoabelling, the model will perform worse then simply training on LLM labels.

So we have to learn how to update and when to update those pseudo labels. That's the key. 🔑
