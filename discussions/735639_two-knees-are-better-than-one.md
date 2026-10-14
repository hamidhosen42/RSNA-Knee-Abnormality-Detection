# Two knees are better than one?

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735639
- **Author:** **Tim Krige** (@timkrige, Expert) · _topic author_ · 2026-08-17 10:12 UTC · ▲8 · LB rank 1202
- **Topic votes:** 8 · **Comments/replies:** 5 · **Pinned:** no
- **Last comment:** 2026-09-11 02:24 UTC by Greg Wehner
- **Cloned:** 2026-10-14 18:23 UTC (topic id 735639, raw JSON `_raw/735639.json`)

---

There are several (7 if I recall/count correctly) training studies where both knees are in the DICOM.

This first caused my very naive crop to show no knee. Now I am wondering if we have some interesting chance to have some clean knees. 

For example: ACL is unlikely to affect both knees, right? I do not think people typically tear both. Has anyone tried labelling and training on the good knees?

---

## Comments (5, threaded, oldest first)

### [1] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-08-17 14:59 UTC · ▲1 · LB rank 636

Nice catch - guessing that most notebooks are converting all knees to a single - to the left in my case.   Probably worthy of a little special processing, but another task for me once I get above 0.90.

But you do remind me that I really do need to confirm that I have knees with my crop :)

#### ↳ [1.1] Tim Krige

**Tim Krige** (@timkrige, Expert) · _topic author_ · 2026-08-17 17:01 UTC · ▲3 · LB rank 1202

@pcjimmmy I take it as a feather in my cap that I caught this nuance before you.

I was curious that a crop of mine was missing the bone and somehow flagging it as fracture... Had a little chuckle about what had happened.

#### ↳ [1.2] Greg Wehner

**Greg Wehner** (@gregwehner, Contributor) · 2026-09-11 02:24 UTC

If there are test cases with both knees in the field of view, do we have any way of knowing which knee corresponds to the test labels that we are supposed to be predicting?

### [2] Enai

**Enai** (@enaimaan, Contributor) · 2026-08-25 09:45 UTC · LB rank 3041

How do you convert a right knee to a left one? and how to detect which is which?

#### ↳ [2.1] Tim Krige

**Tim Krige** (@timkrige, Expert) · _topic author_ · 2026-08-26 10:45 UTC · LB rank 1202

I do not convert a right to a left knee, and thankfully not, I would be a terrible doctor.

I detect which is which with a anatomical model, and patient position. It was reasonably straight forward after looking at the examples and how the knee bends.
