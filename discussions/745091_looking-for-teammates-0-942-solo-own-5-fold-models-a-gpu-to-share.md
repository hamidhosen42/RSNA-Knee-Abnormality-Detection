# Looking for teammates: 0.942 solo, own 5-fold models + a GPU to share

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745091
- **Author:** **KalyanG17** (@kalyang17, Contributor) · _topic author_ · 2026-10-02 07:34 UTC · LB rank 2466
- **Topic votes:** 0 · **Comments/replies:** 6 · **Pinned:** no
- **Last comment:** 2026-10-12 09:14 UTC by kkk
- **Cloned:** 2026-10-14 18:23 UTC (topic id 745091, raw JSON `_raw/745091.json`)

---

Hello, I'm looking for a team to merge with before the Oct 15 deadline.
I'm at 0.942 solo. My own models are a CoAtNet @384 (0.935), a ConvNeXt-T (0.930) and a RadImageNet ResNet-50 trained on my own report labels (0.924); two of them have 5-fold OOF. I also have my own LLM label sets, an RTX 5070 and spare Kaggle GPU hours, so I can run training for the team.
Would especially like to team up with someone whose model is built differently from the public notebooks. If you are interested reply here with your Discord/Email ID. Thanks!

---

## Comments (6, threaded, oldest first)

### [1] (unknown)

**(unknown)** · 2026-10-02 18:01 UTC

_(no content — deleted or empty)_

### [2] tnshpp

**tnshpp** (@tnshpp, Contributor) · 2026-10-05 07:49 UTC · LB rank 2457

Hey I am looking to team up (solo). Currently have LLM labels and trained Dino and radimagenet and got 0.92 on single model. I have got some online compute units. 
Discord:  tnshp4162

### [3] Adel Zebiche

**Adel Zebiche** (@adelzebiche, Contributor) · 2026-10-05 12:39 UTC · LB rank 1093

Hey  I currently have a 93.4 ViT small pretrained with DINOv2 and a ResNet-50 w RadImageNet at 92.8, and I'm also looking for a team

#### ↳ [3.1] Pand

**Pand** (@pand999, Contributor) · 2026-10-07 06:23 UTC · ▲1 · LB rank 111

ive got 5-fold bag on pretrained labels at 0.945, contact me :)
Discord: p.and; officialpand@gmail.com

### [4] Nikita Ageev

**Nikita Ageev** (@truenikita, Contributor) · 2026-10-11 13:04 UTC · ▲-4 · LB rank 1756

Kalyan, Adel, Pand: solo here (0.949 public, but that is the public nartaa notebook as is, so no credit claimed), happy to merge before Oct 15.

What I bring that is built differently from the public notebooks: my own LLM-read labels for all 4,407 reports (12 findings, three reads, stored as calibrated probabilities), a 5-fold CoAtNet-2 @384 trained on them (0.937 public alone, OOF ready), and careful validation: one change per submission, gold kept out of every choice, bootstrap check on each new blend member. Plus a rented RTX 4090 for extra seeds/folds.

If one of you is open to it, send me a team request or reply here and I'll accept the same day.

#### ↳ [4.1] kkk

**kkk** (@kangqinyao, Contributor) · 2026-10-12 09:14 UTC · LB rank 823

I am quite familiar with this competition. I have participated in several similar contests before and achieved decent rankings. To secure high scores, simply submitting open‑source solutions or making trivial tweaks to existing open‑source code will not suffice. My only bottleneck at the moment is computing power. If you have available GPU resources, we can team up to boost our performance. I have studied plenty of relevant solutions and developed many promising ideas and algorithms, so we can work jointly to go for the silver medal.
