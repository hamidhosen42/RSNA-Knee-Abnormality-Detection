# What kind of Hardware are using?(GPU,CPU,RAM,VRAM)

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737333
- **Author:** **JAYASURYA R** (@suryar261, Contributor) · _topic author_ · 2026-08-24 21:14 UTC · ▲-2 · LB rank 5520
- **Topic votes:** -2 · **Comments/replies:** 12 · **Pinned:** no
- **Last comment:** 2026-10-05 21:12 UTC by Chris Deotte
- **Cloned:** 2026-10-14 18:23 UTC (topic id 737333, raw JSON `_raw/737333.json`)

---

what kind of hardware are you guys using for this competition and how much score you guys getting for that hardware?

---

## Comments (12, threaded, oldest first)

### [1] k256.dev

**k256.dev** (@k256net, Master) · 2026-08-24 23:44 UTC · ▲1 · LB rank 3044

GPU: RTX5070
CPU: Ryzen 7 7700
memory 32GB
storage 1TB + 2TB(external)

efficiency LB 18th (0.929)

5-fold training takes around 3 to 15 hours, depending on the setup.
Personally, I think this level of performance is competitive enough to fight with.

#### ↳ [1.1] GG Ayo (AyoGG)

**GG Ayo (AyoGG)** (@ggayoayogg, Expert) · 2026-08-25 05:58 UTC · LB rank 48

Based on my current results, 16GB of VRAM is sufficient.

### [2] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-08-25 05:46 UTC · LB rank 1901

This competition does seem borderline, but can run on kaggle resources - however, starting from scratch not sure you could get to a 0.95 you must use some of the shared work, for sure, the label files that have been shared.

My label making has been a huge local compute - and not done yet.  Dual 16GB graphics with I9 and 128RAM.  

I sent Elon Musk a note asking if I could borrow the Nashville facility but so far no returned message.

### [3] tingyi

**tingyi** (@chengtingyi, Contributor) · 2026-08-25 05:55 UTC · ▲2 · LB rank 48

9950x
rog 5090
64g
2t + 1t
0.938：）

#### ↳ [3.1] GG Ayo (AyoGG)

**GG Ayo (AyoGG)** (@ggayoayogg, Expert) · 2026-08-25 05:59 UTC · LB rank 48

He is a total whale in this game.

### [4] Kareem Khazal

**Kareem Khazal** (@kareemkhazal, Contributor) · 2026-08-26 21:10 UTC · LB rank 556

4050 laptop gpu, i7 13th gen, 16 gb ram, and 8gb vram.

### [5] OmerZalman

**OmerZalman** (@omerzalman, Contributor) · 2026-10-05 03:15 UTC · LB rank 4700

96gb vram -> rtx 6k, and 96 core cpu,

### [6] Navneet

**Navneet** (@navneetbende, Contributor) · 2026-10-05 04:08 UTC · ▲-2

Please let us know as well what kind of Hardware are using? @suryar261

### [7] Steve421471

**Steve421471** (@steve421471, Contributor) · 2026-10-05 19:05 UTC · LB rank 609

RTX 3060 but most of my work in this competition is done using Kaggle's free GPU quota.

### [8] Chris Deotte

**Chris Deotte** (@cdeotte, Grandmaster) · 2026-10-05 20:14 UTC · ▲1 · LB rank 131

I'm using 1xL40S GPU 48GB VRAM. It's a sandbox secure VM which allows me to autorun autonomous agents. When building models with small backbones, it can run an experiment in a few hours. So in a loop, it can run about a dozen experiments per day.

#### ↳ [8.1] OmerZalman

**OmerZalman** (@omerzalman, Contributor) · 2026-10-05 21:09 UTC · ▲1 · LB rank 4700

Whos your provider? What website are you using?

##### ↳ ↳ [8.1.1] Chris Deotte

**Chris Deotte** (@cdeotte, Grandmaster) · 2026-10-05 21:12 UTC · LB rank 131

Normally, I use NVIDIA cloud for Kaggle competitions. But for agent security reasons, I am starting to use other cloud providers with my autonomous agents. In this competition, I am using [Crusoe AI][1] (via NVIDIA Brev)

[1]: https://www.crusoe.ai/
