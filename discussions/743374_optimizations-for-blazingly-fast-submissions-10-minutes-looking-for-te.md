# Optimizations for blazingly fast submissions (<10 minutes!) + Looking for teammates :)

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743374
- **Author:** **Prateek Grover** (@prateekgroverin, Contributor) · _topic author_ · 2026-09-25 19:18 UTC · ▲11 · LB rank 1526
- **Topic votes:** 11 · **Comments/replies:** 7 · **Pinned:** no
- **Last comment:** 2026-10-10 11:48 UTC by Nikita Ageev
- **Cloned:** 2026-10-14 18:23 UTC (topic id 743374, raw JSON `_raw/743374.json`)

---

On inference, I went from ~40 min to under 10 with a CoAtNet. A few things that helped, which might transfer:

- One model copy per GPU, each with its own shard of studies - replacing DataParallel, which re-broadcasts every parameter on each forward call. 
- Read each DICOM once, concurrently. One threaded pass per series serves both slice ordering and pixels. A single reader gets ~59 MB/s from /kaggle/input; 16 threads get ~372.
- Parse DICOM headers directly instead of through pydicom: 0.307 → 0.058 ms/file.
- Read pixels straight from the PixelData offset (np.frombuffer) - bit-identical to pydicom, 1.22 → 0.33 ms/slice.
- Take the 1st/99th percentiles on a strided subsample. np.percentile sorts, and that sort was ~60% of decode time.
- fp16 weights with half inputs, no autocast - 2.08× over autocast on a T4.
- uint8 to the GPU, normalise on device, pinned memory, non-blocking copies, persistent workers.

***Looking for teammates strong on labels/targets or bringing a resilient single model submission > 0.940.***

---

## Comments (7, threaded, oldest first)

### [1] Chris Deotte

**Chris Deotte** (@cdeotte, Grandmaster) · 2026-09-25 23:51 UTC · LB rank 194

Nice optimization, good work!

### [2] (unknown)

**(unknown)** · 2026-09-27 04:36 UTC

_(no content — deleted or empty)_

### [3] Lavin Wins

**Lavin Wins** (@lavinwins, Contributor) · 2026-09-27 04:37 UTC · ▲-1 · LB rank 1006

Hey! I read your post about optimizing inference down to under 10 minutes—that's seriously impressive work, especially the custom DICOM parsing and bypassing pydicom overhead.

I'd love to team up. I meet your requirements because I currently have a robust single model submission scoring 0.943 and deep experience working with target engineering and noisy labels for this type of medical imaging data.

With your infrastructure and speed optimizations combined with my model performance and label strategy, I think we'd make a really strong team for the leaderboard. Let me know if you're down to merge forces!

#### ↳ [3.1] Lavin Wins

**Lavin Wins** (@lavinwins, Contributor) · 2026-09-28 12:57 UTC · ▲-1 · LB rank 1006

Are you there @prateekgroverin

### [4] Volkopat

**Volkopat** (@volkopat, Contributor) · 2026-09-28 16:26 UTC · LB rank 32

Niceeeeeee!

### [5] Nikita Ageev

**Nikita Ageev** (@truenikita, Contributor) · 2026-10-04 16:14 UTC · LB rank 1743

Hi Prateek, you asked for someone strong on labels/targets — that's what I've been working on. I'm Nikita (truenikita), solo. To be upfront: my best 0.943 on the board is the public blend, and I have no Kaggle medals.

What I'd bring:
- An independent label set re-extracted from all 4,407 reports (9 languages) with Claude (Haiku 4.5 on every report, Sonnet on ~1,300 ambiguous ones). Honest numbers on the 58 gold: macro-AUC is on par with the public mm3 labels (±0.01 depending on the CV split), but calibration is much better (Brier and log-loss, P≈1.00) — mm3 are poorly calibrated, which matters for soft targets. I also did a manual error analysis of the 20 most disputed studies: e.g. degenerative / insufficiency-fracture edema read as contusion, "small" effusion under-called, impaction read as fracture; in ~6 of 39 cells the gold label contradicts the report text (image-only findings) — a ceiling for any text labels.
- A single CoAtNet-2 (384, ImageNet init, no public competition weights) trained on a mix of these labels and a second LLM relabel, scores **0.937 on the public LB** on its own — 0.006 below the public blend (0.943) — so it's a genuinely different strong member for an ensemble, not a copy of the public graph.
- Validation discipline: I lead retail credit-risk decisioning, where leakage and honest validation are the daily job. In my current pipeline gold is kept out of every choice, and I gate each new blend member with a 1,000-sample bootstrap on gold — useful protection against shake-up with a 0.001–0.002 margin over the medal line.
- GPU outside the Kaggle quota (RTX 4090 on vast.ai; a 16-epoch CoAtNet-2 at 384 takes ~3.7 h), so we can afford more seeds/folds of your model.

What I'm asking: a merge before the deadline. Submissions are fine — I've used 4, you 44. Happy to go through the approach here first; labels and code only after the merge, per the rules.

### [6] Nikita Ageev

**Nikita Ageev** (@truenikita, Contributor) · 2026-10-10 11:48 UTC · ▲-2 · LB rank 1743

Solo, 0.949 public (that is nartaa's public notebook run as is, no credit claimed). Looking to team up before Oct 15.

What I can bring:
- LLM-read labels for all 4,407 training reports (12 findings, three independent reads, stored as probabilities), plus an honest A/B on held-out gold: the triple-read labels did NOT beat a simpler single-read mix. Useful negative result, happy to share.
- A 5-fold CoAtNet-2 384 leg: 0.935 public alone, 0.948 in a rank blend with the public 0.949. Weights and OOF predictions ready to share.
- Careful bookkeeping: one change per submission, a registry with gold-set deltas and bootstrap CIs.
- Background: credit-risk modelling lead, comfortable with validation design and calibration. Small GPU budget.

Looking for someone with a stronger single model or a different backbone/view strategy who wants a careful validation partner. DM or reply here.
