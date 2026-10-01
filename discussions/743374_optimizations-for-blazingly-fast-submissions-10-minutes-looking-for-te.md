# Optimizations for blazingly fast submissions (<10 minutes!) + Looking for teammates :)

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743374
- **Author:** **Prateek Grover** (@prateekgroverin, Contributor) · _topic author_ · 2026-09-25 19:18 UTC · ▲9 · LB rank 185
- **Topic votes:** 9 · **Comments/replies:** 5 · **Pinned:** no
- **Last comment:** 2026-09-28 16:26 UTC by Volkopat
- **Cloned:** 2026-10-01 19:24 UTC (topic id 743374, raw JSON `_raw/743374.json`)

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

## Comments (5, threaded, oldest first)

### [1] Chris Deotte

**Chris Deotte** (@cdeotte, Grandmaster) · 2026-09-25 23:51 UTC · LB rank 100

Nice optimization, good work!

### [2] (unknown)

**(unknown)** · 2026-09-27 04:36 UTC

_(no content — deleted or empty)_

### [3] Lavin Wins

**Lavin Wins** (@lavinwins, Contributor) · 2026-09-27 04:37 UTC · ▲-1 · LB rank 410

Hey! I read your post about optimizing inference down to under 10 minutes—that's seriously impressive work, especially the custom DICOM parsing and bypassing pydicom overhead.

I'd love to team up. I meet your requirements because I currently have a robust single model submission scoring 0.943 and deep experience working with target engineering and noisy labels for this type of medical imaging data.

With your infrastructure and speed optimizations combined with my model performance and label strategy, I think we'd make a really strong team for the leaderboard. Let me know if you're down to merge forces!

#### ↳ [3.1] Lavin Wins

**Lavin Wins** (@lavinwins, Contributor) · 2026-09-28 12:57 UTC · ▲-1 · LB rank 410

Are you there @prateekgroverin

### [4] Volkopat

**Volkopat** (@volkopat, Contributor) · 2026-09-28 16:26 UTC · LB rank 10

Niceeeeeee!
