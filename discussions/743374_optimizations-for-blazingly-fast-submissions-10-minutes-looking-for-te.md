# Optimizations for blazingly fast submissions (<10 minutes!) + Looking for teammates :)

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743374
- **Author:** **Prateek Grover** (@prateekgroverin, Contributor) · _topic author_ · 2026-09-25 19:18 UTC · ▲2 · LB rank 131
- **Topic votes:** 2 · **Comments/replies:** 1 · **Pinned:** no
- **Last comment:** 2026-09-25 23:51 UTC by Chris Deotte
- **Cloned:** 2026-09-26 16:24 UTC (topic id 743374, raw JSON `_raw/743374.json`)

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

## Comments (1, threaded, oldest first)

### [1] Chris Deotte

**Chris Deotte** (@cdeotte, Grandmaster) · 2026-09-25 23:51 UTC · ▲-1 · LB rank 93

Nice optimization, good work!
