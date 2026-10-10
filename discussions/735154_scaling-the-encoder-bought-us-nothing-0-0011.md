# Scaling the encoder bought us nothing (+0.0011)

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735154
- **Author:** **stevenleehans** (@stevenleehans, Expert) · _topic author_ · 2026-08-14 08:05 UTC · ▲14 · LB rank 1902
- **Topic votes:** 14 · **Comments/replies:** 3 · **Pinned:** no
- **Last comment:** 2026-08-17 14:45 UTC by Dung Duong (Tony) Huynh
- **Cloned:** 2026-10-09 09:46 UTC (topic id 735154, raw JSON `_raw/735154.json`)

---

## TL;DR

- Swapping **DINOv2-Small → DINOv2-Base** (22M → 87M params, 3.9× the compute per step)
  moved our CV by **+0.0011** against a measured noise floor of **0.0020**. A null.
- Only **5 of 12** labels moved in Base's favour, and the whole macro gain rests on our
  weakest label. Capacity is not what's binding here.
- Persisting the decoded pixels turns the ~500 GB of DICOM into an **11.12 GiB** array,
  which fits on a laptop. A fold now runs in **67 min on an M4 Pro at zero GPU quota**,
  versus 76 min on a Kaggle T4.
- The one genuinely transferable lesson: **a checkpoint carries its own preprocessing
  contract.** Swapping encoder weights without swapping the normalisation constants is a
  silent handicap that produces a plausible, wrong result.

You can see the process of turning the data here. feel free to fork :) 
https://www.kaggle.com/code/stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache

---

## 1. Why the 500 GB isn't really 500 GB

The dataset is large, but the model never sees most of it. After slice selection,
windowing, the physical-scale crop, and the resize to 224 px, the entire visual input to
our model is:

```
4,407 studies × 6 slots × 9 slices × 224 × 224, uint8  =  11.12 GiB
```

That's it. Everything else is decode work that gets thrown away when the kernel ends —
and then paid again on the next run.

We measured that decode at **55 minutes per run**, which was **72% of a single fold**.
Writing the array to disk once (sharded `.npy`, memory-mapped on load) removes it
permanently.

**This has a side effect worth more than the speed:** 11 GiB fits on a laptop.

| | Kaggle T4 (fp16) | Apple M4 Pro (fp32) |
|---|---|---|
| per training step | 0.227 s | 0.436 s |
| cache build | 55 min, every run | 0 |
| **one fold, wall clock** | **76 min** | **67 min** |
| GPU quota consumed | 21 min | **0** |

The laptop is 1.9× slower per step and still finishes sooner, because it never rebuilds
the cache. Our weekly GPU quota had been exhausted for five days when we ran these.

**⚠ Caveat we can't resolve:** the same config scored **0.7931** locally vs **0.7957** on
Kaggle — Δ = −0.0026, just outside our floor. We can't separate fp32-vs-fp16 numerics from
a small real deficit. So **every arm below is compared to a local control**, never
cross-machine. If you try this, do the same.

---

## 2. The experiment: does a bigger encoder help?

Every experiment we'd run to that point changed how the **pictures** were prepared —
crop geometry, slice position, sampling window. None had changed the encoder itself.

So: identical data, identical seed, identical folds, identical batch size. Only ViT-S → ViT-B.

| Encoder | Params | Trainable | s/step | Derived OOF (2,652 studies) |
|---|---|---|---|---|
| dinov2-small *(control)* | 22M | 10.7M | 0.436 | **0.7931** |
| dinov2-base | 87M | 42.5M | 1.710 | **0.7942** |

**Δ = +0.0011.** Our paired noise floor at 3 folds is **0.0020**. So this is inside the
noise — 4× the parameters for nothing measurable.

### The per-label view is worse than the average

| Target | n pos | small | base | Δ |
|---|---|---|---|---|
| MCL | 342 | 0.7529 | 0.7729 | **+0.0200** |
| Lateral OA | 401 | 0.7603 | 0.7708 | +0.0105 |
| Lateral Meniscus | 485 | 0.7385 | 0.7439 | +0.0054 |
| PF OA | 796 | 0.7423 | 0.7462 | +0.0039 |
| Medial Meniscus | 1073 | 0.7726 | 0.7761 | +0.0035 |
| Effusion | 1457 | 0.8106 | 0.8105 | −0.0001 |
| Medial OA | 525 | 0.8121 | 0.8109 | −0.0012 |
| Synovitis | 881 | 0.8206 | 0.8191 | −0.0015 |
| ACL | 573 | 0.7968 | 0.7942 | −0.0026 |
| Baker's cyst | 637 | 0.8614 | 0.8582 | −0.0032 |
| Contusion | 613 | 0.8010 | 0.7935 | −0.0075 |
| Fracture | 177 | 0.8484 | 0.8341 | −0.0143 |

**Only 5 of 12 labels move in Base's favour.** For comparison, when a crop-geometry fix
earlier in our log paid +0.0059, it moved **10 of 12** — that's what a real effect looks
like. Here the entire macro gain rests on MCL, which has only **9 expert positives** in the
58 labelled studies, i.e. our least reliable label.

No label clears our per-label gate of 0.03. This is the shape of noise, not of capacity.

---

## 3. The bug: a checkpoint carries its own preprocessing contract

The next arm was **RAD-DINO** (`microsoft/rad-dino`) — a ViT-B/14 finetuned from
`dinov2-base` on 882,775 chest radiographs. Since capacity is now priced at zero, and
RAD-DINO is architecturally identical to `dinov2-base` (same 87M params, same 12 blocks,
same patch 14), a gain could only be attributed to **what it was pretrained on**. Clean
single-lever test.

It started losing immediately and the gap widened every epoch:

| fold 0 epoch | small | base | rad-dino |
|---|---|---|---|
| 1 | 0.6997 | 0.6984 | 0.6820 |
| 3 | 0.7399 | 0.7292 | 0.7226 |
| 5 | 0.7666 | 0.7695 | **0.7352** |

Training loss was also *higher* (0.3531 vs 0.3461), so it was fitting worse, not overfitting.

The obvious reading is "medical pretraining doesn't transfer to MRI". **The actual reason
was in a config file:**

| checkpoint | `image_mean` | `image_std` | native px |
|---|---|---|---|
| dinov2-small / base | 0.485, 0.456, 0.406 | 0.229, 0.224, 0.225 | 224 |
| **rad-dino** | **0.5307** (greyscale) | **0.2583** | **518** |

Our model hardcoded the ImageNet constants as buffers:

```python
self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
```

Correct for DINOv2. **Wrong for RAD-DINO**, which was pretrained on greyscale radiographs —
so it was receiving inputs shifted ~0.2 std from its pretraining distribution with a 12%
scale error on top.

The fix is to read it from the checkpoint instead of assuming it:

```python
cfg = json.loads((ckpt_dir / "preprocessor_config.json").read_text())
mean, std = cfg.get("image_mean"), cfg.get("image_std")
```

We verified both DINOv2 checkpoints resolve to byte-identical ImageNet constants
afterwards, so the results in §2 were unaffected and didn't need re-running.

**Why this is the most useful thing in this post:** left to finish, that run would have
produced a real number, from a real 11-hour experiment, supporting a false conclusion about
the exact hypothesis we most wanted to test. It fails *silently* — no error, no warning,
just a plausible curve pointing the wrong way. If you're swapping encoders in this
competition (and plenty of us are), check `preprocessor_config.json` before you trust the
comparison.

---

## 4. Four other ways our measurements lied

Offered because they're all cheap to hit:

1. **The silent backbone fallback.** Off Kaggle there's no `/kaggle/input`, so our weights
   lookup returned `None` and the builder fell back to **ResNet-18** — which trains fine and
   logs a plausible score. Any run that doesn't *print* which backbone it loaded is not
   evidence about that backbone.

2. **The narrow probe.** Timing training steps predicted 0.64 h per fold; the fold took
   **1.11 h**, because the probe ignored validation, memory-mapped reads and augmentation.
   That was our third consecutive underestimate from a narrow probe. A probe that times one
   part of a loop does not time the loop — treat it as go/no-go, never as a schedule.

3. **Non-deterministic hardware.** On Apple's MPS backend, identical config *and* identical
   seed agreed to four decimals for two epochs and then diverged. A noise floor measured on
   one backend does not transfer to another. Measure it where you run.

4. **A tool that answered the wrong question.** Our comparison script prints its absolute
   difference under the heading `NOISE FLOOR`, because it was written to compare two seeds
   of one config. Fed a control and a treatment, it labels the **effect** as the floor —
   it reported "floor = 0.0011" for a result that sits *under* the real floor of 0.0020.

---

## 5. Where this leaves us

The encoder's **size** is not the binding constraint on this problem. That's consistent
with everything else we've measured: our gains came from crop geometry and slice position —
from how the images are prepared, not from how much model is pointed at them.

The RAD-DINO arm is re-running with correct normalisation as of writing; we'll post the
number either way, including if it's a null.

**One limitation we'll state up front so we don't overclaim later:** RAD-DINO's native
input is 518 px and we feed it 224, so it sees 16×16 patches instead of 37×37 with
interpolated position embeddings. That confound isn't cheap to fix (518 px is ~5× the
compute, and 336 px already scored nothing on our pipeline). So whatever we get, the honest
claim is about RAD-DINO **at 224 px**, not RAD-DINO in general.

---

### Notes on validation

All figures are out-of-fold on report-derived targets, 3 folds, 2,652 studies, paired
(same studies, same seed, same folds). Noise floor of 0.0020 macro was measured, not
assumed — same config, different training seed. Only 58 of 4,407 studies carry expert
labels, so that subset is a sanity check and cannot rank two models; a per-label standard
error there is 0.04–0.09.

Nothing above is a leaderboard claim. Happy to answer questions or be told we've got
something wrong — particularly if anyone has run a ViT-B encoder here and seen it pay.

---

## Comments (3, threaded, oldest first)

### [1] Dung Duong (Tony) Huynh

**Dung Duong (Tony) Huynh** (@dungduongtonyhuynh, Contributor) · 2026-08-16 07:58 UTC

Hi OG, thanks for sharing this! May I ask how you reduced each series to 9 slices? Did you randomly sample the slices, or use interpolation? Thank you!

#### ↳ [1.1] stevenleehans

**stevenleehans** (@stevenleehans, Expert) · _topic author_ · 2026-08-16 08:59 UTC · ▲1 · LB rank 1902

Hi Tony — neither, actually. No random sampling and no interpolation; it's a deterministic pick of 9 real slices, in three steps.

1. Sort the stack physically first. This one matters more than the sampling. The filenames in this dataset are SOP Instance UIDs, i.e. random — sorted filename order matches true anatomical order only about 5% of the time on this corpus. So before anything else I re-sort each series by projecting ImagePositionPatient onto the normal of ImageOrientationPatient, with fallbacks to SliceLocation → InstanceNumber → filename. That resolves by geometry on 100% of series here.

2. Place 3 anchors across a window of the sorted stack, evenly spaced with linspace (not random). I clip the ends rather than use the full stack, since the outermost slices are mostly off-joint.

3. Take the 3 physically adjacent slices around each anchor — so 3 × 3 = 9.

The reason for the 3+3+3 layout rather than 9 evenly spread slices: each group of 3 becomes the R/G/B channels of one encoder input. Adjacent slices make that triplet a genuine ~10 mm depth neighbourhood at this dataset's slice gaps. Nine evenly spread slices would cover more of the joint but hand the encoder three views ~20 mm apart, which is a slab, not a 2.5D triplet.

Two caveats worth knowing:

Interpolation isn't used because MRI slice gaps here vary a lot; I'd rather feed real slices and fix scale in-plane instead (I crop to a fixed physical extent in mm before resizing, since field of view ranges over 71 distinct values with a median of 160 mm — so a fixed pixel size alone does not fix physical scale).
On my CV, 3 slices actually beat 9 by a margin ~4× my noise floor. But that result confounds two things — with one anchor the code takes the window centre, with three it spreads them to the window edges — so I can't yet tell you whether it's the slice count or the slice position doing the work. Don't take 9 as tuned.

##### ↳ ↳ [1.1.1] Dung Duong (Tony) Huynh

**Dung Duong (Tony) Huynh** (@dungduongtonyhuynh, Contributor) · 2026-08-17 14:45 UTC

Thanks for sharing my friend!
