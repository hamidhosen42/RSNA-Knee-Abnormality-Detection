# Slice order in the public baseline is basically random (worth +0.028 AUC)

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745759
- **Author:** **willyp74** (@willyp74, Contributor) · _topic author_ · 2026-10-04 07:33 UTC · ▲1 · LB rank 4197
- **Topic votes:** 1 · **Comments/replies:** 1 · **Pinned:** no
- **Last comment:** 2026-10-04 15:38 UTC by OmerZalman
- **Cloned:** 2026-10-09 09:46 UTC (topic id 745759, raw JSON `_raw/745759.json`)

---

I was working from one of the shared baselines and ran into something that I think affects most of the forks.

read_slot() builds the slice stack with sorted(filename). The filenames are SOP Instance UIDs, so that order has nothing to do with anatomy.

I checked it on 60 random series, comparing filename order against ImagePositionPatient projected on the axis with most variation:

median |Spearman rho| 0.164, 0.0% of series above 0.9, 98.3% below 0.5. Nothing reversed either, so it is not a flipped-stack problem, it is just shuffled.

Two things follow from that. The "central 60% of the stack" that read_slot samples is not the middle of the knee, it is 9 arbitrary positions. And the 3 slices stacked as channels are not neighbours. Also normalise_laterality() flips sagittal stacks for right knees, and that flip only means something if the order was right to begin with.

What I measured, one fold, same seed, same folds, 12 epochs, nothing else changed: filename order 0.7695, IPP order 0.7971, so +0.0276.

I also ran the patched version again with a different seed, just to see how much of that is noise: 0.7960. So seed noise is around 0.001 and the effect is about 25x that.

The fix is just sorting by ImagePositionPatient along the through-plane axis, with InstanceNumber as a fallback. One warning if you try it: do it inside read_slot and cache it per series. I first put it in probe() and it ended up reading headers for all ~820k files, which added about 7 minutes to every run before I noticed.

Happy to share the cell if anyone wants it.

---

## Comments (1, threaded, oldest first)

### [1] OmerZalman

**OmerZalman** (@omerzalman, Contributor) · 2026-10-04 15:38 UTC · ▲-1 · LB rank 4704

what seed was best for you?
