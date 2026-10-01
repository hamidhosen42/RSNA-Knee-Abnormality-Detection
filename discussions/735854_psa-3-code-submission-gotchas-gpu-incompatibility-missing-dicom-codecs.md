# [PSA] 3 code-submission gotchas: GPU incompatibility, missing DICOM codecs, input path quirks

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735854
- **Author:** **Istiyak Amin Santo** (@istiyakaminsanto, Contributor) · _topic author_ · 2026-08-18 13:04 UTC · ▲1 · LB rank 3209
- **Topic votes:** 1 · **Comments/replies:** 1 · **Pinned:** no
- **Last comment:** 2026-08-18 14:24 UTC by PC Jimmmy
- **Cloned:** 2026-10-01 19:24 UTC (topic id 735854, raw JSON `_raw/735854.json`)

---

Saving others some debugging time, three environment issues that silently break a submission:

**1. Default GPU (P100) is incompatible with pre-installed PyTorch**
PyTorch 2.10.0+cu128 needs compute capability sm_70+; P100 is sm_60. `cuda.is_available()` still returns `True`, so it fails silently. Fix in `kernel-metadata.json`: 

```"machine_shape": "NvidiaTeslaT4"```

**2. DICOM codec plugins (pylibjpeg/gdcm) aren't installed**, and scored runs have no internet, so you can't pip install them then. This dataset has JPEG Lossless/JPEG2000 series that need them.
Fix — download the matching wheels locally, upload as a Dataset, install offline in-kernel:

```
pip download --only-binary=:all: --platform manylinux2014_x86_64 --python-version 312 \
  --no-deps -d ./wheels pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg python-gdcm

subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                 "--find-links", wheel_dir, "pylibjpeg", "pylibjpeg-libjpeg",
                 "pylibjpeg-openjpeg", "python-gdcm"], check=True)
```

 **3.`/kaggle/input/` paths aren't what you'd expect competition data** is under `/kaggle/input/competitions/<slug>/`, and attached Datasets can land at `/kaggle/input/datasets/<owner>/<dataset>/` rather than the flat path. Don't hardcode discover at runtime with a depth-bounded os.walk (unbounded recursion will crawl all of train_series/ looking for a shallow file).

**Happy to answer questions if others hit the same walls.**

---

## Comments (1, threaded, oldest first)

### [1] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-08-18 14:24 UTC · LB rank 642

Downloading of libraries to a data set is a way to do it.  But kaggle eliminated that need some time ago.  You just pip what you need in the Add-Ons - Install Dependencies while in the notebook editor.  They even show you how to do it with a couple of examples.
