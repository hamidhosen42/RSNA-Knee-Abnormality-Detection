"""fx_prep.py - DICOM study -> fixed 44 x 336 x 336 uint8 stack, built like the public Raptor corpus.

A model trained on `dreaddevelopment/knee-raptor-corpus(+ext)` must see test studies prepared the same way.
This is the corpus author's own inference preprocessing (dreaddevelopment/knee-mri-twelve-findings-from-a-single-model:
`build_study`, `_make_reader`, `_pick_series_for_slot`), unchanged except:
  * the 44-slice slot layout of the public corpus (12/10/8/6/8, the 0.943 notebook's `_SLOTS44`)
  * the corpus's slice span 0.15-0.85 (verified pixel-exact on Kaggle by fx-00-parity-debug)
  * module-level functions so studies can be decoded in parallel worker processes
  * `parity_check()` to verify, on training studies, that this code reproduces the corpus pixel for pixel
"""
import glob
import os

import numpy as np

IMG = 336
CROP_MM = 140.0
# The public 44-slice corpus uses the OLD middle span 0.15-0.85 (not the author's later 0.06-0.94): found by fx-00 on
# 6 training studies, mean |diff| 0.002/255 vs the corpus with 0.15-0.85 against 27.5/255 with 0.06-0.94.
SPAN_LO, SPAN_HI = 0.15, 0.85
# (plane, fluid preference, slices): fluid 1 = prefer fluid-sensitive, 0 = prefer not, -1 = either
SLOTS = [("Sagittal", 1, 12), ("Sagittal", 0, 10), ("Coronal", 1, 8), ("Coronal", 0, 6), ("Axial", -1, 8)]
MAXS = sum(s[2] for s in SLOTS)                                    # 44
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def _apply_modality_lut(arr, ds):
    try:
        from pydicom.pixels import apply_modality_lut           # pydicom >= 3
    except Exception:
        from pydicom.pixel_data_handlers.util import apply_modality_lut
    return apply_modality_lut(arr, ds)


def order_and_meta(sdir):
    """All slices of a series sorted along the slice normal, with pixel spacing (reads headers only)."""
    import pydicom
    recs, ps_list = [], []
    for f in glob.glob(sdir + "/*.dcm"):
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
            iop = getattr(h, "ImageOrientationPatient", None)
            ipp = getattr(h, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                r = np.array(iop[:3], float); c = np.array(iop[3:], float)
                pos = float(np.dot(np.array(ipp, float), np.cross(r, c)))
            else:
                pos = float(getattr(h, "InstanceNumber", 0) or 0)
            ps = getattr(h, "PixelSpacing", None)
            ps = float(ps[0]) if ps is not None else 0.5
            ps_list.append(ps); recs.append((pos, f, ps))
        except Exception:
            recs.append((0.0, f, 0.5))
    recs.sort(key=lambda x: x[0])
    return [(f, ps) for _, f, ps in recs], (float(np.median(ps_list)) if ps_list else 0.5)


def read_px(f):
    import pydicom
    d = pydicom.dcmread(f)
    a = _apply_modality_lut(d.pixel_array, d).astype(np.float32)
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def mm_crop_resize(a, ps):
    import cv2
    h, w = a.shape
    cpx = min(int(round(CROP_MM / max(ps, 1e-3))), min(h, w))
    y0, x0 = (h - cpx) // 2, (w - cpx) // 2
    return cv2.resize(a[y0:y0 + cpx, x0:x0 + cpx], (IMG, IMG), interpolation=cv2.INTER_AREA)


def pick_series_for_slot(rows, plane, fluid, used):
    cands = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def build_study(sid, rows, series_dir):
    """rows: the study's records from *_series.csv. Returns (vol uint8 [44,336,336], mask uint8 [44])."""
    vol = np.zeros((MAXS, IMG, IMG), np.uint8)
    idx, used = 0, set()
    for plane, fluid, k in SLOTS:
        r = pick_series_for_slot(rows, plane, fluid, used)
        if r is None:
            idx += k; continue
        used.add(r["SeriesInstanceUID"])
        files, med_ps = order_and_meta(f"{series_dir}/{sid}/{r['SeriesInstanceUID']}")
        if not files:
            idx += k; continue
        n = len(files)
        lo, hi = int(n * SPAN_LO), int(n * SPAN_HI) - 1
        hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        arrs, pss = [], []
        for p in picks:
            fp, ps = files[min(p, n - 1)]
            try:
                arrs.append(read_px(fp)); pss.append(ps)
            except Exception:                                       # corrupted slice -> left as zeros
                arrs.append(None); pss.append(med_ps)
        valid = [a for a in arrs if a is not None]
        if valid:
            loq, hiq = np.percentile(np.concatenate([a.ravel() for a in valid]), [2.0, 98.0])
        else:
            loq, hiq = 0.0, 1.0
        for a, ps in zip(arrs, pss):
            if idx >= MAXS:
                break
            if a is None:
                idx += 1; continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            vol[idx] = (mm_crop_resize(aw, ps if ps > 0 else med_ps) * 255).astype(np.uint8)
            idx += 1
        if idx >= MAXS:
            break
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask


def _build_one(job):
    i, sid, rows, series_dir = job
    try:
        import cv2
        cv2.setNumThreads(1)                                        # 4 worker processes on 4 cores: no oversubscription
        vol, mask = build_study(sid, rows, series_dir)
        return i, vol, mask, None
    except Exception as e:                                        # the caller fills a neutral prediction
        return i, None, None, f"{type(e).__name__}: {str(e)[:200]}"


def build_many(sids, ser_records, series_dir, workers=4, inflight=12):
    """Yield (index, vol, mask, error) as studies finish, decoding `workers` studies in parallel processes."""
    jobs = ((i, sid, ser_records.get(sid, []), series_dir) for i, sid in enumerate(sids))
    if workers <= 0:
        for job in jobs:
            yield _build_one(job)
        return
    import multiprocessing as mp
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    # "spawn", not fork: forking after PyTorch has started its thread pools can deadlock the children
    # (seen in the local test: the pool hung forever on 6 studies). Spawned workers import this module fresh.
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        pending = set()
        for job in jobs:
            pending.add(ex.submit(_build_one, job))
            if len(pending) >= inflight:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for f in done:
                    yield f.result()
        for f in pending:
            yield f.result()


def series_records(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path, dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    return {k: v.to_dict("records") for k, v in df.groupby("StudyInstanceUID")}


def find_competition_root(input_root="/kaggle/input"):
    for c in (f"{input_root}/competitions/rsna-knee-abnormality-detection", f"{input_root}/rsna-knee-abnormality-detection"):
        if os.path.isfile(f"{c}/test.csv"):
            return c
    for d, dirs, fs in os.walk(input_root):
        dirs[:] = [x for x in dirs if not x.endswith(("_series", "_images"))]
        if "test.csv" in fs and "test_series.csv" in fs:
            return d
    raise FileNotFoundError("competition data not found - attach rsna-knee-abnormality-detection")


def find_files(input_root, pattern):
    """Files matching `pattern` under input_root, never descending into the DICOM trees (*_series / *_images),
    which hold millions of files: a recursive glob over them costs minutes per run."""
    import fnmatch
    hits = []
    for d, dirs, fs in os.walk(input_root):
        dirs[:] = [x for x in dirs if not x.endswith(("_series", "_images"))]
        hits += [os.path.join(d, f) for f in fs if fnmatch.fnmatch(f, pattern)]
    return sorted(set(hits))


def series_dir(root, split):
    d = f"{root}/{split}_series"
    return d if os.path.isdir(d) else f"{root}/{split}_images"


def parity_check(vols, masks, ids, ser_records, sdir, n=16, seed=0, time_budget_s=None):
    """Rebuild n corpus studies from their DICOMs and compare with the corpus arrays (0 = identical)."""
    import time
    rng = np.random.default_rng(seed)
    rows, t0 = [], time.time()
    for r in sorted(rng.choice(len(ids), size=min(n, len(ids)), replace=False)):
        if time_budget_s and rows and time.time() - t0 > time_budget_s:
            break
        sid = str(ids[r])
        vol, mask = build_study(sid, ser_records.get(sid, []), sdir)
        ref, refm = np.asarray(vols[r]), np.asarray(masks[r])
        d = np.abs(vol.astype(np.int16) - ref.astype(np.int16))
        slot_mae, s0 = [], 0
        for _, _, k in SLOTS:
            slot_mae.append(round(float(d[s0:s0 + k].mean()), 2)); s0 += k
        rows.append({"study": sid[-12:], "mean_abs_diff": round(float(d.mean()), 3),
                     "pixels_identical": round(float((d == 0).mean()), 4), "pixels_within_2": round(float((d <= 2).mean()), 4),
                     "mask_equal": bool((mask == (refm > 0).astype(np.uint8)).all()), "slot_mean_abs_diff": slot_mae})
    return rows
