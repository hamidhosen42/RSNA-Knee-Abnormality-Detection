"""E13 preprocessing: ONE implementation, imported by the cache builder (train) and the
submission leg (test), so the two can never drift apart. E11 failed on the LB after its test
inputs were rebuilt by separate code from an unpublished corpus recipe.

A study becomes a uint8 array (N_VIEWS, N_SLICES, RES, RES) plus a presence mask:
  views   Axial fluid, Sagittal non-fluid, Coronal fluid, Sagittal fluid (the four views
          aastikrajan15's public 0.945 leg reads; no laterality flip, which they measured as
          harmful: LB 0.915 -> 0.909)
  series  per view, candidates ordered by DICOM count (most first, then UID); first readable
  order   by position along the slice normal, sign-normalised; InstanceNumber only if the
          geometry is unreadable; never by filename
  slices  N_SLICES evenly spaced over the whole stack (all of them when the stack is shorter,
          then edge-padded by repetition so the array keeps a fixed shape)
  pixels  modality LUT, MONOCHROME1 inverted, one 0.5-99.5 percentile window per series,
          physical centre crop CROP_MM per axis (clamped to the full frame when the FOV is
          already smaller), INTER_AREA resize to RES
"""
import glob
import os

import cv2
import numpy as np

VIEWS = (("Axial", 1), ("Sagittal", 0), ("Coronal", 1), ("Sagittal", 1))
N_SLICES = 32
RES = 224
CROP_MM = 140.0
CLIP_PCT = (0.5, 99.5)
SPEC = {"views": [list(v) for v in VIEWS], "n_slices": N_SLICES, "res": RES, "crop_mm": CROP_MM,
        "clip_pct": list(CLIP_PCT), "version": "e13-prep-v1"}


def _position_key(hdrs):
    try:
        o = np.array(hdrs[0].ImageOrientationPatient, float)
        nrm = np.cross(o[:3], o[3:])
        if nrm[np.argmax(np.abs(nrm))] < 0:      # same direction for every series of a plane
            nrm = -nrm
        return [float(np.dot(np.array(h.ImagePositionPatient, float), nrm)) for h in hdrs]
    except Exception:
        return [float(getattr(h, "InstanceNumber", 0) or 0) for h in hdrs]


def _decode(path):
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut
    d = pydicom.dcmread(path)
    a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def _crop_fracs(hdr, shape):
    """Fraction of rows/cols to keep for a CROP_MM field of view, each axis on its own."""
    try:
        pr, pc = (float(v) for v in hdr.PixelSpacing)
    except Exception:
        return 1.0, 1.0
    h, w = shape
    fr = CROP_MM / (pr * h) if pr > 0 else 1.0
    fc = CROP_MM / (pc * w) if pc > 0 else 1.0
    return min(fr, 1.0), min(fc, 1.0)


def load_series(sdir):
    """(N_SLICES, RES, RES) uint8 for one series directory, or None if nothing is readable."""
    import pydicom
    files = sorted(glob.glob(os.path.join(sdir, "*.dcm")))
    hdrs = []
    for f in files:
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
            h.filename = f
            hdrs.append(h)
        except Exception:
            continue
    if not hdrs:
        return None
    key = _position_key(hdrs)
    hdrs = [hdrs[i] for i in np.argsort(key, kind="stable")]
    n = len(hdrs)
    pick = np.linspace(0, n - 1, min(N_SLICES, n)).round().astype(int)
    arrs = []
    for i in pick:
        try:
            arrs.append((_decode(hdrs[i].filename), hdrs[i]))
        except Exception:
            continue
    if not arrs:
        return None
    lo, hi = np.percentile(np.concatenate([a.ravel() for a, _ in arrs]), CLIP_PCT)
    out = []
    for a, h in arrs:
        a = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
        fr, fc = _crop_fracs(h, a.shape)
        kr, kc = max(int(round(a.shape[0] * fr)), 16), max(int(round(a.shape[1] * fc)), 16)
        y0, x0 = (a.shape[0] - kr) // 2, (a.shape[1] - kc) // 2
        a = cv2.resize(a[y0:y0 + kr, x0:x0 + kc], (RES, RES), interpolation=cv2.INTER_AREA)
        out.append((a * 255.0 + 0.5).astype(np.uint8))
    vol = np.stack(out)
    if len(vol) < N_SLICES:                       # short stack: stretch to the fixed length
        vol = vol[np.linspace(0, len(vol) - 1, N_SLICES).round().astype(int)]
    return vol


def build_study(study_uid, series_rows, series_root):
    """series_rows: records of (train|test)_series.csv for this study.
    Returns (vol uint8 (N_VIEWS, N_SLICES, RES, RES), mask uint8 (N_VIEWS,), chosen series UIDs)."""
    vol = np.zeros((len(VIEWS), N_SLICES, RES, RES), np.uint8)
    mask = np.zeros(len(VIEWS), np.uint8)
    chosen = []
    for v, (plane, fluid) in enumerate(VIEWS):
        cands = [r for r in series_rows
                 if str(r["Anatomical_Plane"]) == plane and int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        sizes = {str(r["SeriesInstanceUID"]): len(glob.glob(os.path.join(series_root, str(study_uid),
                                                                         str(r["SeriesInstanceUID"]), "*.dcm")))
                 for r in cands}
        got = None
        for uid in sorted(sizes, key=lambda u: (-sizes[u], u)):
            try:
                got = load_series(os.path.join(series_root, str(study_uid), uid))
            except Exception:
                got = None
            if got is not None:
                vol[v], mask[v] = got, 1
                chosen.append(uid)
                break
        if got is None:
            chosen.append("")
    return vol, mask, chosen
