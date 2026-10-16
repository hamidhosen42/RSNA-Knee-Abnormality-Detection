"""Turn each DICOM series into a canonical uint8 volume [n_slices, SIZE, SIZE].

Canonical means: fixed physical pixel size, centre crop to a fixed field of view,
standard in-plane orientation, standard slice order and a per-series intensity window.
The same `load_series` is used for the training cache and for inference.

Usage: python src/preprocess.py [train|test] [n_series_limit]
"""
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import pydicom

DATA = "/mnt/ssd/rsna_knee/data"
WORK = "/mnt/ssd/rsna_knee/work"

MM_PER_PX = 0.4
SIZE = 384  # 153.6 mm field of view
MAX_SLICES = 64

# plane -> (normal axis, slice-order sign, (col axis, sign), (row axis, sign)) in LPS coordinates
# Sagittal: slices right->left, image columns anterior->posterior, rows superior->inferior
# Coronal:  slices anterior->posterior, columns patient-right->left, rows superior->inferior
# Axial:    slices superior->inferior, columns patient-right->left, rows anterior->posterior
CANON = {
    0: (+1, (1, +1), (2, -1)),
    1: (+1, (0, +1), (2, -1)),
    2: (-1, (0, +1), (1, +1)),
}
PLANES = ["Sagittal", "Coronal", "Axial"]


def _read(path):
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    if ds.get("PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = arr.max() - arr
    slope, inter = ds.get("RescaleSlope"), ds.get("RescaleIntercept")
    if slope is not None and inter is not None:
        arr = arr * float(slope) + float(inter)
    iop = [float(v) for v in ds.ImageOrientationPatient] if "ImageOrientationPatient" in ds else None
    ipp = [float(v) for v in ds.ImagePositionPatient] if "ImagePositionPatient" in ds else None
    ps = [float(v) for v in ds.PixelSpacing] if "PixelSpacing" in ds else None
    inst = float(ds.get("InstanceNumber") or 0)
    frames = list(arr) if arr.ndim == 3 else [arr]
    return [(f, iop, ipp, ps, inst + i * 1e-3) for i, f in enumerate(frames)]


def _canon_slice(img, iop, ps, plane):
    """Reorient one slice to the canonical in-plane orientation; returns (img, (row_mm, col_mm))."""
    ps = ps or [MM_PER_PX * img.shape[0] / SIZE] * 2  # no spacing: assume the image spans the crop
    if iop is None:
        return img, ps
    r, c = np.array(iop[:3]), np.array(iop[3:])
    _, (cax, csgn), (rax, rsgn) = CANON[plane]
    if abs(r[cax]) < abs(r[rax]):  # rows and columns are swapped relative to canonical
        img, r, c, ps = img.T, c, r, ps[::-1]
    if r[cax] * csgn < 0:
        img = img[:, ::-1]
    if c[rax] * rsgn < 0:
        img = img[::-1]
    return img, ps


def _resample(img, ps):
    """Centre crop to SIZE*MM_PER_PX mm, resample to MM_PER_PX, zero-pad to SIZE x SIZE."""
    out = np.zeros((SIZE, SIZE), np.float32)
    h, w = img.shape
    ch = min(h, int(round(SIZE * MM_PER_PX / ps[0])))
    cw = min(w, int(round(SIZE * MM_PER_PX / ps[1])))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = np.ascontiguousarray(img[y0:y0 + ch, x0:x0 + cw])
    nh = max(1, min(SIZE, int(round(ch * ps[0] / MM_PER_PX))))
    nw = max(1, min(SIZE, int(round(cw * ps[1] / MM_PER_PX))))
    interp = cv2.INTER_AREA if nh < ch else cv2.INTER_LINEAR
    res = cv2.resize(crop, (nw, nh), interpolation=interp)
    oy, ox = (SIZE - nh) // 2, (SIZE - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = res
    return out


def load_series(sdir, plane_name=None):
    """Returns (uint8 volume [n, SIZE, SIZE], info dict). Unreadable slices are skipped."""
    slices, bad = [], 0
    for fn in os.listdir(sdir):
        try:
            slices.extend(_read(os.path.join(sdir, fn)))
        except Exception:
            bad += 1
    info = {"n_raw": len(slices), "n_bad": bad}
    if not slices:
        return np.zeros((1, SIZE, SIZE), np.uint8), info

    iop = next((s[1] for s in slices if s[1] is not None), None)
    if iop is not None:
        normal = np.cross(iop[:3], iop[3:])
        plane = int(np.abs(normal).argmax())
    else:
        normal, plane = None, PLANES.index(plane_name) if plane_name in PLANES else 0
    info["plane"] = PLANES[plane]

    if normal is not None and all(s[2] is not None for s in slices):
        sgn = CANON[plane][0] * np.sign(normal[plane])
        slices.sort(key=lambda s: float(np.dot(s[2], normal)) * sgn)
        pos = [float(np.dot(s[2], normal)) for s in slices]
        info["slice_gap"] = float(np.median(np.abs(np.diff(pos)))) if len(pos) > 1 else 0.0
    else:
        slices.sort(key=lambda s: s[4])
    if len(slices) > MAX_SLICES:
        keep = np.linspace(0, len(slices) - 1, MAX_SLICES).round().astype(int)
        slices = [slices[i] for i in keep]

    vol = np.stack([_resample(*_canon_slice(s[0], s[1], s[3], plane)) for s in slices])
    lo, hi = np.percentile(vol[:, ::4, ::4], [1, 99.5])
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
    info["n"] = len(vol)
    return (vol * 255).round().astype(np.uint8), info


def _job(args):
    split, study, series, plane, out_dir = args
    out = f"{out_dir}/{series}.npy"
    info = {"study": study, "series": series}
    if os.path.exists(out):
        return None
    try:
        vol, extra = load_series(f"{DATA}/{split}_series/{study}/{series}", plane)
        np.save(out + ".tmp.npy", vol)
        os.replace(out + ".tmp.npy", out)
        info.update(extra)
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"[:200]
    return info


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    out_dir = f"{WORK}/cache{SIZE}_{split}"
    os.makedirs(out_dir, exist_ok=True)
    se = pd.read_csv(f"{DATA}/{split}_series.csv")
    if limit:
        se = se.sample(limit, random_state=0)
    tasks = [(split, a, b, p, out_dir) for a, b, p in zip(se.StudyInstanceUID, se.SeriesInstanceUID, se.Anatomical_Plane)]
    rows = []
    with Pool(18) as p:
        for i, r in enumerate(p.imap_unordered(_job, tasks, chunksize=4)):
            if r:
                rows.append(r)
            if i % 1000 == 0:
                print(f"{i}/{len(tasks)}", flush=True)
    df = pd.DataFrame(rows)
    meta = f"{WORK}/cache{SIZE}_{split}_meta.csv"
    if os.path.exists(meta) and len(df):
        df = pd.concat([pd.read_csv(meta), df]).drop_duplicates("series", keep="last")
    if len(df):
        df.to_csv(meta, index=False)
        print("done", len(df), "errors:", int(df["error"].notna().sum()) if "error" in df else 0)





# ======================================================================= shard kernel (ours)
import glob, hashlib, subprocess, time
SHARD, N_SHARDS = 0, 5
BUDGET_S = 11.3 * 3600
MAX_BYTES = 18.5e9
T0 = time.time()


def _find(name):
    for pat in ["/kaggle/input/*/", "/kaggle/input/*/*/", "/kaggle/input/*/*/*/"]:
        for d in sorted(glob.glob(pat)):
            if os.path.exists(os.path.join(d, name)):
                return d.rstrip("/")
    raise FileNotFoundError(name)


def _install_decoders():
    try:
        import pylibjpeg  # noqa: F401
        return "present"
    except ImportError:
        pass
    assets = _find("cnxt_v0_fold0.pt")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--find-links", assets,
                    "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=False)
    return "installed from " + assets


def _encode(vol):
    bufs = [cv2.imencode(".png", s, [cv2.IMWRITE_PNG_COMPRESSION, 6])[1].tobytes() for s in vol]
    off = np.cumsum([0] + [len(b) for b in bufs]).astype(np.int64)
    return np.frombuffer(b"".join(bufs), np.uint8), off


def _shard_job(args):
    root, study, series, plane, out_dir = args
    info = {"study": study, "series": series}
    out = f"{out_dir}/{series}.npz"
    try:
        vol, extra = load_series(f"{root}/train_series/{study}/{series}", plane)
        data, off = _encode(vol)
        np.savez(out + ".tmp.npz", data=data, off=off)
        os.replace(out + ".tmp.npz", out)
        info.update(extra)
        info["bytes"] = int(len(data))
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"[:200]
    return info


if __name__ == "__main__":
    print("decoders:", _install_decoders(), flush=True)
    root = _find("train_series.csv")
    out_dir = "/kaggle/working/cache384"
    os.makedirs(out_dir, exist_ok=True)
    se = pd.read_csv(f"{root}/train_series.csv")
    pick = se.StudyInstanceUID.map(lambda s: int(hashlib.md5(str(s).encode()).hexdigest(), 16) % N_SHARDS == SHARD)
    se = se[pick].sort_values(["StudyInstanceUID", "SeriesInstanceUID"])
    tasks = [(root, a, b, p, out_dir) for a, b, p in zip(se.StudyInstanceUID, se.SeriesInstanceUID, se.Anatomical_Plane)]
    print(f"shard {SHARD}/{N_SHARDS}: {se.StudyInstanceUID.nunique()} studies, {len(tasks)} series | data {root}", flush=True)
    rows, total, stopped = [], 0, None
    with Pool(4) as p:
        for i, r in enumerate(p.imap(_shard_job, tasks, chunksize=2)):
            rows.append(r)
            total += r.get("bytes", 0)
            if i % 250 == 0:
                print(f"{i}/{len(tasks)} {total / 1e9:.2f} GB {(time.time() - T0) / 60:.0f} min", flush=True)
            if time.time() - T0 > BUDGET_S or total > MAX_BYTES:
                stopped = f"stopped at {i + 1}/{len(tasks)}: {'time' if time.time() - T0 > BUDGET_S else 'size'}"
                p.terminate()
                break
    df = pd.DataFrame(rows)
    df.to_csv("/kaggle/working/cache384_meta.csv", index=False)
    print(f"done {len(df)} series, errors {int(df['error'].notna().sum()) if 'error' in df else 0}, "
          f"{total / 1e9:.2f} GB, {(time.time() - T0) / 3600:.2f} h" + (f" | {stopped}" if stopped else ""), flush=True)
