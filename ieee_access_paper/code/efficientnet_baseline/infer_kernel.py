#!/usr/bin/env python3
"""Our leg: inference, run as a subprocess by the submission notebook ("OUR LEG 2/3").

Rebuilds every test study exactly the way the training corpus was built (the spec travels inside
each checkpoint), scores it with every checkpoint found, averages probabilities across
checkpoints and writes per-finding percentile ranks, in the same shape as D4's fleet output.

Environment:
  RSNA_ARMS_DIRS  os.pathsep-separated roots searched for arms/<arm>/weights/*.pt
  RSNA_CKPT_GLOB  checkpoint filename pattern (default *_best.pt)
  RSNA_OUT        output directory for submission.csv and ours_leg_receipt.json
  RSNA_WORKERS    preprocessing worker processes (default 4)
  RSNA_BUDGET_S   wall-clock budget; the leg exits non-zero rather than overrun it
  RSNA_EXPECTED_WEIGHTS  optional exact checkpoint count
  RSNA_COMP_ROOT  optional competition-data root (local tests)
Exit codes: 0 ok; 2 no/invalid checkpoints; 3 budget exceeded; 4 too many failed studies.
"""
import fnmatch, glob, hashlib, json, os, sys, time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

torch.backends.cudnn.benchmark = True
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
T0 = time.time()


def log(msg):
    print(f"[ours-leg {time.time() - T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------
# Model (must match train_ours_leg.py)
# ---------------------------------------------------------------------------------------------
def build_backbone(arch):
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=False, num_classes=0, in_chans=3)
    kw.update(dict(global_pool="token", dynamic_img_size=True) if is_vit else dict(global_pool="avg"))
    return timm.create_model(arch, **kw)


class RaptorClassifier(nn.Module):
    def __init__(self, backbone, F_dim, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))

    def forward(self, x):
        B, K = x.shape[:2]
        h = self.norm(self.backbone(x.flatten(0, 1)).view(B, K, -1))
        a = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.clsW).sum(-1) + self.clsb


def find_checkpoints(roots, pattern):
    """Depth-limited on purpose: a recursive glob would walk the competition's DICOM tree."""
    found = []
    for root in roots:
        for depth in range(4):
            parts = [root] + ["*"] * depth + ["arms", "*", "weights", "*.pt"]
            for p in sorted(glob.glob(os.path.join(*parts))):
                if fnmatch.fnmatch(os.path.basename(p), pattern) and p not in found:
                    found.append(p)
    return found


def sha12(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------------------------
# Preprocessing: the corpus recipe (Raptor inference build_study), parameterized by the spec
# ---------------------------------------------------------------------------------------------
def order_and_meta(sdir):
    import pydicom
    recs, pss = [], []
    for f in glob.glob(sdir + "/*.dcm"):
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
            iop, ipp = getattr(h, "ImageOrientationPatient", None), getattr(h, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                n = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
                pos = float(np.dot(np.array(ipp, float), n))
            else:
                pos = float(getattr(h, "InstanceNumber", 0) or 0)
            ps = getattr(h, "PixelSpacing", None)
            ps = float(ps[0]) if ps is not None else 0.5
            pss.append(ps)
            recs.append((pos, f, ps))
        except Exception:
            recs.append((0.0, f, 0.5))
    recs.sort(key=lambda r: r[0])
    return [(f, ps) for _, f, ps in recs], (float(np.median(pss)) if pss else 0.5)


def read_px(f):
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut
    d = pydicom.dcmread(f)
    a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def mm_crop_resize(a, ps, crop_mm, img):
    import cv2
    h, w = a.shape
    cpx = min(int(round(crop_mm / max(ps, 1e-3))), min(h, w))
    y0, x0 = (h - cpx) // 2, (w - cpx) // 2
    return cv2.resize(a[y0:y0 + cpx, x0:x0 + cpx], (img, img), interpolation=cv2.INTER_AREA)


def pick_series(rows, plane, fluid, used):
    cands = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def build_study(sid, rows, series_dir, spec):
    slots, (span_lo, span_hi) = spec["slots"], spec["span"]
    img, crop_mm, (q_lo, q_hi) = int(spec["img"]), float(spec["crop_mm"]), spec.get("clip_pct", [2.0, 98.0])
    total = sum(int(s[2]) for s in slots)
    vol = np.zeros((total, img, img), np.uint8)
    idx, used = 0, set()
    for plane, fluid, k in slots:
        k = int(k)
        r = pick_series(rows, plane, int(fluid), used)
        if r is None:
            idx += k
            continue
        used.add(r["SeriesInstanceUID"])
        files, med_ps = order_and_meta(f"{series_dir}/{sid}/{r['SeriesInstanceUID']}")
        if not files:
            idx += k
            continue
        n = len(files)
        lo, hi = int(n * span_lo), int(n * span_hi) - 1
        hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        arrs, pss = [], []
        for p in picks:
            fp, ps = files[min(int(p), n - 1)]
            try:
                arrs.append(read_px(fp)); pss.append(ps)
            except Exception:
                arrs.append(None); pss.append(med_ps)
        valid = [x for x in arrs if x is not None]
        loq, hiq = np.percentile(np.concatenate([x.ravel() for x in valid]), [q_lo, q_hi]) if valid else (0.0, 1.0)
        for x, ps in zip(arrs, pss):
            if x is None:
                idx += 1
                continue
            x = np.clip((x - loq) / (hiq - loq + 1e-6), 0, 1)
            vol[idx] = (mm_crop_resize(x, ps if ps > 0 else med_ps, crop_mm, img) * 255).astype(np.uint8)
            idx += 1
    mask = (vol.reshape(total, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask


def eval_centers(mask, D, k):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi] or [max(1, min((lo + hi) // 2, D - 2))]
    return [cs[i] for i in np.linspace(0, len(cs) - 1, k).round().astype(int)]


class TestStudies(Dataset):
    """Builds one study per item on a CPU worker; returns uint8 windows (K,3,img,img)."""

    def __init__(self, ids, series, series_dir, spec, k):
        self.ids, self.series, self.series_dir, self.spec, self.k = ids, series, series_dir, spec, k

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]
        try:
            vol, mask = build_study(sid, self.series.get(sid, []), self.series_dir, self.spec)
            D = vol.shape[0]
            wins = np.stack([np.stack([vol[c - 1], vol[c], vol[c + 1]], 0)
                             for c in (max(1, min(c, D - 2)) for c in eval_centers(mask, D, self.k))], 0)
            return i, torch.from_numpy(wins), int(mask.sum()), ""
        except Exception as e:
            return i, torch.zeros(0, dtype=torch.uint8), 0, f"{type(e).__name__}: {e}"


def find_comp_root():
    cands = [os.environ.get("RSNA_COMP_ROOT", ""), "/kaggle/input/competitions/rsna-knee-abnormality-detection",
             "/kaggle/input/rsna-knee-abnormality-detection"]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "test.csv")):
            return c
    raise RuntimeError("competition data (test.csv) not found")


def main():
    budget = float(os.environ.get("RSNA_BUDGET_S", "1e9"))
    out_dir = os.environ.get("RSNA_OUT", "/kaggle/tmp/ours_leg_out")
    os.makedirs(out_dir, exist_ok=True)
    roots = [r for r in os.environ.get("RSNA_ARMS_DIRS", "/kaggle/input").split(os.pathsep) if r]
    paths = find_checkpoints(roots, os.environ.get("RSNA_CKPT_GLOB", "*_best.pt"))
    expected = os.environ.get("RSNA_EXPECTED_WEIGHTS")
    if not paths or (expected and int(expected) != len(paths)):
        log(f"checkpoints found {len(paths)} (expected {expected or '>0'}) under {roots}")
        sys.exit(2)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if dev == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16

    models, spec, res, k_eval, norm = [], None, None, None, None
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        this = (json.dumps(ck["corpus"], sort_keys=True), int(ck["res"]), int(ck["k_eval"]),
                tuple(ck["norm_mean"]), tuple(ck["norm_std"]))
        if spec is None:
            spec, res, k_eval, norm = ck["corpus"], this[1], this[2], this[3:]
            key0 = this
        elif this != key0:
            log(f"{os.path.basename(p)} was trained with a different input spec; refusing to mix")
            sys.exit(2)
        bb = build_backbone(ck["arch"])
        m = RaptorClassifier(bb, F_dim=bb.num_features)
        m.load_state_dict(ck["model"], strict=True)
        models.append(m.eval().to(dev))
        log(f"loaded {os.path.basename(p)} ({ck['arch']} @ {ck['res']}, fold {ck.get('fold')}, "
            f"OOF {ck.get('oof_auc')}, sha {sha12(p)})")
        del ck
    k_eval = int(os.environ.get("RSNA_K_EVAL", k_eval))
    mean = torch.tensor(norm[0], device=dev).view(1, 3, 1, 1)
    std = torch.tensor(norm[1], device=dev).view(1, 3, 1, 1)

    root = find_comp_root()
    series_dir = os.path.join(root, "test_series")
    if not os.path.isdir(series_dir):
        series_dir = os.path.join(root, "test_images")
    test_ids = pd.read_csv(os.path.join(root, "test.csv"), dtype={"StudyInstanceUID": str})["StudyInstanceUID"].tolist()
    ser = pd.read_csv(os.path.join(root, "test_series.csv"), dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    series = {k: v.to_dict("records") for k, v in ser.groupby("StudyInstanceUID")}
    cols = ["StudyInstanceUID"] + LAB
    if os.path.exists(os.path.join(root, "sample_submission.csv")):
        cols = list(pd.read_csv(os.path.join(root, "sample_submission.csv"), nrows=1).columns)
    N = len(test_ids)
    log(f"{len(models)} checkpoint(s) | {N} test studies | res {res} | k_eval {k_eval} | amp {amp_dtype} | {dev}")

    workers = int(os.environ.get("RSNA_WORKERS", "4"))
    loader = DataLoader(TestStudies(test_ids, series, series_dir, spec, k_eval), batch_size=None,
                        shuffle=False, num_workers=workers, prefetch_factor=2 if workers else None)
    probs = np.full((N, len(LAB)), np.nan, np.float32)
    failed = []
    with torch.no_grad():
        for n_done, (i, wins, n_slices, err) in enumerate(loader, 1):
            if time.time() - T0 > budget:
                log(f"budget exceeded after {n_done - 1}/{N} studies")
                sys.exit(3)
            if err or wins.numel() == 0:
                failed.append((test_ids[i], err or "empty"))
                continue
            try:
                x = wins.to(dev, non_blocking=True).float().div_(255.0)          # (K,3,img,img)
                if x.shape[-1] != res:
                    x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
                x = ((x - mean) / std).unsqueeze(0)
                ps = []
                for m in models:
                    try:
                        with torch.autocast("cuda", dtype=amp_dtype, enabled=dev == "cuda"):
                            ps.append(torch.sigmoid(m(x).float())[0].cpu().numpy())
                    except RuntimeError:                                     # fp32 retry, never drop
                        torch.cuda.empty_cache() if dev == "cuda" else None
                        ps.append(torch.sigmoid(m(x.float()).float())[0].cpu().numpy())
                probs[i] = np.mean(ps, axis=0)
            except Exception as e:
                failed.append((test_ids[i], f"{type(e).__name__}: {e}"))
            if n_done % 100 == 0 or n_done == N:
                log(f"{n_done}/{N} studies ({len(failed)} failed)")

    limit = max(3, int(0.02 * N))
    if N > 10 and len(failed) > limit:
        log(f"{len(failed)} failed studies > {limit}; not writing a leg the blend should not trust")
        sys.exit(4)
    bad = np.isnan(probs).any(axis=1)
    probs[bad] = 0.5
    ranks = pd.DataFrame(probs, columns=LAB).rank(method="average", pct=True)
    sub = pd.concat([pd.DataFrame({"StudyInstanceUID": test_ids}), ranks], axis=1)[cols]
    assert sub["StudyInstanceUID"].tolist() == test_ids and np.isfinite(sub[LAB].to_numpy()).all()
    sub.to_csv(os.path.join(out_dir, "submission.csv"), index=False)
    json.dump({"checkpoints": [{"file": os.path.basename(p), "sha12": sha12(p)} for p in paths],
               "n_test": N, "failed": failed[:50], "n_failed": len(failed), "res": res, "k_eval": k_eval,
               "spec": spec, "seconds": round(time.time() - T0, 1)},
              open(os.path.join(out_dir, "ours_leg_receipt.json"), "w"), indent=1)
    log(f"wrote {out_dir}/submission.csv ({N} rows, {len(failed)} failed -> 0.5)")


if __name__ == "__main__":
    main()
