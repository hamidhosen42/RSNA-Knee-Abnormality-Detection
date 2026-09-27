#!/usr/bin/env python3
"""Our independent CNN leg: training (Kaggle GPU script).

D4 went from the public 0.943 to 0.945/0.946 by rank-blending that anchor with an independently
trained CNN "student fleet" (EfficientNetV2-S at 384 px in D4 Lite) whose weights are private.
This script builds our own leg of that kind from public parts only:

  corpus  dreaddevelopment/knee-raptor-corpus + knee-raptor-corpus-ext (CC0):
          4,407 studies x 44 slices x 336 x 336 uint8
  labels  dreaddevelopment/rsna-knee-labels, labels_llm_soft.parquet (CC0): 4,349 soft-labelled
  recipe  dreaddevelopment "Knee MRI: training the twelve-finding model" v8: 2.5D three-slice
          windows and a per-finding attention-MIL head

Changes from that recipe, and why:
  1. Backbone tf_efficientnetv2_s.in21k_ft_in1k at 384 px (D4 Lite's leg architecture), with a
     backbone LR of 1e-4 instead of 3e-5, because this is a CNN fine-tune rather than CoAtNet.
  2. fp16 + GradScaler on GPUs without native bf16 (Kaggle T4/P100 have no bf16 conv engines).
  3. Inputs normalized with the backbone's own pretrained mean/std, stored in the checkpoint.
  4. Epochs are selected on the held-out fold (OOF) against soft labels, with the uncertain band
     masked. Gold-58 is reported every epoch but never used to select anything.
  5. One fold per GPU in separate processes (OURS_FOLDS, default "0,1"); DataParallel made this
     model 3x slower than a single T4. A wall-clock budget so a 12 h session ends cleanly.
  6. BatchNorm statistics are re-estimated after top-k weight averaging (EfficientNet has BN).
  7. Corpus rows are read with one pread into a reused buffer, and glibc is told to return big
     blocks to the OS; runs v1/v2 were OOM-killed by DataLoader workers growing ~5 MB per study.

Kaggle script kernels take no CLI arguments, so every default below is the intended run and can
be overridden with the OURS_* environment variables. Outputs go to OUT_DIR (/kaggle/working):
  arms/effv2s/weights/effv2s_f{fold}_{best,swa}.pt   checkpoints read by infer_kernel.py
  oof_effv2s_f{fold}_{best,swa}.npz                  held-out predictions + soft truth
  gold_effv2s_f{fold}_{best,swa}.npz                 gold-58 diagnostics
  train_effv2s_f{fold}.json                          per-epoch history and final metrics
"""
import argparse, ctypes, glob, hashlib, json, os, random, re, subprocess, sys, threading, time, unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
# How the public corpus was built (dataset description, v2). infer_kernel.py rebuilds test
# studies with exactly this spec, so it travels inside every checkpoint.
CORPUS_SPEC = {"slots": [["Sagittal", 1, 12], ["Sagittal", 0, 10], ["Coronal", 1, 8],
                         ["Coronal", 0, 6], ["Axial", -1, 8]],
               "span": [0.15, 0.85], "img": 336, "crop_mm": 140.0, "clip_pct": [2.0, 98.0],
               "source": "dreaddevelopment/knee-raptor-corpus v2 + knee-raptor-corpus-ext v1"}
POS, NEG = 0.7, 0.25   # soft-label AUC bands: >= POS positive, <= NEG negative, between = masked


def env(name, default, cast=str):
    v = os.environ.get(name)
    return default if v in (None, "") else cast(v)


FADVISE = env("OURS_FADVISE", 1, int) and hasattr(os, "posix_fadvise")   # Linux only


def tame_malloc(_worker_id=None):
    """Make glibc return large freed blocks to the OS.

    Kaggle diagnostics: DataLoader workers grew ~1.5-5 MB of anonymous memory per study read, and
    released it only when they exited. That is glibc raising its mmap threshold after the first
    big free, then keeping freed multi-MB blocks on the heap. A fixed threshold stops that.
    """
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.mallopt(-3, 1 << 20)       # M_MMAP_THRESHOLD: >= 1 MB via mmap, unmapped on free
        libc.mallopt(-1, 1 << 20)       # M_TRIM_THRESHOLD
    except OSError:
        pass                            # not glibc (local macOS tests)


# ---------------------------------------------------------------------------------------------
# Input discovery (Kaggle mounts every attached source read-only under /kaggle/input)
# ---------------------------------------------------------------------------------------------
_DICOM_TREES = {"train_series", "test_series", "train_images", "test_images"}


def find(root, *names, max_depth=5):
    """First file named one of `names`, never descending into the competition's DICOM trees."""
    base = root.rstrip(os.sep).count(os.sep)
    for d, dirs, fs in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in _DICOM_TREES and d.count(os.sep) - base < max_depth)
        for n in names:
            if n in fs:
                return os.path.join(d, n)
    return None


class NpyRows:
    """Rows of a C-order .npy file, one contiguous pread per row.

    Run 1 used np.memmap and loaded only ~2 studies/s: every 4 KB page fault is a round trip on
    Kaggle's network-backed /kaggle/input. A single 5 MB read per study avoids that. The file is
    opened lazily so each DataLoader worker gets its own descriptor.
    """

    def __init__(self, path):
        with open(path, "rb") as f:
            major, _ = np.lib.format.read_magic(f)
            read_header = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = read_header(f)
            self.offset = f.tell()
        assert not fortran, f"{path} is Fortran-ordered"
        self.path, self.shape, self.dtype, self.fd, self.buf = path, shape, np.dtype(dtype), None, None
        self.row_bytes = int(np.prod(shape[1:])) * self.dtype.itemsize

    def __getitem__(self, row):
        """Returns a view of a reused per-worker buffer: copy what you need before the next call."""
        if self.fd is None:
            self.fd = os.open(self.path, os.O_RDONLY)
            self.buf = np.empty(self.shape[1:], self.dtype)          # no 5 MB allocation per study
        pos = self.offset + row * self.row_bytes
        if hasattr(os, "preadv"):
            n = os.preadv(self.fd, [memoryview(self.buf).cast("B")], pos)
        else:                                                        # macOS (local tests)
            data = os.pread(self.fd, self.row_bytes, pos)
            n = len(data)
            memoryview(self.buf).cast("B")[:n] = data
        if n != self.row_bytes:
            raise IOError(f"short read {n}/{self.row_bytes} at row {row} of {self.path}")
        if FADVISE:                      # drop the NFS page cache we just filled
            os.posix_fadvise(self.fd, pos, self.row_bytes, os.POSIX_FADV_DONTNEED)
        return self.buf


class TwoPartVols:
    """The two corpus parts as one (4407, 44, 336, 336) array, part 1 rows first."""

    def __init__(self, a, b):
        self.a, self.b, self.n_a = a, b, a.shape[0]
        self.shape = (a.shape[0] + b.shape[0],) + tuple(a.shape[1:])

    def __getitem__(self, row):
        return self.a[row] if row < self.n_a else self.b[row - self.n_a]


def corpus_paths(root):
    p = {k: find(root, k) for k in ("all_vols.npy", "extra_vols.npy", "all_masks.npy", "extra_masks.npy")}
    if not all(p.values()):
        raise SystemExit("corpus not found: attach dreaddevelopment/knee-raptor-corpus and "
                         "dreaddevelopment/knee-raptor-corpus-ext")
    return p


def open_corpus(paths):
    vols = TwoPartVols(NpyRows(paths["all_vols.npy"]), NpyRows(paths["extra_vols.npy"]))
    return vols, np.concatenate([np.load(paths["all_masks.npy"]), np.load(paths["extra_masks.npy"])], axis=0)


def open_ids(root):
    a, b = find(root, "all_ids.npy"), find(root, "extra_ids.npy")
    if not (a and b):
        raise SystemExit("corpus ids not found: attach both corpus parts")
    return np.concatenate([np.load(a, allow_pickle=True).astype(str),
                           np.load(b, allow_pickle=True).astype(str)])


# ---------------------------------------------------------------------------------------------
# Frozen folds: an identical copy of ours_leg/folds.py (folds_v1). Recomputed here and checked
# against the checksum, so training can never silently run on a different split.
# ---------------------------------------------------------------------------------------------
K_FOLDS, FOLD_SEED = 5, 20260927
FOLDS_V1_SHA256 = "f4292c4a2dcaed3c43aa50ab5741f485c1f95ab064ca58fca9986ecc62e6a267"


def _missing(v):
    return v is None or v == "" or (isinstance(v, float) and np.isnan(v))


def _norm_report(t):
    t = "" if _missing(t) else str(t)   # pandas NaN and csv "" must normalize identically
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def make_folds(train_rows, soft_rows):
    gold = {str(r["StudyInstanceUID"]) for r in train_rows if not any(_missing(r.get(c)) for c in LAB)}
    report = {str(r["StudyInstanceUID"]): _norm_report(r.get("Report")) for r in train_rows}
    soft = {str(r["StudyInstanceUID"]): r for r in soft_rows}
    ids = sorted(u for u in soft if u not in gold)
    groups = defaultdict(list)
    for u in ids:
        groups[hashlib.sha1(report.get(u, u).encode()).hexdigest()].append(u)
    y = {u: [1 if float(soft[u][c]) >= 0.5 else 0 for c in LAB] for u in ids}
    gl = {g: [sum(y[u][j] for u in m) for j in range(len(LAB))] for g, m in groups.items()}
    tot = [sum(v[j] for v in gl.values()) for j in range(len(LAB))]
    need = [[tot[j] / K_FOLDS for j in range(len(LAB))] for _ in range(K_FOLDS)]
    cap = [len(ids) / K_FOLDS] * K_FOLDS
    rng = random.Random(FOLD_SEED)
    fold_of, left = {}, set(groups)
    while left:
        order = sorted(left)
        rem = [sum(gl[g][j] for g in order) for j in range(len(LAB))]
        cand = [j for j in range(len(LAB)) if rem[j] > 0]
        j = min(cand, key=lambda j: (rem[j], j)) if cand else None
        pool = [g for g in order if j is None or gl[g][j] > 0]
        rng.shuffle(pool)
        for g in sorted(pool, key=lambda g: -len(groups[g])):
            key = (lambda f: (cap[f], -f)) if j is None else (lambda f: (need[f][j], cap[f], -f))
            f = max(range(K_FOLDS), key=key)
            fold_of[g] = f
            left.discard(g)
            cap[f] -= len(groups[g])
            for jj in range(len(LAB)):
                need[f][jj] -= gl[g][jj]
    return {u: fold_of[g] for g, m in groups.items() for u in m}, sorted(gold)


def folds_sha256(folds):
    return hashlib.sha256(json.dumps(sorted(folds.items())).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------
# Data: per-study bag of three-slice windows (window sampling unchanged from the recipe).
# Workers return uint8 windows at the corpus resolution; scaling to [0,1], resizing to `res`, the
# intensity jitter and normalization happen on the GPU in `to_input` (same order as the recipe).
# Run 1 was OOM-killed holding float32 384 px batches (~170 MB each) in 8 worker processes.
# ---------------------------------------------------------------------------------------------
class StudyWindows(Dataset):
    def __init__(self, paths, ids, id2row, labels, k, train):
        self.paths, self.ids, self.id2row, self.labels = paths, ids, id2row, labels
        self.k, self.train = k, train
        self.vols = self.masks = None

    def __len__(self):
        return len(self.ids)

    def _centers(self, valid, D):
        lo, hi = int(valid.min()), int(valid.max())
        cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
        if not cs:
            cs = [max(1, min((lo + hi) // 2, D - 2))]
        if self.train:
            pool = cs * (self.k // len(cs) + 1)
            random.shuffle(pool)
            return pool[:self.k]
        idx = np.linspace(0, len(cs) - 1, self.k).round().astype(int)
        return [cs[i] for i in idx]

    def __getitem__(self, i):
        if self.vols is None:            # opened lazily, once per DataLoader worker
            self.vols, self.masks = open_corpus(self.paths)
        uid = self.ids[i]
        row = self.id2row[uid]
        vol = self.vols[row]
        D = vol.shape[0]
        valid = np.where(self.masks[row] > 0)[0]
        if len(valid) < 3:
            valid = np.arange(min(3, D))
        cs = [max(1, min(c, D - 2)) for c in self._centers(valid, D)]
        wins = np.stack([np.stack([vol[c - 1], vol[c], vol[c + 1]], 0) for c in cs], 0)   # (K,3,H,W) u8
        return torch.from_numpy(wins), torch.from_numpy(self.labels[uid])


def collate(batch):
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])


def to_input(xb, res, mean, std, dev, train):
    """uint8 (B,K,3,H,W) -> normalized float (B,K,3,res,res) on `dev`, as the recipe did on CPU."""
    x = xb.to(dev, non_blocking=True).float().div_(255.0)
    B, K = x.shape[:2]
    x = x.flatten(0, 1)
    if x.shape[-1] != res or x.shape[-2] != res:
        x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
    x = x.view(B, K, *x.shape[1:])
    if train:                            # no flips: laterality is signal; mild intensity jitter only
        x = (x * (1.0 + (torch.rand(B, 1, 1, 1, 1, device=x.device) - 0.5) * 0.20)).clamp_(0, 1)
    return (x - mean) / std


# ---------------------------------------------------------------------------------------------
# Model: timm backbone + per-finding attention-MIL head (verbatim from the recipe)
# ---------------------------------------------------------------------------------------------
def build_backbone(arch, pretrained):
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
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
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.channels_last = False       # training-speed option only; weights are layout-agnostic

    def forward(self, x):
        B, K = x.shape[:2]
        x = x.flatten(0, 1)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        h = self.norm(self.backbone(x).view(B, K, -1))
        a = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.clsW).sum(-1) + self.clsb


# ---------------------------------------------------------------------------------------------
# Metrics and helpers
# ---------------------------------------------------------------------------------------------
def soft_auc(truth, pred):
    """Macro AUC against soft labels, uncertain band masked per finding."""
    aucs = {}
    for j, c in enumerate(LAB):
        t = truth[:, j]
        keep = (t >= POS) | (t <= NEG)
        yb = (t[keep] >= POS).astype(int)
        if keep.sum() and 0 < yb.sum() < keep.sum():
            aucs[c] = float(roc_auc_score(yb, pred[keep, j]))
    return (float(np.mean(list(aucs.values()))) if aucs else float("nan")), aucs


def hard_auc(truth, pred):
    aucs = {}
    for j, c in enumerate(LAB):
        if len(set(truth[:, j].astype(int))) > 1:
            aucs[c] = float(roc_auc_score(truth[:, j].astype(int), pred[:, j]))
    return (float(np.mean(list(aucs.values()))) if aucs else float("nan")), aucs


@torch.no_grad()
def predict(model, loader, prep, dev, amp_dtype):
    model.eval()
    out = []
    for xb, _ in loader:
        with torch.autocast("cuda", dtype=amp_dtype, enabled=dev == "cuda"):
            out.append(torch.sigmoid(model(prep(xb, False)).float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(LAB)), np.float32)


@torch.no_grad()
def recalibrate_bn(core, loader, prep, dev, amp_dtype, max_batches):
    """Averaged weights need fresh BatchNorm statistics; single-device pass, no gradients."""
    bns = [m for m in core.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return 0
    momenta = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None            # cumulative average over the pass
    core.train()
    for i, (xb, _) in enumerate(loader):
        if i >= max_batches:
            break
        with torch.autocast("cuda", dtype=amp_dtype, enabled=dev == "cuda"):
            core(prep(xb, True))
    for m, mo in zip(bns, momenta):
        m.momentum = mo
    core.eval()
    return len(bns)


def host_mem():
    """Host RAM used/total plus this process's and its workers' RSS (runs v1/v2 were OOM-killed)."""
    try:
        info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
        kb = lambda k: int(info[k].split()[0])
        out = f"{(kb('MemTotal') - kb('MemAvailable')) / 2**20:.1f}/{kb('MemTotal') / 2**20:.1f} GB"
    except (OSError, KeyError, ValueError):
        return "n/a"
    try:
        import psutil
        me = psutil.Process()
        kids = sum(k.memory_info().rss for k in me.children(recursive=True)) / 2**30
        out += f" (main {me.memory_info().rss / 2**30:.1f} G, workers {kids:.1f} G)"
    except Exception:
        pass
    return out


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=env("OURS_ARCH", "tf_efficientnetv2_s.in21k_ft_in1k"))
    ap.add_argument("--arm", default=env("OURS_ARM", "effv2s"))
    ap.add_argument("--ckpt", default=env("OURS_CKPT", "timm"), help="'timm' pretrained, 'none' random init")
    ap.add_argument("--res", type=int, default=env("OURS_RES", 384, int))
    ap.add_argument("--k", type=int, default=env("OURS_K", 12, int))
    ap.add_argument("--k_eval", type=int, default=env("OURS_K_EVAL", 24, int))
    ap.add_argument("--epochs", type=int, default=env("OURS_EPOCHS", 16, int))
    ap.add_argument("--bs", type=int, default=env("OURS_BS", 8, int))
    ap.add_argument("--bb_lr", type=float, default=env("OURS_BB_LR", 1e-4, float))
    ap.add_argument("--head_lr", type=float, default=env("OURS_HEAD_LR", 1e-3, float))
    ap.add_argument("--wd", type=float, default=env("OURS_WD", 0.02, float))
    ap.add_argument("--workers", type=int, default=env("OURS_WORKERS", 4, int))
    ap.add_argument("--topk", type=int, default=env("OURS_TOPK", 3, int))
    ap.add_argument("--grad_ckpt", type=int, default=env("OURS_GRAD_CKPT", 1, int))
    ap.add_argument("--channels_last", type=int, default=env("OURS_CHANNELS_LAST", 0, int))
    ap.add_argument("--fold", type=int, default=env("OURS_FOLD", 0, int), help="-1 = fit on all folds")
    ap.add_argument("--seed", type=int, default=env("OURS_SEED", 42, int))
    ap.add_argument("--budget_h", type=float, default=env("OURS_BUDGET_H", 11.0, float))
    ap.add_argument("--bn_batches", type=int, default=env("OURS_BN_BATCHES", 100, int))
    ap.add_argument("--input_root", default=env("OURS_INPUT_ROOT", "/kaggle/input"))
    ap.add_argument("--out", default=env("OUT_DIR", "/kaggle/working"))
    ap.add_argument("--smoke", type=int, default=env("OURS_SMOKE", 0, int))
    ap.add_argument("--skip_fold_check", type=int, default=env("OURS_SKIP_FOLD_CHECK", 0, int),
                    help="local synthetic tests only")
    a = ap.parse_args()
    if a.smoke:
        a.epochs, a.bs, a.k, a.k_eval, a.workers, a.topk, a.bn_batches = 2, 2, 4, 6, 0, 2, 3
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    tame_malloc()
    T0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ngpu = torch.cuda.device_count() if dev == "cuda" else 0   # 1 per process; see run_folds()
    # T4/P100 report bf16 as "supported" through emulation, but have no bf16 conv engines.
    native_bf16 = dev == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8
    amp_dtype = torch.bfloat16 if native_bf16 else torch.float16
    tag = f"{a.arm}_f{a.fold}" if a.fold >= 0 else f"{a.arm}_full_s{a.seed}"
    wdir = os.path.join(a.out, "arms", a.arm, "weights")
    os.makedirs(wdir, exist_ok=True)
    print(f"device {dev} x{ngpu} | amp {amp_dtype} | arch {a.arch} @ {a.res} | k {a.k}/{a.k_eval} "
          f"| bs {a.bs} | epochs {a.epochs} | fold {a.fold} | tag {tag}", flush=True)

    # ---- ids, labels, folds -------------------------------------------------------------------
    ids = open_ids(a.input_root)
    id2row = {u: i for i, u in enumerate(ids)}
    paths = corpus_paths(a.input_root)
    tcsv = find(a.input_root, "train.csv")
    lab_path = find(a.input_root, "labels_llm_soft.parquet")
    if not tcsv or not lab_path:
        raise SystemExit("need train.csv (competition data) and labels_llm_soft.parquet "
                         "(dreaddevelopment/rsna-knee-labels)")
    train_df = pd.read_csv(tcsv, dtype={"StudyInstanceUID": str})
    soft_df = pd.read_parquet(lab_path)
    soft_df["StudyInstanceUID"] = soft_df["StudyInstanceUID"].astype(str)
    folds, gold_ids = make_folds(train_df.to_dict("records"), soft_df.to_dict("records"))
    digest = folds_sha256(folds)
    if digest != FOLDS_V1_SHA256:
        msg = f"fold split differs from folds_v1: {digest}"
        if not a.skip_fold_check:
            raise SystemExit(msg)
        print("WARNING (local test): " + msg, flush=True)
    soft = soft_df.set_index("StudyInstanceUID")
    gold = train_df.set_index("StudyInstanceUID").loc[gold_ids, LAB].astype(np.float32)
    in_corpus = set(ids)
    pool = [u for u in sorted(folds) if u in in_corpus]
    if a.fold >= 0:
        tr_ids = [u for u in pool if folds[u] != a.fold]
        oof_ids = [u for u in pool if folds[u] == a.fold]
    else:
        tr_ids, oof_ids = pool, []
    gold_ids = [u for u in gold_ids if u in in_corpus]
    if a.smoke:
        tr_ids, oof_ids, gold_ids = tr_ids[:16], oof_ids[:8], gold_ids[:8]
    labels = {u: np.array(soft.loc[u, LAB], dtype=np.float32) for u in tr_ids + oof_ids}
    labels.update({u: np.array(gold.loc[u], dtype=np.float32) for u in gold_ids})
    print(f"train {len(tr_ids)} | oof {len(oof_ids)} | gold {len(gold_ids)} | folds sha {digest[:12]}",
          flush=True)

    # ---- model --------------------------------------------------------------------------------
    bb = build_backbone(a.arch, pretrained=(a.ckpt in ("timm", "pretrained")))
    cfg = getattr(bb, "pretrained_cfg", {}) or {}
    mean, std = list(cfg.get("mean", (0.485, 0.456, 0.406))), list(cfg.get("std", (0.229, 0.224, 0.225)))
    if a.grad_ckpt:
        bb.set_grad_checkpointing(True)
    core = RaptorClassifier(bb, F_dim=bb.num_features).to(dev)
    if a.channels_last:
        core.channels_last = True
        core = core.to(memory_format=torch.channels_last)
    # No DataParallel: on Kaggle 2xT4 it made this model 3x slower than one GPU (10x in forward).
    # run_folds() instead trains one fold per GPU in separate processes.
    model = core
    print(f"backbone {a.arch}: {sum(p.numel() for p in bb.parameters()) / 1e6:.1f}M params, "
          f"features {bb.num_features}, norm mean {mean} std {std}", flush=True)

    mean_t = torch.tensor(mean, dtype=torch.float32, device=dev).view(1, 1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32, device=dev).view(1, 1, 3, 1, 1)

    def prep(xb, train):
        return to_input(xb, a.res, mean_t, std_t, dev, train)

    mk = dict(paths=paths, id2row=id2row, labels=labels)
    # Training workers are not persistent, so they exit before each evaluation (run 1 OOM).
    tl = DataLoader(StudyWindows(ids=tr_ids, k=a.k, train=True, **mk), batch_size=a.bs, shuffle=True,
                    num_workers=a.workers, drop_last=True, collate_fn=collate, pin_memory=False,
                    worker_init_fn=tame_malloc)
    ebs, ew = max(2, a.bs // 2), min(2, a.workers)

    def eval_loader(which):
        return DataLoader(StudyWindows(ids=which, k=a.k_eval, train=False, **mk), batch_size=ebs,
                          shuffle=False, num_workers=ew, collate_fn=collate, pin_memory=False,
                          worker_init_fn=tame_malloc)
    ol = eval_loader(oof_ids) if oof_ids else None
    gl = eval_loader(gold_ids) if gold_ids else None
    T_oof = np.stack([labels[u] for u in oof_ids]) if oof_ids else None
    T_gold = np.stack([labels[u] for u in gold_ids]) if gold_ids else None

    prev = np.clip(np.stack([labels[u] for u in tr_ids]).mean(0), 0.03, 0.7)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.clip((1 - prev) / prev, 1, 10),
                                                         dtype=torch.float32, device=dev))
    bb_params = [p for n, p in core.named_parameters() if n.startswith("backbone.")]
    head_params = [p for n, p in core.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": bb_params, "lr": a.bb_lr},
                             {"params": head_params, "lr": a.head_lr}], weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.bb_lr, a.head_lr],
                                                total_steps=max(len(tl) * a.epochs, 1), pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda" and amp_dtype == torch.float16))

    def ckpt(extra):
        return {"model": {k: v.detach().cpu().clone() for k, v in core.state_dict().items()},
                "arch": a.arch, "res": a.res, "lab": LAB, "norm_mean": mean, "norm_std": std,
                "k_eval": a.k_eval, "corpus": CORPUS_SPEC, "fold": a.fold, "folds_sha256": digest,
                "recipe": {k: v for k, v in vars(a).items() if k not in ("input_root", "out")}, **extra}

    def evaluate():
        r = {}
        if ol is not None:
            P = predict(model, ol, prep, dev, amp_dtype)
            r["oof_auc"], r["oof_aucs"], r["P_oof"] = (*soft_auc(T_oof, P), P)
        if gl is not None:
            P = predict(model, gl, prep, dev, amp_dtype)
            r["gold_auc"], r["gold_aucs"], r["P_gold"] = (*hard_auc(T_gold, P), P)
        return r

    def save_preds(kind, r):
        if "P_oof" in r:
            np.savez(os.path.join(a.out, f"oof_{tag}_{kind}.npz"), pred=r["P_oof"], truth=T_oof,
                     ids=np.array(oof_ids), fold=a.fold)
        if "P_gold" in r:
            np.savez(os.path.join(a.out, f"gold_{tag}_{kind}.npz"), pred=r["P_gold"], truth=T_gold,
                     ids=np.array(gold_ids))

    # Selection key: OOF macro AUC when a fold is held out; otherwise the latest epoch (full fit).
    def score(r, ep):
        return r["oof_auc"] if "oof_auc" in r else float(ep)

    hist, topk, best, ep_times = [], [], None, []
    for ep in range(a.epochs):
        if ep_times and (time.time() - T0) + 1.15 * np.mean(ep_times) + 1200 > a.budget_h * 3600:
            print(f"budget stop before epoch {ep} ({(time.time() - T0) / 3600:.2f} h used)", flush=True)
            break
        t_ep = time.time()
        model.train()
        tot, n, t_wait, t_mark = 0.0, 0, 0.0, time.time()
        for it, (xb, y) in enumerate(tl):
            t_wait += time.time() - t_mark           # time blocked on the loader
            x, y = prep(xb, True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=dev == "cuda"):
                logits = model(x)
            loss = lossf(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(core.parameters(), 3.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot, n = tot + loss.item(), n + 1         # .item() syncs, so step time includes the GPU
            if it % 50 == 0:
                el = max(time.time() - t_ep, 1e-6)
                print(f"  ep{ep} it{it}/{len(tl)} loss {tot / n:.4f} | {(it + 1) * a.bs / el:.1f} studies/s "
                      f"| waiting on data {100 * t_wait / el:.0f}% | RAM {host_mem()}", flush=True)
            t_mark = time.time()
        t_eval = time.time()
        r = evaluate()
        print(f"  ep{ep} eval {(time.time() - t_eval) / 60:.1f} min | RAM {host_mem()}", flush=True)
        ep_times.append(time.time() - t_ep)
        rec = {"ep": ep, "loss": tot / max(n, 1), "sec": round(ep_times[-1]),
               **{k: v for k, v in r.items() if not k.startswith("P_")}}
        hist.append(rec)
        s = score(r, ep)
        if best is None or s > best["score"]:
            best = {"score": s, "ep": ep}
            atomic_save(ckpt({"epoch": ep, "kind": "best", **rec}), os.path.join(wdir, f"{tag}_best.pt"))
            save_preds("best", r)
        topk.append({"score": s, "ep": ep,
                     "sd": {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}})
        topk = sorted(topk, key=lambda t: -t["score"])[:max(1, a.topk)]
        print(f"ep{ep} loss {rec['loss']:.4f} | OOF {r.get('oof_auc', float('nan')):.4f} | "
              f"gold(diag) {r.get('gold_auc', float('nan')):.4f} | best ep{best['ep']} | "
              f"{ep_times[-1] / 60:.1f} min | {(time.time() - T0) / 3600:.2f} h", flush=True)
        json.dump({"tag": tag, "hist": hist, "best": best, "partial": True},
                  open(os.path.join(a.out, f"train_{tag}.json"), "w"), indent=1)

    summary = {"tag": tag, "arch": a.arch, "res": a.res, "fold": a.fold, "folds_sha256": digest,
               "n_train": len(tr_ids), "n_oof": len(oof_ids), "n_gold": len(gold_ids),
               "hist": hist, "best": best, "hours": None, "partial": False}
    # ---- top-k weight average + fresh BN statistics ------------------------------------------
    if len(topk) > 1:
        sds = [t["sd"] for t in topk]
        avg = {k: (sum(sd[k].double() for sd in sds) / len(sds)).to(sds[0][k].dtype)
               if sds[0][k].is_floating_point() else sds[0][k].clone() for k in sds[0]}
        core.load_state_dict(avg)
        bn_loader = DataLoader(StudyWindows(ids=tr_ids, k=a.k, train=True, **mk), batch_size=a.bs,
                               shuffle=True, num_workers=ew, collate_fn=collate, worker_init_fn=tame_malloc)
        n_bn = recalibrate_bn(core, bn_loader, prep, dev, amp_dtype, a.bn_batches)
        r = evaluate()
        eps = [t["ep"] for t in topk]
        rec = {k: v for k, v in r.items() if not k.startswith("P_")}
        atomic_save(ckpt({"epoch": eps, "kind": "swa", "bn_layers_recalibrated": n_bn, **rec}),
                    os.path.join(wdir, f"{tag}_swa.pt"))
        save_preds("swa", r)
        summary["swa"] = {"epochs": eps, "bn_layers": n_bn, **rec}
        print(f"SWA over epochs {eps}: OOF {r.get('oof_auc', float('nan')):.4f} | "
              f"gold(diag) {r.get('gold_auc', float('nan')):.4f}", flush=True)
    summary["hours"] = round((time.time() - T0) / 3600, 3)
    json.dump(summary, open(os.path.join(a.out, f"train_{tag}.json"), "w"), indent=1)
    marker = os.path.join(a.out, "arms", a.arm, "ours_leg_arm.json")
    json.dump({"arm": a.arm, "arch": a.arch, "res": a.res, "k_eval": a.k_eval, "corpus": CORPUS_SPEC,
               "marker": "ours_leg_arm"}, open(marker + f".{os.getpid()}", "w"))
    os.replace(marker + f".{os.getpid()}", marker)       # two fold processes may write it at once
    b = hist[best["ep"]] if best and hist else {}
    print(f"\nDONE {tag} in {summary['hours']} h | best ep{best['ep'] if best else '-'} "
          f"OOF {b.get('oof_auc', float('nan')):.4f} gold(diag) {b.get('gold_auc', float('nan')):.4f}", flush=True)
    for c in LAB:
        print(f"   {c:17s} OOF {b.get('oof_aucs', {}).get(c, float('nan')):.3f} | "
              f"gold {b.get('gold_aucs', {}).get(c, float('nan')):.3f}", flush=True)


def _script_path():
    cands = [globals().get("__file__"), sys.argv[0] if sys.argv and sys.argv[0].endswith(".py") else None,
             "/kaggle/src/script.py", *sorted(glob.glob("/kaggle/src/*.py"))]
    return next((p for p in cands if p and os.path.isfile(p)), None)


def run_folds():
    """Train the folds in OURS_FOLDS, one per GPU, as separate processes (no DataParallel)."""
    folds = [int(f) for f in env("OURS_FOLDS", "0,1").split(",") if f.strip()]
    if os.environ.get("OURS_CHILD") or os.environ.get("OURS_SMOKE") == "1":
        return main()                                   # a child, or a local smoke test
    ngpu = torch.cuda.device_count()
    src = _script_path()
    print(f"folds {folds} | gpus {ngpu} | script {src}", flush=True)
    if len(folds) == 1 or ngpu < 2 or src is None:
        os.environ["OURS_FOLD"] = str(folds[0])         # one fold only: a second one would overrun
        return main()                                   # the 12 h session in sequence
    procs, relays = [], []
    for i, f in enumerate(folds[:ngpu]):
        child_env = {**os.environ, "OURS_CHILD": "1", "OURS_FOLD": str(f), "CUDA_VISIBLE_DEVICES": str(i),
                     "OURS_WORKERS": str(max(1, (os.cpu_count() or 4) // ngpu))}
        p = subprocess.Popen([sys.executable, "-u", src], env=child_env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)

        def relay(p=p, f=f):
            for line in p.stdout:
                print(f"[f{f}] {line}", end="", flush=True)
        t = threading.Thread(target=relay, daemon=True)
        t.start()
        procs.append((f, p)); relays.append(t)
    codes = {f: p.wait() for f, p in procs}
    for t in relays:
        t.join(timeout=10)
    print(f"fold exit codes: {codes}", flush=True)
    if all(c != 0 for c in codes.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    run_folds()
