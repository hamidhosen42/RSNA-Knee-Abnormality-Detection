"""Shared dataset and model code for training and inference."""
import math

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
PLANES = ["Sagittal", "Coronal", "Axial"]
# slot = plane index + 3 * (1 - fat_suppressed): fat-suppressed planes first
N_SLOTS = 6
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def slot_of(plane, fat_sup):
    return PLANES.index(plane) + 3 * (1 - int(fat_sup))


def build_study_table(series_df, n_slices):
    """series_df: StudyInstanceUID, SeriesInstanceUID, Fat_Suppression, Anatomical_Plane.
    n_slices: dict series -> cached slice count. Returns {study: [[series ids of slot 0], ...]}
    with each slot's candidates sorted by slice count (largest first)."""
    table = {}
    for st, sr, fs, pl in zip(series_df.StudyInstanceUID, series_df.SeriesInstanceUID,
                              series_df.Fat_Suppression, series_df.Anatomical_Plane):
        if n_slices.get(sr, 0) < 3:
            continue
        table.setdefault(st, [[] for _ in range(N_SLOTS)])[slot_of(pl, fs)].append(sr)
    for slots in table.values():
        for cands in slots:
            cands.sort(key=lambda s: -n_slices[s])
    return table


def window_centres(n, k, train, lo=0.04, hi=0.96):
    """k slice indices (centres of 3-slice windows) spread over the series."""
    a, b = max(1.0, lo * (n - 1)), min(n - 2.0, hi * (n - 1))
    if b < a:
        a = b = (n - 1) / 2
    if train:
        edges = np.linspace(a, b, k + 1)
        c = edges[:-1] + np.random.rand(k) * (edges[1:] - edges[:-1])
    else:
        c = np.linspace(a, b, k)
    return np.clip(np.round(c).astype(int), 1, max(1, n - 2))


class StudyDataset(torch.utils.data.Dataset):
    """One item = one study: uint8 windows [N_SLOTS, k, 3, H, W], slot mask, slice positions, target."""

    def __init__(self, studies, table, cache_dir, targets=None, k=8, train=False, loader=None):
        self.studies, self.table, self.cache_dir = list(studies), table, cache_dir
        self.targets, self.k, self.train = targets, k, train
        self.loader = loader or (lambda s: np.load(f"{self.cache_dir}/{s}.npy", mmap_mode="r"))

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        st = self.studies[i]
        slots = self.table.get(st, [[] for _ in range(N_SLOTS)])
        x, mask, pos = None, np.zeros(N_SLOTS, bool), np.zeros((N_SLOTS, self.k), np.float32)
        for s, cands in enumerate(slots):
            if not cands:
                continue
            sr = cands[np.random.randint(len(cands))] if self.train else cands[0]
            vol = self.loader(sr)
            n = len(vol)
            c = window_centres(n, self.k, self.train)
            win = np.stack([vol[np.clip([j - 1, j, j + 1], 0, n - 1)] for j in c])  # [k, 3, H, W]
            if x is None:
                x = np.zeros((N_SLOTS, self.k) + win.shape[1:], np.uint8)
            x[s], mask[s], pos[s] = win, True, c / max(n - 1, 1)
        if x is None:
            x = np.zeros((N_SLOTS, self.k, 3, 384, 384), np.uint8)
        y = self.targets[i] if self.targets is not None else np.zeros(len(LABELS), np.float32)
        return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(pos), torch.from_numpy(np.asarray(y, np.float32))


def make_views(x, res, train, group, fov=0.92):
    """x: uint8 [N, 3, H, W] with N = n_series * group. Crops a centred `fov` fraction of the cached
    field of view and resizes to res; in training adds rotation/scale/shift and gain/gamma jitter
    that is shared within a series."""
    n = x.shape[0]
    ns = n // group
    dev = x.device
    x = x.float().div_(255.0)
    if train:
        ang = (torch.rand(ns, device=dev) - 0.5) * 2 * math.radians(12)
        sc = fov * (1 + (torch.rand(ns, device=dev) - 0.5) * 0.24)
        tx = (torch.rand(ns, device=dev) - 0.5) * 0.12
        ty = (torch.rand(ns, device=dev) - 0.5) * 0.12
        gamma = torch.exp((torch.rand(ns, device=dev) - 0.5) * 0.5)
        gain = 1 + (torch.rand(ns, device=dev) - 0.5) * 0.2
    else:
        ang = torch.zeros(ns, device=dev)
        sc = torch.full((ns,), fov, device=dev)
        tx = ty = torch.zeros(ns, device=dev)
        gamma = gain = None
    cos, sin = torch.cos(ang) * sc, torch.sin(ang) * sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    theta = theta.repeat_interleave(group, 0)
    grid = F.affine_grid(theta, (n, 3, res, res), align_corners=False)
    if x.shape[-1] > 1.5 * res:  # avoid aliasing when shrinking a lot
        x = F.avg_pool2d(x, 2) if x.shape[-1] >= 2 * res else F.interpolate(x, scale_factor=0.75, mode="bilinear", antialias=True)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    if train:
        g = gamma.repeat_interleave(group).view(n, 1, 1, 1)
        x = x.clamp_min(1e-4).pow(g) * gain.repeat_interleave(group).view(n, 1, 1, 1)
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    return (x - mean) / std


class KneeNet(nn.Module):
    """2D backbone on 3-slice windows -> tokens tagged with series slot and slice position ->
    small transformer over all windows of the study -> one attention pooling per finding."""

    def __init__(self, backbone="convnext_tiny", pretrained=True, dim=384, depth=2, drop=0.2, drop_path=0.1):
        super().__init__()
        kw = {"drop_path_rate": drop_path} if drop_path else {}
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, **kw)
        self.proj = nn.Sequential(nn.Dropout(drop), nn.Linear(self.enc.num_features, dim), nn.LayerNorm(dim))
        self.slot_emb = nn.Embedding(N_SLOTS, dim)
        self.pos_emb = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(dim, 8, dim * 2, dropout=0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False) if depth else None
        self.att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, len(LABELS)))
        self.cls_w = nn.Parameter(torch.randn(len(LABELS), dim) * 0.02)
        self.cls_b = nn.Parameter(torch.zeros(len(LABELS)))

    def head(self, feat, mask, pos):
        """feat [B, S, K, D], mask [B, S] bool, pos [B, S, K] in 0..1 -> logits [B, 12]."""
        b, s, k, _ = feat.shape
        slot = torch.arange(s, device=feat.device).view(1, s, 1).expand(b, s, k)
        h = self.proj(feat) + self.slot_emb(slot) + self.pos_emb(pos.unsqueeze(-1).to(feat.dtype))
        h = h.view(b, s * k, -1)
        pad = ~mask.unsqueeze(-1).expand(b, s, k).reshape(b, s * k)
        pad = pad & ~pad.all(1, keepdim=True)  # a study with no series at all attends to everything
        if self.tf is not None:
            h = self.tf(h, src_key_padding_mask=pad)
        a = self.att(h).float().masked_fill(pad.unsqueeze(-1), -1e4).softmax(1)  # [B, T, C]
        pooled = torch.einsum("btc,btd->bcd", a.to(h.dtype), h)
        return (pooled * self.cls_w).sum(-1) + self.cls_b

    def forward(self, x, mask, pos, res=256, train=False):
        """x uint8 [B, S, K, 3, H, W]."""
        b, s, k = x.shape[:3]
        keep = mask.view(-1)  # run the backbone only on series that exist
        xv = x.view(b * s, k, *x.shape[3:])[keep].flatten(0, 1)
        f = self.enc(make_views(xv, res, train, k).contiguous(memory_format=torch.channels_last))
        feat = f.new_zeros(b * s, k, f.shape[-1])
        feat[keep] = f.view(-1, k, f.shape[-1])
        return self.head(feat.view(b, s, k, -1), mask, pos)


# ======================================================================= training kernel (ours)
import copy, glob, hashlib, json, math, os, re, subprocess, sys, threading, time
import cv2
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

FOLDS = [3, 4]
CKPT_PREFIX = "cnxt_small"
EVAL_EPOCHS = {0, 4, 8, 11, 12, 13}
PROBE = True
ARGS = dict(backbone="convnext_small", res=256, k=8, k_eval=12, epochs=1 if PROBE else 14, bs=8, micro_bs=2, lr=1e-4,
            lr_head=5e-4, wd=0.02, ema=0.998, workers=2, seed=0, limit=400 if PROBE else 0)
OUT = "/kaggle/working"
T0 = time.time()


def find(name, root="/kaggle/input"):
    for pat in [f"{root}/*/", f"{root}/*/*/", f"{root}/*/*/*/", f"{root}/*/*/*/*/"]:
        for d in sorted(glob.glob(pat)):
            if os.path.exists(os.path.join(d, name)):
                return os.path.join(d, name)
    raise FileNotFoundError(name)


# ---------------------------------------------------------------- labels + folds (make_labels.py v0)
def report_group(text):
    return re.sub(r"\W+", "", str(text).lower())


def build_labels():
    tr = pd.read_csv(find("train.csv"))
    ids = tr.StudyInstanceUID
    gold = tr.set_index("StudyInstanceUID")[LABELS]
    is_gold = gold.notna().all(axis=1)
    src = {"steven": pd.read_csv(find("llm_labels_v4_blend.csv")),
           "pilkwang": pd.read_csv(find("report_labels_v2.csv")),
           "lixin": pd.read_csv(find("labels_llm_gpt56sol.csv"))}
    q = pd.read_csv(find("qwen_knee_weak_labels.csv"))
    src["qwen"] = q[["StudyInstanceUID"] + [f"{c}__label" for c in LABELS]].set_axis(["StudyInstanceUID"] + LABELS, axis=1)
    src = {k: v.set_index("StudyInstanceUID")[LABELS].astype(float).reindex(ids) for k, v in src.items()}
    num = sum(v.fillna(0) for v in src.values())
    den = sum(v.notna().astype(float) for v in src.values())
    soft = (num / den).clip(0.01, 0.99)
    t = pd.DataFrame({k: [roc_auc_score(gold.loc[is_gold, c], v.loc[is_gold, c].fillna(0.5)) for c in LABELS]
                      for k, v in {**src, "consensus": soft}.items()}, index=LABELS)
    print("label sources vs gold-58 (macro):", t.mean().round(3).to_dict(), flush=True)
    out = soft.copy()
    out.insert(0, "is_gold", is_gold.astype(int).values)
    out["fold"] = -1
    weak = ~is_gold.values
    groups = tr.Report.map(report_group).values[weak]
    order = np.random.RandomState(0).permutation(weak.sum())
    folds = np.zeros(weak.sum(), int)
    for k, (_, va) in enumerate(GroupKFold(5).split(order, groups=groups[order])):
        folds[order[va]] = k
    out.loc[weak, "fold"] = folds
    for c in LABELS:
        out[f"gold_{c}"] = gold[c].values
    out.index = ids.values
    return out


# ---------------------------------------------------------------- PNG shard cache
def cache_index():
    metas = sorted(glob.glob("/kaggle/input/**/cache384_meta.csv", recursive=True))
    rows = []
    for m in metas:
        d = pd.read_csv(m)
        d["dir"] = os.path.join(os.path.dirname(m), "cache384")
        rows.append(d)
    meta = pd.concat(rows, ignore_index=True)
    ok = meta["n"].notna() & (meta["error"].isna() if "error" in meta else True)
    meta = meta[ok].drop_duplicates("series")
    print(f"cache: {len(metas)} shard metas, {len(meta)} series, {meta.study.nunique()} studies", flush=True)
    return meta


class PngVol:
    """Indexable like the author's mmap'd uint8 [n, 384, 384] array, decoding only the slices asked for."""

    def __init__(self, path):
        z = np.load(path)
        self.data, self.off = z["data"], z["off"]

    def __len__(self):
        return len(self.off) - 1

    def __getitem__(self, idx):
        idx = np.atleast_1d(np.asarray(idx))
        return np.stack([cv2.imdecode(self.data[self.off[i]:self.off[i + 1]], cv2.IMREAD_UNCHANGED) for i in idx])


def macro_auc(y, p):
    aucs = [roc_auc_score(y[:, i], p[:, i]) if 0 < y[:, i].sum() < len(y) else np.nan for i in range(y.shape[1])]
    return float(np.nanmean(aucs)), aucs


@torch.no_grad()
def predict(model, loader, res):
    model.eval()
    out = []
    for x, mask, pos, _ in loader:
        x, mask, pos = x.cuda(non_blocking=True), mask.cuda(), pos.cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(x, mask, pos, res=res).float().cpu())
    return torch.cat(out).numpy()


def train_fold(fold):
    a = dict(ARGS, fold=fold)
    torch.manual_seed(a["seed"]); np.random.seed(a["seed"])
    log = open(f"{OUT}/log_fold{fold}.txt", "a")

    def say(msg):
        print(f"[f{fold} {(time.time() - T0) / 60:.0f}m] {msg}", flush=True)
        log.write(msg + "\n"); log.flush()

    say(json.dumps(a))
    lab = build_labels()
    say("fold sizes " + json.dumps(lab.fold.value_counts().sort_index().to_dict()))
    meta = cache_index()
    paths = dict(zip(meta.series, [os.path.join(d, f"{s}.npz") for d, s in zip(meta.dir, meta.series)]))
    series = pd.read_csv(find("train_series.csv"))
    table = build_study_table(series, dict(zip(meta.series, meta.n)))
    loader = lambda s: PngVol(paths[s])
    gold = lab[lab.is_gold == 1]
    weak = lab[lab.is_gold == 0]
    tr, va = weak[weak.fold != fold], weak[weak.fold == fold]
    missing = [s for s in lab.index if s not in table]
    say(f"studies without any cached slot: {len(missing)}")
    if a["limit"]:
        tr = tr.iloc[:a["limit"]]
        va = va.iloc[:a["limit"] // 2]
    say(f"train {len(tr)}  val {len(va)}  gold {len(gold)}")
    dl_kw = dict(num_workers=a["workers"], pin_memory=True, persistent_workers=True)
    # bs 8 does not fit a 16 GB T4 (8 studies x 6 slots x 8 windows = 384 images): micro-batches of 4 with
    # gradient accumulation; no BatchNorm anywhere in the model, so the update equals the bs-8 one
    accum = a["bs"] // a["micro_bs"]
    train_dl = torch.utils.data.DataLoader(
        StudyDataset(tr.index, table, None, tr[LABELS].values.astype(np.float32), a["k"], train=True, loader=loader),
        batch_size=a["micro_bs"], shuffle=True, drop_last=True, **dl_kw)
    eval_sets = {"gold": gold, "val": va}
    eval_dl = {k: torch.utils.data.DataLoader(StudyDataset(v.index, table, None, None, a["k_eval"], loader=loader),
                                              batch_size=a["bs"], shuffle=False, **dl_kw) for k, v in eval_sets.items()}
    gold_y = gold[[f"gold_{c}" for c in LABELS]].values
    val_y = (va[LABELS].values > 0.5).astype(int)

    model = KneeNet(a["backbone"]).cuda().to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    enc_params = list(model.enc.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a["lr"]}, {"params": head_params, "lr": a["lr_head"]}],
                            weight_decay=a["wd"])
    per_ep = len(train_dl) // accum
    steps, warm = a["epochs"] * per_ep, per_ep
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
    scaler = torch.amp.GradScaler("cuda")
    step, preds = 0, None
    for ep in range(a["epochs"]):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for it, (x, mask, pos, y) in enumerate(train_dl):
            if it >= per_ep * accum:
                break
            x, mask, pos, y = x.cuda(non_blocking=True), mask.cuda(), pos.cuda(), y.cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                logit = model(x, mask, pos, res=a["res"], train=True)
            loss = F.binary_cross_entropy_with_logits(logit.float(), y)
            scaler.scale(loss / accum).backward()
            tot += loss.item() / accum
            if (it + 1) % accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            d = min(a["ema"], (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - d)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            n += 1
            if n % 100 == 0 or (PROBE and n % 10 == 0):
                say(f"  ep {ep} step {n}/{per_ep} loss {tot / n:.4f} {(time.time() - t0) / n:.2f}s/it "
                    f"mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")
        msg = f"ep {ep:2d} loss {tot / n:.4f} time {time.time() - t0:.0f}s"
        if ep not in EVAL_EPOCHS and ep != a["epochs"] - 1:
            say(msg)             # evaluation only logs; skipping some epochs leaves the final EMA model unchanged
            continue
        te = time.time()
        preds = {k: predict(ema, dl, a["res"]) for k, dl in eval_dl.items()}
        g_auc, g_each = macro_auc(gold_y, preds["gold"])
        v_auc, v_each = macro_auc(val_y, preds["val"])
        say(msg + f" | eval {time.time() - te:.0f}s | gold {g_auc:.4f} | val {v_auc:.4f} [" + " ".join(f"{x:.3f}" for x in v_each) + "]")
    torch.save({"model": ema.state_dict(), "args": a}, f"{OUT}/{CKPT_PREFIX}_fold{fold}.pt")
    rows = []
    for k, v in eval_sets.items():
        df = pd.DataFrame(preds[k], index=v.index, columns=LABELS)
        df.insert(0, "set", k)
        rows.append(df)
    pd.concat(rows).to_csv(f"{OUT}/oof_fold{fold}.csv")
    say("gold per class: " + " ".join(f"{c[:6]}={x:.3f}" for c, x in zip(LABELS, g_each)))
    say(f"DONE fold {fold} in {(time.time() - T0) / 3600:.2f} h")


def script_path():
    cands = [globals().get("__file__"), "/kaggle/src/script.py", *sorted(glob.glob("/kaggle/src/*.py"))]
    return next((p for p in cands if p and os.path.isfile(p)), None)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    if os.environ.get("GPJW_FOLD"):
        train_fold(int(os.environ["GPJW_FOLD"]))
        sys.exit(0)
    ngpu = torch.cuda.device_count()
    src = script_path()
    print(f"gpus {ngpu} | script {src} | folds {FOLDS} | probe {PROBE}", flush=True)
    procs = []
    for i, f in enumerate(FOLDS[:max(ngpu, 1)]):
        env = {**os.environ, "GPJW_FOLD": str(f), "CUDA_VISIBLE_DEVICES": str(i)}
        p = subprocess.Popen([sys.executable, "-u", src], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=lambda p=p: [print(l, end="", flush=True) for l in p.stdout], daemon=True).start()
        procs.append((f, p))
    codes = {f: p.wait() for f, p in procs}
    time.sleep(2)
    print("fold exit codes:", codes, flush=True)
    if all(c != 0 for c in codes.values()):
        raise SystemExit(1)
