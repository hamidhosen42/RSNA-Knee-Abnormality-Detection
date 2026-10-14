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


# ======================================================================= pseudo-label A/B kernel (ours)
import copy, glob, json, math, os, re, subprocess, sys, threading, time
import cv2
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ALPHA = 0.5
PROBE = False
EVAL_EPOCHS = {0, 4, 8, 11, 12, 13}
ARGS = dict(backbone="convnext_tiny", res=256, k=8, k_eval=12, epochs=1 if PROBE else 14, bs=8, micro_bs=2, lr=1e-4,
            lr_head=5e-4, wd=0.02, ema=0.998, workers=2, seed=0, limit=300 if PROBE else 0, fold=-1)
OUT = "/kaggle/working"
T0 = time.time()


def say(tag, msg, log=None):
    print(f"[{tag} {(time.time() - T0) / 60:.0f}m] {msg}", flush=True)
    if log:
        log.write(msg + "\n"); log.flush()


def find(name, root="/kaggle/input"):
    for pat in [f"{root}/*/", f"{root}/*/*/", f"{root}/*/*/*/", f"{root}/*/*/*/*/"]:
        for d in sorted(glob.glob(pat)):
            if os.path.exists(os.path.join(d, name)):
                return os.path.join(d, name)
    raise FileNotFoundError(name)


# ---------------------------------------------------------------- labels + folds (make_labels.py v0, as in E16)
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


# ---------------------------------------------------------------- PNG shard cache (as in E16)
def cache_index():
    metas = sorted(glob.glob("/kaggle/input/**/cache384_meta.csv", recursive=True))
    rows = []
    for m in metas:
        d = pd.read_csv(m)
        d["dir"] = os.path.join(os.path.dirname(m), "cache384")
        rows.append(d)
    meta = pd.concat(rows, ignore_index=True)
    ok = meta["n"].notna() & (meta["error"].isna() if "error" in meta else True)
    return meta[ok].drop_duplicates("series")


class PngVol:
    def __init__(self, path):
        z = np.load(path)
        self.data, self.off = z["data"], z["off"]

    def __len__(self):
        return len(self.off) - 1

    def __getitem__(self, idx):
        idx = np.atleast_1d(np.asarray(idx))
        return np.stack([cv2.imdecode(self.data[self.off[i]:self.off[i + 1]], cv2.IMREAD_UNCHANGED) for i in idx])


def data_table():
    meta = cache_index()
    paths = dict(zip(meta.series, [os.path.join(d, f"{s}.npz") for d, s in zip(meta.dir, meta.series)]))
    table = build_study_table(pd.read_csv(find("train_series.csv")), dict(zip(meta.series, meta.n)))
    return table, (lambda s: PngVol(paths[s]))


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


# ---------------------------------------------------------------- phase 1: OOF of the existing 5 folds
def ckpt_path(fold):
    pats = (["/kaggle/input/**/rsna-knee-gpjw-train-f34/cnxt_v0_fold%d.pt", "/kaggle/input/**/rsna-knee-gpjw-train-f34/**/cnxt_v0_fold%d.pt"]
            if fold >= 3 else ["/kaggle/input/**/rsna-knee-2-5d-convnext-reader/**/cnxt_v0_fold%d.pt",
                               "/kaggle/input/**/rsna-knee-2-5d-convnext-reader/cnxt_v0_fold%d.pt"])
    hits = sorted({p for pat in pats for p in glob.glob(pat % fold, recursive=True)})
    if not hits:   # fall back to a unique file of that name anywhere
        hits = sorted(glob.glob(f"/kaggle/input/**/cnxt_v0_fold{fold}.pt", recursive=True))
    assert len(hits) >= 1, f"no checkpoint for fold {fold}"
    return hits[0]


def oof_job(folds):
    tag = "oof" + "".join(map(str, folds))
    lab = build_labels()
    table, loader = data_table()
    for fold in folds:
        path = ckpt_path(fold)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = KneeNet(ck["args"]["backbone"].split(".")[0], pretrained=False)
        m.load_state_dict(ck["model"])
        m = m.cuda().to(memory_format=torch.channels_last).eval()
        va = lab[(lab.is_gold == 0) & (lab.fold == fold)]
        if PROBE:
            va = va.iloc[:60]
        dl = torch.utils.data.DataLoader(StudyDataset(va.index, table, None, None, ARGS["k_eval"], loader=loader),
                                         batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
        p = predict(m, dl, ck["args"]["res"])
        auc, _ = macro_auc((va[LABELS].values > 0.5).astype(int), p)
        pd.DataFrame(p, index=va.index, columns=LABELS).to_csv(f"{OUT}/oofck_fold{fold}.csv")
        say(tag, f"fold {fold}: {path} | {len(va)} studies | val AUC vs v0 {auc:.4f}")
        del m
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- phase 2: targets
def build_targets():
    lab = build_labels()
    weak = lab[lab.is_gold == 0]
    oof = pd.concat([pd.read_csv(f"{OUT}/oofck_fold{k}.csv", index_col=0) for k in range(5)])
    assert oof.index.is_unique
    if not PROBE:
        assert set(oof.index) == set(weak.index), (len(oof), len(weak))
    prob = 1 / (1 + np.exp(-oof[LABELS]))
    pl = lab.copy()
    idx = prob.index
    pl.loc[idx, LABELS] = (ALPHA * lab.loc[idx, LABELS].values + (1 - ALPHA) * prob.values).clip(0.01, 0.99)
    pl.to_csv(f"{OUT}/labels_pl.csv")
    y = (lab.loc[idx, LABELS].values > 0.5).astype(int)
    auc, each = macro_auc(y, prob.values)
    say("targets", f"5-fold OOF vs v0 macro {auc:.4f} [" + " ".join(f"{a:.3f}" for a in each) + "]")
    say("targets", "mean target v0 " + json.dumps(lab.loc[idx, LABELS].mean().round(3).to_dict()))
    say("targets", "mean target B  " + json.dumps(pl.loc[idx, LABELS].mean().round(3).to_dict()))
    say("targets", f"mean |B - v0| {np.abs(pl.loc[idx, LABELS].values - lab.loc[idx, LABELS].values).mean():.4f}")


# ---------------------------------------------------------------- phase 3: one full-data arm
ARM_CFG = {"C": dict(backbone="convnext_tiny", targets="0.5 v0 + 0.5 Mirror teacher", tw=0.5),
           "B2": dict(backbone="convnext_tiny.fb_in22k_ft_in1k", targets="B labels_pl.csv (0.5 v0 + 0.5 OOF)"),
           # round 2: C2 = C with seed 1 (replicate / ensemble member); C3 = C with the teacher weighted 0.75
           "C2": dict(backbone="convnext_tiny", targets="0.5 v0 + 0.5 Mirror teacher, seed 1", tw=0.5, seed=1),
           "C3": dict(backbone="convnext_tiny", targets="0.25 v0 + 0.75 Mirror teacher", tw=0.75),
           # round 3 (hosen42): the two Colab arms on Kaggle T4 - a new backbone with C3's targets, and C3 with seed 1
           "S75": dict(backbone="convnext_small", targets="0.25 v0 + 0.75 Mirror teacher", tw=0.75),
           "T75s1": dict(backbone="convnext_tiny", targets="0.25 v0 + 0.75 Mirror teacher, seed 1", tw=0.75, seed=1),
           # round 4 (final): students of the 0.956-level ensemble teacher, C3's target mix, two backbones
           "E75t": dict(backbone="convnext_tiny", targets="0.25 v0 + 0.75 ensemble teacher (Mirror384/M224/R7)", tw=0.75, teacher="ens"),
           "E75s": dict(backbone="convnext_small", targets="0.25 v0 + 0.75 ensemble teacher (Mirror384/M224/R7)", tw=0.75, teacher="ens")}

# ens = T2 from gate_ens_teacher.py: the 0.956 blend's rank form (Raptor 0.5 = 0.65 x Mirror384 + 0.35 x M224 with mirror,
# R7 0.5) on the train studies, mapped onto Mirror384's marginal, so only the teacher's ordering differs from C3's
TEACHER_FILES = {"mirror": "mirror_train_raw.csv", "ens": "ens_train_raw.csv"}


def build_targets_C(tw=0.5, teacher="mirror"):
    lab = build_labels()
    m = pd.read_csv(find(TEACHER_FILES[teacher]), dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[LABELS]
    m = m.reindex(lab.index.astype(str))
    if teacher == "ens":
        cover = float(m.notna().all(axis=1).mean())
        say("targets", f"ensemble teacher T2 covers {cover:.3f} of the train studies")
        assert cover > 0.9, f"ensemble teacher file covers only {cover:.3f}"
    m.index = lab.index
    ok = m.notna().all(axis=1) & ~(m == 0.5).all(axis=1)
    weak = lab.is_gold == 0
    idx = lab.index[weak & ok]
    y = (lab.loc[idx, LABELS].values > 0.5).astype(int)
    t_auc, t_each = macro_auc(y, m.loc[idx].values)
    say("targets", f"teacher scored {int(ok[weak].sum())}/{int(weak.sum())} weak studies | teacher vs v0 macro {t_auc:.4f} [" +
        " ".join(f"{x:.3f}" for x in t_each) + "]")
    gold = lab[lab.is_gold == 1]
    gok = ok[gold.index]
    if gok.sum() > 10:
        g_auc, _ = macro_auc(gold.loc[gok, [f"gold_{c}" for c in LABELS]].values, m.loc[gold.index[gok]].values)
        say("targets", f"teacher on gold-58 (diagnostic, teacher was selected on gold): {g_auc:.4f}")
    new = ((1 - tw) * lab.loc[idx, LABELS].values + tw * m.loc[idx].values).clip(0.01, 0.99)
    say("targets", f"mean |C - v0| {np.abs(new - lab.loc[idx, LABELS].values).mean():.4f}")
    lab.loc[idx, LABELS] = new
    lab.to_csv(f"{OUT}/labels_C_tw{tw}_{teacher}.csv")
    return lab


def train_arm(arm):
    a = dict(ARGS, arm=arm, **ARM_CFG[arm])
    torch.manual_seed(a["seed"]); np.random.seed(a["seed"])
    log = open(f"{OUT}/log_arm{arm}.txt", "a")
    say(arm, json.dumps(a), log)
    lab = build_targets_C(a["tw"], a.get("teacher", "mirror")) if "tw" in a else pd.read_csv(find("labels_pl.csv"), index_col=0)
    table, loader = data_table()
    gold = lab[lab.is_gold == 1]
    tr = lab[lab.is_gold == 0]
    if a["limit"]:
        tr = tr.iloc[:a["limit"]]
    say(arm, f"train {len(tr)}  gold {len(gold)} | mean target {tr[LABELS].values.mean():.4f}", log)
    dl_kw = dict(num_workers=a["workers"], pin_memory=True, persistent_workers=True)
    accum = a["bs"] // a["micro_bs"]
    train_dl = torch.utils.data.DataLoader(
        StudyDataset(tr.index, table, None, tr[LABELS].values.astype(np.float32), a["k"], train=True, loader=loader),
        batch_size=a["micro_bs"], shuffle=True, drop_last=True, **dl_kw)
    gold_dl = torch.utils.data.DataLoader(StudyDataset(gold.index, table, None, None, a["k_eval"], loader=loader),
                                          batch_size=a["bs"], shuffle=False, **dl_kw)
    gold_y = gold[[f"gold_{c}" for c in LABELS]].values

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
    step, gp = 0, None
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
                say(arm, f"  ep {ep} step {n}/{per_ep} loss {tot / n:.4f} {(time.time() - t0) / n:.2f}s/it "
                         f"mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")
        msg = f"ep {ep:2d} loss {tot / n:.4f} time {time.time() - t0:.0f}s"
        if ep in EVAL_EPOCHS or ep == a["epochs"] - 1:
            gp = predict(ema, gold_dl, a["res"])
            g_auc, g_each = macro_auc(gold_y, gp)
            msg += f" | gold {g_auc:.4f}"
        say(arm, msg, log)
        # checkpoint after every epoch, so a stopped session still leaves a usable (if less trained) model
        torch.save({"model": ema.state_dict(), "args": a, "epoch": ep}, f"{OUT}/cnxt_pl{arm}_full.pt.tmp")
        os.replace(f"{OUT}/cnxt_pl{arm}_full.pt.tmp", f"{OUT}/cnxt_pl{arm}_full.pt")
    pd.DataFrame(gp, index=gold.index, columns=LABELS).to_csv(f"{OUT}/gold_arm{arm}.csv")
    say(arm, "gold per class: " + " ".join(f"{c[:6]}={x:.3f}" for c, x in zip(LABELS, g_each)), log)
    say(arm, f"DONE arm {arm} in {(time.time() - T0) / 3600:.2f} h", log)


def script_path():
    cands = [globals().get("__file__"), "/kaggle/src/script.py", *sorted(glob.glob("/kaggle/src/*.py"))]
    return next((p for p in cands if p and os.path.isfile(p)), None)


def run_parallel(jobs):
    """jobs: list of (env_key, value); one subprocess per GPU, output streamed; returns exit codes."""
    src, procs = script_path(), []
    for i, (key, val) in enumerate(jobs):
        env = {**os.environ, key: val, "CUDA_VISIBLE_DEVICES": str(i % max(torch.cuda.device_count(), 1))}
        p = subprocess.Popen([sys.executable, "-u", src], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
        threading.Thread(target=lambda p=p: [print(l, end="", flush=True) for l in p.stdout], daemon=True).start()
        procs.append((val, p))
    codes = {v: p.wait() for v, p in procs}
    time.sleep(2)
    return codes


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    if os.environ.get("GPJW_OOF"):
        oof_job([int(f) for f in os.environ["GPJW_OOF"].split(",")])
        sys.exit(0)
    if os.environ.get("GPJW_ARM"):
        train_arm(os.environ["GPJW_ARM"])
        sys.exit(0)
    print(f"gpus {torch.cuda.device_count()} | script {script_path()} | probe {PROBE} | arms ['E75t', 'E75s']", flush=True)
    codes = run_parallel([("GPJW_ARM", x) for x in ['E75t', 'E75s']])
    print("arm exit codes:", codes, flush=True)
    if all(c != 0 for c in codes.values()):
        raise SystemExit(1)