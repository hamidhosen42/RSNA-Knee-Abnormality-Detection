#!/usr/bin/env python3
"""fx_train.py - Experiment 2 (fast, Scott-Willis style): the Experiment 1 trainer (nx_train.py), used with small CNNs.

nx_train.py docstring follows.

nx_train.py - Experiment 1: Raptor-style CoAtNet trained on consensus labels.

Derived from the public recipe (dreaddevelopment/knee-mri-training-the-twelve-finding-model). The model
class, checkpoint format and window sampling are kept identical to it so the 0.943 inference notebook's
`load_model` / `_ke_prepare_windows` / `eval_windows` can run the result unchanged. Deliberate changes:

  * ImageNet normalisation (the 0.943 notebook feeds every Raptor view ImageNet-normalised input)
  * fp16 + GradScaler on GPUs without bf16 (Kaggle T4)
  * fixed epochs + weight average of the last N epochs; gold58 is never used to select anything
  * resize / normalise on the GPU (Kaggle GPU machines have 4 CPU cores)
  * --mode cv writes out-of-fold predictions for self-distillation; --mode evalref scores a public checkpoint

Modes
  cv       train one model per fold in --folds (comma list) of --folds_json; write OOF + gold predictions
  final    train on every labelled study with seed --seed; write gold predictions
  evalref  score an existing checkpoint (--ckpt) on gold58 with the same windows
"""
import os, sys, time, json, random, argparse
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
_SKIP = ("train_series", "test_series", "train_images", "test_images")
INPUT_ROOT = os.environ.get("NX_INPUT_ROOT", "/kaggle/input")   # override only for local smoke tests


_FOUND = {}


def find(name, root=None):
    """First file called `name` under root (cached: DataLoader workers inherit the cache)."""
    root = root or INPUT_ROOT
    key = (name, root)
    if key not in _FOUND:
        _FOUND[key] = None
        for d, dirs, fs in os.walk(root):
            dirs[:] = [x for x in dirs if x not in _SKIP]
            if name in fs:
                _FOUND[key] = os.path.join(d, name)
                break
    return _FOUND[key]


# ------------------------------------------------------------------ corpus
class TwoPart:
    """The two public corpus parts as one (4407, 44, 336, 336) uint8 array, both memory-mapped."""
    def __init__(self, a, b):
        self.a, self.b, self.n_a = a, b, a.shape[0]
        self.shape = (a.shape[0] + b.shape[0],) + tuple(a.shape[1:])

    def __getitem__(self, row):
        return self.a[row] if row < self.n_a else self.b[row - self.n_a]


def open_corpus(root=None):
    p = {n: find(n, root) for n in ("all_vols.npy", "extra_vols.npy", "all_masks.npy",
                                    "extra_masks.npy", "all_ids.npy", "extra_ids.npy")}
    if not all(p.values()):
        raise SystemExit("attach dreaddevelopment/knee-raptor-corpus AND dreaddevelopment/knee-raptor-corpus-ext; "
                         f"missing: {[k for k, v in p.items() if not v]}")
    vols = TwoPart(np.load(p["all_vols.npy"], mmap_mode="r"), np.load(p["extra_vols.npy"], mmap_mode="r"))
    masks = np.concatenate([np.load(p["all_masks.npy"]), np.load(p["extra_masks.npy"])])
    ids = np.concatenate([np.load(p["all_ids.npy"], allow_pickle=True).astype(str),
                          np.load(p["extra_ids.npy"], allow_pickle=True).astype(str)])
    if not (len(ids) == vols.shape[0] == len(masks)):
        raise RuntimeError(f"corpus parts disagree: ids {len(ids)} vols {vols.shape[0]} masks {len(masks)}")
    return vols, masks, ids


def eval_centers(valid, depth, k):
    """Deterministic evaluation windows: k centres spread evenly over the valid slices (== 0.943 _eval_centers).
    Shared by training-time evaluation and the fast submission notebook so both pick identical windows."""
    lo, hi = int(valid.min()), int(valid.max())
    cs = list(range(lo + 1, hi))
    if not cs:
        cs = [max(1, min((lo + hi) // 2, depth - 2))]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [cs[i] for i in idx]


class StudyWindows(Dataset):
    """k three-slice windows per study as uint8 (k,3,H,W); resize/normalise happen on the GPU."""
    def __init__(self, ids, id2row, labels, k, train, root=None):
        self.ids, self.id2row, self.labels, self.k, self.train, self.root = ids, id2row, labels, k, train, root
        self.vols = self.masks = None

    def __len__(self):
        return len(self.ids)

    def _centers(self, valid, depth):
        lo, hi = int(valid.min()), int(valid.max())
        cs = list(range(lo + 1, hi))                      # both neighbours inside [lo, hi]
        if not cs:
            cs = [max(1, min((lo + hi) // 2, depth - 2))]
        if self.train:                                    # public recipe: random windows each epoch
            pool = cs * (self.k // len(cs) + 1)
            random.shuffle(pool)
            return sorted(pool[:self.k])
        return eval_centers(valid, depth, self.k)

    def __getitem__(self, i):
        if self.vols is None:
            self.vols, self.masks, _ = open_corpus(self.root)
        uid = self.ids[i]
        row = self.id2row[uid]
        vol = self.vols[row]
        depth = vol.shape[0]
        valid = np.where(self.masks[row] > 0)[0]
        if len(valid) < 3:
            valid = np.arange(min(3, depth))
        cs = np.clip(np.asarray(self._centers(valid, depth)), 1, depth - 2)
        tri = np.stack([cs - 1, cs, cs + 1], 1)
        uniq, inv = np.unique(tri.reshape(-1), return_inverse=True)
        planes = np.asarray(vol[uniq])                    # only the needed slices are read
        x = torch.from_numpy(planes[inv.reshape(-1, 3)])  # (k,3,H,W) uint8
        y = torch.from_numpy(self.labels[uid]) if self.labels is not None else torch.zeros(len(LAB))
        return x, y


def collate(batch):
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])


_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def to_input(xu8, res, train):
    """uint8 (B,k,3,H,W) on GPU -> normalised float (B,k,3,res,res); same math as the 0.943 eval_windows."""
    B, K, C, H, W = xu8.shape
    x = xu8.float().div_(255.0)
    if (H, W) != (res, res):
        x = F.interpolate(x.view(B * K * C, 1, H, W), size=(res, res), mode="bilinear",
                          align_corners=False).view(B, K, C, res, res)
    if train:                                             # the recipe's only augmentation: mild gain
        g = 1.0 + (torch.rand(B, 1, 1, 1, 1, device=x.device) - 0.5) * 0.20
        x = (x * g).clamp_(0, 1)
    return (x - _MEAN.to(x.device)) / _STD.to(x.device)


# ------------------------------------------------------------------ model (identical to the public arms)
def build_backbone(arch, pretrained=False):
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    import timm
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    kw.update(global_pool="token", dynamic_img_size=True) if is_vit else kw.update(global_pool="avg")
    return timm.create_model(arch, **kw)


class RaptorClassifier(nn.Module):
    """Backbone + per-finding attention-MIL head over windows."""
    def __init__(self, backbone, F_dim=768, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def encode(self, x):
        B, K = x.shape[:2]
        return self.backbone(x.flatten(0, 1)).view(B, K, -1)

    def head(self, feats):
        h = self.norm(feats)
        a = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.clsW).sum(-1) + self.clsb

    def forward(self, x):
        return self.head(self.encode(x))


# ------------------------------------------------------------------ metrics
def fast_auc(y, p):
    y = np.asarray(y) > 0.5
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    ps = np.asarray(p)[order]                              # average ranks for ties
    i = 0
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return (ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def macro_auc(Y, P):
    a = {LAB[j]: fast_auc(Y[:, j], P[:, j]) for j in range(Y.shape[1])}
    a = {k: float(v) for k, v in a.items() if np.isfinite(v)}
    return (float(np.mean(list(a.values()))) if a else float("nan")), a


# ------------------------------------------------------------------ train / predict
def amp_dtype():
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@torch.no_grad()
def predict(model, ids, id2row, a, dev):
    if not ids:
        return np.zeros((0, len(LAB)), np.float32)
    ds = StudyWindows(ids, id2row, None, a.k_eval, train=False)
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=a.workers, collate_fn=collate)
    model.eval(); out = []
    for x, _ in dl:
        x = to_input(x.to(dev, non_blocking=True), a.res, train=False)
        with torch.autocast("cuda", dtype=amp_dtype(), enabled=dev.type == "cuda"):
            out.append(torch.sigmoid(model(x).float()).cpu().numpy())
    return np.concatenate(out)


def train_one(run_name, train_ids, oof_ids, gold_ids, labels, gold_Y, id2row, a=None, dev=None, t_start=None):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    Ytr = np.stack([labels[u] for u in train_ids])
    prev = np.clip(Ytr.mean(0), 0.03, 0.7)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.clip((1 - prev) / prev, 1, 10),
                                                          dtype=torch.float32, device=dev))
    ds = StudyWindows(train_ids, id2row, labels, a.k, train=True)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                    collate_fn=collate, pin_memory=True, persistent_workers=a.workers > 0,
                    prefetch_factor=4 if a.workers > 0 else None)
    bb = build_backbone(a.arch, pretrained=not (a.no_pretrained or a.init_ckpt or a.resume_state))
    if a.grad_ckpt:
        try:
            bb.set_grad_checkpointing(True)
        except Exception as e:
            print(f"[{run_name}] grad checkpointing unsupported: {e}", flush=True)
    model = RaptorClassifier(bb, F_dim=bb.num_features).to(dev)
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    body = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": body, "lr": a.bb_lr}, {"params": head, "lr": a.head_lr}],
                            weight_decay=a.wd)
    steps = max(len(dl) * a.epochs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.bb_lr, a.head_lr], total_steps=steps,
                                                pct_start=a.pct_start)
    dt = amp_dtype()
    scaler = torch.amp.GradScaler("cuda", enabled=(dt == torch.float16 and dev.type == "cuda"))
    print(f"[{run_name}] train {len(train_ids)} | oof {len(oof_ids)} | gold {len(gold_ids)} | {a.arch}@{a.res} "
          f"k {a.k}/{a.k_eval} bs {a.bs} ep {a.epochs} amp {dt} | steps/epoch {len(dl)}", flush=True)
    avg, n_avg, hist, stopped = None, 0, [], False
    first_ep = max(0, a.epochs - a.avg_last)
    start_ep = 0
    if a.resume_state:                                     # exact resume: weights + optimizer + schedule + averages
        st = torch.load(a.resume_state, map_location="cpu", weights_only=False)
        if (st["arch"], st["epochs"], st["steps_per_epoch"]) != (a.arch, a.epochs, len(dl)):
            raise SystemExit(f"[{run_name}] resume state is for {st['arch']} / {st['epochs']} epochs / "
                             f"{st['steps_per_epoch']} steps, this run is {a.arch} / {a.epochs} / {len(dl)}")
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        avg, n_avg, hist, start_ep = st["avg"], st["n_avg"], st["hist"], st["epoch"] + 1
        print(f"[{run_name}] RESUMED from {os.path.basename(a.resume_state)}: continuing at epoch {start_ep} of {a.epochs}",
              flush=True)
    elif a.init_ckpt:                                      # warm start: previous weights, fresh schedule
        ck0 = torch.load(a.init_ckpt, map_location="cpu", weights_only=False)
        if ck0.get("arch") != a.arch:
            raise SystemExit(f"[{run_name}] warm-start checkpoint is {ck0.get('arch')}, this run is {a.arch}")
        model.load_state_dict(ck0["model"], strict=True)
        print(f"[{run_name}] WARM START from {os.path.basename(a.init_ckpt)} (its gold58 {ck0.get('gold_auc', float('nan')):.4f})",
              flush=True)
        del ck0
    for ep in range(start_ep, a.epochs):
        model.train(); tot = 0.0; t_ep = time.time(); step = -1
        for step, (x, y) in enumerate(dl):
            x = to_input(x.to(dev, non_blocking=True), a.res, train=True)
            y = y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dt, enabled=dev.type == "cuda"):
                logits = model(x)
            loss = lossf(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
            if ep == start_ep and step == 4:
                t_warm = time.time()                       # skip warm-up steps (cudnn autotune, first batches)
            if ep == start_ep and step == min(40, len(dl) - 1) and step > 4:
                per = (time.time() - t_warm) / (step - 4)
                print(f"[{run_name}] timing: {per:.2f}s/step (measured over {step - 4} steps) -> about "
                      f"{per * len(dl) / 60:.1f} min/epoch, {per * steps / 3600:.2f} h of training for this run", flush=True)
            if time.time() - t_start > a.time_limit_h * 3600:
                stopped = True
                break
        if ep >= first_ep or stopped:                      # running mean of the last epochs' weights
            sd = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}
            if avg is None:
                avg = sd; n_avg = 1
            else:
                n_avg += 1
                for k in avg:
                    if avg[k].is_floating_point():
                        avg[k] += (sd[k] - avg[k]) / n_avg
                    else:
                        avg[k] = sd[k]
        g_auc = float("nan")
        if a.monitor_gold and gold_ids:                    # monitoring only, never used for selection
            g_auc, _ = macro_auc(gold_Y, predict(model, gold_ids, id2row, a, dev))
        hist.append({"ep": ep, "loss": tot / max(1, step + 1), "gold_monitor": g_auc,
                     "min": (time.time() - t_ep) / 60})
        print(f"[{run_name}] ep{ep} loss {hist[-1]['loss']:.4f} gold(monitor) {g_auc:.4f} "
              f"{hist[-1]['min']:.1f} min", flush=True)
        if a.save_state and not stopped:                  # only completed epochs: a resume restarts cleanly
            state = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "opt": opt.state_dict(),
                     "sched": sched.state_dict(), "scaler": scaler.state_dict(), "avg": avg, "n_avg": n_avg,
                     "hist": hist, "epoch": ep, "epochs": a.epochs, "steps_per_epoch": len(dl), "arch": a.arch,
                     "res": a.res}
            tmp = os.path.join(a.out, f"{run_name}_state.pt.tmp")
            torch.save(state, tmp)
            os.replace(tmp, os.path.join(a.out, f"{run_name}_state.pt"))
            del state
        if stopped:
            print(f"[{run_name}] TIME LIMIT reached - stopping after epoch {ep}; the state of the last completed "
                  f"epoch is saved for --resume_state", flush=True)
            break
        if a.epochs_this_run and ep + 1 - start_ep >= a.epochs_this_run and ep + 1 < a.epochs:
            print(f"[{run_name}] chunk of {a.epochs_this_run} epoch(s) done at epoch {ep} of {a.epochs} - stopping; "
                  f"continue with --resume_state {run_name}_state.pt", flush=True)
            break
    ref = model.state_dict()
    if avg is None:                                        # stopped before the averaging window: use the current weights
        avg, n_avg = {k: v.detach().float().cpu() for k, v in ref.items()}, 1
    model.load_state_dict({k: avg[k].to(ref[k].dtype) for k in ref})
    Pg = predict(model, gold_ids, id2row, a, dev)
    Po = predict(model, oof_ids, id2row, a, dev)
    g_macro, g_per = macro_auc(gold_Y, Pg) if len(Pg) else (float("nan"), {})
    ck = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "arch": a.arch, "res": a.res,
          "lab": LAB, "norm": "imagenet", "avg_epochs": n_avg, "gold_auc": g_macro, "aucs": g_per,
          "stopped_early": stopped, "hist": hist}
    torch.save(ck, os.path.join(a.out, f"{run_name}.pt"))
    np.savez(os.path.join(a.out, f"{run_name}_preds.npz"), gold_pred=Pg, gold_ids=np.array(gold_ids),
             oof_pred=Po, oof_ids=np.array(oof_ids))
    json.dump({k: v for k, v in ck.items() if k != "model"}, open(os.path.join(a.out, f"{run_name}.json"), "w"),
              indent=1)
    print(f"[{run_name}] DONE gold macro-AUC {g_macro:.4f} (avg of {n_avg} epochs)", flush=True)
    del model, opt
    torch.cuda.empty_cache()


def train_with_fallback(run_name, *args, a, **kw):
    """Run train_one; on CUDA out-of-memory retry with gradient checkpointing, then with half the batch size."""
    import gc
    for _ in range(4):
        oom = None
        try:
            return train_one(run_name, *args, a=a, **kw)
        except torch.cuda.OutOfMemoryError as e:
            oom = str(e).splitlines()[0][:120]
        gc.collect()                                       # outside the except block: failed frames are released
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not a.grad_ckpt:
            a.grad_ckpt, change = True, "gradient checkpointing ON"
        elif a.bs > 2:
            a.bs, change = a.bs // 2, f"batch size {a.bs // 2}"
        else:
            raise RuntimeError(f"[{run_name}] out of memory even with checkpointing at batch size {a.bs}")
        print(f"[{run_name}] CUDA out of memory ({oom}) -> restarting this run with {change}", flush=True)
    raise RuntimeError(f"[{run_name}] out-of-memory fallbacks exhausted")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cv", "final", "evalref"], required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--folds_json", default=None)
    ap.add_argument("--folds", default="0")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--tag", default="fx1")
    ap.add_argument("--arch", default="coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k")
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--k_eval", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--bb_lr", type=float, default=3e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--avg_last", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--monitor_gold", action="store_true")
    ap.add_argument("--time_limit_h", type=float, default=11.0)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use only this many training studies")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--init_ckpt", default=None, help="warm start: load these weights, then train a fresh schedule")
    ap.add_argument("--resume_state", default=None, help="exact resume from a *_state.pt of the same run")
    ap.add_argument("--save_state", action="store_true", help="save the full training state after every epoch")
    ap.add_argument("--pct_start", type=float, default=0.15, help="OneCycle warm-up fraction")
    ap.add_argument("--epochs_this_run", type=int, default=0,
                    help="train only this many epochs now, save the state and stop (resume later); 0 = all")
    ap.add_argument("--train_csv", default=None)
    ap.add_argument("--out", default="/kaggle/working")
    a = ap.parse_args()
    t_start = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)

    _, masks, ids = open_corpus()
    id2row = {u: i for i, u in enumerate(ids)}
    tr = pd.read_csv(a.train_csv or find("train.csv"), dtype={"StudyInstanceUID": str})
    gmask = tr[LAB].notna().all(axis=1)
    gold = tr[gmask].set_index("StudyInstanceUID")[LAB].astype(np.float32)
    gold_ids = [u for u in gold.index if u in id2row]
    gold_Y = gold.loc[gold_ids].to_numpy(np.float32)
    print(f"corpus {len(ids)} studies | gold in corpus {len(gold_ids)}/{len(gold)}", flush=True)

    if a.mode == "evalref":
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        a.arch, a.res = ck.get("arch", a.arch), int(ck.get("res", a.res))
        bb = build_backbone(a.arch, pretrained=False)
        model = RaptorClassifier(bb, F_dim=bb.num_features)
        model.load_state_dict(ck["model"], strict=True)
        model.to(dev)
        P = predict(model, gold_ids, id2row, a, dev)
        m, per = macro_auc(gold_Y, P)
        np.savez(os.path.join(a.out, f"{a.tag}_preds.npz"), gold_pred=P, gold_ids=np.array(gold_ids))
        json.dump({"ckpt": a.ckpt, "arch": a.arch, "res": a.res, "gold_auc": m, "aucs": per},
                  open(os.path.join(a.out, f"{a.tag}.json"), "w"), indent=1)
        print(f"[{a.tag}] reference {os.path.basename(a.ckpt)} gold macro-AUC {m:.4f}", flush=True)
        return

    lab = pd.read_parquet(a.labels)
    if "StudyInstanceUID" in lab.columns:
        lab = lab.set_index("StudyInstanceUID")
    lab.index = lab.index.astype(str)
    goldset = set(gold.index)
    pool = [u for u in ids if u in lab.index and u not in goldset]
    labels = {u: np.clip(lab.loc[u, LAB].to_numpy(np.float32), 0.0, 1.0) for u in pool}
    if a.limit:
        pool = pool[:a.limit]
    print(f"labelled pool {len(pool)} (gold excluded)", flush=True)

    if a.mode == "cv":
        fmap = json.load(open(a.folds_json))["folds"]
        for f in [int(x) for x in a.folds.split(",") if x.strip()]:
            tr_ids = [u for u in pool if fmap.get(u, -1) != f]
            oof = [u for u in pool if fmap.get(u, -1) == f]
            train_with_fallback(f"{a.tag}_cv_fold{f}", tr_ids, oof, gold_ids, labels, gold_Y, id2row, a=a, dev=dev, t_start=t_start)
    else:
        train_with_fallback(f"{a.tag}_final_seed{a.seed}", pool, [], gold_ids, labels, gold_Y, id2row, a=a, dev=dev, t_start=t_start)


if __name__ == "__main__":
    main()
