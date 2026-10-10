"""R2 (audit/15): out-of-fold soft-bootstrap ("pseudo-label") retrain of goodpjw2008's reader, as a clean A/B.

One T4x2 GPU kernel, three phases:
  1. 5-fold OOF of the existing reader: the author's cnxt_v0_fold{0,1,2}.pt (goodpjw2008/rsna-knee-2-5d-convnext-reader
     v1) and ours cnxt_v0_fold{3,4}.pt (esfersami50/rsna-knee-gpjw-train-f34 v1), each on its own held-out fold of
     the make_labels v0 split (the split all five were trained with), k_eval 12, fp16 - folds 0/2/4 on GPU 0, 1/3 on 1.
  2. Targets, fixed before running: B = clip(0.5 * v0 soft label + 0.5 * sigmoid(OOF logit), 0.01, 0.99) on the 4,349
     weak studies; A = v0 unchanged.  Gold-58 rows are never trained on.
  3. Two full-data models in parallel (fold -1 = all weak studies), identical except for the targets:
     arm A on GPU 0, arm B on GPU 1; the reader's recipe as in E16 (convnext_tiny, 256 px, k 8, 14 epochs, bs 8 as
     micro 2 x accum 4, AdamW 1e-4/5e-4, wd 0.02, warm-up + cosine, EMA 0.998, clip 3, seed 0, fp16 + GradScaler).
     Gold-58 is evaluated as a diagnostic only.

Output: cnxt_plA_full.pt, cnxt_plB_full.pt (infer.py format), labels_pl.csv, oofck_fold{k}.csv, gold_arm{A,B}.csv, logs.
Pre-registered: adopt B only if LB(B) - LB(A) >= +0.003 (each leg alone) and B's gold-58 is not more than 0.01 below A's.

    python make_pseudo_kernel.py [--owner=esfersami50] [--probe]  ->  kaggle/gpjwpseudo[_probe]/
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = "--probe" in sys.argv
OWNER = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--owner=")), "esfersami50")
CACHE_OWNER = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--cache-owner=")), "esfersami50")
ALPHA = 0.5
N_SHARDS = 5
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"

knee = open(os.path.join(HERE, "src", "knee.py")).read()

MAIN = r'''

# ======================================================================= pseudo-label A/B kernel (ours)
import copy, glob, json, math, os, re, subprocess, sys, threading, time
import cv2
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ALPHA = __ALPHA__
PROBE = __PROBE__
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
def train_arm(arm):
    a = dict(ARGS, arm=arm, alpha=ALPHA if arm == "B" else None)
    torch.manual_seed(a["seed"]); np.random.seed(a["seed"])
    log = open(f"{OUT}/log_arm{arm}.txt", "a")
    say(arm, json.dumps(a), log)
    lab = pd.read_csv(f"{OUT}/labels_pl.csv", index_col=0) if arm == "B" else build_labels()
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
    print(f"gpus {torch.cuda.device_count()} | script {script_path()} | probe {PROBE} | alpha {ALPHA}", flush=True)
    codes = run_parallel([("GPJW_OOF", "0,2,4"), ("GPJW_OOF", "1,3")])
    print("oof exit codes:", codes, flush=True)
    if any(c != 0 for c in codes.values()):
        raise SystemExit("OOF phase failed")
    build_targets()
    codes = run_parallel([("GPJW_ARM", "A"), ("GPJW_ARM", "B")])
    print("arm exit codes:", codes, flush=True)
    if all(c != 0 for c in codes.values()):
        raise SystemExit(1)
'''

src = knee + MAIN.replace("__PROBE__", "True" if PROBE else "False").replace("__ALPHA__", repr(ALPHA))
compile(src, "gpjwpseudo.py", "exec")
d = os.path.join(HERE, "..", "kaggle", "gpjwpseudo" + ("_probe" if PROBE else "") + ("" if OWNER == "esfersami50" else f"_{OWNER}"))
os.makedirs(d, exist_ok=True)
open(os.path.join(d, "gpjwpseudo.py"), "w").write(src)
slug = "rsna-knee-gpjw-pseudo" + ("-probe" if PROBE else "")
meta = {"id": f"{OWNER}/{slug}", "title": slug, "code_file": "gpjwpseudo.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4", "docker_image": DOCKER, "docker_image_pinning_type": "original",
        "dataset_sources": ["stevenleehans/rsna-knee-llm-report-labels/6", "pilkwang/rsna-knee-llm-labels/1",
                            "lixin73/rsna-knee-llm-report-labels-sol56/1",
                            "laymond/rsna-knee-abnormality-qwen3-8b-weak-labels/2",
                            "goodpjw2008/rsna-knee-2-5d-convnext-reader/1"],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "kernel_sources": [f"{CACHE_OWNER}/rsna-knee-gpjw-cache-{i}/1" for i in range(N_SHARDS)]
                          + ["esfersami50/rsna-knee-gpjw-train-f34/1"], "model_sources": []}
json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
print("wrote", d)
