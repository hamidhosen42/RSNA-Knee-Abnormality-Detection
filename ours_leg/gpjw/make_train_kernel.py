"""Build the GPU kernel that trains goodpjw2008's reader on its missing folds 3 and 4.

Faithful to the author's public code (dataset goodpjw2008/rsna-knee-2-5d-convnext-reader v1):
  * knee.py embedded verbatim (model, dataset, window sampling, augmentation);
  * labels + folds rebuilt exactly as make_labels.py v0 (mean of 4 public LLM tables, clip 0.01-0.99;
    GroupKFold(5) on normalised report text after RandomState(0) permutation; gold rows fold -1);
  * train.py's loop and hyper-parameters (14 epochs, bs 8, k 8 / k_eval 12, AdamW 1e-4 / 5e-4,
    wd 0.02, warm-up 1 epoch + cosine, EMA 0.998, grad clip 3, 256 px, convnext_tiny ImageNet).
Deviations, all forced by Kaggle: T4 has no bf16 -> fp16 autocast + GradScaler; the cache is read from
the PNG shard kernels (pixel-identical to the author's .npy cache); 2 data workers per fold process;
folds 3 and 4 run in parallel, one per GPU.

Output: cnxt_v0_fold{3,4}.pt ({"model": ema state_dict, "args": ...}, the format infer.py loads),
oof_fold{k}.csv, log_fold{k}.txt.

    python make_train_kernel.py [--probe]   ->  kaggle/gpjwtrain[_probe]/
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = "--probe" in sys.argv
# E17: --backbone convnext_small (default convnext_tiny = the author's reader, E16)
BACKBONE = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--backbone=")), "convnext_tiny")
TAG = "" if BACKBONE == "convnext_tiny" else "-" + BACKBONE.replace("convnext_", "")
OWNER = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--owner=")), "esfersami50")
# whose cache shard kernels to mount (sami's are shared with hosen42)
CACHE_OWNER = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--cache-owner=")), OWNER)
N_SHARDS = 5
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"

knee = open(os.path.join(HERE, "src", "knee.py")).read()

MAIN = r'''

# ======================================================================= training kernel (ours)
import copy, glob, hashlib, json, math, os, re, subprocess, sys, threading, time
import cv2
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

FOLDS = [3, 4]
CKPT_PREFIX = "__CKPT_PREFIX__"
EVAL_EPOCHS = {0, 4, 8, 11, 12, 13}
PROBE = __PROBE__
ARGS = dict(backbone="__BACKBONE__", res=256, k=8, k_eval=12, epochs=1 if PROBE else 14, bs=8, micro_bs=2, lr=1e-4,
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
'''

src = knee + (MAIN.replace("__PROBE__", "True" if PROBE else "False").replace("__BACKBONE__", BACKBONE)
               .replace("__CKPT_PREFIX__", "cnxt_v0" if not TAG else "cnxt" + TAG.replace("-", "_")))
compile(src, "gpjwtrain.py", "exec")
d = os.path.join(HERE, "..", "kaggle", ("gpjwtrain_probe" if PROBE else "gpjwtrain") + TAG.replace("-", "_")
                 + ("" if OWNER == "esfersami50" else f"_{OWNER}"))
os.makedirs(d, exist_ok=True)
open(os.path.join(d, "gpjwtrain.py"), "w").write(src)
slug = ("rsna-knee-gpjw-train-probe" if PROBE else "rsna-knee-gpjw-train-f34") + TAG
meta = {"id": f"{OWNER}/{slug}", "title": slug, "code_file": "gpjwtrain.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4", "docker_image": DOCKER, "docker_image_pinning_type": "original",
        "dataset_sources": ["stevenleehans/rsna-knee-llm-report-labels/6", "pilkwang/rsna-knee-llm-labels/1",
                            "lixin73/rsna-knee-llm-report-labels-sol56/1",
                            "laymond/rsna-knee-abnormality-qwen3-8b-weak-labels/2"],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "kernel_sources": [f"{CACHE_OWNER}/rsna-knee-gpjw-cache-{i}/1" for i in range(N_SHARDS)], "model_sources": []}
json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
print("wrote", d)
