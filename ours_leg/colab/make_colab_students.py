"""Build the Colab Pro+ notebook that trains more Mirror-student readers on an A100/H100 and auto-submits the ensemble.

The student recipe is exactly the one that gave reader C (leg alone 0.948) and C3 (0.949): goodpjw2008's knee.py
(embedded verbatim), full data (fold -1), 14 epochs, effective batch 8, AdamW 1e-4 / 5e-4, wd 0.02, warm-up 1 epoch +
cosine, EMA 0.998, clip 3, 256 px, k 8 / k_eval 12, targets clip((1 - tw) * v0 + tw * Mirror teacher).  Colab-only
differences: bf16 autocast (the author's own precision; T4 needed fp16), a bigger micro-batch where memory allows (same
effective batch; the model has no BatchNorm), more data workers, and resumable per-epoch state on Google Drive.

The notebook then
  * uploads the finished checkpoints to the private Kaggle dataset <user>/rsna-knee-colab-students,
  * pulls the H955C notebook (esfersami50/rsna-knee-h955c = 0.8 x rank(0.954) + 0.2 x rank(reader C), LB 0.955),
    swaps its reader for the student ensemble (C, C2, C3 + every new student that passes the gold-58 guard; infer.py
    averages their probabilities), pushes it as <user>/rsna-knee-h955ens, waits for the commit, checks its log and
    submits it when AUTO_SUBMIT is on.

    python make_colab_students.py   ->  colab/RSNA_Colab_MirrorStudents.ipynb
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
knee = open(os.path.join(ROOT, "ours_leg", "gpjw", "src", "knee.py")).read()

TRAIN = r'''

# ======================================================================= Colab student trainer (ours)
import copy, glob, json, math, os, re, sys, time
import cv2
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

CFG = json.load(open(os.environ["RSNA_STUDENT_CFG"]))
EVAL_EPOCHS = {0, 4, 8, 11, 12, 13}
BASE = dict(res=256, k=8, k_eval=12, epochs=14, bs=8, lr=1e-4, lr_head=5e-4, wd=0.02, ema=0.998, fold=-1)
T0 = time.time()


def say(tag, msg, log=None):
    print(f"[{tag} {(time.time() - T0) / 60:.0f}m] {msg}", flush=True)
    if log:
        log.write(msg + "\n"); log.flush()


def report_group(text):
    return re.sub(r"\W+", "", str(text).lower())


def build_labels():
    """make_labels.py v0: mean of the four public LLM label tables, clip 0.01-0.99; GroupKFold folds; gold columns."""
    D, L = CFG["data_dir"], CFG["label_dir"]
    tr = pd.read_csv(f"{D}/train.csv")
    ids = tr.StudyInstanceUID
    gold = tr.set_index("StudyInstanceUID")[LABELS]
    is_gold = gold.notna().all(axis=1)
    src = {"steven": pd.read_csv(f"{L}/llm_labels_v4_blend.csv"), "pilkwang": pd.read_csv(f"{L}/report_labels_v2.csv"),
           "lixin": pd.read_csv(f"{L}/labels_llm_gpt56sol.csv")}
    q = pd.read_csv(f"{L}/qwen_knee_weak_labels.csv")
    src["qwen"] = q[["StudyInstanceUID"] + [f"{c}__label" for c in LABELS]].set_axis(["StudyInstanceUID"] + LABELS, axis=1)
    src = {k: v.set_index("StudyInstanceUID")[LABELS].astype(float).reindex(ids) for k, v in src.items()}
    num = sum(v.fillna(0) for v in src.values())
    den = sum(v.notna().astype(float) for v in src.values())
    out = (num / den).clip(0.01, 0.99)
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


def macro_auc(y, p):
    aucs = [roc_auc_score(y[:, i], p[:, i]) if 0 < y[:, i].sum() < len(y) else np.nan for i in range(y.shape[1])]
    return float(np.nanmean(aucs)), aucs


def build_targets(tw):
    lab = build_labels()
    m = pd.read_csv(CFG["teacher_csv"], dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[LABELS]
    m = m.reindex(lab.index.astype(str))
    m.index = lab.index
    ok = m.notna().all(axis=1) & ~(m == 0.5).all(axis=1)
    weak = lab.is_gold == 0
    idx = lab.index[weak & ok]
    t_auc, _ = macro_auc((lab.loc[idx, LABELS].values > 0.5).astype(int), m.loc[idx].values)
    say("targets", f"tw {tw} | teacher scored {int(ok[weak].sum())}/{int(weak.sum())} weak studies | teacher vs v0 {t_auc:.4f}")
    lab.loc[idx, LABELS] = ((1 - tw) * lab.loc[idx, LABELS].values + tw * m.loc[idx].values).clip(0.01, 0.99)
    return lab


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
    metas = sorted(glob.glob(f"{CFG['cache_dir']}/**/cache384_meta.csv", recursive=True))
    rows = []
    for mf in metas:
        d = pd.read_csv(mf)
        d["dir"] = os.path.join(os.path.dirname(mf), "cache384")
        rows.append(d)
    meta = pd.concat(rows, ignore_index=True)
    ok = meta["n"].notna() & (meta["error"].isna() if "error" in meta else True)
    meta = meta[ok].drop_duplicates("series")
    paths = dict(zip(meta.series, [os.path.join(d, f"{s}.npz") for d, s in zip(meta.dir, meta.series)]))
    table = build_study_table(pd.read_csv(f"{CFG['data_dir']}/train_series.csv"), dict(zip(meta.series, meta.n)))
    return table, (lambda s: PngVol(paths[s])), len(metas), len(meta)


@torch.no_grad()
def predict(model, loader, res, amp):
    model.eval()
    out = []
    for x, mask, pos, _ in loader:
        x, mask, pos = x.cuda(non_blocking=True), mask.cuda(), pos.cuda()
        with torch.autocast("cuda", dtype=amp):
            out.append(model(x, mask, pos, res=res).float().cpu())
    return torch.cat(out).numpy()


def train_arm(arm):
    spec = CFG["arms"][arm]
    a = dict(BASE, arm=arm, **spec)
    os.makedirs(CFG["out_dir"], exist_ok=True); os.makedirs(CFG["state_dir"], exist_ok=True)
    final = f"{CFG['out_dir']}/cnxt_{arm}_full.pt"
    if os.path.exists(final):
        say(arm, "already finished: " + final); return
    a.setdefault("seed", 0)
    torch.manual_seed(a["seed"]); np.random.seed(a["seed"])
    log = open(f"{CFG['out_dir']}/log_{arm}.txt", "a")
    bf16 = torch.cuda.is_bf16_supported()
    amp = torch.bfloat16 if bf16 else torch.float16
    gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    # measured on T4 (E16/E17): peak 8.7 GB (tiny) / 13.6 GB (small) at 2 studies per micro-batch
    per_study_gb = {"convnext_tiny": 4.5, "convnext_small": 7.0}.get(a["backbone"].split(".")[0], 8.0)
    micro = int(a.get("micro_bs") or max(m for m in (1, 2, 4, 8) if m * per_study_gb <= 0.8 * gb))
    accum = a["bs"] // micro
    a.update(micro_bs=micro, amp="bf16" if bf16 else "fp16")
    say(arm, json.dumps(a) + f" | gpu {torch.cuda.get_device_name(0)} {gb:.0f} GB", log)
    lab = build_targets(a["tw"])
    table, loader, n_meta, n_series = data_table()
    say(arm, f"cache: {n_meta} shard metas, {n_series} series", log)
    gold = lab[lab.is_gold == 1]
    tr = lab[lab.is_gold == 0]
    missing = [s for s in tr.index if s not in table]
    assert len(missing) < 0.01 * len(tr), f"{len(missing)} train studies have no cached series - is the cache complete?"
    say(arm, f"train {len(tr)}  gold {len(gold)} | micro {micro} x accum {accum} | amp {a['amp']}", log)
    nw = int(CFG.get("workers") or max(2, (os.cpu_count() or 4) - 1))
    dl_kw = dict(num_workers=nw, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    train_dl = torch.utils.data.DataLoader(
        StudyDataset(tr.index, table, None, tr[LABELS].values.astype(np.float32), a["k"], train=True, loader=loader),
        batch_size=micro, shuffle=True, drop_last=True, **dl_kw)
    gold_dl = torch.utils.data.DataLoader(StudyDataset(gold.index, table, None, None, a["k_eval"], loader=loader),
                                          batch_size=8, shuffle=False, **dl_kw)
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
    scaler = torch.amp.GradScaler("cuda", enabled=not bf16)
    step, start, gp, g_auc, g_each = 0, 0, None, float("nan"), []
    state_path = f"{CFG['state_dir']}/state_{arm}.pt"
    if os.path.exists(state_path):   # Colab disconnects: continue after the last finished epoch
        st = torch.load(state_path, map_location="cuda", weights_only=False)
        model.load_state_dict(st["model"]); ema.load_state_dict(st["ema"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        step, start = st["step"], st["epoch"] + 1
        say(arm, f"resumed after epoch {st['epoch']}", log)
    for ep in range(start, a["epochs"]):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for it, (x, mask, pos, y) in enumerate(train_dl):
            if it >= per_ep * accum:
                break
            x, mask, pos, y = x.cuda(non_blocking=True), mask.cuda(), pos.cuda(), y.cuda()
            with torch.autocast("cuda", dtype=amp):
                logit = model(x, mask, pos, res=a["res"], train=True)
            loss = F.binary_cross_entropy_with_logits(logit.float(), y)
            scaler.scale(loss / accum).backward()
            tot += loss.item() / accum
            if (it + 1) % accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(opt); scaler.update()
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
            if n % 100 == 0:
                say(arm, f"  ep {ep} step {n}/{per_ep} loss {tot / n:.4f} {(time.time() - t0) / n:.2f}s/it "
                         f"mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")
        msg = f"ep {ep:2d} loss {tot / max(n, 1):.4f} time {time.time() - t0:.0f}s"
        if ep in EVAL_EPOCHS or ep == a["epochs"] - 1:
            gp = predict(ema, gold_dl, a["res"], amp)
            g_auc, g_each = macro_auc(gold_y, gp)
            msg += f" | gold {g_auc:.4f}"
        say(arm, msg, log)
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(), "step": step, "epoch": ep},
                   state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)
    torch.save({"model": ema.state_dict(), "args": a, "epoch": a["epochs"] - 1, "gold": g_auc}, final + ".tmp")
    os.replace(final + ".tmp", final)
    pd.DataFrame(gp, index=gold.index, columns=LABELS).to_csv(f"{CFG['out_dir']}/gold_{arm}.csv")
    json.dump({"arm": arm, "gold": g_auc, "gold_each": dict(zip(LABELS, g_each)), "args": a},
              open(f"{CFG['out_dir']}/result_{arm}.json", "w"), indent=1)
    say(arm, "gold per class: " + " ".join(f"{c[:6]}={x:.3f}" for c, x in zip(LABELS, g_each)), log)
    say(arm, f"DONE arm {arm} in {(time.time() - T0) / 3600:.2f} h -> {final}", log)
    os.remove(state_path)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    train_arm(sys.argv[1])
'''
compile(knee + TRAIN, "rsna_colab_student.py", "exec")


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.splitlines(keepends=True)}


INTRO = """# RSNA Knee: Mirror-student readers on Colab Pro+ (A100/H100) with auto-submission

Trains more **Mirror-student readers** with the exact recipe of reader C (leg alone 0.948) and C3 (0.949). The
2.5D ConvNeXt reader learns from targets `(1 - tw) * v0 LLM labels + tw * Mirror teacher`. The notebook then:
1. uploads the checkpoints to the private Kaggle dataset `<user>/rsna-knee-colab-students`;
2. pulls H955C (LB 0.955) and swaps its reader for the student ensemble. That ensemble is C, C2, C3 plus every new
   student that passes the gold-58 guard; infer.py averages their probabilities.
3. pushes the result as `<user>/rsna-knee-h955ens`, waits for the Kaggle commit, checks its log and **submits** it
   when `AUTO_SUBMIT = True`.

**Before you run**
1. **Runtime → Change runtime type → A100** (or H100). Pick High-RAM if offered. L4 works but is slower. T4 is not
   enough (it has no bf16).
2. Open 🔑 **Secrets** in the left panel and add `KAGGLE_API_TOKEN` (the `KGAT_…` token of **esfersami50**). Turn on
   *Notebook access*. Never paste the token into a cell or a chat.
   - The training data (cache shards, Mirror teacher) and the H955C notebook are on esfersami50, so this has to be
     that account's token.
3. Run all.

**Time and cost**
- Data download takes about 30–60 min: about 54 GB of PNG cache.
- Training takes roughly 1–2 h per arm on an A100.
- The upload takes a few minutes.
- The Kaggle commit uses about 15 min of the token owner's Kaggle GPU quota. Submitting uses one of the day's
  submissions.

**Resume:** every finished epoch is saved to Drive. If Colab disconnects, run all again and finished work is skipped.

**Rules note:** the Mirror teacher is OAI-trained, so these students are OAI-derived. They are for the public-LB line,
not a rule-clean final.
"""

CONFIG = '''# ===== CONFIG =====
KAGGLE_USER = "esfersami50"          # owner of the token in Colab Secrets (data + H955C + dataset + submission live here)
DRIVE_DIR   = "/content/drive/MyDrive/15-Kaggle/RSNA Knee Abnormality Detection/colab_students"
LOCAL       = "/content/rsna"         # fast local disk (cache ~54 GB)
# New students: tw = teacher weight in the targets (C: 0.5, C3: 0.75). Diversity matters more than seeds:
# C, C2, C3 correlate ~0.96 on gold-58, so a different backbone is the most useful new member.
ARMS = {
    "S75": dict(backbone="convnext_small", tw=0.75, seed=0),   # new backbone, C3 targets
    "T75s1": dict(backbone="convnext_tiny", tw=0.75, seed=1),  # C3 recipe, second seed
}
GOLD_GUARD   = 0.9127                 # a new student joins the ensemble only if gold-58 >= C's 0.9227 - 0.01 (guard, not tuning)
AUTO_SUBMIT  = True                   # submit H955ENS automatically when its commit log passes every check
DATASET_SLUG = "rsna-knee-colab-students"
SUBMIT_SLUG  = "rsna-knee-h955ens"
BASE_KERNEL  = "esfersami50/rsna-knee-h955c"   # LB 0.955: 0.8 x rank(0.954 Mirror+S6) + 0.2 x rank(reader C)
# LB-validated students already on Kaggle (always in the ensemble): C 0.948, C2 0.948, C3 0.949
BASE_STUDENTS = [("esfersami50/rsna-knee-gpjw-student/1", "cnxt_plC_full.pt"),
                 ("esfersami50/rsna-knee-gpjw-student2/1", "cnxt_plC2_full.pt"),
                 ("esfersami50/rsna-knee-gpjw-student2/1", "cnxt_plC3_full.pt")]
'''

SETUP = '''import os, subprocess, json, shutil, time, re, glob
from google.colab import drive
drive.mount("/content/drive")
for d in (DRIVE_DIR, f"{DRIVE_DIR}/out", f"{DRIVE_DIR}/state", LOCAL):
    os.makedirs(d, exist_ok=True)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
import torch
assert torch.cuda.is_available(), "No GPU - Runtime > Change runtime type > A100"
assert torch.cuda.is_bf16_supported(), "This GPU has no bf16 (T4/P100) - use A100/H100/L4"
free_gb = shutil.disk_usage("/content").free / 1e9
print(f"torch {torch.__version__} | local disk free {free_gb:.0f} GB | cpus {os.cpu_count()}")
assert free_gb > 75, "need ~75 GB free local disk for the cache"
'''

PIP = '''!pip -q install -U "kaggle>=2.2" timm
from google.colab import userdata
_tok = userdata.get("KAGGLE_API_TOKEN").strip()      # never printed
os.environ["KAGGLE_API_TOKEN"] = _tok
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
_p = os.path.expanduser("~/.kaggle/access_token"); open(_p, "w").write(_tok); os.chmod(_p, 0o600)
del _tok


def sh(cmd, check=False, quiet=False):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if not quiet:
        print("$", cmd[:150], "->", (r.stdout + r.stderr).strip()[-400:])
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed: {cmd[:150]}")
    return r


import timm; print("timm", timm.__version__)
sh("kaggle --version")
'''

DATA = '''# ===== DATA: competition csvs, 4 public label tables, Mirror teacher, 5 PNG cache shards (local disk) =====
from concurrent.futures import ThreadPoolExecutor
C = "rsna-knee-abnormality-detection"
D, L, TCH, CACHE = f"{LOCAL}/data", f"{LOCAL}/labels", f"{LOCAL}/teacher", f"{LOCAL}/cache"
for d in (D, L, TCH, CACHE):
    os.makedirs(d, exist_ok=True)


def get_comp(f):
    for attempt in range(5):
        if os.path.exists(f"{D}/{f}"):
            return
        sh(f"kaggle competitions download {C} -f {f} -p {D} -q")
        if os.path.exists(f"{D}/{f}.zip"):
            sh(f"cd {D} && unzip -o -q {f}.zip && rm {f}.zip")
        time.sleep(10 * attempt)
    assert os.path.exists(f"{D}/{f}"), f


for f in ("train.csv", "train_series.csv"):
    get_comp(f)
for ds, f in [("stevenleehans/rsna-knee-llm-report-labels", "llm_labels_v4_blend.csv"),
              ("pilkwang/rsna-knee-llm-labels", "report_labels_v2.csv"),
              ("lixin73/rsna-knee-llm-report-labels-sol56", "labels_llm_gpt56sol.csv"),
              ("laymond/rsna-knee-abnormality-qwen3-8b-weak-labels", "qwen_knee_weak_labels.csv")]:
    if not os.path.exists(f"{L}/{f}"):
        sh(f"kaggle datasets download {ds} -f {f} -p {L} -q")
        if os.path.exists(f"{L}/{f}.zip"):
            sh(f"cd {L} && unzip -o -q {f}.zip && rm {f}.zip")
    assert os.path.exists(f"{L}/{f}"), f
if not os.path.exists(f"{TCH}/mirror_train_raw.csv"):
    sh(f"kaggle kernels output esfersami50/rsna-knee-mirror-teacher -p {TCH} --file-pattern 'mirror_train_raw.csv'")
assert os.path.exists(f"{TCH}/mirror_train_raw.csv"), "teacher csv missing (is the token esfersami50's?)"


def shard_ok(i):
    metas = glob.glob(f"{CACHE}/{i}/**/cache384_meta.csv", recursive=True)
    if not metas:
        return False, 0, 0
    import pandas as pd
    m = pd.read_csv(metas[0]); m = m[m["n"].notna()]
    have = len(glob.glob(f"{os.path.dirname(metas[0])}/cache384/*.npz"))
    return have >= len(m), have, len(m)


# Robust shard downloader (the kaggle CLI stopped near 4,000 of ~4,870 files per shard and restarted from page 1 on
# every retry): list every page with retries, skip files already complete (npz opened to catch truncated ones),
# download the rest with retries / back-off, re-list when signed URLs expire.
import requests
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
_api = KaggleApi(); _api.authenticate()


def list_outputs(kernel, page_size=100):
    owner, slug = kernel.split("/")
    files, tok = [], ""
    while True:
        for attempt in range(8):
            try:
                with _api.build_kaggle_client() as kc:
                    req = ApiListKernelSessionOutputRequest()
                    req.user_name, req.kernel_slug, req.page_size, req.page_token = owner, slug, page_size, tok
                    resp = kc.kernels.kernels_api_client.list_kernel_session_output(req)
                break
            except Exception as e:
                print(f"  list {slug} retry {attempt}: {type(e).__name__} {str(e)[:100]}", flush=True)
                time.sleep(min(120, 5 * 2 ** attempt))
        else:
            raise RuntimeError(f"listing {kernel} failed")
        files += [(f.file_name, f.url) for f in (resp.files or [])]
        tok = resp.next_page_token
        if not tok:
            return files


def file_ok(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    if path.endswith(".npz"):
        try:
            with np.load(path) as z:
                z["off"]
        except Exception:
            return False
    return True


def fetch(path, url):
    for attempt in range(6):
        try:
            r = requests.get(url, timeout=180)
            if r.status_code == 200:
                tmp = path + ".part"
                with open(tmp, "wb") as fh:
                    fh.write(r.content)
                os.replace(tmp, path)
                return True
            if r.status_code in (401, 403, 404):
                return False                      # signed URL expired: the next round re-lists
        except Exception:
            pass
        time.sleep(min(60, 3 * 2 ** attempt))
    return False


def get_shard(i, workers=8):
    kernel, root = f"esfersami50/rsna-knee-gpjw-cache-{i}", f"{CACHE}/{i}"
    for rnd in range(6):
        files = list_outputs(kernel)
        todo = [(os.path.join(root, n), u) for n, u in files
                if n.endswith((".npz", "_meta.csv")) and not file_ok(os.path.join(root, n))]
        print(f"shard {i}: listed {len(files)} files, {len(todo)} to download (round {rnd}) | {(time.time() - t0) / 60:.0f} min", flush=True)
        if not todo:
            break
        for pth, _ in todo:
            os.makedirs(os.path.dirname(pth), exist_ok=True)
        with ThreadPoolExecutor(workers) as ex:
            got = sum(ex.map(lambda t: fetch(*t), todo))
        print(f"shard {i}: downloaded {got}/{len(todo)}", flush=True)
    ok, have, want = shard_ok(i)
    return f"shard {i}: {have}/{want} {'ok' if ok else 'INCOMPLETE'}"


t0 = time.time()
for i in range(5):                                  # one shard at a time (8 parallel downloads) to stay under rate limits
    print(get_shard(i), flush=True)
print("cache size:", sh(f"du -sh {CACHE}", quiet=True).stdout.strip())
assert all(shard_ok(i)[0] for i in range(5)), "a cache shard is incomplete - re-run this cell"
'''

WRITE = "%%writefile /content/rsna_colab_student.py\n" + knee + TRAIN

TRAINCELL = '''# ===== TRAIN every arm (one process per arm; finished arms are skipped; resumes from Drive) =====
cfg = {"data_dir": D, "label_dir": L, "teacher_csv": f"{TCH}/mirror_train_raw.csv", "cache_dir": CACHE,
       "out_dir": f"{DRIVE_DIR}/out", "state_dir": f"{DRIVE_DIR}/state", "arms": ARMS}
json.dump(cfg, open("/content/student_cfg.json", "w"))
for arm in ARMS:
    p = subprocess.Popen(["python", "-u", "/content/rsna_colab_student.py", arm],
                         env=dict(os.environ, RSNA_STUDENT_CFG="/content/student_cfg.json"),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="")
    assert p.wait() == 0, f"arm {arm} failed"
'''

SUMMARY = '''# ===== which new students join the ensemble (gold-58 guard only; LB decides afterwards) =====
results = {}
for arm in ARMS:
    f = f"{DRIVE_DIR}/out/result_{arm}.json"
    if os.path.exists(f):
        results[arm] = json.load(open(f))["gold"]
adopted = [a for a, g in results.items() if g >= GOLD_GUARD]
print("gold-58:", {a: round(g, 4) for a, g in results.items()}, "| guard", GOLD_GUARD, "| adopted:", adopted)
assert adopted, "no new student passed the guard - nothing to submit"
'''

UPLOAD = '''# ===== upload the adopted checkpoints as a private Kaggle dataset =====
up = f"{LOCAL}/upload"
shutil.rmtree(up, ignore_errors=True); os.makedirs(up)
for a in adopted:
    shutil.copy(f"{DRIVE_DIR}/out/cnxt_{a}_full.pt", up)
json.dump({"id": f"{KAGGLE_USER}/{DATASET_SLUG}", "title": DATASET_SLUG, "licenses": [{"name": "unknown"}]},
          open(f"{up}/dataset-metadata.json", "w"))
r = sh(f"kaggle datasets version -p {up} -m 'students {adopted}' --dir-mode zip")
if r.returncode != 0 or "not found" in (r.stdout + r.stderr).lower():
    sh(f"kaggle datasets create -p {up} --dir-mode zip", check=True)   # private by default
for _ in range(60):
    st = sh(f"kaggle datasets status {KAGGLE_USER}/{DATASET_SLUG}", quiet=True).stdout.strip().lower()
    print("dataset status:", st)
    if "ready" in st:
        break
    time.sleep(30)
assert "ready" in st, "dataset not ready"
'''

SUBMIT = '''# ===== H955ENS: pull H955C, swap its reader for the student ensemble, push, verify the commit, submit =====
kd = f"{LOCAL}/h955ens"
shutil.rmtree(kd, ignore_errors=True); os.makedirs(kd)
sh(f"kaggle kernels pull {BASE_KERNEL} -p {kd} -m", check=True)
nbf = [f for f in os.listdir(kd) if f.endswith(".ipynb")][0]
nb = json.load(open(f"{kd}/{nbf}"))
n_ck = len(BASE_STUDENTS) + len(adopted)
lines = ["# [h955ens] reader = probability mean of our Mirror students: C, C2, C3 (LB 0.948/0.948/0.949) + " + ", ".join(adopted),
         "_ck = []"]
for kern, f in BASE_STUDENTS:
    lines.append(f'_ck += sorted(glob.glob("/kaggle/input/**/{kern.split("/")[1]}/{f}", recursive=True))')
for a in adopted:
    lines.append(f'_ck += sorted(glob.glob("/kaggle/input/**/{DATASET_SLUG}/cnxt_{a}_full.pt", recursive=True))')
lines.append(f'assert len(_ck) == {n_ck}, _ck; print("reader ckpts", _ck, flush=True)')
hit = 0
for c in nb["cells"]:
    s = "".join(c["source"])
    if "cnxt_plC_full.pt" in s and "_ck = sorted(" in s:
        out = []
        for ln in s.splitlines():
            if ln.startswith("# [h954r-c]") or ln.startswith("_ck = sorted(glob.glob(") or ln.startswith("assert len(_ck) == 1"):
                if ln.startswith("_ck = sorted("):
                    out.extend(lines)
                continue
            out.append(ln)
        c["source"] = [x + "\\n" for x in out]
        hit += 1
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
assert hit == 1, f"reader cell not found exactly once ({hit})"
nb["cells"][0]["source"] = ["# RSNA Knee H955ENS: 0.8 x rank(0.954 Mirror+S6) + 0.2 x rank(Mirror-student ensemble)\\n"]
json.dump(nb, open(f"{kd}/{SUBMIT_SLUG}.ipynb", "w"), indent=1)
os.remove(f"{kd}/{nbf}")
meta = json.load(open(f"{kd}/kernel-metadata.json"))
for k in ("id_no", "keywords"):
    meta.pop(k, None)
meta.update(id=f"{KAGGLE_USER}/{SUBMIT_SLUG}", title=SUBMIT_SLUG, code_file=f"{SUBMIT_SLUG}.ipynb", is_private=True,
            enable_gpu=True, enable_internet=False, machine_shape="NvidiaTeslaT4", docker_image_pinning_type="original")
meta["dataset_sources"] = sorted(set(meta.get("dataset_sources", [])) | {f"{KAGGLE_USER}/{DATASET_SLUG}"})
meta["kernel_sources"] = sorted(set(meta.get("kernel_sources", [])) | {k for k, _ in BASE_STUDENTS})
json.dump(meta, open(f"{kd}/kernel-metadata.json", "w"), indent=2)
print(json.dumps({k: meta[k] for k in ("id", "dataset_sources", "kernel_sources", "docker_image")}, indent=1))
r = sh(f"kaggle kernels push -p {kd}", check=True)
ver = int(re.search(r"version (\\d+)", r.stdout + r.stderr).group(1))
print("pushed version", ver)

status = ""
for _ in range(120):                      # commit on 3 visible studies: ~10-20 min (+ queue)
    status = sh(f"kaggle kernels status {KAGGLE_USER}/{SUBMIT_SLUG}", quiet=True).stdout
    print(time.strftime("%H:%M"), status.strip()[-40:], flush=True)
    if any(x in status for x in ("COMPLETE", "ERROR", "CANCEL")):
        break
    time.sleep(60)
logdir = f"{LOCAL}/h955ens_out"
shutil.rmtree(logdir, ignore_errors=True)
sh(f"kaggle kernels output {KAGGLE_USER}/{SUBMIT_SLUG} -p {logdir} -q", quiet=True)
txt = ""
for f in glob.glob(f"{logdir}/*.log"):
    raw = open(f).read()
    try:
        txt += "".join(e.get("data", "") for e in json.loads(raw))
    except Exception:
        txt += raw
checks = {"commit COMPLETE": "COMPLETE" in status,
          "no Traceback": "Traceback" not in txt,
          "S6 stage PASS": '"status": "PASS"' in txt and "OURS_STAGE_COMPLETE" in txt,
          f"{n_ck} reader ckpts": bool(re.search(r"reader ckpts \\[" + ", ".join(["'[^']+'"] * n_ck) + r"\\]", txt)),
          "reader used, 0 failed": "READER_LEG used failed studies 0" in txt,
          "nested blend done": "NOTEBOOK_COMPLETE" in txt and '"mode": "nested"' in txt,
          "submission.csv written": os.path.exists(f"{logdir}/submission.csv")}
for k, v in checks.items():
    print(("OK   " if v else "FAIL ") + k)
if all(checks.values()) and AUTO_SUBMIT:
    ad = ",".join(adopted)
    sh(f"kaggle competitions submit rsna-knee-abnormality-detection -k {KAGGLE_USER}/{SUBMIT_SLUG} -v {ver} -f submission.csv "
       f"-m 'H955ENS: 0.8 x rank(0.954) + 0.2 x rank(students C,C2,C3,{ad})'", check=True)
    print(sh("kaggle competitions submissions rsna-knee-abnormality-detection", quiet=True).stdout[:600])
elif not all(checks.values()):
    print("NOT submitted: a check failed - look at the log above / on Kaggle")
else:
    print(f"AUTO_SUBMIT is off - submit {KAGGLE_USER}/{SUBMIT_SLUG} version {ver} by hand")
'''

cells = [md(INTRO), code(CONFIG), code(SETUP), code(PIP), code(DATA), code(WRITE), code(TRAINCELL), code(SUMMARY),
         code(UPLOAD), code(SUBMIT)]
for t in (CONFIG, SETUP.replace("from google.colab import drive", "").replace('drive.mount("/content/drive")', ""),
          DATA, TRAINCELL, SUMMARY, UPLOAD, SUBMIT):
    compile(t, "cell", "exec")
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm", "provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(ROOT, "colab")
os.makedirs(out, exist_ok=True)
json.dump(nb, open(os.path.join(out, "RSNA_Colab_MirrorStudents.ipynb"), "w"), indent=1)
print("wrote", os.path.join(out, "RSNA_Colab_MirrorStudents.ipynb"))
