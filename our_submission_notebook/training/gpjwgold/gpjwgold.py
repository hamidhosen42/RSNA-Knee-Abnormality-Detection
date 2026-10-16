"""Score each reader checkpoint (author's folds 0-2 + our folds 3-4) on the 58 gold studies straight from DICOM
with the author's infer.py, to compare per-fold gold AUC and to prove our checkpoints load in the inference path."""
import glob, os, shutil, subprocess, sys
import numpy as np, pandas as pd


def find(name):
    for pat in ["/kaggle/input/*/", "/kaggle/input/*/*/", "/kaggle/input/*/*/*/"]:
        for d in sorted(glob.glob(pat)):
            if os.path.exists(os.path.join(d, name)):
                return d.rstrip("/")
    raise FileNotFoundError(name)


assets, data, ours = find("cnxt_v0_fold0.pt"), find("train_series.csv"), find("cnxt_v0_fold3.pt")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--find-links", assets,
                "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=False)
os.makedirs("/kaggle/working/src", exist_ok=True)
for f in ("knee.py", "preprocess.py", "infer.py"):
    shutil.copy(f"{assets}/{f}", f"/kaggle/working/src/{f}")
sys.path.insert(0, "/kaggle/working/src")
from sklearn.metrics import roc_auc_score
import infer
from knee import LABELS

tr = pd.read_csv(f"{data}/train.csv")
gold = tr[tr[LABELS].notna().all(axis=1)].set_index("StudyInstanceUID")[LABELS]
ids = gold.index.tolist()
print("gold studies", len(ids), "| assets", assets, "| ours", ours, flush=True)
ckpts = sorted(glob.glob(f"{assets}/cnxt_v0_fold*.pt")) + sorted(glob.glob(f"{ours}/cnxt_v0_fold*.pt"))
preds = {}
for c in ckpts:
    name = os.path.basename(os.path.dirname(c)).split("-")[-1] + ":" + os.path.basename(c)
    P = infer.run(data, "train", [c], studies=ids)
    preds[name] = P
    P.to_csv(f"/kaggle/working/gold_{os.path.basename(c).replace('.pt', '')}.csv")
    auc = np.mean([roc_auc_score(gold[l], P.loc[ids, l]) for l in LABELS])
    print(f"{name}: gold macro AUC {auc:.4f}", flush=True)
mean = lambda ks: sum(preds[k] for k in ks) / len(ks)
ks = list(preds)
for label, sel in (("author folds 0-2", [k for k in ks if k.endswith(("fold0.pt", "fold1.pt", "fold2.pt"))]),
                   ("our folds 3-4", [k for k in ks if k.endswith(("fold3.pt", "fold4.pt"))]), ("all 5", ks)):
    P = mean(sel)
    print(f"{label} ({len(sel)}): gold macro AUC {np.mean([roc_auc_score(gold[l], P.loc[ids, l]) for l in LABELS]):.4f}", flush=True)
