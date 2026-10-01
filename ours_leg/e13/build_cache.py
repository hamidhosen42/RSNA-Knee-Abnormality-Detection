#!/usr/bin/env python3
"""E13 training cache: every train study through knee_prep.build_study (Kaggle CPU kernel).

knee_prep.py is embedded at push time (kaggle/push_e13.sh) and also written to the output, so
training and the submission leg import the byte-identical file this cache was built with.

Output (/kaggle/working):
  cache/<StudyInstanceUID>.npz   vol uint8 (4, 32, 224, 224), mask uint8 (4,)
  manifest.csv                   study, views present, chosen series per view, seconds
  knee_prep.py, spec.json        the exact preprocessing used, and its parameters
"""
import base64, json, os, sys, time
from multiprocessing import Pool

import numpy as np
import pandas as pd

KNEE_PREP_B64 = "__KNEE_PREP_B64__"            # filled in by push_e13.sh
OUT = os.environ.get("OUT_DIR", "/kaggle/working")
os.makedirs(os.path.join(OUT, "cache"), exist_ok=True)
src = base64.b64decode(KNEE_PREP_B64).decode()
open(os.path.join(OUT, "knee_prep.py"), "w").write(src)
sys.path.insert(0, OUT)
import knee_prep                                # noqa: E402  the file just written

ROOT = next(r for r in (os.environ.get("RSNA_COMP_ROOT", ""),
                        "/kaggle/input/competitions/rsna-knee-abnormality-detection",
                        "/kaggle/input/rsna-knee-abnormality-detection")
            if r and os.path.exists(os.path.join(r, "train.csv")))
SPLIT = os.environ.get("E13_SPLIT", "train")
SERIES_ROOT = os.path.join(ROOT, f"{SPLIT}_series")
studies = pd.read_csv(os.path.join(ROOT, f"{SPLIT}.csv"), dtype={"StudyInstanceUID": str})["StudyInstanceUID"].tolist()
ser = pd.read_csv(os.path.join(ROOT, f"{SPLIT}_series.csv"), dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
SER = {k: v.to_dict("records") for k, v in ser.groupby("StudyInstanceUID")}
LIMIT = int(os.environ.get("E13_LIMIT", "0"))
if LIMIT:
    studies = studies[:LIMIT]
# The full cache (~19 GB) is too close to Kaggle's 20 GB output cap, so it is built as
# N_SHARDS kernels, each taking a contiguous block of the UID-sorted study list.
SHARD, N_SHARDS = int(os.environ.get("E13_SHARD", "0")), int(os.environ.get("E13_N_SHARDS", "1"))
studies = sorted(studies)
studies = studies[SHARD * len(studies) // N_SHARDS:(SHARD + 1) * len(studies) // N_SHARDS]


def work(uid):
    t = time.time()
    try:
        vol, mask, chosen = knee_prep.build_study(uid, SER.get(uid, []), SERIES_ROOT)
        np.savez_compressed(os.path.join(OUT, "cache", f"{uid}.npz"), vol=vol, mask=mask)
        return uid, mask.tolist(), chosen, round(time.time() - t, 2), ""
    except Exception as e:
        return uid, [0] * len(knee_prep.VIEWS), [""] * len(knee_prep.VIEWS), round(time.time() - t, 2), f"{type(e).__name__}: {e}"


if __name__ == "__main__":
    T0 = time.time()
    print(f"root {ROOT} | split {SPLIT} | {len(studies)} studies | spec {knee_prep.SPEC}", flush=True)
    rows = []
    with Pool(int(os.environ.get("E13_PROCS", os.cpu_count() or 4))) as pool:
        for i, (uid, mask, chosen, sec, err) in enumerate(pool.imap_unordered(work, studies, chunksize=4), 1):
            rows.append({"StudyInstanceUID": uid, "n_views": sum(mask), **{f"view{v}": m for v, m in enumerate(mask)},
                         **{f"series{v}": s for v, s in enumerate(chosen)}, "seconds": sec, "error": err})
            if err:
                print(f"  ERROR {uid}: {err}", flush=True)
            if i % 200 == 0 or i == len(studies):
                el = time.time() - T0
                print(f"  {i}/{len(studies)} | {el / 60:.1f} min | eta {(len(studies) - i) * el / i / 60:.1f} min", flush=True)
    man = pd.DataFrame(rows).sort_values("StudyInstanceUID")
    man.to_csv(os.path.join(OUT, "manifest.csv"), index=False)
    json.dump({**knee_prep.SPEC, "split": SPLIT, "n_studies": len(man), "minutes": round((time.time() - T0) / 60, 1)},
              open(os.path.join(OUT, "spec.json"), "w"), indent=1)
    size = sum(os.path.getsize(os.path.join(OUT, "cache", f)) for f in os.listdir(os.path.join(OUT, "cache")))
    print(f"DONE {len(man)} studies in {(time.time() - T0) / 60:.1f} min | cache {size / 2**30:.2f} GB | "
          f"errors {int((man.error != '').sum())} | views present per study: {man.n_views.value_counts().sort_index().to_dict()} | "
          f"per view: {[int(man[f'view{v}'].sum()) for v in range(len(knee_prep.VIEWS))]}", flush=True)
