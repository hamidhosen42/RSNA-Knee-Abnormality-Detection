#!/usr/bin/env python3
"""E11 diagnosis: does our test-time DICOM rebuild reproduce the training corpus?

E11 (0.943 base + our EfficientNetV2-S leg at w=0.30) scored 0.935 on the public LB. The leg was
trained on dreaddevelopment's precomputed 44x336x336 corpus but, at test time, infer_kernel.py
rebuilds each study from raw DICOM with its own copy of the corpus recipe. If that rebuild differs
from the corpus, the model sees out-of-distribution inputs only on the test set.

This CPU script answers that on training studies, where both paths exist:
  1. pixels:      rebuild a sample (+ all gold studies) with infer_kernel.build_study, VERBATIM
                  (its source is embedded at push time), and compare slot by slot with the corpus;
  2. variants:    if they differ, try plausible alternative recipes to find what the corpus did;
  3. predictions: score the same studies with our checkpoints through both paths and compare
                  (rank agreement per finding, gold-58 AUC as a diagnostic only).

Inputs: competition data, knee-raptor-corpus (+ -ext), and the E11 training kernel output (v3).
Outputs (/kaggle/working): verify_report.json, verify_per_study.csv, verify_variants.csv.
"""
import base64, json, os, random, time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

T0 = time.time()
INFER_KERNEL_B64 = "__INFER_KERNEL_B64__"      # filled in by kaggle/push.sh verify
ik = {"__name__": "infer_kernel"}
exec(compile(base64.b64decode(INFER_KERNEL_B64).decode(), "infer_kernel.py", "exec"), ik)

LAB = ik["LAB"]
N_SAMPLE = int(os.environ.get("VERIFY_N", "160"))
N_VARIANT = int(os.environ.get("VERIFY_N_VARIANT", "60"))
N_PRED = int(os.environ.get("VERIFY_N_PRED", "60"))   # random studies scored (all gold studies are always scored)
SEED = 20261002
OUT = "/kaggle/working"
_DICOM_TREES = {"train_series", "test_series", "train_images", "test_images"}


def log(msg):
    print(f"[verify {time.time() - T0:7.1f}s] {msg}", flush=True)


def find(root, name, max_depth=6):
    base = root.rstrip(os.sep).count(os.sep)
    for d, dirs, fs in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in _DICOM_TREES and d.count(os.sep) - base < max_depth)
        if name in fs:
            return os.path.join(d, name)
    return None


# --------------------------------------------------------------------------------------------- data
comp = next(p for p in ["/kaggle/input/competitions/rsna-knee-abnormality-detection",
                        "/kaggle/input/rsna-knee-abnormality-detection"] if os.path.exists(p + "/train.csv"))
series_dir = comp + ("/train_series" if os.path.isdir(comp + "/train_series") else "/train_images")
train = pd.read_csv(comp + "/train.csv", dtype={"StudyInstanceUID": str})
ser = pd.read_csv(comp + "/train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
rows_of = {k: v.to_dict("records") for k, v in ser.groupby("StudyInstanceUID")}
gold = train.dropna(subset=LAB)
gold_ids = gold.StudyInstanceUID.tolist()

parts = []
for vols_n, masks_n, ids_n in [("all_vols.npy", "all_masks.npy", "all_ids.npy"),
                               ("extra_vols.npy", "extra_masks.npy", "extra_ids.npy")]:
    paths = [find("/kaggle/input", n) for n in (vols_n, masks_n, ids_n)]
    assert all(paths), f"corpus file missing: {vols_n}"
    parts.append((np.load(paths[0], mmap_mode="r"), np.load(paths[1]), [str(x) for x in np.load(paths[2], allow_pickle=True)]))
where = {}
for pi, (_, _, ids) in enumerate(parts):
    for r, u in enumerate(ids):
        where[u] = (pi, r)
log(f"corpus studies {len(where)} | train studies {len(train)} | gold {len(gold_ids)} in corpus: {sum(u in where for u in gold_ids)}")


def corpus_row(uid):
    pi, r = where[uid]
    return np.asarray(parts[pi][0][r]), np.asarray(parts[pi][1][r])


rng = random.Random(SEED)
pool = sorted(u for u in where if u not in set(gold_ids) and u in rows_of)
sample = sorted(set(rng.sample(pool, min(N_SAMPLE, len(pool))))) + [u for u in gold_ids if u in where]
SPEC = None   # taken from the checkpoint below, i.e. exactly what infer_kernel used

# ------------------------------------------------------------------------------------- checkpoints
ckpts = ik["find_checkpoints"](["/kaggle/input"], "*_swa.pt")
assert ckpts, "attach hosen42/rsna-knee-ours-leg-train-effv2s v3"
models, res, k_eval, mean, std = [], None, None, None, None
for p in ckpts:
    ck = torch.load(p, map_location="cpu", weights_only=False)
    SPEC = SPEC or ck["corpus"]
    res, k_eval = int(ck["res"]), int(ck["k_eval"])
    mean = torch.tensor(ck["norm_mean"]).view(1, 3, 1, 1)
    std = torch.tensor(ck["norm_std"]).view(1, 3, 1, 1)
    bb = ik["build_backbone"](ck["arch"])
    m = ik["RaptorClassifier"](bb, F_dim=bb.num_features)
    m.load_state_dict(ck["model"], strict=True)
    models.append(m.eval())
    log(f"loaded {os.path.basename(p)} fold {ck.get('fold')} res {res} k_eval {k_eval}")
torch.set_num_threads(max(1, os.cpu_count() or 1))
SLOT_EDGES = np.cumsum([0] + [int(s[2]) for s in SPEC["slots"]])
log(f"spec {json.dumps(SPEC)} | slot edges {SLOT_EDGES.tolist()} | sample {len(sample)}")


@torch.no_grad()
def predict(vol, mask):
    D = vol.shape[0]
    cs = [max(1, min(c, D - 2)) for c in ik["eval_centers"](mask, D, k_eval)]
    wins = np.stack([np.stack([vol[c - 1], vol[c], vol[c + 1]], 0) for c in cs], 0)
    x = torch.from_numpy(wins).float().div_(255.0)
    if x.shape[-1] != res:
        x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
    x = ((x - mean) / std).unsqueeze(0)
    return np.mean([torch.sigmoid(m(x))[0].numpy() for m in models], axis=0)


# ------------------------------------------------------------------- 1+3. pixels and predictions
recs, p_corpus, p_dicom = [], {}, {}
for n, uid in enumerate(sample, 1):
    cv, cm = corpus_row(uid)
    rec = {"uid": uid, "gold": uid in set(gold_ids)}
    try:
        dv, dm = ik["build_study"](uid, rows_of.get(uid, []), series_dir, SPEC)
    except Exception as e:                                    # what the test-time leg would do: 0.5
        rec["error"] = f"{type(e).__name__}: {e}"
        recs.append(rec)
        continue
    for s in range(len(SPEC["slots"])):
        a, b = SLOT_EDGES[s], SLOT_EDGES[s + 1]
        both = (cm[a:b] > 0) & (dm[a:b] > 0)
        rec[f"s{s}_mask_eq"] = bool((cm[a:b] > 0).tolist() == (dm[a:b] > 0).tolist())
        rec[f"s{s}_mad"] = float(np.abs(cv[a:b][both].astype(np.int16) - dv[a:b][both].astype(np.int16)).mean()) if both.any() else np.nan
        rec[f"s{s}_exact"] = bool(both.any() and np.array_equal(cv[a:b][both], dv[a:b][both]))
    rec["mad_all"] = float(np.abs(cv.astype(np.int16) - dv.astype(np.int16)).mean())
    if rec["gold"] or n <= N_PRED:
        p_corpus[uid], p_dicom[uid] = predict(cv, cm), predict(dv, dm)
    recs.append(rec)
    if n % 20 == 0 or n == len(sample):
        log(f"{n}/{len(sample)} studies")
per = pd.DataFrame(recs)
per.to_csv(f"{OUT}/verify_per_study.csv", index=False)

ok = [u for u in sample if u in p_dicom]   # scored studies
pc = pd.DataFrame([p_corpus[u] for u in ok], index=ok, columns=LAB)
pd_ = pd.DataFrame([p_dicom[u] for u in ok], index=ok, columns=LAB)
agree = {t: float(pc[t].rank().corr(pd_[t].rank())) for t in LAB}
report = {
    "n_studies": len(sample), "n_scored": len(ok), "n_errors": int(per.get("error", pd.Series(dtype=str)).notna().sum()),
    "pixel": {
        "studies_all_slots_exact": float(np.mean([all(r.get(f"s{s}_exact", False) or np.isnan(r.get(f"s{s}_mad", np.nan))
                                                      for s in range(len(SPEC["slots"]))) for r in recs if "error" not in r])),
        "mad_all_median": float(per["mad_all"].median()), "mad_all_p90": float(per["mad_all"].quantile(.9)),
        "per_slot": {f"slot{s}:{SPEC['slots'][s][0]}/{SPEC['slots'][s][1]}": {
            "mask_equal": float(per[f"s{s}_mask_eq"].mean()), "exact": float(per[f"s{s}_exact"].mean()),
            "mad_median": float(per[f"s{s}_mad"].median())} for s in range(len(SPEC["slots"]))},
    },
    "prediction_rank_agreement_corpus_vs_dicom": agree,
    "prediction_rank_agreement_macro": float(np.mean(list(agree.values()))),
}
g = [u for u in gold_ids if u in p_dicom]
if g:
    from sklearn.metrics import roc_auc_score
    y = gold.set_index("StudyInstanceUID").loc[g, LAB].astype(int)
    report["gold58_auc_diagnostic"] = {
        "corpus_path": float(np.mean([roc_auc_score(y[t], pc.loc[g, t]) for t in LAB])),
        "dicom_path": float(np.mean([roc_auc_score(y[t], pd_.loc[g, t]) for t in LAB])), "n": len(g)}
log("REPORT " + json.dumps(report, indent=1))


# ------------------------------------------------------------------------------- 2. recipe variants
def build_variant(uid, order_desc=False, clip=(2.0, 98.0), pick="first", span_round=False):
    """Independent re-implementation; the default arguments must equal build_study (checked)."""
    img, crop_mm = int(SPEC["img"]), float(SPEC["crop_mm"])
    lo_f, hi_f = SPEC["span"]
    total = int(SLOT_EDGES[-1])
    vol, used, idx = np.zeros((total, img, img), np.uint8), set(), 0
    rows = rows_of.get(uid, [])
    for plane, fluid, k in SPEC["slots"]:
        k, fluid = int(k), int(fluid)
        cands = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
        if fluid in (0, 1):
            pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
            cands = pref or cands
        if pick == "most_slices" and cands:
            cands = sorted(cands, key=lambda r: -len(os.listdir(f"{series_dir}/{uid}/{r['SeriesInstanceUID']}")))
        r = cands[0] if cands else None
        if r is None:
            idx += k
            continue
        used.add(r["SeriesInstanceUID"])
        files, med_ps = ik["order_and_meta"](f"{series_dir}/{uid}/{r['SeriesInstanceUID']}")
        if order_desc:
            files = files[::-1]
        if not files:
            idx += k
            continue
        n = len(files)
        lo = int(round(n * lo_f)) if span_round else int(n * lo_f)
        hi = max((int(round(n * hi_f)) if span_round else int(n * hi_f)) - 1, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        arrs = []
        for p in picks:
            try:
                arrs.append((ik["read_px"](files[min(int(p), n - 1)][0]), files[min(int(p), n - 1)][1]))
            except Exception:
                arrs.append((None, med_ps))
        valid = [a for a, _ in arrs if a is not None]
        loq, hiq = np.percentile(np.concatenate([a.ravel() for a in valid]), list(clip)) if valid else (0.0, 1.0)
        for a, ps in arrs:
            if a is not None:
                a = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
                vol[idx] = (ik["mm_crop_resize"](a, ps if ps > 0 else med_ps, crop_mm, img) * 255).astype(np.uint8)
            idx += 1
    return vol


VARIANTS = {"as_infer_kernel": {}, "order_desc": {"order_desc": True}, "clip_1_99": {"clip": (1.0, 99.0)},
            "clip_0.5_99.5": {"clip": (0.5, 99.5)}, "minmax": {"clip": (0.0, 100.0)},
            "pick_most_slices": {"pick": "most_slices"}, "span_round": {"span_round": True}}
vrec = []
rebuilt = per.loc[per["error"].isna(), "uid"].tolist() if "error" in per else per["uid"].tolist()
for uid in rebuilt[:N_VARIANT]:
    cv, _ = corpus_row(uid)
    for name, kw in VARIANTS.items():
        try:
            v = build_variant(uid, **kw)
        except Exception as e:
            vrec.append({"uid": uid, "variant": name, "error": str(e)})
            continue
        row = {"uid": uid, "variant": name}
        for s in range(len(SPEC["slots"])):
            a, b = SLOT_EDGES[s], SLOT_EDGES[s + 1]
            row[f"s{s}_mad"] = float(np.abs(cv[a:b].astype(np.int16) - v[a:b].astype(np.int16)).mean())
        vrec.append(row)
vdf = pd.DataFrame(vrec)
vdf.to_csv(f"{OUT}/verify_variants.csv", index=False)
summary = vdf.groupby("variant")[[c for c in vdf.columns if c.endswith("_mad")]].median().round(3)
report["variants_median_mad_per_slot"] = summary.to_dict(orient="index")
report["variant_reimplementation_check"] = "see as_infer_kernel row: must equal the build_study MADs above"
json.dump(report, open(f"{OUT}/verify_report.json", "w"), indent=1)
log("VARIANTS (median per-slot MAD vs corpus; lower = closer)\n" + summary.to_string())
log("done")
