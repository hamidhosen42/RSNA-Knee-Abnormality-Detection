"""CPU gate and builder of the ensemble teacher T2, run locally before any GPU is spent on student4.

T2 = the public 0.956 blend's own form on the TRAIN studies: per finding, a rank blend of the members' percentile ranks
with the blend's weights (Raptor slot 0.5 = 0.65 x Mirror384 + 0.35 x M224 with mirror; R7 0.5 -> 0.325 / 0.175 / 0.5).
Each member is ranked over the studies it scored; per study the weights are renormalised over the members that scored
it.  The blended rank is then mapped onto Mirror384's per-finding marginal (its sorted train probabilities), so the
student targets keep exactly C3's scale (0.25 v0 + 0.75 teacher) and E75 differs from C3 only in the teacher's ORDER.
Decided 2026-10-15, before any teacher ran (validation item T2): a probability average would give R7 an effective share
of 0.31-0.58 instead of 0.5, because the members are calibrated differently (Mirror ~0.45 mean vs v0 ~0.29; R7 lr-prior).

Stop rules (exit 1, nothing written), fixed before seeing the data:
  * a member covers <= 90% of the train studies, or the R7 shards are not exactly 0..N-1;
  * a member's column is (almost) constant (SD < 1e-3);
  * a member's per-finding Spearman vs Mirror384 < 0.2 (a swapped or misaligned column);
  * with --labels: a member's macro AUC vs v0 < 0.85 (Mirror384 alone: 0.9255).
Diagnostics only (they never change T2): per member and finding the mean, SD, AUC vs v0 and mean |member - v0|; each
member's effective share in a probability average; Spearman(probability average, rank blend); gold-58 AUC of every
member and of both T2 forms.

    python gate_ens_teacher.py --teachers DIR [--labels DIR] [--out DIR]
      --teachers DIR: mirror_train_raw.csv, m224_train_raw.csv, r7_train_shard<k>.csv
      --labels DIR:   train.csv, llm_labels_v4_blend.csv, report_labels_v2.csv, labels_llm_gpt56sol.csv,
                      qwen_knee_weak_labels.csv (the student kernel's v0 sources)
    -> OUT/ens_train_raw.csv (StudyInstanceUID + 12 findings, studies no member scored left out) + OUT/ens_gate_report.json
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
WEIGHTS = {"mirror384": 0.325, "m224": 0.175, "r7": 0.5}
MIN_COVER, MIN_SD, MIN_RHO, MIN_AUC = 0.9, 1e-3, 0.2, 0.85


def auc(y, p):
    """ROC AUC (Mann-Whitney); NaN when only one class is present."""
    y = np.asarray(y, float) > 0.5
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(np.asarray(p, float)).rank(method="average").values
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(a, b):
    return float(np.corrcoef(pd.Series(np.asarray(a, float)).rank().values, pd.Series(np.asarray(b, float)).rank().values)[0, 1])


def read_member(paths):
    d = pd.concat([pd.read_csv(p, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[LABELS] for p in paths])
    d = d[~d.index.duplicated()].astype(float)
    d = d[~(d == 0.5).all(axis=1)]          # a Raptor study that failed or was not reached stays 0.5 everywhere
    return d[np.isfinite(d.values).all(axis=1)]


def v0_labels(L, ids):
    """The student kernel's build_labels(): mean of the four public LLM label sources, clipped; gold = train.csv labels."""
    tr = pd.read_csv(os.path.join(L, "train.csv"), dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    src = {k: pd.read_csv(os.path.join(L, f), dtype={"StudyInstanceUID": str}) for k, f in
           (("steven", "llm_labels_v4_blend.csv"), ("pilkwang", "report_labels_v2.csv"), ("lixin", "labels_llm_gpt56sol.csv"))}
    q = pd.read_csv(os.path.join(L, "qwen_knee_weak_labels.csv"), dtype={"StudyInstanceUID": str})
    src["qwen"] = q[["StudyInstanceUID"] + [f"{c}__label" for c in LABELS]].set_axis(["StudyInstanceUID"] + LABELS, axis=1)
    src = {k: v.set_index("StudyInstanceUID")[LABELS].astype(float).reindex(ids) for k, v in src.items()}
    num = sum(v.fillna(0) for v in src.values())
    den = sum(v.notna().astype(float) for v in src.values())
    v0 = (num / den).clip(0.01, 0.99)
    gold = tr[LABELS].reindex(ids)
    return v0, gold, gold.notna().all(axis=1).values


def build_t2(mem, ids, mirror):
    """Rank blend (weights renormalised per study) mapped onto Mirror's per-finding marginal; also the probability mean."""
    n = len(ids)
    num, den = np.zeros((n, len(LABELS))), np.zeros((n, 1))
    pnum = np.zeros((n, len(LABELS)))
    for name, d in mem.items():
        dd = d.reindex(ids)
        ok = dd.notna().all(axis=1).values
        pr = np.zeros((n, len(LABELS)))
        pr[ok] = dd[ok].rank(method="average", pct=True).values
        num += WEIGHTS[name] * pr * ok[:, None]
        pnum += WEIGHTS[name] * np.nan_to_num(dd.values) * ok[:, None]
        den += WEIGHTS[name] * ok[:, None]
    has = den[:, 0] > 0
    blend = num[has] / den[has]
    prob = pnum[has] / den[has]
    q = np.sort(mirror.values, axis=0)
    grid = (np.arange(len(q)) + 0.5) / len(q)
    t2 = np.empty_like(blend)
    for j in range(len(LABELS)):
        r = (pd.Series(blend[:, j]).rank(method="average").values - 0.5) / len(blend)
        t2[:, j] = np.interp(r, grid, q[:, j])
    return has, pd.DataFrame(t2, index=ids[has], columns=LABELS), pd.DataFrame(prob, index=ids[has], columns=LABELS)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--teachers", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    T, out = a.teachers, a.out or a.teachers
    shards = sorted(glob.glob(os.path.join(T, "r7_train_shard*.csv")))
    ks = sorted(int(re.search(r"shard(\d+)\.csv$", p).group(1)) for p in shards)
    stops, rep = [], {"form": "rank blend 0.325/0.175/0.5 (renormalised per study) mapped onto Mirror384's marginal",
                      "weights": WEIGHTS, "members": {}}
    if not shards or ks != list(range(len(ks))):
        stops.append(f"R7 shards are not 0..N-1: {ks}")
    full_mirror = pd.read_csv(os.path.join(T, "mirror_train_raw.csv"), dtype={"StudyInstanceUID": str})
    mem = {"mirror384": read_member([os.path.join(T, "mirror_train_raw.csv")]),
           "m224": read_member([os.path.join(T, "m224_train_raw.csv")]),
           "r7": read_member(shards) if shards else pd.DataFrame(columns=LABELS, dtype=float)}
    if a.labels:
        ids = pd.Index(pd.read_csv(os.path.join(a.labels, "train.csv"), dtype={"StudyInstanceUID": str}).StudyInstanceUID)
        v0, gold, is_gold = v0_labels(a.labels, ids)
    else:
        ids, v0, gold, is_gold = pd.Index(full_mirror.StudyInstanceUID), None, None, None
    rep["train_studies"] = len(ids)
    mir = mem["mirror384"]
    for name, d in mem.items():
        dd = d.reindex(ids)
        ok = dd.notna().all(axis=1).values
        r = {"rows": int(len(d)), "cover": round(float(ok.mean()), 4), "per_finding": {}}
        if r["cover"] <= MIN_COVER:
            stops.append(f"{name} covers only {r['cover']:.3f} of the train studies")
        common = d.index.intersection(mir.index)
        aucs = []
        for c in LABELS:
            f = {"mean": round(float(d[c].mean()), 4) if len(d) else None, "sd": round(float(d[c].std()), 4) if len(d) > 1 else 0.0}
            if not len(d) or f["sd"] < MIN_SD:
                stops.append(f"{name} {c}: constant column (sd {f['sd']})")
            if name != "mirror384" and len(common) > 2:
                f["spearman_vs_mirror"] = round(spearman(d.loc[common, c], mir.loc[common, c]), 4)
                if f["spearman_vs_mirror"] < MIN_RHO:
                    stops.append(f"{name} {c}: Spearman vs Mirror384 {f['spearman_vs_mirror']} (swapped or misaligned?)")
            if v0 is not None:
                w = ok & ~is_gold
                f["auc_vs_v0"] = round(auc(v0[c].values[w], dd[c].values[w]), 4)
                f["mean_abs_vs_v0"] = round(float(np.abs(dd[c].values[w] - v0[c].values[w]).mean()), 4)
                aucs.append(f["auc_vs_v0"])
            r["per_finding"][c] = f
        if v0 is not None:
            r["macro_auc_vs_v0"] = round(float(np.nanmean(aucs)), 4)
            if not r["macro_auc_vs_v0"] >= MIN_AUC:
                stops.append(f"{name}: macro AUC vs v0 {r['macro_auc_vs_v0']} < {MIN_AUC}")
            g = ok & is_gold
            if g.sum() > 10:
                r["gold58_macro_auc"] = round(float(np.nanmean([auc(gold[c].values[g], dd[c].values[g]) for c in LABELS])), 4)
        rep["members"][name] = r
    if stops:
        rep["status"] = "STOP"
        rep["stops"] = stops
        print(json.dumps(rep, indent=1))
        print("ENS_GATE STOP:", *stops, sep="\n  ")
        return 1
    has, t2, prob = build_t2(mem, ids, mir)
    both = pd.concat([m.reindex(ids) for m in mem.values()], axis=1).notna().all(axis=1).values
    rep["t2_rows"] = int(has.sum())
    rep["rows_scored_by_all_members"] = int(both.sum())
    rep["spearman_prob_vs_rank"] = {c: round(spearman(prob.loc[ids[both], c], t2.loc[ids[both], c]), 4) for c in LABELS}
    sd = {n: mem[n].reindex(ids[both]).std() for n in mem}
    rep["effective_share_in_prob_average"] = {c: {n: round(float(WEIGHTS[n] * sd[n][c] / sum(WEIGHTS[k] * sd[k][c] for k in mem)), 3)
                                                  for n in mem} for c in LABELS}
    if v0 is not None:
        for form, t in (("t2_rank_on_mirror", t2), ("t2_prob_average", prob)):
            tt = t.reindex(ids)
            ok = tt.notna().all(axis=1).values
            w = ok & ~is_gold
            rep[form] = {"macro_auc_vs_v0": round(float(np.nanmean([auc(v0[c].values[w], tt[c].values[w]) for c in LABELS])), 4)}
            g = ok & is_gold
            if g.sum() > 10:
                rep[form]["gold58_macro_auc"] = round(float(np.nanmean([auc(gold[c].values[g], tt[c].values[g]) for c in LABELS])), 4)
    os.makedirs(out, exist_ok=True)
    t2.index.name = "StudyInstanceUID"
    t2.reset_index().to_csv(os.path.join(out, "ens_train_raw.csv"), index=False, float_format="%.6g")
    rep["status"] = "PASS"
    open(os.path.join(out, "ens_gate_report.json"), "w").write(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("members", "effective_share_in_prob_average")}, indent=1))
    print("ENS_GATE PASS | T2 rows", int(has.sum()), "->", os.path.join(out, "ens_train_raw.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
