"""Frozen 5-fold split for our leg (folds_v1).

Universe: the 4,349 non-gold studies that have soft labels in dreaddevelopment/rsna-knee-labels.
Rules: gold-58 excluded; studies whose normalized report text is identical share a fold;
iterative stratification on soft >= 0.5 over the 12 findings. Fully deterministic (no set or
hash-order dependence), so the training kernel recomputes it and checks FOLDS_V1_SHA256.

    python folds.py train.csv labels_llm_soft.csv folds_v1.json
"""
import csv, hashlib, json, random, re, sys, unicodedata
from collections import defaultdict

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
K, SEED = 5, 20260927
FOLDS_V1_SHA256 = "f4292c4a2dcaed3c43aa50ab5741f485c1f95ab064ca58fca9986ecc62e6a267"


def _norm_report(t):
    t = unicodedata.normalize("NFKD", t or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def gold_ids(train_rows):
    return sorted(r["StudyInstanceUID"] for r in train_rows
                  if all(r.get(c) not in ("", None) for c in LAB))


def make_folds(train_rows, soft_rows):
    """train_rows / soft_rows: lists of dicts (csv.DictReader rows or DataFrame records)."""
    gold = set(gold_ids(train_rows))
    report = {str(r["StudyInstanceUID"]): _norm_report(str(r.get("Report") or "")) for r in train_rows}
    soft = {str(r["StudyInstanceUID"]): r for r in soft_rows}
    ids = sorted(u for u in soft if u not in gold)
    groups = defaultdict(list)
    for u in ids:
        groups[hashlib.sha1(report.get(u, u).encode()).hexdigest()].append(u)
    y = {u: [1 if float(soft[u][c]) >= 0.5 else 0 for c in LAB] for u in ids}
    gl = {g: [sum(y[u][j] for u in m) for j in range(len(LAB))] for g, m in groups.items()}
    tot = [sum(v[j] for v in gl.values()) for j in range(len(LAB))]
    need = [[tot[j] / K for j in range(len(LAB))] for _ in range(K)]
    cap = [len(ids) / K] * K
    rng = random.Random(SEED)
    fold_of, left = {}, set(groups)
    while left:
        order = sorted(left)
        rem = [sum(gl[g][j] for g in order) for j in range(len(LAB))]
        cand = [j for j in range(len(LAB)) if rem[j] > 0]
        j = min(cand, key=lambda j: (rem[j], j)) if cand else None
        pool = [g for g in order if j is None or gl[g][j] > 0]
        rng.shuffle(pool)
        for g in sorted(pool, key=lambda g: -len(groups[g])):  # stable sort keeps the shuffle
            key = (lambda f: (cap[f], -f)) if j is None else (lambda f: (need[f][j], cap[f], -f))
            f = max(range(K), key=key)
            fold_of[g] = f
            left.discard(g)
            cap[f] -= len(groups[g])
            for jj in range(len(LAB)):
                need[f][jj] -= gl[g][jj]
    return {u: fold_of[g] for g, m in groups.items() for u in m}, sorted(gold)


def folds_sha256(folds):
    return hashlib.sha256(json.dumps(sorted(folds.items())).encode()).hexdigest()


if __name__ == "__main__":
    train_rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
    soft_rows = list(csv.DictReader(open(sys.argv[2], encoding="utf-8")))
    folds, gold = make_folds(train_rows, soft_rows)
    digest = folds_sha256(folds)
    json.dump({"version": "folds_v1", "k": K, "seed": SEED, "sha256": digest,
               "universe": "4349 non-gold studies of dreaddevelopment/rsna-knee-labels "
                           "labels_llm_soft (dataset v3)",
               "rules": "gold-58 excluded; identical normalized reports share a fold; "
                        "iterative stratification on soft>=0.5",
               "gold_ids": gold, "folds": dict(sorted(folds.items()))},
              open(sys.argv[3], "w"), indent=0)
    sizes = [sum(1 for v in folds.values() if v == f) for f in range(K)]
    print(f"{len(folds)} studies, fold sizes {sizes}, gold {len(gold)}, sha256 {digest}")
