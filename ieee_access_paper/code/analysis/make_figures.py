"""Statistics and figures for the IEEE Access manuscript (ieee_access_paper/manuscript/main.tex).

Inputs, all in data/ next to this script (copied from ours_leg/results/):
  - gold_effv2s_f{0,1}_swa.npz, oof_effv2s_f{0,1}_swa.npz   E11 EfficientNetV2-S, folds 0+1 SWA (from train_v3/)
  - gold_effv2s_cons_f{0,1}_swa.npz                          E03b, consensus labels (from e03b_train_v1/)
Stand-alone test scores in LEG_ALONE are transcribed from audit/07_experiments.csv.

Outputs: ../../manuscript/figures/{leg_alone,yardstick_noise}.pdf and stats.json next to this script.

    uv run --no-project --with numpy --with matplotlib python -I ieee_access_paper/code/analysis/make_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIG = os.path.join(os.path.dirname(os.path.dirname(HERE)), "manuscript", "figures")
os.makedirs(FIG, exist_ok=True)
LABELS = ["ACL", "MCL", "Med. Men.", "Lat. Men.", "Med. OA", "Lat. OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
RNG = np.random.default_rng(20261103)
N_BOOT = 2000

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "axes.linewidth": 0.6})


def auc(y, p):
    """Mann-Whitney ROC AUC with average ranks for ties; nan if one class is missing."""
    y = np.asarray(y) > 0.5
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    sp = np.asarray(p)[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def macro(y, p):
    return float(np.nanmean([auc(y[:, i], p[:, i]) for i in range(y.shape[1])]))


def load(path):
    d = np.load(path, allow_pickle=True)
    return d["pred"].astype(np.float64), d["truth"].astype(np.float64), d["ids"]


def gold_pair(arm):
    """Mean of the fold-0 and fold-1 SWA models on gold-58 (same study order checked)."""
    p0, y0, i0 = load(f"{DATA}/gold_{arm}_f0_swa.npz")
    p1, y1, i1 = load(f"{DATA}/gold_{arm}_f1_swa.npz")
    assert (i0 == i1).all() and np.allclose(y0, y1)
    return (p0 + p1) / 2, y0


def boot_ci(y, p, n=N_BOOT):
    vals = []
    for _ in range(n):
        idx = RNG.integers(0, len(y), len(y))
        vals.append(macro(y[idx], p[idx]))
    vals = np.array(vals)
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5)), vals


stats = {}

# 1. gold-58: prevalence, point estimate and bootstrap CI of the E11 and E03b legs (diagnostic set)
pg, yg = gold_pair("effv2s")
pc, yc = gold_pair("effv2s_cons")
assert np.allclose(yg, yc)
stats["gold58_n"] = int(len(yg))
stats["gold58_positives"] = dict(zip(LABELS, yg.sum(0).astype(int).tolist()))
stats["gold58_mean_positive_findings_per_study"] = float(yg.sum(1).mean())
lo, hi, boot_e11 = boot_ci(yg, pg)
stats["gold58_E11"] = {"macro": macro(yg, pg), "ci95": [lo, hi]}
lo, hi, boot_e03 = boot_ci(yc, pc)
stats["gold58_E03b"] = {"macro": macro(yc, pc), "ci95": [lo, hi]}
# paired bootstrap of the difference E03b - E11 on the same resamples
diffs = []
for _ in range(N_BOOT):
    idx = RNG.integers(0, len(yg), len(yg))
    diffs.append(macro(yc[idx], pc[idx]) - macro(yg[idx], pg[idx]))
diffs = np.array(diffs)
stats["gold58_E03b_minus_E11"] = {"point": stats["gold58_E03b"]["macro"] - stats["gold58_E11"]["macro"],
                                  "ci95": [float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))]}

# 2. resolution of a ~385-study leaderboard split: SD of macro AUC over random subsamples of a fixed predictor.
# Proxy: the E11 OOF predictions (folds 0+1, n = 1740) against the binarised dd soft labels; the soft-label band
# 0.25-0.7 is kept, so this is a conservative (noisier-label) proxy, not the hidden test distribution.
po, yo = [], []
for f in (0, 1):
    p, y, _ = load(f"{DATA}/oof_effv2s_f{f}_swa.npz")
    po.append(p)
    yo.append(y)
po, yo = np.concatenate(po), (np.concatenate(yo) > 0.5).astype(float)
stats["oof_proxy_n"] = int(len(yo))
stats["oof_proxy_macro"] = macro(yo, po)
n_pub = round(0.3 * 1284)
sub = np.array([macro(yo[idx], po[idx]) for idx in (RNG.choice(len(yo), n_pub, replace=False) for _ in range(N_BOOT))])
stats["public_split_size"] = n_pub
stats["public_split_macro_sd"] = float(sub.std())
# two predictors whose test AUCs differ by d: how often does a 385-study split rank them wrongly? Paired proxy: the
# E11 predictor vs the same predictor with rank noise added (calibrated to lose ~0.001-0.005 macro AUC on the full OOF).
flip = {}
for scale in (0.008, 0.012, 0.018, 0.03):
    noisy = po + RNG.normal(0, scale, po.shape)
    full_gap = macro(yo, po) - macro(yo, noisy)
    wrong = 0
    for _ in range(1000):
        idx = RNG.choice(len(yo), n_pub, replace=False)
        wrong += macro(yo[idx], noisy[idx]) >= macro(yo[idx], po[idx])
    flip[f"{full_gap:.4f}"] = wrong / 1000
stats["public_split_wrong_order_rate_by_true_gap"] = flip

# per-finding OOF of E11 (pooled folds 0+1)
stats["E11_oof_per_finding"] = dict(zip(LABELS, [float(auc(yo[:, i], po[:, i])) for i in range(12)]))

# 3. optional: paired gold-58 bootstrap for the principal readers.  Put the training kernels' gold_arm{A,B,C,C3}.csv
# (index StudyInstanceUID, the 12 finding columns, probabilities) into data/; the truth comes from the E11 gold file.
_, _, gold_ids = load(f"{DATA}/gold_effv2s_f0_swa.npz")
row = {str(u): k for k, u in enumerate(gold_ids)}
arms = {}
for arm in ("A", "B", "C", "C3"):
    path = f"{DATA}/gold_arm{arm}.csv"
    if os.path.exists(path):
        import csv
        with open(path) as fh:
            rows = list(csv.reader(fh))
        pred = np.full(yg.shape, np.nan)
        for r in rows[1:]:
            pred[row[r[0]]] = [float(v) for v in r[1:13]]
        assert not np.isnan(pred).any(), f"{path} does not cover all 58 gold studies"
        arms[arm] = pred
for a, b in (("B", "A"), ("C", "B"), ("C3", "A")):
    if a in arms and b in arms:
        d = []
        for _ in range(N_BOOT):
            idx = RNG.integers(0, len(yg), len(yg))
            d.append(macro(yg[idx], arms[a][idx]) - macro(yg[idx], arms[b][idx]))
        stats[f"gold58_{a}_minus_{b}"] = {"point": macro(yg, arms[a]) - macro(yg, arms[b]),
                                          "ci95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]}

with open(os.path.join(HERE, "stats.json"), "w") as fh:
    json.dump(stats, fh, indent=1)
print(json.dumps(stats, indent=1))

# ---------------------------------------------------------------------------------------------------------------
# Figure: leg-alone public LB and gold-58 for every study-level reader we trained (audit/07_experiments.csv)
LEG_ALONE = [  # name, public LB (leg alone), gold-58 (diagnostic), family
    ("E11a", 0.896, 0.9000, "own"),
    ("E03b", 0.899, 0.8826, "own"),
    ("A", 0.933, 0.9066, "reader"),
    ("B2", 0.938, 0.9065, "reader"),
    ("B", 0.939, 0.9037, "reader"),
    ("C", 0.948, 0.9227, "student"),
    ("C2", 0.948, 0.9187, "student"),
    ("C3", 0.949, 0.9266, "student"),
]
col = {"own": "#9aa5b1", "reader": "#5b8db8", "student": "#d0743c"}
fig, ax = plt.subplots(figsize=(3.45, 2.2))
x = np.arange(len(LEG_ALONE))
ax.bar(x, [r[1] for r in LEG_ALONE], color=[col[r[3]] for r in LEG_ALONE], width=0.68, zorder=2)
for xi, r in zip(x, LEG_ALONE):   # bars that reach the teacher line get their value inside the bar
    inside = r[1] > 0.945
    ax.text(xi, r[1] - 0.0012 if inside else r[1] + 0.0012, f"{r[1]:.3f}", ha="center", va="top" if inside else "bottom",
            fontsize=6, color="white" if inside else "black")
ax.axhline(0.949, color="#444", lw=0.7, ls="--", zorder=1)
ax.text(-0.3, 0.9505, "Mirror teacher alone (0.949)", ha="left", va="bottom", fontsize=6)
ax.set_xticks(x)
ax.set_xticklabels([r[0] for r in LEG_ALONE], fontsize=6.5)
ax.set_ylim(0.885, 0.962)
ax.set_ylabel("Development AUC (stand-alone)")
ax.grid(axis="y", lw=0.3, color="#ddd", zorder=0)
from matplotlib.patches import Patch  # noqa: E402
ax.legend(handles=[Patch(color=col["own"], label="EfficientNetV2-S (Sec. VI-E)"), Patch(color=col["reader"], label="2.5D reader"),
                   Patch(color=col["student"], label="distilled student")], fontsize=5.8, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.62))
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(FIG, "leg_alone.pdf"))

# Figure: noise of the two available yardsticks
fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.6))
axs[0].hist(boot_e11, bins=40, color="#9aa5b1")
axs[0].axvline(stats["gold58_E11"]["macro"], color="#d0743c", lw=0.8)
axs[0].set_title("gold-58 bootstrap (E11)", fontsize=6.5)
axs[0].set_xlabel("macro AUC", fontsize=6)
axs[1].hist(sub, bins=40, color="#5b8db8")
axs[1].axvline(stats["oof_proxy_macro"], color="#d0743c", lw=0.8)
axs[1].set_title(f"{n_pub}-study subsamples (OOF proxy)", fontsize=6.5)
axs[1].set_xlabel("macro AUC", fontsize=6)
for a in axs:
    a.set_yticks([])
    a.tick_params(labelsize=5.5)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(FIG, "yardstick_noise.pdf"))
print("figures written to", FIG)
