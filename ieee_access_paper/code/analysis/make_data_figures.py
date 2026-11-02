"""Exploratory label analysis for Section III-E of the manuscript (Figs. 6-9) and data_stats.json.

Inputs in data/ (copied from team_shared/esfersami50/nx1-01-consensus-labels/, the team's label notebook):
  - labels_consensus.parquet   soft consensus labels of the 4,349 non-reference training studies (label set B of Table 2)
  - label_benchmark.csv        per-source, per-finding AUC of public report-label tables against the 58 expert-labelled
                               studies, as computed by that notebook (gold_copy = the table reproduces the expert labels)
  - gold_effv2s_f0_swa.npz     its 'truth' array holds the expert labels of the 58 studies

    uv run --no-project --with numpy --with pandas --with pyarrow --with matplotlib python -I \
        ieee_access_paper/code/analysis/make_data_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIG = os.path.join(os.path.dirname(os.path.dirname(HERE)), "manuscript", "figures")
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SHORT = ["ACL", "MCL", "Med. men.", "Lat. men.", "Med. OA", "Lat. OA", "PF OA", "Effusion", "Synovitis",
         "Baker's", "Contusion", "Fracture"]
# ordinal ramp (validated: monotone, light end 2.06:1), sequential blue, diverging blue <-> grey <-> red
ORD = ["#86b6ef", "#3987e5", "#184f95"]
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ef8f8e", "#b8302f"])
INK, MUTED = "#1f1f1e", "#6b6b66"

plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "axes.linewidth": 0.6, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

stats = {}
lab = pd.read_parquet(os.path.join(DATA, "labels_consensus.parquet")).set_index("StudyInstanceUID")[LABELS]
assert lab.shape == (4349, 12) and lab.notna().all().all()
y = lab.values
pos, unc = y >= 0.7, (y > 0.25) & (y < 0.7)
neg = ~pos & ~unc
stats["n_studies"] = int(len(lab))
stats["positive_rate_ge_0.7"] = dict(zip(LABELS, pos.mean(0).round(4).tolist()))
stats["uncertain_rate_0.25_0.7"] = dict(zip(LABELS, unc.mean(0).round(4).tolist()))

# ---- Fig. A: negative / uncertain / positive share of each finding (100% stacked, ordinal) -----------------------
order = np.argsort(pos.mean(0))
fig, ax = plt.subplots(figsize=(3.45, 2.3))
left = np.zeros(12)
for share, col, name in ((neg.mean(0), ORD[0], "negative (< 0.25)"), (unc.mean(0), ORD[1], "uncertain (0.25-0.7)"),
                         (pos.mean(0), ORD[2], "positive (>= 0.7)")):
    ax.barh(np.arange(12), share[order], left=left[order], color=col, height=0.72, edgecolor="#fcfcfb",
            linewidth=0.8, label=name)
    left += share
for i, j in enumerate(order):
    ax.text(1.01, i, f"{pos.mean(0)[j]:.0%}", va="center", fontsize=6, color=INK)
ax.set_yticks(np.arange(12))
ax.set_yticklabels([SHORT[j] for j in order])
ax.set_xlim(0, 1)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
ax.set_xlabel("Share of 4,349 studies (consensus report label)")
ax.text(1.01, 12.0, "pos.", fontsize=6, color=MUTED)
ax.legend(fontsize=5.6, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.45, -0.2))
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(FIG, "label_shares.pdf"))

# ---- Fig. B: co-occurrence (Pearson correlation of the soft labels) ------------------------------------------------
corr = np.corrcoef(y.T)
stats["corr_top_pairs"] = sorted(((round(float(corr[a, b]), 3), LABELS[a], LABELS[b]) for a in range(12)
                                  for b in range(a + 1, 12)), reverse=True)[:6]
stats["corr_min_pair"] = min((round(float(corr[a, b]), 3), LABELS[a], LABELS[b]) for a in range(12) for b in range(a + 1, 12))
fig, ax = plt.subplots(figsize=(3.45, 3.0))
m = np.where(np.tril(np.ones((12, 12), bool), -1), corr, np.nan)[1:, :11]   # lower triangle without the empty row/column
im = ax.imshow(m, cmap=DIV, vmin=-0.6, vmax=0.6)
for a in range(1, 12):
    for b in range(a):
        v = corr[a, b]
        ax.text(b, a - 1, f"{v:.2f}".replace("0.", "."), ha="center", va="center", fontsize=4.6,
                color="white" if abs(v) > 0.4 else INK)
ax.set_xticks(range(11))
ax.set_xticklabels(SHORT[:11], rotation=60, ha="right", fontsize=5.6)
ax.set_yticks(range(11))
ax.set_yticklabels(SHORT[1:], fontsize=5.6)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.ax.tick_params(labelsize=5.5)
cb.set_label("Pearson r (soft labels)", fontsize=6)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(FIG, "label_cooccurrence.pdf"))

# ---- Fig. C: positive findings per study, report-labelled set vs expert-labelled set -------------------------------
g = np.load(os.path.join(DATA, "gold_effv2s_f0_swa.npz"), allow_pickle=True)["truth"] > 0.5
k_rep, k_gold = pos.sum(1), g.sum(1)
stats["positives_per_study_report_mean"] = float(k_rep.mean())
stats["positives_per_study_report_zero_share"] = float((k_rep == 0).mean())
stats["positives_per_study_gold_mean"] = float(k_gold.mean())
stats["positives_per_study_gold_zero_share"] = float((k_gold == 0).mean())
bins = np.arange(0, 13)
fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.6), sharey=True)
for ax, k, col, title in ((axs[0], k_rep, "#2a78d6", "4,349 studies, report labels"),
                          (axs[1], k_gold, "#eb6834", "58 studies, expert labels")):
    h = np.bincount(k, minlength=13)[:13] / len(k)
    ax.bar(bins, h, color=col, width=0.8, edgecolor="#fcfcfb", linewidth=0.8)
    ax.axvline(k.mean(), color=INK, lw=0.7, ls="--")
    ax.text(k.mean() + 0.3, max(h) * 0.95, f"mean {k.mean():.2f}", fontsize=5.6)
    ax.set_title(title, fontsize=6.5)
    ax.set_xlabel("positive findings per study", fontsize=6)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
axs[0].set_ylabel("share of studies", fontsize=6)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(FIG, "positives_per_study.pdf"))

# ---- Fig. D: agreement of public report labellers with the expert labels ------------------------------------------
b = pd.read_csv(os.path.join(DATA, "label_benchmark.csv"))
stats["gold_copy_tables"] = b.loc[b.gold_copy, "source"].tolist()
b = b[~b.gold_copy & b.gold_macro.notna()].copy()
cols = [f"auc_{c}" for c in LABELS]
b = b.drop_duplicates(subset=cols)                          # identical tables shipped under two names
V0 = {"rsna-knee-llm-report-labels/llm_labels_v4_blend", "rsna-knee-llm-labels/report_labels_v2",
      "rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol", "rsna-knee-abnormality-qwen3-8b-weak-labels/qwen_knee_weak_labels"}
missing = V0 - set(b.source)
assert not missing, missing
b = b.sort_values("gold_macro", ascending=False)
names = [("* " if s in V0 else "") + s.replace("rsna-knee-", "") for s in b.source]
A = b[cols].values
stats["benchmark_sources"] = int(len(b))
stats["benchmark_macro_range"] = [float(b.gold_macro.min()), float(b.gold_macro.max())]
stats["benchmark_per_finding_median"] = dict(zip(LABELS, np.median(A, 0).round(3).tolist()))
stats["v0_sources_macro"] = {s: float(v) for s, v in zip(b.source, b.gold_macro) if s in V0}
fig, ax = plt.subplots(figsize=(7.0, 3.3))
im = ax.imshow(np.c_[A, b.gold_macro.values], cmap=SEQ, vmin=0.6, vmax=1.0, aspect="auto")
for i in range(A.shape[0]):
    for j in range(13):
        v = A[i, j] if j < 12 else b.gold_macro.values[i]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.2, color="white" if v > 0.86 else INK,
                fontweight="bold" if j == 12 else "normal")
ax.axvline(11.5, color="#fcfcfb", lw=2)
ax.set_xticks(range(13))
ax.set_xticklabels(SHORT + ["macro"], fontsize=6, rotation=35, ha="right")
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=5.6)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
cb.ax.tick_params(labelsize=5.5)
cb.set_label("AUC vs. expert labels (58 studies)", fontsize=6)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(FIG, "labeller_vs_expert.pdf"))

with open(os.path.join(HERE, "data_stats.json"), "w") as fh:
    json.dump(stats, fh, indent=1)
print(json.dumps(stats, indent=1))
