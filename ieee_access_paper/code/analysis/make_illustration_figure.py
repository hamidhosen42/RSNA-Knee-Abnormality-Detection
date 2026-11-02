"""Illustrative figure of knee anatomy and the MRI planes/sequences the reader's slots correspond to (Section III-A).

The images are NOT from the RSNA dataset (whose rules forbid publishing it); they are openly licensed Wikimedia Commons
images, listed with authors and licences in data/illustrative/ATTRIBUTION.md.

    uv run --no-project --with numpy --with matplotlib --with pillow python -I \
        ieee_access_paper/code/analysis/make_illustration_figure.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "illustrative")
FIG = os.path.join(os.path.dirname(os.path.dirname(HERE)), "manuscript", "figures")
plt.rcParams.update({"font.family": "serif", "font.size": 7, "pdf.fonttype": 42})

# file, crop box (left, top, right, bottom) or None, panel title, reader slot
PANELS = [
    ("Blausen_0597_KneeAnatomy_Side.png", None, "(a) Anatomy (lateral)", "illustration"),
    ("Knee_MRI_PD_TSE_FS_Sagittal.jpg", (60, 0, 902, 842), "(b) Sagittal PD FS", "slot: Sag FS"),
    ("Knee_MRI_T1_TSE_Sagittal.jpg", (60, 0, 902, 842), "(c) Sagittal T1", "slot: Sag non-FS"),
    ("Knee_MRI_t2_tse_fs_cor_320_mid.png", (240, 90, 960, 810), "(d) Coronal T2 FS", "slot: Cor FS"),
    ("Knee_MRI_pd_tse_fs_tra_320_mid.png", (260, 120, 900, 760), "(e) Axial PD FS", "slot: Ax FS"),
]

fig, axs = plt.subplots(1, 5, figsize=(7.1, 1.75), gridspec_kw={"width_ratios": [1.33, 1, 1, 1, 1]})
for ax, (fn, box, title, slot) in zip(axs, PANELS):
    im = Image.open(os.path.join(SRC, fn)).convert("RGB")
    if box:
        im = im.crop(box)
    ax.imshow(np.asarray(im), cmap=None if fn.startswith("Blausen") else "gray")
    ax.set_title(title, fontsize=6.3, pad=2)
    ax.set_xlabel(slot, fontsize=6, labelpad=2, color="#444")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
fig.tight_layout(pad=0.2, w_pad=0.4)
fig.savefig(os.path.join(FIG, "illustrative_mri.pdf"), dpi=300)
print("wrote", os.path.join(FIG, "illustrative_mri.pdf"))
