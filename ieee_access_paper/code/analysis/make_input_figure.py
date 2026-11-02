"""Example-input figure: the canonical volumes and the evaluation windows that the reader sees for one study.

Needs the competition DICOMs, so run it where the data are (a Kaggle notebook with the competition attached):

    python make_input_figure.py /kaggle/input/rsna-knee-abnormality-detection <StudyInstanceUID> input_example.pdf

Each row is one plane/fat-suppression slot (Algorithm 2 preprocessing, reader/preprocess.py); the columns are the centre
slices of K = 5 evaluation windows (reader/knee.py window_centres).  Missing slots are drawn empty.

Before putting the output in the paper, check that the dataset's terms allow publishing images, and pick a study from
the training split (never the test split).  The figure shows no text from the DICOM header.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reader"))
from knee import N_SLOTS, PLANES, slot_of, window_centres  # noqa: E402
from preprocess import load_series  # noqa: E402

K_SHOW = 5
SLOT_NAMES = [f"{p}{' FS' if s < 3 else ' non-FS'}" for s, p in enumerate(PLANES * 2)]


def main(data_dir, study, out):
    se = pd.read_csv(f"{data_dir}/train_series.csv")
    se = se[se.StudyInstanceUID.astype(str) == study]
    assert len(se), f"study {study} not in train_series.csv (use a training study)"
    vols = {}
    for sr, fs, pl in zip(se.SeriesInstanceUID, se.Fat_Suppression, se.Anatomical_Plane):
        s = slot_of(pl, fs)
        vol, _ = load_series(f"{data_dir}/train_series/{study}/{sr}", pl)
        if len(vol) >= 3 and (s not in vols or len(vol) > len(vols[s])):   # inference keeps the series with most slices
            vols[s] = vol

    fig, axs = plt.subplots(N_SLOTS, K_SHOW, figsize=(K_SHOW * 1.1, N_SLOTS * 1.1))
    for s in range(N_SLOTS):
        cols = window_centres(len(vols[s]), K_SHOW, train=False) if s in vols else [None] * K_SHOW
        for k, j in enumerate(cols):
            ax = axs[s, k]
            ax.set_xticks([])
            ax.set_yticks([])
            if j is not None:
                ax.imshow(vols[s][j], cmap="gray", vmin=0, vmax=255)
            else:
                ax.set_facecolor("#eeeeee")
            if k == 0:
                ax.set_ylabel(SLOT_NAMES[s], fontsize=6)
    fig.tight_layout(pad=0.2)
    fig.savefig(out, dpi=300)
    print("wrote", out, "| slots present:", sorted(vols))


if __name__ == "__main__":
    main(*sys.argv[1:4])
