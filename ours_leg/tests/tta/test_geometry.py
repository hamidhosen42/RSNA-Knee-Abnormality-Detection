"""Test 1: preprocess.load_series of a patient-mirrored acquisition == mirror_op(load_series of the original).
Test 2: window_centres symmetry (sagittal mirror = channel reversal + pos -> 1 - pos on the same windows).

    uv run --no-project --with numpy --with pydicom --with opencv-python-headless --with pandas \
        python -I test_geometry.py <runtime_src dir> <scratch dir>
"""
import itertools
import json
import os
import shutil
import sys

import numpy as np

src_dir, work = sys.argv[1], sys.argv[2]
sys.path.insert(0, src_dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preprocess  # noqa: E402
from synth import acquisition, mirrored_variant, phantom, rot, write_series  # noqa: E402

BASE = {"Sagittal": ([0, 1, 0], [0, 0, -1]), "Coronal": ([1, 0, 0], [0, 0, -1]), "Axial": ([1, 0, 0], [0, 1, 0])}
TILT = {"Sagittal": ("z", 15), "Coronal": ("z", 12), "Axial": ("x", 10)}   # typical oblique planning
results = []


class _FixedPct:
    """stand-in for numpy inside preprocess: identical except a fixed intensity window (isolates the geometry)"""
    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def percentile(a, q):
        return np.array([0.0, 3000.0])


MODE = {"fixed": False}


def canon(d):
    if MODE["fixed"]:
        preprocess.np = _FixedPct()
    try:
        vol, info = preprocess.load_series(d)
    finally:
        preprocess.np = np
    return vol, info


def one(plane, oblique, conv, cols, n_slices, gap):
    r, c = (np.array(v, float) for v in BASE[plane])
    if oblique:
        R = rot(*TILT[plane])
        r, c = R @ r, R @ c
    if conv == "transposed":
        r, c = c, r
    elif conv == "neg_r":
        r = -r
    acq = acquisition(r, c, 200, cols, 0.8, 0.8, n_slices, gap, centre=(3.0, -4.0, 6.0))
    arrays = [np.round(phantom(acq["P"][t])) for t in range(n_slices)]
    root = os.path.join(work, "geom")
    shutil.rmtree(root, ignore_errors=True)
    write_series(os.path.join(root, "orig"), arrays, acq["iop"], acq["ipps"], acq["ps"])
    v0, i0 = canon(os.path.join(root, "orig"))
    ax = {"Sagittal": 0, "Coronal": 2, "Axial": 2}[i0["plane"]]
    expect = v0[::-1] if ax == 0 else v0[:, :, ::-1]
    for var in ("v0", "v1", "v2", "v3"):
        arr, iop, ipps, ps = mirrored_variant(acq, arrays, var)
        inst = list(range(n_slices, 0, -1)) if var == "v3" else None
        write_series(os.path.join(root, var), arr, iop, ipps, ps, inst_order=inst)
        vm, im = canon(os.path.join(root, var))
        d = np.abs(vm.astype(int) - expect.astype(int))
        # a naive mirror for contrast: flip the image columns of every plane (wrong for sagittal: that is anterior <-> posterior)
        naive = v0[:, :, ::-1]
        dn = np.abs(vm.astype(int) - naive.astype(int))
        results.append({"mode": "fixed_window" if MODE["fixed"] else "real", "plane": plane, "oblique": oblique, "conv": conv, "cols": cols, "n": n_slices, "variant": var,
                        "plane_detected": [i0["plane"], im["plane"]], "slice_gap_set": ["slice_gap" in i0, "slice_gap" in im],
                        "shape": list(vm.shape), "exact": bool(d.max() == 0), "max_abs": int(d.max()),
                        "mean_abs": round(float(d.mean()), 4), "naive_colflip_mean_abs": round(float(dn.mean()), 2)})
    shutil.rmtree(root, ignore_errors=True)


for fixed in (False, True):
    MODE["fixed"] = fixed
    for plane, oblique, conv, cols in itertools.product(BASE, (False, True), ("typical", "transposed", "neg_r"), (200, 201)):
        one(plane, oblique, conv, cols, 24, 3.5)
    for plane in BASE:   # > MAX_SLICES: the 64-slice subsampling must commute with the reversal
        one(plane, True, "typical", 200, 80, 1.0)
json.dump(results, open(os.path.join(work, "test1_results.json"), "w"), indent=1)

for mode in ("real", "fixed_window"):
    rr = [r for r in results if r["mode"] == mode]
    for plane in BASE:
        for cols in (200, 201):
            g = [r for r in rr if r["plane"] == plane and r["cols"] == cols]
            print(f"TEST1 {mode:12s} {plane:8s} cols {cols}: series {len(g)}, exact {sum(r['exact'] for r in g)}, "
                  f"max|d| {max(r['max_abs'] for r in g)}, mean|d| {np.mean([r['mean_abs'] for r in g]):.4f}")
print("  plane detection always consistent:", all(r["plane_detected"][0] == r["plane_detected"][1] == r["plane"] for r in results))
print("  naive all-plane column flip, sagittal mean |diff| (should be large):",
      round(float(np.mean([r["naive_colflip_mean_abs"] for r in results if r["plane"] == "Sagittal"])), 2))
print("  correct mirror, mean |diff| over all configs:", round(float(np.mean([r["mean_abs"] for r in results])), 5))

# ---- Test 2: windows
from knee import window_centres  # noqa: E402

viol = []
for k in (8, 12, 16):
    for n in range(3, 400):
        c = window_centres(n, k, False)
        if sorted((n - 1 - c).tolist()) != sorted(c.tolist()):
            viol.append((k, n))
        # windows of the reversed volume at the same centres == channel-reversed windows at n-1-c, pos' = 1 - pos
        vol = np.arange(n)
        w_rev = np.stack([vol[::-1][np.clip([j - 1, j, j + 1], 0, n - 1)] for j in c])
        w_alt = np.stack([vol[np.clip([j - 1, j, j + 1], 0, n - 1)][::-1] for j in (n - 1 - c)])
        if not np.array_equal(w_rev, w_alt):
            viol.append(("win", k, n))
print("TEST2 window_centres symmetric for n=3..399, k in (8,12,16): violations", len(viol), viol[:10])
