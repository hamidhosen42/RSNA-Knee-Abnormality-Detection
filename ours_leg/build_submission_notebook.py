"""Build the submission notebook: our 0.943 base + our own leg, rank-blended the way D4 does it.

Reads ../rsna-knee-d4-blend_patched.ipynb (never modified) and writes
submission/rsna-knee-base-plus-ours.ipynb:
  cells 0-30  unchanged: the public pipeline (Jiwei v11 + Fracture token) and the stash of its output
  new         markdown note, then infer_kernel.py written to /kaggle/tmp/ours_leg/ (%%writefile)
  new 2/3     run our leg from the attached training-kernel output; soft-fail to public
  new 3/3     final = rank((1 - w) * rank(public) + w * rank(ours)), w pre-registered at 0.30

    python build_submission_notebook.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "rsna-knee-d4-blend_patched.ipynb"
OUT = HERE / "submission" / "rsna-knee-base-plus-ours.ipynb"
N_KEEP = 31   # cells 0..30; cell 30 stashes the public submission as _public_probe22.csv

MARKDOWN = """## Our leg (replaces D4's private student fleet)

D4 lifted the public 0.943 pipeline to 0.945/0.946 by rank-blending it with an independently trained
EfficientNetV2-S "student fleet" whose weights are private. The next cells do the same with **our own**
leg: EfficientNetV2-S @ 384 px trained by `hosen42/rsna-knee-ours-leg-train-effv2s` on public inputs only
(`dreaddevelopment/knee-raptor-corpus` + `-ext`, `dreaddevelopment/rsna-knee-labels`), attached as that
notebook's output. Test studies are rebuilt with the exact corpus recipe stored in each checkpoint.

The blend weight is pre-registered at **0.30** (`RSNA_OURS_W`; D4 used 0.45). If the checkpoints are
missing or the leg fails in any way, the public submission from cell 30 is kept unchanged."""

MKDIR = "import os\nos.makedirs('/kaggle/tmp/ours_leg', exist_ok=True)"

RUN = r'''# ============================ OUR LEG — 2/3: our EfficientNetV2-S leg =====================
# Finds checkpoints in any attached input that has arms/<arm>/ours_leg_arm.json (the training
# kernel's output), runs infer_kernel.py as a subprocess so its GPU memory is released before the
# blend, and leaves _OV_SKIP=True (public only) on any problem. Depth-limited globs on purpose:
# a recursive glob under /kaggle/input would walk the competition's DICOM tree.
import glob as _g, os as _o, pathlib as _pl, subprocess as _sp, sys as _sys, time as _t

_OV_INPUT = _o.environ.get("RSNA_INPUT_ROOT", "/kaggle/input")
_OV_LEG_SRC = _pl.Path(_o.environ.get("RSNA_OURS_SRC", "/kaggle/tmp/ours_leg/infer_kernel.py"))
_OV_LEG_OUT = _pl.Path(_o.environ.get("RSNA_OURS_OUT", "/kaggle/tmp/ours_leg_out"))
_OV_OURS = _OV_LEG_OUT / "submission.csv"
_OV_CKPT_GLOB = _o.environ.get("RSNA_OURS_CKPT_GLOB", "*_swa.pt")


def _proc_age_h():
    try:
        up = float(open('/proc/uptime').read().split()[0])
        start = int(open('/proc/self/stat').read().split(')')[-1].split()[19]) / _o.sysconf('SC_CLK_TCK')
        return (up - start) / 3600
    except OSError:
        return 0.0          # not on Kaggle (local test)


_markers = []
for _depth in range(4):
    _parts = [_OV_INPUT] + ["*"] * _depth + ["arms", "*", "ours_leg_arm.json"]
    _markers += [m for m in sorted(_g.glob(_o.path.join(*_parts))) if "/competitions/" not in m]
_roots = sorted({str(_pl.Path(m).parents[2]) for m in _markers})
_ckpts = sorted({p for r in _roots for p in _g.glob(f"{r}/arms/*/weights/{_OV_CKPT_GLOB}")})
_left_s = int((9.0 - _proc_age_h() - 0.4) * 3600)
print(f"[ours 2/3] arms roots {_roots} | {len(_ckpts)} checkpoint(s) matching {_OV_CKPT_GLOB} | "
      f"budget left {_left_s / 3600:.2f} h", flush=True)

_OV_SKIP = True
if not _ckpts:
    print("[ours 2/3] SKIP: no checkpoints attached -> public only", flush=True)
elif not _OV_LEG_SRC.is_file():
    print(f"[ours 2/3] SKIP: {_OV_LEG_SRC} missing -> public only", flush=True)
elif _left_s < 900:
    print("[ours 2/3] SKIP: under 15 min of the 9 h budget left -> public only", flush=True)
else:
    _env = {**_o.environ, "RSNA_ARMS_DIRS": _o.pathsep.join(_roots), "RSNA_CKPT_GLOB": _OV_CKPT_GLOB,
            "RSNA_OUT": str(_OV_LEG_OUT), "RSNA_WORKERS": _o.environ.get("RSNA_WORKERS", "4"),
            "RSNA_BUDGET_S": str(_left_s), "RSNA_EXPECTED_WEIGHTS": str(len(_ckpts))}
    _t0 = _t.time()
    try:
        _rc = _sp.run([_sys.executable, str(_OV_LEG_SRC)], env=_env).returncode
    except Exception as _exc:
        _rc = -1
        print(f"[ours 2/3] could not start the leg: {type(_exc).__name__}: {_exc}", flush=True)
    print(f"[ours 2/3] leg rc={_rc} in {(_t.time() - _t0) / 60:.1f} min", flush=True)
    _OV_SKIP = _rc != 0 or not _OV_OURS.is_file()
    if _OV_SKIP:
        print("[ours 2/3] SOFT FAIL -> public only", flush=True)
'''

BLEND = r'''# ============================ OUR LEG — 3/3: blend in rank space =========================
# Rank space, because the two sides are calibrated differently and macro AUC reads only the order
# within each column. One global weight, pre-registered (E10 rule: 0.30 first; 0.45 is D4's value
# and the only other candidate). Any failed check restores the public submission.
import numpy as _ov_np
import os as _ov_os, shutil as _ov_shutil, pandas as _ov_pd

_OV_SKIP = bool(globals().get('_OV_SKIP', True))
_OV_W = float(_ov_os.environ.get("RSNA_OURS_W", "0.30"))


def _ov_rank(frame):
    return frame[_OV_TARGETS].rank(method="average", pct=True)


_ov_ok, _ov_why = False, "not attempted"
try:
    if _OV_SKIP:
        raise RuntimeError("our leg was skipped or failed")
    if not 0.0 < _OV_W < 1.0:
        raise RuntimeError(f"blend weight {_OV_W} outside (0, 1)")
    _ov_ours = _ov_pd.read_csv(_OV_OURS, dtype={"StudyInstanceUID": str})
    if _ov_ours.columns.tolist() != ["StudyInstanceUID", *_OV_TARGETS]:
        raise RuntimeError(f"our schema drift: {_ov_ours.columns.tolist()}")
    _ov_ours = _ov_ours.set_index("StudyInstanceUID").reindex(_ov_pub["StudyInstanceUID"]).reset_index()
    if _ov_ours[_OV_TARGETS].isna().any().any():
        raise RuntimeError("our leg is missing studies or has nulls")
    _ov_sd = _ov_ours[_OV_TARGETS].to_numpy(float).std(axis=0)
    if float(_ov_sd.min()) < 1e-9 and len(_ov_pub) > 10:
        raise RuntimeError(f"our leg has a constant column (min sd {_ov_sd.min():.2e})")
    _ov_out = _ov_pub.copy()
    _ov_out[_OV_TARGETS] = ((1.0 - _OV_W) * _ov_rank(_ov_pub).to_numpy()
                            + _OV_W * _ov_rank(_ov_ours).to_numpy())
    _ov_out[_OV_TARGETS] = _ov_out[_OV_TARGETS].rank(method="average", pct=True)
    assert _ov_np.isfinite(_ov_out[_OV_TARGETS].to_numpy(float)).all()
    _ov_out.to_csv(_OV_SUB, index=False)
    _ov_ok, _ov_why = True, "blended"
    if len(_ov_pub) > 2:
        _ov_corr = float(_ov_np.nanmean([_ov_rank(_ov_pub)[c].corr(_ov_rank(_ov_ours)[c], method="spearman")
                                         for c in _OV_TARGETS]))
        print(f"[ours 3/3] mean per-finding rank correlation public vs ours: {_ov_corr:.4f}", flush=True)
except Exception as _ov_exc:
    _ov_why = f"{type(_ov_exc).__name__}: {_ov_exc}"
    _ov_shutil.copy2(_OV_PUBLIC, _OV_SUB)
    print(f"[ours 3/3] NOT blended ({_ov_why}); submission.csv restored to the public output", flush=True)

print(f"[ours] final: {'public %.2f + ours %.2f' % (1 - _OV_W, _OV_W) if _ov_ok else 'PUBLIC ONLY'}",
      flush=True)
_ov_final = _ov_pd.read_csv(_OV_SUB, dtype={"StudyInstanceUID": str})
assert _ov_final.columns.tolist() == ["StudyInstanceUID", *_OV_TARGETS]
assert _ov_final[_OV_TARGETS].notna().all().all() and len(_ov_final) == len(_ov_pub)
print(f"[ours] submission.csv verified: {_ov_final.shape}, no nulls, column order ok", flush=True)
'''


def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


def main():
    nb = json.loads(BASE.read_text())
    stash = "".join(nb["cells"][N_KEEP - 1]["source"])
    assert "OUR LEG — 1/3" in stash and "_public_probe22.csv" in stash, "base notebook layout changed"
    infer = (HERE / "infer_kernel.py").read_text()
    nb["cells"] = nb["cells"][:N_KEEP] + [
        {"cell_type": "markdown", "metadata": {}, "source": MARKDOWN.splitlines(keepends=True)},
        code_cell(MKDIR),
        code_cell("%%writefile /kaggle/tmp/ours_leg/infer_kernel.py\n" + infer),
        code_cell(RUN),
        code_cell(BLEND),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"wrote {OUT.relative_to(HERE.parent)} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
