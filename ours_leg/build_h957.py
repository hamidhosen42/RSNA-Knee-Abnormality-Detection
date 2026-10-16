"""H957: heliosli's public 0.956 notebook (0-957-latest-dataset-score-in-one-click v1 = nartaa's Mirror Raptor 0.5 +
heliosli's R7 compact 0.5, bundle 8c7b666e27dc0e8f) kept VERBATIM, plus our Mirror-student reader ensemble blended on
top the way the public 0.955 notebook (bliverrigpdoge/velci-kcd25-20) and our H955C (LB 0.955 = 0.954 + 0.001) did:

    final = 0.8 x rank(R7 notebook's own submission.csv) + 0.2 x rank(reader),   per finding, weights fixed before running

Reader = infer.py's probability mean of our students C, C2, C3 (leg alone 0.948 / 0.948 / 0.949), run by the reader
cells of samanyu1808/rsna-knee-blend3 v3 (goodpjw2008's DICOM inference code; identical to H955C's reader stage).  A
reader that failed as a whole leaves the R7 file exactly as written; a study the reader could not score keeps its R7
rank.  The checkpoints come from the private dataset <owner>/rsna-knee-student-readers.

Pre-registered: keep if LB > the R7 notebook's own score (0.956), neutral if equal, reject below.

    python build_h957.py [--owner hosen42] [--readers rsna-knee-student-readers] [--n-ckpt 3]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import NB, inp, out_dir, write_json  # noqa: E402  (one place for every path: ours_leg/paths.py)
ap = argparse.ArgumentParser()
ap.add_argument("--parent", default="public_notebooks/0.956_heliosli__0-957-latest-dataset-score-in-one-click.ipynb")
ap.add_argument("--blend3", default="public_notebooks/0.951_2v_samanyu1808__rsna-knee-blend3-please-upvote-if-useful.ipynb")
ap.add_argument("--owner", default="hosen42")
ap.add_argument("--readers", default="rsna-knee-student-readers")
ap.add_argument("--n-ckpt", type=int, default=3)
ap.add_argument("--pattern", default="cnxt_plC*_full.pt", help="H957: C, C2, C3; H957x5: cnxt_pl*_full.pt (+ S75, T75s1)")
ap.add_argument("--w-reader", type=float, default=0.2,
                help="students' rank weight (H957/H957x5: 0.2). 2026-10-14 user: raise the score any way -> LB probes 0.3 / 0.4")
ap.add_argument("--out", default="submission_h957")
a = ap.parse_args()

nb = json.load(open(inp(a.parent)))
cells = nb["cells"]
src = lambda c: "".join(c["source"])
assert len(cells) == 7 and "BUNDLE_ID = '8c7b666e27dc0e8f'" in src(cells[4]) and "C_SLUG = 'rsna-knee-compact-r7cnxfsss'" in src(cells[4])
assert src(cells[6]).startswith("# Stage 3: per-finding rank blend with weights fixed before any run: Raptor 0.5, ours 0.5.")
assert "sub.to_csv(WORK / 'submission.csv', index=False)" in src(cells[6])

b3 = json.load(open(inp(a.blend3)))["cells"]
assert src(b3[9]).startswith("# Stage 2b: goodpjw2008 ConvNeXt 2.5D reader")
reader = [json.loads(json.dumps(b3[i])) for i in (5, 6, 7, 8, 9)]
r = src(reader[-1])
old = '_ck = sorted(glob.glob(_assets + "/cnxt_v0_fold*.pt")); print("reader ckpts", _ck, flush=True)'
assert r.count(old) == 1
r = r.replace(old, f'# [h957] reader = probability mean of our Mirror students {a.pattern} (private dataset {a.readers})\n'
                   f'_ck = sorted(glob.glob("/kaggle/input/**/{a.readers}/{a.pattern}", recursive=True))\n'
                   f'assert len(_ck) == {a.n_ckpt}, _ck; print("reader ckpts", _ck, flush=True)')
reader[-1]["source"] = r.splitlines(keepends=True)
compile(r, "reader", "exec")

FINAL = '''# [h957] final: 0.8 x rank(the R7 notebook's own submission.csv) + 0.2 x rank(our Mirror-student ensemble), per
# finding, weights fixed before any run (public 0.955 notebook / our H955C).  A failed reader leaves the R7 file as is;
# a study the reader could not score keeps its R7 rank.
import json as _hj, shutil as _hs
import numpy as _hn, pandas as _hp
W_PARENT, W_READER = __WP__, __WR__
_hw = WORK
_hs.copyfile(_hw / 'submission.csv', _hw / '_r7_submission.csv')
_par = _hp.read_csv(_hw / '_r7_submission.csv', dtype={'StudyInstanceUID': str})
_cols = list(_par.columns)
_labs = _cols[1:]
_ids = _par['StudyInstanceUID'].tolist()
try:
    if not READER_OK:
        raise RuntimeError('reader stage failed')
    _rd = _hp.read_csv(_hw / 'reader_sub.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    _rd = _rd.reindex(_ids)[_labs]
    if _rd.notna().values.mean() < 0.97 or not (_rd.nunique() > 1).any():
        raise RuntimeError('reader file unusable (mostly NaN or constant)')
    _bad = ~_hn.isfinite(_rd.values.astype(float)).all(axis=1)
    _out = _par.copy()
    _rho = {}
    for _c in _labs:
        _p0 = _hp.Series(_par[_c].values.astype(_hn.float64)).rank(method='average', pct=True).values
        _r = _hn.full(len(_ids), _hn.nan)
        _r[~_bad] = _hp.Series(_rd[_c].values.astype(_hn.float64)[~_bad]).rank(method='average', pct=True).values
        _out[_c] = _hn.where(_bad, _p0, W_PARENT * _p0 + W_READER * _hn.nan_to_num(_r)).astype(_hn.float32)
        try:   # diagnostic only (rank correlation with numpy; never allowed to change the submission)
            _ok = ~_bad
            if _ok.sum() >= 3:
                _rho[_c] = round(float(_hn.corrcoef(_hp.Series(_par[_c].values[_ok]).rank().values,
                                                     _hp.Series(_rd[_c].values[_ok]).rank().values)[0, 1]), 4)
        except Exception:
            pass
    _v = _out[_labs].values
    assert _hn.isfinite(_v).all() and ((_v >= 0) & (_v <= 1)).all() and _out['StudyInstanceUID'].tolist() == _ids
    _out[_cols].to_csv(_hw / 'submission.csv', index=False)
    _rep = {'status': 'blended', 'weights': {'r7_notebook': W_PARENT, 'students': W_READER}, 'rows': len(_ids),
            'reader_failed_rows': int(_bad.sum()), 'spearman_r7_vs_students': _rho}
except Exception as _e:
    _hs.copyfile(_hw / '_r7_submission.csv', _hw / 'submission.csv')
    _rep = {'status': f'R7 FILE KEPT: {type(_e).__name__}: {str(_e)[:300]}'}
(_hw / 'h957_receipt.json').write_text(_hj.dumps(_rep, indent=1))
print('H957_FINAL', _hj.dumps(_rep), flush=True)
'''
FINAL = FINAL.replace("__WP__", repr(round(1 - a.w_reader, 4))).replace("__WR__", repr(a.w_reader))
compile(FINAL, "final", "exec")
final = {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": FINAL.splitlines(keepends=True)}
cells[0]["source"] = ("# RSNA Knee H957: heliosli's R7 + Raptor notebook (verbatim) + our Mirror-student readers\n\n"
                      "Cells up to stage 3 are heliosli's public 0.956 notebook unchanged. Then goodpjw2008's 2.5D reader code "
                      "runs our three Mirror-student checkpoints (C, C2, C3), and the last cell blends 0.8 x rank(R7 notebook) + "
                      "0.2 x rank(students), weights fixed before running.\n").splitlines(keepends=True)
nb["cells"] = cells + reader + [final]
for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None

out = out_dir(a.out)
os.makedirs(out, exist_ok=True)
slug = "rsna-knee-" + a.out.replace("submission_", "")
write_json(os.path.join(out, slug + ".ipynb"), nb, indent=1)
meta = {"id": f"{a.owner}/{slug}", "title": slug, "code_file": slug + ".ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f",
        "docker_image_pinning_type": "original",
        "dataset_sources": ["heliosli/rsna-knee-compact-r7cnxfsss", "nartaa/rsna-knee-publication-swa-weights-20261007",
                            "heliosli/rsna-orthofoundation-fold0-epoch7-20261004", "goodpjw2008/rsna-knee-2-5d-convnext-reader",
                            f"{a.owner}/{a.readers}"],
        "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": []}
write_json(os.path.join(out, "kernel-metadata.json"), meta, indent=2)
print("wrote", out, "| cells", len(nb["cells"]))
