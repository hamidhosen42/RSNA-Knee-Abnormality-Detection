"""H957-S6: H957x5 (R7 notebook verbatim + our 5 Mirror students, LB 0.957) plus heliosli's older S6 compact model as
one more leg.

Order inside the notebook (all on hosen42):
  1. heliosli's R7 + Raptor notebook, cells unchanged (writes submission.csv = Mirror 0.5 / R7 0.5);
  2. our 5-student reader (exactly H957x5's reader cells);
  3. the S6 compact stage = the 0.954 notebook's S6 cells (code that matches bundle 209a32c8b50a6b1c in
     heliosli/rsna-knee-compact-v1 v13), run through exec() inside try/except, with its own bundle / source /
     prediction folders (nothing of R7 is reused or overwritten) and a deadline of notebook start + 7.9 h.  If it fails
     or runs late, the S6 leg is dropped and the notebook still finishes;
  4. final per-finding rank blend, weights fixed before running: R7 notebook 0.7, S6 0.1, students 0.2.  A leg that is
     missing for a study (or as a whole) gives its weight to the others (renormalised per study).

Pre-registered: keep if LB > 0.957 (H957x5), neutral if equal, reject below.

    python build_h957s6.py [--owner hosen42]
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--base", default="submission_h957x5/rsna-knee-h957x5.ipynb")
ap.add_argument("--s6", default="submission_h954/rsna-knee-h954.ipynb")
ap.add_argument("--owner", default="hosen42")
ap.add_argument("--out", default="submission_h957s6")
ap.add_argument("--w", default="0.7,0.1,0.2", help="parent (R7 notebook), S6, students")
a = ap.parse_args()
WP, WS6, WST = (float(x) for x in a.w.split(","))
assert abs(WP + WS6 + WST - 1) < 1e-9

base = json.load(open(os.path.join(ROOT, a.base)))
cells = base["cells"]
src = lambda c: "".join(c["source"])
assert len(cells) == 13 and src(cells[12]).startswith("# [h957] final:") and "cnxt_pl*_full.pt" in src(cells[11])
h954 = json.load(open(os.path.join(ROOT, a.s6)))["cells"]
s4, s5 = src(h954[4]), src(h954[5])
assert "BUNDLE_ID = '209a32c8b50a6b1c'" in s4 and "C_SLUG = 'rsna-knee-compact-v1'" in s4


def sub(text, old, new, count=1):
    assert text.count(old) == count, (old, text.count(old))
    return text.replace(old, new)


# separate folders / files so the S6 run can never read or overwrite R7's
s4 = sub(s4, "DEADLINE_HOURS = 8.3", "DEADLINE_HOURS = 7.9   # [h957s6] leave time for the final blend")
s4 = sub(s4, "SRC = TMP / 'probe_src'", "SRC = TMP / 'probe_src_s6'")
s4 = sub(s4, "BUNDLE = TMP / 'bundle'", "BUNDLE = TMP / 'bundle_s6'")
s5 = sub(s5, "pred = TMP / 'prediction'", "pred = TMP / 'prediction_s6'")
s5 = sub(s5, "shutil.copyfile(log, WORK / (log.parent.name + '_log.txt'))", "shutil.copyfile(log, WORK / ('s6_' + log.parent.name + '_log.txt'))")
s5 = sub(s5, "shutil.copyfile(pred / 'submission.csv', WORK / 'ours_sub.csv')", "shutil.copyfile(pred / 'submission.csv', WORK / 's6_sub.csv')")
s5 = sub(s5, "_df = pd.read_csv(WORK / 'ours_sub.csv')", "_df = pd.read_csv(WORK / 's6_sub.csv')")
s5 = sub(s5, "_df.to_csv(WORK / 'ours_sub.csv', index=False)", "_df.to_csv(WORK / 's6_sub.csv', index=False)")
s5 = sub(s5, "(WORK / 'submission_receipt.json').write_text", "(WORK / 's6_submission_receipt.json').write_text")
s5 = sub(s5, "print('OURS_STAGE_COMPLETE'", "print('S6_STAGE_COMPLETE'")
s5 = sub(s5, "display(pd.read_csv(WORK / 'ours_sub.csv').head())", "print(pd.read_csv(WORK / 's6_sub.csv').head())")
S6_CODE = s4 + "\n" + s5
compile(S6_CODE, "s6_stage", "exec")
assert "ours_sub.csv" not in S6_CODE and "TMP / 'prediction'\n" not in S6_CODE

S6_CELL = ("# [h957s6] heliosli's S6 compact (the 0.954 notebook's stage-2 cells, matching code for bundle 209a32c8b50a6b1c),\n"
           "# run in its own folders after the R7 notebook and our students; any failure or a late finish drops the S6 leg.\n"
           "import time as _s6_time\n"
           "S6_OK, _s6_t0 = False, _s6_time.time()\n"
           "S6_CODE = " + repr(S6_CODE) + "\n"
           "try:\n"
           "    exec(compile(S6_CODE, 's6_stage', 'exec'), globals())\n"
           "    S6_OK = (WORK / 's6_sub.csv').is_file()\n"
           "except BaseException as _s6_e:\n"
           "    print('S6_STAGE_FAILED', type(_s6_e).__name__, str(_s6_e)[:300], flush=True)\n"
           "print('S6_STAGE_DONE ok=%s seconds %.0f' % (S6_OK, _s6_time.time() - _s6_t0), flush=True)\n")
compile(S6_CELL, "s6_cell", "exec")

FINAL = '''# [h957s6] final: per-finding rank blend of three legs, weights fixed before running: the R7 notebook's own
# submission __WP__, heliosli's S6 compact __WS6__, our 5 Mirror students __WST__.  A leg missing for a study (or as a whole)
# gives its weight to the others; each leg is ranked over its own scored studies.
import csv as _fc, json as _fj, shutil as _fs
import numpy as _fn, pandas as _fp
W_P, W_S6, W_ST = __WP__, __WS6__, __WST__
_fw = WORK
_fs.copyfile(_fw / 'submission.csv', _fw / '_r7_submission.csv')
_par = _fp.read_csv(_fw / '_r7_submission.csv', dtype={'StudyInstanceUID': str})
_cols = list(_par.columns); _labs = _cols[1:]; _ids = _par['StudyInstanceUID'].tolist(); _n = len(_ids)
_rep = {'weights': {'r7_notebook': W_P, 's6': W_S6, 'students': W_ST}, 'rows': _n}


def _leg(path, ok_flag, name):
    if not ok_flag or not path.is_file():
        _rep[name] = 'missing'; return None
    d = _fp.read_csv(path, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').reindex(_ids)[_labs]
    if d.notna().values.mean() < 0.97 or not (d.nunique() > 1).any():
        _rep[name] = 'unusable'; return None
    return d


try:
    _st = _leg(_fw / 'reader_sub.csv', READER_OK, 'students')
    _s6 = _leg(_fw / 's6_sub.csv', globals().get('S6_OK', False), 's6')
    _st_ok = _fn.zeros(_n, bool) if _st is None else _fn.isfinite(_st.values.astype(float)).all(axis=1)
    _s6_ok = _fn.zeros(_n, bool)
    if _s6 is not None:   # S6 fills an unscored study with the column mean: read what its workers really scored
        _scored = set()
        for _f in (TMP / 'prediction_s6').glob('worker*/predictions.csv'):
            with open(_f, encoding='utf-8') as _fh:
                for _r in _fc.DictReader(_fh):
                    try:
                        _v = [float(_r[_L]) for _L in _labs]
                    except (KeyError, ValueError, TypeError):
                        continue
                    if all(0. <= _x <= 1. for _x in _v):
                        _scored.add(_r['StudyInstanceUID'])
        _s6_ok = _fn.array([u in _scored for u in _ids]) & _fn.isfinite(_s6.values.astype(float)).all(axis=1)
    _rep.update(students_rows=int(_st_ok.sum()), s6_rows=int(_s6_ok.sum()))

    def _pct_on(v, ok):
        r = _fn.zeros(_n)
        if ok.any():
            r[ok] = _fp.Series(_fn.asarray(v, dtype=_fn.float64)[ok]).rank(method='average', pct=True).values
        return r

    _out = _par.copy()
    for _c in _labs:
        _p0 = _fp.Series(_par[_c].values.astype(_fn.float64)).rank(method='average', pct=True).values
        _num = W_P * _p0
        _den = _fn.full(_n, W_P)
        if _st is not None:
            _num = _num + W_ST * _pct_on(_st[_c].values, _st_ok) * _st_ok; _den = _den + W_ST * _st_ok
        if _s6 is not None:
            _num = _num + W_S6 * _pct_on(_s6[_c].values, _s6_ok) * _s6_ok; _den = _den + W_S6 * _s6_ok
        _out[_c] = (_num / _den).astype(_fn.float32)
    _v = _out[_labs].values
    assert _fn.isfinite(_v).all() and ((_v >= 0) & (_v <= 1)).all() and _out['StudyInstanceUID'].tolist() == _ids
    _out[_cols].to_csv(_fw / 'submission.csv', index=False)
    _rep['status'] = 'blended'
except Exception as _e:
    _fs.copyfile(_fw / '_r7_submission.csv', _fw / 'submission.csv')
    _rep['status'] = f'R7 FILE KEPT: {type(_e).__name__}: {str(_e)[:300]}'
(_fw / 'h957s6_receipt.json').write_text(_fj.dumps(_rep, indent=1))
print('H957S6_FINAL', _fj.dumps(_rep), flush=True)
'''.replace("__WP__", repr(WP)).replace("__WS6__", repr(WS6)).replace("__WST__", repr(WST))
compile(FINAL, "final", "exec")


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.splitlines(keepends=True)}


cells[0]["source"] = ("# RSNA Knee H957-S6: R7 + Raptor (verbatim) + our 5 Mirror students + heliosli's S6 compact\n\n"
                      f"Final per-finding rank blend: R7 notebook {WP}, S6 {WS6}, students {WST} (fixed before running). The S6 "
                      "stage runs last in its own folders and is dropped if it fails or runs late.\n").splitlines(keepends=True)
base["cells"] = cells[:12] + [code(S6_CELL), code(FINAL)]
for c in base["cells"]:
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
out = os.path.join(ROOT, a.out)
os.makedirs(out, exist_ok=True)
slug = "rsna-knee-" + a.out.replace("submission_", "")
json.dump(base, open(os.path.join(out, slug + ".ipynb"), "w"), indent=1)
meta = json.load(open(os.path.join(ROOT, os.path.dirname(a.base), "kernel-metadata.json")))
meta.update(id=f"{a.owner}/{slug}", title=slug, code_file=slug + ".ipynb")
meta["dataset_sources"] = meta["dataset_sources"] + ["heliosli/rsna-knee-compact-v1"]
json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
print("wrote", out, "| cells", len(base["cells"]), "| sources", meta["dataset_sources"])
