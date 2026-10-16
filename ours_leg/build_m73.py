"""M73: the 0.954 recipe (Raptor 0.7 + an independent non-Raptor model 0.3, per-finding percentile ranks) with OUR
non-Raptor stack in the place of heliosli's retired compact model.

  stage 0  nartaa's Anatomical Mirror Raptor (public 0.949), exactly as heliosli's 0.954 notebook runs it
           (its cells 2-3: checkpoint-verified source, own subprocess, raptor_raw.csv / raptor_sub.csv);
  stack    E20 (rule-clean 0.945) stages 1-3 unchanged - DINO ensemble, A5 attention pooling, RadImageNet heads +
           calibrator - then Part B, our 7-checkpoint goodpjw-recipe reader with its per-target routed blend.
           E20's stages 4-5 (four/five Raptor views + CoAt readers) are NOT run: the Raptor family is stage 0, and
           dropping them keeps the run well inside 9 h.  The public0033 meniscus bag overlay lived in stage 5, so
           it goes with it (Medial Meniscus weight 0.10).
  final    W_MIRROR 0.7 / W_STACK 0.3 - copied from the 0.954 notebook, fixed before any run, not fitted.

    python build_m73.py [--e20 submission_e20] [--h954 submission_h954] [--out submission_m73] [--owner esfersami50]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import NB, inp, out_dir, write_json  # noqa: E402  (one place for every path: ours_leg/paths.py)
ap = argparse.ArgumentParser()
ap.add_argument("--e20", default="submission_e20")
ap.add_argument("--h954", default="submission_h954")
ap.add_argument("--out", default="submission_m73")
ap.add_argument("--owner", default="esfersami50")
a = ap.parse_args()


def load(d):
    name = next(f for f in os.listdir(inp(d)) if f.endswith(".ipynb"))
    return json.load(open(os.path.join(inp(d), name))), json.load(open(os.path.join(inp(d), "kernel-metadata.json")))


e20, e20_meta = load(a.e20)
h954, h954_meta = load(a.h954)
E, H = e20["cells"], h954["cells"]
src = lambda c: "".join(c["source"])
assert src(H[2]).startswith("%%writefile /tmp/raptor_run.py") and "RAPTOR_STAGE_DONE" in src(H[3])
assert src(E[29]).startswith("## Stages 4–5 — Raptor and complementary CoAtNet readers")
assert src(E[32]).startswith("# Publish submission.csv only after the entire BTKD + CoAt family graph succeeds.")
assert src(E[38]).startswith("# Run our reader and rank-blend it into the stack's submission.")
assert "rsna_phase('radimagenet', 'COMPLETE')" in src(E[28])


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


INTRO = """# RSNA Knee M73: Mirror Raptor 0.7 + our non-Raptor stack 0.3

The 0.954 public recipe (heliosli/rsna-knee-abnormality-detection v1: Raptor 0.7 + an independent model 0.3, per-finding
percentile ranks) with our own stack in the place of the compact model, which its author has retired.

* **Stage 0** runs nartaa's [RSNA Knee 0949 Anatomical Mirror](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror)
  exactly as the 0.954 notebook does (checkpoint-verified source, own subprocess).
* **Stages 1-3 + Part B** are our rule-clean 0.945 stack (E20) without its Raptor / CoAt stages: DINO ensemble, A5
  attention pooling, RadImageNet heads, then our 7-checkpoint 2.5D ConvNeXt reader.
* **Final:** Mirror 0.7, our stack 0.3, weights copied from the 0.954 notebook and fixed before any run.
"""
FINAL_CHECKS = """# [m73] Final checks of our stack WITHOUT stages 4-5 (Raptor views + CoAt readers): the Raptor family is stage 0,
# blended in the last cell.  Same contract as before: schema, study order, finite values, DINO 20/20, calibrator.
_release_path=Path('/kaggle/working/_pipeline_stage.csv')
_release_df=rsna_frame(pd.read_csv(_release_path,dtype={'StudyInstanceUID':str}),_RSNA_TEST_IDS,_RSNA_LABELS,'FINAL')
if _DINOV2_MATCHED_MEMBERS!=20:raise RuntimeError('final DINO member count mismatch')
if not globals().get('V18_CALIBRATOR_APPLIED'):raise RuntimeError('BTKD calibrator was not applied')
rsna_save_predictions('final_no_raptor',_RSNA_TEST_IDS,_release_df[_RSNA_LABELS].to_numpy(),_RSNA_LABELS)
_RSNA_AUDIT.update(status='COMPLETE_NO_RAPTOR_STAGES',dino_members=20,a5_folds=5,raptor_views=0,coat_family_members=0,elapsed_seconds=time.time()-T0)
for cache_file in list(_RSNA_CACHE_FILES):Path(cache_file).unlink(missing_ok=True)
_final=Path('/kaggle/working/submission.csv');_tmp=_final.with_suffix('.csv.tmp')
with _tmp.open('w') as _out_handle:
    _release_df.to_csv(_out_handle,index=False)
    _out_handle.flush();os.fsync(_out_handle.fileno())
os.replace(_tmp,_final)
rsna_json('/kaggle/working/btkd_v559_complete.json',_RSNA_AUDIT)
print(f'STACK_STAGES_1_3_COMPLETE: {_final}, {len(_release_df)} studies, {time.time()-T0:.0f}s since stack start', flush=True)
"""
FINAL_BLEND = """# [m73] Final: per-finding percentile-rank blend, weights fixed before any run (the 0.954 notebook's 0.7 / 0.3):
# Mirror Raptor 0.7, our stack (stages 1-3 + the routed reader blend) 0.3.  A study Mirror failed on (left at 0.5 in
# every column) takes the stack's rank; more than 3 % Mirror failures stops the notebook.
import json as _m_json, shutil as _m_shutil, time as _m_time
import numpy as _m_np
import pandas as _m_pd
W_MIRROR, W_STACK = 0.7, 0.3
_m_work = Path('/kaggle/working')
_m_shutil.copy(_m_work / 'submission.csv', _m_work / '_m73_stack.csv')
_m_ids = [str(u) for u in _RSNA_TEST_IDS]
_m_cols = list(_m_pd.read_csv(_m_work / '_m73_stack.csv', nrows=1).columns)
_m_labels = _m_cols[1:]


def _m_load(p):
    d = _m_pd.read_csv(p, dtype={'StudyInstanceUID': str})
    assert sorted(d['StudyInstanceUID']) == sorted(_m_ids), f'{p}: rows differ from test.csv'
    assert set(_m_labels) <= set(d.columns), f'{p}: columns differ'
    return d.set_index('StudyInstanceUID').loc[_m_ids, _m_labels]


_m_stack = _m_load(_m_work / '_m73_stack.csv')
_m_raw = _m_load(_m_work / 'raptor_raw.csv')
_m_rsub = _m_load(_m_work / 'raptor_sub.csv')
_m_n = len(_m_ids)
_m_rap_fail = (_m_raw.values == 0.5).all(axis=1)
if _m_rap_fail.sum() > 0.03 * _m_n:
    raise RuntimeError('Mirror failed on %d of %d studies' % (_m_rap_fail.sum(), _m_n))
assert _m_np.isfinite(_m_stack.values).all(), 'stack predictions not finite'


def _m_pct(v):
    return _m_pd.Series(_m_np.asarray(v, dtype=_m_np.float64)).rank(method='average', pct=True).values


_m_out = _m_np.zeros((_m_n, len(_m_labels)))
_m_rho_rr, _m_rho_rs = {}, {}
for _j, _c in enumerate(_m_labels):
    _rr, _rs = _m_pct(_m_raw[_c].values), _m_pct(_m_stack[_c].values)
    _b = W_MIRROR * _rr + W_STACK * _rs
    _b[_m_rap_fail] = _rs[_m_rap_fail]
    _m_out[:, _j] = _b
    if _m_n >= 3:
        _m_rho_rr[_c] = round(float(_m_pd.Series(_m_raw[_c].values).corr(_m_pd.Series(_m_rsub[_c].values), method='spearman')), 4)
        _m_rho_rs[_c] = round(float(_m_pd.Series(_m_raw[_c].values).corr(_m_pd.Series(_m_stack[_c].values), method='spearman')), 4)
if _m_rho_rr and min(_m_rho_rr.values()) < 0.99:
    raise RuntimeError('Mirror raw dump does not match its own submission ranks: %s' % _m_rho_rr)
_m_sub = _m_pd.DataFrame(_m_out.astype(_m_np.float32), columns=_m_labels)
_m_sub.insert(0, 'StudyInstanceUID', _m_ids)
_m_sub = _m_sub[_m_cols]
assert _m_np.isfinite(_m_sub[_m_labels].values).all()
assert ((_m_sub[_m_labels].values >= 0) & (_m_sub[_m_labels].values <= 1)).all()
_m_sub.to_csv(_m_work / 'submission.csv', index=False)
_m_receipt = {'weights': {'mirror': W_MIRROR, 'stack': W_STACK}, 'rows': _m_n, 'mirror_failed': int(_m_rap_fail.sum()),
              'mirror_seconds': round(RAPTOR_SECONDS, 1), 'total_seconds': round(_m_time.time() - NB_T0, 1),
              'spearman_mirror_raw_vs_sub': _m_rho_rr, 'spearman_mirror_vs_stack': _m_rho_rs}
(_m_work / 'm73_blend_receipt.json').write_text(_m_json.dumps(_m_receipt, indent=1))
print('NOTEBOOK_COMPLETE', _m_json.dumps(_m_receipt), flush=True)
"""

cells = [md(INTRO), H[1], H[2], H[3]]
cells += E[0:29]
cells += [md("## Stages 4-5 — not run in M73\nThe Raptor family is stage 0 (Anatomical Mirror); the CoAt readers and "
             "the public0033 meniscus overlay are dropped with it.\n")]
cells += [E[31], code(FINAL_CHECKS)]
cells += E[33:]
cells += [md("## Final blend — Mirror 0.7 + our stack 0.3\n"), code(FINAL_BLEND)]
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
for t in (FINAL_CHECKS, FINAL_BLEND):
    compile(t, "m73_cell", "exec")

nb = dict(e20)
nb["cells"] = cells
out = out_dir(a.out)
os.makedirs(out, exist_ok=True)
slug = "rsna-knee-" + a.out.replace("submission_", "")
write_json(os.path.join(out, slug + ".ipynb"), nb, indent=1)
meta = dict(e20_meta)
meta.update(id=f"{a.owner}/{slug}", title=slug, code_file=slug + ".ipynb")
mirror_src = "nartaa/rsna-knee-publication-swa-weights-20261007"
assert mirror_src in h954_meta["dataset_sources"]
meta["dataset_sources"] = meta["dataset_sources"] + [mirror_src]
write_json(os.path.join(out, "kernel-metadata.json"), meta, indent=2)
print("wrote", out, "| cells", len(cells), "| sources", len(meta["dataset_sources"]), "| kernels", meta["kernel_sources"])
