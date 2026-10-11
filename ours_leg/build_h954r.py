"""H954R: the 0.954 public notebook (heliosli/rsna-knee-abnormality-detection v1 = Anatomical Mirror Raptor 0.7 +
compact convnext_small S6 0.3) plus goodpjw2008's 2.5D ConvNeXt reader as a third rank-blend leg.

Single change against submission_h954: the reader stage of samanyu1808/rsna-knee-blend3 v3 (its cells 5-9, verbatim:
goodpjw2008/rsna-knee-2-5d-convnext-reader, Apache-2.0, every cnxt_v0_fold*.pt) runs after the compact stage, and the
blend cell becomes a 3-way per-finding percentile-rank mean with blend3's published weights
{raptor 0.5, compact 0.25, reader 0.25} - copied, not fitted on the LB or on any labels.  blend3 used the same
weights with the older compact S5 and scored 0.951 = blend-gold (Mirror 0.7 + S5 0.3), so the reader leg did not
hurt there.  A study a leg failed on is dropped from that study's mean (weights renormalised); a reader that failed
as a whole leaves exactly the parent's 2-way blend, and that is printed.

    python build_h954r.py [--parent submission_h954] [--blend3 <public blend3 ipynb>] [--out submission_h954r]
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--parent", default="submission_h954")
ap.add_argument("--blend3", default="public_notebooks/0.951_2v_samanyu1808__rsna-knee-blend3-please-upvote-if-useful.ipynb")
ap.add_argument("--out", default="submission_h954r")
ap.add_argument("--owner", default="hosen42")
ap.add_argument("--reader", choices=["public", "b"], default="public",
                help="public = goodpjw2008's 3 v0 folds (blend3 verbatim); b = our pseudo-label reader B "
                     "(esfersami50/rsna-knee-gpjw-pseudo v1, leg alone 0.939)")
a = ap.parse_args()

par = os.path.join(ROOT, a.parent)
nb_name = next(f for f in os.listdir(par) if f.endswith(".ipynb"))
nb = json.load(open(os.path.join(par, nb_name)))
b3 = json.load(open(os.path.join(ROOT, a.blend3)))["cells"]
cells = nb["cells"]
assert len(cells) == 7 and "W_RAPTOR, W_OURS = 0.7, 0.3" in "".join(cells[6]["source"])
assert "".join(b3[9]["source"]).startswith("# Stage 2b: goodpjw2008 ConvNeXt 2.5D reader")
reader_cells = [json.loads(json.dumps(b3[i])) for i in (5, 6, 7, 8, 9)]
for c in reader_cells:
    c["outputs"], c["execution_count"] = [], None
if a.reader == "b":
    _r = "".join(reader_cells[-1]["source"])
    _old = '_ck = sorted(glob.glob(_assets + "/cnxt_v0_fold*.pt")); print("reader ckpts", _ck, flush=True)'
    assert _r.count(_old) == 1
    _r = _r.replace(_old, '# [h954r-B] reader = our full-data pseudo-label reader B instead of the 3 public v0 folds\n'
                    '_ck = sorted(glob.glob("/kaggle/input/**/rsna-knee-gpjw-pseudo/cnxt_plB_full.pt", recursive=True))\n'
                    'assert len(_ck) == 1, _ck; print("reader ckpts", _ck, flush=True)')
    reader_cells[-1]["source"] = _r.splitlines(keepends=True)
    compile(_r, "reader_cell", "exec")

blend = cells[6]
s = "".join(blend["source"])


def sub(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)


sub("""# Stage 3: per-finding rank blend with weights fixed before any run: Raptor 0.7, ours 0.3.  Not fitted on any labels:
# Raptor trained on every training study and was selected on the 58 gold, so no clean labelled set exists to fit them.
# A study that one model failed on takes the other model's rank; more than 3 % Raptor failures stops the notebook.""",
    """# Stage 3 [h954r]: 3-way per-finding rank blend, weights copied from samanyu1808/rsna-knee-blend3 v3 (not fitted):
# Raptor 0.5, compact S6 0.25, goodpjw2008 reader 0.25.  A study that a leg failed on is dropped from that study's
# mean and the other weights renormalised; more than 3 % Raptor failures stops the notebook.""")
sub("W_RAPTOR, W_OURS = 0.7, 0.3", "W_RAPTOR, W_OURS, W_READER = 0.5, 0.25, 0.25")
sub("""if rap_fail.sum() > 0.03 * n:
    raise RuntimeError('Raptor failed on %d of %d studies' % (rap_fail.sum(), n))
""", """if rap_fail.sum() > 0.03 * n:
    raise RuntimeError('Raptor failed on %d of %d studies' % (rap_fail.sum(), n))
# [h954r] reader leg: its own failures are studies with no usable series (all-NaN row) or a missing / mostly-bad file
reader = None
if READER_OK:
    reader = pd.read_csv(WORK / 'reader_sub.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    reader = reader.reindex(test_ids)[labels]
    if reader.notna().values.mean() < 0.97 or not (reader.nunique() > 1).any():
        print('READER_REJECTED', float(reader.notna().values.mean()), flush=True)
        reader = None
read_fail = np.ones(n, bool) if reader is None else ~np.isfinite(reader.values).all(axis=1)
print('READER_LEG', 'used' if reader is not None else 'NOT USED (2-way parent blend)', 'failed studies', int(read_fail.sum()), flush=True)
""")
sub("""    rr, ro = pct(raw[c].values), pct(ours[c].values)
    b = W_RAPTOR * rr + W_OURS * ro
    b[rap_fail] = ro[rap_fail]
    b[our_fail] = rr[our_fail]
    b[rap_fail & our_fail] = 0.5
    out[:, j] = b""", """    rr, ro = pct(raw[c].values), pct(ours[c].values)
    rd = pct(reader[c].values) if reader is not None else np.zeros(n)
    wr = W_RAPTOR * ~rap_fail
    wo = W_OURS * ~our_fail
    wd = W_READER * ~read_fail
    den = wr + wo + wd
    b = np.where(den > 0, (wr * rr + wo * ro + wd * np.nan_to_num(rd)) / np.maximum(den, 1e-9), 0.5)
    out[:, j] = b""")
sub("""blend = {'weights': {'raptor': W_RAPTOR, 'ours': W_OURS}, 'rows': n, 'raptor_failed': int(rap_fail.sum()),
         'ours_failed': int(our_fail.sum()),""", """blend = {'weights': {'raptor': W_RAPTOR, 'ours': W_OURS, 'reader': W_READER}, 'rows': n,
         'raptor_failed': int(rap_fail.sum()), 'ours_failed': int(our_fail.sum()),
         'reader_used': reader is not None, 'reader_failed': int(read_fail.sum()),""")
sub("""         'spearman_raptor_vs_ours': rho_ro}""", """         'spearman_raptor_vs_ours': rho_ro,
         'spearman_raptor_vs_reader': {c: round(float(pd.Series(raw[c].values).corr(pd.Series(reader[c].values),
                                       method='spearman')), 4) for c in labels} if reader is not None else None}""")
blend["source"] = s.splitlines(keepends=True)
blend["outputs"], blend["execution_count"] = [], None

cells[0]["source"] = ("".join(cells[0]["source"]) + "\n\n**h954r (ours):** adds goodpjw2008's 2.5D ConvNeXt reader "
                      "(stage 2b, from samanyu1808/rsna-knee-blend3 v3) and blends Raptor 0.5 / compact S6 0.25 / "
                      "reader 0.25, the weights blend3 published.").splitlines(keepends=True)
nb["cells"] = cells[:6] + reader_cells + [blend]
for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
compile(s, "blend_cell", "exec")

out = os.path.join(ROOT, a.out)
os.makedirs(out, exist_ok=True)
slug = "rsna-knee-" + a.out.replace("submission_", "")
json.dump(nb, open(os.path.join(out, slug + ".ipynb"), "w"), indent=1)
meta = json.load(open(os.path.join(par, "kernel-metadata.json")))
meta.update(id=f"{a.owner}/{slug}", title=slug, code_file=slug + ".ipynb")
if "goodpjw2008/rsna-knee-2-5d-convnext-reader" not in meta["dataset_sources"]:
    meta["dataset_sources"] = meta["dataset_sources"] + ["goodpjw2008/rsna-knee-2-5d-convnext-reader"]
if a.reader == "b":
    meta["kernel_sources"] = list(meta.get("kernel_sources") or []) + ["esfersami50/rsna-knee-gpjw-pseudo/1"]
json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
print("wrote", out, "| cells", len(nb["cells"]), "| sources", meta["dataset_sources"])
