"""H967's MaViT leg (writer, runner, blend cells after our final cell) on fake data -- no GPU, no Kaggle.
/kaggle/working is redirected to a temp dir and the real MaViT script is replaced by a stub that writes the scenario's CSV.
11 cases: exact 0.9/0.1 rank formula, unread rows keep their rank, > 3 % unread, script failure, a missing / duplicated study,
a constant column, the 70-min start guard, the timeout kill, a stale file, values outside [0, 1].  Drafted by the 10-19
research workflow, moved here.

    uv run --no-project --with numpy --with pandas python -I ours_leg/tests/test_mavit_leg.py <h967 .ipynb>
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd

src = lambda c: "".join(c["source"])
_nb = [src(c) for c in json.load(open(sys.argv[1]))["cells"]]
writer = next(t for t in _nb if t.startswith("# [h967] backup of the H964 final file"))
runner = next(t for t in _nb if t.startswith("# [h967] MaViT in its own process group"))
blend = next(t for t in _nb if t.startswith("# [h967] final = "))
assert _nb.index(writer) < _nb.index(runner) < _nb.index(blend) and _nb.index(writer) > next(i for i, t in enumerate(_nb) if t.startswith("# [h957] final:"))
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
          "Baker's", 'Contusion', 'Fracture']
rng = np.random.default_rng(0)
N = 400
IDS = ["1.2.826.%d" % i for i in range(N)]
results = []


def run(name, mavit_df=None, script_rc=0, left_min=300, sleep=0, timeout_override=None, stale=False):
    tmp = tempfile.mkdtemp(prefix="h967_")
    W = os.path.join(tmp, "working")
    os.makedirs(W)
    fix = lambda s: s.replace("/kaggle/working", W)
    base = pd.DataFrame(rng.random((N, 12)).astype(np.float32), columns=LABELS)
    base.insert(0, "StudyInstanceUID", IDS)
    base.to_csv(os.path.join(W, "submission.csv"), index=False)
    h964_bytes = open(os.path.join(W, "submission.csv"), "rb").read()
    if stale:   # a stale MaViT file from an earlier run must never be accepted
        os.makedirs(os.path.join(W, "mavit_leg"), exist_ok=True)
        mavit_df.to_csv(os.path.join(W, "mavit_leg", "mavit_test.csv"), index=False)
    g = {"NB_T0": time.time() - (9 * 3600 - left_min * 60), "__name__": "__main__"}
    exec(fix(writer), g)
    assert g["MAVIT_SRC_OK"] and os.path.isfile(os.path.join(W, "mavit_leg", "mavit_infer.py"))
    real = open(os.path.join(W, "mavit_leg", "mavit_infer.py")).read()
    assert "7acd2a990a89c7a7d73ec0488badc5058a61b0c77142742c532ebc196749bdbc" in real and "sys.path.insert(0, _HD)" in real
    stub_csv = os.path.join(tmp, "stub_out.csv")
    if mavit_df is not None:
        mavit_df.to_csv(stub_csv, index=False)
    stub = ("import shutil, sys, time, os\n"
            "time.sleep(%r)\n"
            "if %r != 0:\n    sys.exit(%r)\n"
            "if os.path.exists(%r):\n    shutil.copyfile(%r, %r)\n" % (sleep, script_rc, script_rc, stub_csv, stub_csv,
                                                                         os.path.join(W, "mavit_leg", "mavit_test.csv")))
    open(os.path.join(W, "mavit_leg", "mavit_infer.py"), "w").write(stub)
    r = fix(runner)
    if timeout_override is not None:
        r = r.replace("timeout=max(60, int(_mr_left - 12 * 60))", "timeout=%r" % timeout_override)
    exec(r, g)
    exec(fix(blend), g)
    out = open(os.path.join(W, "submission.csv"), "rb").read()
    rec = json.load(open(os.path.join(W, "h967_receipt.json")))
    shutil.rmtree(tmp)
    return base, h964_bytes, out, rec, g


def mav(nan_rows=0, drop=0, const=False, dup=False):
    m = pd.DataFrame(rng.random((N, 12)).astype(np.float32), columns=LABELS)
    if const:
        m["Effusion"] = 0.5
    m.loc[:nan_rows - 1, LABELS] = np.nan if nan_rows else m.loc[:nan_rows - 1, LABELS]
    m.insert(0, "StudyInstanceUID", IDS)
    if drop:
        m = m.iloc[drop:]
    if dup:
        m = pd.concat([m, m.iloc[:1]])
    return m


def check(name, cond):
    results.append((name, bool(cond)))
    print("OK  " if cond else "FAIL", name)


# 1 success, all rows read: exact formula
m = mav()
base, h964, out, rec, g = run("ok", m)
o = pd.read_csv(pd.io.common.BytesIO(out), dtype={"StudyInstanceUID": str})
exp_ok = True
for c in LABELS:
    rb = pd.Series(base[c].values.astype(np.float64)).rank(method="average", pct=True).values
    rm = pd.Series(m[c].values.astype(np.float64)).rank(method="average", pct=True).values
    exp_ok &= np.allclose(o[c].values, (0.9 * rb + 0.1 * rm).astype(np.float32), atol=1e-6)
check("success: status blended, formula 0.9 rank(H964) + 0.1 rank(MaViT)", rec["status"] == "blended" and exp_ok
      and o["StudyInstanceUID"].tolist() == IDS and list(o.columns) == ["StudyInstanceUID"] + LABELS)
# 2 2 % unread rows: blended, the unread rows keep the H964 rank
m = mav(nan_rows=8)
base, h964, out, rec, g = run("nan2", m)
o = pd.read_csv(pd.io.common.BytesIO(out), dtype={"StudyInstanceUID": str})
rb = pd.Series(base["ACL"].values.astype(np.float64)).rank(method="average", pct=True).values
check("8/400 unread: blended, unread rows = H964 rank", rec["status"] == "blended" and rec["mavit_unread_rows"] == 8
      and np.allclose(o["ACL"].values[:8], rb[:8].astype(np.float32), atol=1e-6))
# 3 > 3 % unread: rejected, H964 file restored byte for byte
base, h964, out, rec, g = run("nan5", mav(nan_rows=20))
check("20/400 unread (> 3 %): H964 restored byte for byte", out == h964 and rec["status"].startswith("MaViT not used - H964 file restored"))
# 4 script fails
base, h964, out, rec, g = run("rc1", mav(), script_rc=1)
check("script rc 1: H964 restored", out == h964 and "stage not ok" in rec["status"] and rec["mavit_info"]["rc"] == 1)
# 5 missing rows
base, h964, out, rec, g = run("drop", mav(drop=1))
check("a study missing: H964 restored", out == h964 and "not exactly the submitted studies" in rec["status"])
# 6 duplicate row
base, h964, out, rec, g = run("dup", mav(dup=True))
check("a duplicated study: H964 restored", out == h964 and "not exactly the submitted studies" in rec["status"])
# 7 constant column
base, h964, out, rec, g = run("const", mav(const=True))
check("a constant column: H964 restored", out == h964 and "constant" in rec["status"])
# 8 time guard
base, h964, out, rec, g = run("late", mav(), left_min=60)
check("60 min left: skipped, H964 restored", out == h964 and "skipped" in rec["mavit_info"].get("error", "") and g["MAVIT_OK"] is False)
# 9 timeout -> process group killed
t0 = time.time()
base, h964, out, rec, g = run("timeout", mav(), sleep=30, timeout_override=2)
check("timeout: killed in ~2 s, H964 restored", out == h964 and rec["mavit_info"]["rc"] == 124 and time.time() - t0 < 20)
# 10 stale file + failing script: never accepted
base, h964, out, rec, g = run("stale", mav(), script_rc=1, stale=True)
check("stale MaViT file + failed run: H964 restored", out == h964 and "stage not ok" in rec["status"])
# 11 values outside [0, 1]
bad = mav()
bad.loc[5, "MCL"] = 1.5
base, h964, out, rec, g = run("range", bad)
check("a value > 1: H964 restored", out == h964 and "outside [0, 1]" in rec["status"])
print("%d/%d passed" % (sum(r for _, r in results), len(results)))
sys.exit(0 if all(r for _, r in results) else 1)
