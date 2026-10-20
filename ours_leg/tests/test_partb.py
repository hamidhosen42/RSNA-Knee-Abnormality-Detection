"""H965's Part B stage on fake data, no GPU: backup cell -> heliosli's Part B cell (with our timeout patch) -> verify cell.
  normal      the reader file is rank-blended at heliosli's per-finding weights, verify accepts it, the infer call got a timeout
  inferfail   the reader subprocess fails          -> Part B restores the stack file, verify accepts it unchanged
  timeout     the reader subprocess times out      -> same
  nan         the reader writes NaN                -> same
  late        more than 8.2 h used before Part B   -> skipped, file unchanged
  badfile     a broken file reaches the verify cell -> the pre-Part-B backup is restored byte for byte
No cell may raise.

    uv run --no-project --with numpy --with pandas python -I ours_leg/tests/test_partb.py <h965 .ipynb> <scratch dir>
"""
import contextlib
import io
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

nbp, tmp = sys.argv[1], Path(sys.argv[2])
src = lambda c: "".join(c["source"])
cells = json.load(open(nbp))["cells"]
backup = next(src(c) for c in cells if src(c).startswith("# [h965] backup"))
partb = next(src(c) for c in cells if src(c).startswith("# Part B (V2)"))
verify = next(src(c) for c in cells if src(c).startswith("# [h965] verify"))
labs = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
        "Contusion", "Fracture"]
W = {"Baker's": 0.035, "Contusion": 0.03, "Medial OA": 0.025, "ACL": 0.025, "Medial Meniscus": 0.02, "MCL": 0.02, "Lateral Meniscus": 0.008,
     "Fracture": 0.008, "Lateral OA": 0.008, "PF OA": 0.008, "Effusion": 0.004, "Synovitis": 0.003}   # heliosli's v11 weights
n = 90
ids = [f"1.2.{i}" for i in range(n)]
rng = np.random.default_rng(3)
MODE, CALLS = {"m": "ok"}, []
real_run = subprocess.run


def fake_run(args, **kw):
    CALLS.append((list(args), kw))
    if "pip" in args:
        return subprocess.CompletedProcess(args, 0)
    if MODE["m"] == "inferfail":
        raise subprocess.CalledProcessError(1, args)
    if MODE["m"] == "timeout":
        raise subprocess.TimeoutExpired(args, kw.get("timeout"))
    v = rng.random((n, 12))
    if MODE["m"] == "nan":
        v[5, 3] = np.nan
    d = pd.DataFrame(v, columns=labs)
    d.insert(0, "StudyInstanceUID", ids[::-1])   # other row order on purpose: Part B must align by study
    d.to_csv(args[4], index=False)
    return subprocess.CompletedProcess(args, 0)


subprocess.run = fake_run


def pct(df):
    return df[labs].rank(method="average", pct=True)


def run(case, elapsed_h=7.0, mode="ok"):
    root = tmp / case
    shutil.rmtree(root, ignore_errors=True)
    work, inp = root / "working", root / "input"
    a, c = inp / "goodpjw2008-reader", inp / "comp"
    for p in (work, a, c):
        p.mkdir(parents=True)
    for f in ("knee.py", "cnxt_v0_fold0.pt", "cnxt_v0_fold1.pt", "cnxt_v0_fold2.pt"):
        (a / f).write_text("x")
    (c / "test_series.csv").write_text("StudyInstanceUID\n")
    st = pd.DataFrame(rng.random((n, 12)), columns=labs)
    st.insert(0, "StudyInstanceUID", ids)
    st.to_csv(work / "submission.csv", index=False)
    before = (work / "submission.csv").read_bytes()
    fix = lambda t: t.replace("/kaggle/working", str(work)).replace("/kaggle/input", str(inp))
    MODE["m"] = mode
    CALLS.clear()
    g = {"NB_T0": time.time() - elapsed_h * 3600}
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        exec(fix(backup), g)
        exec(fix(partb), g)
        if case == "badfile":   # a broken file reaches the verify cell
            (work / "submission.csv").write_text("StudyInstanceUID,ACL\n1.2.0,nan\n")
        exec(fix(verify), g)
    log = buf.getvalue()
    out = (work / "submission.csv").read_bytes()
    return st, before, out, log, g


st, before, out, log, g = run("normal")
infer = [kw for args, kw in CALLS if "infer.py" in " ".join(map(str, args))]
assert len(infer) == 1 and 60 <= infer[0]["timeout"] <= 9 * 3600 - 7.0 * 3600 - 15 * 60 + 5, infer
res = pd.read_csv(io.BytesIO(out), dtype={"StudyInstanceUID": str})
own = pd.read_csv(tmp / "normal" / "working" / "_own.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids].reset_index()
want = pd.DataFrame({c: (1 - W[c]) * pct(st)[c] + W[c] * pct(own)[c] for c in labs})
want = want.rank(method="average", pct=True)
assert res["StudyInstanceUID"].tolist() == ids and np.allclose(res[labs].values, want.values, atol=1e-9)
assert "part B file accepted" in log and '"changed": true' in log, log[-400:]
print("OK   normal     per-finding rank blend at heliosli's weights; infer call timeout %d s (7 h used); verify accepted" % infer[0]["timeout"])
for case, mode, kw in (("inferfail", "inferfail", {}), ("timeout", "timeout", {}), ("nan", "nan", {}), ("late", "ok", {"elapsed_h": 8.3})):
    st, before, out, log, g = run(case, mode=mode, **kw)
    assert out == before, case
    if case == "late":
        assert "Part B skipped" in log and not [1 for a_, _ in CALLS if "infer.py" in " ".join(map(str, a_))]
    else:
        assert "own leg FAILED, public stack submission kept" in log, (case, log[-300:])
    assert "part B file accepted" in log and '"changed": false' in log, (case, log[-300:])
    print("OK   %-10s file byte-identical to the pre-Part-B file, no exception" % case)
st, before, out, log, g = run("badfile")
assert out == before and "pre-Part-B file restored" in log, log[-300:]
print("OK   badfile    verify cell restored the pre-Part-B file byte for byte")
print("ALL PART B CHECKS PASSED")
