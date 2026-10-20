"""H968's student reader block (the [h968] TTA block of cell 16) on fake reader outputs, subprocess mocked, no GPU.
Cases on 1,284 studies: a consistent mirror is accepted (reader file = 0.5 x (plain + mirror)); a side-swapped mirror and a
broken finding are rejected (reader file = the plain pass = H964's students); a crashed infer_tta.py reruns H964's infer.py;
a timeout stays a timeout (READER_OK False, as in H964); a child killed by a signal (rc -9, e.g. out of memory) reruns
H964's infer.py like any crash; a non-finite mirror value is rejected (plain pass); a guard error falls back to the plain
pass; a missing plain file leaves no reader file (READER_OK False).  On 3 studies the guard only logs (accepted).  No case
may raise.

    uv run --no-project --with numpy --with pandas python -I ours_leg/tests/test_student_tta.py <h968 .ipynb> <scratch dir>
"""
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
import types

import numpy as np
import pandas as pd

nbp, work = sys.argv[1], sys.argv[2]
src = lambda c: "".join(c["source"])
run_cell = next(src(c) for c in json.load(open(nbp))["cells"] if src(c).startswith("# [h960r] the whole optional"))
i0 = run_cell.index('    for _stale in ("reader_sub.csv", "reader_sub_plain.csv", "reader_sub_mirror.csv"):')
i1 = run_cell.index('    READER_INFO["reader_rc"] = rc\n') + len('    READER_INFO["reader_rc"] = rc\n')
block = textwrap.dedent(run_cell[i0:i1])
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
          "Synovitis", "Baker's", "Contusion", "Fracture"]
rng = np.random.default_rng(0)


def data(n):
    ids = [f"s{i}" for i in range(n)]
    b = pd.DataFrame(rng.random((n, 12)), index=pd.Index(ids, name="StudyInstanceUID"), columns=LABELS)
    b["Lateral Meniscus"] = 0.6 * b["Medial Meniscus"] + 0.4 * rng.random(n)   # medial/lateral correlated as in real data
    b["Lateral OA"] = 0.6 * b["Medial OA"] + 0.4 * rng.random(n)
    return b


def case(kind, n=1284):
    d = os.path.join(work, kind)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    open(os.path.join(d, "reader_sub.csv"), "w").write("stale\n")   # must be removed before the run
    base = data(n)
    calls = []

    def fake_run(cmd, env=None, timeout=None):
        script, out = os.path.basename(cmd[1]), cmd[4]
        calls.append((script, timeout))
        if kind == "crash" and script == "infer_tta.py":
            return types.SimpleNamespace(returncode=1)
        if kind == "killed" and script == "infer_tta.py":   # SIGKILL (e.g. the out-of-memory killer): rc -9, not a timeout
            return types.SimpleNamespace(returncode=-9)
        if kind == "timeout":
            raise subprocess.TimeoutExpired(cmd, timeout)
        p = base.copy()
        q = (base + 0.03 * rng.standard_normal(base.shape)).clip(0, 1)
        if kind == "swapped":
            q[["Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA"]] = q[["Lateral Meniscus", "Medial Meniscus", "Lateral OA", "Medial OA"]].values
        if kind == "broken":
            q["Synovitis"] = rng.random(n)
        if kind == "mirrornan":   # Spearman skips NaN pairs: only the finiteness check can catch this
            q.iloc[7, 2] = np.nan
        if script == "infer_tta.py":
            if kind != "noplain":
                p.to_csv(out[:-4] + "_plain.csv")
            if kind == "guarderr":
                q.drop(columns=["Medial OA"]).to_csv(out[:-4] + "_mirror.csv")   # a column missing: the guard raises
            else:
                q.to_csv(out[:-4] + "_mirror.csv")
            (0.5 * (p + q)).to_csv(out)
        else:
            p.to_csv(out)
        return types.SimpleNamespace(returncode=0)

    sub = types.SimpleNamespace(run=fake_run, TimeoutExpired=subprocess.TimeoutExpired)
    g = {"os": os, "time": time, "sys": sys, "subprocess": sub, "_rp": pd, "_data": "DATA", "_ck": ["a.pt"], "_env": {},
         "NB_T0": time.time() - 3600, "READER_INFO": {}}
    exec(compile(block.replace("/kaggle/working/", d + "/"), "h968_block", "exec"), g)   # must never raise
    f = os.path.join(d, "reader_sub.csv")
    final = pd.read_csv(f, index_col=0) if os.path.exists(f) else None
    return base, final, g, calls


def is_plain(base, final):
    return final is not None and np.allclose(final.values, base.values)


base, final, g, calls = case("good")
assert g["READER_OK"] and g["READER_INFO"]["tta"]["accepted"] and g["READER_INFO"]["tta"]["used"] == "0.5 x (plain + mirror)"
assert not is_plain(base, final) and [c for c, _ in calls] == ["infer_tta.py"] and 60 <= calls[0][1] <= 8 * 3600 - 20 * 60 + 5
print("OK   good       accepted: reader file = 0.5 x (plain + mirror); one call, timeout %d s" % calls[0][1], g["READER_INFO"]["tta"])
for kind in ("swapped", "broken"):
    base, final, g, calls = case(kind)
    assert g["READER_OK"] and g["READER_INFO"]["tta"]["accepted"] is False and is_plain(base, final), (kind, g["READER_INFO"])
    print("OK   %-10s rejected by the guard: reader file = the plain pass" % kind, {k: g["READER_INFO"]["tta"][k] for k in ("min_spearman", "ml_margin")})
base, final, g, calls = case("crash")
assert g["READER_OK"] and [c for c, _ in calls] == ["infer_tta.py", "infer.py"] and is_plain(base, final) and g["READER_INFO"]["tta_rc"] == 1
print("OK   crash      infer_tta.py rc 1 -> H964's infer.py rerun, reader file = plain")
base, final, g, calls = case("timeout")
assert g["READER_OK"] is False and g["READER_INFO"]["reader_rc"] == "timeout" and [c for c, _ in calls] == ["infer_tta.py"] and final is None
print("OK   timeout    READER_OK False, no rerun (as H964), stale file removed")
base, final, g, calls = case("killed")
assert g["READER_OK"] and [c for c, _ in calls] == ["infer_tta.py", "infer.py"] and is_plain(base, final) and g["READER_INFO"]["tta_rc"] == -9
print("OK   killed     infer_tta.py rc -9 (signal) -> H964's infer.py rerun, reader file = plain")
base, final, g, calls = case("mirrornan")
assert g["READER_OK"] and g["READER_INFO"]["tta"]["accepted"] is False and g["READER_INFO"]["tta"]["finite"] is False and is_plain(base, final)
print("OK   mirrornan  a NaN in the mirror pass -> rejected, reader file = plain (finite)")
base, final, g, calls = case("guarderr")
assert g["READER_OK"] and g["READER_INFO"]["tta"]["accepted"] is False and "guard_error" in g["READER_INFO"]["tta"] and is_plain(base, final)
print("OK   guarderr   guard raised -> plain pass used, no exception")
base, final, g, calls = case("noplain")
assert g["READER_OK"] is False and g["READER_INFO"]["reader_rc"] == -8
print("OK   noplain    no plain file -> no reader file, READER_OK False (final cell keeps the parent), no exception")
base, final, g, calls = case("small3", n=3)
assert g["READER_OK"] and g["READER_INFO"]["tta"]["accepted"] and "logged only" in g["READER_INFO"]["tta"]["guard"] and not is_plain(base, final)
print("OK   3 studies  guard logged only, TTA accepted (the smoke run shows the accepted path)")
print("ALL STUDENT TTA BLOCK CHECKS PASSED")
