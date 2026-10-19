"""H963 / H964 functional check, no GPU: run C2's final blend cell (cell 12) and the V2 version on the same fake stage
outputs.  Findings outside the V2 change must come out byte-identical; the four changed findings must equal the blend with
the V2 Raptor weights (checked by re-running C2's cell with only W_RAPTOR_C swapped by hand).

    uv run --no-project --with numpy --with pandas --with scipy python -I ours_leg/tests/test_v2_blend.py <h963 or h964 .ipynb> <scratch dir>
"""
import contextlib
import io
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

nbp, tmp = sys.argv[1], Path(sys.argv[2])
src = lambda c: "".join(c["source"])
ROOT = Path(__file__).resolve().parents[2]
c2_blend = src(json.load(open(ROOT / "our_submission_notebook/submission_c2/rsna-knee-sub-c2.ipynb"))["cells"][12])
v2_blend = src(json.load(open(nbp))["cells"][12])
OLD = "W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, \"Baker's\": 0.35}"
V2 = "W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.15, 'PF OA': 0.65, 'Synovitis': 0.85, \"Baker's\": 0.30}"
assert v2_blend == c2_blend.replace(OLD, V2) and v2_blend != c2_blend
labs = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
        "Contusion", "Fracture"]
CHANGED = {"Lateral OA", "PF OA", "Synovitis", "Baker's"}
n = 150
ids = [f"1.2.{i}" for i in range(n)]
rng = np.random.default_rng(11)
stage = {k: rng.random((n, 12)) for k in ("raw", "raw224", "ours", "s6f")}
stage["s6f"][[3, 77]] = np.nan   # two studies S6fall-B did not score (keep the 0.958 value)


def frame(v):
    d = pd.DataFrame(v, columns=labs)
    d.insert(0, "StudyInstanceUID", ids)
    return d


def run(code, tag):
    root = tmp / tag
    shutil.rmtree(root, ignore_errors=True)
    DATA, WORK, pred = root / "data", root / "work", root / "pred"
    for p in (DATA, WORK / "k1" / "B", pred / "worker0"):
        p.mkdir(parents=True)
    pd.DataFrame({"StudyInstanceUID": ids}).to_csv(DATA / "test.csv", index=False)
    pd.DataFrame(columns=["StudyInstanceUID"] + labs).to_csv(DATA / "sample_submission.csv", index=False)
    for name, key in (("raptor", "raw"), ("raptor224", "raw224")):
        frame(stage[key]).to_csv(WORK / f"{name}_raw.csv", index=False)
        s = frame(stage[key]); s[labs] = s[labs].rank(pct=True); s.to_csv(WORK / f"{name}_sub.csv", index=False)
    frame(stage["ours"]).to_csv(WORK / "ours_sub.csv", index=False)
    frame(stage["ours"]).to_csv(pred / "worker0" / "predictions.csv", index=False)   # the R6 worker output C2's cell 12 re-reads
    frame(stage["s6f"]).to_csv(WORK / "k1" / "B" / "B_probs.csv", index=False)
    g = {"DATA": DATA, "WORK": WORK, "pred": pred, "NB_T0": time.time(), "RAPTOR_SECONDS": 1.0, "RAPTOR224_SECONDS": 1.0, "S6F_SECONDS": 1.0,
         "time": time, "json": json, "display": lambda *a, **k: None}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(code, g)
    return pd.read_csv(WORK / "submission.csv", dtype={"StudyInstanceUID": str}), json.load(open(WORK / "blend_receipt.json"))


base, _ = run(c2_blend, "c2")
new, rec = run(v2_blend, "v2")
assert base["StudyInstanceUID"].tolist() == new["StudyInstanceUID"].tolist() == ids
for c in labs:
    same = np.array_equal(base[c].values, new[c].values)
    assert same == (c not in CHANGED), (c, same)
assert rec["weights"]["raptor_per_class"] == {"Medial OA": 0.5, "Lateral OA": 0.15, "PF OA": 0.65, "Synovitis": 0.85, "Baker's": 0.30}
assert rec["s6fall_b_failed"] == 2 and rec["weights"]["s6fall_b"] == 1.0 / 3.0
print("OK   8 findings byte-identical to C2; Lateral OA / PF OA / Synovitis / Baker's changed; receipt shows the V2 weights;"
      " S6fall-B 1/3 with 2 unscored studies kept")
print("ALL V2 BLEND CHECKS PASSED")
