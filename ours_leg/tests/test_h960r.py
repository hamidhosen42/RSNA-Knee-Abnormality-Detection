"""The hardened student leg (H960r / H960RM / H961) on fake data, no GPU:
  (A) the parent's own blend cell -> snapshot -> writer -> strict final, 12 cases (accepted blend, every rejection, receipt failure);
  (B) the guarded reader-run cell, 8 cases (checkpoint sha256, missing checkpoint, wrong data, time left, timeout, sources);
  (C) end to end writer -> run (fake reader) -> final, 4 cases.
The parent blend is the 0.958 notebook's stage 3 (H960r / H960RM) or C2's final cell, which also reads S6fall-B (H961).

    uv run --no-project --with numpy --with pandas --with scipy python -I ours_leg/tests/test_h960r.py <notebook.ipynb> <scratch dir>
"""
import json, sys, shutil, time, re, contextlib, io, hashlib, subprocess, types
from pathlib import Path
import numpy as np, pandas as pd
nbp, tmp = sys.argv[1], Path(sys.argv[2])
cells = json.load(open(nbp))["cells"]; src = lambda c: "".join(c["source"])
def pick(cond):
    m = [src(c) for c in cells if c["cell_type"] == "code" and cond(src(c))]
    assert len(m) == 1, len(m)
    return m[0]
blend = pick(lambda t: "W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': " in t and t.startswith(("# Stage 3", "# [C2 10-17] final")))   # 0.2, or 0.15 (V2)
C2 = blend.startswith("# [C2 10-17] final")   # H961: the parent blend also reads nartaa's S6fall-B file


def s6f_file(WORK, n_ids):   # H961: the S6fall-B probabilities C2's stage 1c leaves (k1/B/B_probs.csv)
    if C2:
        (WORK / "k1" / "B").mkdir(parents=True, exist_ok=True)
        frame(rng.random((n_ids, 12))).to_csv(WORK / "k1" / "B" / "B_probs.csv", index=False)
snap = pick(lambda t: t.startswith("# [h960r] the parent notebook"))
writer = pick(lambda t: t.startswith("# [h960r] the student sources"))
run_cell = pick(lambda t: t.startswith("# [h960r] the whole optional"))
final = pick(lambda t: t.startswith("# [h957] final:"))
assert "E1_INVALID" in final and "PARENT_LABEL" in snap
WP, WR = map(float, re.search(r"^W_PARENT, W_READER = ([0-9.]+), ([0-9.]+)", final, re.M).groups())   # 0.8/0.2, or e.g. 0.7/0.3 (H966)
assert abs(WP + WR - 1) < 1e-9, (WP, WR)
order = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and src(c) in (blend, snap, writer, run_cell, final)]
assert [src(cells[i]) for i in order] == [blend, snap, writer, run_cell, final], "cell order"
labs = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
WC = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, "Baker's": 0.35}
n = 120; ids = [f"1.2.{i}" for i in range(n)]; rng = np.random.default_rng(5)
def frame(v, idl=None): d = pd.DataFrame(v, columns=labs); d.insert(0, "StudyInstanceUID", idl or ids); return d
def pct(v): return pd.Series(np.asarray(v, float)).rank(method="average", pct=True).values
N_CK = int(re.search(r"_ri.get\('n_ckpt'\) != (\d+)", final).group(1))   # 5, or 7 with the S1 students (H976)
S1 = "_s1_ckpts(" in run_cell
assert (N_CK == 7) == S1, (N_CK, S1)
GOOD = {"n_ckpt": N_CK, "ckpt_sha_ok": True}
cases = {"snapfail": {"snapfail": True}, "snapchanged": {"snapchg": True}, "receiptfail": {"receipt": True}, "normal": {}, "readerfail": {"ok": False}, "nanrow": {"nan": True}, "dupuid": {"dup": True}, "missinguid": {"miss": True},
         "constcol": {"const": True}, "outofrange": {"oor": True}, "badckpt": {"info": {"n_ckpt": N_CK - 1, "ckpt_sha_ok": True}}, "noinfo": {"info": {}}}
for case, kw in cases.items():
    root = tmp / "A" / case; shutil.rmtree(root, ignore_errors=True)
    DATA, WORK, pred = root / "data", root / "work", root / "pred"
    for d in (DATA, WORK, pred / "worker0"): d.mkdir(parents=True)
    pd.DataFrame({"StudyInstanceUID": ids}).to_csv(DATA / "test.csv", index=False)
    pd.DataFrame(columns=["StudyInstanceUID"] + labs).to_csv(DATA / "sample_submission.csv", index=False)
    raw, raw224, ours = rng.random((n, 12)), rng.random((n, 12)), rng.random((n, 12))
    frame(raw).to_csv(WORK / "raptor_raw.csv", index=False); s = frame(raw); s[labs] = s[labs].rank(pct=True); s.to_csv(WORK / "raptor_sub.csv", index=False)
    frame(raw224).to_csv(WORK / "raptor224_raw.csv", index=False); s = frame(raw224); s[labs] = s[labs].rank(pct=True); s.to_csv(WORK / "raptor224_sub.csv", index=False)
    s6f_file(WORK, n)
    frame(ours).to_csv(WORK / "ours_sub.csv", index=False); frame(ours).to_csv(pred / "worker0" / "predictions.csv", index=False)
    st = rng.random((n, 12)); idl = list(ids)
    if kw.get("nan"): st[3] = np.nan
    if kw.get("const"): st[:, 5] = 0.4
    if kw.get("oor"): st[7, 2] = 1.5
    if kw.get("dup"): idl[10] = idl[11]
    rd = frame(st, idl)
    if kw.get("miss"): rd = rd.iloc[1:]
    rd.to_csv(WORK / "reader_sub.csv", index=False)
    if kw.get("receipt"): (WORK / "h960_receipt.json").mkdir()   # writing the receipt will fail
    g = {"DATA": DATA, "WORK": WORK, "pred": pred, "NB_T0": time.time(), "RAPTOR_SECONDS": 1.0, "RAPTOR224_SECONDS": 1.0, "S6F_SECONDS": 1.0, "time": time,
         "json": json, "display": lambda *a, **k: None, "READER_OK": kw.get("ok", True), "READER_INFO": kw.get("info", GOOD)}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(blend, g)
        if kw.get("snapfail"):   # the snapshot copy itself fails: the whole optional stage must skip without raising
            (WORK / "_baseline_submission.csv").mkdir(); parent_bytes = (WORK / "submission.csv").read_bytes()
            sys.modules["torch"] = types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None))
            exec(snap, g); exec(writer.replace("/kaggle/working", str(WORK)), g)
            exec(run_cell.replace("/kaggle/working", str(WORK)), g)   # V31-3: the reader-run cell is exercised too
            exec(final, g)
        else:
            exec(snap, g)
    if kw.get("snapfail"):
        assert g["BASELINE_SHA256"] is None and g["READER_SRC_OK"] is False and g["READER_OK"] is False and (WORK / "submission.csv").read_bytes() == parent_bytes
        assert "student sources were not written" in g["READER_INFO"].get("error", ""), g["READER_INFO"]
        print(f"A {case:10s} OK | snapshot failed -> writer skip -> reader skip -> final: no exception, parent's submission.csv left in place"); continue
    with contextlib.redirect_stdout(io.StringIO()):
        base_bytes = (WORK / "_baseline_submission.csv").read_bytes()
        if kw.get("snapchg"):
            with open(WORK / "_baseline_submission.csv", "ab") as fh: fh.write(b"\n")   # the snapshot is altered after it was taken
        exec(final, g)
    out = pd.read_csv(WORK / "submission.csv", dtype={"StudyInstanceUID": str})
    if kw.get("receipt"):
        assert not (WORK / "submission.csv").read_bytes() == base_bytes   # the valid blend is kept although the receipt failed
        print(f"A {case:10s} OK | receipt write failed, notebook did not fail, blend kept"); continue
    rec = json.load(open(WORK / "h960_receipt.json"))
    if kw.get("snapchg"):
        assert (WORK / "submission.csv").read_bytes() == base_bytes and "snapshot changed" in rec["status"], rec["status"]
        print(f"A {case:10s} OK | altered snapshot rejected; the parent's own submission.csv left in place"); continue
    if case == "normal":
        par = pd.read_csv(WORK / "_baseline_submission.csv", dtype={"StudyInstanceUID": str})
        for j, c in enumerate(labs):
            assert np.allclose(out[c].values, WP * pct(par[c].values) + WR * pct(st[:, j]), atol=1e-6), c
        assert rec["status"] == "blended"
        print(f"A {case:10s} OK | {WP:g} x rank(snapshot) + {WR:g} x rank(students) | receipt parent_sha {rec['parent_sha256'][:12]}")
    else:
        assert (WORK / "submission.csv").read_bytes() == base_bytes, case   # the snapshot, byte for byte
        assert rec["status"].startswith("E1 INVALID - parent snapshot kept"), rec["status"]
        print(f"A {case:10s} OK | parent snapshot kept byte-identical | {rec['status'][:60]}")
# ---------------- (B) guarded reader-run cell
sys.modules["torch"] = types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None))
calls, MODE = [], {"m": "ok"}
real_run = subprocess.run
def fake_run(args, **kw):
    calls.append((list(args), kw))
    if "pip" in args: return subprocess.CompletedProcess(args, 1)       # pip fails, as in the real logs
    if MODE["m"] == "timeout": raise subprocess.TimeoutExpired(args, kw.get("timeout"))
    Path(args[4]).write_text("StudyInstanceUID,ACL\n1.2.3,0.5\n")
    if str(args[1]).endswith("infer_tta.py"):   # H968: the TTA script also writes its plain and mirror passes
        for _sfx in ("_plain", "_mirror"):
            Path(str(args[4])[:-4] + _sfx + ".csv").write_text("StudyInstanceUID,ACL\n1.2.3,0.5\n")
    return subprocess.CompletedProcess(args, 0)
subprocess.run = fake_run
kw_writer_fail = {"writerfail": True}
OLD = ["cnxt_plC2_full.pt", "cnxt_plC3_full.pt", "cnxt_plC_full.pt", "cnxt_plS75_full.pt", "cnxt_plT75s1_full.pt"]
S1_NAMES = ["cnxt_plE75s_full.pt", "cnxt_plE75t_full.pt"]


def s1_files(inp, fake_sha, drop=None):   # H976: the S1 students, mounted like a kernel source
    if S1:
        d = inp / "notebooks/hosen42/rsna-knee-gpjw-student5"; d.mkdir(parents=True)
        for k in S1_NAMES:
            if k != drop:
                (d / k).write_bytes(k.encode()); fake_sha[k] = hashlib.sha256(k.encode()).hexdigest()
def runB(case, good_sha=True, drop=None, test_ids=None, elapsed_h=4.0, mode="ok"):
    root = tmp / "B" / case; shutil.rmtree(root, ignore_errors=True)
    inp, work = root / "input", root / "working"; (work / "own_src").mkdir(parents=True)
    a = inp / "datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader"; a.mkdir(parents=True); (a / "cnxt_v0_fold0.pt").touch()
    c = inp / "competitions/rsna-knee-abnormality-detection"; (c / "test_series").mkdir(parents=True); (c / "test_series.csv").touch()
    pd.DataFrame({"StudyInstanceUID": test_ids or ["1.2.3", "1.2.4"]}).to_csv(c / "test.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": ["1.2.3", "1.2.4"], "ACL": [0.1, 0.2]}).to_csv(work / "_baseline_submission.csv", index=False)
    o = inp / "datasets/hosen42/rsna-knee-student-readers"; o.mkdir(parents=True); fake_sha = {}
    for k in OLD:
        if k != drop:
            (o / k).write_bytes(k.encode()); fake_sha[k] = hashlib.sha256(k.encode()).hexdigest()
    s1_files(inp, fake_sha, drop)
    wcode = writer.replace("/kaggle/working", str(work))
    if kw_writer_fail.get(case): shutil.rmtree(work / "own_src"); (work / "own_src").write_text("x")   # makedirs/open will fail
    code = run_cell.replace("/kaggle/input", str(inp)).replace("/kaggle/working", str(work))
    if good_sha:   # the test files are not the real checkpoints: point the expected hashes at them
        code = re.sub(r"_SHA = \{[^}]*\}", "_SHA = " + repr(fake_sha), code)
    calls.clear(); MODE["m"] = mode; g = {"NB_T0": time.time() - elapsed_h * 3600, "BASELINE_SHA256": "snapshot-taken"}
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        exec(wcode, g); exec(code, g)    # must never raise
    return g["READER_OK"], g["READER_INFO"], buf.getvalue()
ok, info, log = runB("normal"); assert ok and info["n_ckpt"] == N_CK and info["ckpt_sha_ok"] and info["pip_rc"] == 1 and info["reader_rc"] == 0, info
print(f"B normal      OK | {N_CK} checkpoints sha-verified, data identity ok, pip rc recorded (non-fatal), READER_OK True")
for case, kw, want in ((("s1missing", {"drop": "cnxt_plE75t_full.pt"}, "FileNotFoundError"),) if S1 else ()) + (("realsha", {"good_sha": False}, "differ from the validated"), ("missingckpt", {"drop": "cnxt_plS75_full.pt"}, "FileNotFoundError"),
                       ("wrongdata", {"test_ids": ["9.9.9"]}, "does not hold the submitted studies"), ("late", {"elapsed_h": 8.5}, "min left"),
                       ("timeout", {"mode": "timeout"}, "READER_TIMEOUT"), ("writerfail", {}, "student sources were not written"),
                       ("latestart", {"elapsed_h": 8.2}, "at the start of the student stage")):
    ok, info, log = runB(case, **kw)
    assert ok is False and want in (log + json.dumps(info)), (case, info, log[-300:])
    print(f"B {case:12s} OK | no exception, READER_OK False | {want}")
print("ALL H960r CHECKS PASSED")
# ---------------- (C) end to end: writer -> guarded run (fake reader writes a full 12-target file) -> strict final
def runC(case, bad_reader=None):
    root = tmp / "C" / case; shutil.rmtree(root, ignore_errors=True)
    DATA, WORK, pred = root / "data", root / "work", root / "pred"
    for d in (DATA, WORK, pred / "worker0"): d.mkdir(parents=True)
    inp = root / "input"
    a = inp / "datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader"; a.mkdir(parents=True); (a / "cnxt_v0_fold0.pt").touch()
    c = inp / "competitions/rsna-knee-abnormality-detection"; (c / "test_series").mkdir(parents=True); (c / "test_series.csv").touch()
    pd.DataFrame({"StudyInstanceUID": ids}).to_csv(c / "test.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": ids}).to_csv(DATA / "test.csv", index=False)
    pd.DataFrame(columns=["StudyInstanceUID"] + labs).to_csv(DATA / "sample_submission.csv", index=False)
    raw, raw224, ours = rng.random((n, 12)), rng.random((n, 12)), rng.random((n, 12))
    frame(raw).to_csv(WORK / "raptor_raw.csv", index=False); s_ = frame(raw); s_[labs] = s_[labs].rank(pct=True); s_.to_csv(WORK / "raptor_sub.csv", index=False)
    frame(raw224).to_csv(WORK / "raptor224_raw.csv", index=False); s_ = frame(raw224); s_[labs] = s_[labs].rank(pct=True); s_.to_csv(WORK / "raptor224_sub.csv", index=False)
    s6f_file(WORK, n)
    frame(ours).to_csv(WORK / "ours_sub.csv", index=False); frame(ours).to_csv(pred / "worker0" / "predictions.csv", index=False)
    o = inp / "datasets/hosen42/rsna-knee-student-readers"; o.mkdir(parents=True); fake_sha = {}
    for k in OLD:
        (o / k).write_bytes(k.encode()); fake_sha[k] = hashlib.sha256(k.encode()).hexdigest()
    s1_files(inp, fake_sha)
    st = rng.random((n, 12))
    (WORK / "reader_sub.csv").write_text("stale file from an earlier run\n")        # N1: must be removed before the run
    def reader_run(args, **kw):
        calls.append((list(args), kw))
        if "pip" in args: return subprocess.CompletedProcess(args, 1)
        d = frame(st)
        if bad_reader == "missing": d = d.iloc[2:]
        if bad_reader == "nan": d.iloc[5, 3] = np.nan
        if bad_reader == "nowrite": return subprocess.CompletedProcess(args, 0)     # rc 0 but no fresh file
        d.to_csv(args[4], index=False)
        if str(args[1]).endswith("infer_tta.py"):   # H968: plain = mirror = the reader file, so 0.5 x (plain + mirror) = it
            for _sfx in ("_plain", "_mirror"):
                d.to_csv(str(args[4])[:-4] + _sfx + ".csv", index=False)
        return subprocess.CompletedProcess(args, 0)
    subprocess.run = reader_run
    g = {"DATA": DATA, "WORK": WORK, "pred": pred, "NB_T0": time.time(), "RAPTOR_SECONDS": 1.0, "RAPTOR224_SECONDS": 1.0, "S6F_SECONDS": 1.0, "time": time,
         "json": json, "display": lambda *a, **k: None}
    wcode = writer.replace("/kaggle/working", str(WORK))
    rcode = re.sub(r"_SHA = \{[^}]*\}", "_SHA = " + repr(fake_sha), run_cell.replace("/kaggle/input", str(inp)).replace("/kaggle/working", str(WORK)))
    with contextlib.redirect_stdout(io.StringIO()):
        exec(blend, g); exec(snap, g); base_bytes = (WORK / "_baseline_submission.csv").read_bytes()
        exec(wcode, g); exec(rcode, g); exec(final, g)
    rec = json.load(open(WORK / "h960_receipt.json")); out = pd.read_csv(WORK / "submission.csv", dtype={"StudyInstanceUID": str})
    return g, rec, out, base_bytes, st, WORK
g, rec, out, base_bytes, st, WORK = runC("full")
par = pd.read_csv(WORK / "_baseline_submission.csv", dtype={"StudyInstanceUID": str})
assert rec["status"] == "blended" and g["READER_OK"] and rec["reader_info"]["n_ckpt"] == N_CK and "parent" in rec["weights"], rec
for j, c in enumerate(labs):
    assert np.allclose(out[c].values, WP * pct(par[c].values) + WR * pct(st[:, j]), atol=1e-6), c
assert not (WORK / "_h960_candidate.csv").exists()
print("C full        OK | writer -> run -> 12-target reader file -> strict final = WP/WR blend; stale file removed; atomic replace")
for case in ("missing", "nan", "nowrite"):
    g, rec, out, base_bytes, st, WORK = runC(case, bad_reader=case)
    assert (WORK / "submission.csv").read_bytes() == base_bytes and rec["status"].startswith("E1 INVALID - parent snapshot kept"), (case, rec["status"])
    print(f"C {case:12s} OK | parent snapshot restored byte-identical | {rec['status'][:70]}")
print("ALL H960r END-TO-END CHECKS PASSED")
