"""Our Mirror-student leg for any parent notebook: H957x5's reader cells (7-11) and final cell (12), with the robustness
patches R1-R3 (reader) and R5-R6 (final cell).  Shared by build_h958.py and build_h960.py, so both use one copy.

  R1  reader checkpoints by NAME (other files in the dataset are ignored); known mount layouts before the recursive walk
  R2  reader time guard: timeout = 9 h - elapsed - 20 min; a timeout only drops the student leg
  R3  infer.py: a series stored in fewer than 3 files (e.g. one multi-frame DICOM) is counted by its frames, as in training
  R5  a study the reader could not score gets its expected student rank given its parent rank (linear fit)
  R6  the final cell's header states the weights the code really uses
Each patch is a no-op in a normal run.  The reader cells need NB_T0 (set by the parent's first stage cell) and the final
cell reads WORK / 'submission.csv' written by the parent's last cell.
"""
import json
import re

READERS = "hosen42/rsna-knee-student-readers"   # the 5 LB-validated students (private, hosen42); never re-versioned
READER_CKPTS = ["cnxt_plC2_full.pt", "cnxt_plC3_full.pt", "cnxt_plC_full.pt", "cnxt_plS75_full.pt", "cnxt_plT75s1_full.pt"]
SCORED = {"submission_h957x5": "H957x5", "submission_h957x5w30": "H957x5-w30", "submission_h957x5w40": "H957x5-w40"}
READER_ASSETS = "goodpjw2008/rsna-knee-2-5d-convnext-reader"   # decoder wheels + cnxt_v0_fold0.pt marker the reader cell looks for


def src(c):
    return "".join(c["source"])


def patch(cell, pairs):
    s = src(cell)
    for old, new in pairs:
        assert s.count(old) == 1, (old[:90], s.count(old))
        s = s.replace(old, new)
    cell["source"] = s.splitlines(keepends=True)
    compile("\n".join(l for l in s.splitlines() if not l.startswith(("%%", "!"))), "patched", "exec")


def student_cells(st, extra_readers=None, extra=(), parent_name="the R7 notebook"):
    """st = cells of a scored H957x5-family notebook.  Returns (reader_cells, final_cell, WP, WR), patched."""
    assert len(st) == 13
    reader, final = [json.loads(json.dumps(c)) for c in st[7:12]], json.loads(json.dumps(st[12]))
    assert src(reader[0]).startswith("os.makedirs('/kaggle/working/own_src'")
    assert src(reader[3]).startswith("%%writefile /kaggle/working/own_src/infer.py")
    assert "cnxt_pl*_full.pt" in src(reader[4]) and "assert len(_ck) == 5" in src(reader[4])
    assert src(final).startswith("# [h957] final:")
    _w = re.findall(r"^W_PARENT, W_READER = ([0-9.]+), ([0-9.]+)$", src(final), flags=re.M)
    assert len(_w) == 1, _w
    WP, WR = (float(x) for x in _w[0])
    assert abs(WP + WR - 1) < 1e-9, (WP, WR)

    # R1 + R2: reader checkpoints by name, cheap lookup first; time guard on the reader run
    _lookup = ('# [h958 R1] checkpoints by NAME: other files in a dataset are ignored, so adding files can never break this\n'
               '# run; the two known mount layouts first, the slow recursive walk over /kaggle/input (the DICOM tree) only as a fallback\n'
               'def _ckpts(slug, names):\n'
               '    def _all(d):\n'
               '        return [os.path.join(d, n) for n in names] if all(os.path.isfile(os.path.join(d, n)) for n in names) else None\n'
               '    for d in sorted(glob.glob("/kaggle/input/datasets/*/" + slug)) + ["/kaggle/input/" + slug]:\n'
               '        if _all(d):\n'
               '            return _all(d)\n'
               '    for p in glob.glob("/kaggle/input/**/" + slug + "/" + names[0], recursive=True):\n'
               '        if _all(os.path.dirname(p)):\n'
               '            return _all(os.path.dirname(p))\n'
               '    raise FileNotFoundError("%s: %s" % (slug, names))\n'
               f'_ck = _ckpts("{READERS.split("/")[1]}", {READER_CKPTS!r})\n')
    if extra:
        xslug = extra_readers.split("/")[-1]
        assert extra_readers.count("/") == 1 and xslug != READERS.split("/")[1] and re.fullmatch(r"[a-z0-9-]+", xslug), extra_readers
        assert len(set(extra)) == len(extra) and all(re.fullmatch(r"cnxt_pl\w+_full\.pt", x) for x in extra), extra
        _lookup += f'_ck += _ckpts("{xslug}", {list(extra)!r})   # [h959] the new students, from their own dataset {extra_readers}\n'
    patch(reader[4], [
        ('_ck = sorted(glob.glob("/kaggle/input/**/rsna-knee-student-readers/cnxt_pl*_full.pt", recursive=True))\n'
         'assert len(_ck) == 5, _ck; print("reader ckpts", _ck, flush=True)\n',
         _lookup + 'print("reader ckpts", _ck, flush=True)\n'),
        ('rc = subprocess.run([sys.executable, "/kaggle/working/own_src/infer.py", _data, "test", "/kaggle/working/reader_sub.csv"] + _ck,\n'
         '                    env=_env).returncode\n',
         'try:   # [h958 R2] time guard: the readers never push the session past 9 h; a timeout only drops the student leg\n'
         '    rc = subprocess.run([sys.executable, "/kaggle/working/own_src/infer.py", _data, "test", "/kaggle/working/reader_sub.csv"] + _ck,\n'
         '                        env=_env, timeout=max(60, 9 * 3600 - (time.time() - NB_T0) - 20 * 60)).returncode\n'
         'except subprocess.TimeoutExpired:\n'
         '    rc = -9\n'
         '    print("READER_TIMEOUT after %.0f s - the final cell keeps the notebook\'s own blend" % (time.time() - _t0), flush=True)\n')])

    # R3: infer.py counts a series' slices as training did when the series has fewer than 3 files
    patch(reader[3], [
        ('@torch.no_grad()\ndef run(',
         'def _n_slices(d):\n'
         '    """[h958 R3] slice count as in training (decoded frames, capped at MAX_SLICES).  Counting files is the same except\n'
         '    for a series stored in fewer than 3 files (e.g. one multi-frame DICOM): its frames are read from the headers."""\n'
         '    if not os.path.isdir(d):\n'
         '        return 0\n'
         '    files = os.listdir(d)\n'
         '    if len(files) >= 3:\n'
         '        return min(len(files), MAX_SLICES)\n'
         '    import pydicom\n'
         '    n = 0\n'
         '    for fn in files:\n'
         '        try:\n'
         '            n += int(pydicom.dcmread(os.path.join(d, fn), stop_before_pixels=True).get("NumberOfFrames") or 1)\n'
         '        except Exception:\n'
         '            pass\n'
         '    return min(n, MAX_SLICES)\n\n\n'
         '@torch.no_grad()\ndef run('),
        ('        n_files[sr] = min(len(os.listdir(d)), MAX_SLICES) if os.path.isdir(d) else 0\n',
         '        n_files[sr] = _n_slices(d)\n')])

    # R5 + R6: the final cell
    _head_old = ("# [h957] final: 0.8 x rank(the R7 notebook's own submission.csv) + 0.2 x rank(our Mirror-student ensemble), per\n"
                 "# finding, weights fixed before any run (public 0.955 notebook / our H955C).  A failed reader leaves the R7 file as is;\n"
                 "# a study the reader could not score keeps its R7 rank.\n")
    if (WP, WR) == (0.8, 0.2):
        _head_new = ("# [h957] final: 0.8 x rank(%s's own submission.csv) + 0.2 x rank(our Mirror-student ensemble), per\n"
                     "# finding, weights fixed before any run (public 0.955 notebook / our H955C).  A failed reader leaves the %s file as is;\n"
                     % (parent_name, "R7" if parent_name == "the R7 notebook" else "parent"))
    else:   # R6: say what the code really does
        _head_new = ("# [h957] final: %g x rank(the notebook's own submission.csv) + %g x rank(our Mirror-student ensemble), per\n"
                     "# finding.  The %g student weight was chosen after the 0.957 LB probes (H957x5-w30 / -w40), not before any run.\n"
                     "# A failed reader leaves the notebook's file as is;\n" % (WP, WR, WR))
    _head_new += "# [h958 R5] a study the reader could not score gets its expected student rank given its parent rank.\n"
    patch(final, [
        (_head_old, _head_new),
        ("        _out[_c] = _hn.where(_bad, _p0, W_PARENT * _p0 + W_READER * _hn.nan_to_num(_r)).astype(_hn.float32)\n",
         "        if _bad.any():   # [h958 R5] linear fit of the student rank on the parent rank over the scored studies\n"
         "            _r[_bad] = (_hn.clip(_hn.polyval(_hn.polyfit(_p0[~_bad], _r[~_bad], 1), _p0[_bad]), 0, 1)\n"
         "                        if (~_bad).sum() >= 3 else _p0[_bad])\n"
         "        _out[_c] = (W_PARENT * _p0 + W_READER * _r).astype(_hn.float32)\n")])
    return reader, final, WP, WR


# sha256 of the 5 student checkpoints exactly as uploaded to hosen42/rsna-knee-student-readers v2 (local students_upload/)
CKPT_SHA256 = {
    "cnxt_plC2_full.pt": "07968a382f1e7c724feaa989c31eb4857bb439fb5e9991d36e05e5f0c848bce7",
    "cnxt_plC3_full.pt": "e96f9ad922d0aa3cb7edcfacee34e426bb40ea8ccd5992be3cb4444b57d6cf5d",
    "cnxt_plC_full.pt": "524e3beaaaa3ad9a263f6ba4df4be6806ebfd25e9575d800368adcd8110b20f8",
    "cnxt_plS75_full.pt": "f1866ceddc77f968a7e8b00f2eacb758f0af09882215ad0311dedc7b7693081d",
    "cnxt_plT75s1_full.pt": "1b526073ec72a20a0aea6a18442a79df1f6900f1e17d4bd56651f5cdf823c955",
}

def snapshot_cell(parent_label):
    return ("# [h960r] the parent notebook is complete here (its own submission.csv is written): %s.\n"
            "# Keep an immutable snapshot and its sha256; every later stage is optional and can replace the file only through the\n"
            "# final cell's strict checks.  If the students are not accepted, THIS parent file is the submission.\n"
            "import hashlib as _sh, shutil as _ss\n"
            "_ss.copyfile(WORK / 'submission.csv', WORK / '_baseline_submission.csv')\n"
            "BASELINE_SHA256 = _sh.sha256((WORK / '_baseline_submission.csv').read_bytes()).hexdigest()\n"
            "PARENT_LABEL = %r\n"
            "print('PARENT_SNAPSHOT sha256 %%s | %%s' %% (BASELINE_SHA256, PARENT_LABEL), flush=True)\n" % (parent_label, parent_label))


TRIPLE_QUOTES = ('"' * 3, "'" * 3)


def harden(reader, final, parent_label):
    """Baseline-first hardening of an R1-R3 / R5 student leg (H960r / H960RM), after two independent reviews (10-17):
      * the student stage runs AFTER the parent's last cell and after snapshot_cell(); its source files are written by a
        guarded Python cell (no %%writefile outside the guard), and the run cell is guarded end to end (time left at the
        start and before inference, asset lookup, test-data identity, checkpoint sha256, timeout);
      * the final cell accepts the students only if all 5 verified checkpoints ran and the file holds exactly the parent's
        studies once each, 12 finite targets in [0, 1], none constant; no imputation; otherwise it restores the parent
        snapshot byte for byte; its setup is inside the guard and the receipt write is non-fatal.
    The fallback is the PARENT snapshot (parent_label says what that is), not necessarily the scored 0.958 file.
    Returns (snapshot cell, [writer cell, run cell], final cell)."""
    reader = [json.loads(json.dumps(c)) for c in reader]
    final = json.loads(json.dumps(final))
    # --- 1. the four source cells -> one guarded writer cell (same file contents)
    assert src(reader[0]).startswith("os.makedirs('/kaggle/working/own_src'")
    files = {}
    for c in reader[1:4]:
        first, body = src(c).split("\n", 1)
        assert first.startswith("%%writefile /kaggle/working/own_src/"), first
        files[first.split("/")[-1]] = body
    assert sorted(files) == ["infer.py", "knee.py", "preprocess.py"], sorted(files)
    writer = ("# [h960r] the student sources (H957x5's preprocess.py / knee.py / infer.py, R3 patch included) are written inside a\n"
              "# guard: a failure here only leaves READER_OK False and the parent snapshot stays the submission.\n"
              "import os as _wo\n"
              "READER_OK = False\n"
              "READER_SRC_OK = False\n"
              f"_SOURCES = {files!r}\n"
              "try:\n"
              "    _wo.makedirs('/kaggle/working/own_src', exist_ok=True)\n"
              "    for _n, _s in _SOURCES.items():\n"
              "        with open('/kaggle/working/own_src/' + _n, 'w') as _fh:\n"
              "            _fh.write(_s)\n"
              "    READER_SRC_OK = True\n"
              "except Exception as _we:\n"
              "    print('READER_SOURCES_FAILED', type(_we).__name__, str(_we)[:200], flush=True)\n")
    compile(writer, "writer", "exec")
    # --- 2. the run cell, guarded end to end
    body = src(reader[4])
    assert not any(q in body for q in TRIPLE_QUOTES)
    pairs = [
        ('gc.collect(); torch.cuda.empty_cache()\n',
         'gc.collect(); torch.cuda.empty_cache()\n'
         'if not READER_SRC_OK:\n'
         '    raise RuntimeError("student sources were not written")\n'
         'if 9 * 3600 - (time.time() - NB_T0) < 50 * 60:   # [h960r] too little time left: skip the whole optional stage\n'
         '    raise RuntimeError("only %.0f min left at the start of the student stage" % ((9 * 3600 - (time.time() - NB_T0)) / 60))\n'),
        ('_assets = _find("cnxt_v0_fold0.pt"); _data = _find("test_series.csv")\n',
         '_assets = _find("cnxt_v0_fold0.pt"); _data = _find("test_series.csv")\n'
         '# [h960r] the reader must read exactly the studies the parent notebook submitted\n'
         'import pandas as _rp\n'
         '_test_ids = set(_rp.read_csv(os.path.join(_data, "test.csv"), dtype={"StudyInstanceUID": str})["StudyInstanceUID"])\n'
         '_base_ids = set(_rp.read_csv("/kaggle/working/_baseline_submission.csv", dtype={"StudyInstanceUID": str})["StudyInstanceUID"])\n'
         'if _test_ids != _base_ids or not os.path.isdir(os.path.join(_data, "test_series")):\n'
         '    raise RuntimeError("reader data folder %s does not hold the submitted studies" % _data)\n'
         'READER_INFO["data"] = _data\n'),
        ('print("reader ckpts", _ck, flush=True)\n',
         'print("reader ckpts", _ck, flush=True)\n'
         '# [h960r] the 5 checkpoints must be the exact files the LB-validated students were scored with (sha256)\n'
         'import hashlib as _rh\n'
         f'_SHA = {CKPT_SHA256!r}\n'
         'def _sha256(p):\n'
         '    h = _rh.sha256()\n'
         '    with open(p, "rb") as fh:\n'
         '        for b in iter(lambda: fh.read(1 << 24), b""):\n'
         '            h.update(b)\n'
         '    return h.hexdigest()\n'
         '_bad_ck = [os.path.basename(p) for p in _ck if _sha256(p) != _SHA.get(os.path.basename(p))]\n'
         'if len(_ck) != 5 or _bad_ck:\n'
         '    raise RuntimeError("student checkpoints differ from the validated files: %s" % _bad_ck)\n'
         'READER_INFO["n_ckpt"] = len(_ck); READER_INFO["ckpt_sha_ok"] = True\n'),
        ('subprocess.run([sys.executable, "-m", "pip", "install"', '_pip = subprocess.run([sys.executable, "-m", "pip", "install"'),
        ('"pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=False)\n',
         '"pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=False)\n'
         'READER_INFO["pip_rc"] = _pip.returncode   # non-fatal, as in every scored student run (wheels for this Python may be missing)\n'),
        ('_t0 = time.time()\n',
         'if 9 * 3600 - (time.time() - NB_T0) < 45 * 60:   # [h960r] check again right before inference\n'
         '    raise RuntimeError("only %.0f min left before the 9 h limit" % ((9 * 3600 - (time.time() - NB_T0)) / 60))\n'
         '_t0 = time.time()\n'),
        ('READER_OK = rc == 0 and os.path.isfile("/kaggle/working/reader_sub.csv")\n',
         'READER_OK = rc == 0 and os.path.isfile("/kaggle/working/reader_sub.csv")\n'
         'READER_INFO["reader_rc"] = rc\n')]
    for old, new in pairs:
        assert body.count(old) == 1, (old[:80], body.count(old))
        body = body.replace(old, new)
    guarded = ("# [h960r] the whole optional student stage is guarded: any failure (sources, time left, asset lookup, data identity,\n"
               "# checkpoint sha256, timeout) only leaves READER_OK False, and the final cell then keeps the parent snapshot.\n"
               "import json as _rj\n"
               "READER_OK = False\n"
               "READER_INFO = {}\n"
               "try:\n" + "".join("    " + l if l.strip() else l for l in body.splitlines(keepends=True)) +
               "except Exception as _re:\n"
               "    READER_OK = False\n"
               "    READER_INFO['error'] = '%s: %s' % (type(_re).__name__, str(_re)[:300])\n"
               "    print('READER_STAGE_FAILED', READER_INFO['error'], flush=True)\n"
               "print('READER_INFO', _rj.dumps(READER_INFO, default=str), flush=True)\n")
    compile(guarded, "reader_run", "exec")
    # --- 3. the final cell: strict acceptance, setup inside the guard, non-fatal receipt, no imputation
    patch(final, [
        ("# [h958 R5] a study the reader could not score gets its expected student rank given its parent rank.\n",
         "# [h960r] strict acceptance: the students are used only if all 5 verified checkpoints ran and their file holds exactly the\n"
         "# parent's studies once each, all 12 targets finite in [0, 1] and non-constant; otherwise the PARENT snapshot (see\n"
         "# PARENT_LABEL) is restored byte for byte.  No imputation of missing rows.\n"),
        ("_hw = WORK\n"
         "_hs.copyfile(_hw / 'submission.csv', _hw / '_r7_submission.csv')\n"
         "_par = _hp.read_csv(_hw / '_r7_submission.csv', dtype={'StudyInstanceUID': str})\n"
         "_cols = list(_par.columns)\n"
         "_labs = _cols[1:]\n"
         "_ids = _par['StudyInstanceUID'].tolist()\n"
         "try:\n",
         "_hw = WORK\n"
         "try:\n"
         "    _hs.copyfile(_hw / '_baseline_submission.csv', _hw / '_r7_submission.csv')   # [h960r] the parent = the snapshot\n"
         "    _par = _hp.read_csv(_hw / '_r7_submission.csv', dtype={'StudyInstanceUID': str})\n"
         "    _cols = list(_par.columns)\n"
         "    _labs = _cols[1:]\n"
         "    _ids = _par['StudyInstanceUID'].tolist()\n"),
        ("    _rd = _hp.read_csv(_hw / 'reader_sub.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')\n"
         "    _rd = _rd.reindex(_ids)[_labs]\n"
         "    if _rd.notna().values.mean() < 0.97 or not (_rd.nunique() > 1).any():\n"
         "        raise RuntimeError('reader file unusable (mostly NaN or constant)')\n",
         "    _ri = globals().get('READER_INFO', {})\n"
         "    if _ri.get('n_ckpt') != 5 or not _ri.get('ckpt_sha_ok'):\n"
         "        raise RuntimeError('E1_INVALID: not the 5 verified checkpoints')\n"
         "    _raw_rd = _hp.read_csv(_hw / 'reader_sub.csv', dtype={'StudyInstanceUID': str})\n"
         "    if _raw_rd['StudyInstanceUID'].duplicated().any() or set(_raw_rd['StudyInstanceUID']) != set(_ids):\n"
         "        raise RuntimeError('E1_INVALID: the student rows are not exactly the parent studies')\n"
         "    if not set(_labs) <= set(_raw_rd.columns):\n"
         "        raise RuntimeError('E1_INVALID: target columns missing')\n"
         "    _rd = _raw_rd.set_index('StudyInstanceUID').reindex(_ids)[_labs].astype(float)\n"
         "    if not _hn.isfinite(_rd.values).all() or ((_rd.values < 0) | (_rd.values > 1)).any():\n"
         "        raise RuntimeError('E1_INVALID: non-finite or out-of-range student predictions')\n"
         "    if (_rd.nunique() <= 1).any():\n"
         "        raise RuntimeError('E1_INVALID: a constant target column')\n"),
        ("        if _bad.any():   # [h958 R5] linear fit of the student rank on the parent rank over the scored studies\n"
         "            _r[_bad] = (_hn.clip(_hn.polyval(_hn.polyfit(_p0[~_bad], _r[~_bad], 1), _p0[_bad]), 0, 1)\n"
         "                        if (~_bad).sum() >= 3 else _p0[_bad])\n", ""),
        ("    _hs.copyfile(_hw / '_r7_submission.csv', _hw / 'submission.csv')\n"
         "    _rep = {'status': f'R7 FILE KEPT: {type(_e).__name__}: {str(_e)[:300]}'}\n",
         "    _hs.copyfile(_hw / '_baseline_submission.csv', _hw / 'submission.csv')   # [h960r] the immutable parent snapshot\n"
         "    import hashlib as _fh\n"
         "    assert _fh.sha256((_hw / 'submission.csv').read_bytes()).hexdigest() == BASELINE_SHA256\n"
         "    _rep = {'status': f'E1 INVALID - parent snapshot kept: {type(_e).__name__}: {str(_e)[:300]}'}\n"),
        ("(_hw / 'h957_receipt.json').write_text(_hj.dumps(_rep, indent=1))\nprint('H957_FINAL', _hj.dumps(_rep), flush=True)\n",
         "_rep.update(parent=PARENT_LABEL, reader_info=globals().get('READER_INFO', {}), parent_sha256=BASELINE_SHA256)\n"
         "try:   # [h960r] the receipt is diagnostic only: a failure here must not fail the notebook\n"
         "    (_hw / 'h960_receipt.json').write_text(_hj.dumps(_rep, indent=1, default=str))\n"
         "except Exception as _rw:\n"
         "    print('RECEIPT_WRITE_FAILED', type(_rw).__name__, flush=True)\n"
         "print('H960_FINAL', _hj.dumps(_rep, default=str), flush=True)\n")])
    cell = lambda t: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.splitlines(keepends=True)}
    return cell(snapshot_cell(parent_label)), [cell(writer), cell(guarded)], final
