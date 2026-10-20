"""hengck23's MaViT (MIT, public 0.935) as an optional leg after our final cell (H967): foysal's C3 cell 4 script with marked
[h967] patches, then three guarded cells -- writer (backup of the final file + the script), runner (own process group, >= 70 min
left, killed at 9 h - 12 min) and blend ((1 - W) x rank(final file) + W x rank(MaViT), a study MaViT could not read keeps its
rank, > 3 % unread / wrong studies / out-of-range / constant column / any exception restores the pre-MaViT file).  None of the
cells raises.  Drafted by the 10-19 research workflow (scratchpad mavit_leg/draft), reviewed and moved here.
"""
import hashlib
import json

from paths import inp

C3 = "submission_c3/rsna-knee-sub-c3.ipynb"
C3_SHA = "96064b4e6e7ac856c5faaec18dc072d0bfe3acb954d82e5f974ef00a4a1b7515"      # = foysalemonshanto/rsna-knee-sub-c3 v3 (sub 57307392)
CKPT_SHA = "7acd2a990a89c7a7d73ec0488badc5058a61b0c77142742c532ebc196749bdbc"    # hengck23/hengck23-rnsa-knee v3 00000008.pth
MAVIT_DATASET = "hengck23/hengck23-rnsa-knee"


def mavit_cells(W):
    """Returns (cells to append after our final cell, the patched MaViT script)."""
    raw = open(inp(C3), "rb").read()
    assert hashlib.sha256(raw).hexdigest() == C3_SHA, "C3 notebook changed"
    c4 = "".join(json.loads(raw)["cells"][4]["source"])
    head = "%%writefile /kaggle/working/mavit_test.py\n"
    assert c4.startswith(head)
    s = c4[len(head):]

    def rep(s, old, new):
        assert s.count(old) == 1, old[:80]
        return s.replace(old, new)


    s = rep(s, "# [C3 10-17] hengck23 MaViT (MIT, 0.935 public) on the TEST split: its notebook's model / reader code; own process.  A study it cannot\n"
               "# read stays NaN (it then takes the other members' ranks); more than 3 % NaN stops the notebook in the blend cell.\n",
            "# [h967] hengck23 MaViT (MIT, public 0.935 = his scored version 4) on the TEST split: foysal's C3 cell 4 (C3 v3, sub 57307392), own\n"
            "# process.  [h967] changes: hengck23's folder FIRST on sys.path (and checked), mount search with a shallow glob fallback, the\n"
            "# checkpoint's sha256, output /kaggle/working/mavit_leg/mavit_test.csv written atomically.  A study it cannot read stays NaN.\n")
    s = rep(s, "_HD = [d for d in ('/kaggle/input/datasets/hengck23/hengck23-rnsa-knee', '/kaggle/input/hengck23-rnsa-knee') if os.path.isfile(os.path.join(d, 'model.py'))][0]\n"
               "sys.path.append(_HD)\n",
            "def _h967_dir(cands, need):   # [h967] the two known mount layouts, then a shallow glob (never the DICOM tree)\n"
            "    for d in list(cands) + sorted(glob.glob('/kaggle/input/*/')) + sorted(glob.glob('/kaggle/input/*/*/')) + sorted(glob.glob('/kaggle/input/*/*/*/')):\n"
            "        if all(os.path.exists(os.path.join(d, n)) for n in need):\n"
            "            return d.rstrip('/')\n"
            "    raise FileNotFoundError(need)\n"
            "_HD = _h967_dir(('/kaggle/input/datasets/hengck23/hengck23-rnsa-knee', '/kaggle/input/hengck23-rnsa-knee'),\n"
            "                ('model.py', 'mavit.py', 'my_fastread_knee.py', 'configure.py', '00000008.pth'))\n"
            "sys.path.insert(0, _HD)   # [h967] first, so no other model / configure / mavit module can shadow hengck23's\n")
    s = rep(s, "from my_fastread_knee import *\n",
            "from my_fastread_knee import *\n"
            "import configure as _h967_c, mavit as _h967_v, model as _h967_m, my_fastread_knee as _h967_f   # [h967] origin check\n"
            "for _h967_x in (_h967_c, _h967_v, _h967_m, _h967_f):\n"
            "    assert os.path.dirname(os.path.abspath(_h967_x.__file__)) == os.path.abspath(_HD), ('shadowed', _h967_x.__file__)\n")
    s = rep(s, "_K = [k for k in ('/kaggle/input/competitions/rsna-knee-abnormality-detection', '/kaggle/input/rsna-knee-abnormality-detection') if os.path.isfile(os.path.join(k, 'test.csv'))][0]\n",
            "_K = _h967_dir(('/kaggle/input/competitions/rsna-knee-abnormality-detection', '/kaggle/input/rsna-knee-abnormality-detection'),\n"
            "               ('test.csv', 'test_series.csv', 'test_series'))\n")
    s = rep(s, "_m = KneeModel(pretrained=False); ",
            "import hashlib as _h967_hl   # [h967] the exact checkpoint C3 and hengck23's 0.935 run used\n"
            "_h967_h = _h967_hl.sha256()\n"
            "with open(os.path.join(_HD, '00000008.pth'), 'rb') as _h967_fh:\n"
            "    for _h967_b in iter(lambda: _h967_fh.read(1 << 24), b''):\n"
            "        _h967_h.update(_h967_b)\n"
            "if _h967_h.hexdigest() != '%s':\n"
            "    raise RuntimeError('MaViT checkpoint sha256 differs: %%s' %% _h967_h.hexdigest())\n"
            "_m = KneeModel(pretrained=False); " % CKPT_SHA)
    s = rep(s, "_df.to_csv('/kaggle/working/mavit_test.csv', index=False)\n",
            "_df.to_csv('/kaggle/working/mavit_leg/mavit_test.csv.tmp', index=False)   # [h967] atomic\n"
            "os.replace('/kaggle/working/mavit_leg/mavit_test.csv.tmp', '/kaggle/working/mavit_leg/mavit_test.csv')\n")
    compile(s, "mavit_infer.py", "exec")

    md = lambda t: {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
    code = lambda t: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.splitlines(keepends=True)}

    writer = ("# [h967] backup of the H964 final file, then the MaViT script (see its header).  A failure here only leaves MAVIT_SRC_OK False:\n"
              "# the H964 file stays the submission.  Never raises.\n"
              "import os as _mw_os, shutil as _mw_sh, hashlib as _mw_hl\n"
              "MAVIT_SRC_OK = False\n"
              "MAVIT_PRE_SHA256 = None\n"
              "_MW_SRC = " + repr(s) + "\n"
              "try:\n"
              "    _mw_sh.copyfile('/kaggle/working/submission.csv', '/kaggle/working/_h967_pre_mavit.csv')\n"
              "    MAVIT_PRE_SHA256 = _mw_hl.sha256(open('/kaggle/working/_h967_pre_mavit.csv', 'rb').read()).hexdigest()\n"
              "    _mw_os.makedirs('/kaggle/working/mavit_leg', exist_ok=True)\n"
              "    for _mw_f in ('mavit_test.csv', 'mavit_test.csv.tmp'):   # never accept a stale file\n"
              "        if _mw_os.path.exists('/kaggle/working/mavit_leg/' + _mw_f):\n"
              "            _mw_os.remove('/kaggle/working/mavit_leg/' + _mw_f)\n"
              "    with open('/kaggle/working/mavit_leg/mavit_infer.py', 'w') as _mw_fh:\n"
              "        _mw_fh.write(_MW_SRC)\n"
              "    MAVIT_SRC_OK = True\n"
              "except Exception as _mw_e:\n"
              "    print('MAVIT_WRITE_FAILED', type(_mw_e).__name__, str(_mw_e)[:200], flush=True)\n"
              "print('MAVIT_BACKUP sha256', MAVIT_PRE_SHA256, flush=True)\n")

    runner = ("# [h967] MaViT in its own process group (as C3 cell 5), started only with >= 70 min of the 9 h left and killed at 9 h - 12 min;\n"
              "# a failure, a skip or a timeout only leaves MAVIT_OK False (the H964 file stays the submission).  Never raises.\n"
              "import json as _mr_js, os as _mr_os, signal as _mr_sig, subprocess as _mr_sp, sys as _mr_sys, time as _mr_t\n"
              "MAVIT_OK = False\n"
              "MAVIT_INFO = {}\n"
              "try:\n"
              "    if not MAVIT_SRC_OK:\n"
              "        raise RuntimeError('MaViT script not written')\n"
              "    _mr_left = 9 * 3600 - (_mr_t.time() - NB_T0)\n"
              "    MAVIT_INFO['min_left_at_start'] = round(_mr_left / 60, 1)\n"
              "    if _mr_left < 70 * 60:\n"
              "        raise RuntimeError('only %.0f min left: MaViT skipped' % (_mr_left / 60))\n"
              "    _mr_t0 = _mr_t.time()\n"
              "    _mr_p = _mr_sp.Popen([_mr_sys.executable, '-u', '/kaggle/working/mavit_leg/mavit_infer.py'], start_new_session=True)\n"
              "    try:\n"
              "        _mr_rc = _mr_p.wait(timeout=max(60, int(_mr_left - 12 * 60)))\n"
              "    except _mr_sp.TimeoutExpired:\n"
              "        _mr_os.killpg(_mr_p.pid, _mr_sig.SIGKILL)\n"
              "        _mr_p.wait()\n"
              "        _mr_rc = 124\n"
              "    MAVIT_INFO.update(rc=_mr_rc, seconds=round(_mr_t.time() - _mr_t0, 1))\n"
              "    MAVIT_OK = _mr_rc == 0 and _mr_os.path.isfile('/kaggle/working/mavit_leg/mavit_test.csv')\n"
              "except Exception as _mr_e:\n"
              "    MAVIT_INFO['error'] = '%s: %s' % (type(_mr_e).__name__, str(_mr_e)[:300])\n"
              "print('MAVIT_STAGE', _mr_js.dumps(dict(MAVIT_INFO, ok=MAVIT_OK)), flush=True)\n")

    blend = ("# [h967] final = (1 - W_MAVIT) x rank(the H964 file) + W_MAVIT x rank(MaViT), per finding, W_MAVIT fixed before any run; MaViT is\n"
             "# ranked over the studies it read, and a study it could not read keeps the H964 rank (C3's rule).  Rejected (the H964 file is\n"
             "# restored byte for byte): stage not ok, rows not exactly the submitted studies, > 3 % unread (C3's cap), values outside\n"
             "# [0, 1], a constant column.  Write, re-read, then atomic replace.  Never raises.\n"
             "import hashlib as _mb_hl, json as _mb_js, os as _mb_os, shutil as _mb_sh\n"
             "import numpy as _mb_np, pandas as _mb_pd\n"
             "W_MAVIT = __W__\n"
             "_mb = {'w_mavit': W_MAVIT}\n"
             "_mb_pre = '/kaggle/working/_h967_pre_mavit.csv'\n"
             "_mb_sha = lambda: _mb_hl.sha256(open(_mb_pre, 'rb').read()).hexdigest()\n"
             "try:\n"
             "    if MAVIT_PRE_SHA256 is None or _mb_sha() != MAVIT_PRE_SHA256:\n"
             "        raise RuntimeError('the pre-MaViT backup is missing or changed')\n"
             "    if not MAVIT_OK:\n"
             "        raise RuntimeError('MaViT stage not ok')\n"
             "    _mb_b = _mb_pd.read_csv(_mb_pre, dtype={'StudyInstanceUID': str})\n"
             "    _mb_cols = list(_mb_b.columns)\n"
             "    _mb_labs = _mb_cols[1:]\n"
             "    _mb_ids = _mb_b['StudyInstanceUID'].tolist()\n"
             "    _mb_n = len(_mb_ids)\n"
             "    _mb_m = _mb_pd.read_csv('/kaggle/working/mavit_leg/mavit_test.csv', dtype={'StudyInstanceUID': str})\n"
             "    if _mb_m['StudyInstanceUID'].duplicated().any() or set(_mb_m['StudyInstanceUID']) != set(_mb_ids):\n"
             "        raise RuntimeError('MaViT rows are not exactly the submitted studies')\n"
             "    if not set(_mb_labs) <= set(_mb_m.columns):\n"
             "        raise RuntimeError('MaViT target columns missing')\n"
             "    _mb_v = _mb_m.set_index('StudyInstanceUID').reindex(_mb_ids)[_mb_labs].astype(float).values\n"
             "    _mb_fail = ~_mb_np.isfinite(_mb_v).all(axis=1)\n"
             "    if _mb_fail.sum() > 0.03 * _mb_n:\n"
             "        raise RuntimeError('MaViT did not read %d of %d studies (> 3 %%)' % (int(_mb_fail.sum()), _mb_n))\n"
             "    _mb_ok = _mb_v[~_mb_fail]\n"
             "    if ((_mb_ok < 0) | (_mb_ok > 1)).any():\n"
             "        raise RuntimeError('MaViT probabilities outside [0, 1]')\n"
             "    if _mb_n >= 3 and (_mb_ok.max(axis=0) - _mb_ok.min(axis=0) <= 0).any():\n"
             "        raise RuntimeError('a constant MaViT column')\n"
             "    _mb_out = _mb_b.copy()\n"
             "    for _mb_j, _mb_c in enumerate(_mb_labs):\n"
             "        _mb_rb = _mb_pd.Series(_mb_b[_mb_c].values.astype(_mb_np.float64)).rank(method='average', pct=True).values\n"
             "        _mb_rm = _mb_np.full(_mb_n, _mb_np.nan)\n"
             "        _mb_rm[~_mb_fail] = _mb_pd.Series(_mb_ok[:, _mb_j]).rank(method='average', pct=True).values\n"
             "        _mb_out[_mb_c] = _mb_np.where(_mb_fail, _mb_rb, (1.0 - W_MAVIT) * _mb_rb + W_MAVIT * _mb_rm).astype(_mb_np.float32)\n"
             "    _mb_x = _mb_out[_mb_labs].values\n"
             "    assert _mb_np.isfinite(_mb_x).all() and ((_mb_x >= 0) & (_mb_x <= 1)).all() and _mb_out['StudyInstanceUID'].tolist() == _mb_ids\n"
             "    _mb_out[_mb_cols].to_csv('/kaggle/working/_h967_candidate.csv', index=False)\n"
             "    _mb_chk = _mb_pd.read_csv('/kaggle/working/_h967_candidate.csv', dtype={'StudyInstanceUID': str})\n"
             "    assert list(_mb_chk.columns) == _mb_cols and _mb_chk['StudyInstanceUID'].tolist() == _mb_ids and _mb_np.isfinite(_mb_chk[_mb_labs].values).all()\n"
             "    _mb_os.replace('/kaggle/working/_h967_candidate.csv', '/kaggle/working/submission.csv')\n"
             "    _mb.update(status='blended', rows=_mb_n, mavit_unread_rows=int(_mb_fail.sum()))\n"
             "except Exception as _mb_e:\n"
             "    try:\n"
             "        if MAVIT_PRE_SHA256 is not None and _mb_sha() == MAVIT_PRE_SHA256:\n"
             "            _mb_sh.copyfile(_mb_pre, '/kaggle/working/_h967_restore.csv')\n"
             "            _mb_os.replace('/kaggle/working/_h967_restore.csv', '/kaggle/working/submission.csv')\n"
             "            _mb_kept = 'H964 file restored'\n"
             "        else:\n"
             "            _mb_kept = 'no valid backup - submission.csv left as is'\n"
             "    except Exception as _mb_e2:\n"
             "        _mb_kept = 'restore failed (%s) - submission.csv left as is' % type(_mb_e2).__name__\n"
             "    _mb['status'] = 'MaViT not used - %s: %s: %s' % (_mb_kept, type(_mb_e).__name__, str(_mb_e)[:300])\n"
             "_mb['mavit_info'] = globals().get('MAVIT_INFO', {})\n"
             "try:\n"
             "    open('/kaggle/working/h967_receipt.json', 'w').write(_mb_js.dumps(_mb, indent=1, default=str))\n"
             "except Exception:\n"
             "    pass\n"
             "print('H967_FINAL', _mb_js.dumps(_mb, default=str), flush=True)\n").replace("__W__", repr(W))

    cells = [md("## Stage 5 [h967]: hengck23's MaViT (MIT, public 0.935) after our final cell, optional and guarded: final = %g x rank(the "
                "H964 file) + %g x rank(MaViT), weight fixed before any run; a skipped, failed, late or rejected MaViT stage keeps the H964 "
                "file byte for byte\n" % (round(1 - W, 4), W)), code(writer), code(runner), code(blend)]
    for c in cells[1:]:
        compile("".join(c["source"]), "cell", "exec")
    return cells, s
