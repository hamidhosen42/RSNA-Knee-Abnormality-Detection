import json, sys
from pathlib import Path

NB = Path('/Users/md.hamidhosen/E Drive/1-Kaggle/RSNA Knee Abnormality Detection/rsna-knee-abnormality-detection-v2.ipynb')
nb = json.loads(NB.read_text())
cells = nb['cells']


def src(i):
    return ''.join(cells[i]['source'])


def set_src(i, s):
    cells[i]['source'] = s.splitlines(keepends=True)


def sub(i, old, new):
    s = src(i)
    assert s.count(old) == 1, (i, s.count(old), old[:60])
    set_src(i, s.replace(old, new))


def mkcell(kind, s):
    c = {'cell_type': kind, 'metadata': {}, 'source': s.splitlines(keepends=True)}
    if kind == 'code':
        c['execution_count'] = None
        c['outputs'] = []
    return c


# ------------------------------------------------------------------ 1. CFG (cell 3)
sub(3, """    STAGE = 'eda'  # eda | teacher | preprocess | train | infer""",
       """    # The previous Kaggle run failed with "Submission File Not Found" because the
    # notebook was committed with STAGE='eda'. Only the 'infer' stage writes
    # submission.csv, so a code competition re-run had nothing to score.
    # Anything you submit must be committed with STAGE = 'infer'.
    STAGE = 'infer'  # eda | teacher | preprocess | train | infer""")

sub(3, """import os, re, sys, gc, json, math, time, warnings, hashlib, unicodedata""",
       """import os, re, sys, gc, json, math, time, warnings, hashlib, unicodedata, itertools""")

sub(3, """    extra_ckpt_dirs = []     # e.g. [Path('/kaggle/input/rsna-models-v2')]""",
       """    extra_ckpt_dirs = []     # e.g. [Path('/kaggle/input/rsna-models-v2')]
    # Directories of .whl files for offline DICOM decoders (see the preflight cell).
    wheel_dirs = []          # e.g. [Path('/kaggle/input/pydicom-decoder-wheels')]""")

sub(3, """    comp_dir = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')""",
       """    # Kaggle normally mounts the competition at /kaggle/input/<slug>; the
    # /competitions/ prefix only exists in some environments. Both are probed at the
    # bottom of this cell, so a wrong literal here is no longer fatal.
    comp_dir = Path('/kaggle/input/rsna-knee-abnormality-detection')""")

sub(3, """    seed = 42
    num_workers = 4
    amp = True""",
       """    seed = 42
    num_workers = 4
    amp = True

    # ------------------------------ submission robustness
    # Fit a CPU model on the series descriptors (planes / fluid-sensitivity / fat-sat)
    # that are available for the test set, so the notebook still submits something
    # better than 0.5 before any image checkpoint exists.
    USE_METADATA_FALLBACK = True
    # Horizontal flip is NOT a valid TTA here: preprocessing maps every knee onto a
    # shared lateral -> medial axis, so a flip swaps "Medial OA" with "Lateral OA".
    TTA_HFLIP = False""")

sub(3, """if not CFG.comp_dir.exists():
    candidates = list(Path('/kaggle/input').glob('*knee*')) + \\
                 list(Path('/kaggle/input/competitions').glob('*knee*'))
    if candidates:
        CFG.comp_dir = candidates[0]""",
       """if not (CFG.comp_dir / 'test.csv').exists():
    candidates = [Path('/kaggle/input/rsna-knee-abnormality-detection'),
                  Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')]
    candidates += sorted(Path('/kaggle/input').glob('*knee*'))
    candidates += sorted(Path('/kaggle/input/competitions').glob('*knee*'))
    for _c in candidates:
        if (_c / 'test.csv').exists():
            CFG.comp_dir = _c
            break""")

# ------------------------------------------------- 1b. decoder preflight (cells 4 and 5)
PREFLIGHT_MD = '### Environment and decoder preflight\n\nJPEG Lossless and JPEG 2000 both occur in this data, and the Kaggle base image does not\nalways ship a decoder for them. Rather than guessing from importable module names, this\ncell samples real DICOMs, groups them by transfer syntax, and **attempts an actual pixel\ndecode of each**, because Pillow covers JPEG 2000 in most builds but never JPEG Lossless\nP14.\n\nIt runs before the submission safety net, so it refuses to raise during inference: a\nmissing decoder costs score, but crashing here costs the entire submission. Only the\npreprocess stage, which produces no submission, fails hard.\n'
PREFLIGHT = 'import subprocess\nimport pydicom\n\nprint(\'pydicom:\', pydicom.__version__)\n\n# ---------------------------------------------------------------- 1. offline wheels\n# Internet is disabled at submission time, so a missing decoder has to arrive as an\n# attached Kaggle Dataset of .whl files. Put those directories in CFG.wheel_dirs and\n# they are installed here with --no-index (no network).\nfor _d in [Path(x) for x in getattr(CFG, \'wheel_dirs\', [])]:\n    _whl = sorted(_d.glob(\'*.whl\'))\n    if not _whl:\n        print(f\'[warn] no .whl files under {_d}\')\n        continue\n    _r = subprocess.run([sys.executable, \'-m\', \'pip\', \'install\', \'-q\', \'--no-index\',\n                         f\'--find-links={_d}\', *[str(w) for w in _whl]],\n                        capture_output=True, text=True)\n    print(f\'[wheels] {_d}: {len(_whl)} wheels -> exit {_r.returncode}\')\n    if _r.returncode:\n        print(_r.stderr[-1000:])\n\n# ------------------------------------------------------- 2. which plugins are present\ndecoders = {}\nfor mod in [\'pylibjpeg\', \'openjpeg\', \'libjpeg\', \'gdcm\', \'PIL\']:\n    try:\n        __import__(mod)\n        decoders[mod] = \'ok\'\n    except Exception as e:\n        decoders[mod] = f\'MISSING ({type(e).__name__})\'\nfor k, v in decoders.items():\n    print(f\'  {k:<10}: {v}\')\n\n# --------------------------------------------- 3. ground truth: actually decode a file\n# Module names are a poor proxy. Pillow, for instance, covers JPEG 2000 in most builds\n# but never JPEG Lossless P14 - only a real decode tells you which syntaxes you can read.\nSYNTAX_NAMES = {\n    \'1.2.840.10008.1.2\': \'Implicit VR Little Endian\',\n    \'1.2.840.10008.1.2.1\': \'Explicit VR Little Endian\',\n    \'1.2.840.10008.1.2.1.99\': \'Deflated Explicit VR LE\',\n    \'1.2.840.10008.1.2.2\': \'Explicit VR Big Endian\',\n    \'1.2.840.10008.1.2.4.50\': \'JPEG Baseline\',\n    \'1.2.840.10008.1.2.4.51\': \'JPEG Extended\',\n    \'1.2.840.10008.1.2.4.57\': \'JPEG Lossless P14\',\n    \'1.2.840.10008.1.2.4.70\': \'JPEG Lossless P14 SV1\',\n    \'1.2.840.10008.1.2.4.80\': \'JPEG-LS Lossless\',\n    \'1.2.840.10008.1.2.4.81\': \'JPEG-LS Near-lossless\',\n    \'1.2.840.10008.1.2.4.90\': \'JPEG 2000 Lossless\',\n    \'1.2.840.10008.1.2.4.91\': \'JPEG 2000\',\n}\n\n\ndef probe_decoders(roots, max_series=48, per_series=1):\n    """Sample DICOMs, group them by transfer syntax, and try a real pixel decode."""\n    files = []\n    for root in roots:\n        root = Path(root)\n        if not root.exists():\n            continue\n        studies = list(itertools.islice((p for p in root.iterdir() if p.is_dir()), 400))\n        series = [s for st in studies for s in itertools.islice(\n            (q for q in st.iterdir() if q.is_dir()), 3)]\n        if not series:\n            continue\n        step = max(1, len(series) // max_series)\n        for s in series[::step][:max_series]:\n            files.extend(sorted(s.glob(\'*.dcm\'))[:per_series])\n    by_syntax = {}\n    for f in files:\n        try:\n            ds = pydicom.dcmread(str(f), stop_before_pixels=True)\n            uid = str(ds.file_meta.TransferSyntaxUID)\n        except Exception:\n            continue\n        by_syntax.setdefault(uid, []).append(f)\n    out = {}\n    for uid, fs in by_syntax.items():\n        ok, why = False, \'\'\n        for f in fs[:3]:\n            try:\n                _ = pydicom.dcmread(str(f)).pixel_array\n                ok = True\n                break\n            except Exception as e:\n                why = f\'{type(e).__name__}: {str(e)[:110]}\'\n        out[uid] = {\'name\': SYNTAX_NAMES.get(uid, uid), \'n_sampled\': len(fs),\n                    \'decodes\': ok, \'error\': \'\' if ok else why}\n    return out\n\n\nDECODE_PROBE, compressed_decoder_ok = {}, True\n_roots = [CFG.comp_dir / \'test_series\', CFG.comp_dir / \'train_series\']\nif any(Path(r).exists() for r in _roots):\n    DECODE_PROBE = probe_decoders(_roots)\n    if DECODE_PROBE:\n        _tbl = pd.DataFrame(DECODE_PROBE).T.sort_values(\'decodes\')\n        print(\'\\ntransfer syntaxes actually present, and whether they decode here:\')\n        print(_tbl[[\'name\', \'n_sampled\', \'decodes\', \'error\']].to_string())\n        _broken = [v[\'name\'] for v in DECODE_PROBE.values() if not v[\'decodes\']]\n        compressed_decoder_ok = not _broken\n    else:\n        print(\'\\n[warn] found no readable DICOM to probe; decoder status unknown\')\nelse:\n    print(\'\\n[warn] no train_series/test_series mount; skipping the decode probe\')\n\n# --------------------------------------------------------------- 4. act on the result\n# This cell runs BEFORE the submission safety net, so raising here means Kaggle gets no\n# submission.csv at all. Only the preprocess stage - which produces no submission - is\n# allowed to fail hard. Inference warns and lets the per-study fallback absorb the loss.\nif not compressed_decoder_ok:\n    _msg = (f\'Cannot decode: {_broken}. Install python-gdcm and/or pylibjpeg + \'\n            \'pylibjpeg-libjpeg + pylibjpeg-openjpeg. With internet on, pip install \'\n            \'them directly; for an offline submission, pip download the wheels, \'\n            \'publish them as a Kaggle Dataset, attach it, and set CFG.wheel_dirs.\')\n    if CFG.RUN_PREPROCESS:\n        raise RuntimeError(_msg)\n    if CFG.RUN_INFER:\n        print(f\'\\n[SEVERE] {_msg}\\n[SEVERE] Studies using those syntaxes will fall back \'\n              \'to the metadata prediction, which will cost real score. Fix this before \'\n              \'your final submission.\')\n\ntry:\n    import torch\n    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else \'cpu only\'\n    print(f\'torch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | {gpu}\')\nexcept ImportError:\n    print(\'torch unavailable (acceptable only for EDA/teacher/preprocess)\')\n'
assert cells[5]['cell_type'] == 'code' and 'compressed_decoder_ok' in src(5)
set_src(4, PREFLIGHT_MD)
set_src(5, PREFLIGHT)

# ------------------------------------------------------------------ 2. soften fatal asserts (cell 6)
sub(6, """assert list(sample_sub.columns) == ['StudyInstanceUID'] + TARGETS
assert 'Report' not in test.columns, 'The strategy assumes reports are unavailable at inference.'
assert train.StudyInstanceUID.is_unique and test.StudyInstanceUID.is_unique""",
       """# These were hard asserts. On the hidden re-run an assert means no submission.csv at
# all, so anything that does not actually break inference is now only a warning.
if list(sample_sub.columns) != ['StudyInstanceUID'] + TARGETS:
    print('[warn] sample_submission column order differs from the documented one:',
          list(sample_sub.columns))
if 'Report' in test.columns:
    print('[warn] test.csv contains Report; the image strategy still ignores it.')
assert train.StudyInstanceUID.is_unique and test.StudyInstanceUID.is_unique""")

# ------------------------------------------------------------------ 3. safety-net cell after cell 6
SAFETY = '''# ============================== SUBMISSION SAFETY NET ==============================
# Why the last run failed: a code competition re-runs the whole notebook and then looks
# for /kaggle/working/submission.csv. If the writing cell is skipped (wrong STAGE) or
# any earlier cell raises, the file does not exist and the submission is rejected no
# matter how good the model is.
#
# Rule from here on: a VALID submission.csv exists on disk from this cell onwards, in
# every stage. Later cells only ever overwrite it with something better.
#
# This cell deliberately re-reads its own inputs instead of trusting the loader cell
# above. A safety net that needs an earlier cell to have succeeded is not a safety net,
# and it must also work when you execute this cell on its own.

if 'CFG' not in globals() or 'TARGETS' not in globals():
    raise RuntimeError('Run the configuration cell (Part 0) first, or use Run All.')


def _resolve_comp_dir():
    """Find the competition mount even if the loader cell never ran."""
    seen, roots = set(), []
    for c in [getattr(CFG, 'comp_dir', None),
              Path('/kaggle/input/rsna-knee-abnormality-detection'),
              Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
              *sorted(Path('/kaggle/input').glob('*knee*')),
              *sorted(Path('/kaggle/input/competitions').glob('*knee*'))]:
        if c is not None and str(c) not in seen:
            seen.add(str(c))
            roots.append(Path(c))
    for r in roots:
        if (r / 'test.csv').exists():
            return r
    raise FileNotFoundError(
        'test.csv not found under any candidate mount. Attach the competition data '
        f'(Add Input -> Competitions), then re-run. Searched: {[str(r) for r in roots]}')


_root = _resolve_comp_dir()
if _root != getattr(CFG, 'comp_dir', None):
    print(f'[info] comp_dir corrected to {_root}')
    CFG.comp_dir = _root

if not isinstance(globals().get('test'), pd.DataFrame):
    test = pd.read_csv(_root / 'test.csv')
if not isinstance(globals().get('sample_sub'), pd.DataFrame):
    _p = _root / 'sample_submission.csv'
    sample_sub = pd.read_csv(_p) if _p.exists() else None

SUB_PATH = (Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('.')) / 'submission.csv'
SUB_COLS = ['StudyInstanceUID'] + TARGETS

# The row set the grader expects. sample_submission is authoritative when it agrees
# with test.csv; otherwise test.csv wins, because that is what defines the studies.
if (isinstance(sample_sub, pd.DataFrame)
        and 'StudyInstanceUID' in sample_sub.columns
        and len(sample_sub) == len(test)
        and set(sample_sub.StudyInstanceUID) == set(test.StudyInstanceUID)):
    SUB_INDEX = sample_sub[['StudyInstanceUID']].copy()
else:
    if sample_sub is not None:
        print('[warn] sample_submission and test.csv disagree; using test.csv as the row set')
    SUB_INDEX = test[['StudyInstanceUID']].copy()


def write_submission(pred_df, note=''):
    """Align any prediction frame to the official rows/columns, then write the file.

    Missing or non-finite cells fall back to 0.5 rather than propagating NaN, so a
    partially successful model still produces a scorable file.
    """
    out = SUB_INDEX.merge(pred_df, on='StudyInstanceUID', how='left')
    for c in TARGETS:
        if c not in out.columns:
            out[c] = 0.5
    out[TARGETS] = (out[TARGETS].apply(pd.to_numeric, errors='coerce')
                                .replace([np.inf, -np.inf], np.nan)
                                .fillna(0.5).clip(0.0, 1.0))
    out = out[SUB_COLS]
    assert len(out) == len(SUB_INDEX), (len(out), len(SUB_INDEX))
    assert not out[TARGETS].isna().any().any()
    out.to_csv(SUB_PATH, index=False)
    print(f'submission.csv written {out.shape} -> {SUB_PATH} {note}')
    return out


_baseline = SUB_INDEX.copy()
for _c in TARGETS:
    _baseline[_c] = 0.5
submission = write_submission(_baseline, '(constant 0.5 safety net)')
'''
cells.insert(7, mkcell('code', SAFETY))

# ------------------------------------------------------------------ 4. metadata fallback before the final infer cell
# after the insert above, the "Part 5" markdown is at 47 and the infer code cell at 48
assert cells[47]['cell_type'] == 'markdown' and 'Part 5' in src(47), src(47)[:80]
assert cells[48]['cell_type'] == 'code' and 'RUN_INFER' in src(48)

META_MD = '''## 5.0 Protocol-metadata fallback — a scorable submission before any GPU training

`test_series.csv` ships with the test set, so plane / fluid-sensitivity / fat-suppression
counts are legitimately available at inference time. Which sequences a radiographer
acquired is not independent of the clinical question, so these descriptors carry a small
but real signal, and the model costs a couple of CPU minutes with no weights to attach.

Two honest caveats: the folds below are stratified, **not** grouped by patient/report, so
the printed OOF AUC is optimistic; and a target whose cross-validated AUC does not clear
0.53 is left at 0.5 rather than shipped. Treat this as the floor the image model has to
beat, not as a solution.
'''

META_CODE = '''if CFG.RUN_INFER and CFG.USE_METADATA_FALLBACK:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    PLANES = ['Sagittal', 'Coronal', 'Axial']

    def protocol_features(series_df):
        g = series_df.copy()
        g['Fluid_Sensitive'] = pd.to_numeric(g.Fluid_Sensitive, errors='coerce').fillna(0).astype(int)
        g['Fat_Suppression'] = pd.to_numeric(g.Fat_Suppression, errors='coerce').fillna(0).astype(int)
        g['Anatomical_Plane'] = g.Anatomical_Plane.astype(str).str.strip().str.title()
        recs = {}
        for study, s in g.groupby('StudyInstanceUID', sort=False):
            f = {'n_series': len(s),
                 'n_fluid': int((s.Fluid_Sensitive == 1).sum()),
                 'n_fatsup': int((s.Fat_Suppression == 1).sum()),
                 'frac_fluid': float((s.Fluid_Sensitive == 1).mean()),
                 'frac_fatsup': float((s.Fat_Suppression == 1).mean())}
            for p in PLANES:
                sp = s[s.Anatomical_Plane == p]
                f[f'n_{p}'] = len(sp)
                f[f'n_{p}_fluid'] = int((sp.Fluid_Sensitive == 1).sum())
                f[f'n_{p}_fat'] = int((sp.Fat_Suppression == 1).sum())
                f[f'n_{p}_fluid_fat'] = int(((sp.Fluid_Sensitive == 1) &
                                             (sp.Fat_Suppression == 1)).sum())
            recs[study] = f
        out = pd.DataFrame.from_dict(recs, orient='index')
        out['n_other_plane'] = out.n_series - out[[f'n_{p}' for p in PLANES]].sum(axis=1)
        return out.astype(np.float32)

    try:
        Xtr_meta = protocol_features(train_series).reindex(train.StudyInstanceUID).fillna(0.0)
        Xte_meta = protocol_features(test_series).reindex(SUB_INDEX.StudyInstanceUID).fillna(0.0)
        Xte_meta = Xte_meta[Xtr_meta.columns]
        print('protocol features:', Xtr_meta.shape[1], '| test rows:', len(Xte_meta))

        meta_pred = pd.DataFrame(0.5, index=pd.Index(SUB_INDEX.StudyInstanceUID),
                                 columns=TARGETS, dtype=np.float32)
        rows = []
        for c in TARGETS:
            m = train[c].notna().values
            y = train.loc[m, c].astype(int).values
            Xc = Xtr_meta.values[m]
            if len(np.unique(y)) < 2 or int(np.bincount(y, minlength=2).min()) < 25:
                rows.append({'target': c, 'n': int(m.sum()), 'n_pos': int(y.sum()),
                             'oof_auc': np.nan, 'used': False})
                continue
            oof = np.zeros(len(y), np.float32)
            te = np.zeros(len(Xte_meta), np.float32)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.seed)
            for tr_i, va_i in skf.split(Xc, y):
                clf = HistGradientBoostingClassifier(
                    max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
                    min_samples_leaf=40, l2_regularization=1.0,
                    early_stopping=True, validation_fraction=0.15,
                    random_state=CFG.seed)
                clf.fit(Xc[tr_i], y[tr_i])
                oof[va_i] = clf.predict_proba(Xc[va_i])[:, 1]
                te += clf.predict_proba(Xte_meta.values)[:, 1] / skf.get_n_splits()
            auc = float(roc_auc_score(y, oof))
            used = auc > 0.53          # do not ship noise as if it were signal
            if used:
                meta_pred[c] = te
            rows.append({'target': c, 'n': int(m.sum()), 'n_pos': int(y.sum()),
                         'oof_auc': auc, 'used': used})

        meta_oof = pd.DataFrame(rows).sort_values('oof_auc', ascending=False)
        # Write first, pretty-print second: a rendering problem must never cost the file.
        submission = write_submission(meta_pred.reset_index(names='StudyInstanceUID'),
                                      '(protocol-metadata fallback)')
        print(f'metadata-only macro AUC (stratified OOF, optimistic): '
              f'{meta_oof.oof_auc.mean(skipna=True):.4f} over '
              f'{int(meta_oof.used.sum())}/{len(TARGETS)} usable targets')
        try:
            display(meta_oof.style.format({'oof_auc': '{:.3f}'}))
        except Exception:
            print(meta_oof.round(3).to_string(index=False))
    except Exception as e:
        print(f'[error] metadata fallback failed ({type(e).__name__}: {e}); '
              'the 0.5 safety-net submission is kept.')
'''

cells.insert(47, mkcell('markdown', META_MD))
cells.insert(48, mkcell('code', META_CODE))

# ------------------------------------------------------------------ 5. rewrite the image-inference cell
idx = 50
assert cells[idx]['cell_type'] == 'code' and 'TestDataset' in src(idx), src(idx)[:120]

INFER = '''# ===================================================================================
# Image-model inference. This cell can only IMPROVE the submission that already exists
# on disk: every failure path leaves the previous file in place instead of raising and
# leaving Kaggle with nothing to score.
# ===================================================================================
if CFG.RUN_INFER:
    import traceback

    class TestDataset(Dataset):
        """Reports ok=0 instead of raising, so one undecodable study cannot kill the
        other ~1300 and take the whole submission with it."""

        def __init__(self, manifest_df, root):
            self.df = manifest_df.reset_index(drop=True)
            self.root = Path(root)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, i):
            r = self.df.iloc[i]
            vol = np.zeros((CFG.N_VIEWS, CFG.N_SLICES, CFG.IMG_SIZE, CFG.IMG_SIZE), np.uint8)
            vm = np.zeros(CFG.N_VIEWS, np.float32)
            sm = np.zeros((CFG.N_VIEWS, CFG.N_SLICES), np.float32)
            for vi, slot in enumerate(CFG.VIEWS):
                sid = r.get(slot)
                if not isinstance(sid, str) or not sid:
                    continue
                try:
                    v, s = load_series_volume(self.root / str(r.StudyInstanceUID) / sid,
                                              VIEW_SPEC[slot]['plane'])
                except Exception:
                    v, s = None, None
                if v is not None:
                    vol[vi], sm[vi], vm[vi] = v, s, 1
            ok = int(bool(vm.any()))
            if not ok:
                # Keep the tensor well formed so masked attention cannot produce NaN;
                # the row is excluded from the ensemble by row_ok below.
                vm[0], sm[0, 0] = 1.0, 1.0
            return {'x': torch.from_numpy(vol.astype(np.float32) / 255.0),
                    'view_mask': torch.from_numpy(vm),
                    'slice_mask': torch.from_numpy(sm),
                    'idx': i, 'ok': ok}

    def _rank01(a):
        """Per-target percentile rank. AUC is rank-based, so this removes calibration
        differences between folds before averaging."""
        return pd.DataFrame(np.asarray(a, np.float64)).rank(method='average',
                                                            pct=True).to_numpy(np.float64)

    try:
        test_manifest = test.merge(
            build_manifest(test_series, CFG.comp_dir / 'test_series', count_files=True),
            on='StudyInstanceUID', how='left', validate='one_to_one')
        n_noview = int(test_manifest[CFG.VIEWS].isna().all(axis=1).sum())
        if n_noview:
            print(f'[warn] {n_noview} test studies have no selectable series; '
                  'they keep the fallback prediction')

        ckpt_paths = []
        for d in [CFG.ckpt_dir] + list(CFG.extra_ckpt_dirs):
            if Path(d).exists():
                ckpt_paths.extend(Path(d).rglob('knee_cv2_f*.pt'))
        ckpt_paths = sorted(set(ckpt_paths))
        if not ckpt_paths:
            raise FileNotFoundError(
                'no knee_cv2_f*.pt checkpoints found. Run STAGE="train", publish '
                '/kaggle/working/rsna_models_v2 as a Kaggle Dataset, attach it, and list '
                'it in CFG.extra_ckpt_dirs.')
        print('checkpoints:', [str(x) for x in ckpt_paths])

        models, seen_folds = [], set()
        for path in ckpt_paths:
            try:
                ck = torch.load(path, map_location='cpu', weights_only=False)
            except TypeError:  # torch < 2.0
                ck = torch.load(path, map_location='cpu')
            expected = {'views': CFG.VIEWS, 'targets': TARGETS, 'img_size': CFG.IMG_SIZE,
                        'n_slices': CFG.N_SLICES, 'cache_version': CFG.CACHE_VERSION}
            mismatch = {k: (ck.get(k), v) for k, v in expected.items() if ck.get(k) != v}
            if mismatch:
                print(f'[warn] skipping {path}: preprocessing mismatch {mismatch}')
                continue
            if ck.get('fold') in seen_folds:
                print(f'[warn] skipping duplicate fold-{ck.get("fold")} checkpoint: {path}')
                continue
            seen_folds.add(ck.get('fold'))
            model = KneeNet(backbone=ck['backbone'], pretrained=False).to(DEVICE)
            model.load_state_dict(ck['state_dict'], strict=True)
            model.eval()
            models.append(model)
        if not models:
            raise RuntimeError('every checkpoint was rejected; nothing to run')
        if len(models) < CFG.folds:
            print(f'[warning] {len(models)}/{CFG.folds} folds loaded; train all folds '
                  'for the final entry')
        print(f'loaded {len(models)} models | GPU weights: '
              f'{torch.cuda.memory_allocated() / 2 ** 30:.2f} GiB')

        ds = TestDataset(test_manifest, CFG.comp_dir / 'test_series')
        dl = DataLoader(ds, batch_size=CFG.infer_batch_size, shuffle=False,
                        num_workers=CFG.num_workers, pin_memory=True,
                        persistent_workers=CFG.num_workers > 0)
        fold_pred = np.full((len(models), len(ds), len(TARGETS)), 0.5, np.float32)
        row_ok = np.zeros(len(ds), bool)
        t0 = time.time()
        with torch.inference_mode():
            for step, b in enumerate(dl):
                x = b['x'].to(DEVICE, non_blocking=True)
                vm = b['view_mask'].to(DEVICE, non_blocking=True)
                sm = b['slice_mask'].to(DEVICE, non_blocking=True)
                idx = b['idx'].numpy()
                for mi, model in enumerate(models):
                    with autocast():
                        logits, _, _, _ = model(x, vm, sm)
                    p = logits.float().sigmoid().cpu().numpy()
                    if CFG.TTA_HFLIP:
                        with autocast():
                            lf, _, _, _ = model(torch.flip(x, dims=[-1]), vm, sm)
                        p = 0.5 * (p + lf.float().sigmoid().cpu().numpy())
                    fold_pred[mi, idx] = p
                row_ok[idx] = b['ok'].numpy().astype(bool)
                if (step + 1) % 25 == 0:
                    el = time.time() - t0
                    done = min((step + 1) * CFG.infer_batch_size, len(ds))
                    print(f'{done}/{len(ds)} | {el / 60:.1f} min | '
                          f'ETA {el / done * (len(ds) - done) / 60:.1f} min', flush=True)

        if not row_ok.any():
            raise RuntimeError('no test study produced a decodable view')

        ranked = np.stack([_rank01(np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0))
                           for p in fold_pred]).mean(0)
        img_df = pd.DataFrame(ranked, columns=TARGETS)
        img_df.insert(0, 'StudyInstanceUID', ds.df.StudyInstanceUID.values)
        img_df = img_df[row_ok].set_index('StudyInstanceUID')

        # Start from the best fallback available, put it on the same rank scale, then
        # overwrite the studies the image model could actually read.
        if 'meta_pred' in globals():
            final = meta_pred.copy()
        else:
            final = pd.DataFrame(0.5, index=pd.Index(SUB_INDEX.StudyInstanceUID),
                                 columns=TARGETS)
        # float64 throughout: assigning float64 into a float32 column raises in
        # pandas 3 and warns in pandas 2, which would abort the run at the last step.
        final = final[TARGETS].astype(np.float64)
        final[TARGETS] = _rank01(final[TARGETS].to_numpy(np.float64))
        common = final.index.intersection(img_df.index)
        final.loc[common, TARGETS] = img_df.loc[common, TARGETS].to_numpy(np.float64)
        final = final.reset_index(names='StudyInstanceUID')

        const = [c for c in TARGETS if final[c].nunique() < 2]
        if const:
            print(f'[warn] constant predictions for {const} — check the checkpoints')
        print(f'image rows used: {int(row_ok.sum())}/{len(ds)} | '
              f'inference {(time.time() - t0) / 60:.1f} min')
        submission = write_submission(final, '(image ensemble)')
    except Exception:
        traceback.print_exc()
        print('[error] image inference did not complete; the existing submission.csv '
              '(metadata or 0.5 baseline) is kept, so the run still scores.')

# ---------------------------------------------------------------- final verification
_check = pd.read_csv(SUB_PATH)
assert list(_check.columns) == SUB_COLS, list(_check.columns)
assert len(_check) == len(SUB_INDEX) and not _check[TARGETS].isna().any().any()
print(f'FINAL submission.csv: {_check.shape} at {SUB_PATH}')
display(_check.head())
'''
cells[idx] = mkcell('code', INFER)

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
print('patched. cells:', len(cells))
