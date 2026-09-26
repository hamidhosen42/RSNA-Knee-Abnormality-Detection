#### CELL 2
"""Read-only execution instrumentation; does not change source predictions."""
import hashlib as _rep_hashlib
import json as _rep_json
import os as _rep_os
import re as _rep_re
import sys as _rep_sys
import time as _rep_time
from pathlib import Path as _RepPath

_rep_started = _rep_time.time()
_rep_source_sha256 = "6b03919bbca5fe61e37c5ea6546609a771cd7db6503121f3e2dacdd8c9b6b052"
_rep_messages = []
_rep_stages = {}

class _ReproductionOutput:
    def __init__(self, underlying):
        self.underlying = underlying
    def write(self, value):
        _rep_messages.append(value)
        return self.underlying.write(value)
    def flush(self):
        return self.underlying.flush()
    def __getattr__(self, name):
        return getattr(self.underlying, name)

_rep_sys.stdout = _ReproductionOutput(_rep_sys.stdout)

def _rep_sha(path):
    digest = _rep_hashlib.sha256()
    with _RepPath(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _rep_snapshot(stage):
    import pandas as _pd
    import numpy as _np
    work = _RepPath('/kaggle/working')
    frame = _pd.read_csv(work / 'submission.csv', dtype={'StudyInstanceUID': str})
    candidates = [
        _RepPath('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
        _RepPath('/kaggle/input/rsna-knee-abnormality-detection'),
    ]
    competition = next(p for p in candidates if (p / 'test.csv').is_file())
    test = _pd.read_csv(competition / 'test.csv', dtype={'StudyInstanceUID': str})
    schema = list(_pd.read_csv(competition / 'sample_submission.csv', nrows=0).columns)
    assert list(frame.columns) == schema, (stage, 'columns')
    assert frame.StudyInstanceUID.tolist() == test.StudyInstanceUID.tolist(), (stage, 'UID/order')
    assert frame.StudyInstanceUID.is_unique, (stage, 'duplicate UID')
    values = frame.iloc[:, 1:].to_numpy(_np.float64)
    assert _np.isfinite(values).all(), (stage, 'nonfinite')
    assert ((values >= 0) & (values <= 1)).all(), (stage, 'range')
    destination = work / ('reproduction_' + stage + '.csv')
    destination.write_bytes((work / 'submission.csv').read_bytes())
    _rep_stages[stage] = {'rows': len(frame), 'sha256': _rep_sha(destination)}
    print('[reproduction]', stage, _rep_stages[stage], flush=True)

# Importing torch here creates no random tensors; the source controls its seeds.
for _rep_variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    _rep_os.environ.setdefault(_rep_variable, '4')
import torch as _rep_torch
assert _rep_torch.cuda.device_count() == 2, 'exactly two GPUs required'
_rep_gpu_names = [_rep_torch.cuda.get_device_name(i) for i in range(2)]
assert all('T4' in name for name in _rep_gpu_names), _rep_gpu_names
print('[reproduction] T4 preflight', _rep_gpu_names, flush=True)

#### CELL 4
from __future__ import annotations
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')
import gc
import hashlib
import json
import re
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F

def _cuda_execution_probe(index):
    dev = torch.device(f'cuda:{index}')
    try:
        major, minor = torch.cuda.get_device_capability(index)
        probe = nn.Conv2d(3, 4, kernel_size=3, padding=1).eval().to(dev)
        with torch.inference_mode():
            out = probe(torch.zeros((1, 3, 16, 16), device=dev))
            if tuple(out.shape) != (1, 4, 16, 16):
                raise RuntimeError(f'unexpected CUDA probe shape {tuple(out.shape)}')
        torch.cuda.synchronize(index)
        print(f'cuda:{index} probe PASS (compute {major}.{minor})')
        del probe, out
        torch.cuda.empty_cache()
        return True
    except Exception as exc:
        print(f'cuda:{index} probe FAIL ({type(exc).__name__}: {exc}); using CPU fallback')
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return False
DEVS = []
if torch.cuda.is_available():
    DEVS = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count()) if _cuda_execution_probe(i)]
if not DEVS:
    raise RuntimeError('GPU required')
print(f'devices: {[str(d) for d in DEVS]}')
T0 = time.time()
SEED = 2026
np.random.seed(SEED)
torch.manual_seed(SEED)
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
CROP_MM = 130.0
CACHE_IMG = 336
GROUP = 3
N_GROUP_MAX = 1
CACHE_FRACTION = 0.45
CACHE_BUDGET_MAX_GB = 24.0
CACHE_BUDGET_GB = 12.0
TEST_SHARE = 0.3
HDR_THREADS = 16
PIX_THREADS = 12
ORDER_THREADS = 32
ORDER_BUDGET_S = 5400
RUNS = [{'name': 'r224', 'img': 224}, {'name': 'r336', 'img': 336}]
EPOCHS = 10
BATCH_STUDIES = 8
AUG_ROT_DEG = 8.0
AUG_SCALE = 0.08
AUG_SHIFT = 0.05
AUG_INTENSITY = 0.1
LAT_MIN_OFFSET_MM = 20.0
SLICE_BAND = (0.2, 0.8)
RULES_NATIVE = {'order': 'normal', 'lat': 'centre', 'slot_fallback': False, 'decode_fill': 'nearest'}
RULES_LEGACY = {'order': 'dominant_axis', 'lat': 'corner_x', 'slot_fallback': True, 'decode_fill': 'zero'}
RULES = dict(RULES_NATIVE)
LEGACY_LAT_OFFSET_MM = 5.0
LR_HEAD = 0.001
LR_BACKBONE = 8e-06
UNFREEZE_LAST = 6
WEIGHT_DECAY = 0.02
EVAL_BATCH = 8
TIME_BUDGET = 8.0 * 3600
SLOTS_RECOVERED = [('SAG_FLUID_FS', 'Sagittal', True, True), ('COR_FLUID_FS', 'Coronal', True, True), ('AX_FLUID_FS', 'Axial', True, True), ('SAG_FLUID_NOFS', 'Sagittal', True, False), ('COR_T1', 'Coronal', False, False), ('SAG_T1', 'Sagittal', False, False)]
SLOTS_PUBLIC = [('SAG_FLUID', 'Sagittal', None, True), ('COR_FLUID', 'Coronal', None, True), ('AX_FLUID', 'Axial', None, True), ('SAG_STRUCT', 'Sagittal', None, False), ('COR_STRUCT', 'Coronal', None, False), ('AX_STRUCT', 'Axial', None, False)]
SLOT_SCHEME = os.environ.get('SLOT_SCHEME', 'recovered')
SLOTS = SLOTS_PUBLIC if SLOT_SCHEME == 'public' else SLOTS_RECOVERED
N_SLOT = len(SLOTS)
POOL_PARTS = {'cls_mean': 2, 'cls_mean_focal': 3}
SLOT_PRIOR_TABLE = {'ACL': (0, 3, 5), 'MCL': (1, 4), 'Medial Meniscus': (0, 1, 3, 4), 'Lateral Meniscus': (0, 1, 3, 4), 'Medial OA': (1, 4, 5), 'Lateral OA': (1, 4, 5), 'PF OA': (0, 2, 5), 'Effusion': (0, 2), 'Synovitis': (0, 2), "Baker's": (0,), 'Contusion': (0, 1, 2), 'Fracture': (0, 1, 2, 4, 5)}
SLOT_PRIOR_STRENGTH = 0.55
FATSAT_OPTS = {'FS', 'FATSAT', 'FAT_SAT', 'FSAT'}
_SEP = re.compile('[_\\-.]')
_FATSAT_RX = re.compile('\\bfs\\b|fatsat|fat sat|\\bstir\\b|\\bspair\\b|\\bspir\\b|\\bwe\\b|water excit|\\btirm\\b|\\bsting\\b|\\bfatsup\\b')
_T1_RX = re.compile('\\bt1\\b|\\bt1w\\b')
_T2_RX = re.compile('\\bt2\\b|\\bt2w\\b')
_PD_RX = re.compile('\\bpd\\b|\\bpdw\\b|proton|\\bdp\\b|dens')
#### CELL 5
def log(msg):
    print(f'[{time.time() - T0:7.1f}s] {msg}', flush=True)

def find_root():
    for c in [Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'), Path('/kaggle/input/rsna-knee-abnormality-detection'), Path('data'), Path('.')]:
        if (c / 'test.csv').is_file() and (c / 'test_series').is_dir():
            return c
    base = Path('/kaggle/input')
    if base.is_dir():
        for depth1 in sorted((p for p in base.iterdir() if p.is_dir())):
            for cand in [depth1] + sorted((p for p in depth1.iterdir() if p.is_dir())):
                if (cand / 'test.csv').is_file():
                    return cand
    raise FileNotFoundError(f'competition mount not found (cwd {Path.cwd()}); expected a directory holding test.csv and test_series/')

def find_dinov2(variant='small'):
    base = Path('/kaggle/input')
    if not base.is_dir():
        return None
    hits = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ('train_series', 'test_series')]
        if 'config.json' in files and 'dinov2' in root.lower():
            hits.append(Path(root))
    for h in hits:
        if variant in str(h).lower():
            return h
    return hits[0] if hits else None
ROOT = find_root()
log(f'input root: {ROOT}')
IMG = CACHE_IMG

def available_gb():
    try:
        with open('/proc/meminfo') as fh:
            info = {k.strip(): v for k, v in (l.split(':', 1) for l in fh if ':' in l)}
        return int(info['MemAvailable'].split()[0]) / 1024 ** 2
    except Exception:
        return CACHE_BUDGET_GB / CACHE_FRACTION

def plan_cache(n_study, n_test=0):
    avail = available_gb()
    budget = min(avail * CACHE_FRACTION, CACHE_BUDGET_MAX_GB)
    n_total = n_study + max(n_test, int(TEST_SHARE * n_study))
    per_slice = n_total * N_SLOT * IMG * IMG
    afford = int(budget * 1024 ** 3 // max(per_slice, 1))
    groups = max(1, min(N_GROUP_MAX, afford // GROUP))
    log(f'memory: {avail:.1f} GB available, {budget:.1f} GB to the cache; sizing for {n_study} train + {n_total - n_study} test studies -> {groups} group(s) of {GROUP} = {groups * GROUP} slices per slot' + (f' (wanted {N_GROUP_MAX})' if groups < N_GROUP_MAX else ''))
    return groups
N_GROUP = plan_cache(len(pd.read_csv(ROOT / 'train.csv')), len(pd.read_csv(ROOT / 'test.csv')))
CACHE_SLICES = GROUP * N_GROUP
log(f'cache layout: {N_GROUP} groups x {GROUP} slices = {CACHE_SLICES} per slot')
#### CELL 6
HDR_TAGS = ['SeriesDescription', 'SequenceName', 'ScanOptions', 'ScanningSequence', 'RepetitionTime', 'EchoTime', 'Laterality', 'PixelSpacing', 'Rows', 'Columns', 'RescaleSlope', 'RescaleIntercept', 'ImagePositionPatient', 'ImageOrientationPatient']

def _hdr_vec(s, n):
    if not isinstance(s, str):
        return None
    try:
        v = [float(x) for x in s.split('|')]
    except ValueError:
        return None
    return np.array(v) if len(v) >= n else None

def side_from_geometry(h):
    cx = {}
    for r in h.itertuples(index=False):
        ipp = _hdr_vec(getattr(r, 'ImagePositionPatient', None), 3)
        iop = _hdr_vec(getattr(r, 'ImageOrientationPatient', None), 6)
        ps = _hdr_vec(getattr(r, 'PixelSpacing', None), 2)
        rows, cols = (getattr(r, 'Rows', None), getattr(r, 'Columns', None))
        if ipp is None or iop is None or ps is None or (not rows) or (not cols):
            continue
        try:
            c = ipp[:3] + iop[:3] * ps[1] * float(cols) / 2 + iop[3:6] * ps[0] * float(rows) / 2
        except (TypeError, ValueError):
            continue
        cx.setdefault(r.StudyInstanceUID, []).append(float(c[0]))
    out = {}
    for st, xs in cx.items():
        m = float(np.median(xs))
        out[st] = None if abs(m) < LAT_MIN_OFFSET_MM else 'R' if m < 0 else 'L'
    return out

def side_from_corner_x(h):
    out = {}
    for st, g in h.groupby('StudyInstanceUID'):
        xs = []
        for r in g.itertuples(index=False):
            ipp = _hdr_vec(getattr(r, 'ImagePositionPatient', None), 3)
            if ipp is not None and np.isfinite(ipp).all():
                xs.append(float(ipp[0]))
        if not xs:
            out[st] = None
            continue
        x = float(np.median(xs))
        out[st] = None if abs(x) < LEGACY_LAT_OFFSET_MM else 'R' if x < 0 else 'L'
    return out

def lat_of(h, tag=''):
    geo = side_from_corner_x(h) if RULES['lat'] == 'corner_x' else side_from_geometry(h)
    d, n_tag, n_geo, n_none, n_disagree = ({}, 0, 0, 0, 0)
    for st, g in h.groupby('StudyInstanceUID'):
        v = [str(x).strip().upper() for x in g['Laterality'].dropna()]
        if RULES['lat'] == 'corner_x' and 'ImageLaterality' in g.columns:
            v += [str(x).strip().upper() for x in g['ImageLaterality'].dropna()]
        v = [x[0] for x in v if x and x[0] in ('L', 'R')]
        side = v[0] if v else None
        if side is not None:
            n_tag += 1
            if geo.get(st) is not None and geo[st] != side:
                n_disagree += 1
        else:
            side = geo.get(st)
            n_geo += side is not None
            n_none += side is None
        d[st] = side
    log(f'{tag}laterality: {n_tag} from the tag, {n_geo} from geometry, {n_none} unresolved; tag and geometry disagree on {n_disagree} ({n_disagree / max(n_tag, 1):.1%} of the tagged)')
    return d

def probe(item):
    split, study, series, path = item
    row = {'split': split, 'StudyInstanceUID': study, 'SeriesInstanceUID': series, 'dir': path}
    try:
        files = sorted((e.name for e in os.scandir(path) if e.name.endswith('.dcm')))
        row['files'] = files
        row['n_slices'] = len(files)
        if not files:
            return row
        ds = pydicom.dcmread(os.path.join(path, files[len(files) // 2]), stop_before_pixels=True, force=True)
        for t in HDR_TAGS:
            v = getattr(ds, t, None)
            if v is None:
                row[t] = None
            elif isinstance(v, (list, tuple)) or type(v).__name__ == 'MultiValue':
                row[t] = '|'.join((str(x) for x in v))
            else:
                row[t] = str(v)
    except Exception as exc:
        row['err'] = str(exc)[:120]
    return row

def walk(split):
    base = ROOT / split
    items = []
    if not base.is_dir():
        return pd.DataFrame(columns=['split', 'StudyInstanceUID', 'SeriesInstanceUID', 'dir', 'files', 'n_slices'] + HDR_TAGS)
    for study in os.scandir(base):
        if study.is_dir():
            for series in os.scandir(study.path):
                if series.is_dir():
                    items.append((split, study.name, series.name, series.path))
    with ThreadPoolExecutor(max_workers=HDR_THREADS) as pool:
        rows = list(pool.map(probe, items))
    return pd.DataFrame(rows)

def annotate(df):
    desc = df['SeriesDescription'].fillna('') + ' ' + df['SequenceName'].fillna('')
    desc = desc.str.lower().str.replace(_SEP, ' ', regex=True)
    opts = df['ScanOptions'].fillna('').str.upper().str.split('|')
    opts_fs = opts.apply(lambda ts: any((t.strip() in FATSAT_OPTS for t in ts)))
    df['fatsat'] = desc.str.contains(_FATSAT_RX) | opts_fs
    tr = pd.to_numeric(df['RepetitionTime'], errors='coerce')
    te = pd.to_numeric(df['EchoTime'], errors='coerce')
    gre = df['ScanningSequence'].fillna('').str.upper().str.contains('GR')
    t1, t2, pdw = (desc.str.contains(_T1_RX), desc.str.contains(_T2_RX), desc.str.contains(_PD_RX))
    df['weight'] = np.where(t1 & ~t2 & ~pdw, 'T1', np.where(t2 & ~pdw, 'T2', np.where(pdw, 'PD', np.where(gre, 'GRE', np.where(tr < 800, 'T1', np.where(te > 60, 'T2', np.where(tr >= 800, 'PD', 'UNK')))))))
    df['fluid'] = np.isin(df['weight'], ['PD', 'T2'])
    df['px'] = pd.to_numeric(df['PixelSpacing'].fillna('').str.split('|').str[0].replace('', np.nan), errors='coerce')
    return df
#### CELL 7
def pick_slots(series_df, plane_map):
    series_df = series_df.copy()
    series_df['plane'] = series_df['SeriesInstanceUID'].map(plane_map)
    out = {}
    for study, g in series_df.groupby('StudyInstanceUID'):
        chosen = {}
        for name, plane, fluid, fs in SLOTS:
            sel = (g['plane'] == plane) & (g['fatsat'] == fs)
            if fluid is not None:
                sel &= g['fluid'] == fluid
            cand = g[sel]
            if len(cand) == 0 and RULES['slot_fallback'] and (fluid is False):
                cand = g[(g['plane'] == plane) & ~g['fatsat']]
            if len(cand):
                chosen[name] = cand.sort_values('n_slices', ascending=False).iloc[0]
        out[study] = chosen
    return out
#### CELL 8
ORDER_TAGS = [(32, 50), (32, 55), (32, 19)]
DECODE_FAILED = []


def _natural_key(name):
    return tuple((int(x) if x.isdigit() else x.lower() for x in re.split('(\\d+)', str(name))))

def _order_dominant_axis(rec):
    files, d = (rec['files'], rec['dir'])
    rows = []
    for pos, f in enumerate(files):
        ipp = inst = None
        try:
            ds = pydicom.dcmread(os.path.join(d, f), force=True, stop_before_pixels=True, specific_tags=['ImagePositionPatient', 'InstanceNumber'])
            raw = getattr(ds, 'ImagePositionPatient', None)
            if raw is not None and len(raw) >= 3:
                c = np.asarray(raw[:3], dtype=np.float64)
                if np.isfinite(c).all():
                    ipp = c
            n = getattr(ds, 'InstanceNumber', None)
            if n is not None:
                inst = float(n)
        except Exception:
            pass
        rows.append((f, ipp, inst, pos))
    placed = [r for r in rows if r[1] is not None]
    need = max(2, int(0.8 * len(rows)))
    if len(placed) >= need:
        xyz = np.stack([r[1] for r in placed])
        axis = int(np.argmax(np.ptp(xyz, axis=0)))
        spare = float(np.nanmedian(xyz[:, axis]))
        rows.sort(key=lambda r: (float(r[1][axis]) if r[1] is not None else spare, r[2] if r[2] is not None else float('inf'), r[3]))
    elif sum((r[2] is not None for r in rows)) >= need:
        rows.sort(key=lambda r: (r[2] if r[2] is not None else float('inf'), r[3]))
    else:
        rows.sort(key=lambda r: _natural_key(r[0]))
    return ([r[0] for r in rows], True)

def order_slices(rec):
    if RULES['order'] == 'dominant_axis':
        return _order_dominant_axis(rec)
    files, d = (rec['files'], rec['dir'])
    keyed = []
    for f in files:
        k = None
        try:
            ds = pydicom.dcmread(os.path.join(d, f), force=True, stop_before_pixels=True, specific_tags=ORDER_TAGS)
            iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
            ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
            k = float(np.dot(ipp, np.cross(iop[:3], iop[3:])))
        except Exception:
            try:
                k = float(ds.InstanceNumber)
            except Exception:
                k = None
        keyed.append((k, f))
    if any((k is None for k, _ in keyed)):
        return (files, False)
    return ([f for _, f in sorted(keyed, key=lambda t: t[0])], True)

def read_slot(rec, n_slice=None, out_size=None):
    n_slice = GROUP if n_slice is None else n_slice
    out_size = IMG if out_size is None else out_size
    files, d, px = (rec.get('ordered') or rec['files'], rec['dir'], rec['px'])
    n = len(files)
    if n == 0:
        return None
    lo, hi = (int(SLICE_BAND[0] * (n - 1)), int(SLICE_BAND[1] * (n - 1)))
    idx = np.unique(np.linspace(lo, hi, n_slice).astype(int)) if hi > lo else np.array([n // 2])
    while len(idx) < n_slice:
        idx = np.append(idx, idx[-1])
    planes = []
    for i in idx[:n_slice]:
        try:
            ds = pydicom.dcmread(os.path.join(d, files[int(i)]), force=True)
            a = ds.pixel_array.astype(np.float32)
            sl = float(getattr(ds, 'RescaleSlope', 1) or 1)
            ic = float(getattr(ds, 'RescaleIntercept', 0) or 0)
            a = a * sl + ic
        except Exception:
            a = None
        planes.append(a)
    got = [k for k, p in enumerate(planes) if p is not None]
    if RULES['decode_fill'] == 'zero':
        if not got:
            DECODE_FAILED.append(rec.get('SeriesInstanceUID', d))
        planes = [np.zeros((out_size, out_size), np.float32) if p is None else p for p in planes]
        got = list(range(len(planes)))
    if not got:
        DECODE_FAILED.append(rec.get('SeriesInstanceUID', d))
        return None
    if len(got) < len(planes):
        DECODE_FAILED.append(rec.get('SeriesInstanceUID', d))
        for k, p in enumerate(planes):
            if p is None:
                planes[k] = planes[min(got, key=lambda j: abs(j - k))]
    shp = planes[0].shape
    planes = [p if p.shape == shp else np.zeros(shp, np.float32) for p in planes]
    vol = np.stack(planes)
    if px and np.isfinite(px) and (px > 0):
        want = int(round(CROP_MM / px))
        h, w = shp
        if 16 < want < min(h, w):
            cy, cx = (h // 2, w // 2)
            half = want // 2
            vol = vol[:, max(0, cy - half):cy + half, max(0, cx - half):cx + half]
    lo_v, hi_v = np.percentile(vol, [1, 99])
    vol = np.clip((vol - lo_v) / max(hi_v - lo_v, 1e-06), 0, 1)
    t = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)
    t = F.interpolate(t, size=(out_size, out_size), mode='bilinear', align_corners=False)
    return (t.squeeze(0) * 255).round().clamp(0, 255).to(torch.uint8)
#### CELL 9
def normalise_laterality(img, plane, lat):
    if lat != 'R':
        return img
    if plane in ('Coronal', 'Axial'):
        return torch.flip(img, dims=[-1])
    return torch.flip(img, dims=[0])
#### CELL 10
ORDER_CACHE = os.environ.get('RSNA_ORDER_CACHE') or None

def build_cache(slot_map, plane_map, lat_map, tag):
    studies = sorted(slot_map)
    sidx = {s: i for i, s in enumerate(studies)}
    cache = np.zeros((len(studies), N_SLOT, CACHE_SLICES, IMG, IMG), np.uint8)
    mask = np.zeros((len(studies), N_SLOT), np.float32)
    log(f'{tag}: cache {cache.shape} = {cache.nbytes / 1024 ** 3:.1f} GB')
    jobs = [(st, k, plane, slot_map[st][name]) for st in studies for k, (name, plane, _, _) in enumerate(SLOTS) if name in slot_map[st]]
    n_job = len(jobs)
    t_ord = time.time()
    n_slice_total = sum((len(j[3]['files']) for j in jobs))
    log(f'{tag}: ordering {len(jobs)} slot-series ({n_slice_total} slice headers)')
    ok = done = 0
    CHUNK_O = 1024
    seen = {}
    if ORDER_CACHE and Path(ORDER_CACHE).is_file():
        try:
            import json as _json
            seen = _json.loads(Path(ORDER_CACHE).read_text())
        except (OSError, ValueError):
            seen = {}
        hit = 0
        for _, _, _, rec in jobs:
            e = seen.get(rec['SeriesInstanceUID'])
            if e and len(e['files']) == len(rec['files']):
                rec['ordered'] = e['files']
                ok += int(e['good'])
                hit += 1
        jobs = [j for j in jobs if 'ordered' not in j[3]]
        log(f'{tag}: {hit} slot-series ordered from {ORDER_CACHE}, {len(jobs)} to read')
    with ThreadPoolExecutor(max_workers=ORDER_THREADS) as pool:
        for c0 in range(0, len(jobs), CHUNK_O):
            block = jobs[c0:c0 + CHUNK_O]
            for (_, _, _, rec), (files, good) in zip(block, pool.map(lambda j: order_slices(j[3]), block)):
                rec['ordered'] = files
                ok += int(good)
                done += 1
                if ORDER_CACHE:
                    seen[rec['SeriesInstanceUID']] = {'files': files, 'good': bool(good)}
            budget = min(ORDER_BUDGET_S, max(60.0, (TIME_BUDGET - (time.time() - T0)) * 0.35))
            if time.time() - t_ord > budget:
                log(f'{tag}: ordering budget spent at {done}/{len(jobs)}; the rest keep file order')
                break
    if ORDER_CACHE and done:
        import json as _json
        _t = Path(ORDER_CACHE).with_suffix('.tmp')
        _t.write_text(_json.dumps(seen))
        _t.replace(Path(ORDER_CACHE))
    log(f'{tag}: ordered {ok}/{n_job} by geometry ({n_job - ok} kept arbitrary) in {time.time() - t_ord:.0f}s')
    jobs = [(st, k, plane, slot_map[st][name]) for st in studies for k, (name, plane, _, _) in enumerate(SLOTS) if name in slot_map[st]]
    log(f'{tag}: decoding {len(jobs)} slot-series')
    n_failed_before = len(DECODE_FAILED)
    CHUNK = 512
    done = 0
    with ThreadPoolExecutor(max_workers=PIX_THREADS) as pool:
        for c0 in range(0, len(jobs), CHUNK):
            block = jobs[c0:c0 + CHUNK]
            for (st, k, plane, _), img in zip(block, pool.map(lambda j: read_slot(j[3], CACHE_SLICES, IMG), block)):
                done += 1
                if img is None:
                    continue
                cache[sidx[st], k] = normalise_laterality(img, plane, lat_map.get(st)).numpy()
                mask[sidx[st], k] = 1.0
            if done % 4096 < CHUNK:
                log(f'  {tag} {done}/{len(jobs)}')
            if time.time() - T0 > TIME_BUDGET:
                log(f'  {tag}: time budget reached during decode')
                break
    n_failed = len(DECODE_FAILED) - n_failed_before
    log(f'{tag}: {int(mask.sum())}/{len(jobs)} slots filled' + (f'; {n_failed} series had a slice that would not decode' if n_failed else ''))
    gc.collect()
    return (studies, cache, mask)
#### CELL 11
class SlotHead(nn.Module):

    def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2, prior=False):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden
        p_ = torch.zeros(n_out, n_slot)
        if prior and n_slot == len(SLOTS) and (n_out == len(TARGETS)):
            for t, slots in SLOT_PRIOR_TABLE.items():
                if t in TARGETS:
                    p_[TARGETS.index(t), list(slots)] = SLOT_PRIOR_STRENGTH
        self.prior = prior
        if prior:
            self.register_buffer('slot_prior', p_)

    def forward(self, x, mask):
        h = self.proj(x) + self.slot_emb
        att = torch.einsum('bsh,oh->bos', h, self.query) / self.hidden ** 0.5
        if self.prior:
            att = att + self.slot_prior.unsqueeze(0)
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -10000.0).softmax(-1)
        ctx = self.drop(torch.einsum('bos,bsh->boh', att, h))
        return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias
#### CELL 12
class Model(nn.Module):

    def __init__(self, backbone, dim, pool='cls_mean', prior=False):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.head = SlotHead(dim * POOL_PARTS[pool], N_SLOT, len(TARGETS), prior=prior)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, imgs, mask, img_size=None):
        B, S = imgs.shape[:2]
        x = imgs.reshape(B * S, *imgs.shape[2:]).float().div_(255.0)
        if img_size is not None and img_size != x.shape[-1]:
            x = F.interpolate(x, size=(img_size, img_size), mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        out = self.backbone(pixel_values=x).last_hidden_state
        patch = out[:, 1:]
        parts = [out[:, 0], patch.mean(1)]
        if self.pool == 'cls_mean_focal':
            k = max(1, patch.shape[1] // 8)
            parts.append(patch.topk(k, dim=1).values.mean(1))
        feat = torch.cat(parts, dim=1).reshape(B, S, -1)
        return self.head(feat, mask)
#### CELL 13
def build_model(unfreeze_last, source=None, variant='small', pool='cls_mean', prior=False):
    from transformers import AutoModel
    p = source if source is not None else find_dinov2(variant)
    if p is None:
        raise FileNotFoundError('DINOv2 weights not attached')
    bb = AutoModel.from_pretrained(str(p))
    n_layer = len(bb.encoder.layer)
    for prm in bb.parameters():
        prm.requires_grad = False
    for blk in bb.encoder.layer[max(0, n_layer - unfreeze_last):]:
        for prm in blk.parameters():
            prm.requires_grad = True
    for prm in bb.layernorm.parameters():
        prm.requires_grad = True
    dim = bb.config.hidden_size
    trainable = sum((p.numel() for p in bb.parameters() if p.requires_grad))
    log(f'backbone: {n_layer} blocks, last {unfreeze_last} trainable ({trainable / 1000000.0:.1f}M params), feature dim {dim * POOL_PARTS[pool]}')
    return Model(bb, dim, pool=pool, prior=prior)
#### CELL 14
FINGERPRINT_TOL = 0.002

def fingerprint(model, dev, img_size, n_slot=None, group=None, seed=None):
    n_slot = N_SLOT if n_slot is None else n_slot
    group = GROUP if group is None else group
    seed = SEED if seed is None else seed
    g = torch.Generator().manual_seed(seed)
    imgs = torch.randint(0, 256, (2, n_slot, group, img_size, img_size), generator=g, dtype=torch.uint8).to(dev)
    mask = torch.ones(2, n_slot, device=dev)
    mask[1, -1] = 0.0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(imgs, mask, img_size).float().cpu().numpy()
    if was_training:
        model.train()
    return out

def check_fingerprint(model, dev, img_size, expected, tol=FINGERPRINT_TOL, tag=''):
    got = fingerprint(model, dev, img_size)
    exp = np.asarray(expected, np.float32)
    if got.shape != exp.shape:
        raise WeightsError(f'{tag}fingerprint shape {got.shape} != stored {exp.shape}: the architecture is not the one these weights were fitted to')
    d = float(np.abs(got - exp).max())
    if d > tol:
        raise WeightsError(f'{tag}fingerprint differs by {d:.4g} (tolerance {tol:g}). The weights load but do not compute what they computed when fitted - preprocessing, resolution or architecture has moved between the two runs.')
    log(f'{tag}fingerprint matches within {d:.2g}')
    return d

class WeightsError(RuntimeError):
    pass

def find_weights(name='manifest.json'):
    import json
    base = Path('/kaggle/input')
    if not base.is_dir():
        return None
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ('train_series', 'test_series')]
        if name not in files:
            continue
        try:
            man = json.loads((Path(root) / name).read_text())
        except (OSError, ValueError):
            continue
        if isinstance(man.get('members'), list) and man['members']:
            missing = [m['file'] for m in man['members'] if not (Path(root) / m['file']).is_file()]
            if missing:
                raise WeightsError(f"{root} holds a manifest listing {len(man['members'])} members but {len(missing)} of their files are absent (first {missing[0]!r})")
            return Path(root)
    return None
TTA_OVERLAP = True
TTA_POOL = 'prob'
PUBLIC_FRONTIER_TARGET_POOL = {'Fracture': 'max', 'Contusion': 'max', 'Medial Meniscus': 'max', 'Lateral Meniscus': 'max', 'ACL': 'top2', 'MCL': 'top2', "Baker's": 'max'}
TTA_TARGET_POOL = {**PUBLIC_FRONTIER_TARGET_POOL, 'Synovitis': 'original_mean'}
# No-extra-pass diversity branch: smooth focal pooling is evaluated from
# the same no-jitter public-member windows already used by the parent.
LEGACY_FOLD_SOFTPOOL_BETA = {
    'ACL': 6.0, 'MCL': 6.0,
    'Medial Meniscus': 8.0, 'Lateral Meniscus': 8.0,
    "Baker's": 8.0, 'Contusion': 8.0, 'Fracture': 10.0,
}
LEGACY_FOLD_SOFTPOOL_ALPHA = {
    'ACL': 0.20, 'MCL': 0.20,
    'Medial Meniscus': 0.25, 'Lateral Meniscus': 0.25,
    "Baker's": 0.20, 'Contusion': 0.20, 'Fracture': 0.15,
}
LEGACY_MEMBER_WEIGHT_BY_TARGET = {'Lateral Meniscus': 15.0, 'Medial OA': 2.5, 'Lateral OA': 15.0, 'Contusion': 5.0}

def window_starts(n_slice, group, overlap=None):
    overlap = TTA_OVERLAP if overlap is None else overlap
    if overlap and n_slice >= group:
        return list(range(n_slice - group + 1))
    return [g * group for g in range(max(n_slice // group, 1))]

def apply_target_window_pool(values, probs, logits, original_probs, mapping, target_idx):
    for target, mode in mapping.items():
        j = target_idx[target]
        if mode == 'max':
            values[:, j] = probs[:, :, j].max(0).values
        elif mode == 'mean':
            values[:, j] = probs[:, :, j].mean(0)
        elif mode == 'logit_mean':
            values[:, j] = torch.sigmoid(logits[:, :, j].mean(0))
        elif mode == 'original_mean':
            values[:, j] = original_probs[:, :, j].mean(0)
        elif mode in ('top2', 'top3'):
            k = min(int(mode[3:]), probs.shape[0])
            values[:, j] = probs[:, :, j].topk(k, dim=0).values.mean(0)
        else:
            raise ValueError(f'unknown TTA pooling mode for {target}: {mode}')
    return values

def legacy_fold_soft_window_pool(original_probs, target_idx):
    values = original_probs.mean(0).clone()
    for target, beta in LEGACY_FOLD_SOFTPOOL_BETA.items():
        j = target_idx[target]
        x = original_probs[:, :, j]
        weight = torch.softmax(float(beta) * x, dim=0)
        values[:, j] = (weight * x).sum(0)
    return values

@torch.no_grad()
def predict_member(model, cache, mask, idx, dev, img_size, group=None, pool=None, starts=None, jitter=False, jitter_seed=SEED, return_public_frontier=False):
    group = GROUP if group is None else group
    pool = TTA_POOL if pool is None else pool
    starts = window_starts(cache.shape[2], group) if starts is None else list(starts)
    if not starts:
        raise ValueError('predict_member was given no windows to average over')
    target_idx = {t: j for j, t in enumerate(TARGETS)}
    unknown = (set(TTA_TARGET_POOL) | set(PUBLIC_FRONTIER_TARGET_POOL)) - set(target_idx)
    if unknown:
        raise ValueError(f'unknown target(s) in TTA_TARGET_POOL: {unknown}')
    jitter_gen = torch.Generator(device=dev)
    jitter_gen.manual_seed(int(jitter_seed) % (2 ** 63 - 1))
    model.eval()
    out, public_frontier_out, public_soft_out = ([], [], [])
    for b in range(0, len(idx), EVAL_BATCH):
        sel = idx[b:b + EVAL_BATCH]
        m = torch.from_numpy(mask[sel]).to(dev)
        win_probs, win_logits, win_original_probs = ([], [], [])
        for st in starts:
            rows = torch.from_numpy(np.ascontiguousarray(cache[sel, :, st:st + group])).to(dev)
            views = [rows] + ([augment(rows, generator=jitter_gen)] if jitter else [])
            view_probs, view_logits = ([], [])
            for view in views:
                with torch.autocast('cuda', enabled=dev.type == 'cuda'):
                    z = model(view, m, img_size).float()
                view_logits.append(z)
                view_probs.append(torch.sigmoid(z))
            win_logits.append(torch.stack(view_logits).mean(0))
            win_probs.append(torch.stack(view_probs).mean(0))
            win_original_probs.append(view_probs[0])
        probs = torch.stack(win_probs)
        logits = torch.stack(win_logits)
        original_probs = torch.stack(win_original_probs)
        v = torch.sigmoid(logits.mean(0)) if pool == 'logit' else probs.mean(0)
        v = apply_target_window_pool(v, probs, logits, original_probs, TTA_TARGET_POOL, target_idx)
        out.append(v.cpu().numpy())
        if return_public_frontier:
            public_v = apply_target_window_pool(original_probs.mean(0), original_probs, logits, original_probs, PUBLIC_FRONTIER_TARGET_POOL, target_idx)
            public_frontier_out.append(public_v.cpu().numpy())
            public_soft = legacy_fold_soft_window_pool(original_probs, target_idx)
            public_soft_out.append(public_soft.cpu().numpy())
    primary = np.concatenate(out) if out else np.zeros((0, len(TARGETS)), np.float32)
    if not return_public_frontier:
        return primary
    public_frontier = np.concatenate(public_frontier_out) if public_frontier_out else np.zeros((0, len(TARGETS)), np.float32)
    public_soft = np.concatenate(public_soft_out) if public_soft_out else np.zeros((0, len(TARGETS)), np.float32)
    return (primary, public_frontier, public_soft)
BUILD_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()
LEGACY_BUNDLE_FILE = 'rsna_20260807_v1.pt'
LEGACY_WEIGHT = 0.5

def find_legacy_bundle():
    base = Path('/kaggle/input')
    if not base.is_dir():
        return None
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ('train_series', 'test_series')]
        if LEGACY_BUNDLE_FILE in files:
            return Path(root) / LEGACY_BUNDLE_FILE
    return None

def legacy_group_members():
    return {}

def _run_member(path, m, dev, Cte, Mte, idx, starts, jitter):
    t0 = time.time()
    with BUILD_LOCK:
        if 'state' in m:
            state, fp = (m['state'], None)
        else:
            ck = torch.load(Path(path) / m['file'], map_location='cpu', weights_only=False)
            state, fp = (ck['model'], ck.get('fingerprint'))
        model = build_model(int(m['config']['unfreeze_last']), variant=m['config']['variant'], pool=m['config'].get('pool', 'cls_mean'), prior=bool(m['config'].get('prior', False))).to(dev)
        model.load_state_dict(state)
        if fp is not None:
            check_fingerprint(model, dev, IMG, fp, tag=f"{m['id']}: ")
        else:
            log(f"  {m['id']}: no stored fingerprint (legacy bundle) -- accepted at reduced weight")
    t_ready = time.time()
    jitter_seed = SEED + int(hashlib.sha256(str(m['id']).encode()).hexdigest()[:8], 16)
    public_member = 'state' not in m
    predicted = predict_member(model, Cte, Mte, idx, dev, IMG, starts=starts, jitter=jitter, jitter_seed=jitter_seed, return_public_frontier=public_member)
    if public_member:
        p, public_p, public_soft = predicted
    else:
        p, public_p, public_soft = (predicted, None, None)
    t_done = time.time()
    del model, state
    gc.collect()
    if dev.type == 'cuda':
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()
    passes = len(starts) * (2 if jitter else 1)
    return (p, public_p, public_soft, (t_ready - t0, (t_done - t_ready) / max(passes, 1)))

def _combine(per_member):
    all_ids = sorted({s for m in per_member for s in m['ids']})
    pos = {s: i for i, s in enumerate(all_ids)}
    acc = np.zeros((len(all_ids), len(TARGETS)), np.float64)
    tot = np.zeros(len(TARGETS), np.float64)
    for m in per_member:
        target_weight = m.get('target_weight')
        w = np.asarray(target_weight if target_weight is not None else [float(m.get('weight', 1.0))] * len(TARGETS), dtype=np.float64)
        if w.shape != (len(TARGETS),) or np.any(w < 0):
            raise ValueError(f"invalid target weights for {m.get('id')}: {w}")
        r = pd.DataFrame(m['pred']).rank(pct=True).to_numpy()
        acc[[pos[s] for s in m['ids']]] += r * w[None, :]
        tot += w
    if np.any(tot <= 0):
        raise ValueError(f'at least one target has no ensemble vote: {tot}')
    return (all_ids, acc / tot[None, :])

def combine_public_members_by_fold(per_member, pred_key='pred'):
    # Raw-average the four members within each fold, rank each fold,
    # then give all five folds equal weight.
    all_ids = sorted({study for member in per_member for study in member['ids']})
    position = {study: i for i, study in enumerate(all_ids)}
    groups = {}
    for i, member in enumerate(per_member):
        fold = member.get('fold')
        key = f'fold_{fold}' if fold is not None else f'member_{i}'
        groups.setdefault(key, []).append(member)
    fold_ranks, diagnostics = ([], [])
    for key, members_in_fold in sorted(groups.items()):
        matrices = []
        for member in members_in_fold:
            values = np.full((len(all_ids), len(TARGETS)), np.nan, np.float64)
            values[[position[study] for study in member['ids']]] = np.asarray(member[pred_key], np.float64)
            if np.isnan(values).any():
                raise WeightsError(f"{member.get('id')}: incomplete {pred_key} coverage")
            matrices.append(values)
        raw_fold_mean = np.mean(matrices, axis=0)
        fold_ranks.append(pd.DataFrame(raw_fold_mean).rank(method='average', pct=True).to_numpy(np.float64))
        diagnostics.append({'ensemble_group': key, 'members': len(members_in_fold)})
    if len(fold_ranks) != 5:
        raise WeightsError(f'legacy branch requires five folds, found {len(fold_ranks)}')
    return all_ids, np.mean(fold_ranks, axis=0), pd.DataFrame(diagnostics)

def blend_legacy_frontier_and_soft(frontier_rank, soft_rank):
    output = np.asarray(frontier_rank, np.float64).copy()
    for j, target in enumerate(TARGETS):
        alpha = float(LEGACY_FOLD_SOFTPOOL_ALPHA.get(target, 0.0))
        if alpha:
            output[:, j] = (1.0 - alpha) * frontier_rank[:, j] + alpha * soft_rank[:, j]
    return output

def infer_from_package(path, dev=None):
    man = json.loads((Path(path) / 'manifest.json').read_text())
    members = man['members']
    log(f'weights package: {len(members)} member(s) from {path}; {len(DEVS)} device(s)')
    test_df = pd.read_csv(ROOT / 'test.csv')
    test_series = pd.read_csv(ROOT / 'test_series.csv')
    plane_map = dict(zip(test_series['SeriesInstanceUID'], test_series['Anatomical_Plane']))
    hte = annotate(walk('test_series'))
    log(f'test header pass: {len(hte)} series')
    groups = {}
    for m in members:
        groups.setdefault(m['pixel_group'], []).append(m)
    groups.update(legacy_group_members())
    per_member, public_frontier_members = ([], [])
    est = {'fixed': None, 'win': None}

    def bank(m, ids, pred, starts, jitter, public_pred=None, public_soft=None):
        if float(np.std(pred)) < 1e-09:
            log(f"  {m['id']}: degenerate predictions; not banked")
            return
        with STATE_LOCK:
            per_member.append({'id': m['id'], 'fold': m.get('fold'), 'ids': ids, 'pred': pred, 'weight': m.get('weight', 1.0), 'target_weight': m.get('target_weight'), 'holdout': m.get('holdout')})
            if public_pred is not None and len(starts) == len(starts_full):
                if float(np.std(public_pred)) < 1e-09:
                    raise WeightsError(f"{m['id']}: degenerate public-frontier prediction")
                public_frontier_members.append({'id': m['id'], 'fold': m.get('fold'), 'ids': ids, 'pred': public_pred, 'soft_pred': public_soft})
            elif public_pred is not None:
                log(f"  {m['id']}: public-frontier vote omitted because only {len(starts)} / {len(starts_full)} windows completed")
            log(f"  banked {m['id']} fold {m.get('fold', '?')} ({len(starts)} window(s); {len(per_member)} member(s)")
    for gi, (key, gm) in enumerate(groups.items(), 1):
        cfg = json.loads(key)
        adopt_config_globals(cfg)
        log(f"decode group {gi}/{len(groups)}: {cfg['img']}px x {cfg['slices']} slices, crop {cfg['crop_mm']} mm -> {len(gm)} member(s)")
        st_te, Cte, Mte = build_cache(pick_slots(hte, plane_map), plane_map, lat_of(hte, 'test '), f'test g{gi}')
        idx = np.arange(len(st_te))
        starts_full = window_starts(Cte.shape[2], GROUP)
        pending = sorted(gm, key=lambda m: -(m.get('holdout') or 0))
        left_after = sum((len(g) for j, (_, g) in enumerate(groups.items(), 1) if j > gi))

        def pop_next():
            with STATE_LOCK:
                if not pending:
                    return (None, None, False)
                left = TIME_BUDGET - (time.time() - T0)
                remaining = len(pending) + left_after
                slots_left = -(-remaining // len(DEVS))
                starts, jit = (starts_full, False)
                if est['fixed'] is not None and est['win'] is not None:
                    afford = max(left * 0.9, 0.0)
                    room = afford / max(slots_left, 1)
                    if est['fixed'] + est['win'] > room:
                        log(f'  {left / 60:.0f} min left: surrendering {len(pending)} member(s); not one more fits')
                        pending.clear()
                        return (None, None, False)
                    per_win = est['win'] * (2 if jit else 1)
                    n_win = int((room - est['fixed']) / per_win) if per_win > 0 else len(starts_full)
                    n_win = max(1, min(len(starts_full), n_win))
                    if n_win < len(starts_full):
                        mid = (len(starts_full) - n_win) // 2
                        starts = starts_full[mid:mid + n_win]
                return (pending.pop(0), starts, jit)

        def worker(dev):
            others = [d for d in DEVS if d is not dev]
            while True:
                m, starts, jit = pop_next()
                if m is None:
                    return
                for attempt, d in enumerate([dev] + others[:1]):
                    try:
                        p, public_p, public_soft, (fs, ws) = _run_member(path, m, d, Cte, Mte, idx, starts, jit)
                        with STATE_LOCK:
                            est['fixed'], est['win'] = (fs, ws)
                        bank(m, st_te, p, starts, jit, public_p, public_soft)
                        break
                    except Exception as exc:
                        log(f"  MEMBER {m['id']} failed on {d} ({type(exc).__name__}: {exc}); " + ('retrying on peer device' if attempt == 0 and others else 'dropped -- costs one vote, not the run'))
                        if d.type == 'cuda':
                            with torch.cuda.device(d):
                                torch.cuda.empty_cache()
        threads = [threading.Thread(target=worker, args=(d,)) for d in DEVS]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        del Cte, Mte
        gc.collect()
    if len(public_frontier_members) != len(members):
        raise WeightsError(f'public-frontier inference incomplete: {len(public_frontier_members)} / {len(members)} members')
    frontier_ids, frontier_acc = _combine(public_frontier_members)
    sub = write_submission(frontier_acc, frontier_ids, test_df, 'submission.csv')
    log(f'submission.csv = exact no-jitter public-frontier rank mean of {len(public_frontier_members)} member(s); {sub.shape}')
    return sub

def adopt_config_globals(cfg):
    global IMG, CACHE_IMG, GROUP, CACHE_SLICES, N_GROUP, CROP_MM, SLICE_BAND, RULES
    CACHE_IMG = IMG = int(cfg['img'])
    GROUP = int(cfg['group'])
    CACHE_SLICES = int(cfg['slices'])
    N_GROUP = max(CACHE_SLICES // GROUP, 1)
    CROP_MM = float(cfg['crop_mm'])
    SLICE_BAND = tuple((float(x) for x in cfg['band']))
    rules = cfg.get('rules') or RULES_NATIVE
    unknown = {k: v for k, v in rules.items() if k not in RULES_NATIVE or v not in (RULES_NATIVE[k], RULES_LEGACY[k])}
    if unknown:
        raise WeightsError(f'the members record pixel rules this pipeline cannot reproduce: {unknown}')
    RULES = {**RULES_NATIVE, **rules}
    if [s[0] for s in SLOTS] != list(cfg['slots']):
        raise WeightsError(f"the members were fitted on slots {cfg['slots']} and this pipeline defines {[s[0] for s in SLOTS]}; a weight would be read against the wrong slot")
#### CELL 15
def write_submission(pred, studies, test_df, path):
    sub = pd.DataFrame(pd.DataFrame(pred).rank(pct=True).values, columns=TARGETS)
    sub.insert(0, 'StudyInstanceUID', studies)
    sub = test_df[['StudyInstanceUID']].merge(sub, on='StudyInstanceUID', how='left')
    sub[TARGETS] = sub[TARGETS].fillna(0.5)
    sub.to_csv(path, index=False)
    return sub

def _v37_validate_submission(path, test_df, tag):
    frame = pd.read_csv(path)
    expected = ['StudyInstanceUID'] + TARGETS
    if list(frame.columns) != expected:
        raise ValueError(f'{tag}: columns differ from the competition contract')
    if len(frame) != len(test_df) or not frame['StudyInstanceUID'].is_unique:
        raise ValueError(f'{tag}: row count or StudyInstanceUID uniqueness failed')
    if set(frame['StudyInstanceUID'].astype(str)) != set(test_df['StudyInstanceUID'].astype(str)):
        raise ValueError(f'{tag}: StudyInstanceUID set differs from test.csv')
    values = frame[TARGETS].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f'{tag}: non-finite prediction')
    return frame

def main():
    pkg = find_weights()
    if pkg is None:
        raise WeightsError('required public checkpoint manifest not found')
    infer_from_package(pkg, DEVS[0])
    _v37_validate_submission('submission.csv', pd.read_csv(ROOT / 'test.csv'), 'public frontier')

#### CELL 16
"""Check raw DINO output before the inherited merge/fill/rank writer."""
_rep_original_dino_writer = write_submission
_rep_original_dino_combine = _combine
_rep_dino_raw_members_checked = 0

def _rep_checked_dino_combine(members):
    global _rep_dino_raw_members_checked
    import numpy as _np
    import pandas as _pd
    wanted = _pd.read_csv(ROOT / 'test.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()
    assert len(members) == 20, 'DINO raw member count'
    assert len({member['id'] for member in members}) == 20, 'DINO duplicate members'
    for member in members:
        actual = [str(study) for study in member['ids']]
        raw = _np.asarray(member['pred'])
        assert raw.shape == (len(wanted), 12), 'DINO member raw shape'
        assert _np.isfinite(raw).all(), 'DINO member raw nonfinite'
        assert len(actual) == len(set(actual)) == len(wanted), 'DINO member raw UID uniqueness'
        assert set(actual) == set(wanted), 'DINO member missing test studies'
        _rep_dino_raw_members_checked += 1
    return _rep_original_dino_combine(members)

def _rep_checked_dino_writer(pred, studies, test_df, path):
    import numpy as _np
    raw = _np.asarray(pred)
    actual = [str(study) for study in studies]
    wanted = test_df['StudyInstanceUID'].astype(str).tolist()
    assert raw.shape == (len(wanted), 12), 'DINO raw prediction shape'
    assert _np.isfinite(raw).all(), 'DINO raw nonfinite predictions'
    assert len(actual) == len(set(actual)) == len(wanted), 'DINO raw UID uniqueness'
    assert set(actual) == set(wanted), 'DINO raw missing test studies'
    return _rep_original_dino_writer(pred, studies, test_df, path)

write_submission = _rep_checked_dino_writer
_combine = _rep_checked_dino_combine
_rep_dino_guard_installed = True

#### CELL 17
main()
log('done')
#### CELL 18
_rep_snapshot('dino')

#### CELL 20
_A5_SAVED = dict(globals())
import gc, os, time, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings('ignore')
cv2.setNumThreads(1)
CROP_MM = 130.0
SIZE = 336
SLICE_BAND = (0.12, 0.88)
N_SLICE = 16
INTENSITY = 'slice'
SLOTS = [('Sagittal', 1), ('Sagittal', 0), ('Coronal', 1), ('Coronal', 0), ('Axial', 1), ('Axial', 0)]
N_SLOT = len(SLOTS)
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']

def _find_dir(*names):
    root = Path('/kaggle/input')
    cand = []
    for n in names:
        cand += [root / n, root / 'competitions' / n, root / 'datasets' / n]
        for parent in (root / 'datasets', root / 'competitions', root):
            if parent.is_dir():
                try:
                    cand += [d / n for d in parent.iterdir() if d.is_dir()]
                except OSError:
                    pass
    for p in cand:
        if p.is_dir():
            return p
    return None
COMP = _find_dir('rsna-knee-abnormality-detection')
CKPT = _find_dir('knee-mri-fold-weights')
assert COMP is not None, 'competition data not attached'
assert CKPT is not None, 'fold weights not attached'
assert (COMP / 'sample_submission.csv').exists(), f'no competition data at {COMP}'
assert list(CKPT.glob('*_f*.pt')), f'no checkpoints at {CKPT}'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

#### CELL 21
SERIES_ROOT = COMP / 'test_series'
if not SERIES_ROOT.exists():
    SERIES_ROOT = COMP / 'train_series'
print('series root:', SERIES_ROOT)

def ordered_files(sdir, cap=64):
    keyed = []
    for f in sdir.glob('*.dcm'):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            keyed.append((int(ds.InstanceNumber), str(f)))
        except Exception:
            continue
        if len(keyed) >= cap * 4:
            break
    return [f for _, f in sorted(keyed)]

def series_side(path):
    try:
        return float(pydicom.dcmread(path, stop_before_pixels=True).ImagePositionPatient[0])
    except Exception:
        return 0.0

def read_crop(path):
    try:
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
    except Exception:
        return None
    try:
        ps = float(ds.PixelSpacing[0])
    except Exception:
        ps = CROP_MM / max(arr.shape)
    half = int(round(CROP_MM / ps / 2))
    cy, cx = (arr.shape[0] // 2, arr.shape[1] // 2)
    y0, y1 = (max(0, cy - half), min(arr.shape[0], cy + half))
    x0, x1 = (max(0, cx - half), min(arr.shape[1], cx + half))
    crop = arr[y0:y1, x0:x1]
    return None if crop.size == 0 else crop

def window(crop, lo, hi, flip):
    c = np.clip((crop - lo) / max(hi - lo, 1e-06), 0, 1)
    img = cv2.resize(c, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return img[:, ::-1].copy() if flip else img

def render(path, flip):
    crop = read_crop(path)
    if crop is None:
        return None
    lo, hi = np.percentile(crop[::4, ::4], [1, 99])
    return window(crop, lo, hi, flip)

def build_study(args):
    idx, study, recs = args
    out = np.zeros((N_SLOT, N_SLICE, SIZE, SIZE), np.uint8)
    mask = np.zeros(N_SLOT, np.uint8)
    rows = pd.DataFrame(recs)
    if len(rows):
        for s_i, (plane, fs) in enumerate(SLOTS):
            sub = rows[(rows.Anatomical_Plane == plane) & (rows.Fat_Suppression == fs)]
            if sub.empty:
                continue
            files = ordered_files(SERIES_ROOT / study / sub.iloc[0].SeriesInstanceUID)
            if not files:
                continue
            flip = plane != 'Sagittal' and series_side(files[0]) < 0
            lo, hi = SLICE_BAND
            i0 = int(round(lo * (len(files) - 1)))
            i1 = int(round(hi * (len(files) - 1)))
            avail = list(range(i0, i1 + 1))
            if len(avail) >= N_SLICE:
                picks = [avail[int(round(t))] for t in np.linspace(0, len(avail) - 1, N_SLICE)]
                off = 0
            else:
                picks, off = (avail, (N_SLICE - len(avail)) // 2)
            if INTENSITY == 'series':
                crops = [read_crop(files[p]) for p in picks]
                got = [x for x in crops if x is not None]
                if got:
                    samp = np.concatenate([x[::4, ::4].ravel() for x in got])
                    lo_, hi_ = np.percentile(samp, [1, 99])
                    for c, x in enumerate(crops):
                        if x is None:
                            x = read_crop(files[min(len(files) - 1, picks[c] + 1)])
                        if x is not None:
                            out[s_i, off + c] = (window(x, lo_, hi_, flip) * 255).astype(np.uint8)
            else:
                for c, p in enumerate(picks):
                    img = render(files[p], flip)
                    if img is None:
                        img = render(files[min(len(files) - 1, p + 1)], flip)
                    if img is not None:
                        out[s_i, off + c] = (img * 255).astype(np.uint8)
            mask[s_i] = len(picks)
    return (idx, out, mask)
sub_df = pd.read_csv(COMP / 'sample_submission.csv')
ser_csv = pd.read_csv(COMP / 'test_series.csv')
if not (COMP / 'test_series').exists():
    ser_csv = pd.read_csv(COMP / 'train_series.csv')
ser_csv = ser_csv.loc[:, ~ser_csv.columns.duplicated()]
studies = sub_df.StudyInstanceUID.tolist()
by = {s: g.to_dict('records') for s, g in ser_csv[ser_csv.StudyInstanceUID.isin(set(studies))].groupby('StudyInstanceUID')}
print(f'{len(studies):,} test studies, {len(by):,} with series metadata')
#### CELL 22
N_SLOT_TYPES, MASK_IDX = (6, 0)

def segment_softmax(scores, sidx, B):
    T, K = scores.shape
    idx = sidx.unsqueeze(1).expand(-1, K)
    m = torch.full((B, K), float('-inf'), device=scores.device, dtype=scores.dtype)
    m = m.scatter_reduce(0, idx, scores, reduce='amax', include_self=True)
    e = (scores - m[sidx]).exp()
    s = torch.zeros(B, K, device=scores.device, dtype=scores.dtype).index_add_(0, sidx, e)
    return e / s[sidx].clamp(min=1e-06)

class MeanMaxPool(nn.Module):

    def forward(self, f, sidx, B, slot=None, return_attn=False):
        D = f.shape[1]
        cnt = torch.zeros(B, device=f.device, dtype=f.dtype).index_add_(0, sidx, torch.ones(f.shape[0], device=f.device, dtype=f.dtype))
        mean = torch.zeros(B, D, device=f.device, dtype=f.dtype).index_add_(0, sidx, f)
        mean = mean / cnt.clamp(min=1).unsqueeze(1)
        mx = torch.full((B, D), -10000.0, device=f.device, dtype=f.dtype)
        mx = mx.scatter_reduce(0, sidx.unsqueeze(1).expand(-1, D), f, reduce='amax', include_self=True)
        return (torch.cat([mean, mx], 1), None)

class LabelAttentionPool(nn.Module):

    def __init__(self, d, n_labels=12, n_heads=4, slot_bias=True):
        super().__init__()
        self.d, self.k, self.h = (d, n_labels, n_heads)
        self.q = nn.Parameter(torch.randn(n_labels, d) * 0.02)
        self.key, self.val = (nn.Linear(d, d), nn.Linear(d, d))
        self.slot_bias = nn.Parameter(torch.zeros(n_labels, N_SLOT_TYPES + 1)) if slot_bias else None

    def forward(self, f, sidx, B, slot=None, return_attn=False):
        scores = self.key(f) @ self.q.t() / self.d ** 0.5
        if self.slot_bias is not None and slot is not None:
            scores = scores + self.slot_bias.t()[slot]
        a = segment_softmax(scores, sidx, B)
        out = torch.zeros(B, self.k, self.d, device=f.device, dtype=f.dtype)
        out = out.index_add_(0, sidx, a.unsqueeze(-1) * self.val(f).unsqueeze(1))
        return (out, a)

class TokenXAttnPool(nn.Module):

    def __init__(self, d, n_labels=12, n_heads=6, dropout=0.2):
        super().__init__()
        self.d, self.k = (d, n_labels)
        self.q = nn.Parameter(torch.randn(n_labels, d) * 0.02)
        self.slot_emb = nn.Embedding(N_SLOT_TYPES + 1, d, padding_idx=0)
        self.kv_norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)

    def forward(self, tok, sidx, B, slot=None, return_attn=False):
        T, N, D = tok.shape
        cnt = torch.bincount(sidx, minlength=B)
        S = int(cnt.max().item())
        starts = torch.cumsum(cnt, 0) - cnt
        pos = torch.arange(T, device=tok.device) - starts[sidx]
        kv = tok + self.slot_emb(slot).unsqueeze(1)
        pad = tok.new_zeros(B, S, N, D)
        pad[sidx, pos] = kv
        keep = torch.zeros(B, S, dtype=torch.bool, device=tok.device)
        keep[sidx, pos] = True
        kpm = ~keep.repeat_interleave(N, dim=1)
        pad = self.kv_norm(pad.reshape(B, S * N, D))
        q = self.q.unsqueeze(0).expand(B, -1, -1)
        att, w = self.attn(q, pad, pad, key_padding_mask=kpm, need_weights=return_attn, average_attn_weights=True)
        cls = tok[:, 0]
        mean = torch.zeros(B, D, device=tok.device, dtype=tok.dtype).index_add_(0, sidx, cls) / cnt.clamp(min=1).unsqueeze(1)
        mx = torch.full((B, D), -10000.0, device=tok.device, dtype=tok.dtype)
        mx = mx.scatter_reduce(0, sidx.unsqueeze(1).expand(-1, D), cls, reduce='amax', include_self=True)
        base = torch.cat([mean, mx], 1).unsqueeze(1).expand(-1, self.k, -1)
        return (torch.cat([att, base], -1), w)

class ViTSlotToken(nn.Module):

    def __init__(self, vit, n_cat, dim=None):
        super().__init__()
        self.vit = vit
        d = dim or vit.embed_dim
        self.tok = nn.Embedding(n_cat + 1, d, padding_idx=MASK_IDX)
        self.num_features = vit.num_features
        self._orig_prefix = getattr(vit, 'num_prefix_tokens', 1)
        vit.num_prefix_tokens = self._orig_prefix + 1
        for blk in vit.blocks:
            a = getattr(blk, 'attn', None)
            if a is not None and hasattr(a, 'num_prefix_tokens'):
                a.num_prefix_tokens = a.num_prefix_tokens + 1

    @staticmethod
    def _maybe(mod, x):
        return x if mod is None else mod(x)

    def forward_features(self, x, cat):
        v = self.vit
        x = v.patch_embed(x)
        pos = v._pos_embed(x)
        rope = None
        if isinstance(pos, tuple):
            x, rope = pos
        else:
            x = pos
        x = self._maybe(getattr(v, 'patch_drop', None), x)
        x = self._maybe(getattr(v, 'norm_pre', None), x)
        npt = self._orig_prefix
        tok = self.tok(cat).unsqueeze(1)
        x = torch.cat([x[:, :npt], tok, x[:, npt:]], dim=1)
        if rope is not None:
            if getattr(v, 'rope_mixed', False):
                for i, blk in enumerate(v.blocks):
                    x = blk(x, rope=rope[i])
            else:
                for blk in v.blocks:
                    x = blk(x, rope=rope)
        else:
            x = v.blocks(x)
        return v.norm(x)

    def forward_head(self, x, pre_logits=True):
        return self.vit.forward_head(x, pre_logits=pre_logits)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class _GatedDepthBlock(nn.Module):

    def __init__(self, n_slice, dropout=0.0, ls_init=0.1):
        super().__init__()
        self.norm = nn.GroupNorm(1, n_slice)
        self.v = nn.Conv2d(n_slice, n_slice, 1)
        self.g = nn.Conv2d(n_slice, n_slice, 1)
        self.out = nn.Conv2d(n_slice, n_slice, 1)
        self.gamma = nn.Parameter(torch.full((n_slice, 1, 1), ls_init))
        self.drop = nn.Dropout2d(dropout) if dropout else nn.Identity()

    def forward(self, x):
        z = self.norm(x)
        return x + self.gamma * self.drop(self.out(self.v(z) * F.silu(self.g(z))))

class DepthCompress(nn.Module):

    def __init__(self, n_slice=16, out_ch=3, depth=1, dropout=0.0, ls_init=0.1, imagenet=True, proj_noise=0.25):
        super().__init__()
        self.imagenet = imagenet
        self.blocks = nn.ModuleList([_GatedDepthBlock(n_slice, dropout, ls_init) for _ in range(depth)])
        self.proj = nn.Conv2d(n_slice, out_ch, 1, bias=True)
        if imagenet:
            self.register_buffer('mu', torch.tensor(IMAGENET_MEAN).view(1, -1, 1, 1))
            self.register_buffer('sd', torch.tensor(IMAGENET_STD).view(1, -1, 1, 1))

    def forward(self, x):
        keep = (x.amax(dim=1, keepdim=True) > 0).to(x.dtype)
        z = x
        for b in self.blocks:
            z = b(z)
        z = self.proj(z)
        if self.imagenet:
            z = (z - self.mu.to(z.dtype)) / self.sd.to(z.dtype)
        return z * keep
N_PLANE, N_CONTRAST = (3, 2)
_PLANE_OF = lambda s: torch.clamp(s - 1, 0, 5) // 2
_CONTRAST_OF = lambda s: torch.clamp(s - 1, 0, 5) % 2

class SlotDepthMixer(nn.Module):

    def __init__(self, n_slice=16, ksize=5, alpha_max=0.25):
        super().__init__()
        self.n_slice, self.ksize, self.r = (n_slice, ksize, ksize // 2)
        self.alpha_max = alpha_max
        b = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        self.register_buffer('base', b.log()[self.r:])
        n_u = self.r + 1
        self.shared = nn.Parameter(torch.zeros(n_u))
        self.plane_k = nn.Parameter(torch.zeros(N_PLANE, n_u))
        self.contrast_k = nn.Parameter(torch.zeros(N_CONTRAST, n_u))
        self.g0 = nn.Parameter(torch.zeros(()))
        self.gate_p = nn.Parameter(torch.zeros(N_PLANE))
        self.gate_c = nn.Parameter(torch.zeros(N_CONTRAST))
        idx = torch.arange(n_slice)
        self.register_buffer('off', idx[None, :] - idx[:, None])

    def kernel(self, slot):
        p, c = (_PLANE_OF(slot), _CONTRAST_OF(slot))
        half = self.base + self.shared + self.plane_k[p] + self.contrast_k[c]
        full = torch.cat([half.flip(-1)[..., :self.r], half], dim=-1)
        return F.softmax(full, dim=-1)

    def alpha(self, slot):
        p, c = (_PLANE_OF(slot), _CONTRAST_OF(slot))
        return self.alpha_max * torch.tanh(self.g0 + self.gate_p[p] + self.gate_c[c])

    def forward(self, x, slot, vmask):
        T, S, H, W = x.shape
        if vmask is None:
            raise ValueError('stem=mixer requires the padding mask')
        k = self.kernel(slot)
        v = vmask.to(k.dtype)
        d = self.off + self.r
        inb = (d >= 0) & (d < self.ksize)
        kk = k[:, d.clamp(0, self.ksize - 1)] * inb
        M = kk * v[:, None, :]
        den = M.sum(-1, keepdim=True)
        eye = torch.eye(S, device=x.device, dtype=M.dtype).expand(T, S, S)
        ok = (den > 1e-06) & v[:, :, None].bool()
        M = torch.where(ok, M / den.clamp(min=1e-06), eye)
        a = self.alpha(slot)[:, None, None]
        Aop = ((1.0 - a) * eye + a * M).to(x.dtype)
        if x.is_contiguous(memory_format=torch.channels_last) and (not x.is_contiguous()):
            y = torch.bmm(x.permute(0, 2, 3, 1).reshape(T, H * W, S), Aop.transpose(1, 2))
            return y.reshape(T, H, W, S).permute(0, 3, 1, 2)
        return torch.bmm(Aop, x.reshape(T, S, H * W)).reshape(T, S, H, W)

def _seg_mean_max(v, sidx, B):
    D = v.shape[1]
    cnt = torch.zeros(B, device=v.device, dtype=v.dtype).index_add_(0, sidx, torch.ones(v.shape[0], device=v.device, dtype=v.dtype))
    mean = torch.zeros(B, D, device=v.device, dtype=v.dtype).index_add_(0, sidx, v)
    mean = mean / cnt.clamp(min=1).unsqueeze(1)
    mx = torch.full((B, D), -10000.0, device=v.device, dtype=v.dtype)
    mx = mx.scatter_reduce(0, sidx.unsqueeze(1).expand(-1, D), v, reduce='amax', include_self=True)
    return torch.cat([mean, mx], 1)

def _pad_kv(x, sidx, B, norm):
    T, P, D = x.shape
    cnt = torch.bincount(sidx, minlength=B)
    S = int(cnt.max().item())
    starts = torch.cumsum(cnt, 0) - cnt
    pos = torch.arange(T, device=x.device) - starts[sidx]
    pad = x.new_zeros(B, S, P, D)
    pad[sidx, pos] = x
    keep = torch.zeros(B, S, dtype=torch.bool, device=x.device)
    keep[sidx, pos] = True
    return (norm(pad.reshape(B, S * P, D)), ~keep.repeat_interleave(P, dim=1))

class _GatedDelta(nn.Module):

    def __init__(self, d, n_labels, n_heads, dropout):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_labels, d) * 0.02)
        self.kv_norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.d_norm = nn.LayerNorm(d)
        self.dw = nn.Parameter(torch.randn(n_labels, d) * (1.0 / d ** 0.5))
        self.db = nn.Parameter(torch.zeros(n_labels))
        self.gate = nn.Parameter(torch.zeros(n_labels))

    def delta(self, pat, sidx, B, return_attn):
        kv, kpm = _pad_kv(pat, sidx, B, self.kv_norm)
        q = self.q.unsqueeze(0).expand(B, -1, -1)
        att, w = self.attn(q, kv, kv, key_padding_mask=kpm, need_weights=return_attn, average_attn_weights=True)
        return ((self.d_norm(att) * self.dw).sum(-1) + self.db, w)

class TokenResidualPool(_GatedDelta):

    def __init__(self, d, n_labels=12, n_heads=6, pe=64, dropout=0.2):
        super().__init__(d, n_labels, n_heads, dropout)
        self.base = nn.Sequential(nn.LayerNorm(2 * d + pe), nn.Dropout(dropout), nn.Linear(2 * d + pe, n_labels))

    def forward(self, tok, slot, sidx, B, pres, return_attn=False):
        base = self.base(torch.cat([_seg_mean_max(tok[:, 1:].mean(1), sidx, B), pres], 1))
        d_, w = self.delta(tok[:, 1:], sidx, B, return_attn)
        return (base + self.gate * d_, w)

class CodexResidualPool(_GatedDelta):

    def __init__(self, d, n_labels=12, n_heads=6, pe=64, dropout=0.2):
        super().__init__(d, n_labels, n_heads, dropout)
        self.base = nn.Sequential(nn.LayerNorm(2 * d + pe), nn.Dropout(dropout), nn.Linear(2 * d + pe, n_labels))

    def forward(self, tok, slot, sidx, B, pres, return_attn=False):
        base = self.base(torch.cat([_seg_mean_max(tok[:, 0], sidx, B), pres], 1))
        d_, w = self.delta(tok[:, 1:], sidx, B, return_attn)
        return (base + self.gate * d_, w)

class ClsAddPool(nn.Module):

    def __init__(self, d, n_labels=12, pe=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(4 * d + pe), nn.Dropout(dropout), nn.Linear(4 * d + pe, n_labels))

    def forward(self, tok, slot, sidx, B, pres, return_attn=False):
        return (self.net(torch.cat([_seg_mean_max(tok[:, 1:].mean(1), sidx, B), _seg_mean_max(tok[:, 0], sidx, B), pres], 1)), None)

class Readout(nn.Module):

    def __init__(self, pool, d, n_labels=12, pe=64):
        super().__init__()
        self.pool_kind, self.k = (pool, n_labels)
        self.pres_emb = nn.Embedding(N_SLOT_TYPES + 1, pe, padding_idx=0)
        if pool in ('xres', 'clsadd', 'xcodex'):
            self.pool = {'xres': TokenResidualPool, 'clsadd': ClsAddPool, 'xcodex': CodexResidualPool}[pool](d, n_labels, pe=pe)
        elif pool in ('attn', 'xattn'):
            if pool == 'xattn':
                self.pool = TokenXAttnPool(d, n_labels)
                wd = 3 * d + pe
            else:
                self.pool = LabelAttentionPool(d, n_labels)
                wd = d + pe
            self.norm = nn.LayerNorm(wd)
            self.w = nn.Parameter(torch.randn(n_labels, wd) * (1.0 / wd ** 0.5))
            self.b = nn.Parameter(torch.zeros(n_labels))
        else:
            self.pool = MeanMaxPool()
            self.net = nn.Sequential(nn.LayerNorm(2 * d + pe), nn.Dropout(0.2), nn.Linear(2 * d + pe, n_labels))
        self.drop = nn.Dropout(0.2)

    def forward(self, f, slot, sidx, B, return_attn=False):
        pe = self.pres_emb(slot)
        pres = torch.zeros(B, pe.shape[1], device=f.device, dtype=f.dtype).index_add_(0, sidx, pe)
        if self.pool_kind in ('xres', 'clsadd', 'xcodex'):
            return self.pool(f, slot, sidx, B, pres)[0]
        pooled, attn = self.pool(f, sidx, B, slot=slot, return_attn=return_attn)
        if self.pool_kind in ('attn', 'xattn'):
            x = torch.cat([pooled, pres.unsqueeze(1).expand(-1, self.k, -1)], -1)
            x = self.drop(self.norm(x))
            return (x * self.w).sum(-1) + self.b
        return self.net(torch.cat([pooled, pres], 1))

class Net(nn.Module):

    def __init__(self, enc, cond, n_meta=0, pool='mean_max', stem='native', n_slice=16):
        super().__init__()
        self.enc, self.cond = (enc, cond)
        self.compress = DepthCompress(n_slice, 3) if stem == 'compress' else None
        self.mixer = SlotDepthMixer(n_slice) if stem == 'mixer' else None
        self.tokens = pool in ('xattn', 'xres', 'clsadd', 'xcodex')
        D = enc.num_features
        self.meta_mlp = nn.Sequential(nn.LayerNorm(n_meta), nn.Linear(n_meta, 128), nn.GELU(), nn.Linear(128, D)) if n_meta > 0 else None
        self.readout = Readout(pool, D)
        if cond == 'post':
            self.slot_emb = nn.Embedding(N_SLOT_TYPES + 1, D, padding_idx=MASK_IDX)

    def forward(self, im, slot, smeta, sidx, B, vm=None):
        if self.mixer is not None:
            im = self.mixer(im, slot, vm)
        if self.compress is not None:
            im = self.compress(im)
        f = self.enc.forward_features(im, slot) if self.cond == 'token' else self.enc.forward_features(im)
        if self.tokens:
            inner = getattr(self.enc, 'vit', self.enc)
            orig = getattr(self.enc, '_orig_prefix', getattr(inner, 'num_prefix_tokens', 1))
            f = torch.cat([f[:, :1], f[:, orig:]], 1)
        else:
            f = self.enc.forward_head(f, pre_logits=True)
            if f.dim() > 2:
                f = f.flatten(1)
        ex = (lambda v: v.unsqueeze(1)) if self.tokens else lambda v: v
        if self.cond == 'post':
            f = f + ex(self.slot_emb(slot))
        if self.meta_mlp is not None and smeta.shape[1] > 0:
            mt = self.meta_mlp(smeta)
            f = torch.cat([f, mt.unsqueeze(1)], 1) if self.tokens else f + mt
        return self.readout(f, slot, sidx, B)
models = []
for ckpt_path in sorted(CKPT.glob('*_f*.pt')):
    z = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = z['cfg']
    _stem = cfg.get('stem', 'native')
    _in = 3 if _stem == 'compress' else cfg.get('n_slice', 16)
    enc = timm.create_model(cfg['backbone'], pretrained=False, num_classes=0, in_chans=_in, **{'img_size': cfg['img']} if 'vit_' in cfg['backbone'] else {})
    if cfg['cond'] == 'token':
        enc = ViTSlotToken(enc, N_SLOT_TYPES)
    m = Net(enc, cfg['cond'], cfg.get('n_meta', 0), cfg['pool'], stem=_stem, n_slice=cfg.get('n_slice', 16))
    missing, unexpected = m.load_state_dict(z['state_dict'], strict=False)
    assert not [k for k in missing if not k.startswith('enc.')], f'missing {missing[:5]}'
    assert not unexpected, f'unexpected {unexpected[:5]}'
    models.append(m.eval())
    print(f"loaded {ckpt_path.name}  fold {z['fold']}  {cfg['backbone']} pool={cfg['pool']} meta={cfg['meta']}")
CFG = cfg
assert CFG.get('n_meta', 0) == 0, f"checkpoint expects {CFG['n_meta']} metadata features -- build slot_meta for the TEST studies and pass it to predict() before submitting"
print(f"\n{len(models)} fold models ready | input norm: {CFG.get('norm', 'none')}")
#### CELL 23
AMP_PREF = 'bf16'

def amp_for(dev):
    if not str(dev).startswith('cuda'):
        return (torch.float32, False)
    cc = torch.cuda.get_device_capability(dev)
    if AMP_PREF == 'bf16':
        return (torch.bfloat16, True)
    if AMP_PREF == 'fp16':
        return (torch.float16, True)
    if AMP_PREF == 'fp32':
        return (torch.float32, False)
    return (torch.bfloat16 if cc >= (8, 0) else torch.float16, True)
AMP_DT, AMP_ON = amp_for(DEV)
WORKERS = max(1, min(4, os.cpu_count() or 4))
CHUNK = 48
MICRO = 8
models = [m.to(DEV).eval() for m in models]
print(f"device {DEV} | amp {str(AMP_DT).split('.')[-1]} (on={AMP_ON}) | workers {WORKERS} | chunk {CHUNK} | micro {MICRO}")

def _norm_(im):
    k = CFG.get('norm', 'none')
    if k == 'zscore':
        m = (im > 0).float()
        n = m.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
        mu = (im * m).sum(dim=(1, 2, 3), keepdim=True) / n
        var = (((im - mu) * m) ** 2).sum(dim=(1, 2, 3), keepdim=True) / n
        return (im - mu) / (var.sqrt() + 1e-06) * m
    if k == 'imagenet':
        m = (im > 0).float()
        return (im - 0.485) / 0.229 * m
    return im

@torch.no_grad()
def _micro(images, masks):
    dev = DEV
    ims, slots, sidx, vms = ([], [], [], [])
    for b in range(len(masks)):
        present = np.nonzero(masks[b] > 0)[0]
        if len(present) == 0:
            continue
        blk = images[b][present]
        ims.append(torch.from_numpy(blk))
        vms.append(torch.from_numpy(blk.reshape(blk.shape[0], blk.shape[1], -1).max(2) > 0))
        slots.append(torch.from_numpy(present + 1).long())
        sidx.append(torch.full((len(present),), b, dtype=torch.long))
    out = np.full((len(models), len(masks), len(LABELS)), np.nan, np.float32)
    if not ims:
        return out
    im = _norm_(torch.cat(ims).to(dev, non_blocking=True).float().div_(255.0))
    sl = torch.cat(slots).to(dev)
    si = torch.cat(sidx).to(dev)
    vm = torch.cat(vms).to(dev)
    sm = torch.zeros(len(sl), CFG.get('n_meta', 0), device=dev)
    per = torch.zeros(len(models), len(masks), len(LABELS), device=dev, dtype=torch.float32)
    with torch.autocast('cuda' if str(dev).startswith('cuda') else 'cpu', dtype=AMP_DT, enabled=AMP_ON):
        for fold_index, model in enumerate(models):
            per[fold_index] = torch.sigmoid(
                model(im, sl, sm, si, len(masks), vm=vm).float()
            )
    got = per.cpu().numpy()
    keep = np.array([(masks[b] > 0).any() for b in range(len(masks))])
    out[:, keep] = got[:, keep]
    return out

def predict(images, masks):
    out = np.full((len(models), len(masks), len(LABELS)), np.nan, np.float32)
    for a in range(0, len(masks), MICRO):
        b = min(a + MICRO, len(masks))
        out[:, a:b] = _micro(images[a:b], masks[a:b])
    return out

# Macro ROC-AUC depends on ordering, so combine fold orderings rather
# than allowing a fold's probability scale to dominate the mean.
preds = np.full((len(models), len(studies), len(LABELS)), np.nan, np.float32)
t0, done = (time.time(), 0)
with ProcessPoolExecutor(max_workers=WORKERS) as ex:
    for c0 in range(0, len(studies), CHUNK):
        block = studies[c0:c0 + CHUNK]
        imgs = np.zeros((len(block), N_SLOT, N_SLICE, SIZE, SIZE), np.uint8)
        msks = np.zeros((len(block), N_SLOT), np.uint8)
        futs = [ex.submit(build_study, (i, s, by.get(s, []))) for i, s in enumerate(block)]
        for f in as_completed(futs):
            try:
                i, a, k = f.result()
                imgs[i], msks[i] = (a, k)
            except Exception as e:
                print(f'  study failed: {type(e).__name__}: {e}')
        preds[:, c0:c0 + len(block)] = predict(imgs, msks)
        done += len(block)
        el = time.time() - t0
        print(f'  {done:,}/{len(studies):,}  {el / 60:.1f}m  eta {el / done * (len(studies) - done) / 60:.1f}m', flush=True)
        del imgs, msks
        gc.collect()
print(f'\ninference done in {(time.time() - t0) / 60:.1f} min')
A5_W = 0.45
A5_LABELS = list(LABELS)
_a5_ok = np.isfinite(preds).all(axis=(0, 2))
_a5_rank_mean = np.zeros((len(studies), len(LABELS)), np.float64)
for fold_index in range(preds.shape[0]):
    fold = preds[fold_index][_a5_ok]
    ordinal = fold.argsort(0).argsort(0).astype(np.float64)
    _a5_rank_mean[_a5_ok] += ordinal / max(len(fold) - 1, 1)
_a5_rank_mean /= preds.shape[0]
_a5_rank_mean[~_a5_ok] = np.nan
A5_PREDS = dict(zip(
    sub_df['StudyInstanceUID'].astype(str), _a5_rank_mean.astype(np.float32)
))
for _a5k, _a5v in _A5_SAVED.items():
    globals()[_a5k] = _a5v
del _A5_SAVED, _a5k, _a5v
#### CELL 24
_a5_sub = pd.read_csv('/kaggle/working/submission.csv',
                      dtype={'StudyInstanceUID': str})
assert _a5_sub.columns.tolist()[1:] == A5_LABELS, 'submission schema drift'
if A5_W > 0:
    _a5_ours = np.stack([A5_PREDS[_u]
                         for _u in _a5_sub['StudyInstanceUID'].astype(str)])
    _a5_base_rank = _a5_sub[A5_LABELS].rank(method='average', pct=True)
    _a5_ours_rank = pd.DataFrame(_a5_ours, columns=A5_LABELS,
                                 index=_a5_sub.index).rank(method='average', pct=True)
    _a5_sub[A5_LABELS] = (1.0 - A5_W) * _a5_base_rank + A5_W * _a5_ours_rank
    assert np.isfinite(_a5_sub[A5_LABELS].to_numpy()).all()
    _a5_sub.to_csv('/kaggle/working/submission.csv', index=False)
#### CELL 25
_rep_snapshot('a5')

#### CELL 27
"""Check every Rad matrix before ranking can conceal nonfinite values."""
_rep_rad_rank_checks = 0
_rep_rad_guard_installed = False
_rep_rad_previous_profile = _rep_sys.getprofile()

def _rep_on_rad_main(frame, event, arg):
    global _rep_rad_guard_installed
    if event != 'call' or frame.f_code.co_name != '_rad_main' or '_rad_rank_columns' not in frame.f_globals:
        if _rep_rad_previous_profile is not None:
            _rep_rad_previous_profile(frame, event, arg)
        return
    namespace = frame.f_globals
    original_rank = namespace['_rad_rank_columns']

    def checked_rad_rank(values):
        global _rep_rad_rank_checks
        import numpy as _np
        raw = _np.asarray(values)
        assert raw.ndim == 2 and raw.shape[1] == 12, 'Rad raw shape'
        assert _np.isfinite(raw).all(), 'Rad raw nonfinite before ranking'
        _rep_rad_rank_checks += 1
        return original_rank(values)

    namespace['_rad_rank_columns'] = checked_rad_rank
    _rep_rad_guard_installed = True
    _rep_sys.setprofile(_rep_rad_previous_profile)

_rep_sys.setprofile(_rep_on_rad_main)

#### CELL 28
%%writefile /kaggle/working/rad_calibrator.json
{
 "coef": [
  [
   0.054988985480198795,
   0.0,
   -0.0,
   0.0,
   -0.006905034746797682,
   0.0,
   -0.0,
   0.0,
   0.0,
   -0.007598718307712681,
   0.004775188808852552,
   -0.0,
   0.0,
   0.0,
   -0.004984520904425084,
   0.0,
   -0.006541710133337969,
   -0.0,
   -0.003752718730904078,
   0.0,
   -0.0,
   -0.0,
   0.0028342800789145934,
   -0.0,
   0.012842247615159527,
   0.0,
   -0.0,
   0.004720465029524873,
   -0.0,
   -0.0,
   -0.003820481837620495,
   0.021610275070444572,
   -0.0,
   -0.009703704271490837,
   0.0,
   -0.0007892533771826419,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   0.0,
   -0.002190521845853738,
   -0.0,
   0.0,
   -0.0,
   0.0018799414367212483,
   -0.0,
   -0.0,
   -0.0,
   0.0,
   0.008708677037306389,
   0.0,
   0.0,
   -0.0006364643665921704,
   -0.0,
   0.0,
   -0.0007433491597744793,
   -0.0,
   -0.0,
   -0.0022552694692117214,
   0.08996657756223722,
   0.0,
   -0.0,
   0.0027210612157672017,
   -0.0,
   0.0,
   -0.009935986032994056,
   0.0014937801726411662,
   -0.0,
   -0.0,
   0.0,
   -0.0,
   0.0,
   0.0,
   -0.0,
   0.0,
   0.0041838897324036756,
   0.0,
   0.0,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   0.0
  ],
  [
   -0.0,
   0.05407410902522933,
   -0.0,
   -0.0,
   -0.00955325444490328,
   -0.0,
   -0.0,
   0.0,
   0.0,
   -0.002741621763298323,
   0.0,
   0.0030651354435245284,
   0.0,
   0.005261907711238927,
   -0.0,
   -0.0,
   -2.246328164835325e-06,
   -0.0,
   -0.007235628651266947,
   0.0,
   -0.0,
   -0.0,
   0.002244318533778753,
   0.0009848822815680745,
   0.0,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   0.0017347234462540013,
   0.0,
   -0.0,
   0.0,
   0.0,
   0.0,
   -0.0,
   -0.0,
   0.0,
   0.0,
   -0.0,
   -0.0,
   0.0,
   -0.0,
   0.0,
   0.0,
   -0.0,
   0.001643286147786208,
   -0.0,
   0.0,
   0.0,
   0.0,
   -0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   -0.0,
   0.0,
   0.032048814954222495,
   -0.0,
   -0.0,
   -0.002201853534088179,
   -0.0,
   -0.0,
   0.0,
   0.0,
   -0.0,
   0.0,
   0.0,
   0.0,
   -0.0,
   -0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0
  ],
  [
   -0.0,
   -0.0,
   0.0591118991425821,
   -0.006355044471722653,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.007699542581442222,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   0.048274735575991476,
   -0.0,
   -0.0,
   -0.008539297840185648,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.004472020944327244,
   0.0,
   -0.0,
   -0.00347437691921257,
   0.05861944258061976,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.006976785974184935,
   -0.0,
   -0.0,
   0.0,
   0.0,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   -0.004010865327087766,
   0.0,
   0.0,
   0.0,
   -0.0005908157372598818,
   0.0,
   0.0,
   -0.0,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   -0.00500875228252724,
   0.0,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   0.12402375784783327,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0015649029295778055,
   -0.0,
   -0.0,
   -0.0,
   0.0,
   -0.0,
   -0.0027029555164037418,
   0.0,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   0.0,
   -0.0,
   0.0,
   -0.0,
   0.0,
   -0.0,
   0.0
  ],
  [
   0.0,
   -0.0,
   -0.0,
   0.020022960936525995,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0017738172621121892,
   -0.0,
   -0.0023183591962150705,
   0.0,
   -0.007515949427429685,
   -0.010155250767449142,
   0.0004984806412923339,
   -0.0,
   0.0,
   -0.005354229345626853,
   0.0,
   0.0,
   -0.0007398489398975964,
   -0.0,
   -0.0,
   0.0,
   -0.004071983093585694,
   -0.0,
   0.01700346063545301,
   -0.0,
   0.0058490781650871355,
   -0.001005834776031484,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   0.0,
   -0.0009063914656936253,
   0.000214420591619211,
   0.0005584969828176026,
   -0.0,
   -0.0020443851541547286,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   0.0,
   0.007754847683752665,
   -0.0005033977078919417,
   0.000708682810146001,
   -0.0,
   0.0,
   -0.003502795873998033,
   0.0,
   0.0,
   -0.0028597700818967357,
   -0.0,
   0.11847279874112572,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0015607762716186155,
   -0.0,
   -0.005456887523612534,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   0.0,
   0.0,
   -0.0,
   -0.0,
   0.0,
   0.0
  ],
  [
   0.0,
   -0.0,
   0.011905299490447545,
   -0.0,
   0.051327525539440685,
   0.0,
   0.006490303077739882,
   0.0,
   0.009189655557113569,
   -0.0032555342419950226,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   0.0,
   0.0,
   0.015598360106187889,
   0.0,
   0.0,
   0.0,
   0.0,
   -0.0007355085953963836,
   0.0,
   0.0,
   0.0,
   -0.005692959829707796,
   0.008383567458721805,
   -0.0004342984045223054,
   0.040683776777621715,
   0.0,
   0.0,
   0.0,
   0.0,
   -0.0,
   0.003946854023482639,
   0.0,
   -0.0,
   0.0,
   -0.0,
   0.003675523046569264,
   -0.0,
   -0.0,
   -0.006683593070962812,
   0.001598200174796656,
   -0.0,
   -0.0,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0009560338374132588,
   0.0,
   0.0014865501499527398,
   0.0,
   0.0,
   -0.006465356514571239,
   0.0,
   -0.0,
   0.08801358135090863,
   0.0,
   0.0,
   0.0,
   0.0,
   -0.0,
   0.0,
   0.0,
   -0.0,
   0.0,
   0.0,
   0.0,
   0.0032578169698303334,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0022856634143569687,
   -0.0008211041163776929,
   0.0,
   0.0,
   0.001979434428479871,
   -0.0006904492760499002
  ],
  [
   0.0,
   -0.0010201300673726335,
   -0.0,
   0.003330913654546868,
   0.0,
   0.03804589769802522,
   0.0036582135089306815,
   -0.0,
   0.0,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0074698905145500535,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.004181180657534594,
   -0.0,
   0.0,
   -0.0014097935467749104,
   0.0,
   -0.008364014702270928,
   -0.0,
   0.0,
   0.0,
   0.031234909820757367,
   0.0,
   0.0,
   0.001761560363567976,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   0.0,
   -0.00010302276929695168,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   -0.0,
   0.0,
   0.0,
   0.0,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   0.0036722429881927794,
   0.0,
   0.0,
   -0.0,
   -0.0,
   0.0,
   -0.028070173174009682,
   -0.0,
   0.03932353957184419,
   0.0,
   0.052709270489857116,
   4.30670771118039e-05,
   0.0,
   0.0,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   -0.0,
   0.0,
   0.005897795403501839,
   -0.0,
   -0.0017091642652794067,
   0.0,
   -0.0,
   -0.0,
   -0.0011445129464464573,
   0.0,
   -3.599923112263568e-05
  ],
  [
   -0.0,
   -0.0,
   0.0,
   0.0,
   0.0,
   0.00398886886838279,
   0.024531745408578224,
   0.0,
   0.003472383623711957,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   -0.0024216642234806546,
   -0.0,
   0.0,
   0.0,
   0.0,
   0.04441771278056577,
   0.0,
   0.0,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   0.0,
   -0.0,
   0.0,
   0.0,
   0.04208386438405934,
   0.0,
   0.0,
   -0.0,
   0.0,
   -0.0,
   -8.871585477522171e-05,
   -0.0,
   -0.0,
   -0.0,
   -0.01794873111002335,
   -0.010965712486455531,
   0.0,
   -0.0,
   -0.0003112348219671695,
   -0.0,
   0.00033734018896491223,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.002767382390020047,
   -0.0,
   -0.0,
   0.0,
   -0.0,
   -0.006553844259659735,
   -0.0,
   0.002859144951920367,
   -0.0,
   -0.0,
   -0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.1080204096279449,
   0.0,
   0.0,
   -0.0,
   0.0,
   -0.008286727880437668,
   -0.0,
   0.0,
   0.0,
   0.0,
   0.0016705366518583718,
   0.0,
   0.003966228970353484,
   0.0,
   -0.0,
   -0.0036942492702868837,
   0.0,
   2.0158865558700288e-05,
   -0.0,
   -0.003290955791230868,
   0.0,
   2.931550215656576e-05
  ],
  [
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   0.15721758161271948,
   0.0,
   -0.0009277854332745394,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   0.0,
   0.0,
   -0.0,
   0.0,
   -0.0035078908538144857,
   0.0,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   0.0029922774103666346,
   0.0,
   -0.0,
   0.0,
   -0.0,
   0.0,
   0.0,
   -0.0,
   0.0,
   0.0,
   0.0015069965777220378,
   0.0,
   -0.0,
   0.0,
   0.0,
   0.0,
   -0.0,
   0.0034575286435870912,
   0.0,
   0.0,
   0.0,
   0.0,
   0.004879501088574802,
   0.0,
   -0.0,
   0.0,
   0.00107584083385987,
   0.0,
   0.0,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   0.07386105032184348,
   0.0,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   0.0,
   -0.0,
   0.0,
   0.0,
   -0.0006098366519528874,
   0.000710602647614561,
   0.0,
   0.00029648470600281623,
   0.0,
   0.0006669555200166979,
   0.0,
   0.0001247447196589612,
   0.0
  ],
  [
   0.0,
   -0.0,
   -0.0,
   0.0,
   0.0,
   0.010971916263113718,
   0.0,
   0.0005950879974343855,
   0.020934939880106,
   -0.0,
   0.0,
   -0.0,
   0.0,
   -0.00716089007036752,
   -0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.012582689029033865,
   -0.0,
   -0.0,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   0.0,
   -0.0,
   0.0,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.05233942478784284,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.003853509415892567,
   -0.009802578580289266,
   0.0,
   0.0,
   0.013526997046656204,
   0.0064884417795260645,
   -0.0,
   0.0002747792719802145,
   0.00794104167516065,
   0.0032425867028947433,
   -0.0
  ],
  [
   0.0,
   -0.0,
   -0.0,
   0.0,
   -0.0,
   0.0,
   -0.0,
   0.0,
   0.0,
   0.07766030485486214,
   0.0,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   0.0,
   -0.0,
   0.0,
   -0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   -0.0,
   0.0016984387848328866,
   -0.02318816122584484,
   -0.0,
   0.0008232018620112374,
   -0.0,
   0.0,
   -0.0,
   0.0030093149725252876,
   0.0,
   0.015496588831608236,
   0.0,
   -0.0,
   0.0,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   0.0,
   0.0,
   0.0,
   -0.0,
   -0.0,
   0.0,
   0.0,
   -0.0,
   -0.0,
   0.0,
   0.0,
   -0.0,
   0.0,
   0.0,
   0.0,
   -0.003985489123804257,
   -0.0,
   0.0,
   -0.0,
   0.0,
   -0.0,
   0.0,
   0.0,
   0.08934738817249994,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   0.0,
   0.0,
   -0.0,
   0.0031358753239660443,
   -0.0,
   0.0,
   0.0,
   0.0,
   -0.002757717666564488,
   0.0,
   0.0,
   0.0,
   -0.001572020817838309
  ],
  [
   0.0,
   0.0,
   -0.007645376480798367,
   -0.0,
   -0.013152846307009906,
   -0.0,
   -0.013405770463176186,
   0.0,
   -0.0,
   -0.0126185896490272,
   0.08456110280686102,
   0.0010498490952287943,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   0.026761069960320228,
   0.004881051675806471,
   0.0,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   0.0,
   -0.0,
   -0.0029949120807847875,
   0.0,
   0.0,
   -0.0013947178093899313,
   -0.0,
   0.0,
   -0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   -0.0,
   0.0,
   0.0,
   -0.0,
   0.0,
   -0.0,
   0.0,
   -0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   -0.0,
   -0.0,
   0.0,
   0.0,
   -0.00942848401064722,
   -0.0,
   -0.0,
   -0.00022519045066539146,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   0.0457033966423892,
   0.027947704687921546,
   0.0,
   -0.0017566658004398543,
   -0.011670049510825595,
   -0.0,
   -0.0,
   -0.0,
   0.0,
   -0.0007825317116991779,
   -0.0,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   0.0,
   -0.0
  ],
  [
   0.0,
   0.0,
   -0.0,
   -0.0,
   -0.001502844459484492,
   -0.0,
   -0.0,
   0.0,
   -0.0,
   -0.0020872198960683284,
   0.009847417955540042,
   0.004860013798480458,
   0.0,
   -0.0,
   -0.0012632551037073973,
   0.0,
   -0.007111539097901386,
   -0.0,
   -0.00195199372867498,
   0.0,
   -0.0,
   -0.0,
   0.003484155321535836,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   0.0,
   -0.0005385599274495165,
   -0.0,
   0.0,
   0.0,
   0.0,
   -0.0,
   -0.0,
   0.0,
   -0.0,
   0.0,
   -0.0,
   0.0,
   -0.0,
   0.0,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   0.0,
   -0.0,
   0.0,
   -0.0,
   -0.002005337879973018,
   0.0,
   -0.0,
   -0.0,
   0.0,
   0.0,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   0.0,
   -0.0,
   -0.0,
   0.0,
   0.04784159508857366,
   0.0,
   -0.0,
   -0.0,
   -0.0,
   -0.0,
   0.0,
   -0.0,
   0.0,
   0.0,
   0.0,
   -0.0,
   0.0,
   0.0,
   0.0,
   -0.0,
   0.0
  ]
 ],
 "gate": [
  "ACL",
  "Medial OA",
  "Lateral OA",
  "PF OA",
  "Effusion",
  "Baker's",
  "Contusion"
 ],
 "groups": [
  [
   "ACL",
   "MCL",
   "Contusion",
   "Fracture"
  ],
  [
   "Medial Meniscus",
   "Lateral Meniscus"
  ],
  [
   "Medial OA",
   "Lateral OA",
   "PF OA"
  ],
  [
   "Effusion",
   "Synovitis",
   "Baker's"
  ]
 ],
 "intercept": [
  0.27594726240322665,
  0.19664147502522833,
  0.46103077516667196,
  0.26000148037766385,
  0.348880476077348,
  0.267855131253255,
  0.4114411126395088,
  0.4677221586179633,
  0.2775079245121116,
  0.3059966406870837,
  0.26281611268502275,
  0.1933670474221893
 ],
 "mean": [
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  -8.169035821569329e-19,
  4.0845179107846643e-19,
  8.169035821569329e-19,
  0.0,
  -2.0422589553923322e-19,
  -7.147906343873162e-19,
  1.0211294776961661e-19,
  2.0422589553923322e-19,
  -6.1267768661769965e-19,
  -8.169035821569329e-19,
  3.0633884330884983e-19,
  -1.2253553732353993e-18,
  -4.0845179107846643e-19,
  3.0633884330884983e-19,
  6.1267768661769965e-19,
  -1.021129477696166e-18,
  -2.0422589553923322e-19,
  -9.190165299265494e-19,
  2.0422589553923322e-19,
  2.0422589553923322e-19,
  -2.0422589553923322e-19,
  -1.2253553732353993e-18,
  -1.0211294776961661e-19,
  0.0,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  0.5001149689583813,
  5.526557829386066,
  2.2368360542653485,
  1.9521729133134054,
  1.337548861807312,
  3.1772821338238675,
  1.0584042308576684,
  1.0482869625201197,
  1.0705909404460796,
  3.1772821338238675,
  1.0584042308576684,
  1.0482869625201197,
  1.0705909404460796
 ],
 "protocol_columns": [
  "n_series",
  "n_Sag",
  "n_Cor",
  "n_Axi",
  "Fat",
  "Fat_Sag",
  "Fat_Cor",
  "Fat_Axi",
  "Flu",
  "Flu_Sag",
  "Flu_Cor",
  "Flu_Axi"
 ],
 "scale": [
  0.28867512696347697,
  0.28867512696347697,
  0.28867512696347697,
  0.28867512695294856,
  0.28867512696347697,
  0.28867512696347697,
  0.28867512696347697,
  0.28867512696347697,
  0.28867512696347697,
  0.28867512696347697,
  0.28867512696347697,
  0.28867512695294856,
  0.28867512338382056,
  0.28867512338382056,
  0.28867512338382056,
  0.28867512338382056,
  0.28867512338382056,
  0.28867512338382056,
  0.288675123394349,
  0.288675123394349,
  0.288675123394349,
  0.288675123394349,
  0.28867512336276374,
  0.28867512338382056,
  0.28867512696347697,
  0.28867512695294856,
  0.28867512695294856,
  0.28867512696347697,
  0.28867512696347697,
  0.28867512696347697,
  0.28867512696347697,
  0.28867512695294856,
  0.2886751269529485,
  0.28867512695294856,
  0.28867512696347697,
  0.28867512694242015,
  0.23019299127577333,
  0.20945811170519504,
  0.18896664844677993,
  0.22034647621868486,
  0.15727493499021805,
  0.19246986757904713,
  0.18959582474964737,
  0.1500338770229888,
  0.17335524281933992,
  0.21499556517074797,
  0.18526691377924037,
  0.24722042176621192,
  0.23300663477492883,
  0.2351154922532429,
  0.19851192071060803,
  0.22614032204035772,
  0.16824789069383667,
  0.19316845808174274,
  0.19078421415473018,
  0.17455770295025105,
  0.1939184152151192,
  0.2380093052256761,
  0.2054846773727961,
  0.25238186441777205,
  0.2560260929356755,
  0.2563420389842119,
  0.2669423487470595,
  0.2578640024055912,
  0.2718501229846065,
  0.26593176523552875,
  0.26652763985208816,
  0.2716787563858938,
  0.26822049981618185,
  0.2551672579287264,
  0.26457231958493127,
  0.24970143393957778,
  0.21746552440992822,
  0.24075616342318434,
  0.2586368483721014,
  0.23201516104802145,
  1.392601296101108,
  0.5943047831182455,
  0.7471955639897663,
  0.6579157483703758,
  0.6336751078357071,
  0.41835483650318195,
  0.35622110896336706,
  0.26841408540232553,
  0.6336751078357071,
  0.41835483650318195,
  0.35622110896336706,
  0.26841408540232553
 ]
}

#### CELL 29
from __future__ import annotations
# ---------------------------------------------------------------------------
# Rad branch calibrator - INHERITED, NOT FITTED BY THIS NOTEBOOK.
# Carried in the parent notebook as a zlib+base64 string; written out above as
# plain JSON instead. Same numbers, readable. It standardises 88 protocol
# features (mean/scale) and applies a 12x88 linear map + intercept per finding.
# Origin: Mattia Angeli, "Bend the Knee to the Dinosaurs" (Apache-2.0).
# ---------------------------------------------------------------------------
# Surgical reproduction of V48's deployed prediction branch.
#
# The pinned reference Rad family is fused with correct-contract E13, then
# the same E13 heads run on the E11 layout at 0.15. No twin/legacy wrapper
# follows it, matching the branch that produced V48's visible submission.

import contextlib as _rad_contextlib
import gc as _rad_gc
import hashlib as _rad_hashlib
import json as _rad_json
import os as _rad_os
import re as _rad_re
import time as _rad_time
from concurrent.futures import ThreadPoolExecutor as _RadThreadPool
from pathlib import Path as _RadPath

import numpy as _rad_np
import pandas as _rad_pd
import pydicom as _rad_pydicom
import torch as _rad_torch
import torch.nn as _rad_nn
import torch.nn.functional as _rad_F
from torchvision.models import resnet50 as _rad_resnet50

_RAD_LABELS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
    'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
    'Contusion', 'Fracture',
]
_RAD_ALPHA = 0.50
_RAD_EXCLUDE = ("Baker's", 'Fracture')
_RAD_HEADS_SHA256 = '54f657826b3458a7ba3d462e198ba380732f2b136246182312704929874a9a2c'
_RAD_REFERENCE_HEADS_SHA256 = '0f465649799ecfbccaac1767844639e7ced44e1bc9babde6e4bac7c5d9b89eaa'
_RAD_ENCODER_SHA256 = '08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734'
_RAD_E13_HEADS_SHA256 = 'ad9f19af73bfdf4e49263c0e45060dc3cb239e1195039b26dc8c0a3a6bcd1a8a'
_RAD_E13_MEMBER_WEIGHT = 0.50
_RAD_V48_SECOND_ALPHA = 0.15
_RAD_TWIN_ALT_WEIGHT = 0.500001
_RAD_TOKEN_DIM, _RAD_HEAD_DIM = 2048, 512

_RAD_E11_SLOTS = [
    ('SAG_NOFS', 'Sagittal', None, False),
    ('COR_NOFS', 'Coronal', None, False),
    ('AX_NOFS', 'Axial', None, False),
    ('SAG_FS', 'Sagittal', None, True),
]
_RAD_E11_CROP_MM = 130.0
_RAD_E11_CACHE_SLICES = 8
_RAD_E11_IMG = 224

_RAD_E13_SLOTS = [
    ('SAG_FS', 'Sagittal', None, True),
    ('COR_FS', 'Coronal', None, True),
    ('AX_FS', 'Axial', None, True),
    ('SAG_NOFS', 'Sagittal', None, False),
]
_RAD_E13_CROP_MM = 130.0
_RAD_E13_CACHE_SLICES = 8
_RAD_E13_IMG = 224

# Our independently trained five-fold family.  Its preprocessing and estimator
# are preserved from V35: native DICOM geometry/fat-sat handling and a mean of
# per-fold percentile ranks (rather than v15's rank of the probability mean).
_OUR_N_SLOT, _OUR_N_SLICE, _OUR_IMG = 3, 8, 224

# Exact V40/E10 test representation: three fat-suppressed planes, eight
# acquired slices per plane, full frame, legacy ordering/laterality/fill.
SLOTS = [
    ('SAG_FS', 'Sagittal', None, True),
    ('COR_FS', 'Coronal', None, True),
    ('AX_FS', 'Axial', None, True),
]
N_SLOT = len(SLOTS)
CACHE_SLICES = 8
IMG = CACHE_IMG = 224
CROP_MM = 10_000.0
SLICE_BAND = (0.2, 0.8)
RULES = dict(RULES_LEGACY)
TIME_BUDGET = 8.0 * 3600


def _rad_log(message):
    print(f'[Rad-dual5] {message}', flush=True)


def _rad_sha256(path, chunk=8 << 20):
    digest = _rad_hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


def _rad_find_file(name, expected_sha=None, explicit_env=None):
    if explicit_env and _rad_os.environ.get(explicit_env):
        candidates = [_RadPath(_rad_os.environ[explicit_env])]
    else:
        candidates = []
        base = _RadPath('/kaggle/input')
        if base.is_dir():
            for root, dirs, files in _rad_os.walk(base):
                dirs[:] = [d for d in dirs if d not in ('train_series', 'test_series')]
                if name in files:
                    candidates.append(_RadPath(root) / name)
    if not candidates:
        raise FileNotFoundError(f'V36 missing input artifact {name}')
    for path in candidates:
        if expected_sha is None or _rad_sha256(path) == expected_sha:
            return path
    raise RuntimeError(f'V36 found {name}, but no copy has the required SHA-256')


class _RadEncoder(_rad_nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _rad_nn.Sequential(
            *list(_rad_resnet50(weights=None).children())[:-2]
        )

    def forward(self, image):
        return self.backbone(image).mean(dim=(2, 3))


class _RadHead(_rad_nn.Module):
    def __init__(self):
        super().__init__()
        self.project = _rad_nn.Sequential(
            _rad_nn.LayerNorm(_RAD_TOKEN_DIM),
            _rad_nn.Linear(_RAD_TOKEN_DIM, _RAD_HEAD_DIM),
            _rad_nn.GELU(),
        )
        self.plane = _rad_nn.Parameter(_rad_torch.randn(N_SLOT, _RAD_HEAD_DIM) * .01)
        self.position = _rad_nn.Parameter(_rad_torch.randn(CACHE_SLICES, _RAD_HEAD_DIM) * .01)
        self.query = _rad_nn.Parameter(_rad_torch.randn(len(_RAD_LABELS), _RAD_HEAD_DIM) * .02)
        self.attn = _rad_nn.MultiheadAttention(
            _RAD_HEAD_DIM, 8, dropout=.10, batch_first=True
        )
        self.fuse = _rad_nn.Sequential(
            _rad_nn.LayerNorm(_RAD_HEAD_DIM * 4),
            _rad_nn.Linear(_RAD_HEAD_DIM * 4, _RAD_HEAD_DIM),
            _rad_nn.GELU(),
            _rad_nn.Dropout(.15),
        )
        self.weight = _rad_nn.Parameter(
            _rad_torch.randn(len(_RAD_LABELS), _RAD_HEAD_DIM) * .02
        )
        self.bias = _rad_nn.Parameter(_rad_torch.zeros(len(_RAD_LABELS)))

    def forward(self, feature, mask):
        token = self.project(feature.float())
        token = token.view(len(token), N_SLOT, CACHE_SLICES, _RAD_HEAD_DIM)
        token = token + self.plane[None, :, None] + self.position[None, None]
        token = token.flatten(1, 2)
        key_padding = mask <= 0
        all_empty = key_padding.all(1)
        if all_empty.any():
            key_padding = key_padding.clone()
            key_padding[all_empty, 0] = False
        query = self.query.unsqueeze(0).expand(len(token), -1, -1)
        attended = query + self.attn(
            query, token, token, key_padding_mask=key_padding, need_weights=False
        )[0]
        denominator = mask.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)
        mean = (token * mask.unsqueeze(-1)).sum(1, keepdim=True) / denominator
        mean = mean.expand(-1, len(_RAD_LABELS), -1)
        fused = self.fuse(_rad_torch.cat(
            [attended, mean, _rad_torch.abs(attended - mean), attended * mean], dim=-1
        ))
        return (fused * self.weight.unsqueeze(0)).sum(-1) + self.bias


def _rad_load_public_heads(device, expected_sha):
    heads_path = _rad_find_file('v52_radimagenet_heads.pt', expected_sha)
    payload = _rad_torch.load(heads_path, map_location='cpu', weights_only=True)
    expected = {
        'version': 'v52-radimagenet-resnet50-official-1',
        'targets': _RAD_LABELS,
        'encoder_sha256': _RAD_ENCODER_SHA256,
        'encoder_source_commit': '0ce16f7375db4236e646829d1eca61cdb4282133',
        'img': 224,
        'slices_per_plane': 8,
        'feature': 'global_average_pool',
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f'public-v15 head contract drift for {key}')
    folds = payload.get('folds')
    if not isinstance(folds, list) or len(folds) != 5:
        raise RuntimeError('public-v15 bundle requires exactly five heads')
    if sorted(int(record.get('fold', -1)) for record in folds) != list(range(5)):
        raise RuntimeError('public-v15 fold identity drift')
    heads = []
    for record in folds:
        head = _RadHead().to(device).eval()
        head.load_state_dict(record['state_dict'], strict=True)
        heads.append(head)
    return heads, str(heads_path)


def _rad_load_e13_heads(device):
    # V48 used an unqualified filename shared by E11 and E13. Resolve the
    # intended E13 bundle by content and validate its complete pixel contract.
    heads_path = _rad_find_file('v52_e11_heads.pt', _RAD_E13_HEADS_SHA256)
    payload = _rad_torch.load(heads_path, map_location='cpu', weights_only=False)
    expected = {
        'version': 'e11-radimagenet-resnet50-diverse-1',
        'targets': _RAD_LABELS,
        'encoder_sha256': _RAD_ENCODER_SHA256,
        'slots': [list(slot) for slot in _RAD_E13_SLOTS],
        'crop_mm': _RAD_E13_CROP_MM,
        'img': _RAD_E13_IMG,
        'slices_per_plane': _RAD_E13_CACHE_SLICES,
        'feature': 'global_average_pool',
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f'E13 head contract drift for {key}')
    folds = payload.get('folds')
    if not isinstance(folds, list) or len(folds) != 5:
        raise RuntimeError('E13 bundle requires exactly five heads')
    if sorted(int(record.get('fold', -1)) for record in folds) != list(range(5)):
        raise RuntimeError('E13 fold identity drift')
    heads = []
    for record in folds:
        head = _RadHead().to(device).eval()
        head.load_state_dict(record['state_dict'], strict=True)
        heads.append(head)
    return heads, str(heads_path)


def _rad_load_models(device):
    encoder_path = _rad_find_file(
        'ResNet50.pt', _RAD_ENCODER_SHA256, explicit_env='RSNA_RAD_WEIGHT_PATH'
    )
    encoder = _RadEncoder()
    encoder.load_state_dict(
        _rad_torch.load(encoder_path, map_location='cpu', weights_only=True), strict=True
    )
    if sum(parameter.numel() for parameter in encoder.parameters()) != 23_508_032:
        raise RuntimeError('V36 RadImageNet encoder parameter-count drift')
    encoder.eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    if device.type == 'cuda' and _rad_torch.cuda.device_count() > 1:
        encoder = _rad_nn.DataParallel(
            encoder, device_ids=list(range(_rad_torch.cuda.device_count()))
        )

    alt_heads, alt_path = _rad_load_public_heads(device, _RAD_HEADS_SHA256)
    reference_heads, reference_path = _rad_load_public_heads(
        device, _RAD_REFERENCE_HEADS_SHA256
    )
    if alt_path == reference_path:
        raise RuntimeError('twin E10 branches resolved to the same artifact')
    return encoder, alt_heads, reference_heads, str(encoder_path), alt_path, reference_path


@_rad_torch.inference_mode()
def _rad_encode(encoder, pixels, slot_mask, device):
    n, slots, slices, height, width = pixels.shape
    features = _rad_np.zeros(
        (n, slots * slices, _RAD_TOKEN_DIM), _rad_np.float16
    )
    token_mask = _rad_np.repeat(slot_mask[:, :, None], slices, axis=2).reshape(n, -1)
    valid = _rad_np.flatnonzero(token_mask.reshape(-1) > 0)
    flat = pixels.reshape(-1, height, width)
    batch = 192 if device.type == 'cuda' and _rad_torch.cuda.device_count() > 1 else (
        96 if device.type == 'cuda' else 8
    )
    for start in range(0, len(valid), batch):
        indices = valid[start:start + batch]
        image = _rad_torch.from_numpy(flat[indices]).to(device).float().div_(127.5).sub_(1.0)
        image = image.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
        amp = (_rad_torch.autocast('cuda')
               if device.type == 'cuda' else _rad_contextlib.nullcontext())
        with amp:
            feature = encoder(image)
        values = feature.float().cpu().numpy()
        if not _rad_np.isfinite(values).all():
            raise RuntimeError('V36 non-finite RadImageNet feature')
        features.reshape(-1, _RAD_TOKEN_DIM)[indices] = values.astype(_rad_np.float16)
    return features, token_mask.astype(_rad_np.float32)


@_rad_torch.inference_mode()
def _rad_predict_head(head, features, masks, device, batch=64):
    predictions = []
    for start in range(0, len(features), batch):
        image = _rad_torch.from_numpy(features[start:start + batch]).to(device)
        mask = _rad_torch.from_numpy(masks[start:start + batch]).to(device)
        amp = (_rad_torch.autocast('cuda')
               if device.type == 'cuda' else _rad_contextlib.nullcontext())
        with amp:
            predictions.append(_rad_torch.sigmoid(head(image, mask)).float().cpu())
    return _rad_torch.cat(predictions).numpy()


def _rad_rank_columns(values):
    return _rad_pd.DataFrame(
        _rad_np.asarray(values, dtype=_rad_np.float64)
    ).rank(method='average', pct=True).to_numpy(_rad_np.float64)


def _rad_validate(frame, expected_ids):
    if frame.columns.tolist() != ['StudyInstanceUID', *_RAD_LABELS]:
        raise RuntimeError('V36 submission schema drift')
    ids = frame['StudyInstanceUID'].astype(str).tolist()
    if ids != list(map(str, expected_ids)) or len(ids) != len(set(ids)):
        raise RuntimeError('V36 submission study identity/order drift')
    values = frame[_RAD_LABELS].to_numpy(_rad_np.float64)
    if not _rad_np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise RuntimeError('V36 invalid submission values')


def _rad_main():
    started = _rad_time.time()
    work = _RadPath(_rad_os.environ.get('RSNA_RAD_OUTPUT_DIR', '/kaggle/working'))
    primary = work / 'submission.csv'
    if not primary.is_file():
        raise FileNotFoundError('V37 requires the completed DINO parent submission.csv')
    test = _rad_pd.read_csv(ROOT / 'test.csv', dtype={'StudyInstanceUID': str})
    expected_ids = test['StudyInstanceUID'].astype(str).tolist()
    baseline = _rad_pd.read_csv(primary, dtype={'StudyInstanceUID': str})
    _rad_validate(baseline, expected_ids)

    device = _rad_torch.device('cuda:0' if _rad_torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('V37 RadImageNet inference requires CUDA')
    (encoder, public_heads, reference_heads, encoder_path,
     public_heads_path, reference_heads_path) = _rad_load_models(device)

    # Family 1: public v15/E10 legacy pixels.  Keep this path bit-for-bit as in
    # V36, including rank(mean(fold probability)).
    test_series = _rad_pd.read_csv(
        ROOT / 'test_series.csv',
        dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str},
    )
    plane = dict(zip(test_series.SeriesInstanceUID, test_series.Anatomical_Plane))
    headers = annotate(walk('test_series'))
    studies, pixels, slot_mask = build_cache(
        pick_slots(headers, plane), plane, lat_of(headers, 'test-e10 '), 'test-e10'
    )
    by_uid = {str(uid): index for index, uid in enumerate(studies)}
    missing = [uid for uid in expected_ids if uid not in by_uid]
    if missing:
        raise RuntimeError(f'{len(missing)} test studies absent from public-v15 cache')
    order = _rad_np.asarray([by_uid[uid] for uid in expected_ids], dtype=_rad_np.int64)
    pixels, slot_mask = pixels[order], slot_mask[order]
    token_count = int(
        _rad_np.repeat(slot_mask[:, :, None], CACHE_SLICES, axis=2).sum()
    )
    if token_count < int(0.85 * len(test) * N_SLOT * CACHE_SLICES):
        raise RuntimeError(f'insufficient acquired public-v15 test slices: {token_count}')

    features, token_mask = _rad_encode(encoder, pixels, slot_mask, device)
    del pixels, slot_mask, headers
    _rad_gc.collect()
    public_fold_predictions = [
        _rad_predict_head(head, features, token_mask, device)
        for head in public_heads
    ]
    reference_fold_predictions = [
        _rad_predict_head(head, features, token_mask, device)
        for head in reference_heads
    ]
    if len(public_fold_predictions) != 5 or len(reference_fold_predictions) != 5:
        raise RuntimeError('twin E10 inference did not use all ten heads')

    # Preserve each public recipe's rank(mean(fold probability)) estimator.
    public_probability = _rad_np.mean(_rad_np.stack(public_fold_predictions), axis=0)
    reference_probability = _rad_np.mean(
        _rad_np.stack(reference_fold_predictions), axis=0
    )
    public_rank = _rad_rank_columns(public_probability)
    reference_rank = _rad_rank_columns(reference_probability)
    del (public_heads, reference_heads, public_fold_predictions,
         reference_fold_predictions, public_probability, reference_probability,
         features, token_mask)
    _rad_gc.collect()
    _rad_torch.cuda.empty_cache()
    _rad_log(
        f'twin public-v15 families complete ({public_heads_path}; {reference_heads_path})'
    )

    # One new member from V48: three fat-sensitive planes plus a sagittal
    # structural anchor, all at a 130 mm crop. Average ranks inside the Rad block
    # and re-rank the result exactly as V48 does before the unchanged E10 vote.
    globals().update(
        SLOTS=list(_RAD_E13_SLOTS),
        N_SLOT=len(_RAD_E13_SLOTS),
        CACHE_SLICES=int(_RAD_E13_CACHE_SLICES),
        IMG=int(_RAD_E13_IMG),
        CACHE_IMG=int(_RAD_E13_IMG),
        CROP_MM=float(_RAD_E13_CROP_MM),
        RULES=dict(RULES_LEGACY),
    )
    e13_heads, e13_path = _rad_load_e13_heads(device)
    headers = annotate(walk('test_series'))
    studies, pixels, slot_mask = build_cache(
        pick_slots(headers, plane), plane, lat_of(headers, 'test-e13 '), 'test-e13'
    )
    by_uid = {str(uid): index for index, uid in enumerate(studies)}
    missing = [uid for uid in expected_ids if uid not in by_uid]
    if missing:
        raise RuntimeError(f'{len(missing)} test studies absent from E13 cache')
    order = _rad_np.asarray([by_uid[uid] for uid in expected_ids], dtype=_rad_np.int64)
    pixels, slot_mask = pixels[order], slot_mask[order]
    e13_token_count = int(
        _rad_np.repeat(slot_mask[:, :, None], CACHE_SLICES, axis=2).sum()
    )
    if e13_token_count < int(0.85 * len(test) * N_SLOT * CACHE_SLICES):
        raise RuntimeError(f'insufficient acquired E13 test slices: {e13_token_count}')
    e13_features, e13_token_mask = _rad_encode(
        encoder, pixels, slot_mask, device
    )
    del pixels, slot_mask, headers
    _rad_gc.collect()
    e13_predictions = [
        _rad_predict_head(head, e13_features, e13_token_mask, device)
        for head in e13_heads
    ]
    if len(e13_predictions) != 5:
        raise RuntimeError('E13 inference did not use all five heads')
    e13_probability = _rad_np.mean(_rad_np.stack(e13_predictions), axis=0)
    if (
        e13_probability.shape != (len(test), len(_RAD_LABELS))
        or not _rad_np.isfinite(e13_probability).all()
    ):
        raise RuntimeError(f'invalid E13 prediction shape/value: {e13_probability.shape}')
    e13_rank = _rad_rank_columns(e13_probability)
    public_rank = _rad_rank_columns(
        (1.0 - _RAD_E13_MEMBER_WEIGHT) * public_rank
        + _RAD_E13_MEMBER_WEIGHT * e13_rank
    )
    reference_rank = _rad_rank_columns(
        (1.0 - _RAD_E13_MEMBER_WEIGHT) * reference_rank
        + _RAD_E13_MEMBER_WEIGHT * e13_rank
    )
    # V48 resolves this same bundle again after switching pixel layouts.
    del (e13_predictions, e13_probability, e13_rank,
         e13_features, e13_token_mask)
    _rad_gc.collect()
    _rad_torch.cuda.empty_cache()
    _rad_log(
        f'E13 FS-crop member complete at Rad-block weight '
        f'{_RAD_E13_MEMBER_WEIGHT:.2f} ({e13_path})'
    )

    # E10 keeps its audited 0.50 parent/Rad vote. The two excluded findings
    # remain the raw parent values, matching the audited deployment.
    baseline_rank = _rad_rank_columns(baseline[_RAD_LABELS].to_numpy())

    def _rad_e10_branch(head_rank):
        branch = baseline.copy()
        for index, target in enumerate(_RAD_LABELS):
            if target not in _RAD_EXCLUDE:
                branch[target] = (
                    (1.0 - _RAD_ALPHA) * baseline_rank[:, index]
                    + _RAD_ALPHA * head_rank[:, index]
                )
        return branch

    candidate_alt = _rad_e10_branch(public_rank)
    candidate_reference = _rad_e10_branch(reference_rank)
    for branch in (candidate_alt, candidate_reference):
        for target in _RAD_EXCLUDE:
            if not _rad_np.array_equal(
                branch[target].to_numpy(), baseline[target].to_numpy()
            ):
                raise RuntimeError(f'E10 failed to preserve raw parent values for {target}')
        _rad_validate(branch, expected_ids)
    # Diagnostic E10 output only; the final equal rank mean is formed after E11
    # and the legacy-DINO tie-break have completed independently in each branch.
    candidate = baseline.copy()
    alt_e10_rank = _rad_rank_columns(candidate_alt[_RAD_LABELS].to_numpy())
    reference_e10_rank = _rad_rank_columns(
        candidate_reference[_RAD_LABELS].to_numpy()
    )
    candidate[_RAD_LABELS] = (
        _RAD_TWIN_ALT_WEIGHT * alt_e10_rank
        + (1.0 - _RAD_TWIN_ALT_WEIGHT) * reference_e10_rank
    )
    _rad_validate(candidate, expected_ids)
    e10_path = work / 'submission_e10_v2.csv'
    candidate.to_csv(e10_path, index=False)
    _rad_log(
        f'twin E10 branches complete at alpha={_RAD_ALPHA:.2f}; '
        f'preserved raw={list(_RAD_EXCLUDE)}'
    )

    # V48's successful run selected the E13 bundle a second time after
    # installing the older E11 slot order. Express that observed behavior
    # directly, without relying on duplicate-filename directory order.
    globals().update(
        SLOTS=list(_RAD_E11_SLOTS),
        N_SLOT=len(_RAD_E11_SLOTS),
        CACHE_SLICES=int(_RAD_E11_CACHE_SLICES),
        IMG=int(_RAD_E11_IMG),
        CACHE_IMG=int(_RAD_E11_IMG),
        CROP_MM=float(_RAD_E11_CROP_MM),
        RULES=dict(RULES_LEGACY),
    )
    headers = annotate(walk('test_series'))
    studies, pixels, slot_mask = build_cache(
        pick_slots(headers, plane), plane,
        lat_of(headers, 'test-v48-pass2 '), 'test-v48-pass2'
    )
    by_uid = {str(uid): index for index, uid in enumerate(studies)}
    missing = [uid for uid in expected_ids if uid not in by_uid]
    if missing:
        raise RuntimeError(f'{len(missing)} test studies absent from V48 pass-2 cache')
    order = _rad_np.asarray([by_uid[uid] for uid in expected_ids], dtype=_rad_np.int64)
    pixels, slot_mask = pixels[order], slot_mask[order]
    v48_pass2_token_count = int(
        _rad_np.repeat(slot_mask[:, :, None], CACHE_SLICES, axis=2).sum()
    )
    if v48_pass2_token_count < int(0.55 * len(test) * N_SLOT * CACHE_SLICES):
        raise RuntimeError(
            f'insufficient acquired V48 pass-2 slices: {v48_pass2_token_count}'
        )
    v48_features, v48_token_mask = _rad_encode(
        encoder, pixels, slot_mask, device
    )
    del pixels, slot_mask, headers
    _rad_gc.collect()
    v48_pass2_predictions = [
        _rad_predict_head(head, v48_features, v48_token_mask, device)
        for head in e13_heads
    ]
    if len(v48_pass2_predictions) != 5:
        raise RuntimeError('V48 second pass did not use all five E13 heads')
    v48_pass2_probability = _rad_np.mean(
        _rad_np.stack(v48_pass2_predictions), axis=0
    )
    if (
        v48_pass2_probability.shape != (len(test), len(_RAD_LABELS))
        or not _rad_np.isfinite(v48_pass2_probability).all()
    ):
        raise RuntimeError(
            f'invalid V48 pass-2 prediction: {v48_pass2_probability.shape}'
        )
    v48_pass2_rank = _rad_rank_columns(v48_pass2_probability)

    reference_branch = candidate_reference.copy()
    reference_branch[_RAD_LABELS] = _rad_rank_columns(
        (1.0 - _RAD_V48_SECOND_ALPHA)
        * _rad_rank_columns(candidate_reference[_RAD_LABELS].to_numpy())
        + _RAD_V48_SECOND_ALPHA * v48_pass2_rank
    )
    _rad_validate(reference_branch, expected_ids)
    _rad_log(
        f'V48 second E13 pass complete at alpha '
        f'{_RAD_V48_SECOND_ALPHA:.2f} on the E11 slot layout'
    )

    # V48 deploys the pinned reference branch directly after the second pass.
    # The alternative-head twin and legacy-DINO tie-break are not part of .917.
    _RAD_CAL = _rad_json.loads(
        _RadPath('/kaggle/working/rad_calibrator.json').read_text())
    _RAD_CAL_GATE = set(_RAD_CAL['gate'])
    _RAD_CAL_W = 0.40

    def _rad_cal_protocol(uids):
        frame = _rad_pd.read_csv(
            ROOT / 'test_series.csv',
            dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str},
        )
        frame['StudyInstanceUID'] = frame['StudyInstanceUID'].astype(str)
        index = _rad_pd.Index([str(u) for u in uids], name='StudyInstanceUID')
        table = _rad_pd.DataFrame(index=index)
        table['n_series'] = frame.groupby(
            'StudyInstanceUID').size().reindex(index).fillna(0)
        for plane in ('Sagittal', 'Coronal', 'Axial'):
            part = frame[frame['Anatomical_Plane'].astype(str) == plane]
            table[f'n_{plane[:3]}'] = part.groupby(
                'StudyInstanceUID').size().reindex(index).fillna(0)
        for flag in ('Fat_Suppression', 'Fluid_Sensitive'):
            marked = frame[_rad_pd.to_numeric(
                frame[flag], errors='coerce').fillna(0) > 0]
            table[flag[:3]] = marked.groupby(
                'StudyInstanceUID').size().reindex(index).fillna(0)
            for plane in ('Sagittal', 'Coronal', 'Axial'):
                part = marked[marked['Anatomical_Plane'].astype(str) == plane]
                table[f'{flag[:3]}_{plane[:3]}'] = part.groupby(
                    'StudyInstanceUID').size().reindex(index).fillna(0)
        if list(table.columns) != list(_RAD_CAL['protocol_columns']):
            raise RuntimeError('calibration protocol layout mismatch')
        return table.to_numpy(_rad_np.float64)

    def _rad_calibrate(branch):
        base = baseline_rank
        public = reference_rank
        pass2 = v48_pass2_rank
        mean = (base + public + pass2) / 3.0
        blocks = [base, public, pass2, public - base, pass2 - base, mean]
        for _grp in _RAD_CAL['groups']:
            cols = [_RAD_LABELS.index(t) for t in _grp]
            blocks.append(mean[:, cols].mean(axis=1, keepdims=True))
        blocks.append(_rad_cal_protocol(expected_ids))
        x = _rad_np.concatenate(blocks, axis=1)
        centre = _rad_np.asarray(_RAD_CAL['mean'], _rad_np.float64)
        spread = _rad_np.asarray(_RAD_CAL['scale'], _rad_np.float64)
        coef = _rad_np.asarray(_RAD_CAL['coef'], _rad_np.float64)
        bias = _rad_np.asarray(_RAD_CAL['intercept'], _rad_np.float64)
        if x.shape[1] != coef.shape[1]:
            raise RuntimeError(
                f'calibration expects {coef.shape[1]} columns, built {x.shape[1]}')
        adjusted = _rad_rank_columns(((x - centre) / spread) @ coef.T + bias)
        out = branch.copy()
        values = out[_RAD_LABELS].to_numpy(_rad_np.float64).copy()
        for index, target in enumerate(_RAD_LABELS):
            if target in _RAD_CAL_GATE:
                values[:, index] = (
                    (1.0 - _RAD_CAL_W) * values[:, index]
                    + _RAD_CAL_W * adjusted[:, index]
                )
        out[_RAD_LABELS] = _rad_rank_columns(values)
        _rad_validate(out, expected_ids)
        return out

    final = _rad_calibrate(reference_branch)
    globals()['V18_CALIBRATOR_APPLIED'] = True
    globals()['V18_CAL_GATE'] = tuple(sorted(_RAD_CAL_GATE))
    _rad_validate(final, expected_ids)
    temporary = primary.with_suffix('.csv.tmp')
    final.to_csv(temporary, index=False)
    _rad_os.replace(temporary, primary)

    receipt = {
        'recipe': 'V48 deployed reference branch: correct E13@0.50-inside-Rad -> E10@0.50 -> same E13 on E11 layout@0.15',
        'e13_member_weight_inside_rad': _RAD_E13_MEMBER_WEIGHT,
        'e10_alpha': _RAD_ALPHA,
        'e10_preserved_targets': list(_RAD_EXCLUDE),
        'v48_second_alpha': _RAD_V48_SECOND_ALPHA,
        'reference_heads_sha256': _RAD_REFERENCE_HEADS_SHA256,
        'e13_heads_sha256': _RAD_E13_HEADS_SHA256,
        'v48_second_heads_sha256': _RAD_E13_HEADS_SHA256,
        'v48_second_slots': [list(slot) for slot in _RAD_E11_SLOTS],
        'encoder_sha256': _RAD_ENCODER_SHA256,
        'test_studies': len(expected_ids),
        'e10_tokens': token_count,
        'v48_second_tokens': v48_pass2_token_count,
        'e13_tokens': e13_token_count,
        'submission_sha256': _rad_sha256(primary),
    }
    (work / 'v50_v2_repro_receipt.json').write_text(
        _rad_json.dumps(receipt, indent=2, sort_keys=True) + '\n'
    )
    del (encoder, e13_heads, v48_pass2_predictions,
         v48_pass2_probability, v48_pass2_rank, v48_features, v48_token_mask)
    _rad_gc.collect()
    _rad_torch.cuda.empty_cache()
    _rad_log(
        f'V48 reference branch complete; reference-v15={reference_heads_path}; '
        f'e13-two-pass={e13_path}; second_alpha='
        f'{_RAD_V48_SECOND_ALPHA:.2f}; encoder={encoder_path}; '
        f'elapsed={(_rad_time.time()-started)/60:.1f}m'
    )


_rad_main()

#### CELL 30
_rep_snapshot('rad')

#### CELL 32
%%writefile /kaggle/working/raptor_arm.py
# ===========================================================================
# Raptor CoAtNet arm - INHERITED CODE, NOT WRITTEN BY THIS NOTEBOOK'S AUTHOR.
#
# Origin : Mattia Angeli, "Bend the Knee to the Dinosaurs"
#          https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-the-dinosaurs
# Weights: dreaddevelopment/raptor-knee-{maxspan,native384,native384dense}
#          https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-maxspan
# License: Apache-2.0 (Kaggle default for public notebooks).
#
# In the parent notebook this module was carried as a single escaped string
# literal and run with exec(compile(...)). It is reproduced here verbatim as
# ordinary readable source so that it can be read, diffed and audited. The
# code is unchanged; only its presentation is.
# ===========================================================================
import os, glob, time, gc, hashlib
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
IMG = 336
CROP_MM = 140.0
SPAN_LO, SPAN_HI = 0.02, 0.98
SLOTS = [("Sagittal", 1, 18), ("Sagittal", 0, 14),
         ("Coronal", 1, 12), ("Coronal", 0, 8), ("Axial", -1, 12)]
MAXS = sum(slot[2] for slot in SLOTS)
K_EVAL = 62
NORM = "imagenet"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
       "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
       "Contusion", "Fracture"]
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
_SLOTS64 = [("Sagittal", 1, 18), ("Sagittal", 0, 14),
            ("Coronal", 1, 12), ("Coronal", 0, 8), ("Axial", -1, 12)]
_SLOTS44 = [("Sagittal", 1, 12), ("Sagittal", 0, 10),
            ("Coronal", 1, 8), ("Coronal", 0, 6), ("Axial", -1, 8)]
ARMS = [
    {"name": "maxspan-v5", "file": "raptor_ft_coatnet_v5_full_swa.pt",
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,
     "img": 336, "slots": _SLOTS64, "span": (0.02, 0.98), "k_eval": 62,
     "reverse": False, "w": 0.55},
    {"name": "native384dense-v10", "file": "raptor_ft_coatnet_v10_full.pt",
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,
     "img": 384, "slots": _SLOTS64, "span": (0.02, 0.98), "k_eval": 62,
     "reverse": False, "w": 0.1},
    {"name": "maxspan-v5-reverse", "file": "raptor_ft_coatnet_v5_full_swa.pt",
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,
     "img": 336, "slots": _SLOTS64, "span": (0.02, 0.98), "k_eval": 62,
     "reverse": True, "w": 0.15},
    {"name": "native384-v8", "file": "raptor_ft_coatnet_v8_full_swa.pt",
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,
     "img": 384, "slots": _SLOTS44, "span": (0.06, 0.94), "k_eval": 42,
     "reverse": False, "w": 0.2},
]

def build_backbone(arch, pretrained=False):
    hybrid = arch.startswith(('maxvit', 'maxxvit', 'coatnet', 'coat_', 'convnext'))
    is_vit = not hybrid and any((k in arch for k in ('vit', 'deit', 'dinov2', 'eva', 'beit')))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    if is_vit:
        kw.update(global_pool='token', dynamic_img_size=True)
    else:
        kw.update(global_pool='avg')
    return timm.create_model(arch, **kw)

class RaptorClassifier(nn.Module):

    def __init__(self, backbone, F_dim=768, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def encode(self, x):
        B, K = x.shape[:2]
        f = self.backbone(x.flatten(0, 1))
        return f.view(B, K, -1)

    def head(self, feats):
        h = self.norm(feats)
        a = self.att(h)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum('bkn,bkf->bnf', a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

    def forward(self, x):
        return self.head(self.encode(x))

def load_model(pt_path, arch_default, res_default, device, ngpu=1):
    ck = torch.load(pt_path, map_location='cpu', weights_only=False)
    arch = ck.get('arch', arch_default)
    ck_res = int(ck.get('res', res_default))
    bb = build_backbone(arch, pretrained=False)
    model = RaptorClassifier(bb, F_dim=bb.num_features)
    model.load_state_dict(ck['model'], strict=True)
    model.eval().to(device)
    del ck
    gc.collect()
    return (model, ck_res)

def load_refit_head(pt_path, feature_dim, device):
    ck = torch.load(pt_path, map_location='cpu', weights_only=False)
    state = ck.get('model', ck)
    head = RaptorClassifier(nn.Identity(), F_dim=int(feature_dim))
    head_state = {name: tensor for name, tensor in state.items()
                  if not name.startswith('backbone.')}
    head.load_state_dict(head_state, strict=True)
    head.eval().to(device)
    del ck, state, head_state
    gc.collect()
    return head

def _eval_centers(mask, D, k):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = (int(valid.min()), int(valid.max()))
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, D - 2))]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [cs[i] for i in idx]

def eval_windows(vol, mask, k, res, norm=NORM):
    D = vol.shape[0]
    cs = _eval_centers(mask, D, k)
    wins = np.empty((len(cs), 3, res, res), np.float32)
    for j, c in enumerate(cs):
        c = max(1, min(c, D - 2))
        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        t = torch.from_numpy(tri)
        if t.shape[-1] != res:
            t = F.interpolate(t[None], size=(res, res), mode='bilinear', align_corners=False)[0]
        wins[j] = t.numpy()
    x = torch.from_numpy(wins)
    if norm == 'imagenet':
        x = (x - _MEAN) / _STD
    return x

@torch.no_grad()
def infer_probs(model, xwins, device):
    x = xwins.unsqueeze(0).to(device)
    use_cuda = device != 'cpu' and str(device).startswith('cuda')
    if use_cuda:
        try:
            with torch.autocast('cuda', dtype=torch.float16):
                o = torch.sigmoid(model(x).float())
            return o[0].cpu().numpy()
        except RuntimeError:
            torch.cuda.empty_cache()
            o = torch.sigmoid(model(x).float())
            return o[0].cpu().numpy()
    o = torch.sigmoid(model(x).float())
    return o[0].cpu().numpy()

@torch.no_grad()
def infer_probs_two_heads(model, refit_head, xwins, device):
    x = xwins.unsqueeze(0).to(device)
    use_cuda = device != 'cpu' and str(device).startswith('cuda')
    def forward_heads():
        features = model.encode(x)
        original = torch.sigmoid(model.head(features).float())
        refitted = torch.sigmoid(refit_head.head(features).float())
        return (original[0].cpu().numpy(), refitted[0].cpu().numpy())
    if use_cuda:
        try:
            with torch.autocast('cuda', dtype=torch.float16):
                return forward_heads()
        except RuntimeError:
            torch.cuda.empty_cache()
            return forward_heads()
    return forward_heads()

def rankpct(x):
    order = x.argsort(0).argsort(0).astype(np.float64)
    return order / max(1, x.shape[0] - 1)

def _make_reader():
    import pydicom, cv2
    from pydicom.pixel_data_handlers.util import apply_modality_lut

    def order_and_meta(sdir):
        fs = glob.glob(sdir + '/*.dcm')
        recs = []
        ps_list = []
        for f in fs:
            try:
                h = pydicom.dcmread(f, stop_before_pixels=True)
                iop = getattr(h, 'ImageOrientationPatient', None)
                ipp = getattr(h, 'ImagePositionPatient', None)
                if iop is not None and ipp is not None and (len(iop) == 6):
                    r = np.array(iop[:3], float)
                    c = np.array(iop[3:], float)
                    n = np.cross(r, c)
                    pos = float(np.dot(np.array(ipp, float), n))
                else:
                    pos = float(getattr(h, 'InstanceNumber', 0) or 0)
                ps = getattr(h, 'PixelSpacing', None)
                ps = float(ps[0]) if ps is not None else 0.5
                ps_list.append(ps)
                recs.append((pos, f, ps))
            except Exception:
                recs.append((0.0, f, 0.5))
        recs.sort(key=lambda x: x[0])
        med_ps = float(np.median(ps_list)) if ps_list else 0.5
        return ([(f, ps) for _, f, ps in recs], med_ps)

    def read_px(f):
        d = pydicom.dcmread(f)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
            a = a.max() - a
        return a

    def mm_crop_resize(a, ps):
        h, w = a.shape
        cpx = int(round(CROP_MM / max(ps, 0.001)))
        cpx = min(cpx, min(h, w))
        y0 = (h - cpx) // 2
        x0 = (w - cpx) // 2
        a = a[y0:y0 + cpx, x0:x0 + cpx]
        return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)
    return (order_and_meta, read_px, mm_crop_resize)

def _pick_series_for_slot(rows, plane, fluid, used):
    cands = [r for r in rows if r['Anatomical_Plane'] == plane and r['SeriesInstanceUID'] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get('Fluid_Sensitive', 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None

def build_study(sid, ser_records, tsdir, reader):
    order_and_meta, read_px, mm_crop_resize = reader
    rows = ser_records.get(sid, [])
    vol = np.zeros((MAXS, IMG, IMG), np.uint8)
    idx = 0
    used = set()
    for plane, fluid, k in SLOTS:
        r = _pick_series_for_slot(rows, plane, fluid, used)
        if r is None:
            idx += k
            continue
        used.add(r['SeriesInstanceUID'])
        files, med_ps = order_and_meta(f"{tsdir}/{sid}/{r['SeriesInstanceUID']}")
        if not files:
            idx += k
            continue
        n = len(files)
        lo, hi = (int(n * SPAN_LO), int(n * SPAN_HI) - 1)
        hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        arrs = []
        pss = []
        for p in picks:
            fp, ps = files[min(p, n - 1)]
            try:
                arrs.append(read_px(fp))
                pss.append(ps)
            except Exception:
                arrs.append(None)
                pss.append(med_ps)
        valid = [a for a in arrs if a is not None]
        if valid:
            allpx = np.concatenate([a.ravel() for a in valid])
            loq, hiq = np.percentile(allpx, [2.0, 98.0])
        else:
            loq, hiq = (0.0, 1.0)
        for a, ps in zip(arrs, pss):
            if idx >= MAXS:
                break
            if a is None:
                idx += 1
                continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-06), 0, 1)
            aw = mm_crop_resize(aw, ps if ps > 0 else med_ps)
            vol[idx] = (aw * 255).astype(np.uint8)
            idx += 1
        if idx >= MAXS:
            break
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    return (vol, mask)

def find_test_root():
    cands = ['/kaggle/input/competitions/rsna-knee-abnormality-detection', '/kaggle/input/rsna-knee-abnormality-detection']
    for b in cands:
        if os.path.exists(b + '/test.csv'):
            return b
    for d, _, f in os.walk('/kaggle/input'):
        if 'test.csv' in f and (os.path.isdir(d + '/test_series') or os.path.isdir(d + '/test_images')):
            return d
    for d, _, f in os.walk('/kaggle/input'):
        if 'test.csv' in f:
            return d
    raise RuntimeError('no test root under /kaggle/input')

def find_weight_file(fname):
    direct = [f'/kaggle/input/raptor-knee-maxspan/{fname}', f'/kaggle/input/raptor-knee-native384dense/{fname}', f'/kaggle/input/raptor-knee-native384/{fname}', f'/kaggle/input/raptor-knee-arms/{fname}', f'/kaggle/input/raptor-knee-arms/1/{fname}', f'/kaggle/input/raptor-cnn336/{fname}']
    for p in direct:
        if os.path.exists(p):
            return p
    for d in sorted(glob.glob('/kaggle/input/*/')):
        if 'competition' in d.lower():
            continue
        hits = glob.glob(os.path.join(d, '**', fname), recursive=True)
        if hits:
            return hits[0]
    raise RuntimeError(f'{fname} not found under /kaggle/input')

def find_optional_verified_weight(fname, expected_sha256, root='/kaggle/input'):
    hits = []
    for directory in sorted(glob.glob(os.path.join(root, '*/'))):
        if 'competition' in directory.lower():
            continue
        hits.extend(glob.glob(os.path.join(directory, '**', fname), recursive=True))
    for path in sorted(set(hits)):
        digest = hashlib.sha256()
        with open(path, 'rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() == expected_sha256:
            return path
        print(f'[head-refit] ignored hash-mismatched optional checkpoint: {path}', flush=True)
    return None

def main():
    import pandas as pd
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {dev} | gpus {torch.cuda.device_count()} | torch {torch.__version__}", flush=True)
    root = find_test_root()
    tsdir = root + "/test_series"
    if not os.path.isdir(tsdir):
        tsdir = root + "/test_images"
    print("test root:", root, "| series dir:", tsdir, flush=True)
    test = pd.read_csv(root + "/test.csv")
    test["StudyInstanceUID"] = test["StudyInstanceUID"].astype(str)
    test_ids = test["StudyInstanceUID"].tolist()
    tser = pd.read_csv(root + "/test_series.csv")
    tser["StudyInstanceUID"] = tser["StudyInstanceUID"].astype(str)
    tser["SeriesInstanceUID"] = tser["SeriesInstanceUID"].astype(str)
    series = {key: frame.to_dict("records") for key, frame in tser.groupby("StudyInstanceUID")}
    print(f"test studies {len(test_ids)} | test series {len(tser)}", flush=True)
    sub_cols = ["StudyInstanceUID"] + LAB
    sample = os.path.join(root, "sample_submission.csv")
    if os.path.exists(sample):
        sub_cols = list(pd.read_csv(sample, nrows=1).columns)
    reader = _make_reader()
    n_study, n_arm = len(test_ids), len(ARMS)
    arm_probs = [np.full((n_study, len(LAB)), 0.5, np.float32) for _ in range(n_arm)]
    for arm_index, arm in enumerate(ARMS):
        globals()["IMG"] = int(arm["img"])
        globals()["SLOTS"] = list(arm["slots"])
        globals()["MAXS"] = sum(slot[2] for slot in SLOTS)
        globals()["SPAN_LO"], globals()["SPAN_HI"] = map(float, arm["span"])
        globals()["K_EVAL"] = int(arm["k_eval"])
        weight_path = find_weight_file(arm["file"])
        model, resolution = load_model(weight_path, arm["arch"], arm["res"], dev)
        print(f"[arm {arm_index}] {arm['name']} | img {IMG} | slices {MAXS} | "
              f"span {SPAN_LO:.2f}-{SPAN_HI:.2f} | windows {K_EVAL} | "
              f"res {resolution} | {time.time() - t0:.0f}s", flush=True)
        for study_index, study_uid in enumerate(test_ids):
            try:
                volume, mask = build_study(study_uid, series, tsdir, reader)
                windows = eval_windows(volume, mask, k=K_EVAL, res=resolution, norm=NORM)
                if bool(arm.get("reverse", False)):
                    windows = windows.flip(1).contiguous()
                arm_probs[arm_index][study_index] = infer_probs(model, windows, dev)
                del volume, mask, windows
            except Exception as error:
                print(f"  [arm {arm_index}] study {study_index} {study_uid[:16]} FALLBACK "
                      f"({type(error).__name__}: {error})", flush=True)
            if (study_index + 1) % 100 == 0 or study_index + 1 == n_study:
                print(f"  [arm {arm_index}] {study_index + 1}/{n_study} | "
                      f"{time.time() - t0:.0f}s", flush=True)
        del model
        gc.collect()
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[arm {arm_index}] done + freed | {time.time() - t0:.0f}s", flush=True)
    weights = np.array([float(arm.get("w", 1.0)) for arm in ARMS], dtype=np.float64)
    weights /= weights.sum()
    print(f"[blend] global probability mean w="
          f"{dict(zip([arm['name'] for arm in ARMS], weights.round(4)))}", flush=True)
    probability_blend = np.tensordot(
        weights, np.stack([np.clip(values, 0, 1) for values in arm_probs]), axes=(0, 0))
    ranks = rankpct(probability_blend)
    if not np.isfinite(ranks).all():
        ranks[~np.isfinite(ranks)] = 0.5
    submission = pd.DataFrame(ranks.astype(np.float32), columns=LAB)
    submission.insert(0, "StudyInstanceUID", test_ids)
    submission = submission[sub_cols]
    assert submission["StudyInstanceUID"].tolist() == test_ids
    assert np.isfinite(submission[LAB].values).all()
    out = "/kaggle/working/_raptor.csv"
    submission.to_csv(out, index=False)
    print("wrote", out, "|", len(submission), "rows x", len(submission.columns), "cols", flush=True)
    print(submission.head().to_string(index=False), flush=True)
    print(f"DONE {time.time() - t0:.0f}s", flush=True)

#### CELL 33
# Import the Raptor arm as a real module (its own globals, exactly as the
# parent's exec-into-a-fresh-dict provided) and install the raw-output guards.
import sys as _ke_sys
if '/kaggle/working' not in _ke_sys.path:
    _ke_sys.path.insert(0, '/kaggle/working')
import raptor_arm as _ke_raptor

_rep_raptor_checked_predictions = 0
_rep_raptor_checked_ranks = 0

def _rep_wrap_raptor(module):
    """Audit wrapper: count and range-check raw outputs without altering them."""
    global _rep_raptor_guard_installed
    import numpy as _np
    _orig_predict, _orig_rank = module.infer_probs, module.rankpct

    def checked_predict(*a, **k):
        global _rep_raptor_checked_predictions
        out = _orig_predict(*a, **k)
        assert _np.asarray(out).shape == (12,), 'Raptor raw prediction shape'
        assert _np.isfinite(out).all(), 'Raptor raw nonfinite predictions'
        _rep_raptor_checked_predictions += 1
        return out

    def checked_rank(values):
        global _rep_raptor_checked_ranks
        assert _np.isfinite(values).all(), 'Raptor raw nonfinite blend'
        _rep_raptor_checked_ranks += 1
        return _orig_rank(values)

    module.infer_probs, module.rankpct = checked_predict, checked_rank
    _rep_raptor_guard_installed = True

_rep_wrap_raptor(_ke_raptor)
_ke_raptor.main()

#### CELL 34
import numpy as _ke_np
import pandas as _ke_pd
from pathlib import Path as _KePath

_ke_primary = _KePath('/kaggle/working/submission.csv')
_ke_ours = _ke_pd.read_csv(_ke_primary, dtype={'StudyInstanceUID': str})
_KE_LAB = [c for c in _ke_ours.columns if c != 'StudyInstanceUID']

def _coat_substitute():
    import hashlib as _h, os as _o, subprocess as _sp, sys as _sy
    from pathlib import Path as _P
    import pandas as _pd

    MAN_SHA = '98511a8fdeb9da0e6e70c78d013ff636e1476f31c80b5dc134d294b18c3f284e'
    WHL_SHA = '236c8df54a90f4d02076e6f9c1cc763d794542e886c576a6fee46ec8ff75a7a9'
    raptor = _P('/kaggle/working/_raptor.csv')

    def sha(p):
        d = _h.sha256()
        with _P(p).open('rb') as f:
            for b in iter(lambda: f.read(8 << 20), b''):
                d.update(b)
        return d.hexdigest()

    def find(name, want):
        root = _P('/kaggle/input')
        if not root.is_dir():
            return None
        for base in sorted(root.iterdir()):
            if base.name in ('competitions', 'train_series', 'test_series'):
                continue
            for p in sorted(base.rglob(name)):
                if p.is_file() and sha(p) == want:
                    return p
        return None

    man = find('coat_resgated_ep10_top3_manifest.json', MAN_SHA)
    if man is None:
        raise RuntimeError('coat manifest absent or hash mismatch')
    art = man.parent
    whl = find('opencv_python_headless-4.12.0.88-*.whl', WHL_SHA)
    if whl is None:
        for c in sorted(_P('/kaggle/input').rglob('opencv_python_headless-4.12.0.88-*.whl')):
            if sha(c) == WHL_SHA:
                whl = c
                break
    if whl is None:
        raise RuntimeError('pinned opencv wheel absent or hash mismatch')

    envd = _P('/kaggle/working/_coat_env')
    _sp.run([_sy.executable, '-m', 'pip', 'install', '--no-deps', '--quiet',
             '--target', str(envd), str(whl)], check=True)

    out = _P('/kaggle/working/_coat_arm.csv')
    child = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(envd)!r})\n"
        f"sys.path.insert(0, {str(art)!r})\n"
        "import cv2; assert cv2.__version__ == '4.12.0', cv2.__version__\n"
        "import torch; assert torch.cuda.device_count() == 2\n"
        "import coatnet_resgated_ep10_top3_inference as rt\n"
        "assert rt.base.cv2.__version__ == '4.12.0'\n"
        "from pathlib import Path\n"
        "r = rt.run_submission(competition_root=rt.base.find_competition_root(),\n"
        f"    artifact_root=Path({str(art)!r}), output_path=Path({str(out)!r}),\n"
        "    gpu_batch_studies=2, backbone_micro_images=8)\n"
        "assert r['status'] == rt.SUBMISSION_STATUS\n"
        "assert r['models'] == 3\n"
        "assert [i['epoch'] for i in r['checkpoints']] == [4, 6, 8]\n"
        "assert r['fallback_studies'] == 0, r['failures']\n"
        "Path('/kaggle/working/_coat_arm_receipt.json').write_text(json.dumps(r, indent=2))\n")
    env = dict(_o.environ)
    env['PYTHONPATH'] = f"{envd}:{art}:" + env.get('PYTHONPATH', '')
    proc = _sp.run([_sy.executable, '-c', child], env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f'coat child failed: {proc.stderr[-700:]}')

    pub = _pd.read_csv(raptor, dtype={'StudyInstanceUID': str})
    ours = _pd.read_csv(out, dtype={'StudyInstanceUID': str})
    if list(ours.columns) != list(pub.columns):
        raise RuntimeError('coat arm column drift')
    ours = ours.set_index('StudyInstanceUID').reindex(
        pub.StudyInstanceUID.astype(str).tolist()).reset_index()
    lab = [c for c in pub.columns if c != 'StudyInstanceUID']
    if ours[lab].isna().any().any():
        raise RuntimeError('coat arm does not cover every study')
    import json as _j
    import numpy as _np
    private_alpha = 0.40000000000000002
    public_rank = pub[lab].rank(method='average', pct=True)
    private_rank = ours[lab].rank(method='average', pct=True)
    hybrid = pub.copy()
    hybrid[lab] = (
        (1.0 - private_alpha) * public_rank
        + private_alpha * private_rank
    )
    if not _np.isfinite(hybrid[lab].to_numpy(_np.float64)).all():
        raise RuntimeError('CoAt/Raptor hybrid contains non-finite values')
    tmp = raptor.with_name('.raptor_coat_hybrid.csv')
    hybrid.to_csv(tmp, index=False)
    _o.replace(tmp, raptor)
    raptor.with_name('_coat_raptor_blend_receipt.json').write_text(_j.dumps({
        'contract': 'public_raptor_private_residual_coat_global_rank_blend_v1',
        'private_alpha': private_alpha,
        'public_raptor_alpha': 1.0 - private_alpha,
        'study_count': len(ours),
        'finding_specific_weights': False,
    }, indent=2, sort_keys=True) + '\n')
    return len(ours)


_coat_public_path = _KePath('/kaggle/working/_raptor.csv')
_coat_n = _coat_substitute()
print(f'[coat-arm] blended OUR resgated e4/e6/e8 into the public Raptor arm '
      f'(private alpha 0.400; {_coat_n} studies)', flush=True)

_ke_theirs = _ke_pd.read_csv('/kaggle/working/_raptor.csv',
                             dtype={'StudyInstanceUID': str})
assert list(_ke_theirs.columns) == list(_ke_ours.columns), 'column drift'
_ke_theirs = _ke_theirs.set_index('StudyInstanceUID').reindex(
    _ke_ours['StudyInstanceUID']).reset_index()
assert _ke_theirs[_KE_LAB].notna().all().all(), 'study identity drift'


_ke_tr = _ke_ours[_KE_LAB].rank(method='average', pct=True)
_ke_cr = _ke_theirs[_KE_LAB].copy()
_blend_transformer = _ke_ours.copy()
_blend_coatnet = _ke_theirs.copy()
_blend_labels = list(_KE_LAB)
_blend_tr = _ke_tr.copy()
_blend_cr = _ke_cr.copy()
_coatnet_weight = {label: 0.60 for label in _blend_labels}
# probe22 per-finding outer weights (maverickss26, Apache-2.0): raising these
# five is what took their notebook from 0.939 to 0.941.
_coatnet_weight.update({'ACL': 0.75, 'Medial Meniscus': 0.8, 'Lateral Meniscus': 1.0, 'Lateral OA': 0.75, 'Fracture': 0.75})
_blend_output = _blend_transformer.copy()
for _blend_label in _blend_labels:
    _blend_w = float(_coatnet_weight[_blend_label])
    _blend_output[_blend_label] = (
        (1.0 - _blend_w) * _blend_tr[_blend_label]
        + _blend_w * _blend_cr[_blend_label]
    )
_blend_output[_blend_labels] = _blend_output[_blend_labels].rank(
    method='average', pct=True
)
assert _ke_np.isfinite(
    _blend_output[_blend_labels].to_numpy(_ke_np.float64)
).all()
_blend_output.to_csv(_ke_primary, index=False)

#### CELL 39
"""Validate the executed source graph and bind the actual final CSV."""
import numpy as _rep_np
import pandas as _rep_pd

_rep_snapshot('final')
_rep_work = _RepPath('/kaggle/working')
_rep_log = ''.join(_rep_messages)
_rep_failures = _rep_re.findall(
    r'^.*(?:\[arm \d+\].*FALLBACK|study failed:|degenerate predictions; not banked|'
    r'public-frontier vote omitted|surrendering \d+ member|probe FAIL).*$',
    _rep_log, flags=_rep_re.MULTILINE)
assert not _rep_failures, _rep_failures
assert 'rank mean of 20 member(s)' in _rep_log, 'DINO member coverage'
assert sum(('loaded m_f' + str(i) + '.pt') in _rep_log for i in range(5)) == 5, 'A5 folds'
assert all(('[arm ' + str(i) + '] done + freed') in _rep_log for i in range(4)), 'Raptor views'
assert V18_CALIBRATOR_APPLIED is True, 'source calibrator did not run'
assert _rep_dino_guard_installed and _rep_raptor_guard_installed, 'raw guards'
assert _rep_dino_raw_members_checked == 20, 'DINO raw coverage'
assert _rep_rad_guard_installed and _rep_rad_rank_checks > 0, 'Rad raw rank guards'
assert _rep_raptor_checked_predictions == 4 * _rep_stages['final']['rows'], 'raw Raptor coverage'
assert _rep_raptor_checked_ranks == 1, 'raw Raptor blend coverage'

_rep_coat = _rep_json.loads((_rep_work / '_coat_arm_receipt.json').read_text())
assert _rep_coat['status'] == 'VALID_COAT_RESGATED_EP10_TOP3_RANK_SUBMISSION'
assert _rep_coat['models'] == 3
assert [c['epoch'] for c in _rep_coat['checkpoints']] == [4, 6, 8]
assert _rep_coat['fallback_studies'] == 0 and not _rep_coat['failures']
assert _rep_coat['output_sha256'] == _rep_sha(_rep_work / '_coat_arm.csv')
_rep_blend = _rep_json.loads((_rep_work / '_coat_raptor_blend_receipt.json').read_text())
assert _rep_blend['private_alpha'] == 0.4
assert _rep_blend['finding_specific_weights'] is False
_rep_rad = _rep_json.loads((_rep_work / 'v50_v2_repro_receipt.json').read_text())
assert _rep_rad['submission_sha256'] == _rep_stages['rad']['sha256']

# Recompute the final fusion independently from saved intermediate outputs.
_rep_parent = _rep_pd.read_csv(_rep_work / 'reproduction_rad.csv', dtype={'StudyInstanceUID': str})
_rep_hybrid = _rep_pd.read_csv(_rep_work / '_raptor.csv', dtype={'StudyInstanceUID': str})
_rep_final = _rep_pd.read_csv(_rep_work / 'submission.csv', dtype={'StudyInstanceUID': str})
_rep_labels = list(_rep_final.columns[1:])
assert _rep_hybrid.StudyInstanceUID.tolist() == _rep_final.StudyInstanceUID.tolist()
_rep_parent_rank = _rep_parent[_rep_labels].rank(method='average', pct=True)
_rep_rebuilt = _rep_parent_rank.copy()
for _rep_label in _rep_labels:
    _rep_weight = {'ACL': 0.75, 'Medial Meniscus': 0.8, 'Lateral Meniscus': 1.0, 'Lateral OA': 0.75, 'Fracture': 0.75}.get(_rep_label, 0.6)
    _rep_rebuilt[_rep_label] = ((1.0 - _rep_weight) * _rep_parent_rank[_rep_label]
                              + _rep_weight * _rep_hybrid[_rep_label])
_rep_rebuilt = _rep_rebuilt.rank(method='average', pct=True)
assert _rep_np.allclose(_rep_final[_rep_labels], _rep_rebuilt, rtol=0, atol=1e-15)
assert _rep_coat['studies'] == len(_rep_final)
assert _rep_rad['test_studies'] == len(_rep_final)
assert _rep_blend['study_count'] == len(_rep_final)

_rep_receipt = {
    'contract': 'exact_public_mattia_v27_reproduction_v1',
    'status': 'PASS',
    'source_notebook_sha256': _rep_source_sha256,
    'source_version': 27,
    'source_script_version_id': 348195103,
    'prediction_source_modified': False,
    'inference_only': True,
    'new_gold_evaluation': False,
    'gpu_names': _rep_gpu_names,
    'rows': len(_rep_final),
    'columns': list(_rep_final.columns),
    'uid_sha256': _rep_hashlib.sha256(('\n'.join(_rep_final.StudyInstanceUID) + '\n').encode()).hexdigest(),
    'stages': _rep_stages,
    'final_submission_sha256': _rep_sha(_rep_work / 'submission.csv'),
    'coat_receipt_sha256': _rep_sha(_rep_work / '_coat_arm_receipt.json'),
    'rad_receipt_sha256': _rep_sha(_rep_work / 'v50_v2_repro_receipt.json'),
    'final_fusion_reconstruction_passed': True,
    'execution_fallbacks': _rep_failures,
    'raptor_raw_predictions_checked': _rep_raptor_checked_predictions,
    'dino_raw_members_checked': _rep_dino_raw_members_checked,
    'rad_raw_rank_checks': _rep_rad_rank_checks,
    'elapsed_seconds': _rep_time.time() - _rep_started,
    'score': None,
    'score_note': 'Only a completed competition submissions API row establishes our score.',
}
(_rep_work / 'reproduction_receipt.json').write_text(_rep_json.dumps(_rep_receipt, indent=2, sort_keys=True) + '\n')
print('[reproduction] PASS', _rep_receipt['final_submission_sha256'], flush=True)

