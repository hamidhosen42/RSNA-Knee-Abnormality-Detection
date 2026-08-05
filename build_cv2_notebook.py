"""Build the score-focused cv2 notebook from the user's original cv1 notebook.

The source notebook stores each cell as one string.  Keeping the upgrade in this
small builder makes the substantial cell replacements reviewable and lets us
recreate the .ipynb deterministically without requiring nbformat.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "rsna-knee-abnormality-detection-cv1.ipynb"
DEST = ROOT / "rsna-knee-abnormality-detection-cv2.ipynb"


def clean(source: str) -> str:
    return textwrap.dedent(source).strip() + "\n"


with SOURCE.open(encoding="utf-8") as f:
    nb = json.load(f)


def set_cell(index: int, source: str, cell_type: str | None = None) -> None:
    cell = nb["cells"][index]
    if cell_type is not None:
        cell["cell_type"] = cell_type
    cell["source"] = clean(source)
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []


def guard_cell(index: int, condition: str) -> None:
    cell = nb["cells"][index]
    source = cell["source"]
    if isinstance(source, list):
        source = "".join(source)
    cell["source"] = f"if {condition}:\n" + textwrap.indent(source.rstrip(), "    ") + "\n"
    cell["execution_count"] = None
    cell["outputs"] = []


set_cell(0, r'''
<div style="background:linear-gradient(90deg,#1c3f6e 0%,#4d6d9a 45%,#b4636b 78%,#d97e5a 100%);
            padding:22px 26px;border-radius:10px;color:#fff;font-family:sans-serif">
<h1 style="margin:0 0 6px 0;font-size:30px">🦵 RSNA Knee Abnormality Detection — CV2</h1>
<h3 style="margin:0;font-weight:400;opacity:.92">Leakage-safe report teacher · geometry-aware DICOM cache · target-aware 2.5D student</h3>
</div>

**Task** — predict per-study probabilities for 12 knee MRI findings.  
**Metric** — the unweighted mean of the 12 per-target ROC AUCs.  
**Inference data** — images and series descriptors only; reports are a training-time source of weak labels.

This version fixes the score and reliability blockers in CV1:

1. every observed target is retained, including partially labelled studies;
2. folds are made from observed labels and patient/report groups *before* pseudo-labelling;
3. each image fold gets a report teacher that never sees that fold's labels;
4. DICOM orientation, physical aspect ratio, foreground crop, slice masks and cache version are explicit;
5. the student uses pretrained-input normalization and target-specific slice/view attention;
6. missing cache shards, missing checkpoints and failed submission inference raise errors instead of
   silently training on zeros or writing a dummy submission.

The notebook is intentionally staged. Set `CFG.STAGE` to one of `eda`, `teacher`, `preprocess`,
`train`, or `infer`. Publish outputs from one stage as a Kaggle Dataset and attach them to the next.
''', "markdown")


set_cell(1, r'''
## What changes the score most

| Priority | CV1 problem | CV2 correction |
|---|---|---|
| 1 | partially observed rows are discarded | per-target observed masks in teacher, loss and AUC |
| 2 | report teacher is fitted on validation labels | fold-specific cross-fitted teachers |
| 3 | pretrained encoder receives unnormalised `[0,1]` input | use the backbone's own mean/std |
| 4 | one attention map is shared by all findings | learned target-specific slice and view queries |
| 5 | one series per plane loses complementary contrast | five sequence slots: sagittal/coronal FS + non-FS, axial |
| 6 | series ties are CSV-order dependent | real DICOM counts and deterministic quality ranking |
| 7 | incomplete cache becomes blank studies | strict cache coverage and shape/version validation |
| 8 | every checkpoint decodes test DICOM again | decode each batch once, run all folds on that batch |
| 9 | last gradient accumulation is dropped | exact accumulation and scheduler step counts |
| 10 | no checkpoint still produces a CSV | fail closed and validate finite, non-constant predictions |

Macro AUC is invariant to a pure change in class prevalence, but not to site/protocol or
class-conditional distribution shift. Prefer stable grouped OOF results to public-leaderboard tuning.
''', "markdown")


set_cell(2, r'''
## 0. Configuration

Run one stage at a time. The submission stage must have internet disabled and must receive model
weights and decoder dependencies through attached Kaggle Datasets. Training all folds is deliberate;
set `TRAIN_FOLDS=[0]` only for a smoke test, never for the final ensemble.
''', "markdown")


set_cell(3, r'''
import os, re, sys, gc, json, math, time, warnings, hashlib, unicodedata
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 80)


class CFG:
    # ------------------------------ choose exactly one stage
    STAGE = 'eda'  # eda | teacher | preprocess | train | infer
    VALID_STAGES = {'eda', 'teacher', 'preprocess', 'train', 'infer'}

    # ------------------------------ paths
    comp_dir = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
    work_dir = Path('/kaggle/working')
    teacher_dir = work_dir / 'rsna_teacher_v2'
    cache_dir = work_dir / 'rsna_cache_v2'
    ckpt_dir = work_dir / 'rsna_models_v2'

    # Attach published stage outputs here. Do not recursively scan the 570 GB competition mount.
    extra_teacher_dirs = []  # e.g. [Path('/kaggle/input/rsna-teacher-v2')]
    extra_cache_dirs = []    # e.g. [Path('/kaggle/input/rsna-cache-v2-shard00'), ...]
    extra_ckpt_dirs = []     # e.g. [Path('/kaggle/input/rsna-models-v2')]

    # ------------------------------ preprocessing
    CACHE_VERSION = 'cv2-orient-roi-letterbox-256-12-v1'
    IMG_SIZE = 256
    N_SLICES = 12
    VIEWS = ['SAG_FS', 'SAG_NFS', 'COR_FS', 'COR_NFS', 'AX']
    N_VIEWS = len(VIEWS)
    SHARD_ID = 0
    NUM_SHARDS = 8
    PREPROCESS_SMOKE_ONLY = False

    # ------------------------------ model
    backbone = [
        'tf_efficientnetv2_s.in21k_ft_in1k',
        'tf_efficientnetv2_s',
        'efficientnetv2_rw_s',
        'tf_efficientnetv2_b0',
    ]
    pretrained = True
    drop_rate = 0.25
    drop_path_rate = 0.15
    grad_ckpt = True

    # ------------------------------ training
    folds = 5
    TRAIN_FOLDS = None       # None = all folds; [0] = smoke test
    mixed_epochs = 6         # hard + report-derived targets
    hard_epochs = 2          # final hard-label-only fine-tuning
    batch_size = 2
    infer_batch_size = 4
    accum = 4
    lr = 3e-4
    backbone_lr = 8e-5
    wd = 1e-2
    pseudo_w = 0.45
    pseudo_conf_floor = 0.20
    aux_w = 0.20
    max_pos_weight = 5.0
    ema_decay = 0.999
    seed = 42
    num_workers = 4
    amp = True


assert CFG.STAGE in CFG.VALID_STAGES, (CFG.STAGE, CFG.VALID_STAGES)
CFG.RUN_EDA = CFG.STAGE == 'eda'
CFG.RUN_TEACHER = CFG.STAGE == 'teacher'
CFG.RUN_PREPROCESS = CFG.STAGE == 'preprocess'
CFG.RUN_TRAIN = CFG.STAGE == 'train'
CFG.RUN_INFER = CFG.STAGE == 'infer'

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
           'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
           'Synovitis', "Baker's", 'Contusion', 'Fracture']
WEIGHT_COLS = [f'w__{c}' for c in TARGETS]
OBS_COLS = [f'obs__{c}' for c in TARGETS]

for d in [CFG.work_dir, CFG.teacher_dir, CFG.cache_dir, CFG.ckpt_dir]:
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ plotting theme
ICEFIRE = sns.color_palette('icefire', 12)
ICE, FIRE, MIDLINE = ICEFIRE[2], ICEFIRE[9], '#39414d'
CMAP = 'icefire'
CMAP_SEQ = sns.blend_palette(['#eef1f6', ICE, FIRE], as_cmap=True)
sns.set_theme(style='white', palette=ICEFIRE, font_scale=0.95)
mpl.rcParams.update({
    'figure.dpi': 120, 'savefig.dpi': 120,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#c8cdd4', 'axes.labelcolor': MIDLINE,
    'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.titlepad': 10,
    'text.color': MIDLINE, 'xtick.color': MIDLINE, 'ytick.color': MIDLINE,
    'grid.color': '#e8eaee', 'axes.grid': True, 'axes.grid.axis': 'y',
    'legend.frameon': False, 'figure.facecolor': 'white',
})


def grad(n, reverse=False):
    cm = plt.get_cmap(CMAP)
    cold = np.linspace(0.02, 0.28, n - n // 2)
    warm = np.linspace(0.78, 0.99, n // 2)
    cols = [cm(p) for p in np.concatenate([cold, warm])][:n]
    return cols[::-1] if reverse else cols


def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def find_artifact(name, primary, extras):
    hits = [Path(primary) / name] + [Path(d) / name for d in extras]
    hits = [p for p in hits if p.exists()]
    if not hits:
        raise FileNotFoundError(
            f'{name} not found. Attach the preceding-stage dataset and list its directory in CFG.')
    if len(hits) > 1:
        print(f'[artifact] multiple copies of {name}; using {hits[0]}')
    return hits[0]


seed_everything(CFG.seed)
if not CFG.comp_dir.exists():
    candidates = list(Path('/kaggle/input').glob('*knee*')) + \
                 list(Path('/kaggle/input/competitions').glob('*knee*'))
    if candidates:
        CFG.comp_dir = candidates[0]
print('stage:', CFG.STAGE, '| comp_dir:', CFG.comp_dir, '| exists:', CFG.comp_dir.exists())
''')


set_cell(4, r'''
### Environment and decoder preflight

JPEG Lossless and JPEG 2000 occur in the data. The inference notebook is offline, so decoder wheels,
model weights and any non-default dependency must be attached beforehand. This check now fails during
preprocess/inference if no compressed-pixel decoder is available; silent missing views are not accepted.
''', "markdown")


set_cell(5, r'''
import pydicom

print('pydicom:', pydicom.__version__)
decoders = {}
for mod in ['pylibjpeg', 'openjpeg', 'libjpeg', 'gdcm', 'PIL']:
    try:
        __import__(mod)
        decoders[mod] = 'ok'
    except Exception as e:
        decoders[mod] = f'MISSING ({type(e).__name__})'
for k, v in decoders.items():
    print(f'  {k:<10}: {v}')

compressed_decoder_ok = any(decoders[k] == 'ok' for k in ['openjpeg', 'libjpeg', 'gdcm'])
if (CFG.RUN_PREPROCESS or CFG.RUN_INFER) and not compressed_decoder_ok:
    raise RuntimeError(
        'No JPEG/JPEG-2000 DICOM decoder. Attach offline wheels for pylibjpeg, '
        'pylibjpeg-openjpeg and pylibjpeg-libjpeg (or python-gdcm).')

try:
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu only'
    print(f'torch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | {gpu}')
except ImportError:
    print('torch unavailable (acceptable only for EDA/teacher/preprocess)')
''')


set_cell(6, r'''
required = ['train.csv', 'train_series.csv', 'test.csv', 'test_series.csv', 'sample_submission.csv']
missing = [x for x in required if not (CFG.comp_dir / x).exists()]
if missing:
    raise FileNotFoundError(f'Competition mount is incomplete: {missing}')

train = pd.read_csv(CFG.comp_dir / 'train.csv')
train_series = pd.read_csv(CFG.comp_dir / 'train_series.csv')
test = pd.read_csv(CFG.comp_dir / 'test.csv')
test_series = pd.read_csv(CFG.comp_dir / 'test_series.csv')
sample_sub = pd.read_csv(CFG.comp_dir / 'sample_submission.csv')

assert list(sample_sub.columns) == ['StudyInstanceUID'] + TARGETS
assert 'Report' not in test.columns, 'The strategy assumes reports are unavailable at inference.'
assert train.StudyInstanceUID.is_unique and test.StudyInstanceUID.is_unique

observed = train[TARGETS].notna()
has_any_label = observed.any(axis=1)
has_all_labels = observed.all(axis=1)
has_report = train['Report'].fillna('').str.strip().str.len() > 0
_label_values = train[TARGETS].where(observed).stack().astype(float).values
if len(_label_values) and not np.isin(_label_values, [0.0, 1.0]).all():
    raise ValueError('Every observed target must be binary 0/1.')

print(f'train studies: {len(train):,} | train series: {len(train_series):,}')
print(f'test example studies: {len(test):,} | test series: {len(test_series):,}')
print(f'any observed label: {has_any_label.sum():,} | all 12: {has_all_labels.sum():,} '
      f'| report present: {has_report.sum():,}')
''')


set_cell(8, r'''
if CFG.RUN_EDA:
    per_target = pd.DataFrame(index=TARGETS)
    per_target['n_observed'] = observed.sum()
    per_target['missing_pct'] = (1 - observed.mean()) * 100
    per_target['n_pos'] = train[TARGETS].fillna(0).sum().astype(int)
    per_target['n_neg'] = per_target.n_observed - per_target.n_pos
    per_target['prevalence_observed'] = per_target.n_pos / per_target.n_observed.clip(lower=1)
    per_target['expected_pos_per_fold'] = per_target.n_pos / CFG.folds
    per_target['expected_neg_per_fold'] = per_target.n_neg / CFG.folds
    display(per_target.style.background_gradient(cmap=CMAP_SEQ,
                                                  subset=['prevalence_observed'])
                            .format({'missing_pct': '{:.1f}',
                                     'prevalence_observed': '{:.3f}',
                                     'expected_pos_per_fold': '{:.1f}',
                                     'expected_neg_per_fold': '{:.1f}'}))

    known_per_study = observed.sum(axis=1)
    print('\nknown labels per study:')
    print(known_per_study.value_counts().sort_index().to_string())
    print(f'neither report nor any label: {(~has_report & ~has_any_label).sum():,}')

    # Compatibility aliases used only by the original EDA plots below.
    has_label = has_all_labels
    lab = train[has_all_labels].copy()
    prev = pd.Series({c: train.loc[observed[c], c].mean() for c in TARGETS}).sort_values(ascending=False)
''')


# Existing EDA-only code cells. Core masks are defined above, so these can be skipped in other stages.
for _idx in [9, 11, 13, 15, 16, 18, 20, 22]:
    guard_cell(_idx, 'CFG.RUN_EDA')


set_cell(17, r'''
**Audit checklist**

- Count observed positives *and negatives per target*; a row need not have all twelve labels.
- Verify pixel decoding, not only header reading, for every observed transfer syntax.
- Use `PatientID` when retained and exact normalized report duplicates as fold groups.
- Treat `Fluid_Sensitive=0` as “not marked fluid-sensitive,” not automatically as T1.
- Compare CSV series identifiers to directories and count physical/unique slices before selecting a series.
- Inspect canonical coronal/axial/sagittal montages on both left and right knees before a long cache run.
''', "markdown")


set_cell(23, r'''
# Part 2 — study/series data model

CV2 keeps complementary contrasts instead of forcing one series per plane:

| Slot | Selection | Main signal |
|---|---|---|
| `SAG_FS` | sagittal, fat-suppressed; fluid-sensitive preferred | ACL, contusion, effusion |
| `SAG_NFS` | sagittal, non-fat-suppressed; fluid-sensitive preferred | menisci, cartilage, fracture |
| `COR_FS` | coronal, fat-suppressed; fluid-sensitive preferred | MCL, contusion, OA |
| `COR_NFS` | coronal, non-fat-suppressed; fluid-sensitive preferred | menisci, OA, fracture |
| `AX` | axial; fluid-sensitive and fat suppression preferred | PF OA, synovitis, effusion, Baker's |

Selections are unique, use actual `.dcm` counts, and reject obvious single-slice/localizer series.
The learned target-aware pooling can ignore an unavailable or unhelpful slot using masks.
''', "markdown")


set_cell(24, r'''
VIEW_SPEC = {
    'SAG_FS':  dict(plane='Sagittal', fat=1),
    'SAG_NFS': dict(plane='Sagittal', fat=0),
    'COR_FS':  dict(plane='Coronal',  fat=1),
    'COR_NFS': dict(plane='Coronal',  fat=0),
    'AX':      dict(plane='Axial',    fat=None),
}


def _series_file_count(root, study, series):
    return sum(1 for _ in (Path(root) / str(study) / str(series)).glob('*.dcm'))


def assign_views(series_df: pd.DataFrame) -> dict:
    """Choose distinct, deterministic series for the five sequence slots."""
    g = series_df.copy()
    for c in ['Fluid_Sensitive', 'Fat_Suppression', 'n_files']:
        if c not in g:
            g[c] = 0
    g['Fluid_Sensitive'] = pd.to_numeric(g.Fluid_Sensitive, errors='coerce').fillna(0).astype(int)
    g['Fat_Suppression'] = pd.to_numeric(g.Fat_Suppression, errors='coerce').fillna(0).astype(int)
    g['n_files'] = pd.to_numeric(g.n_files, errors='coerce').fillna(0).astype(int)
    used, out = set(), {}

    for slot, spec in VIEW_SPEC.items():
        cand = g[(g.Anatomical_Plane == spec['plane']) & (~g.SeriesInstanceUID.isin(used))].copy()
        if spec['fat'] is not None:
            cand = cand[cand.Fat_Suppression == spec['fat']]
        cand = cand[cand.n_files >= 4]  # excludes most scouts/localizers
        if cand.empty:
            out[slot] = None
            continue
        cand['fluid_rank'] = (cand.Fluid_Sensitive != 1).astype(int)
        cand = cand.sort_values(['fluid_rank', 'n_files', 'SeriesInstanceUID'],
                                ascending=[True, False, True], kind='stable')
        sid = str(cand.iloc[0].SeriesInstanceUID)
        out[slot] = sid
        used.add(sid)
    return out


def build_manifest(series_df: pd.DataFrame, root: Path, count_files=True) -> pd.DataFrame:
    df = series_df.drop_duplicates(['StudyInstanceUID', 'SeriesInstanceUID']).copy()
    if count_files:
        df['n_files'] = [_series_file_count(root, r.StudyInstanceUID, r.SeriesInstanceUID)
                         for r in df.itertuples()]
    elif 'n_files' not in df:
        raise ValueError('n_files is required for deterministic series selection')
    rows = []
    for study, g in df.groupby('StudyInstanceUID', sort=False):
        rec = {'StudyInstanceUID': study, 'n_series': len(g),
               'n_dicom_total': int(g.n_files.sum())}
        rec.update(assign_views(g))
        rows.append(rec)
    return pd.DataFrame(rows)


_demo = pd.DataFrame([
    dict(SeriesInstanceUID='a', Anatomical_Plane='Sagittal', Fluid_Sensitive=1,
         Fat_Suppression=1, n_files=30),
    dict(SeriesInstanceUID='b', Anatomical_Plane='Sagittal', Fluid_Sensitive=1,
         Fat_Suppression=0, n_files=28),
    dict(SeriesInstanceUID='c', Anatomical_Plane='Coronal', Fluid_Sensitive=1,
         Fat_Suppression=1, n_files=25),
])
_assigned = assign_views(_demo)
assert _assigned['SAG_FS'] == 'a' and _assigned['SAG_NFS'] == 'b'
assert len([x for x in _assigned.values() if x is not None]) == len(set(x for x in _assigned.values() if x))

manifest = None
if CFG.RUN_EDA or CFG.RUN_PREPROCESS:
    manifest = build_manifest(train_series, CFG.comp_dir / 'train_series', count_files=True)
    fill = (manifest[CFG.VIEWS].notna().mean() * 100).round(1)
    print('view fill rate (%):', fill.to_dict())
    print('studies with no selected view:', int(manifest[CFG.VIEWS].isna().all(axis=1).sum()))
''')


set_cell(25, r'''
## 2.1 Geometry-aware DICOM preprocessing

The loader now:

1. deduplicates physical slice positions and orders depth in patient coordinates;
2. uses IOP direction cosines to standardize the two in-plane axes;
3. maps left and right knees onto the same lateral→medial coordinate;
4. rescales pixel axes to physical spacing, crops the shared foreground ROI, then letterboxes
   (never stretches) into a square;
5. returns a slice-valid mask and records the preprocessing version in every cache item.

Oblique acquisitions are handled by selecting the array axis most aligned with the requested patient
coordinate. Always inspect a left/right montage: DICOM orientation bugs are high-impact and visual.
''', "markdown")


set_cell(26, r'''
import cv2
cv2.setNumThreads(0)


def _side(headers):
    for ds in headers:
        v = str(getattr(ds, 'Laterality', '') or getattr(ds, 'ImageLaterality', '')).upper()
        if v in {'L', 'R'}:
            return v
    xs = []
    for ds in headers:
        try:
            xs.append(float(ds.ImagePositionPatient[0]))
        except Exception:
            pass
    if xs:
        return 'R' if float(np.median(xs)) < 0 else 'L'
    return 'R'  # unknown: preserve pixels instead of applying an arbitrary mirror


def _position(ds, plane, side):
    """Monotonic patient-coordinate position for deterministic depth order."""
    try:
        ipp = np.asarray(ds.ImagePositionPatient, np.float64)
        if plane == 'Sagittal':
            return float(ipp[0] * (1 if side == 'R' else -1))  # lateral -> medial
        if plane == 'Coronal':
            return float(ipp[1])                               # anterior -> posterior
        return float(ipp[2])                                   # inferior -> superior
    except Exception:
        return float(getattr(ds, 'InstanceNumber', 0) or 0)


def _orient_frame(a, ds, plane, side):
    """Map array H/W to a stable patient-coordinate display convention."""
    try:
        iop = np.asarray(ds.ImageOrientationPatient, np.float64)
        width_dir, height_dir = iop[:3].copy(), iop[3:].copy()
    except Exception:
        # Common radiological convention fallback.
        if side == 'L' and plane in {'Coronal', 'Axial'}:
            a = a[:, ::-1]
        return np.ascontiguousarray(a), False

    side_factor = 1.0 if side == 'R' else -1.0
    if plane == 'Sagittal':
        h_axis, h_sign, v_axis, v_sign = 1, +1, 2, -1  # posterior right; inferior down
    elif plane == 'Coronal':
        h_axis, h_sign, v_axis, v_sign = 0, side_factor, 2, -1  # lateral -> medial
    else:
        h_axis, h_sign, v_axis, v_sign = 0, side_factor, 1, +1  # lateral -> medial; posterior down

    # If the desired horizontal patient axis is carried by array rows, transpose first.
    transposed = abs(height_dir[h_axis]) > abs(width_dir[h_axis])
    if transposed:
        a = a.T
        width_dir, height_dir = height_dir, width_dir
    if width_dir[h_axis] * h_sign < 0:
        a = a[:, ::-1]
    if height_dir[v_axis] * v_sign < 0:
        a = a[::-1, :]
    return np.ascontiguousarray(a), transposed


def _resize_physical(a, ds, transposed=False):
    ps = getattr(ds, 'PixelSpacing', None)
    sy, sx = (float(ps[0]), float(ps[1])) if ps and len(ps) >= 2 else (1.0, 1.0)
    if transposed:
        sy, sx = sx, sy
    base = max(min(sy, sx), 1e-6)
    nh = max(1, int(round(a.shape[0] * sy / base)))
    nw = max(1, int(round(a.shape[1] * sx / base)))
    if (nh, nw) != a.shape:
        interp = cv2.INTER_AREA if nh < a.shape[0] or nw < a.shape[1] else cv2.INTER_LINEAR
        a = cv2.resize(a, (nw, nh), interpolation=interp)
    return a


def _shared_foreground_bbox(vol, pad=0.08):
    mean_img = np.mean(vol, axis=0)
    positive = mean_img[mean_img > np.min(mean_img)]
    if positive.size < 64:
        return 0, vol.shape[1], 0, vol.shape[2]
    threshold = np.percentile(positive, 12)
    mask = (mean_img > threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    ys, xs = np.where(mask > 0)
    if len(xs) < 64:
        return 0, vol.shape[1], 0, vol.shape[2]
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    py, px = int((y1 - y0) * pad), int((x1 - x0) * pad)
    return max(0, y0 - py), min(vol.shape[1], y1 + py), \
           max(0, x0 - px), min(vol.shape[2], x1 + px)


def _letterbox(a, size):
    scale = min(size / a.shape[0], size / a.shape[1])
    nh, nw = max(1, int(round(a.shape[0] * scale))), max(1, int(round(a.shape[1] * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    b = cv2.resize(a, (nw, nh), interpolation=interp)
    out = np.zeros((size, size), dtype=b.dtype)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = b
    return out


def load_series_volume(series_dir: Path, plane: str, size=CFG.IMG_SIZE,
                       n_slices=CFG.N_SLICES):
    """Return `(uint8[D,H,W], valid_slice_mask[D])`, or `(None, None)`."""
    files = list(Path(series_dir).glob('*.dcm'))
    if not files:
        return None, None

    pairs = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            pairs.append((f, ds))
        except Exception:
            continue
    if not pairs:
        return None, None

    # Reject embedded localizers/mixed geometry inside a nominal series.
    def geometry_key(ds):
        try:
            iop = tuple(np.round(np.asarray(ds.ImageOrientationPatient, float), 3))
        except Exception:
            iop = None
        return (int(getattr(ds, 'Rows', 0) or 0),
                int(getattr(ds, 'Columns', 0) or 0), iop)
    geom_counts = Counter(geometry_key(ds) for _, ds in pairs)
    dominant = max(geom_counts, key=lambda k: (geom_counts[k], k[2] is not None,
                                               k[0] * k[1], str(k)))
    pairs = [(f, ds) for f, ds in pairs if geometry_key(ds) == dominant]

    side = _side([x[1] for x in pairs])
    keyed = sorted(((_position(ds, plane, side), str(f), f, ds) for f, ds in pairs),
                   key=lambda x: (x[0], x[1]))
    unique = []
    for item in keyed:
        if not unique or abs(item[0] - unique[-1][0]) > 1e-3:
            unique.append(item)
    if not unique:
        return None, None

    # Short series are spread across the fixed depth grid. Unfilled positions are
    # nearest-filled for 2.5D context but remain masked out of attention/encoding.
    n_take = min(n_slices, len(unique))
    take = np.unique(np.linspace(0, len(unique) - 1, n_take).round().astype(int))
    out_positions = np.linspace(0, n_slices - 1, len(take)).round().astype(int)
    frames, valid = [None] * n_slices, np.zeros(n_slices, np.uint8)
    for out_i, src_i in zip(out_positions, take):
        _, _, f, _ = unique[src_i]
        try:
            ds = pydicom.dcmread(str(f))
            a = ds.pixel_array.astype(np.float32)
            if a.ndim == 3:
                a = a[..., 0] if a.shape[-1] <= 4 else a[0]
            a = a * float(getattr(ds, 'RescaleSlope', 1) or 1) + \
                float(getattr(ds, 'RescaleIntercept', 0) or 0)
            if str(getattr(ds, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
                a = a.max() - a
            a, transposed = _orient_frame(a, ds, plane, side)
            a = _resize_physical(a, ds, transposed)
            frames[out_i] = a
            valid[out_i] = 1
        except Exception:
            frames[out_i] = None

    good = [x for x in frames if x is not None]
    if not good:
        return None, None
    # Orientation/spacing should make shapes identical; center-pad rare one-pixel discrepancies.
    mh, mw = max(x.shape[0] for x in good), max(x.shape[1] for x in good)
    aligned = []
    valid_positions = np.flatnonzero(valid)
    for out_i, a in enumerate(frames):
        if a is None:
            nearest = valid_positions[np.argmin(np.abs(valid_positions - out_i))]
            a = frames[int(nearest)]
        canvas = np.zeros((mh, mw), np.float32)
        y0, x0 = (mh - a.shape[0]) // 2, (mw - a.shape[1]) // 2
        canvas[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] = a
        aligned.append(canvas)
    vol = np.stack(aligned)

    vox = vol[valid.astype(bool)]
    lo, hi = np.percentile(vox, [0.5, 99.5])
    vol = np.clip((vol - lo) / (hi - lo + 1e-6), 0, 1)
    y0, y1, x0, x1 = _shared_foreground_bbox(vol[valid.astype(bool)])
    vol = np.stack([_letterbox(a[y0:y1, x0:x1], size) for a in vol])
    vol = (vol * 255).round().astype(np.uint8)
    return np.ascontiguousarray(vol), valid


def _valid_cached(path):
    try:
        with np.load(path, allow_pickle=False) as z:
            vm, sm, vol = z['view_mask'], z['slice_mask'], z['vol']
            return (str(z['version'].item()) == CFG.CACHE_VERSION and
                    vol.shape == (CFG.N_VIEWS, CFG.N_SLICES, CFG.IMG_SIZE, CFG.IMG_SIZE) and
                    vol.dtype == np.uint8 and vm.shape == (CFG.N_VIEWS,) and
                    sm.shape == (CFG.N_VIEWS, CFG.N_SLICES) and
                    np.array_equal(vm.astype(bool), sm.astype(bool).any(1)) and
                    bool(vm.any()))
    except Exception:
        return False


def cache_study(study: str, mrow, root: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f'{study}.npz'
    if dst.exists() and _valid_cached(dst):
        return True, 'cached'
    vol = np.zeros((CFG.N_VIEWS, CFG.N_SLICES, CFG.IMG_SIZE, CFG.IMG_SIZE), np.uint8)
    view_mask = np.zeros(CFG.N_VIEWS, np.uint8)
    slice_mask = np.zeros((CFG.N_VIEWS, CFG.N_SLICES), np.uint8)
    for vi, slot in enumerate(CFG.VIEWS):
        sid = mrow.get(slot)
        if not isinstance(sid, str) or not sid:
            continue
        v, sm = load_series_volume(root / str(study) / sid, VIEW_SPEC[slot]['plane'])
        if v is not None:
            vol[vi], slice_mask[vi], view_mask[vi] = v, sm, 1
    if not view_mask.any():
        return False, 'no_decodable_view'
    tmp = dst.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, vol=vol, view_mask=view_mask, slice_mask=slice_mask,
                        version=np.asarray(CFG.CACHE_VERSION))
    os.replace(tmp, dst)
    return True, 'written'


print('geometry-aware preprocessing helpers ready:', CFG.CACHE_VERSION)
''')


set_cell(27, r'''
### Preprocessing validation

The smoke montage only runs in the preprocessing stage. Inspect at least one left and one right knee
in each plane. A cache version is embedded in each item, so changing orientation/geometry cannot
silently reuse stale arrays.
''', "markdown")


set_cell(28, r'''
if CFG.RUN_PREPROCESS and CFG.PREPROCESS_SMOKE_ONLY:
    assert manifest is not None and len(manifest)
    _row = manifest.iloc[0]
    _out = CFG.cache_dir / f'shard{CFG.SHARD_ID:02d}'
    _t0 = time.time()
    _ok, _why = cache_study(str(_row.StudyInstanceUID), _row,
                            CFG.comp_dir / 'train_series', _out)
    print('smoke:', _ok, _why, f'{time.time() - _t0:.1f}s')
    if not _ok:
        raise RuntimeError(f'Preprocessing smoke test failed: {_why}')
    with np.load(_out / f'{_row.StudyInstanceUID}.npz') as z:
        vol, vm, sm = z['vol'], z['view_mask'], z['slice_mask']
    fig, axes = plt.subplots(CFG.N_VIEWS, 6, figsize=(14, 2.1 * CFG.N_VIEWS))
    show = np.linspace(0, CFG.N_SLICES - 1, 6).astype(int)
    for v in range(CFG.N_VIEWS):
        for c, s in enumerate(show):
            axes[v, c].imshow(vol[v, s], cmap='gray', vmin=0, vmax=255)
            axes[v, c].axis('off')
            axes[v, c].grid(False)
            if c == 0:
                axes[v, c].set_title(f'{CFG.VIEWS[v]} view={vm[v]} valid={sm[v].sum()}',
                                     loc='left', fontsize=8, color=FIRE)
    plt.tight_layout(); plt.show()
''')


set_cell(29, r'''
if CFG.RUN_PREPROCESS and not CFG.PREPROCESS_SMOKE_ONLY:
    assert manifest is not None and len(manifest)
    if not (0 <= CFG.SHARD_ID < CFG.NUM_SHARDS):
        raise ValueError((CFG.SHARD_ID, CFG.NUM_SHARDS))

    def shard_of(uid):
        return int(hashlib.sha1(str(uid).encode()).hexdigest()[:12], 16) % CFG.NUM_SHARDS

    out_dir = CFG.cache_dir / f'shard{CFG.SHARD_ID:02d}'
    todo = manifest[manifest.StudyInstanceUID.map(shard_of) == CFG.SHARD_ID].reset_index(drop=True)
    print(f'stable shard {CFG.SHARD_ID}/{CFG.NUM_SHARDS}: {len(todo):,} studies -> {out_dir}')
    root = CFG.comp_dir / 'train_series'
    records, t0 = [], time.time()
    for i, r in todo.iterrows():
        try:
            ok, reason = cache_study(str(r.StudyInstanceUID), r, root, out_dir)
        except Exception as e:
            ok, reason = False, f'{type(e).__name__}: {str(e)[:160]}'
        records.append({'StudyInstanceUID': r.StudyInstanceUID, 'ok': int(ok), 'reason': reason})
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f'{i+1:,}/{len(todo):,} | ok={sum(x["ok"] for x in records):,} '
                  f'| {elapsed/60:.1f} min | ETA '
                  f'{elapsed/(i+1)*(len(todo)-i-1)/60:.1f} min', flush=True)

    cache_log = pd.DataFrame(records)
    log_path = CFG.work_dir / f'cache_manifest_shard{CFG.SHARD_ID:02d}.csv'
    cache_log.to_csv(log_path, index=False)
    failures = cache_log[cache_log.ok == 0]
    print('cache complete:', cache_log.ok.sum(), '/', len(cache_log), '| log:', log_path)
    if len(failures):
        display(failures.head(20))
        raise RuntimeError(f'{len(failures)} studies have no valid cache; fix before training.')
''')


set_cell(30, r'''
> **Stage workflow:** run all stable hash shards, publish each `/kaggle/working/rsna_cache_v2/shardXX`
> directory and its manifest, then attach all shards to training. Do not train until coverage is 100%
> for the requested rows.

# Part 3 — leakage-safe report supervision

Folds are constructed first from every observed positive and negative target, grouping studies that
share a retained `PatientID` or an exact normalized non-empty report. For image fold `f`, its report
teacher is fitted only on observed labels outside `f`; it fills missing labels for training rows and
fully unlabelled rows. Validation always uses original observed labels only.

The character/word TF-IDF vocabulary may use all *unlabelled text* (transductive representation
learning), but no held-out target enters the fold-specific classifier. Blank reports use zero rule
features and are never sent to `predict_proba`.
''', "markdown")


# Cell 31 (the multilingual lexicon) is retained. Replace only the blank-report behavior.
_lex = nb['cells'][31]['source']
if isinstance(_lex, list):
    _lex = ''.join(_lex)
_lex = _lex.replace("return {t: np.nan for t in TARGETS}",
                    "return {t: 0.0 for t in TARGETS}")
nb['cells'][31]['source'] = _lex


set_cell(33, r'''
if CFG.RUN_TEACHER:
    from sklearn.metrics import roc_auc_score, average_precision_score

    t0 = time.time()
    rules = pd.DataFrame([rule_score(x) for x in train['Report']], index=train.index)[TARGETS]
    rules = rules.fillna(0.0).astype(np.float32)
    print(f'rules computed in {time.time()-t0:.1f}s | finite: {np.isfinite(rules.values).all()}')

    ev = []
    for c in TARGETS:
        m = observed[c] & has_report
        y, p = train.loc[m, c].astype(int).values, rules.loc[m, c].values
        if len(np.unique(y)) < 2:
            continue
        ev.append({'label': c, 'n': int(m.sum()), 'n_pos': int(y.sum()),
                   'rule_auc': roc_auc_score(y, p),
                   'rule_ap': average_precision_score(y, p)})
    ev = pd.DataFrame(ev).sort_values('rule_auc')
    display(ev.style.background_gradient(cmap=CMAP_SEQ, subset=['rule_auc'], vmin=.5, vmax=1)
                    .format({'rule_auc': '{:.3f}', 'rule_ap': '{:.3f}'}))
''')


set_cell(34, r'''
## 3.1 Patient/report groups and partial-label multilabel folds

Positive and negative observations are separate stratification features. This matters because
missing does not mean negative. A small union-find joins studies by retained `PatientID` and by exact
normalized report duplicates before the greedy group assignment.
''', "markdown")


set_cell(35, r'''
def _normal_report(text):
    if not isinstance(text, str):
        return ''
    text = unicodedata.normalize('NFKC', text).casefold()
    return re.sub(r'\s+', ' ', text).strip()


def collect_patient_ids(studies, series_df, root):
    """Read at most one header per study; return blank when PatientID was stripped."""
    first_series = series_df.drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID') \
                            .SeriesInstanceUID.to_dict()
    out = {}
    for i, uid in enumerate(studies):
        sid = first_series.get(uid)
        pid = ''
        if sid is not None:
            files = list((Path(root) / str(uid) / str(sid)).glob('*.dcm'))
            if files:
                try:
                    ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True,
                                         specific_tags=['PatientID'])
                    pid = str(getattr(ds, 'PatientID', '') or '').strip()
                except Exception:
                    pass
        out[uid] = pid
        if (i + 1) % 1000 == 0:
            print('patient headers:', i + 1, '/', len(studies), flush=True)
    return pd.Series(out)


class UnionFind:
    def __init__(self, n):
        self.p = np.arange(n)
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return int(x)
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a


def make_groups(df, patient_ids):
    uf = UnionFind(len(df))
    pid_values = patient_ids.reindex(df.StudyInstanceUID).fillna('').astype(str)
    # Some anonymizers replace every PatientID with one constant. Do not merge an
    # implausibly large fraction of the dataset into one artificial patient.
    pid_counts = pid_values[pid_values != ''].value_counts()
    oversized = pid_counts[pid_counts > max(25, int(.01 * len(df)))].index
    pid_values = pid_values.mask(pid_values.isin(oversized), '')
    if len(oversized):
        print(f'ignored {len(oversized)} implausibly common PatientID value(s)')
    report_values = df.Report.map(_normal_report).where(lambda s: s.str.len() > 30, '')
    for values in [pid_values.values, report_values.values]:
        seen = {}
        for i, value in enumerate(values):
            if not value:
                continue
            if value in seen:
                uf.union(i, seen[value])
            else:
                seen[value] = i
    return np.asarray([f'g{uf.find(i):07d}' for i in range(len(df))])


def group_multilabel_folds(df, groups, n_splits=5, seed=42):
    """Greedy group split balancing observed positives and observed negatives per target."""
    obs = df[TARGETS].notna().values
    pos = (df[TARGETS].fillna(0).values > .5) & obs
    feat = np.concatenate([pos, obs & ~pos], axis=1).astype(np.int16)
    tmp = pd.DataFrame(feat)
    tmp['group'] = groups
    group_feat = tmp.groupby('group').sum()
    group_size = pd.Series(groups).value_counts().reindex(group_feat.index).values
    labelled_group = group_feat.values.sum(1) > 0

    X = group_feat.values[labelled_group]
    names = group_feat.index.values[labelled_group]
    sizes = group_size[labelled_group]
    total = X.sum(0).astype(float)
    rarity = (X / np.maximum(total, 1)).sum(1)
    rng = np.random.default_rng(seed)
    tie = rng.random(len(X))
    order = np.lexsort((tie, -sizes, -rarity))

    assigned = np.zeros((n_splits, X.shape[1]), float)
    fold_sizes = np.zeros(n_splits, float)
    target = total / n_splits
    target_size = sizes.sum() / n_splits
    group_fold = {}
    for j in order:
        active = X[j] > 0
        deficit = ((target[None, :] - assigned)[:, active] /
                   np.maximum(target[active], 1)).sum(1) if active.any() else np.zeros(n_splits)
        size_need = (target_size - fold_sizes) / max(target_size, 1)
        score = deficit + 0.25 * size_need
        best = np.flatnonzero(score == score.max())
        f = int(best[np.argmin(fold_sizes[best])])
        group_fold[names[j]] = f
        assigned[f] += X[j]
        fold_sizes[f] += sizes[j]

    # Purely unlabelled groups remain -1 and are used in every fold's training set.
    return np.asarray([group_fold.get(g, -1) for g in groups], dtype=np.int8)


def fold_balance(df):
    rows = []
    for f in sorted(x for x in df.fold.unique() if x >= 0):
        for c in TARGETS:
            m = (df.fold == f) & df[c].notna()
            rows.append({'fold': f, 'target': c, 'n': int(m.sum()),
                         'pos': int(df.loc[m, c].sum()),
                         'neg': int(m.sum() - df.loc[m, c].sum())})
    return pd.DataFrame(rows)
''')


set_cell(36, r'''
if CFG.RUN_TEACHER:
    patient_ids = collect_patient_ids(train.StudyInstanceUID.tolist(), train_series,
                                      CFG.comp_dir / 'train_series')
    groups = make_groups(train, patient_ids)
    folds = group_multilabel_folds(train, groups, CFG.folds, CFG.seed)
    fold_df = train[['StudyInstanceUID'] + TARGETS].copy()
    fold_df['group'] = groups
    fold_df['fold'] = folds
    fold_df.to_csv(CFG.teacher_dir / 'folds.csv', index=False)

    bal = fold_balance(fold_df)
    display(bal.pivot(index='fold', columns='target', values='pos'))
    bad = bal[(bal['pos'] == 0) | (bal['neg'] == 0)]
    print('fold sizes (labelled groups):', fold_df[fold_df.fold >= 0].fold.value_counts().sort_index().to_dict())
    print('unlabelled-only rows used by every training fold:', int((fold_df.fold < 0).sum()))
    if len(bad):
        display(bad)
        raise RuntimeError('At least one fold/target lacks a positive or negative; reduce CFG.folds.')
''')


set_cell(37, r'''
## 3.2 Fold-specific TF-IDF teachers

Character 3–5 grams handle multilingual morphology; word 1–2 grams retain phrase context. Balanced
logistic regression is useful for ranking rare findings, then its logits are adjusted back to the
training prior before becoming BCE soft targets. Pseudo-label weights depend on grouped OOF AUC and
sample confidence. Known cells always keep weight 1.0.
''', "markdown")


set_cell(38, r'''
if CFG.RUN_TEACHER:
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    txt = train.Report.fillna('').map(_normal_report).values
    report_rows = has_report.values
    vec_char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=4,
                               max_features=350_000, sublinear_tf=True, dtype=np.float32)
    vec_word = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=3,
                               max_features=200_000, sublinear_tf=True, dtype=np.float32)
    X = sparse.hstack([
        vec_char.fit_transform(txt),
        vec_word.fit_transform(txt),
        sparse.csr_matrix(rules.fillna(0).values.astype(np.float32)),
    ], format='csr')
    assert np.isfinite(X.data).all()
    print('teacher feature matrix:', X.shape, '| nnz:', f'{X.nnz:,}')

    def prior_correct(p, prevalence):
        p = np.clip(np.asarray(p), 1e-5, 1 - 1e-5)
        prevalence = float(np.clip(prevalence, 1e-4, 1 - 1e-4))
        logit = np.log(p / (1 - p)) + np.log(prevalence / (1 - prevalence))
        return (1 / (1 + np.exp(-logit))).astype(np.float32)

    def fit_predict_teacher(X, y, fit_mask, pred_mask, fallback):
        yfit = y[fit_mask].astype(int)
        n_pos, n_neg = int(yfit.sum()), int(len(yfit) - yfit.sum())
        out = np.asarray(fallback, np.float32).copy()
        if min(n_pos, n_neg) < 2:
            return out
        model = LogisticRegression(C=3.0, max_iter=1500, class_weight='balanced',
                                   solver='liblinear', random_state=CFG.seed)
        model.fit(X[fit_mask], yfit)
        raw = model.predict_proba(X[pred_mask])[:, 1]
        out[pred_mask] = prior_correct(raw, yfit.mean())
        return out
''')


set_cell(39, r'''
if CFG.RUN_TEACHER:
    n, T, FOLDS = len(train), len(TARGETS), list(range(CFG.folds))
    teacher_pred = np.full((CFG.folds, n, T), np.nan, np.float32)
    teacher_fit_ok = np.zeros((CFG.folds, T), bool)
    oof = np.full((n, T), np.nan, np.float32)

    for f in FOLDS:
        print(f'\nreport teacher for image fold {f}')
        for ti, c in enumerate(TARGETS):
            fit_mask = (train[c].notna().values & report_rows & (folds != f))
            yfit = train.loc[fit_mask, c].astype(int).values
            teacher_fit_ok[f, ti] = (len(yfit) > 0 and yfit.sum() >= 2 and
                                      (len(yfit) - yfit.sum()) >= 2)
            fallback = rules[c].values.astype(np.float32)
            pred = fit_predict_teacher(X, train[c].fillna(0).values,
                                       fit_mask, report_rows, fallback)
            teacher_pred[f, :, ti] = pred
            val_mask = train[c].notna().values & report_rows & (folds == f)
            oof[val_mask, ti] = pred[val_mask]

    quality = []
    for ti, c in enumerate(TARGETS):
        m = np.isfinite(oof[:, ti]) & train[c].notna().values
        y = train.loc[m, c].astype(int).values
        auc = roc_auc_score(y, oof[m, ti]) if len(np.unique(y)) == 2 else 0.5
        quality.append(auc)
    quality = np.asarray(quality, np.float32)
    qdf = pd.DataFrame({'target': TARGETS,
                        'grouped_oof_auc_diagnostic': quality}).sort_values(
                            'grouped_oof_auc_diagnostic')
    display(qdf.style.background_gradient(cmap=CMAP_SEQ,
                                          subset=['grouped_oof_auc_diagnostic'], vmin=.5, vmax=1)
                 .format({'grouped_oof_auc_diagnostic': '{:.4f}'}))
    qdf.to_csv(CFG.teacher_dir / 'teacher_quality.csv', index=False)

    original = train[TARGETS].values.astype(np.float32)
    obs_np = observed.values
    for f in FOLDS:
        pred = teacher_pred[f]
        target = np.where(obs_np, original, pred).astype(np.float32)
        confidence = np.abs(pred - .5) * 2
        # OOF AUC is diagnostic only: using held-out outcomes to scale this fold's
        # training loss would leak validation information through 12 scalars.
        fit_scale = np.where(teacher_fit_ok[f], 1.0, 0.25)[None, :]
        pseudo_weight = (CFG.pseudo_w * fit_scale *
                         (CFG.pseudo_conf_floor + (1 - CFG.pseudo_conf_floor) * confidence))
        pseudo_allowed = report_rows[:, None] & (folds[:, None] != f)
        weights = np.where(obs_np, 1.0, np.where(pseudo_allowed, pseudo_weight, 0.0)) \
                    .astype(np.float32)
        out = pd.DataFrame({'StudyInstanceUID': train.StudyInstanceUID,
                            'fold': folds, 'group': groups})
        out[TARGETS] = target
        out[WEIGHT_COLS] = weights
        out[OBS_COLS] = obs_np.astype(np.uint8)
        out = out[weights.sum(1) > 0].reset_index(drop=True)
        out.to_csv(CFG.teacher_dir / f'targets_fold{f}.csv', index=False)
        print(f'fold {f}: wrote {len(out):,} supervised rows')

    meta_out = {
        'cache_version': CFG.CACHE_VERSION,
        'folds': CFG.folds,
        'targets': TARGETS,
        'teacher_oof_auc': dict(zip(TARGETS, map(float, quality))),
    }
    with (CFG.teacher_dir / 'teacher_meta.json').open('w') as f:
        json.dump(meta_out, f, indent=2)
    print('teacher artifacts:', CFG.teacher_dir)
''')


set_cell(40, r'''
# Part 4 — target-aware 2.5D student

```
(B,V,D,H,W)
  → adjacent-slice 2.5D channels
  → shared pretrained encoder + backbone-specific normalization
  → add view and normalized depth embeddings
  → 12 learned slice queries: (B,V,T,D)
  → 12 learned view queries:  (B,T,V)
  → 12 target-specific linear heads
```

ACL, MCL, PF OA and Baker's cyst no longer share the same attention distribution. Clinical view priors
only initialize the learned view logits; masks and data decide the final weights. The auxiliary loss is
routed only to plausible target/view pairs. Final epochs use observed hard targets only.
''', "markdown")


set_cell(41, r'''
if CFG.RUN_TRAIN or CFG.RUN_INFER:
    import copy
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    import timm

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if DEVICE != 'cuda':
        raise RuntimeError('Train/infer stage requires a GPU for this configuration.')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    print('device:', DEVICE, '| timm:', timm.__version__, '| torch:', torch.__version__)

    _NEW_AMP = hasattr(torch.amp, 'GradScaler')

    def make_scaler():
        return (torch.amp.GradScaler('cuda', enabled=CFG.amp) if _NEW_AMP
                else torch.cuda.amp.GradScaler(enabled=CFG.amp))

    def autocast():
        return (torch.amp.autocast('cuda', enabled=CFG.amp) if _NEW_AMP
                else torch.cuda.amp.autocast(enabled=CFG.amp))

    def resolve_backbone(candidates, pretrained):
        available = set(timm.list_models())
        weighted = set(timm.list_models(pretrained=True))
        for name in candidates:
            base = name.split('.')[0]
            if pretrained and name in weighted:
                return name
            if not pretrained and (name in available or base in available):
                return name if name in available else base
        for name in candidates:
            base = name.split('.')[0]
            if base in available:
                return base
        raise RuntimeError(f'No configured backbone exists in timm {timm.__version__}: {candidates}')

    BACKBONE = resolve_backbone(CFG.backbone, CFG.pretrained if CFG.RUN_TRAIN else False)
    print('backbone:', BACKBONE)
''')


set_cell(42, r'''
if CFG.RUN_TRAIN or CFG.RUN_INFER:
    _VALIDATED_CACHE_PATHS = set()

    def discover_cache(cache_dirs):
        index = {}
        duplicates = []
        for d in cache_dirs:
            d = Path(d)
            if not d.exists():
                continue
            for p in d.rglob('*.npz'):
                if p.name.endswith('.tmp.npz'):
                    continue
                if p.stem in index:
                    duplicates.append(p.stem)
                else:
                    index[p.stem] = p
        if duplicates:
            print(f'[cache] {len(set(duplicates))} duplicate UIDs; first configured directory wins')
        return index


    class KneeDataset(Dataset):
        def __init__(self, df, cache_dirs, train_mode, hard_only=False, strict=True):
            self.df = df.reset_index(drop=True).copy()
            self.train_mode = train_mode
            self.hard_only = hard_only
            self.index = discover_cache(cache_dirs)
            missing = [str(x) for x in self.df.StudyInstanceUID if str(x) not in self.index]
            if missing and strict:
                raise FileNotFoundError(
                    f'{len(missing)}/{len(self.df)} requested cache items are missing; first: {missing[:5]}. '
                    'Attach every shard. Blank-study fallback is intentionally disabled.')
            self.df = self.df[self.df.StudyInstanceUID.astype(str).isin(self.index)].reset_index(drop=True)
            invalid = []
            for uid in self.df.StudyInstanceUID.astype(str):
                p = self.index[uid]
                key = (str(p), p.stat().st_size, p.stat().st_mtime_ns)
                if key not in _VALIDATED_CACHE_PATHS:
                    if not _valid_cached(p):
                        invalid.append(str(p))
                    else:
                        _VALIDATED_CACHE_PATHS.add(key)
            if invalid:
                raise RuntimeError(f'{len(invalid)} stale/corrupt cache files; first: {invalid[:5]}')
            if hard_only:
                w = self.df[WEIGHT_COLS].values
                self.df = self.df[(w >= .999).any(1)].reset_index(drop=True)
            print(f'dataset: {len(self.df):,} rows | hard_only={hard_only} | cache={len(self.index):,}')

        def __len__(self):
            return len(self.df)

        @staticmethod
        def _augment(vol, view_mask):
            out = np.zeros_like(vol)
            for v in range(CFG.N_VIEWS):
                if not view_mask[v]:
                    continue
                angle = np.random.uniform(-8, 8)
                scale = np.random.uniform(.93, 1.07)
                tx, ty = np.random.uniform(-.04, .04, 2) * CFG.IMG_SIZE
                M = cv2.getRotationMatrix2D((CFG.IMG_SIZE / 2, CFG.IMG_SIZE / 2), angle, scale)
                M[0, 2] += tx
                M[1, 2] += ty
                for d in range(CFG.N_SLICES):
                    out[v, d] = cv2.warpAffine(vol[v, d], M,
                                               (CFG.IMG_SIZE, CFG.IMG_SIZE),
                                               flags=cv2.INTER_LINEAR,
                                               borderMode=cv2.BORDER_CONSTANT)
            x = out.astype(np.float32) / 255.0
            # Contrast/intensity varies independently by sequence.
            gain = np.random.uniform(.88, 1.12, (CFG.N_VIEWS, 1, 1, 1))
            bias = np.random.uniform(-.05, .05, (CFG.N_VIEWS, 1, 1, 1))
            return np.clip(x * gain + bias, 0, 1).astype(np.float32)

        def __getitem__(self, i):
            r = self.df.iloc[i]
            path = self.index[str(r.StudyInstanceUID)]
            with np.load(path, allow_pickle=False) as z:
                vol = z['vol']
                view_mask = z['view_mask'].astype(np.float32)
                slice_mask = z['slice_mask'].astype(np.float32)
            if not view_mask.any():
                raise RuntimeError(f'cache item has no valid view: {path}')
            x = self._augment(vol, view_mask) if self.train_mode else vol.astype(np.float32) / 255.0
            y = r[TARGETS].values.astype(np.float32)
            w = r[WEIGHT_COLS].values.astype(np.float32)
            if self.hard_only:
                w = w * (w >= .999)
            return {
                'x': torch.from_numpy(np.ascontiguousarray(x)),
                'view_mask': torch.from_numpy(view_mask),
                'slice_mask': torch.from_numpy(slice_mask),
                'y': torch.from_numpy(y),
                'w': torch.from_numpy(w),
                'idx': i,
            }


    def masked_softmax(logits, mask, dim=-1):
        """AMP-safe masked softmax; an all-masked row returns exactly zero."""
        mask = torch.broadcast_to(mask.to(device=logits.device, dtype=torch.bool), logits.shape)
        z = logits.float().masked_fill(~mask, -torch.inf)
        zmax = z.amax(dim=dim, keepdim=True)
        zmax = torch.where(torch.isfinite(zmax), zmax, torch.zeros_like(zmax))
        p = torch.exp(z - zmax) * mask.to(z.dtype)
        den = p.sum(dim=dim, keepdim=True)
        p = torch.where(den > 0, p / den.clamp_min(torch.finfo(p.dtype).tiny),
                        torch.zeros_like(p))
        return p.to(logits.dtype)


    # Rows=targets, columns=CFG.VIEWS. Used for initialization and auxiliary routing only.
    _R = {
        'ACL':             [1, 1, 1, 0, 0],
        'MCL':             [0, 0, 1, 1, 0],
        'Medial Meniscus': [1, 1, 1, 1, 0],
        'Lateral Meniscus':[1, 1, 1, 1, 0],
        'Medial OA':       [1, 1, 1, 1, 1],
        'Lateral OA':      [1, 1, 1, 1, 1],
        'PF OA':           [1, 1, 0, 0, 1],
        'Effusion':        [1, 0, 1, 0, 1],
        'Synovitis':       [1, 0, 1, 0, 1],
        "Baker's":        [1, 1, 0, 0, 1],
        'Contusion':       [1, 0, 1, 0, 1],
        'Fracture':        [1, 1, 1, 1, 1],
    }
    TARGET_VIEW_ROUTE = torch.tensor([_R[c] for c in TARGETS], dtype=torch.float32)


    class KneeNet(nn.Module):
        def __init__(self, backbone=BACKBONE, pretrained=False):
            super().__init__()
            self.backbone_name = backbone
            self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0,
                                         in_chans=3, drop_rate=CFG.drop_rate,
                                         drop_path_rate=CFG.drop_path_rate)
            if CFG.grad_ckpt:
                try:
                    self.enc.set_grad_checkpointing(True)
                except Exception:
                    pass
            C, T, V = self.enc.num_features, len(TARGETS), CFG.N_VIEWS
            try:
                pcfg = timm.data.resolve_model_data_config(self.enc)
            except Exception:
                pcfg = getattr(self.enc, 'pretrained_cfg', {}) or {}
            mean = torch.tensor(pcfg.get('mean', (0.485, 0.456, 0.406))).view(1, 3, 1, 1)
            std = torch.tensor(pcfg.get('std', (0.229, 0.224, 0.225))).view(1, 3, 1, 1)
            self.register_buffer('input_mean', mean, persistent=True)
            self.register_buffer('input_std', std, persistent=True)

            self.view_emb = nn.Parameter(torch.randn(V, C) * .02)
            self.depth_mlp = nn.Sequential(nn.Linear(1, C), nn.Tanh(), nn.Linear(C, C))
            self.slice_query = nn.Parameter(torch.randn(T, C) * .02)
            self.view_query = nn.Parameter(torch.randn(T, C) * .02)
            init_route = torch.where(TARGET_VIEW_ROUTE > 0, 0.75, -0.75)
            self.view_prior = nn.Parameter(init_route)
            self.norm = nn.LayerNorm(C)
            self.drop = nn.Dropout(CFG.drop_rate)
            self.head_weight = nn.Parameter(torch.randn(T, C) * (1 / math.sqrt(C)))
            self.head_bias = nn.Parameter(torch.zeros(T))

        @staticmethod
        def to_25d(x):
            prev = torch.cat([x[:, :, :1], x[:, :, :-1]], 2)
            nxt = torch.cat([x[:, :, 1:], x[:, :, -1:]], 2)
            return torch.stack([prev, x, nxt], 3)

        def forward(self, x, view_mask, slice_mask):
            B, V, D, H, W = x.shape
            x = self.to_25d(x).reshape(B * V * D, 3, H, W)
            x = (x - self.input_mean) / self.input_std
            valid = ((slice_mask > 0) & (view_mask[:, :, None] > 0)).reshape(-1)
            valid_idx = valid.nonzero(as_tuple=False).squeeze(1)
            if not len(valid_idx):
                raise RuntimeError('A batch contains no valid DICOM slices')
            encoded = self.enc(x.index_select(0, valid_idx))
            flat = encoded.new_zeros((B * V * D, self.enc.num_features))
            flat.index_copy_(0, valid_idx, encoded)
            f = flat.reshape(B, V, D, -1)
            depth = torch.linspace(-1, 1, D, device=f.device, dtype=f.dtype).view(D, 1)
            f = self.norm(f + self.view_emb[None, :, None, :] +
                          self.depth_mlp(depth)[None, None, :, :])

            # Target-specific attention over depth: B,V,T,D.
            s = torch.einsum('bvdc,tc->bvtd', f, self.slice_query) / math.sqrt(f.shape[-1])
            a_slice = masked_softmax(s.permute(0, 1, 3, 2),
                                     slice_mask[:, :, None, :], dim=-1)
            fv = torch.einsum('bvtd,bvdc->bvtc', a_slice, f)

            # Target-specific attention over sequence slots: B,T,V.
            sv = torch.einsum('bvtc,tc->btv', fv, self.view_query) / math.sqrt(f.shape[-1])
            sv = sv + self.view_prior[None, :, :]
            a_view = masked_softmax(sv, view_mask[:, None, :], dim=-1)
            ft = torch.einsum('btv,bvtc->btc', a_view, fv)
            logits = (self.drop(ft) * self.head_weight[None]).sum(-1) + self.head_bias
            aux = (self.drop(fv) * self.head_weight[None, None]).sum(-1) + \
                  self.head_bias[None, None]
            return logits, aux, a_view, a_slice


    @torch.no_grad()
    def ema_update(ema_model, model, decay, num_updates):
        # Avoid an initial-model bias when a fold has only a few hundred updates.
        decay = min(float(decay), (1 + num_updates) / (10 + num_updates))
        ema_state, state = ema_model.state_dict(), model.state_dict()
        for key, value in ema_state.items():
            src = state[key].detach()
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(src, alpha=1 - decay)
            else:
                value.copy_(src)


    def masked_multitask_loss(logits, aux, y, weight, view_mask, pos_weight):
        """Macro target reduction; pseudo reliability is not renormalized away."""
        valid = weight > 0
        raw = F.binary_cross_entropy_with_logits(logits, y, reduction='none',
                                                 pos_weight=pos_weight)
        denom = valid.sum(0)
        main_per_target = (raw * weight).sum(0) / denom.clamp_min(1)
        present = denom > 0
        main = main_per_target[present].mean() if present.any() else logits.sum() * 0

        aux_raw = F.binary_cross_entropy_with_logits(aux, y[:, None, :].expand_as(aux),
                                                     reduction='none', pos_weight=pos_weight)
        route = TARGET_VIEW_ROUTE.T.to(aux.device)[None]
        routed_view = view_mask[:, :, None] * route
        per_study_aux = (aux_raw * routed_view).sum(1) / routed_view.sum(1).clamp_min(1)
        aux_valid = valid & (routed_view.sum(1) > 0)
        aux_denom = aux_valid.sum(0)
        aux_per_target = (per_study_aux * weight * aux_valid).sum(0) / aux_denom.clamp_min(1)
        aux_present = aux_denom > 0
        aux_loss = (aux_per_target[aux_present].mean() if aux_present.any()
                    else logits.sum() * 0)
        return main + CFG.aux_w * aux_loss, main.detach(), aux_loss.detach()


    print('dataset and target-aware model defined')
''')


set_cell(43, r'''
### Model smoke test

The smoke test checks exact output shapes, finite loss, normalization buffers, and zero attention for
masked slices/views. It runs for both training and inference stages before loading a long job.
''', "markdown")


set_cell(44, r'''
if CFG.RUN_TRAIN or CFG.RUN_INFER:
    _m = KneeNet(pretrained=False).to(DEVICE)
    _B = min(CFG.batch_size, 2)
    _x = torch.rand(_B, CFG.N_VIEWS, CFG.N_SLICES, CFG.IMG_SIZE, CFG.IMG_SIZE, device=DEVICE)
    _vm = torch.ones(_B, CFG.N_VIEWS, device=DEVICE)
    _vm[:, -1] = 0
    _sm = torch.ones(_B, CFG.N_VIEWS, CFG.N_SLICES, device=DEVICE)
    _sm[:, -1] = 0
    _sm[:, 0, -2:] = 0
    _y = (torch.rand(_B, len(TARGETS), device=DEVICE) > .7).float()
    _w = torch.ones_like(_y)
    with autocast():
        _logit, _aux, _av, _as = _m(_x, _vm, _sm)
        _loss, _, _ = masked_multitask_loss(_logit, _aux, _y, _w, _vm,
                                            torch.ones(len(TARGETS), device=DEVICE))
    _loss.backward()
    assert _logit.shape == (_B, len(TARGETS))
    assert _aux.shape == (_B, CFG.N_VIEWS, len(TARGETS))
    assert _av.shape == (_B, len(TARGETS), CFG.N_VIEWS)
    assert torch.isfinite(_loss)
    assert float(_av[:, :, -1].abs().max()) < 1e-5
    assert float(_as[:, -1].abs().max()) < 1e-5
    print(f'smoke ok | params={sum(p.numel() for p in _m.parameters())/1e6:.1f}M '
          f'| loss={float(_loss):.4f} | peak={torch.cuda.max_memory_allocated()/2**30:.2f} GiB')
    del _m, _x, _vm, _sm, _y, _w, _logit, _aux, _av, _as, _loss
    gc.collect(); torch.cuda.empty_cache()
''')


set_cell(45, r'''
if CFG.RUN_TRAIN:
    from sklearn.metrics import roc_auc_score

    cache_dirs = [CFG.cache_dir] + list(CFG.extra_cache_dirs)

    def load_fold_targets(fold):
        p = find_artifact(f'targets_fold{fold}.csv', CFG.teacher_dir, CFG.extra_teacher_dirs)
        df = pd.read_csv(p)
        required = {'StudyInstanceUID', 'fold', *TARGETS, *WEIGHT_COLS}
        if not required.issubset(df.columns):
            raise ValueError(f'{p} is missing {sorted(required - set(df.columns))}')
        return df

    def macro_auc(y, p, observed_mask):
        per = {}
        for i, c in enumerate(TARGETS):
            m = observed_mask[:, i].astype(bool)
            yi = y[m, i].astype(int)
            per[c] = (np.nan if len(np.unique(yi)) < 2
                      else roc_auc_score(yi, p[m, i]))
        if any(not np.isfinite(v) for v in per.values()):
            raise RuntimeError(f'Invalid per-target validation AUC: {per}')
        return float(np.mean(list(per.values()))), per

    def make_sampler(df, hard_only):
        y = df[TARGETS].values
        w = df[WEIGHT_COLS].values
        if hard_only:
            active = w >= .999
        else:
            active = w > 0
        positive = (y > .6) & active
        prevalence = positive.sum(0) / np.maximum(active.sum(0), 1)
        rarity = np.sqrt(1 / np.maximum(prevalence, .01))
        sw = 1 + (positive * rarity[None]).max(1)
        sw = np.clip(sw, 1, 6)
        return WeightedRandomSampler(torch.as_tensor(sw, dtype=torch.double),
                                     num_samples=len(sw), replacement=True)

    def make_loader(ds, batch_size, train_mode):
        sampler = make_sampler(ds.df, ds.hard_only) if train_mode else None
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, shuffle=False,
                          num_workers=CFG.num_workers, pin_memory=True, drop_last=False,
                          persistent_workers=CFG.num_workers > 0)

    def validation(model, loader):
        model.eval(); P, Y, O = [], [], []
        with torch.inference_mode():
            for b in loader:
                with autocast():
                    logits, _, _, _ = model(b['x'].to(DEVICE, non_blocking=True),
                                            b['view_mask'].to(DEVICE, non_blocking=True),
                                            b['slice_mask'].to(DEVICE, non_blocking=True))
                P.append(logits.float().sigmoid().cpu().numpy())
                Y.append(b['y'].numpy())
                O.append((b['w'].numpy() >= .999).astype(np.uint8))
        return macro_auc(np.concatenate(Y), np.concatenate(P), np.concatenate(O))

    def train_fold(fold):
        df = load_fold_targets(fold)
        tr = df[df.fold != fold].reset_index(drop=True)  # includes fold=-1 pseudo-only studies
        va = df[df.fold == fold].reset_index(drop=True)
        if not len(va):
            raise RuntimeError(f'fold {fold} has no validation rows')

        ds_mix = KneeDataset(tr, cache_dirs, train_mode=True, hard_only=False, strict=True)
        ds_hard = KneeDataset(tr, cache_dirs, train_mode=True, hard_only=True, strict=True)
        ds_val = KneeDataset(va, cache_dirs, train_mode=False, hard_only=False, strict=True)
        dl_mix = make_loader(ds_mix, CFG.batch_size, True)
        dl_hard = make_loader(ds_hard, CFG.batch_size, True)
        dl_val = make_loader(ds_val, CFG.batch_size, False)

        hard = tr[WEIGHT_COLS].values >= .999
        y = tr[TARGETS].values
        pos = ((y > .5) & hard).sum(0)
        neg = ((y <= .5) & hard).sum(0)
        pos_weight = np.sqrt((neg + 1) / (pos + 1)).clip(1, CFG.max_pos_weight)
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=DEVICE)
        print(f'\n=== fold {fold} | mix={len(ds_mix):,} hard={len(ds_hard):,} val={len(ds_val):,} ===')
        print('capped sqrt pos_weight:', dict(zip(TARGETS, pos_weight.cpu().numpy().round(2))))

        model = KneeNet(pretrained=CFG.pretrained).to(DEVICE)
        ema = copy.deepcopy(model).eval()
        for p in ema.parameters():
            p.requires_grad_(False)
        bb = [p for n, p in model.named_parameters() if n.startswith('enc.')]
        hd = [p for n, p in model.named_parameters() if not n.startswith('enc.')]
        opt = torch.optim.AdamW([{'params': bb, 'lr': CFG.backbone_lr},
                                 {'params': hd, 'lr': CFG.lr}], weight_decay=CFG.wd)

        loaders = [dl_mix] * CFG.mixed_epochs + [dl_hard] * CFG.hard_epochs
        updates_per_epoch = [math.ceil(len(dl) / CFG.accum) for dl in loaders]
        total_updates = sum(updates_per_epoch)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[CFG.backbone_lr, CFG.lr], total_steps=total_updates,
            pct_start=.10, div_factor=10, final_div_factor=100)
        scaler = make_scaler()
        history, best, update_count = [], -np.inf, 0

        for epoch, dl in enumerate(loaders):
            model.train(); opt.zero_grad(set_to_none=True)
            running, n_batches, t0 = 0.0, 0, time.time()
            group_start = 0
            for it, b in enumerate(dl):
                # Scale by the actual final accumulation group size, not always CFG.accum.
                group_end = min(group_start + CFG.accum, len(dl))
                group_size = group_end - group_start
                x = b['x'].to(DEVICE, non_blocking=True)
                vm = b['view_mask'].to(DEVICE, non_blocking=True)
                sm = b['slice_mask'].to(DEVICE, non_blocking=True)
                yy = b['y'].to(DEVICE, non_blocking=True)
                ww = b['w'].to(DEVICE, non_blocking=True)
                with autocast():
                    logits, aux, _, _ = model(x, vm, sm)
                    loss, main_loss, aux_loss = masked_multitask_loss(
                        logits, aux, yy, ww, vm, pos_weight)
                    scaled_loss = loss / group_size
                scaler.scale(scaled_loss).backward()
                running += float(loss.detach()); n_batches += 1

                do_step = (it + 1 == group_end)
                if do_step:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                    sched.step(); update_count += 1
                    ema_update(ema, model, CFG.ema_decay, update_count)
                    group_start = it + 1

            auc, per = validation(ema, dl_val)
            row = {'epoch': epoch, 'phase': 'mixed' if epoch < CFG.mixed_epochs else 'hard',
                   'loss': running / max(n_batches, 1), 'auc': auc, **per}
            history.append(row)
            print(f'epoch {epoch:02d} {row["phase"]:<5} | loss={row["loss"]:.4f} '
                  f'| macro AUC={auc:.4f} | {time.time()-t0:.0f}s', flush=True)
            if auc > best:
                best = auc
                payload = {
                    'state_dict': ema.state_dict(), 'backbone': BACKBONE,
                    'views': CFG.VIEWS, 'targets': TARGETS,
                    'img_size': CFG.IMG_SIZE, 'n_slices': CFG.N_SLICES,
                    'cache_version': CFG.CACHE_VERSION, 'fold': fold,
                    'epoch': epoch, 'val_auc': auc, 'per_target_auc': per,
                }
                torch.save(payload, CFG.ckpt_dir / f'knee_cv2_f{fold}.pt')

        hist = pd.DataFrame(history)
        hist.to_csv(CFG.ckpt_dir / f'history_f{fold}.csv', index=False)
        print(f'fold {fold} best macro AUC: {best:.4f}')
        del model, ema, opt, scaler, dl_mix, dl_hard, dl_val
        gc.collect(); torch.cuda.empty_cache()
        return best

    train_folds = list(range(CFG.folds)) if CFG.TRAIN_FOLDS is None else list(CFG.TRAIN_FOLDS)
    scores = [train_fold(f) for f in train_folds]
    print('CV folds:', train_folds, '| scores:', np.round(scores, 4),
          '| mean:', float(np.mean(scores)))
''')


set_cell(46, r'''
# Part 5 — inference and submission

All fold models are loaded together and each decoded test batch is passed through them sequentially.
This avoids decoding every DICOM once per checkpoint. Fold outputs are percentile-ranked per target
before averaging, matching the rank-based metric while reducing calibration differences.

The cell validates checkpoint preprocessing metadata, requires at least one checkpoint, requires every
study to have a decoded view, and refuses non-finite or constant target predictions.
''', "markdown")


set_cell(47, r'''
if CFG.RUN_INFER:
    class TestDataset(Dataset):
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
                v, s = load_series_volume(self.root / str(r.StudyInstanceUID) / sid,
                                          VIEW_SPEC[slot]['plane'])
                if v is not None:
                    vol[vi], sm[vi], vm[vi] = v, s, 1
            if not vm.any():
                raise RuntimeError(f'No decodable selected view for test study {r.StudyInstanceUID}')
            return {'x': torch.from_numpy(vol.astype(np.float32) / 255.0),
                    'view_mask': torch.from_numpy(vm),
                    'slice_mask': torch.from_numpy(sm), 'idx': i}

    test_manifest = test.merge(
        build_manifest(test_series, CFG.comp_dir / 'test_series', count_files=True),
        on='StudyInstanceUID', how='left', validate='one_to_one')
    if test_manifest[CFG.VIEWS].isna().all(axis=1).any():
        bad = test_manifest.loc[test_manifest[CFG.VIEWS].isna().all(axis=1), 'StudyInstanceUID'].tolist()
        raise RuntimeError(f'Test studies have no selected series: {bad[:10]}')

    ckpt_paths = []
    for d in [CFG.ckpt_dir] + list(CFG.extra_ckpt_dirs):
        if Path(d).exists():
            ckpt_paths.extend(Path(d).rglob('knee_cv2_f*.pt'))
    ckpt_paths = sorted(set(ckpt_paths))
    if not ckpt_paths:
        raise FileNotFoundError('No knee_cv2_f*.pt checkpoints. Attach weights and set CFG.extra_ckpt_dirs.')
    print('checkpoints:', [str(x) for x in ckpt_paths])

    models = []
    for path in ckpt_paths:
        try:
            ck = torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:  # torch < 2.0 compatibility
            ck = torch.load(path, map_location='cpu')
        expected = {'views': CFG.VIEWS, 'targets': TARGETS, 'img_size': CFG.IMG_SIZE,
                    'n_slices': CFG.N_SLICES, 'cache_version': CFG.CACHE_VERSION}
        mismatch = {k: (ck.get(k), v) for k, v in expected.items() if ck.get(k) != v}
        if mismatch:
            raise ValueError(f'Checkpoint/preprocessing mismatch in {path}: {mismatch}')
        model = KneeNet(backbone=ck['backbone'], pretrained=False).to(DEVICE)
        model.load_state_dict(ck['state_dict'], strict=True)
        model.eval()
        models.append(model)
    print(f'loaded {len(models)} models | GPU weights: '
          f'{torch.cuda.memory_allocated()/2**30:.2f} GiB')

    ds = TestDataset(test_manifest, CFG.comp_dir / 'test_series')
    dl = DataLoader(ds, batch_size=CFG.infer_batch_size, shuffle=False,
                    num_workers=CFG.num_workers, pin_memory=True,
                    persistent_workers=CFG.num_workers > 0)
    fold_pred = np.zeros((len(models), len(ds), len(TARGETS)), np.float32)
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
                fold_pred[mi, idx] = logits.float().sigmoid().cpu().numpy()
            if (step + 1) % 25 == 0:
                elapsed = time.time() - t0
                print(f'{min((step+1)*CFG.infer_batch_size, len(ds))}/{len(ds)} '
                      f'| {elapsed/60:.1f} min', flush=True)

    ranked = np.stack([
        pd.DataFrame(p, columns=TARGETS).rank(method='average', pct=True).values
        for p in fold_pred
    ]).mean(0).astype(np.float32)
    if not np.isfinite(ranked).all():
        raise RuntimeError('Non-finite ensemble prediction')
    constant = [TARGETS[i] for i in range(len(TARGETS)) if np.unique(ranked[:, i]).size < 2]
    if constant:
        raise RuntimeError(f'Constant target predictions indicate broken inference: {constant}')

    pred_df = pd.DataFrame(ranked, columns=TARGETS)
    pred_df.insert(0, 'StudyInstanceUID', test_manifest.StudyInstanceUID.values)
    sub = sample_sub[['StudyInstanceUID']].merge(pred_df, on='StudyInstanceUID', how='left',
                                                 validate='one_to_one')
    sub = sub[sample_sub.columns]
    if sub[TARGETS].isna().any().any():
        raise RuntimeError('Submission merge produced missing predictions')
    sub.to_csv('submission.csv', index=False)
    print(f'submission.csv written: {sub.shape} | total inference {(time.time()-t0)/60:.1f} min')
    display(sub.head())
else:
    print(f'STAGE={CFG.STAGE!r}: submission.csv was intentionally not written.')
''')


set_cell(48, r'''
# Score-improvement ladder

Run ablations against the same grouped folds; do not assume every larger model helps.

1. **Make CV trustworthy first:** inspect per-target fold positives/negatives, report-teacher OOF AUC,
   cache coverage, and left/right orientation montages.
2. **Baseline ablation:** CV1 attention versus CV2 target-aware attention with identical cache/backbone.
3. **Resolution/depth:** compare 256×12 with 288×16. Meniscus/OA often benefit, but measure runtime.
4. **Teacher:** replace TF-IDF with an offline multilingual encoder, still cross-fitted by the same
   groups. Never send competition reports to an external API.
5. **Series MIL:** keep the top two qualifying series per slot and add a series-level attention layer.
6. **Backbone/seed ensemble:** train a complementary ConvNeXt/MaxViT or second seed and rank-blend.
7. **Efficiency entry:** use EfficientNetV2-B0, 192×10, fewer folds and two/three core slots; time it
   on a full-size synthetic/test manifest because the example test has only three studies.

Threshold tuning cannot improve ROC AUC. Calibration is useful only when it improves cross-model
blending; any monotonic recalibration leaves an individual target's AUC unchanged.
''', "markdown")


nb.setdefault('metadata', {})['cv2_notes'] = {
    'source': SOURCE.name,
    'builder': Path(__file__).name,
    'cache_version': 'cv2-orient-roi-letterbox-256-12-v1',
}
with DEST.open('w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write('\n')
print(f'wrote {DEST} ({len(nb["cells"])} cells)')
