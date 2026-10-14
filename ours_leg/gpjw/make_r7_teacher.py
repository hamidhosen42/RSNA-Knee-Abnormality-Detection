"""Teacher predictions of heliosli's R7 compact model (bundle 8c7b666e27dc0e8f) on the TRAIN studies, in shards.

heliosli's R7 setup cell (cell 4 of 0-957-latest-dataset-score-in-one-click v1: code/wheels/sources/bundle checks) runs
verbatim.  Its controller (probe_dual.py) and worker (compact_infer.py) take --data/--split and read <split>.csv,
<split>_series/ and sample_submission.csv, and they insist that sample_submission lists exactly the studies of
<split>.csv.  So each shard builds a small view folder: train.csv = this shard's rows, train_series(.csv) = symlinks to
the competition files, sample_submission.csv = this shard's ids.  Studies the worker could not score (or did not reach
before the deadline) are dropped from the output, never filled.

Output: /kaggle/working/r7_train_shard<k>.csv (probabilities of the scored studies) + worker logs.

    python make_r7_teacher.py [--shards=2] [--owner=hosen42]  ->  kaggle/r7teacher_<k>[_owner]/
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OWNER = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--owner=")), "hosen42")
N = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--shards=")), "2"))
DEADLINE_H = 11.0

cells = json.load(open(os.path.join(ROOT, "public_notebooks", "0.956_heliosli__0-957-latest-dataset-score-in-one-click.ipynb")))["cells"]
setup = "".join(cells[4]["source"])
assert setup.startswith("ARM = 'E_lrprior'\nBUNDLE_ID = '8c7b666e27dc0e8f'") and "T0 = NB_T0" in setup
assert "display(" not in setup and "\n%" not in setup and "\n!" not in setup

RUN = r'''

# ======================================================================= [r7-teacher] score the TRAIN studies (ours)
import csv, hashlib
SHARD, N_SHARDS = __SHARD__, __N__
VIEW = TMP / 'train_view'
shutil.rmtree(VIEW, ignore_errors=True)
VIEW.mkdir(parents=True)
with open(DATA / 'train.csv', encoding='utf-8', newline='') as fh:
    _rd = csv.DictReader(fh)
    _fields, _rows = _rd.fieldnames, list(_rd)
_mine = [r for r in _rows if int(hashlib.md5(r['StudyInstanceUID'].encode()).hexdigest(), 16) % N_SHARDS == SHARD]
with open(VIEW / 'train.csv', 'w', encoding='utf-8', newline='') as fh:
    _w = csv.DictWriter(fh, fieldnames=_fields)
    _w.writeheader()
    _w.writerows(_mine)
os.symlink(DATA / 'train_series', VIEW / 'train_series')
if (DATA / 'train_series.csv').is_file():
    os.symlink(DATA / 'train_series.csv', VIEW / 'train_series.csv')
with open(DATA / 'sample_submission.csv', encoding='utf-8') as fh:
    _header = next(csv.reader(fh))
with open(VIEW / 'sample_submission.csv', 'w', encoding='utf-8', newline='') as fh:
    _w = csv.writer(fh)
    _w.writerow(_header)
    for r in _mine:
        _w.writerow([r['StudyInstanceUID']] + ['0.5'] * (len(_header) - 1))
print('R7_TEACHER_SHARD', SHARD, 'of', N_SHARDS, '| studies', len(_mine), 'of', len(_rows), flush=True)

deadline = NB_T0 + __DEADLINE_H__ * 3600
env = dict(os.environ, PYTHONPATH=str(CODE) + os.pathsep + os.environ.get('PYTHONPATH', ''), PYTHONDONTWRITEBYTECODE='1')
pred = TMP / 'prediction_train'
shutil.rmtree(pred, ignore_errors=True)   # never mix in a stale prediction folder
rc = subprocess.run([sys.executable, '-u', str(SRC / 'probe_dual.py'), '--data', str(VIEW), '--split', 'train',
                     '--code', str(CODE), '--bundle', str(BUNDLE), '--arm', ARM, '--output', str(pred),
                     '--amp', 'auto', '--loader-workers', '2', '--deadline', str(deadline),
                     '--worker', WORKER, '--worker-args', json.dumps(WORKER_ARGS),
                     '--max-fallback-fraction', '1.0'], env=env).returncode
for log in sorted(pred.glob('worker*/log.txt')):
    shutil.copyfile(log, WORK / (log.parent.name + '_log.txt'))
if rc != 0:
    raise RuntimeError('controller failed with exit code %d' % rc)
_scored = {}
for f in pred.glob('worker*/predictions.csv'):
    with open(f, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            try:
                v = [float(r[L]) for L in _header[1:]]
            except (KeyError, ValueError, TypeError):
                continue
            if all(0. <= x <= 1. for x in v):
                _scored[r['StudyInstanceUID']] = v
_out = WORK / ('r7_train_shard%d.csv' % SHARD)
with open(_out, 'w', encoding='utf-8', newline='') as fh:
    _w = csv.writer(fh)
    _w.writerow(_header)
    for r in _mine:
        u = r['StudyInstanceUID']
        if u in _scored:
            _w.writerow([u] + ['%.6g' % x for x in _scored[u]])
print('R7_TEACHER_DONE shard', SHARD, '| scored', len(_scored), 'of', len(_mine), '| hours %.2f' % ((time.time() - NB_T0) / 3600),
      '->', _out, flush=True)
'''

for k in range(N):
    src = "import time\nNB_T0 = time.time()   # [r7-teacher] heliosli's setup counts its deadline from here\n\n" + setup + \
          RUN.replace("__SHARD__", str(k)).replace("__N__", str(N)).replace("__DEADLINE_H__", repr(DEADLINE_H))
    compile(src, "r7_teacher.py", "exec")
    d = os.path.join(HERE, "..", "kaggle", f"r7teacher_{k}" + ("" if OWNER == "hosen42" else f"_{OWNER}"))
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "r7_teacher.py"), "w").write(src)
    slug = f"rsna-knee-r7-teacher-{k}"
    meta = {"id": f"{OWNER}/{slug}", "title": slug, "code_file": "r7_teacher.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
            "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f",
            "docker_image_pinning_type": "original",
            "dataset_sources": ["heliosli/rsna-knee-compact-r7cnxfsss", "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"],
            "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": []}
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
    print("wrote", d)
