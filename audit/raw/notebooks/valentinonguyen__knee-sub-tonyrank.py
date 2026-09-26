#### CELL 2
import hashlib
import json
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '4')
from pathlib import Path
import subprocess
import sys
import time
import pandas as pd

ASSET_SLUG = 'rsna-knee-bend-dinov3-0917-repro-assets'
ASSET = next((p for p in [
    Path('/kaggle/input/datasets/tonylica') / ASSET_SLUG,
    Path('/kaggle/input') / ASSET_SLUG,
] if (p / 'bundle_manifest.json').is_file()), None)
ROOT = next((p for p in [
    Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
    Path('/kaggle/input/rsna-knee-abnormality-detection'),
] if (p / 'test.csv').is_file()), None)
assert ASSET is not None, 'Attach the consolidated reproduction dataset'
assert ROOT is not None and (ROOT / 'test_series').is_dir(), 'Attach the competition data'

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

manifest_path = ASSET / 'bundle_manifest.json'
observed_manifest_sha256 = sha256_file(manifest_path)
assert observed_manifest_sha256 == '3b6279ab50ca8c3ce100644eadda9bc14e0359c05120990e28a43edc6a5364a8', f'Dataset version or manifest mismatch: {observed_manifest_sha256}'
manifest = json.loads(manifest_path.read_text())
for record in manifest['files']:
    path = ASSET / record['path']
    assert path.is_relative_to(ASSET) and path.is_file(), record['path']
    assert path.stat().st_size == record['bytes'], f"Size mismatch: {record['path']}"
    assert sha256_file(path) == record['sha256'], f"Hash mismatch: {record['path']}"
print(f"Verified {manifest['file_count']} files; all models come from public Baseline v15.")
os.environ.update(RSNA_ASSET_ROOT=str(ASSET), RSNA_COMPETITION_ROOT=str(ROOT),
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
WORK = Path('/kaggle/working')
(WORK / 'branches').mkdir(exist_ok=True)
# Remove only a stale primary artifact from a previous interactive run.
(WORK / 'submission.csv').unlink(missing_ok=True)
STAGE_TIMES = {}

def run_stage(script):
    started = time.monotonic()
    subprocess.run([sys.executable, '-u', str(ASSET / 'pipeline' / script)],
                   cwd=WORK, check=True, env=dict(os.environ))
    STAGE_TIMES[script] = round(time.monotonic() - started, 3)
    print(f"Completed {script} in {STAGE_TIMES[script]:.1f}s")
    (WORK / 'stage_times.json').write_text(json.dumps(STAGE_TIMES, indent=2))

#### CELL 4
run_stage('transformer_rad.py')
#### CELL 6
run_stage('public_raptor.py')
#### CELL 8
run_stage('residual_coat.py')
#### CELL 10
run_stage('blend.py')
#### CELL 12
submission = pd.read_csv('/kaggle/working/submission.csv', dtype={'StudyInstanceUID': str})
print(f'Ready: {len(submission)} studies, {len(submission.columns)-1} targets')
display(submission.head())
display(json.loads(Path('/kaggle/working/run_receipt.json').read_text()))
