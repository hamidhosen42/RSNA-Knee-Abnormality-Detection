#### CELL 1
import os
for _thread_env in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_thread_env, '4')
os.environ.setdefault('CUDNN_CONV_WSCAP_DBG', '1024')

def _d4_check_runtime(rt, artifact_root):
    import hashlib
    import json
    from pathlib import Path
    import torch
    root = Path(artifact_root)
    manifest_path = root / 'coatnet_pairfilm_manifest.json'
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if digest != '7ada0605bca6b0530569c6454e988ace479606a3328ed591d090e5764fea661d':
        raise RuntimeError('D4 reference manifest identity changed')
    expected = json.loads(manifest_path.read_text())
    manifest, paths = rt.validate_artifact_root(root, verify_hashes=True)
    if manifest != expected or len(paths) != 1 or rt.N_MODELS != 1:
        raise RuntimeError('D4 must load exactly the reference SWA model')
    if Path(paths[0]).name != 'coatnet_d4_parent_rank8_swa2.pt':
        raise RuntimeError('D4 must use its reference rank-8 parent checkpoint')
    if manifest['slice_depth']['adapter_sha256'] != 'b7c48b19997bc7ecb7e997f8dcf491ec5701ff917d1dac4b369eb85ac97ae89a':
        raise RuntimeError('D4 reference adapter identity changed')
    if rt.GPU_BATCH_STUDIES != 2 or rt.BACKBONE_MICRO_IMAGES != 8:
        raise RuntimeError('D4 reference batching changed')
    if rt.CUDNN_BENCHMARK is not False:
        raise RuntimeError('D4 reference convolution policy changed')
    if rt.base.cv2.__version__ != '4.12.0' or rt.parent.timm.__version__ != '1.0.22':
        raise RuntimeError('D4 reference dependency versions changed')
    if torch.cuda.device_count() != 2:
        raise RuntimeError('D4 requires exactly two T4 GPUs')
    names = [torch.cuda.get_device_name(i) for i in range(2)]
    if not all(('T4' in name for name in names)):
        raise RuntimeError(f'D4 reference device mismatch: {names}')
    return (manifest, paths)

def _d4_check_outputs(rt, manifest, receipt, output, competition):
    import hashlib
    from pathlib import Path
    import numpy as np
    import pandas as pd

    def need(condition, message):
        if not condition:
            raise RuntimeError('D4 release check: ' + message)
    need(receipt['status'] == rt.SUBMISSION_STATUS, 'unsuccessful status')
    expected_fields = {'models': 1, 'fallback_studies': 0, 'dicom_preparations_per_study': 1, 'model_passes_per_study': 1, 'rank_after_probability_average': True, 'canonical_sagittal': True, 'common_physical_triplet_warp': False, 'resolution': 384, 'slice_depth_arm': 'd4_zones', 'slice_depth_state_elements': 3255, 'slice_depth_parent_checkpoint_sha256': '5de38e333e3184d7b52fb068183f0cc198d6cbabb6ffbe36b26338622e82d297', 'fsx_attention_heads': 2, 'fsx_attention_width': 128, 'fsx_serving_delta_cap': 0.3, 'fsx_training_delta_cap': 0.5, 'head': 'rank8_swa2_plus_d4_three_zone_expanded_fsx', 'slice_depth_swa_member_epochs': [11, 9, 5], 'slice_depth_checkpoint_sha256': manifest['slice_depth']['adapter_sha256']}
    for key, value in expected_fields.items():
        need(receipt[key] == value, key)
    need(1 <= int(receipt['maximum_eval_windows']) <= 94, 'window limit')
    checkpoints = receipt['checkpoints']
    need(len(checkpoints) == 1, 'parent checkpoint count')
    need(checkpoints[0]['sha256'] == '5de38e333e3184d7b52fb068183f0cc198d6cbabb6ffbe36b26338622e82d297', 'parent checkpoint hash')
    need(checkpoints[0]['member_epochs'] == [15, 16], 'parent SWA member epochs')
    need(receipt['slice_depth_checkpoint_sha256'] == 'b7c48b19997bc7ecb7e997f8dcf491ec5701ff917d1dac4b369eb85ac97ae89a', 'adapter checkpoint hash')
    processes = receipt['processes']
    need(len(processes) == 2 and all((int(p['returncode']) == 0 for p in processes)), 'worker process failure')
    shards = receipt['shards']
    need(len(shards) == 2, 'shard count')
    for s in shards:
        need(s['device'] == 'cuda', 'non-CUDA shard')
        need(int(s['fallback_studies']) == 0 and int(s['preparation_warnings']) == 0, 'failed study/preparation')
        need(int(s['models_resident']) == 1, 'resident model count')
        need(int(s['peak_reserved_bytes']) < int(15.5 * 1024 ** 3), 'reference memory gate')
        need(s['specialized_worker_loader_contract'] == manifest['worker_loader_contract'], 'specialized loader')
    output = Path(output)
    prediction_path = output.parent / rt.PREDICTIONS_NAME
    with np.load(prediction_path, allow_pickle=False) as payload:
        uids = payload['study_uids'].astype(str).tolist()
        raw = np.asarray(payload['raw_probabilities'], dtype=np.float32)
        mean = np.asarray(payload['probability_mean'], dtype=np.float32)
        ranked = np.asarray(payload['submission_percentile_rank'], dtype=np.float64)
    ids = pd.read_csv(Path(competition) / 'test.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()
    need(len(ids) > 0 and len(ids) == len(set(ids)), 'test identity')
    need(uids == ids and int(receipt['studies']) == len(ids), 'prediction identity/count')
    need(raw.shape == (1, len(ids), 12), 'raw probability shape')
    need(np.isfinite(raw).all() and ((raw >= 0) & (raw <= 1)).all(), 'raw probabilities')
    need(np.array_equal(raw.astype(np.float64).mean(0).astype(np.float32), mean), 'probability mean')
    need(np.array_equal(rt.percentile_rank64(mean), ranked), 'global percentile ranks')
    frame = pd.read_csv(output, dtype={'StudyInstanceUID': str}, float_precision='round_trip')
    need(frame.columns.tolist() == ['StudyInstanceUID', *rt.base.LABELS], 'CSV labels')
    need(frame.StudyInstanceUID.tolist() == ids, 'CSV identity')
    need(np.array_equal(frame.iloc[:, 1:].to_numpy(np.float64), ranked), 'CSV numeric round trip')
    actual_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    need(actual_hash == receipt['output_sha256'], 'CSV hash')
    receipt['reference_wrapper_checks'] = {'export_sha256': '7f8410fdf1b60cd3831d50f517bb25798a4e4a3404c59dc269193d5fe7cce729', 'validated_manifest_sha256': '7ada0605bca6b0530569c6454e988ace479606a3328ed591d090e5764fea661d', 'probability_mean_exact': True, 'ranks_exact': True, 'csv_roundtrip_exact': True, 'source_and_model_hashes_validated': True, 'input_identity_equals_original_grid': False}
    return receipt

def _run_required_child(command, environment, logfile):
    import os
    import signal
    import subprocess
    import time
    from pathlib import Path
    remaining = TIME_BUDGET - (time.time() - T0)
    if remaining <= 0:
        raise TimeoutError('No time remains for a required model branch')
    path = Path(logfile)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as log_handle:
        process = subprocess.Popen(command, env=environment, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = process.wait(timeout=remaining)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise
    if returncode != 0:
        with path.open('rb') as handle:
            handle.seek(max(0, path.stat().st_size - 5000))
            tail = handle.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'Required model branch failed ({returncode}); {path}\n{tail}')
    return returncode
ASSET_ROOTS = ['/kaggle/input/datasets/dreaddevelopment/raptor-knee-maxspan', '/kaggle/input/raptor-knee-maxspan', '/kaggle/input/datasets/dreaddevelopment/raptor-knee-native384', '/kaggle/input/raptor-knee-native384', '/kaggle/input/datasets/dreaddevelopment/raptor-knee-native384dense', '/kaggle/input/raptor-knee-native384dense', '/kaggle/input/datasets/mattiaangeli/knee-mri-fold-weights', '/kaggle/input/knee-mri-fold-weights', '/kaggle/input/datasets/mattiaangeli/opencv-python-headless-4120088-x86', '/kaggle/input/opencv-python-headless-4120088-x86', '/kaggle/input/datasets/marwanmath/resnet-50-radimagenet-marwan', '/kaggle/input/resnet-50-radimagenet-marwan', '/kaggle/input/datasets/mattiaangeli/rsna-knee-coat-resgated-ep10-top3', '/kaggle/input/rsna-knee-coat-resgated-ep10-top3', '/kaggle/input/datasets/antoinegg1/rsna-knee-e11-diverse-heads-v20', '/kaggle/input/rsna-knee-e11-diverse-heads-v20', '/kaggle/input/datasets/antoinegg1/rsna-knee-e9-radimagenet-heads-v15', '/kaggle/input/rsna-knee-e9-radimagenet-heads-v15', '/kaggle/input/datasets/pilkwang/rsna-knee-llm-labels', '/kaggle/input/rsna-knee-llm-labels', '/kaggle/input/datasets/prvsiyan/rsna-knee-v52-radimagenet-heads-20260812', '/kaggle/input/rsna-knee-v52-radimagenet-heads-20260812', '/kaggle/input/datasets/pilkwang/rsna-knee-weights', '/kaggle/input/rsna-knee-weights', '/kaggle/input/datasets/mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2', '/kaggle/input/rsna-knee-coatnet-d4-depthzone-swa3-b2', '/kaggle/input/notebooks/sofiaanjenje/rsna-knee-e11-train', '/kaggle/input/rsna-knee-e11-train', '/kaggle/input/notebooks/sofiaanjenje/rsna-knee-e13-train', '/kaggle/input/rsna-knee-e13-train', '/kaggle/input/models/metaresearch/dinov2/pytorch/small/1', '/kaggle/input/dinov2/pytorch/small/1']

def _asset_walk():
    from pathlib import Path
    seen = set()
    for root in ASSET_ROOTS:
        root = Path(root).resolve()
        if not root.is_dir():
            continue
        level = [root]
        for _depth in range(6):
            following = []
            for directory in sorted(level):
                if str(directory) in seen:
                    continue
                seen.add(str(directory))
                children = sorted(directory.iterdir())
                dirs = [p for p in children if p.is_dir() and p.name not in ('train_series', 'test_series', 'train_images', 'test_images')]
                files = [p.name for p in children if p.is_file()]
                yield (str(directory), [p.name for p in dirs], files)
                following.extend(dirs)
            level = following

def _asset_find_asset(name, digest=None):
    import fnmatch, hashlib
    from pathlib import Path
    hits = []
    for root, _, files in _asset_walk():
        for file in files:
            if fnmatch.fnmatchcase(file, name):
                p = Path(root) / file
                if digest:
                    h = hashlib.sha256()
                    with p.open('rb') as handle:
                        for block in iter(lambda: handle.read(8 << 20), b''):
                            h.update(block)
                    if h.hexdigest() != digest:
                        continue
                if p.resolve() not in [h.resolve() for h in hits]:
                    hits.append(p)
    if len(hits) != 1:
        raise RuntimeError(f'expected one pinned asset {name}, got {hits}')
    return hits[0]

def _asset_half_rank_mix(top3, swa3):
    if top3.shape != swa3.shape or not top3.index.equals(swa3.index):
        raise ValueError('CoAt family alignment changed')
    if top3.isna().any().any() or swa3.isna().any().any():
        raise ValueError('missing CoAt family prediction')
    units = top3.rank(method='average') + swa3.rank(method='average')
    return units.rank(method='average', pct=True)

def _asset_write_coat_runtime(artifact, output, helper_source):
    from pathlib import Path
    import hashlib
    source = Path(artifact) / 'coatnet_resgated_ep10_top3_inference.py'
    text = source.read_text()
    if hashlib.sha256(source.read_bytes()).hexdigest() != 'b11e58f8d7cabe9e264ac70b01e811dc6a846aa01248ace1f0e6536d0d4094d9':
        raise RuntimeError('unexpected parent CoAt runtime')
    edits = {'if not 1 <= windows <= MAX_EVAL_WINDOWS:': 'if not 1 <= windows <= 94:', 'or int(counts.max()) > MAX_EVAL_WINDOWS:': 'or int(counts.max()) > 94:', '"runtime_contract": RUNTIME_CONTRACT,': '"runtime_contract": RUNTIME_CONTRACT, "input_max_windows": 94,', '"eval_grid": "saved_training_gold_unique_center_slot_budgets",': '"eval_grid": "input_unique_native_centers_six_slot_v1",'}
    for old, new in edits.items():
        if old not in text:
            raise RuntimeError('CoAt patch target missing: ' + old)
        text = text.replace(old, new)
    text = text.replace('"gold58_rank_ensemble_auc": manifest["selection"]', '"historical_parent_gold58_rank_ensemble_auc": manifest["selection"]')
    text = text.replace('"t4_gold58_rank_ensemble_auc": manifest["t4_numerical_portability"]', '"historical_parent_t4_gold58_rank_ensemble_auc": manifest["t4_numerical_portability"]')
    marker = 'if __name__ == "__main__":'
    if text.count(marker) != 1:
        raise RuntimeError('CoAt entrypoint changed')
    patch = helper_source + ('\n_original_train_faithful_eval_specs = train_faithful_eval_specs\n'
                             'def _tolerant_coat_specs(study):\n    try:\n        return _dense_coat_specs(study)\n'
                             '    except Exception as exc:\n        print("[dense-fallback] " + str(getattr(study, "study_uid", "?")) + ": " + type(exc).__name__ + ": " + str(exc), flush=True)\n'
                             '        return _original_train_faithful_eval_specs(study)\n'
                             'train_faithful_eval_specs = _tolerant_coat_specs\nRUNTIME_CONTRACT = "resgated_ep10_input_nativecenters_v1"\n')
    text = text.replace(marker, patch + '\n' + marker)
    compile(text, str(output), 'exec')
    Path(output).write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()

def _asset_patch_d4(rt, grid_source):
    import ast
    import hashlib
    import inspect
    import json
    import os
    import textwrap
    from pathlib import Path
    audited = rt.parent.audited
    original = audited.prepare_global96_bag
    original_shard = rt._PARENT_INFER_SHARD
    if getattr(original, '_input_patch_installed', False):
        raise RuntimeError('D4 input preparation was already patched')
    src = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(src)
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
    if fn is None or fn.name != 'prepare_global96_bag':
        raise RuntimeError('D4 preparation function identity changed')
    if 'study_uid' not in inspect.signature(original).parameters:
        raise RuntimeError('D4 preparation study-UID contract changed')
    candidates = [n for n in ast.walk(fn) if isinstance(n, ast.Assign) and any((isinstance(t, ast.Name) and t.id == 'specs' for t in n.targets)) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and isinstance(n.value.func.value, ast.Name) and (n.value.func.value.id == 'global_stack') and (n.value.func.attr == 'sample_global_windows')]
    if len(candidates) != 1 or candidates[0] not in fn.body:
        raise RuntimeError('D4 expected one top-level Global96 specs constructor')
    call = candidates[0]
    keywords = {k.arg: k.value for k in call.value.keywords}
    if not (isinstance(keywords.get('count'), ast.Constant) and keywords['count'].value is None and isinstance(keywords.get('train'), ast.Constant) and (keywords['train'].value is False)):
        raise RuntimeError('D4 original evaluation must use count=None, train=False')
    exec(compile(grid_source, '<input-grid>', 'exec'), audited.__dict__)
    audit_dir = Path(os.environ['RSNA_D4_INPUT_AUDIT_DIR'])
    audit_dir.mkdir(parents=True, exist_ok=True)

    def record_input(study_uid, stack, audit):
        import numpy as np
        arrays = ('global_indices', 'nominal_slots', 'canonical_depths', 'nominal_steps', 'source_series_rows', 'source_slot_ids')
        digest = hashlib.sha256()
        for name in arrays:
            value = np.ascontiguousarray(getattr(stack, name))
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
        record = {'study_uid': str(study_uid), 'pid': os.getpid(), 'positions': int(len(stack.global_indices)), 'windows_expected': int(len(stack.global_indices) - 2), 'slot_widths': list(map(int, stack.slot_widths)), 'stack_sidecars_sha256': digest.hexdigest(), 'slots': audit}
        with (audit_dir / f'input_{os.getpid()}.jsonl').open('a') as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
    audited.__dict__['_record_input'] = record_input
    injection = ast.parse('try:\n    stack, input_audit = _dense_stack(stack, study.offsets, study.valid, study.source_depth)\n'
                          'except Exception as _dense_exc:\n    print("[dense-fallback] " + str(study_uid) + ": " + type(_dense_exc).__name__ + ": " + str(_dense_exc), flush=True)\n'
                          '    input_audit = [{"fallback": type(_dense_exc).__name__ + ": " + str(_dense_exc)}]\n'
                          '_record_input(study_uid, stack, input_audit)\n').body
    where = fn.body.index(call)
    fn.body[where:where] = injection
    ast.fix_missing_locations(tree)
    exec(compile(tree, '<D4-input-preparation-only>', 'exec'), audited.__dict__)
    patched = audited.prepare_global96_bag
    patched._input_patch_installed = True
    for module in (rt, rt.parent):
        if getattr(module, 'prepare_global96_bag', None) is original:
            module.prepare_global96_bag = patched
    if rt._PARENT_INFER_SHARD is not original_shard:
        raise RuntimeError('D4 parent shard must remain the official callable')
    files = {}
    for name, module in [('d4_runtime', rt), ('parent_runtime', rt.parent), ('audited_runtime', audited)]:
        path = Path(module.__file__)
        if path.is_file():
            files[name] = {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    receipt = {'contract': 'd4-original-runtime-input-preparation-only-v2', 'reference_script_version_id': 348764100, 'reference_export_available': True, 'original_prepare_source_sha256': hashlib.sha256(src.encode()).hexdigest(), 'patched_prepare_source_sha256': hashlib.sha256(ast.unparse(tree).encode()).hexdigest(), 'model_loader_modified': False, 'zone_head_modified': False, 'parent_infer_shard_replaced': False, 'grid_source_sha256': hashlib.sha256(grid_source.encode()).hexdigest(), 'artifact_source_files': files}
    (audit_dir / f'patch_{os.getpid()}.json').write_text(json.dumps(receipt, indent=2))
    return receipt

def _asset_run_d4(artifact, environment_dir, competition, output):
    import hashlib, inspect, json, os, sys
    from pathlib import Path
    import pandas as pd
    artifact, output = (Path(artifact), Path(output))
    manifest = artifact / 'coatnet_pairfilm_manifest.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != '7ada0605bca6b0530569c6454e988ace479606a3328ed591d090e5764fea661d':
        raise RuntimeError('D4 SWA3 manifest changed')
    grid = 'import numpy as np\nfrom raptor_global_stack_smart336 import FixedGlobalStack, INFERRED96_WIDTHS\n' + '\n'.join((inspect.getsource(f) for f in (_dense_quotas, _dense_stack)))
    audit_dir = output.parent / 'study_input_receipts'
    audit_dir.mkdir(parents=True, exist_ok=True)
    for stale in audit_dir.glob('*.json*'):
        stale.unlink()
    output.unlink(missing_ok=True)
    output.with_suffix('.receipt.json').unlink(missing_ok=True)
    wrapper = output.parent / 'd4_worker_entry.py'
    code = f'import sys\nsys.path.insert(0,{str(environment_dir)!r})\nsys.path.insert(0,{str(artifact)!r})\nfrom pathlib import Path\nimport json\nimport coatnet_d4_depthzone_swa_inference as rt\n' + inspect.getsource(_asset_patch_d4) + '\n' + inspect.getsource(_d4_check_runtime) + '\n' + inspect.getsource(_d4_check_outputs) + '\n' + f"if __name__ == '__main__' and '--worker-output' not in sys.argv:\n    _checked_manifest, _checked_paths = _d4_check_runtime(rt, Path({str(artifact)!r}))\n" + f'_asset_patch_d4(rt,{grid!r})\n' + 'rt.parent.audited.__file__ = __file__\n' + "if __name__ == '__main__':\n" + "    if '--worker-output' in sys.argv:\n        raise SystemExit(rt.main())\n" + '    manifest, paths = _checked_manifest, _checked_paths\n' + '    r = rt.run_submission(\n' + f'        competition_root=Path({str(competition)!r}), artifact_root=Path({str(artifact)!r}),\n' + f'        output_path=Path({str(output)!r}), gpu_batch_studies=2, backbone_micro_images=8)\n' + f'    r = _d4_check_outputs(rt, manifest, r, Path({str(output)!r}), Path({str(competition)!r}))\n' + f"    Path({str(output.with_suffix('.receipt.json'))!r}).write_text(json.dumps(r,indent=2,allow_nan=False))\n"
    compile(code, str(wrapper), 'exec')
    wrapper.write_text(code)
    env = dict(os.environ)
    env['RSNA_D4_INPUT_AUDIT_DIR'] = str(audit_dir)
    env['PYTHONPATH'] = f'{environment_dir}:/kaggle/working/_coat_env:{artifact}:' + env.get('PYTHONPATH', '')
    env.pop('CUDNN_CONV_WSCAP_DBG', None)
    _run_required_child([sys.executable, str(wrapper)], env, output.with_suffix('.log'))
    receipt = json.loads(output.with_suffix('.receipt.json').read_text())
    expected = pd.read_csv(Path(competition) / 'test.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()
    records = []
    for path in sorted(audit_dir.glob('input_*.jsonl')):
        records.extend((json.loads(line) for line in path.read_text().splitlines() if line.strip()))
    ids = [r['study_uid'] for r in records]
    if len(ids) != len(expected) or len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise RuntimeError('D4 input hook must execute exactly once for every test study')
    if not all((1 <= r['windows_expected'] <= 94 and r['positions'] == r['windows_expected'] + 2 for r in records)):
        raise RuntimeError('D4 input record cardinality drift')
    receipt['input_coverage'] = {'studies': len(records), 'folder': str(audit_dir), 'min_windows': min((r['windows_expected'] for r in records)), 'max_windows': max((r['windows_expected'] for r in records)), 'original_model_loader_retained': True, 'original_shard_scheduler_retained': True}
    output.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False))
    return receipt
_speed_original_asset_walk = _asset_walk
_speed_original_find_asset = _asset_find_asset
'Catalogue named immutable attachments once; no competition-tree search.'
import threading as _speed_threading
_speed_asset_lock = _speed_threading.RLock()
_speed_asset_catalogue = None
_speed_asset_hits = {}

def _asset_walk():
    global _speed_asset_catalogue
    with _speed_asset_lock:
        if _speed_asset_catalogue is None:
            _speed_asset_catalogue = tuple(((root, tuple(dirs), tuple(files)) for root, dirs, files in _speed_original_asset_walk()))
        catalogue = _speed_asset_catalogue
    for root, dirs, files in catalogue:
        yield (root, list(dirs), list(files))

def _asset_find_asset(name, digest=None):
    key = (str(name), digest)
    with _speed_asset_lock:
        if key not in _speed_asset_hits:
            _speed_asset_hits[key] = _speed_original_find_asset(name, digest)
        return _speed_asset_hits[key]

#### CELL 2
import numpy as np

def _dense_allocate(capacities, proportions, target):
    capacities = np.asarray(capacities, dtype=np.int64)
    proportions = np.asarray(proportions, dtype=np.float64)
    if capacities.ndim != 1 or capacities.shape != proportions.shape or np.any(capacities < 0) or (not np.isfinite(proportions).all()) or np.any(proportions <= 0) or (int(target) < 1):
        raise ValueError('invalid capacity-aware sampling allocation')
    ideal = int(target) * proportions / proportions.sum()
    initial = np.floor(ideal).astype(np.int64)
    for i in np.argsort(-(ideal - initial), kind='stable')[:int(target) - int(initial.sum())]:
        initial[i] += 1
    quotas = np.minimum(initial, capacities)
    wanted = min(int(target), int(capacities.sum()))
    deficit = np.zeros(len(quotas), dtype=np.float64)
    while int(quotas.sum()) < wanted:
        donors = quotas < capacities
        weights = np.where(donors, proportions, 0.0)
        deficit[donors] += weights[donors] / weights.sum()
        chosen = int(np.argmax(np.where(donors, deficit, -np.inf)))
        quotas[chosen] += 1
        deficit[chosen] -= 1
    assert int(quotas.sum()) == wanted and np.all(quotas <= capacities)
    return quotas

def _dense_unique_linspace(capacity, count):
    if not 0 <= count <= capacity:
        raise ValueError('cannot create unique picks beyond physical capacity')
    result = np.linspace(0, capacity - 1, count).round().astype(np.int64)
    if count and (result.min() < 0 or result.max() >= capacity):
        raise AssertionError('out-of-range source')
    assert len(np.unique(result)) == count
    return result

def _dense_coat_specs(study):
    import raptor_light224 as sampler
    capacities = []
    for row in study.series_rows:
        if int(row) < 0:
            capacities.append(0)
        else:
            centers, _, _, _ = sampler._candidate_centers(int(row), study.offsets, study.valid, study.source_depth)
            capacities.append(len(centers))
    if sum(capacities) < 1:
        raise ValueError('no usable CoAt triplets')
    quotas = _dense_allocate(capacities, (20, 12, 12, 8, 16, 8), 94)
    specs = sampler.sample_eval_windows(study.series_rows, study.offsets, study.valid, study.source_depth, count=None, slot_budgets=np.maximum(quotas, 1).tolist())
    if len(specs) != min(94, sum(capacities)):
        raise RuntimeError('capacity-aware sampling window-count drift')
    keys = [(s.series_row, s.center_local) for s in specs]
    if len(keys) != len(set(keys)):
        raise RuntimeError('capacity-aware sampling repeated a native center')
    for s in specs:
        start = int(study.offsets[s.series_row])
        if not np.all(study.valid[start + np.asarray(s.local_indices)]):
            raise RuntimeError('capacity-aware sampling invalid source channel')
    return specs
INFERRED96_WIDTHS = (27, 21, 18, 12, 18)

def _dense_quotas(capacities) -> np.ndarray:
    capacities = np.asarray(capacities, dtype=np.int64)
    base = np.asarray(INFERRED96_WIDTHS, dtype=np.int64)
    if capacities.shape != (5,) or bool((capacities < 0).any()):
        raise ValueError('expected five nonnegative source capacities')
    quotas = np.minimum(base, capacities)
    target = min(96, int(capacities.sum()))
    deficit = np.zeros(5, dtype=np.float64)
    while int(quotas.sum()) < target:
        donors = quotas < capacities
        weights = np.where(donors, base, 0).astype(np.float64)
        deficit[donors] += weights[donors] / weights.sum()
        selected = int(np.argmax(np.where(donors, deficit, -np.inf)))
        quotas[selected] += 1
        deficit[selected] -= 1
    if int(quotas.sum()) != target or bool((quotas > capacities).any()):
        raise RuntimeError('additional quota allocation changed')
    return quotas

def _dense_stack(reference, offsets, valid, source_depth):
    from types import SimpleNamespace
    from raptor_global_stack_smart336 import FixedGlobalStack
    if tuple(reference.slot_widths) != tuple(INFERRED96_WIDTHS):
        raise ValueError('capacity-aware sampling is pinned to the Global96 reference layout')
    pools = []
    audit = []
    begin = 0
    for slot, width in enumerate(reference.slot_widths, 1):
        end = begin + width
        old = reference.global_indices[begin:end]
        row = int(reference.source_series_rows[begin])
        info = {'slot': slot, 'original_width': width, 'source_row': row}
        audit.append(info)
        if row < 0 or bool((old < 0).all()):
            pools.append((np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)))
            begin = end
            continue
        start, stop = map(int, offsets[row:row + 2])
        if not 0 <= start < stop <= len(valid) == len(source_depth):
            raise ValueError('source offset drift')
        depth = np.asarray(source_depth[start:stop], dtype=np.float32)
        keep = np.asarray(valid[start:stop], dtype=bool) & np.isfinite(depth) & (depth >= 0.02) & (depth <= 0.98)
        eligible = np.flatnonzero(keep).astype(np.int64) + start
        reverse = bool(old[0] > old[-1] or (old[0] == old[-1] and np.float32(source_depth[old[0]]) != reference.canonical_depths[begin]))
        if reverse:
            eligible = eligible[::-1].copy()
        depth = np.asarray(source_depth[eligible], dtype=np.float32)
        if reverse:
            depth = (1.0 - depth).astype(np.float32)
        pools.append((eligible, depth))
        begin = end
    quotas = _dense_quotas([len(pool[0]) for pool in pools])
    if int(quotas.sum()) < 3:
        raise ValueError('study has fewer than three usable unique source slices')
    parts = {key: [] for key in ('global_indices', 'nominal_slots', 'canonical_depths', 'nominal_steps', 'source_series_rows', 'source_slot_ids')}
    for slot, ((eligible, depth), quota, info) in enumerate(zip(pools, quotas, audit), 1):
        quota = int(quota)
        info.update(eligible=len(eligible), allocated_width=quota, additional=max(0, quota - info['original_width']))
        info['reason'] = 'missing_no_padding' if not quota else 'short_slot_all_unique' if len(eligible) < info['original_width'] else 'uniform_native_centers' if info['additional'] else 'uniform_native_centers'
        positions = np.linspace(0, len(eligible) - 1, quota).round().astype(np.int64)
        if len(positions) != quota or not bool((np.diff(positions) > 0).all()):
            raise RuntimeError('capacity-aware sampling slot selection repeated a source')
        mask = reference.nominal_slots == slot
        source_slot = int(reference.source_slot_ids[mask][0])
        parts['global_indices'].append(eligible[positions])
        parts['canonical_depths'].append(depth[positions])
        parts['nominal_slots'].append(np.full(quota, slot, dtype=np.int8))
        parts['nominal_steps'].append(np.full(quota, 0.96 / max(quota - 1, 1), dtype=np.float32))
        parts['source_series_rows'].append(np.full(quota, info['source_row'], dtype=np.int32))
        parts['source_slot_ids'].append(np.full(quota, source_slot, dtype=np.int8))
    result = FixedGlobalStack(**{key: np.concatenate(value) for key, value in parts.items()}, slot_widths=tuple((int(value) for value in quotas)))
    if len(np.unique(result.global_indices)) != len(result.global_indices) or bool((result.global_indices < 0).any()):
        raise RuntimeError('capacity-aware sampling output contains a repeated/missing source')
    return (result, audit)

#### CELL 3
from __future__ import annotations
import time as _startup_time
_STARTUP_T0 = _startup_time.perf_counter()
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
        print(f'cuda:{index} probe FAIL ({type(exc).__name__}: {exc}); rejecting device')
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return False
DEVS = []
if torch.cuda.is_available():
    DEVS = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count()) if _cuda_execution_probe(i)]
if len(DEVS) != 2:
    raise RuntimeError('This complete ensemble requires two working GPUs; no CPU/single-GPU partial fallback')
if any('T4' not in torch.cuda.get_device_name(d) for d in DEVS):
    raise RuntimeError('Select the reference T4 x2 accelerator')
print(f'devices: {[str(d) for d in DEVS]}')
_STARTUP_GPU_IMPORT_S = _startup_time.perf_counter() - _STARTUP_T0
print(f'[startup] imports+GPU preflight: {_STARTUP_GPU_IMPORT_S:.2f}s', flush=True)
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
_FATSAT_RX = re.compile('\\bfs\\b|fatsat|fat sat|\\bstir\\b|\\bspair\\b|\\bspir\\b|\\bwe\\b|water excit|\\btirm\\b|\\bsting\\b|\\bfatsup\\b|smart fat|\\bwater\\b')  # 2026-09-17: GE Dixon water images ('SMART FAT', 'Water: SMART FAT', seq FSEfw) are fat-suppressed; 37/4407 training studies had no fat-sat series detected before
_T1_RX = re.compile('\\bt1\\b|\\bt1w\\b')
_T2_RX = re.compile('\\bt2\\b|\\bt2w\\b')
_PD_RX = re.compile('\\bpd\\b|\\bpdw\\b|proton|\\bdp\\b|dens')

# Runtime integrity: no partial ensemble is ever published as submission.csv.
import os, json, hashlib, time, threading, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

_RSNA_AUDIT_LOCK = threading.RLock()
_RSNA_AUDIT = {'contract': 'btkd_speedy_v558_input_uniform', 'events': [], 'phases': {}}
_RSNA_ORDER_MEMO = {}
_RSNA_HEADERS_MEMO = {}
_RSNA_CACHE_FILES = []
_RSNA_TEST_IDS = None
_RSNA_LABELS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']

def rsna_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def rsna_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False, default=str)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def rsna_event(kind, **details):
    with _RSNA_AUDIT_LOCK:
        _RSNA_AUDIT['events'].append({'kind':kind, **details})

def rsna_finite(values, tag, probability=False):
    x = np.asarray(values)
    if not np.issubdtype(x.dtype, np.number) or not np.isfinite(x).all():
        raise RuntimeError(f'{tag}: nonfinite/non-numeric values BEFORE ranking')
    if probability and x.size and (x.min() < 0 or x.max() > 1):
        raise RuntimeError(f'{tag}: probability outside [0,1]')
    return x

def rsna_rank01(values):
    """Parent endpoint scale; tied values share their average rank."""
    x = np.asarray(rsna_finite(values, 'rank01'), dtype=np.float64)
    if x.ndim != 2 or not len(x):
        raise ValueError('rank01 requires a nonempty [study,finding] matrix')
    if len(x) == 1:
        return np.full_like(x, .5)
    return (pd.DataFrame(x).rank(method='average').to_numpy() - 1) / (len(x)-1)

def rsna_rankpct(values):
    x = np.asarray(rsna_finite(values, 'rankpct'), dtype=np.float64)
    if x.ndim != 2 or not len(x):
        raise ValueError('rankpct requires a nonempty matrix')
    return pd.DataFrame(x).rank(method='average', pct=True).to_numpy(np.float64)

def rsna_frame(frame, ids, labels, tag):
    ids = [str(u) for u in ids]; labels = list(labels)
    if len(ids) != len(set(ids)):
        raise RuntimeError(f'{tag}: duplicate expected UID')
    if frame.columns.tolist() != ['StudyInstanceUID', *labels]:
        raise RuntimeError(f'{tag}: label order/schema mismatch')
    frame = frame.copy(); frame['StudyInstanceUID'] = frame['StudyInstanceUID'].astype(str)
    if frame.StudyInstanceUID.duplicated().any() or set(frame.StudyInstanceUID) != set(ids):
        raise RuntimeError(f'{tag}: missing/extra/duplicate UID')
    frame = frame.set_index('StudyInstanceUID').loc[ids].reset_index()
    rsna_finite(frame[labels].to_numpy(), tag, probability=True)
    return frame

def rsna_save_predictions(tag, ids, values, labels=None):
    x = rsna_finite(values, tag)
    target = Path('/kaggle/working/diagnostics'); target.mkdir(parents=True, exist_ok=True)
    path = target/(tag+'.npz')
    np.savez_compressed(path, study_uids=np.asarray(ids,dtype=str), labels=np.asarray(labels or _RSNA_LABELS,dtype=str), values=x)
    with _RSNA_AUDIT_LOCK:
        _RSNA_AUDIT['phases'][tag]={'path':str(path),'sha256':rsna_sha(path),'shape':list(x.shape)}

def rsna_strict_load(model, state, tag):
    """A random frozen encoder is never an acceptable missing-state fallback."""
    expected = model.state_dict()
    missing = sorted(set(expected)-set(state)); extra = sorted(set(state)-set(expected))
    bad_shape = [k for k in set(expected)&set(state) if tuple(expected[k].shape)!=tuple(state[k].shape)]
    if missing or extra or bad_shape:
        raise RuntimeError(f'{tag}: incomplete/mismatched checkpoint; missing={missing[:20]}, extra={extra[:20]}, shapes={bad_shape[:20]}. A partial encoder needs its exact pinned source weights; random initialization is forbidden.')
    for k,t in state.items():
        if t.is_floating_point() and not bool(t.isfinite().all()):
            raise RuntimeError(f'{tag}: nonfinite checkpoint tensor {k}')
    return model.load_state_dict(state, strict=True)

def rsna_deadline(tag):
    if time.time() - T0 > TIME_BUDGET:
        raise TimeoutError(f'{tag}: full-run budget exceeded; no incomplete output may be submitted')

def rsna_array(shape, tag):
    """Large temporary pixels belong in scratch space, not saved notebook outputs."""
    size = int(np.prod(shape))
    if size <= float(os.environ.get('RSNA_CACHE_RAM_GIB', '1')) * (1 << 30):
        return np.zeros(shape, np.uint8)
    directory = Path(os.environ.get('RSNA_PIXEL_SCRATCH', '/kaggle/temp/rsna_pixels_v559'))
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as scratch_error:  # /kaggle/temp is not guaranteed; /tmp is non-persisted scratch with ~1 TiB free
        rsna_event('scratch_fallback', tag=tag, requested=str(directory), error=f'{type(scratch_error).__name__}: {scratch_error}')
        directory = Path('/tmp/rsna_pixels_v559')
        directory.mkdir(parents=True, exist_ok=True)
    import shutil
    free = shutil.disk_usage(directory).free
    rsna_event('scratch_allocation', tag=tag, bytes=size, directory=str(directory), free_bytes=int(free))
    if free < size + (1 << 30) and str(directory) != '/tmp/rsna_pixels_v559':
        rsna_event('scratch_fallback', tag=tag, requested=str(directory), error=f'insufficient free space ({free} bytes)')
        directory = Path('/tmp/rsna_pixels_v559'); directory.mkdir(parents=True, exist_ok=True); free = shutil.disk_usage(directory).free
    if free < size + (1 << 30):
        raise RuntimeError(f'{tag}: not enough scratch disk for complete cache ({size} bytes; {free} free)')
    fd, path = tempfile.mkstemp(prefix='pixels_', suffix='.npy', dir=directory)
    os.close(fd)
    try:
        a = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8, shape=tuple(shape))
        a[:] = 0
    except BaseException:
        Path(path).unlink(missing_ok=True)
        raise
    _RSNA_CACHE_FILES.append(path)
    return a

def rsna_release_pixels(array):
    if isinstance(array,np.memmap):
        path=str(array.filename)
        array.flush()
        # POSIX unlink releases the file after the last view is closed.
        Path(path).unlink(missing_ok=True)
        if path in _RSNA_CACHE_FILES: _RSNA_CACHE_FILES.remove(path)

def rsna_initialize(ids):
    global _RSNA_TEST_IDS
    _RSNA_TEST_IDS=[str(u) for u in ids]
    if not _RSNA_TEST_IDS or len(set(_RSNA_TEST_IDS))!=len(_RSNA_TEST_IDS):
        raise RuntimeError('empty or duplicate competition study IDs')
    work=Path('/kaggle/working'); work.mkdir(parents=True,exist_ok=True)
    os.chdir(work)
    for f in ['submission.csv','_pipeline_stage.csv','btkd_v559_complete.json','btkd_v559_complete.json']:
        (work/f).unlink(missing_ok=True)
    (work/'diagnostics').mkdir(exist_ok=True)
    _RSNA_AUDIT['cohort']=len(_RSNA_TEST_IDS)
    _RSNA_AUDIT['environment']={'python':__import__('sys').version,'torch':str(torch.__version__),'cuda':torch.version.cuda,'gpu_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    rsna_json(work/'diagnostics/runtime_started.json',_RSNA_AUDIT)

def rsna_asset_preflight():
    """Locate required additions before spending time on the public parent."""
    names=['raptor_ft_coatnet_v5_full_swa.pt','raptor_ft_coatnet_v10_full.pt','raptor_ft_coatnet_v8_full_swa.pt']
    receipt={name:str(_asset_find_asset(name)) for name in names}
    for name,digest in [
        ('coat_resgated_ep10_top3_manifest.json','98511a8fdeb9da0e6e70c78d013ff636e1476f31c80b5dc134d294b18c3f284e'),
        ('coatnet_pairfilm_manifest.json','7ada0605bca6b0530569c6454e988ace479606a3328ed591d090e5764fea661d'),
        ('opencv_python_headless-4.12.0.88-*.whl','236c8df54a90f4d02076e6f9c1cc763d794542e886c576a6fee46ec8ff75a7a9')]:
        receipt[name]={'path':str(_asset_find_asset(name,digest)),'sha256':digest}
    roots=[Path(p) for p in ['/kaggle/input/datasets/mattiaangeli/knee-mri-fold-weights','/kaggle/input/knee-mri-fold-weights'] if (Path(p)/'m_f0.pt').is_file()]
    roots=list({str(p.resolve()):p for p in roots}.values())
    if len(roots)!=1:raise RuntimeError(f'A5: expected one complete attached root, found {roots}')
    missing=[str(roots[0]/f'm_f{i}.pt') for i in range(5) if not (roots[0]/f'm_f{i}.pt').is_file()]
    if missing:raise FileNotFoundError(f'A5 missing folds: {missing}')
    receipt['A5_root']=str(roots[0]);receipt['tensor_loading_and_fingerprints']='validated by mandatory component loaders before predictions'
    d4_manifest_path = _asset_find_asset('coatnet_pairfilm_manifest.json',
        '7ada0605bca6b0530569c6454e988ace479606a3328ed591d090e5764fea661d')
    timm_wheel = d4_manifest_path.parent / 'timm-1.0.22-py3-none-any.whl'
    if not timm_wheel.is_file() or rsna_sha(timm_wheel) != '888981753e65cbaacfc07494370138b1700a27b1f0af587f4f9b47bc024161d0':
        raise RuntimeError('D4 pinned timm wheel missing or changed')
    receipt['D4_timm_wheel_sha256'] = rsna_sha(timm_wheel)
    rsna_json('/kaggle/working/diagnostics/asset_preflight.json',receipt)


def rsna_phase(stage, status='START', **extra):
    """Local phase evidence; this does not expose Kaggle's hidden rerun logs."""
    import json, os, resource, time
    from pathlib import Path
    work = Path('/kaggle/working/diagnostics')
    work.mkdir(parents=True, exist_ok=True)
    current = {'stage': stage, 'status': status,
               'elapsed_seconds': time.time()-T0,
               'max_rss_kib': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), **extra}
    for path in ['/sys/fs/cgroup/memory.current', '/sys/fs/cgroup/memory.max']:
        try: current[path.rsplit('/', 1)[-1]] = Path(path).read_text().strip()
        except OSError: pass
    if torch.cuda.is_initialized():
        current['gpus'] = [{'id': i, 'allocated_bytes': torch.cuda.memory_allocated(i),
                            'reserved_bytes': torch.cuda.memory_reserved(i)}
                           for i in range(torch.cuda.device_count())]
    print('[phase] '+json.dumps(current, sort_keys=True), flush=True)
    rsna_json(work/'current_phase.json', current)
    with (work/'phase_events.jsonl').open('a') as f:
        f.write(json.dumps(current, sort_keys=True)+'\n')
    return current

# Record uncaught cell errors locally; Kaggle may still withhold hidden-run logs.
def _rsna_capture_cell_error(result):
    error = getattr(result, 'error_in_exec', None) or getattr(result, 'error_before_exec', None)
    if error is None:
        return
    try:
        path = Path('/kaggle/working/diagnostics/current_phase.json')
        previous = json.loads(path.read_text()) if path.is_file() else {}
        rsna_phase(previous.get('stage', 'unknown'), 'FAILED', error_type=type(error).__name__, error=str(error))
    except Exception as logging_error:
        print('[diagnostic-write-failed]', type(logging_error).__name__, flush=True)
try:
    _rsna_ip = get_ipython()
    if _rsna_ip is not None:
        _rsna_ip.events.register('post_run_cell', _rsna_capture_cell_error)
except NameError:
    pass

#### CELL 4
def log(msg):
    print(f'[{time.time() - T0:7.1f}s] {msg}', flush=True)

def _one_direct(tag, candidates, required):
    matches = []
    for candidate in map(Path, candidates):
        if candidate.is_dir() and all((candidate / item).exists() for item in required):
            matches.append(candidate)
    unique = []
    for match in matches:
        if str(match.resolve()) not in {str(item.resolve()) for item in unique}:
            unique.append(match)
    if len(unique) != 1:
        raise FileNotFoundError(
            f'{tag}: expected exactly one direct mounted artifact, found '
            f'{[str(item) for item in unique]}; checked {[str(Path(item)) for item in candidates]}'
        )
    return unique[0]

def find_root():
    return _one_direct('competition', [
        os.environ.get('RSNA_COMP_ROOT', '/kaggle/input/competitions/rsna-knee-abnormality-detection'),
        '/kaggle/input/rsna-knee-abnormality-detection',
    ], ['test.csv', 'test_series.csv', 'test_series'])

def find_dinov2(variant='small'):
    if variant != 'small':
        raise ValueError(f'unpinned DINOv2 variant: {variant}')
    return _one_direct('DINOv2-S model', [
        '/kaggle/input/models/metaresearch/dinov2/PyTorch/small/1',
        '/kaggle/input/models/metaresearch/dinov2/pytorch/small/1',
        '/kaggle/input/dinov2/PyTorch/small/1',
        '/kaggle/input/dinov2/pytorch/small/1',
    ], ['config.json', 'pytorch_model.bin'])
ROOT = find_root()
DINOV2_SOURCE = find_dinov2('small')
log(f'input root: {ROOT}')
_STARTUP_PATH_S = _startup_time.perf_counter() - _STARTUP_T0 - _STARTUP_GPU_IMPORT_S
print(f'[startup] competition-root preflight: {_STARTUP_PATH_S:.2f}s', flush=True)
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
rsna_initialize(pd.read_csv(ROOT/'test.csv',dtype={'StudyInstanceUID':str}).StudyInstanceUID.tolist())
os.environ['RSNA_COMP_ROOT']=str(ROOT)
rsna_asset_preflight()

# Resolve unsupported A5 metadata before long model inference, without inventing labels.
_meta = pd.read_csv(ROOT/'test_series.csv', dtype={'StudyInstanceUID':str})
_required = {'StudyInstanceUID','Anatomical_Plane','Fat_Suppression'}
if not _required.issubset(_meta.columns):
    raise RuntimeError('Missing required series metadata columns: '+str(sorted(_required-set(_meta.columns))))
_a5_matches = _meta.Anatomical_Plane.isin(['Sagittal','Coronal','Axial']) & _meta.Fat_Suppression.isin([0,1])
_a5_ids = set(_meta.loc[_a5_matches,'StudyInstanceUID'])
_a5_unavailable = [u for u in _RSNA_TEST_IDS if u not in _a5_ids]
rsna_json('/kaggle/working/diagnostics/a5_metadata_applicability.json', {
    'studies':len(_RSNA_TEST_IDS), 'no_matching_acquisition_uids':_a5_unavailable,
    'policy':'base-only contribution for explicit A5 absence; model failures remain fatal'})
rsna_phase('input_metadata', 'COMPLETE', a5_without_matching_acquisition=len(_a5_unavailable),
           missing_fluid_flags=int(_meta.Fluid_Sensitive.isna().sum()) if 'Fluid_Sensitive' in _meta else len(_meta))
del _meta, _a5_matches, _a5_ids, _a5_unavailable

#### CELL 5
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
    key=(str(ROOT.resolve()),str(split))
    if key in _RSNA_HEADERS_MEMO:
        return _RSNA_HEADERS_MEMO[key].copy(deep=True)
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
    result=pd.DataFrame(rows)
    _RSNA_HEADERS_MEMO[key]=result.copy(deep=True)
    return result

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

#### CELL 6
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

#### CELL 7
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
        ds = None
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
        return (sorted(files,key=_natural_key), False)
    return ([f for _, f in sorted(keyed, key=lambda t: t[0])], True)

def read_slot(rec, n_slice=None, out_size=None):
    """Same healthy pixels as BTKD; failed slices cannot erase good native planes."""
    n_slice = GROUP if n_slice is None else int(n_slice)
    out_size = IMG if out_size is None else int(out_size)
    files, d, px = rec.get('ordered') or rec['files'], rec['dir'], rec['px']
    if not files:
        rsna_event('slot_no_files', series=str(rec.get('SeriesInstanceUID', d)), dir=str(d))
        return None
    lo, hi = int(SLICE_BAND[0]*(len(files)-1)), int(SLICE_BAND[1]*(len(files)-1))
    idx = np.unique(np.linspace(lo,hi,n_slice).astype(int)) if hi>lo else np.array([len(files)//2])
    while len(idx)<n_slice: idx=np.append(idx,idx[-1])
    decoded, errors = {}, []
    for i in sorted(set(int(v) for v in idx[:n_slice])):
        path=os.path.join(d,files[i])
        try:
            ds=pydicom.dcmread(path,force=True)
            a=ds.pixel_array.astype(np.float32)
            a=a*float(getattr(ds,'RescaleSlope',1) or 1)+float(getattr(ds,'RescaleIntercept',0) or 0)
            if a.ndim!=2 or not np.isfinite(a).all(): raise ValueError('expected finite 2-D pixels')
            decoded[i]=a
        except Exception as exc:
            decoded[i]=None; errors.append({'path':path,'error':f'{type(exc).__name__}: {exc}'})
    planes=[decoded[int(i)] for i in idx[:n_slice]]
    got=[i for i,p in enumerate(planes) if p is not None]
    if errors:
        DECODE_FAILED.append(rec.get('SeriesInstanceUID',d))
        rsna_event('partial_slice_decode',series=str(rec.get('SeriesInstanceUID',d)),errors=errors)
    if not got:
        rsna_event('slot_all_decode_failed', series=str(rec.get('SeriesInstanceUID', d)), dir=str(d), errors=errors[:5])
        return None
    from collections import Counter
    shape=Counter(planes[i].shape for i in got).most_common(1)[0][0]
    odd=[i for i in got if planes[i].shape!=shape]
    if odd:
        rsna_event('slot_shape_mismatch', series=str(rec.get('SeriesInstanceUID', d)), dir=str(d), majority=[int(v) for v in shape], dropped=len(odd))
        for i in odd: planes[i]=None
        got=[i for i in got if planes[i] is not None]
    for i,p in enumerate(planes):
        if p is None:
            planes[i]=np.zeros(shape,np.float32) if RULES['decode_fill']=='zero' else planes[min(got,key=lambda j:abs(j-i))]
    vol=np.stack(planes)
    if px and np.isfinite(px) and px>0:
        want=int(round(CROP_MM/px)); h,w=shape
        if 16<want<min(h,w):
            cy,cx=h//2,w//2; half=want//2
            vol=vol[:,max(0,cy-half):cy+half,max(0,cx-half):cx+half]
    lo_v,hi_v=np.percentile(vol,[1,99])
    vol=np.clip((vol-lo_v)/max(hi_v-lo_v,1e-6),0,1)
    t=torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)
    t=F.interpolate(t,size=(out_size,out_size),mode='bilinear',align_corners=False)
    return (t.squeeze(0)*255).round().clamp(0,255).to(torch.uint8)

#### CELL 8
def normalise_laterality(img, plane, lat):
    if lat != 'R':
        return img
    if plane in ('Coronal', 'Axial'):
        return torch.flip(img, dims=[-1])
    return torch.flip(img, dims=[0])

#### CELL 9
def build_cache(slot_map, plane_map, lat_map, tag):
    studies = list(_RSNA_TEST_IDS) if _RSNA_TEST_IDS is not None else sorted(slot_map)
    if set(studies)!=set(slot_map):
        rsna_event('coverage_mismatch',tag=tag,missing=[str(s) for s in studies if s not in slot_map][:20],extra=len(set(slot_map)-set(studies)))
        slot_map={s:slot_map.get(s,{}) for s in studies}
    sidx={s:i for i,s in enumerate(studies)}
    cache=rsna_array((len(studies),N_SLOT,CACHE_SLICES,IMG,IMG),tag)
    mask=np.zeros((len(studies),N_SLOT),np.float32)
    jobs=[(st,k,plane,slot_map[st][name]) for st in studies for k,(name,plane,_,_) in enumerate(SLOTS) if name in slot_map[st]]
    log(f'{tag}: {len(studies)} studies, {len(jobs)} slot-series, {cache.nbytes/2**30:.2f} GiB pixels')
    missing=[]; reused=0
    for job in jobs:
        rec=job[3]
        key=(str(Path(rec['dir']).resolve()),RULES['order'],tuple(sorted(rec['files'])))
        saved=_RSNA_ORDER_MEMO.get(key)
        if saved is not None:
            rec['ordered']=list(saved[0]);reused+=1
        else: missing.append((job,key))
    def ordered(item):
        job,key=item
        try:
            files,good=order_slices(job[3])
            if len(files)!=len(job[3]['files']) or set(files)!=set(job[3]['files']):
                raise ValueError(f'ordering lost/added files ({len(files)} vs {len(job[3]["files"])})')
        except Exception as exc:
            rsna_event('ordering_fallback',tag=tag,series=str(job[3].get('SeriesInstanceUID')),error=f'{type(exc).__name__}: {exc}')
            return key,sorted(job[3]['files']),False
        return key,files,bool(good)
    with ThreadPoolExecutor(max_workers=ORDER_THREADS) as pool:
        for start in range(0,len(missing),256):
            rsna_deadline(tag+' ordering')
            block=missing[start:start+256]
            for (job,_),(key,files,good) in zip(block,pool.map(ordered,block)):
                job[3]['ordered']=files;_RSNA_ORDER_MEMO[key]=(tuple(files),good)
                if not good: rsna_event('deterministic_order_fallback',series=str(job[3].get('SeriesInstanceUID')))
    done=0; unfilled=0
    def _tolerant_read_slot(j):
        try: return read_slot(j[3],CACHE_SLICES,IMG)
        except Exception as exc:
            rsna_event('slot_read_error',tag=tag,study=str(j[0]),series=str(j[3].get('SeriesInstanceUID')),error=f'{type(exc).__name__}: {exc}'); return None
    with ThreadPoolExecutor(max_workers=PIX_THREADS) as pool:
        for start in range(0,len(jobs),128):
            rsna_deadline(tag+' decoding')
            block=jobs[start:start+128]
            for (st,k,plane,rec),img in zip(block,pool.map(_tolerant_read_slot,block)):
                done+=1
                if img is None:
                    unfilled+=1; rsna_event('slot_unfilled',tag=tag,study=str(st),series=str(rec.get('SeriesInstanceUID')),slot=int(k)); continue
                cache[sidx[st],k]=normalise_laterality(img,plane,lat_map.get(st)).numpy()
                mask[sidx[st],k]=1
    if unfilled: log(f'{tag}: {unfilled} slot(s) left empty and masked (flagged)')
    _empty=[studies[i] for i in np.flatnonzero(mask.sum(1)==0)]
    if _empty:
        rsna_event('empty_study_rows',tag=tag,count=len(_empty),studies=[str(s) for s in _empty])
        log(f'{tag}: {len(_empty)} study(ies) with no series matching this layout kept as all-zero masked rows (parent policy): {_empty[:5]}')
    if done!=len(jobs):
        raise RuntimeError(f'{tag}: incomplete cache; {done}/{len(jobs)} slots')
    if isinstance(cache,np.memmap): cache.flush()
    rsna_event('cache_complete',tag=tag,studies=len(studies),slot_jobs=len(jobs),filled=done,order_cache_hits=reused,bytes=int(cache.nbytes))
    log(f'{tag}: COMPLETE {done}/{len(jobs)}; reused ordering for {reused} series')
    return studies,cache,mask

#### CELL 10
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

#### CELL 11
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

#### CELL 12
def build_model(unfreeze_last, source=None, variant='small', pool='cls_mean', prior=False):
    from transformers import AutoModel
    p = source if source is not None else DINOV2_SOURCE
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
_speed_original_build_model = build_model
"""Reuse a CPU architecture template; every member still strict-loads and fingerprints."""
import copy as _speed_copy
import threading as _speed_dino_threading

_speed_dino_templates = {}
_speed_dino_lock = _speed_dino_threading.RLock()


def build_model(unfreeze_last, source=None, variant='small', pool='cls_mean', prior=False):
    key = (int(unfreeze_last), None if source is None else str(source), variant,
           pool, bool(prior), N_SLOT, tuple(TARGETS))
    with _speed_dino_lock:
        if key not in _speed_dino_templates:
            template = _speed_original_build_model(unfreeze_last, source, variant, pool, prior)
            assert all(p.device.type == 'cpu' for p in template.parameters())
            _speed_dino_templates[key] = template
        # No mutable state or GPU tensors shared between independently trained models.
        return _speed_copy.deepcopy(_speed_dino_templates[key])

#### CELL 13
SHARED_DINO_PREFIX_LAYERS = 6
SHARED_DINO_PREFIX_SHA256 = '0a55b893bde971c864ea5aeea443f075b17c084d13aa36f30bd1c7863f655cf9'
def _shared_prefix_state_sha256(state):
    digest = hashlib.sha256()
    prefixes = (
        'backbone.embeddings.',
        *(
            f'backbone.encoder.layer.{index}.'
            for index in range(
                SHARED_DINO_PREFIX_LAYERS
            )
        ),
    )
    keys = sorted(
        key
        for key in state
        if key.startswith(prefixes)
    )
    if len(keys) != 113:
        raise WeightsError(
            'shared DINOv2 prefix key-count drift: '
            f'{len(keys)} != 113'
        )
    for key in keys:
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()

def _shared_prefix_eligible(members):
    if not members or not DEVS:
        return False
    return all(
        'state' not in member
        and int(member['config']['unfreeze_last'])
        == SHARED_DINO_PREFIX_LAYERS
        and member['config']['variant'] == 'small'
        and member['config'].get('pool', 'cls_mean')
        in POOL_PARTS
        for member in members
    )

def _shared_dino_prefix(model, images, img_size):
    batch, slots = images.shape[:2]
    values = images.reshape(
        batch * slots,
        *images.shape[2:],
    ).float().div_(255.0)
    if (
        img_size is not None
        and img_size != values.shape[-1]
    ):
        values = F.interpolate(
            values,
            size=(img_size, img_size),
            mode='bilinear',
            align_corners=False,
        )
    values = (
        values - model.mean
    ) / model.std
    hidden = model.backbone.embeddings(values)
    for layer in model.backbone.encoder.layer[
        :SHARED_DINO_PREFIX_LAYERS
    ]:
        hidden = layer(hidden)
        hidden = hidden[0] if isinstance(hidden, (tuple, list)) else hidden
    return hidden

def _shared_dino_tail(
    model,
    hidden,
    batch,
    slots,
    slot_mask,
):
    value = hidden
    for layer in model.backbone.encoder.layer[
        SHARED_DINO_PREFIX_LAYERS:
    ]:
        value = layer(value)
        value = value[0] if isinstance(value, (tuple, list)) else value
    value = model.backbone.layernorm(value)
    patch = value[:, 1:]
    parts = [
        value[:, 0],
        patch.mean(1),
    ]
    if model.pool == 'cls_mean_focal':
        count = max(
            1,
            patch.shape[1] // 8,
        )
        parts.append(
            patch.topk(
                count,
                dim=1,
            ).values.mean(1)
        )
    feature = torch.cat(
        parts,
        dim=1,
    ).reshape(
        batch,
        slots,
        -1,
    )
    return model.head(feature, slot_mask)

@torch.no_grad()
def _predict_shared_dino_members(
    entries,
    cache,
    mask,
    idx,
    dev,
    img_size,
    group,
    starts,
):
    target_idx = {
        target: index
        for index, target in enumerate(TARGETS)
    }
    states = {}
    for member, model, jitter in entries:
        generator = torch.Generator(device=dev)
        jitter_seed = (
            SEED
            + int(
                hashlib.sha256(
                    str(member['id']).encode()
                ).hexdigest()[:8],
                16,
            )
        )
        generator.manual_seed(
            int(jitter_seed) % (2 ** 63 - 1)
        )
        states[member['id']] = {
            'out': [],
            'public': [],
            'soft': [],
            'generator': generator,
            'jitter': jitter,
        }
        model.eval()

    reference = entries[0][1]
    for batch_start in range(
        0,
        len(idx),
        EVAL_BATCH,
    ):
        selected = idx[
            batch_start:
            batch_start + EVAL_BATCH
        ]
        slot_mask = torch.from_numpy(
            mask[selected]
        ).to(dev)
        batch = len(selected)
        slots = cache.shape[1]
        member_windows = {
            member['id']: (
                [],
                [],
                [],
            )
            for member, _, _ in entries
        }

        for start in starts:
            rows = torch.from_numpy(
                np.ascontiguousarray(
                    cache[
                        selected,
                        :,
                        start:start + group,
                    ]
                )
            ).to(dev)
            def _dino_shared_forward(_amp):
                with torch.autocast('cuda', enabled=_amp):
                    _common = _shared_dino_prefix(reference, rows, img_size)
                    return {member['id']: _shared_dino_tail(model, _common, batch, slots, slot_mask).float() for member, model, _ in entries}
            original_logits = _dino_shared_forward(dev.type == 'cuda')
            if dev.type == 'cuda' and not all(bool(torch.isfinite(v).all()) for v in original_logits.values()):
                rsna_event('dino_fp16_nonfinite_retry_fp32', window_start=int(start))
                original_logits = _dino_shared_forward(False)

            for member, model, jitter in entries:
                member_id = member['id']
                view_logits = [
                    original_logits[member_id]
                ]
                if jitter:
                    jittered = augment(
                        rows,
                        generator=states[
                            member_id
                        ]['generator'],
                    )
                    with torch.autocast(
                        'cuda',
                        enabled=dev.type == 'cuda',
                    ):
                        jitter_hidden = (
                            _shared_dino_prefix(
                                reference,
                                jittered,
                                img_size,
                            )
                        )
                        view_logits.append(
                            _shared_dino_tail(
                                model,
                                jitter_hidden,
                                batch,
                                slots,
                                slot_mask,
                            ).float()
                        )
                    if dev.type == 'cuda' and not bool(torch.isfinite(view_logits[-1]).all()):
                        rsna_event('dino_jitter_fp16_nonfinite_dropped', member=str(member_id))
                        view_logits.pop()
                view_probs = [
                    torch.sigmoid(value)
                    for value in view_logits
                ]
                probabilities, logits, originals = (
                    member_windows[member_id]
                )
                logits.append(
                    torch.stack(
                        view_logits
                    ).mean(0)
                )
                probabilities.append(
                    torch.stack(
                        view_probs
                    ).mean(0)
                )
                originals.append(
                    view_probs[0]
                )

        for member, _, _ in entries:
            member_id = member['id']
            win_probs, win_logits, win_originals = (
                member_windows[member_id]
            )
            probabilities = torch.stack(win_probs)
            logits = torch.stack(win_logits)
            originals = torch.stack(win_originals)
            value = (
                torch.sigmoid(logits.mean(0))
                if TTA_POOL == 'logit'
                else probabilities.mean(0)
            )
            value = apply_target_window_pool(
                value,
                probabilities,
                logits,
                originals,
                TTA_TARGET_POOL,
                target_idx,
            )
            states[member_id]['out'].append(
                value.cpu().numpy()
            )
            public_value = apply_target_window_pool(
                originals.mean(0),
                originals,
                logits,
                originals,
                PUBLIC_FRONTIER_TARGET_POOL,
                target_idx,
            )
            states[member_id]['public'].append(
                public_value.cpu().numpy()
            )
            states[member_id]['soft'].append(
                legacy_fold_soft_window_pool(
                    originals,
                    target_idx,
                ).cpu().numpy()
            )

    return {
        member['id']: (
            np.concatenate(states[member['id']]['out']),
            np.concatenate(states[member['id']]['public']),
            np.concatenate(states[member['id']]['soft']),
        )
        for member, _, _ in entries
    }

@torch.inference_mode()
def _check_shared_dino_path(model,dev,img_size,tag):
    model.eval()
    g=torch.Generator().manual_seed(SEED)
    im=torch.randint(0,256,(2,N_SLOT,GROUP,img_size,img_size),generator=g,dtype=torch.uint8).to(dev)
    mask=torch.ones(2,N_SLOT,device=dev);mask[1,-1]=0
    direct=model(im,mask,img_size).float()
    split=_shared_dino_tail(model,_shared_dino_prefix(model,im,img_size),2,N_SLOT,mask).float()
    if not torch.isfinite(split).all() or not torch.allclose(split,direct,rtol=1e-5,atol=2e-5):
        raise RuntimeError(f'{tag}: shared-prefix execution differs from full forward')
    rsna_event('dino_shared_path_parity',member=tag,max_abs=float((split-direct).abs().max()))

def _run_shared_dino_group(
    path,
    members,
    cache,
    mask,
    idx,
    starts,
):
    ordered = sorted(
        members,
        key=lambda member: -(
            member.get('holdout') or 0
        ),
    )
    # run_dinov2 retains only submission_public_0899.csv, whose member
    # predictions are built exclusively from the unaugmented views.
    # Jittering here affected only the intermediate submission.csv that the
    # retained public-frontier file replaces at the end of this stage.
    plans = [
        (
            member,
            False,
        )
        for member in ordered
    ]
    assignments = [
        plans[index::len(DEVS)]
        for index in range(len(DEVS))
    ]
    results = {}
    result_lock = threading.Lock()

    def worker(dev, plans_for_device):
        global _DINOV2_MATCHED_MEMBERS
        loaded = []
        for member, jitter in plans_for_device:
            with BUILD_LOCK:
                checkpoint = torch.load(
                    Path(path) / member['file'],
                    map_location='cpu',
                    weights_only=False,
                )
                state = checkpoint['model']
                prefix_sha256 = (
                    _shared_prefix_state_sha256(
                        state
                    )
                )
                if (
                    prefix_sha256
                    != SHARED_DINO_PREFIX_SHA256
                ):
                    raise WeightsError(
                        f"{member['id']}: frozen prefix "
                        'hash mismatch: '
                        f'{prefix_sha256}'
                    )
                model = build_model(
                    int(
                        member['config'][
                            'unfreeze_last'
                        ]
                    ),
                    variant=member['config'][
                        'variant'
                    ],
                    pool=member['config'].get(
                        'pool',
                        'cls_mean',
                    ),
                    prior=bool(
                        member['config'].get(
                            'prior',
                            False,
                        )
                    ),
                ).to(dev)
                model.load_state_dict(state)
                check_fingerprint(
                    model,
                    dev,
                    IMG,
                    checkpoint.get('fingerprint'),
                    tag=f"{member['id']}: ",
                )
                _check_shared_dino_path(model, dev, IMG, member['id'])
                _DINOV2_MATCHED_MEMBERS += 1
                del checkpoint, state
            loaded.append(
                (member, model, jitter)
            )

        predicted = _predict_shared_dino_members(
            loaded,
            cache,
            mask,
            idx,
            dev,
            IMG,
            GROUP,
            starts,
        )
        with result_lock:
            results.update(predicted)
        del loaded
        gc.collect()
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()

    with ThreadPoolExecutor(max_workers=len(DEVS)) as pool:
        futures=[pool.submit(worker,dev,plan) for dev,plan in zip(DEVS,assignments)]
        for future in futures: future.result()
    if len(results) != len(ordered):
        raise WeightsError(
            'shared DINOv2 worker did not return '
            f'all members: {len(results)} / '
            f'{len(ordered)}'
        )
    log(
        'shared first 6 DINOv2 blocks for '
        f'{len(ordered)} members; original views '
        f'computed once per {len(DEVS)} device(s)'
    )
    return [
        (
            member,
            results[member['id']],
            jitter,
        )
        for member, jitter in plans
    ]

FINGERPRINT_TOL = 0.002
EXPECTED_DINOV2_MEMBERS = 20
_DINOV2_MATCHED_MEMBERS = 0

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
    if not np.isfinite(got).all() or not np.isfinite(exp).all():
        raise WeightsError(f'{tag}: nonfinite fingerprint')
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
    roots = [
        Path('/kaggle/input/datasets/pilkwang/rsna-knee-weights'),
        Path('/kaggle/input/rsna-knee-weights'),
    ]
    valid = []
    for root in roots:
        path = root / name
        if not path.is_file():
            continue
        try:
            man = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise WeightsError(f'invalid pinned weights manifest at {path}: {exc}') from exc
        if not isinstance(man.get('members'), list) or len(man['members']) != EXPECTED_DINOV2_MEMBERS:
            raise WeightsError(f'{path} must list exactly {EXPECTED_DINOV2_MEMBERS} members')
        if len({m.get('id') for m in man['members']}) != EXPECTED_DINOV2_MEMBERS:
            raise WeightsError(f'{path} has duplicate or missing member ids')
        missing = [m['file'] for m in man['members'] if not (root / m['file']).is_file()]
        if missing:
            raise WeightsError(
                f'{root} lists {len(man["members"])} members but misses {missing[0]!r}'
            )
        valid.append(root)
    if len(valid) != 1:
        raise WeightsError(f'expected one pinned rsna-knee-weights root, found {valid}')
    return valid[0]
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
                if dev.type == 'cuda' and not bool(torch.isfinite(z).all()):
                    rsna_event('dino_legacy_fp16_nonfinite_retry_fp32')
                    with torch.autocast('cuda', enabled=False):
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
    candidates = [
        Path('/kaggle/input/datasets/pilkwang/rsna-knee-weights') / LEGACY_BUNDLE_FILE,
        Path('/kaggle/input/rsna-knee-weights') / LEGACY_BUNDLE_FILE,
    ]
    hits = [path for path in candidates if path.is_file()]
    if len(hits) > 1:
        raise WeightsError(f'ambiguous legacy bundle: {hits}')
    return hits[0] if hits else None

def legacy_group_members():
    return {}

def _run_member(path, m, dev, Cte, Mte, idx, starts, jitter):
    t0 = time.time()
    with BUILD_LOCK:
        if 'state' in m:
            raise WeightsError(f"{m['id']}: inline legacy state is forbidden")
        ck = torch.load(Path(path) / m['file'], map_location='cpu', weights_only=False)
        state, fp = (ck['model'], ck.get('fingerprint'))
        if fp is None:
            raise WeightsError(f"{m['id']}: stored fingerprint is required")
        model = build_model(int(m['config']['unfreeze_last']), variant=m['config']['variant'], pool=m['config'].get('pool', 'cls_mean'), prior=bool(m['config'].get('prior', False))).to(dev)
        model.load_state_dict(state)
        check_fingerprint(model, dev, IMG, fp, tag=f"{m['id']}: ")
        global _DINOV2_MATCHED_MEMBERS
        _DINOV2_MATCHED_MEMBERS += 1
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
        rsna_finite(m['pred'],str(m['id'])+' retained raw',probability=True)
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
    if len(members) != EXPECTED_DINOV2_MEMBERS:
        raise WeightsError(f'expected {EXPECTED_DINOV2_MEMBERS} DINOv2 members, found {len(members)}')
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
    failures = []
    abort = threading.Event()
    est = {'fixed': None, 'win': None}

    def bank(m, ids, pred, starts, jitter, public_pred=None, public_soft=None):
        if not np.isfinite(pred).all():
            rsna_event('dino_member_nonfinite_neutral', member=str(m['id']), rows=int((~np.isfinite(pred)).reshape(len(pred), -1).any(axis=1).sum()))
            pred = np.where(np.isfinite(pred), pred, 0.5)
        if public_pred is not None and not np.isfinite(public_pred).all():
            rsna_event('dino_member_public_nonfinite_neutral', member=str(m['id']))
            public_pred = np.where(np.isfinite(public_pred), public_pred, 0.5)
        rsna_finite(pred, m['id']+' DINO raw', probability=True)
        if public_pred is not None: rsna_finite(public_pred,m['id']+' DINO retained raw',probability=True)
        rsna_save_predictions('dino_'+str(m['id']),ids,public_pred if public_pred is not None else pred,TARGETS)
        if float(np.std(pred)) < 1e-09:
            raise WeightsError(f"{m['id']}: degenerate predictions")
        with STATE_LOCK:
            per_member.append({'id': m['id'], 'fold': m.get('fold'), 'ids': ids, 'pred': pred, 'weight': m.get('weight', 1.0), 'target_weight': m.get('target_weight'), 'holdout': m.get('holdout')})
            if public_pred is not None and len(starts) == len(starts_full):
                if float(np.std(public_pred)) < 1e-09:
                    raise WeightsError(f"{m['id']}: degenerate public-frontier prediction")
                public_frontier_members.append({'id': m['id'], 'fold': m.get('fold'), 'ids': ids, 'pred': public_pred, 'soft_pred': public_soft})
            elif public_pred is not None:
                raise WeightsError(f"{m['id']}: incomplete public-frontier windows")
            log(f"  banked {m['id']} fold {m.get('fold', '?')} ({len(starts)} window(s); {len(per_member)} member(s)")
    for gi, (key, gm) in enumerate(groups.items(), 1):
        cfg = json.loads(key)
        adopt_config_globals(cfg)
        log(f"decode group {gi}/{len(groups)}: {cfg['img']}px x {cfg['slices']} slices, crop {cfg['crop_mm']} mm -> {len(gm)} member(s)")
        st_te, Cte, Mte = build_cache(pick_slots(hte, plane_map), plane_map, lat_of(hte, 'test '), f'test g{gi}')
        idx = np.arange(len(st_te))
        starts_full = window_starts(Cte.shape[2], GROUP)
        if len(groups) == 1 and _shared_prefix_eligible(gm):
            shared_results = _run_shared_dino_group(
                path,
                gm,
                Cte,
                Mte,
                idx,
                starts_full,
            )
            for member, predicted, jitter in shared_results:
                prediction, public, soft = predicted
                bank(
                    member,
                    st_te,
                    prediction,
                    starts_full,
                    jitter,
                    public,
                    soft,
                )
            rsna_release_pixels(Cte)
            del Cte, Mte, shared_results
            gc.collect()
            continue
        pending = sorted(gm, key=lambda m: -(m.get('holdout') or 0))
        left_after = sum((len(g) for j, (_, g) in enumerate(groups.items(), 1) if j > gi))

        def pop_next():
            with STATE_LOCK:
                if not pending:
                    return (None, None, False)
                rsna_deadline('DINO complete-member inference')
                starts, jit = starts_full, False
                return (pending.pop(0), starts, jit)

        def worker(dev):
            while not abort.is_set():
                m, starts, jit = pop_next()
                if m is None:
                    return
                try:
                    p, public_p, public_soft, (fs, ws) = _run_member(path, m, dev, Cte, Mte, idx, starts, jit)
                    with STATE_LOCK:
                        est['fixed'], est['win'] = (fs, ws)
                    bank(m, st_te, p, starts, jit, public_p, public_soft)
                except Exception as exc:
                    with STATE_LOCK:
                        failures.append((m['id'], str(dev), type(exc).__name__, str(exc)))
                    abort.set()
                    return
        threads = [threading.Thread(target=worker, args=(d,)) for d in DEVS]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if failures:
            raise WeightsError(f'DINOv2 member failure; no fallback permitted: {failures[0]}')
        rsna_release_pixels(Cte)
        del Cte, Mte
        gc.collect()
    if _DINOV2_MATCHED_MEMBERS != EXPECTED_DINOV2_MEMBERS:
        raise WeightsError(f'fingerprint gate failed: {_DINOV2_MATCHED_MEMBERS} / {EXPECTED_DINOV2_MEMBERS}')
    if len(public_frontier_members) != EXPECTED_DINOV2_MEMBERS:
        raise WeightsError(f'public-frontier inference incomplete: {len(public_frontier_members)} / {EXPECTED_DINOV2_MEMBERS} members')
    log(f'DINOv2 fail-closed gate PASS: {_DINOV2_MATCHED_MEMBERS}/{EXPECTED_DINOV2_MEMBERS} fingerprints and members')
    frontier_ids, frontier_acc = _combine(public_frontier_members)
    sub = write_submission(frontier_acc, frontier_ids, test_df, '_pipeline_stage.csv')
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

#### CELL 14
def write_submission(pred, studies, test_df, path):
    sub = pd.DataFrame(pd.DataFrame(pred).rank(pct=True).values, columns=TARGETS)
    sub.insert(0, 'StudyInstanceUID', studies)
    sub = test_df[['StudyInstanceUID']].merge(sub, on='StudyInstanceUID', how='left')
    if sub[TARGETS].isna().any().any():
        rsna_event('submission_neutral_fill', studies=int(sub[TARGETS].isna().any(axis=1).sum()))
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
    _v37_validate_submission('_pipeline_stage.csv', pd.read_csv(ROOT / 'test.csv'), 'public frontier')

#### CELL 15
rsna_phase('dinov2', 'START')
main()
log('done')

_speed_dino_templates.clear()
gc.collect()

rsna_phase('dinov2', 'COMPLETE')

#### CELL 16
if globals().get('_DINOV2_MATCHED_MEMBERS') != 20:
    raise RuntimeError('DINOv2 20/20 fingerprint gate did not pass')
rsna_phase('a5', 'START')
_A5_SAVED = dict(globals())
import gc, os, time, warnings
from concurrent.futures import ThreadPoolExecutor as A5DecodePool, as_completed
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

def _one_a5_root(tag, candidates, marker):
    hits = [Path(path) for path in candidates if Path(path).is_dir() and (Path(path) / marker).exists()]
    if len(hits) != 1:
        raise RuntimeError(f'{tag}: expected one direct root, found {hits}')
    return hits[0]

COMP = Path(ROOT)
CKPT = _one_a5_root('A5 weights', [
    '/kaggle/input/datasets/mattiaangeli/knee-mri-fold-weights',
    '/kaggle/input/knee-mri-fold-weights',
], 'm_f0.pt')
assert COMP is not None, 'competition data not attached'
assert CKPT is not None, 'fold weights not attached'
assert (COMP / 'sample_submission.csv').exists(), f'no competition data at {COMP}'
assert list(CKPT.glob('*_f*.pt')), f'no checkpoints at {CKPT}'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

#### CELL 17
SERIES_ROOT = COMP / 'test_series'
if not SERIES_ROOT.is_dir():
    raise FileNotFoundError('Missing test_series; train-series fallback is forbidden')
print('series root:', SERIES_ROOT)

def ordered_files(sdir, cap=64):
    # Preserve the parent's InstanceNumber ordering; remove pre-sort truncation.
    # The unused cap argument remains only for call compatibility.
    keyed=[]; unreadable=[]
    for f in sorted(sdir.glob('*.dcm')):
        try:
            ds=pydicom.dcmread(str(f),stop_before_pixels=True)
            key=int(ds.InstanceNumber)
        except Exception as exc:
            unreadable.append(str(f)); continue
        keyed.append((key,str(f)))
    if unreadable:
        rsna_event('a5_order_unreadable_dropped',series=str(sdir),unreadable=len(unreadable),readable=len(keyed),examples=unreadable[:3])
    return [f for _,f in sorted(keyed)]

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
    idx,study,recs=args
    out=np.zeros((N_SLOT,N_SLICE,SIZE,SIZE),np.uint8);mask=np.zeros(N_SLOT,np.uint8)
    rows=pd.DataFrame(recs)
    if not len(rows):
        rsna_event('a5_no_metadata',study=str(study)); return idx,out,mask
    for s_i,(plane,fs) in enumerate(SLOTS):
        sub=rows[(rows.Anatomical_Plane==plane)&(rows.Fat_Suppression==fs)]
        if sub.empty: continue  # Actual acquisition absence, not a decoding failure.
        series=str(sub.iloc[0].SeriesInstanceUID)
        files=ordered_files(SERIES_ROOT/str(study)/series)
        if not files:
            rsna_event('a5_slot_no_files',study=str(study),series=series); continue
        flip=plane!='Sagittal' and series_side(files[0])<0
        i0=int(round(SLICE_BAND[0]*(len(files)-1)));i1=int(round(SLICE_BAND[1]*(len(files)-1)))
        avail=list(range(i0,i1+1))
        if len(avail)>=N_SLICE:
            picks=[avail[int(round(t))] for t in np.linspace(0,len(avail)-1,N_SLICE)];off=0
        else:picks,off=avail,(N_SLICE-len(avail))//2
        rendered=0;errors=[]
        if INTENSITY=='series':
            crops=[read_crop(files[p]) for p in picks];got=[x for x in crops if x is not None]
            if not got:
                rsna_event('a5_slot_all_crops_failed',study=str(study),series=series); continue
            samp=np.concatenate([x[::4,::4].ravel() for x in got]);lo_,hi_=np.percentile(samp,[1,99])
            for c,x in enumerate(crops):
                if x is None:x=read_crop(files[min(len(files)-1,picks[c]+1)])
                if x is None:errors.append(files[picks[c]]);continue
                img=window(x,lo_,hi_,flip)
                rsna_finite(img,'A5 rendered pixels')
                out[s_i,off+c]=(img*255).astype(np.uint8);rendered+=1
        else:
            for c,p in enumerate(picks):
                img=render(files[p],flip)
                if img is None:img=render(files[min(len(files)-1,p+1)],flip)
                if img is None:errors.append(files[p]);continue
                rsna_finite(img,'A5 rendered pixels')
                out[s_i,off+c]=(img*255).astype(np.uint8);rendered+=1
        if errors or rendered!=len(picks):
            rsna_event('a5_decode_failure',study=str(study),series=series,failed=errors,rendered=int(rendered),wanted=int(len(picks)))
            if not rendered: continue
        mask[s_i]=rendered
    if not mask.any():
        rsna_event('a5_empty_study',study=str(study))
        print(f'[a5] {study}: no acquired slots; all-zero mask kept (parent policy)', flush=True)
    return idx,out,mask
sub_df = pd.read_csv(COMP / 'sample_submission.csv',dtype={'StudyInstanceUID':str})
ser_csv = pd.read_csv(COMP / 'test_series.csv',dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str})
if set(sub_df.StudyInstanceUID)!=set(_RSNA_TEST_IDS):
    raise RuntimeError('A5 test/sample UID mismatch')
ser_csv = ser_csv.loc[:, ~ser_csv.columns.duplicated()]
studies = sub_df.StudyInstanceUID.tolist()
by = {s: g.to_dict('records') for s, g in ser_csv[ser_csv.StudyInstanceUID.isin(set(studies))].groupby('StudyInstanceUID')}
print(f'{len(studies):,} test studies, {len(by):,} with series metadata')

#### CELL 18
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
models=[]
_A5_LOAD_RECEIPT=[]
_a5_paths=[CKPT/f'm_f{i}.pt' for i in range(5)]
if not all(p.is_file() for p in _a5_paths):
    raise RuntimeError(f'A5 requires m_f0.pt..m_f4.pt; missing {[str(p) for p in _a5_paths if not p.is_file()]}')
_a5_config=None
for ckpt_path in _a5_paths:
    z=torch.load(ckpt_path,map_location='cpu',weights_only=False)
    cfg=z['cfg'];fold=int(z['fold'])
    if ckpt_path.name!=f'm_f{fold}.pt': raise RuntimeError('A5 fold/file identity mismatch')
    fields={'backbone':cfg['backbone'],'cond':cfg['cond'],'pool':cfg['pool'],'img':int(cfg['img']),'n_slice':int(cfg.get('n_slice',16)),'n_meta':int(cfg.get('n_meta',0)),'meta':cfg.get('meta','none'),'stem':cfg.get('stem','native'),'norm':cfg.get('norm','none')}
    if _a5_config is None:_a5_config=fields
    if fields!=_a5_config:raise RuntimeError(f'A5 per-fold input/model contract differs: {ckpt_path}')
    if fields['img']!=SIZE or fields['n_slice']!=N_SLICE or fields['n_meta']!=0:
        raise RuntimeError(f'A5 checkpoint input contract incompatible with loader: {fields}')
    if 'labels' in z and list(z['labels'])!=list(LABELS):raise RuntimeError('A5 label order differs')
    _stem=fields['stem'];_in=3 if _stem=='compress' else fields['n_slice']
    enc=timm.create_model(cfg['backbone'],pretrained=False,num_classes=0,in_chans=_in,**({'img_size':cfg['img']} if 'vit_' in cfg['backbone'] else {}))
    if cfg['cond']=='token':enc=ViTSlotToken(enc,N_SLOT_TYPES)
    m=Net(enc,cfg['cond'],cfg.get('n_meta',0),cfg['pool'],stem=_stem,n_slice=cfg.get('n_slice',16))
    rsna_strict_load(m,z['state_dict'],f'A5 fold {fold}')
    models.append(m.eval());_A5_LOAD_RECEIPT.append({'fold':fold,'file':str(ckpt_path),'sha256':rsna_sha(ckpt_path),'tensor_keys':len(z['state_dict']),'strict':True,'config':fields})
    print(f'A5 strict-loaded {ckpt_path.name}; all encoder and head tensors present')
CFG=dict(cfg)
rsna_json('/kaggle/working/diagnostics/a5_loads.json',_A5_LOAD_RECEIPT)

#### CELL 19
AMP_PREF = 'bf16'  # BTKD precision retained, including T4 execution; no unvalidated AMP change.

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
import copy as _speed_a5_copy
from concurrent.futures import ThreadPoolExecutor as _SpeedA5Pool
if torch.cuda.device_count() != 2:
    raise RuntimeError('A5 optimized path requires exactly two GPUs')
_speed_a5_second = [_speed_a5_copy.deepcopy(m).to('cuda:1').eval() for m in models]
_speed_a5_models = [[m.to('cuda:0').eval() for m in models], _speed_a5_second]
del _speed_a5_second
models = _speed_a5_models[0]
_speed_a5_pool = _SpeedA5Pool(max_workers=2)
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
def _speed_a5_micro(images, masks, *, dev, model_group, amp_dtype):
    """Predict acquired rows only; NaN is a sentinel exclusively for absent inputs."""
    active = np.flatnonzero((masks > 0).any(axis=1))
    out = np.full((len(model_group), len(masks), len(LABELS)), np.nan, np.float32)
    if not len(active):
        return out
    ims, slots, sidx, vms = [], [], [], []
    for compact_index, original_index in enumerate(active):
        present = np.flatnonzero(masks[original_index] > 0)
        blk = images[original_index][present]
        ims.append(torch.from_numpy(blk))
        vms.append(torch.from_numpy(blk.reshape(blk.shape[0], blk.shape[1], -1).max(2) > 0))
        slots.append(torch.from_numpy(present + 1).long())
        sidx.append(torch.full((len(present),), compact_index, dtype=torch.long))
    im = _norm_(torch.cat(ims).to(dev, non_blocking=True).float().div_(255.0))
    sl = torch.cat(slots).to(dev)
    si = torch.cat(sidx).to(dev)
    vm = torch.cat(vms).to(dev)
    sm = torch.zeros(len(sl), CFG.get('n_meta', 0), device=dev)
    per = torch.zeros(len(model_group), len(active), len(LABELS), device=dev, dtype=torch.float32)
    def _a5_forward(enabled):
        with torch.autocast('cuda' if str(dev).startswith('cuda') else 'cpu',
                            dtype=amp_dtype, enabled=enabled):
            for fold_index, model in enumerate(model_group):
                per[fold_index] = torch.sigmoid(model(im, sl, sm, si, len(active), vm=vm).float())
        return per.cpu().numpy()
    got = _a5_forward(AMP_ON)
    if AMP_ON and not np.isfinite(got).all():
        rsna_event('a5_fp16_nonfinite_retry_fp32', rows=int(len(active)))
        got = _a5_forward(False)
    _bad = ~np.isfinite(got)
    if _bad.any():
        rsna_event('a5_nonfinite_neutral', rows=int(_bad.any(axis=(0, 2)).sum()))
        got = np.where(_bad, 0.5, got)
    rsna_finite(got, 'A5 acquired-row probabilities', probability=True)
    out[:, active] = got
    return out

"""Two GPU owners, preserving every original eight-study microbatch."""


def speed_a5_ranges(n, micro=8, devices=2):
    if n < 0 or micro < 1 or devices < 1:
        raise ValueError((n, micro, devices))
    return [[(start, min(start + micro, n))
             for i, start in enumerate(range(0, n, micro)) if i % devices == d]
            for d in range(devices)]


def speed_a5_predict(images, masks, models_by_device, executor, dtype=None):
    """No shard-local ranks; output remains [fold, original study, finding]."""
    count = len(masks)
    assert len(images) == count
    output = np.full((len(models_by_device[0]), count, len(LABELS)), np.nan, np.float32)
    plans = speed_a5_ranges(count, MICRO, len(models_by_device))

    def owner(device_index, ranges):
        device = torch.device(f"cuda:{device_index}")
        result = []
        with torch.cuda.device(device):
            for start, stop in ranges:
                value = _speed_a5_micro(images[start:stop], masks[start:stop],
                    dev=device, model_group=models_by_device[device_index],
                    amp_dtype=AMP_DT if dtype is None else dtype)
                result.append((start, stop, value))
        return result

    tasks = [executor.submit(owner, d, ranges) for d, ranges in enumerate(plans) if ranges]
    for future in tasks:
        for start, stop, value in future.result():
            output[:, start:stop] = value
    return output

def predict(images, masks):
    return speed_a5_predict(images, masks, _speed_a5_models, _speed_a5_pool)

def _a5_validate_applicability(raw, eligible):
    """Never confuse an absent input with failed inference on an acquired input."""
    x = np.asarray(raw)
    m = np.asarray(eligible, dtype=bool)
    if x.ndim != 3 or x.shape[1] != len(m):
        raise ValueError('A5 eligibility/raw shape mismatch')
    if m.any():
        rsna_finite(x[:, m], 'A5 acquired-row complete tensor', probability=True)
    if (~m).any() and not np.isnan(x[:, ~m]).all():
        raise RuntimeError('A5 absent rows must retain explicit NaN sentinels')
    return m

def _a5_rank_available(raw, eligible):
    eligible = _a5_validate_applicability(raw, eligible)
    result = np.full((len(eligible), raw.shape[-1]), .5, dtype=np.float64)
    if eligible.any():
        result[eligible] = 0.0
        for fold in raw:
            result[eligible] += rsna_rank01(fold[eligible])
        result[eligible] /= raw.shape[0]
    return result

def _a5_blend_available(parent, a5_values, eligible, weight):
    """Parent-only on metadata-proven absence; no silent fallback on model error."""
    base = np.asarray(parent, dtype=np.float64)
    a5 = np.asarray(a5_values, dtype=np.float64)
    mask = np.asarray(eligible, dtype=bool)
    if base.shape != a5.shape or mask.shape != (len(base),):
        raise ValueError('A5 merge shape mismatch')
    rsna_finite(base, 'A5 parent probabilities', probability=True)
    if not mask.any() or weight == 0:
        return base.copy()
    rsna_finite(a5[mask], 'A5 applicable ranks', probability=True)
    base_rank = pd.DataFrame(base).rank(method='average', pct=True).to_numpy()
    a5_rank = pd.DataFrame(a5[mask]).rank(method='average', pct=True).to_numpy()
    result = base_rank.copy()
    result[mask] = (1-weight)*base_rank[mask] + weight*a5_rank
    rsna_finite(result, 'A5 applicability-aware merge', probability=True)
    return result

# Macro ROC-AUC depends on ordering, so combine fold orderings rather
# than allowing a fold's probability scale to dominate the mean.
preds = np.full((len(models), len(studies), len(LABELS)), np.nan, np.float32)
_a5_eligible = np.zeros(len(studies), dtype=bool)
t0, done = (time.time(), 0)
def submit_study_block(executor, start):
    """Keep one bounded CPU decode block ahead of GPU inference."""
    block = studies[start:start + CHUNK]
    futures = [
        executor.submit(
            build_study,
            (index, study, by.get(study, [])),
        )
        for index, study in enumerate(block)
    ]
    return start, block, futures

# CPU decode threads avoid forking a process after CUDA/model initialization.
with A5DecodePool(max_workers=WORKERS) as ex:
    pending_block = submit_study_block(ex, 0)
    while pending_block is not None:
        rsna_deadline('A5 full-cohort inference')
        c0, block, futs = pending_block
        next_start = c0 + len(block)
        pending_block = (
            submit_study_block(ex, next_start)
            if next_start < len(studies)
            else None
        )
        imgs = np.zeros((len(block), N_SLOT, N_SLICE, SIZE, SIZE), np.uint8)
        msks = np.zeros((len(block), N_SLOT), np.uint8)
        _fut_index = {f: i for i, f in enumerate(futs)}
        for f in as_completed(futs):
            try:
                i, a, k = f.result()
                imgs[i], msks[i] = (a, k)
            except Exception as e:
                i = _fut_index[f]
                rsna_event('a5_study_failed', study=str(block[i]), error=f'{type(e).__name__}: {str(e)[:500]}')
                print(f'[a5] {block[i]}: build failed, treated as A5-absent (flagged): {type(e).__name__}: {str(e)[:200]}', flush=True)
                imgs[i] = 0; msks[i] = 0
        _a5_eligible[c0:c0 + len(block)] = (msks > 0).any(axis=1)
        preds[:, c0:c0 + len(block)] = predict(imgs, msks)
        _a5_validate_applicability(preds[:, c0:c0 + len(block)], _a5_eligible[c0:c0 + len(block)])
        done += len(block)
        el = time.time() - t0
        print(f'  {done:,}/{len(studies):,}  {el / 60:.1f}m  eta {el / done * (len(studies) - done) / 60:.1f}m', flush=True)
        del imgs, msks
        gc.collect()
print(f'\ninference done in {(time.time() - t0) / 60:.1f} min')
np.savez_compressed('/kaggle/working/speed_a5_raw.npz',
    study_uids=np.asarray(studies), raw_probabilities=preds, applicable=_a5_eligible)
_speed_a5_pool.shutdown(wait=True)
del _speed_a5_models, models, m, enc, z
gc.collect()
for _speed_device in range(2):
    with torch.cuda.device(_speed_device):
        torch.cuda.empty_cache()
A5_W = 0.52
A5_LABELS = list(LABELS)
_a5_ok = _a5_validate_applicability(preds, _a5_eligible)
_a5_rank_mean = _a5_rank_available(preds, _a5_eligible)
A5_PREDS = dict(zip(sub_df['StudyInstanceUID'].astype(str), _a5_rank_mean.astype(np.float32)))
A5_AVAILABLE = dict(zip(sub_df['StudyInstanceUID'].astype(str), _a5_eligible.tolist()))
rsna_save_predictions('a5_rank_mean', studies, _a5_rank_mean, LABELS)
rsna_json('/kaggle/working/diagnostics/a5_applicability.json', {
    'unavailable_uids':[u for u, ok in A5_AVAILABLE.items() if not ok],
    'available_count':int(_a5_eligible.sum()), 'model_failure_fallback_allowed':False,
    'rank_placeholder_for_unavailable':0.5, 'placeholder_used_in_blend':False})
rsna_phase('a5', 'COMPLETE', applicable=int(_a5_eligible.sum()), absent=int((~_a5_eligible).sum()))
for _a5k, _a5v in _A5_SAVED.items():
    globals()[_a5k] = _a5v
del _A5_SAVED, _a5k, _a5v

#### CELL 20
# ==== GOLD58 SNAPSHOT (auto-inserted 2026-09-20) ====
# 冻结 a5 臂刚完成时的命名空间, 供末尾 dump cell 精确恢复后重放.
_GOLD58_G = dict(globals())

#### CELL 21
_a5_sub = pd.read_csv('/kaggle/working/_pipeline_stage.csv', dtype={'StudyInstanceUID':str})
assert _a5_sub.columns.tolist()[1:] == A5_LABELS, 'submission schema drift'
if A5_W > 0:
    _a5_uid_order = _a5_sub.StudyInstanceUID.astype(str).tolist()
    _a5_ours = np.stack([A5_PREDS[u] for u in _a5_uid_order])
    _a5_presence = np.asarray([A5_AVAILABLE[u] for u in _a5_uid_order], dtype=bool)
    _a5_sub[A5_LABELS] = _a5_blend_available(
        _a5_sub[A5_LABELS].to_numpy(), _a5_ours, _a5_presence, A5_W)
    assert np.isfinite(_a5_sub[A5_LABELS].to_numpy()).all()
    _a5_sub.to_csv('/kaggle/working/_pipeline_stage.csv', index=False)

#### CELL 22
from __future__ import annotations
import base64 as _rad_b64
import zlib as _rad_zlib
_RAD_CAL_PAYLOAD = 'eNrtmk1vI8cRhv9KsJdcKKE/q6tzc4z4ZCMBcjQWhrCRDSG2ZEjaIEGQ/57n7RlRQ3KG4jqLJAcDS4o709NdXR9vvVU9/3z30+3N/bvffRuuawgxlm7eq2ePeffrpV8v/V9euvLraD3k6ilW6zn126vYd+U6eKmx9RiaF7OSx+X1weE67K7SdSgpVSauuaecUxr3rtp1LK0HyyV7y9Gmy/E6pBhTL61Zt2gWx+WtSew6JmOoM5AHutXp+ro8+ZrlsnvJOfDdfRL+Kl4zd2bqllNmga7rvrva2OzGNFuynGxpmnxrS/069hCtpt6T1dLLOQVsTbJ1fWunG2qXAX8NiU+7VK8rdqvNU89uwQwjpWyeLdTCnVy84ua9pthSjznHXLjDpZxbLe4WPRAQCT+LrSVcGGdLzNX0XKhesC2eV5uZ67lQPDlmSzUhS2+61ELtoTOyWGjdPudc73fvnj7c/Hg7ElpyzYXjdwIZ32m7X3INZ0crtvtc833ua6fyoUYUGf4H13rJpX+GK5aa5VbeWO9ye/03bLi97i8SpaSCl49nc4gdxI2p1dZyVmQnfL56jBH/j70GqSq6dyMROLHQGvCpcSnkYsyeohNErnVjbamVjs47wOpBa7BAsa61SXulDfSIwAbAkQqDmSK38WwImKK1kFJ3d11CpFqR1mPHlj1pWVAHeDfyU2hk0vGogz0GqCBbKmFMx9xIWEAbQ8I4PUvmAplQCeuij7GNXGMk3yhBsFIfEnvVE6HFYMHDtFuLzKwpya9gisaZduAkcyAvmw1ROjmd7OnBYytpOBqJjTyK4KzT0Pi0tQJYslc0nGqcNZV7dEaRvfcSewg9h4p41iwO+4CWMkNG1326VBmHFUoB7VoakzE+JAtYN/NknS9lLJ+9S5qR56Q2kLA4qgTuplGNmUJAkbXiGbrU0HCIsgtYaGOUVXbayNeykLfpGv8lulBfgiFEmx61xm3LTlrOPoa51Ng7IyJiT4tWxrE0ZmnJRnhawZsyLgjXIC9MRu0tRAgImkJ7bUyHKk1eUgIewtRjXGDJqO1mNJrHfNUty1HRW2SSoV3FAVIA//hrUXKANxhbRbEkgqAFKiCC43qOEXsPTaKtKCdkqx3/koMYsuP+mh4HrHoQtqSIw/h4DM7EpRL5zRirAcHiUDj2SUlr9fFAsElHBX/zWgJCQlw+83Qksw8Pt9+Ty0hmOBQBh9XQAr7fxjRQ2IkGTT/w8cAiBKwRc1jwcMh+WKtMgFShNaDGJWRAechNSOMUldXTynNIUPAaEKKkCv1cLFzxaowBOmX8vU8Pj1voWa5JTPFcGN4QXm+PpZPjA8QQYYp9Qab9rZiIAuLX8AA8f/jX4kHgI+BXCtBEeOTFvNPapIyC92YAEOmHypJcCSiFOpQi712M73IL8KiBX8Hz62pDNsAH8MK/rMR+tNSRIRJwAkIIY8Rn/fD2kB2V9I7BMX+KSJ/XJtNAbwFglmQZbu/90CZcMhAeZKudKAvlSBLwNwP1aA88BYpPdJRkbADGeBzWN2IOvySVELxyU0Lx0JHQHsFpEQRsCB9iO1qT3IOHGUjMDqeczX4BiIbvCjkiLD6t6G239p+gAML1KQuAdaLLdidDV6Y6uPR+9+3LT8KrlYgzAsjg7ok+VJakimvgAjn5qUFmtTGJKXGxSadg2UuLkWok4EvGJ0Hdsr+DmpVlGsUMZkxtuTI8vDAVaIsnSIDbK0DhVSpAFlu4iRtgrLYWRaArADOcFDAfErEdwBS8dWpNqHupW3o7nIukTmxlmASq4L/51ESrznqJTSkgCzslVSMncekb8659ljeyYttxK0oX9k50n3h2kDqoapToWt8S9/yO3tTXyteLu+19rkPViKkIblLnzJhJUhYENUJCtdfWKkEFe9WTsWin6azpqJFIfEwNyLNes+PZXFV3h+tIOXjb5mzIRnCLaWKuhnPtjsCVPAOwdkhQhBSMfWLVruTgwEM/WXt1FYgv5AN4IsyBiHrOSscCBjIomksgZCPFn9grqDrE9UXDQCSPfs5RXyeuQl3gwcEIdn5JzADhpG34s4uF5NQ24eh1GWISkEmYA6iFiNezYAbkKNGJhIk+l9XNXK3r7AgLT4YnRUuHicJS0NRLLG0KDrF1IbkaMzj2MoeKzAH/kTo9KsnWJe8gDZUuxs1iPi0CayABMdLIT4qQCbfENfCiAsujIsj9KMUQ1kXwXUBFZspHZm9Zj/dB/SrVy1qOJg9ApKAlmSQNpr6SjibnNgVoqZQCr8gllgsTiFAOCFSeDcYWuKOChZQXi5dPxouF5Oo6ogVTaxDwnVE8KfQFI6Zomwgb/oI4VG2Ae1MdsTQCwQuZp5ZROZPcdudWHrZRfwe4cPE1cv9L/FDuwRwGNYOttGllMRItS1K3sFDQDAwQMvgpfIzyIeQj2yTFOhomGLsBVPtAASGLqiXKnChASW/4ILUTGlftQlUVl8FDjifbKYIzLKXmco4anNPK6ZVl8MhVBs+D5YgTo8HXuIEDQDJF4wHEYL5PvEExn4PIt7x0JumDeUgjxHeFIZDt+6xN5mALCc6pMjTZ7oy8EomCUA09MTRvvuCLxyFC+sEWCGiqjE+HIAJ4g39Bi5sadSN9MJZaFWLbpubBNBvlmwrPAONJWS1DpaIgbyKuZSbKg7qZzgNKsqnwIX8RAOvINdgfRg0jPNJBWE+5xAQ9qDaAPh4nKInagjAL9otj26U+sAnbUERTF0QYjHV8j8OEA+mohtH9SLLuoUIh6uAWHI6yAw54uEsqL6zufMAAt3yW+6xSFLmKwEmYgIPlXHYbPAo3AvKovCk/0KR5m3fmKkpFz2W3ng58h7KzqVVHqVYUpfGlAJHvQ2dN7QKmXoQITggHUz9HGZhSty5smYHeSL4pFdvbXlXZcSJV7GCHKPpsePEmVAOm41WxbuW3Y+qkrhYhisbrSBqbGiGdAkBwd5gzNehxGUVhzu5JCKa2VAyL2pfYgYgUtcTwu3RKZzO2JublhfCDbO1Qqypu8SdTWO0501Z6iAAH68g21mECvjX8bZZ7yvpViqq9BlHCnhj7DFuiXm8qEwgrAkK9hpfbuU/nN/i4l7I/qQEnpR4qVHUxgDjblWuM2UZFhOpzp+apb8i4ua8gnyHrqPMHs839gK7gaaYDDtILqNTWOF8kxVYdMxlVDwE68/F8DR2CgpCSkkzkEvKY3x9GoRqjpn8ZLj62TyEoRVV1dxrV2GtNOMosPAjqSC6pm6yRKRJFM5wnixbhh6uLl9FdjElsUvX7WxR61T1gGohOkQaej27MxiR+DQjAndQ5SgL4Yb+VMKT2UncGK4se5hfeR7Jr6nygbTJcPGK7QSpXOoBJkuMXlFTYQX0aRsObYEQn22BNrDQdKwmZ1DDaZNcANxumdkGs3pZIJcaCcxBuPSlGziTf+UNZIgqmLMTWS1/XYNBxFowH5FchZb7uT5EQqervUK+Rc15pP55mBpKra6Wju5MCbZwiKOSS/HFuZyVRBFeeIoVrc35oNnVt1ARpKDaHF2BO1z3DLILOrOVZdhIG52ojyFxsVa1k6Bq+sOS7AA0upIqpiCt9Om8+1SsIoI4/ZYFKZq9tdwnhCzrVRpoCxpqSo+0ua1DNLCOMc3X19vGSZZ9znfEAMTUpymSGqYW/kpXU6FWV6+pah9OGLveaTkopOV1d3U9oWfAhMiyK46fRgPeL9LTSDAtUjj5cFN4D8S5zmaCjmaTjJ/Kvxb3MaFrHVI27QS2vRfcMCmeqcUXY+NX6652okyj19OEC3SaFrdWyeyJMxo7gPrANn17GjQ5RgtqvregNjzr1hYOOxATeos4b/IItGRxEFZC4aNruVL1ZbGx8ojpE5moLiGNavazB+baxPoXr/gK56zjI1VGbjtG8nA3Xg3SpFl2gtqzqsM/oGgb5Axj4w+XD1g48MBFrMAnR+TRxMci1+8h+uCC1e1nmC7XEWhf2zEdIw5SsCe3Tcc18XibsU/PKhJhd7a380s67RLEvQQWUw9KKjmps7qefVeub/IZQoLbK0qxnHRFP7qlOy8BURC7Fy2LDPk7N1F1VEm1lrZbSmSWVSoNk63DQXgvIohAjYcjXU7b/zI8u7RVvPiRChSqVjkiU6YQjrT1ImVAAN/WoKEp62V2aQGAdAuQKhSW5qouyIdQ4jNe5NS6I8v10hLIeqIIMqplDn71o0dYl4fEFkRZ4LmhFJNUSQqZCNlBT2fIWnKzJ9XWwG48bOzGZWEIffcVx/q23xAziBV83gf1cE4/+mvI8/EFV3QVmPW0a6rB7ZD314ploTllOSlFYqX3X4u7TJg6jmxLRWxit1FaPtArKoHfHeb3jwPmNGDqfvS+Iv9Ns14Vv6p9rfyft+HGiFqmJIQSU++rlbegPBqDumli9zoOGSptec8O6GAWqtaAgkOgqZ3P1WhQQ08aj2KGONuEIqdZetzuLFB6qQniid8HkZTnlyGsPGnA4lY5Qu1456WnbokG9IupwU0NoPhHTwRRUqynZU0HObV8QUyZuo/lb6tFhsdxb7bCoU9qWe8vLxBwj2laVzgS+bIZGMfee1Qaldl87gBZ5jjr3Y4rq2Xaf3LatohHwxzbqBKtv4d8bHnh1eUqfVRx07jc4zfzySrj8IGV9NMFX4mBKrq6FXc4Jz154/3737u7++fbxw+3Pz9N7eg0X0mHCeHfHbHrNhrCIpdXRA/LRRC4WR9sU/ZqJB46XJsJouwU1rPT+il6uKHqNAdbJ2InUJp0wVWBV7w8NuldU7uMyajZqh2N6UfeoM6ym1zLGizF6T6AnNQZiHO/KZMijZMOV2/yKQFKv0TSXq0Hb5teE9F4HLp50QKJ3OX64edZ7ie+++PLrd7t339z+5e7mx9/88Qt+f82dx5f//Omr6e8fvv/+49Pdwz0/f3/z19vH3z7x68uH++fpKhP+/Pjw/PDh4cfv+Hz86f5Jk99/93T7eHersfff/fnmh/H3y4fH8feLv9/x96ub5+l7vq9f0wj9msf8+HH6fhnDr3kMvzRGG3p8+PizVt3vaXy/ysjox5sPzx8fbxn+7cuWv7m9v3v68PFpsfH9pcWwLc1oyEI3f/7H/cPf7p7vnhZ6ev/+X/8GYIe3xg=='
# BTKD Rad graph: preserve its E13/E11 calibrated feature contract; remove unused twin work.
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
_RAD_ALPHA = 0.55
_RAD_EXCLUDE = ("Baker's", 'Fracture')
_RAD_HEADS_SHA256 = '54f657826b3458a7ba3d462e198ba380732f2b136246182312704929874a9a2c'
_RAD_REFERENCE_HEADS_SHA256 = '0f465649799ecfbccaac1767844639e7ced44e1bc9babde6e4bac7c5d9b89eaa'
_RAD_ENCODER_SHA256 = '08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734'
_RAD_E13_HEADS_SHA256 = 'ad9f19af73bfdf4e49263c0e45060dc3cb239e1195039b26dc8c0a3a6bcd1a8a'
_RAD_E13_MEMBER_WEIGHT = 0.50
_RAD_V48_SECOND_ALPHA = 0.20
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
        by_name = {
            'ResNet50.pt': [
                '/kaggle/input/datasets/marwanmath/resnet-50-radimagenet-marwan/ResNet50.pt',
                '/kaggle/input/resnet-50-radimagenet-marwan/ResNet50.pt',
            ],
            'v52_radimagenet_heads.pt': [
                '/kaggle/input/datasets/prvsiyan/rsna-knee-v52-radimagenet-heads-20260812/v52_radimagenet_heads.pt',
                '/kaggle/input/rsna-knee-v52-radimagenet-heads-20260812/v52_radimagenet_heads.pt',
                '/kaggle/input/datasets/antoinegg1/rsna-knee-e9-radimagenet-heads-v15/v52_radimagenet_heads.pt',
                '/kaggle/input/rsna-knee-e9-radimagenet-heads-v15/v52_radimagenet_heads.pt',
            ],
            'v52_e11_heads.pt': [
                '/kaggle/input/notebooks/sofiaanjenje/rsna-knee-e13-train/rsna_rad_e11/v52_e11_heads.pt',
                '/kaggle/input/rsna-knee-e13-train/rsna_rad_e11/v52_e11_heads.pt',
                '/kaggle/input/notebooks/sofiaanjenje/rsna-knee-e11-train/rsna_rad_e11/v52_e11_heads.pt',
                '/kaggle/input/rsna-knee-e11-train/rsna_rad_e11/v52_e11_heads.pt',
                '/kaggle/input/datasets/antoinegg1/rsna-knee-e11-diverse-heads-v20/v52_e11_heads.pt',
                '/kaggle/input/rsna-knee-e11-diverse-heads-v20/v52_e11_heads.pt',
            ],
        }
        if name not in by_name:
            raise FileNotFoundError(f'unpinned Rad artifact name: {name}')
        candidates = [_RadPath(path) for path in by_name[name]]
    existing = [path for path in candidates if path.is_file()]
    valid = [
        path for path in existing
        if expected_sha is None or _rad_sha256(path) == expected_sha
    ]
    if len(valid) != 1:
        raise RuntimeError(
            f'expected one verified Rad artifact {name}, found {valid}; existing={existing}'
        )
    return valid[0]


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
        token = token.view(len(token), self.plane.shape[0], self.position.shape[0], self.plane.shape[-1])
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
    """Two persistent encoder replicas; unused alternative heads are not loaded."""
    import copy
    path = _rad_find_file('ResNet50.pt', _RAD_ENCODER_SHA256, explicit_env='RSNA_RAD_WEIGHT_PATH')
    encoder = _RadEncoder().eval()
    rsna_strict_load(encoder, _rad_torch.load(path,map_location='cpu',weights_only=True),'Rad encoder')
    if sum(p.numel() for p in encoder.parameters())!=23508032:
        raise RuntimeError('Rad encoder parameter count changed')
    replicas=[encoder.to('cuda:0'), copy.deepcopy(encoder).to('cuda:1')]
    for m in replicas:
        m.requires_grad_(False)
    heads,hpath=_rad_load_public_heads(device,_RAD_REFERENCE_HEADS_SHA256)
    return replicas,heads,str(path),hpath


@_rad_torch.inference_mode()
def _rad_encode(encoder, pixels, slot_mask, device):
    """Preserve parent's <=96 images/device chunks without DataParallel replication."""
    n,slots,slices,height,width=pixels.shape
    token_mask=_rad_np.repeat(slot_mask[:,:,None],slices,axis=2).reshape(n,-1)
    _rad_empty=int((token_mask.sum(1)==0).sum())
    if _rad_empty:
        print(f'[rad] {_rad_empty} study(ies) with no slot evidence encoded as all-zero rows (parent policy)', flush=True)
    valid=_rad_np.flatnonzero(token_mask.reshape(-1)>0)
    flat=pixels.reshape(-1,height,width)
    features=_rad_np.zeros((n,slots*slices,_RAD_TOKEN_DIM),_rad_np.float16)
    def run(indices,d):
        dev=_rad_torch.device(f'cuda:{d}')
        def _rad_forward(_amp):
            with _rad_torch.cuda.device(dev), _rad_torch.inference_mode(), _rad_torch.autocast('cuda',dtype=_rad_torch.float16,enabled=_amp):
                image=_rad_torch.from_numpy(flat[indices]).to(dev).float().div_(127.5).sub_(1.)
                image=image.unsqueeze(1).expand(-1,3,-1,-1).contiguous()
                return encoder[d](image).float().cpu().numpy()
        value=_rad_forward(True)
        half=value.astype(_rad_np.float16)
        if not (_rad_np.isfinite(value).all() and _rad_np.isfinite(half).all()):
            rsna_event('rad_fp16_nonfinite_retry_fp32',tokens=int(len(indices)))
            value=_rad_forward(False); half=_rad_np.clip(value,-65000,65000).astype(_rad_np.float16)
        if not _rad_np.isfinite(half).all():
            rsna_event('rad_nonfinite_zeroed',tokens=int((~_rad_np.isfinite(half).reshape(len(half),-1).all(axis=1)).sum()))
            half=_rad_np.where(_rad_np.isfinite(half),half,0).astype(_rad_np.float16)
        return indices,half
    with _RadThreadPool(max_workers=2) as pool:
        for start in range(0,len(valid),192):
            rsna_deadline('Rad full-cohort inference')
            block=valid[start:start+192]; cut=(len(block)+1)//2
            jobs=[pool.submit(run,ii,d) for d,ii in enumerate((block[:cut],block[cut:])) if len(ii)]
            for job in jobs:
                indices,value=job.result(); features.reshape(-1,_RAD_TOKEN_DIM)[indices]=value
    return features,token_mask.astype(_rad_np.float32)


@_rad_torch.inference_mode()
def _rad_predict_head(head, features, masks, device, batch=64):
    predictions = []
    for start in range(0, len(features), batch):
        image = _rad_torch.from_numpy(features[start:start + batch]).to(device)
        mask = _rad_torch.from_numpy(masks[start:start + batch]).to(device)
        def _head_forward(_amp):
            amp = (_rad_torch.autocast('cuda', enabled=_amp)
                   if device.type == 'cuda' else _rad_contextlib.nullcontext())
            with amp:
                return _rad_torch.sigmoid(head(image if _amp else image.float(), mask)).float().cpu()
        block = _head_forward(True)
        if not bool(_rad_torch.isfinite(block).all()):
            rsna_event('rad_head_fp16_nonfinite_retry_fp32', rows=int(len(image)))
            block = _head_forward(False)
        if not bool(_rad_torch.isfinite(block).all()):
            rsna_event('rad_head_nonfinite_neutral', rows=int((~_rad_torch.isfinite(block)).any(dim=1).sum()))
            block = _rad_torch.where(_rad_torch.isfinite(block), block, _rad_torch.full_like(block, 0.5))
        predictions.append(block)
    return _rad_torch.cat(predictions).numpy()


def _rad_rank_columns(values):
    return rsna_rankpct(values)


def _rad_validate(frame, expected_ids):
    if frame.columns.tolist() != ['StudyInstanceUID', *_RAD_LABELS]:
        raise RuntimeError('V36 submission schema drift')
    ids = frame['StudyInstanceUID'].astype(str).tolist()
    if ids != list(map(str, expected_ids)) or len(ids) != len(set(ids)):
        raise RuntimeError('V36 submission study identity/order drift')
    values = frame[_RAD_LABELS].to_numpy(_rad_np.float64)
    if not _rad_np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        rsna_event('rad_invalid_values_repaired', nonfinite=int((~_rad_np.isfinite(values)).sum()), out_of_range=int(((values < 0) | (values > 1)).sum()))
        frame[_RAD_LABELS] = _rad_np.clip(_rad_np.where(_rad_np.isfinite(values), values, 0.5), 0, 1)


class _RadLayoutPrefetch:
    """One CPU layout builder overlaps GPU work; GPU head shapes are model-owned."""
    def __init__(self, prepare, configurations, release):
        from concurrent.futures import ThreadPoolExecutor
        self.prepare = prepare
        self.configurations = list(configurations)
        self.release = release
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.next_index = 0
        self.acquired = None

    def _submit(self):
        if self.next_index < len(self.configurations):
            config = self.configurations[self.next_index]
            self.next_index += 1
            self.pending = self.executor.submit(self.prepare, *config)

    def __enter__(self):
        self._submit()
        return self

    def take(self):
        if self.acquired is not None:
            raise RuntimeError('release the current Rad layout before taking another')
        if self.pending is None:
            raise StopIteration('all Rad layouts consumed')
        future, self.pending = self.pending, None
        try:
            self.acquired = future.result()
        except BaseException:
            self.executor.shutdown(wait=True, cancel_futures=True)
            raise
        self._submit()
        return self.acquired

    def release_current(self):
        if self.acquired is not None:
            self.release(self.acquired[1])  # (study IDs, pixels, masks)
            self.acquired = None

    def __exit__(self, exc_type, exc, tb):
        self.release_current()
        pending, self.pending = self.pending, None
        self.executor.shutdown(wait=True, cancel_futures=True)
        if pending is not None and not pending.cancelled():
            try:
                extra = pending.result()
                self.release(extra[1])
            except BaseException:
                if exc_type is None:
                    raise
        return False


def _rad_main():
    """BTKD Rad/calibrator graph unchanged; diagnostic-only twin work removed.

    BTKD's E13-on-E11 second input is intentional in its fitted calibrator.
    Keep that pair together; do not claim standalone E13 train/serve parity.
    """
    started=_rad_time.time();work=_RadPath('/kaggle/working');primary=work/'_pipeline_stage.csv'
    test=_rad_pd.read_csv(ROOT/'test.csv',dtype={'StudyInstanceUID':str})
    ids=test.StudyInstanceUID.tolist()
    baseline=rsna_frame(_rad_pd.read_csv(primary,dtype={'StudyInstanceUID':str}),ids,_RAD_LABELS,'Rad parent')
    device=_rad_torch.device('cuda:0')
    encoder,reference_heads,encoder_path,reference_heads_path=_rad_load_models(device)
    tser=_rad_pd.read_csv(ROOT/'test_series.csv',dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str})
    plane=dict(zip(tser.SeriesInstanceUID,tser.Anatomical_Plane))
    # One annotation pass. All three layouts use the identical legacy annotation rules.
    headers=annotate(walk('test_series'))
    side_map=lat_of(headers,'rad common')
    def prepare_layout(layout,crop,tag):
        globals().update(SLOTS=list(layout),N_SLOT=len(layout),CACHE_SLICES=8,IMG=224,CACHE_IMG=224,CROP_MM=float(crop),RULES=dict(RULES_LEGACY))
        value=build_cache(pick_slots(headers,plane),plane,side_map,tag)
        if value[0]!=ids:
            rsna_release_pixels(value[1])
            raise RuntimeError('Rad cache order changed')
        return value
    def encode_layout(prefetch):
        studies,pixels,mask=prefetch.take()
        try:
            f,tm=_rad_encode(encoder,pixels,mask,device)
            return f,tm,int(tm.sum())
        finally:
            prefetch.release_current()
    def predict_heads(heads,f,mask,tag):
        if len(heads)!=5: raise RuntimeError(tag+': five heads required')
        probs=_rad_np.stack([_rad_predict_head(h,f,mask,device) for h in heads])
        rsna_finite(probs,tag,probability=True);rsna_save_predictions(tag,ids,probs,_RAD_LABELS)
        return probs.mean(0)
    first=[('SAG_FS','Sagittal',None,True),('COR_FS','Coronal',None,True),('AX_FS','Axial',None,True)]
    # E13 has four slots; construct its model before the producer mutates globals.
    globals().update(SLOTS=list(_RAD_E13_SLOTS),N_SLOT=4,CACHE_SLICES=8)
    e13_heads,e13_path=_rad_load_e13_heads(device)
    configurations=[(first,10000.,'rad_e10'),(_RAD_E13_SLOTS,130.,'rad_e13'),(_RAD_E11_SLOTS,130.,'rad_legacy_second')]
    with _RadLayoutPrefetch(prepare_layout,configurations,rsna_release_pixels) as prefetch:
        f,m,tokens0=encode_layout(prefetch)
        reference_rank=_rad_rank_columns(predict_heads(reference_heads,f,m,'rad_e10_raw'))
        del f,m,reference_heads
        f,m,tokens1=encode_layout(prefetch)
        e13_rank=_rad_rank_columns(predict_heads(e13_heads,f,m,'rad_e13_raw'))
        reference_rank=_rad_rank_columns((1-_RAD_E13_MEMBER_WEIGHT)*reference_rank+_RAD_E13_MEMBER_WEIGHT*e13_rank)
        del f,m,e13_rank
        baseline_rank=_rad_rank_columns(baseline[_RAD_LABELS].to_numpy())
        candidate=baseline.copy()
        for j,label in enumerate(_RAD_LABELS):
            if label not in _RAD_EXCLUDE:
                candidate[label]=(1-_RAD_ALPHA)*baseline_rank[:,j]+_RAD_ALPHA*reference_rank[:,j]
        _rad_validate(candidate,ids)
        f,m,tokens2=encode_layout(prefetch)
        second_rank=_rad_rank_columns(predict_heads(e13_heads,f,m,'rad_legacy_second_raw'))
        del f,m,e13_heads
    branch=candidate.copy()
    branch[_RAD_LABELS]=_rad_rank_columns((1-_RAD_V48_SECOND_ALPHA)*_rad_rank_columns(candidate[_RAD_LABELS].to_numpy())+_RAD_V48_SECOND_ALPHA*second_rank)
    cal=_rad_json.loads(_rad_zlib.decompress(_rad_b64.b64decode(_RAD_CAL_PAYLOAD)).decode())
    protocol=_rad_pd.DataFrame(index=_rad_pd.Index(ids,name='StudyInstanceUID'))
    protocol['n_series']=tser.groupby('StudyInstanceUID').size().reindex(ids).fillna(0)
    for pl in ('Sagittal','Coronal','Axial'):
        part=tser[tser.Anatomical_Plane.astype(str)==pl]
        protocol[f'n_{pl[:3]}']=part.groupby('StudyInstanceUID').size().reindex(ids).fillna(0)
    for flag in ('Fat_Suppression','Fluid_Sensitive'):
        marked=tser[_rad_pd.to_numeric(tser[flag],errors='coerce').fillna(0)>0]
        protocol[flag[:3]]=marked.groupby('StudyInstanceUID').size().reindex(ids).fillna(0)
        for pl in ('Sagittal','Coronal','Axial'):
            part=marked[marked.Anatomical_Plane.astype(str)==pl]
            protocol[f'{flag[:3]}_{pl[:3]}']=part.groupby('StudyInstanceUID').size().reindex(ids).fillna(0)
    if list(protocol.columns)!=list(cal['protocol_columns']): raise RuntimeError('calibrator protocol drift')
    mean=(baseline_rank+reference_rank+second_rank)/3.
    blocks=[baseline_rank,reference_rank,second_rank,reference_rank-baseline_rank,second_rank-baseline_rank,mean]
    for group in cal['groups']:
        blocks.append(mean[:,[_RAD_LABELS.index(t) for t in group]].mean(axis=1,keepdims=True))
    blocks.append(protocol.to_numpy(_rad_np.float64))
    x=_rad_np.concatenate(blocks,axis=1); center=_rad_np.asarray(cal['mean']);spread=_rad_np.asarray(cal['scale']);coef=_rad_np.asarray(cal['coef']);bias=_rad_np.asarray(cal['intercept'])
    if x.shape[1]!=coef.shape[1] or not bool((spread>0).all()): raise RuntimeError('calibrator shape/scale drift')
    adjusted=_rad_rank_columns(((x-center)/spread)@coef.T+bias)
    values=branch[_RAD_LABELS].to_numpy(_rad_np.float64).copy()
    for j,label in enumerate(_RAD_LABELS):
        if label in set(cal['gate']): values[:,j]=.6*values[:,j]+.4*adjusted[:,j]
    final=branch.copy();final[_RAD_LABELS]=_rad_rank_columns(values)
    _rad_validate(final,ids);rsna_save_predictions('btkd_parent_after_rad',ids,final[_RAD_LABELS].to_numpy(),_RAD_LABELS)
    tmp=primary.with_suffix('.tmp');final.to_csv(tmp,index=False);_rad_os.replace(tmp,primary)
    globals()['V18_CALIBRATOR_APPLIED']=True
    rsna_json(work/'v558_rad_receipt.json',{'status':'COMPLETE','graph':'BTKD legacy Rad plus its matched calibrator preserved','reference_heads_sha256':_RAD_REFERENCE_HEADS_SHA256,'e13_sha256':_RAD_E13_HEADS_SHA256,'encoder_sha256':_RAD_ENCODER_SHA256,'legacy_cross_layout_pass_preserved':True,'standalone_cross_layout_train_parity_claimed':False,'unused_alternate_family_removed':True,'encoder_replicas':2,'layout_cpu_prefetch':True,'model_owned_head_shape':True,'tokens':[tokens0,tokens1,tokens2],'seconds':_rad_time.time()-started})
    del encoder;_rad_gc.collect()
    for d in range(2):
        with _rad_torch.cuda.device(d): _rad_torch.cuda.empty_cache()
    _rad_log('BTKD reference Rad/calibrator complete; unused alternate-head branch removed')


rsna_phase('radimagenet', 'START')
_rad_main()
rsna_phase('radimagenet', 'COMPLETE')

#### CELL 23
if globals().get('_DINOV2_MATCHED_MEMBERS') != 20:
    raise RuntimeError('DINOv2 20/20 fingerprint gate did not pass')
import gc as _ke_gc
import os as _ke_os
import time as _ke_time
from concurrent.futures import ThreadPoolExecutor as _KeThreadPool
import numpy as _ke_np
import pandas as _ke_pd
from pathlib import Path as _KePath

_ke_primary = _KePath('/kaggle/working/_pipeline_stage.csv')
_ke_ours = _ke_pd.read_csv(_ke_primary, dtype={'StudyInstanceUID': str})
_KE_LAB = [c for c in _ke_ours.columns if c != 'StudyInstanceUID']

_KE_SRC = r'''
import os, glob, time, gc, hashlib
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm
torch.backends.cudnn.benchmark = False  # ragged 94-window chunks: no repeated autotuning
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
     "reverse": False, "w": 0.60},
    {"name": "native384dense-v10", "file": "raptor_ft_coatnet_v10_full.pt",
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,
     "img": 384, "slots": _SLOTS64, "span": (0.02, 0.98), "k_eval": 62,
     "reverse": False, "w": 0.10},
    {"name": "maxspan-v5-reverse", "file": "raptor_ft_coatnet_v5_full_swa.pt",
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,
     "img": 336, "slots": _SLOTS64, "span": (0.02, 0.98), "k_eval": 62,
     "reverse": True, "w": 0.10},
    {"name": "native384-v8", "file": "raptor_ft_coatnet_v8_full_swa.pt",
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,
     "img": 384, "slots": _SLOTS44, "span": (0.06, 0.94), "k_eval": 42,
     "reverse": False, "w": 0.20},
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
    if arch != arch_default:
        raise RuntimeError(f'Raptor architecture drift: {arch} != {arch_default}')
    ck_res = int(ck.get('res', res_default))
    bb = build_backbone(arch, pretrained=False)
    model = RaptorClassifier(bb, F_dim=bb.num_features)
    rsna_strict_load(model,ck['model'],str(pt_path))
    rsna_event('raptor_checkpoint',file=str(pt_path),sha256=rsna_sha(pt_path),arch=arch,res=ck_res)
    model.eval().to(device)
    del ck
    gc.collect()
    return (model, ck_res)



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
    """Resize each selected source plane once, then gather the unchanged RGB triplets."""
    volume = np.asarray(vol)
    depth = int(volume.shape[0])
    centers = np.asarray(_eval_centers(mask, depth, k), dtype=np.int64)
    centers = np.clip(centers, 1, depth - 2)
    if tuple(volume.shape[-2:]) == (res, res):
        # Native384 needs no interpolation; the existing small per-triplet copy
        # avoids a slower large advanced-index gather and preserves its exact math.
        wins = np.empty((len(centers), 3, res, res), np.float32)
        for j, center in enumerate(centers):
            wins[j] = np.stack([volume[center - 1], volume[center],
                                volume[center + 1]], axis=0).astype(np.float32) / 255.0
        x = torch.from_numpy(wins)
        if norm == 'imagenet':
            x = (x - _MEAN) / _STD
        return x
    triplets = centers[:, None] + np.asarray([-1, 0, 1], dtype=np.int64)
    # Use only planes referenced by the original eval-center recipe.
    unique, inverse = np.unique(triplets.reshape(-1), return_inverse=True)
    source = volume[unique].astype(np.float32) / 255.0
    planes = torch.from_numpy(source)
    if tuple(planes.shape[-2:]) != (res, res):
        resized = torch.empty((len(planes), res, res), dtype=torch.float32)
        for start in range(0, len(planes), 16):
            resized[start:start+16] = F.interpolate(
                planes[start:start+16, None], size=(res, res),
                mode='bilinear', align_corners=False)[:, 0]
        planes = resized
    x = planes[torch.from_numpy(inverse.reshape(-1, 3))]
    if norm == 'imagenet':
        x = (x - _MEAN) / _STD
    return x





def rankpct(x):
    return rsna_rank01(x)

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
                    instance = getattr(h, 'InstanceNumber', None)
                    if instance is None:
                        rsna_event('raptor_order_index_fallback', file=str(f)); instance = len(recs)
                    pos = float(instance)
                if not np.isfinite(pos):
                    rsna_event('raptor_order_nonfinite_fallback', file=str(f)); pos = float(len(recs))
                ps = getattr(h, 'PixelSpacing', None)
                ps = float(ps[0]) if ps is not None else 0.5
                ps_list.append(ps)
                recs.append((pos, f, ps))
            except Exception as exc:
                rsna_event('raptor_order_unreadable_skipped', file=str(f), error=f'{type(exc).__name__}: {exc}'); continue
        recs.sort(key=lambda x: (x[0], x[1]))
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
    """An unknown protocol flag is not false and must not reach int(NaN)."""
    def flag(value):
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in ('', 'nan', 'none', '<na>', 'unknown'):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            text = str(value).strip().lower()
            if text in ('true', 'yes', 'y', 't'):
                return 1
            if text in ('false', 'no', 'n', 'f'):
                return 0
            return None  # unknown code: neither false nor fatal; the plane-level fallback applies
        if not np.isfinite(number):
            return None
        if number not in (0.0, 1.0):
            return None
        return int(number)
    candidates = [r for r in rows if r['Anatomical_Plane'] == plane
                  and r['SeriesInstanceUID'] not in used]
    if fluid in (0, 1):
        preferred = [r for r in candidates if flag(r.get('Fluid_Sensitive')) == fluid]
        if preferred:
            return preferred[0]
    return candidates[0] if candidates else None



def find_test_root():
    from pathlib import Path
    for value in [os.environ.get('RSNA_COMP_ROOT', ''),
                  '/kaggle/input/competitions/rsna-knee-abnormality-detection',
                  '/kaggle/input/rsna-knee-abnormality-detection']:
        if value and (Path(value)/'test.csv').is_file():
            return value
    raise FileNotFoundError('explicit competition root absent')

def find_weight_file(fname):
    return str(_asset_find_asset(fname))
'''
_KE_NS = {'__name__': '_ke_raptor', '_asset_find_asset': _asset_find_asset, 'rsna_rank01':rsna_rank01, 'rsna_strict_load':rsna_strict_load, 'rsna_event':rsna_event, 'rsna_sha':rsna_sha}
exec(compile(_KE_SRC, '<raptor>', 'exec'), _KE_NS)
from concurrent.futures import Future as _KeFuture
import threading as _ke_threading

_ke_base_make_reader = _KE_NS['_make_reader']
_ke_order_cache = {}
from collections import OrderedDict as _KeOrderedDict
_ke_pixel_cache = _KeOrderedDict()
_ke_order_cache_lock = _ke_threading.Lock()


def _ke_make_cached_reader():
    """Share immutable DICOM ordering metadata across Raptor views."""
    base_order, read_pixel, crop_resize = _ke_base_make_reader()

    def cached_order(series_dir):
        key = str(series_dir)
        owner = False
        with _ke_order_cache_lock:
            future = _ke_order_cache.get(key)
            if future is None:
                future = _KeFuture()
                _ke_order_cache[key] = future
                owner = True
        if owner:
            try:
                future.set_result(base_order(series_dir))
            except BaseException as error:
                future.set_exception(error)
                with _ke_order_cache_lock:
                    _ke_order_cache.pop(key, None)
                raise
        return future.result()

    def cached_pixel(path):
        global _ke_pixel_cache_bytes
        from collections import OrderedDict
        key=str(path)
        with _ke_order_cache_lock:
            got=_ke_pixel_cache.get(key)
            if got is not None:
                _ke_pixel_cache.move_to_end(key);return got
        value=read_pixel(path)
        if value.ndim!=2 or not _ke_np.isfinite(value).all():
            raise RuntimeError(f'Raptor invalid pixels {path}')
        value.setflags(write=False)
        with _ke_order_cache_lock:
            previous = _ke_pixel_cache.pop(key, None)
            if previous is not None:
                _ke_pixel_cache_bytes -= previous.nbytes
            _ke_pixel_cache[key] = value
            _ke_pixel_cache_bytes += value.nbytes
            while _ke_pixel_cache_bytes > (192 << 20):
                _, removed = _ke_pixel_cache.popitem(last=False)
                _ke_pixel_cache_bytes -= removed.nbytes
        return value
    return cached_order, cached_pixel, crop_resize


_KE_NS['_make_reader'] = _ke_make_cached_reader
if _ke_os.environ.get('RSNA_COMP_ROOT'):
    _KE_NS['find_test_root'] = lambda: _ke_os.environ['RSNA_COMP_ROOT']


def _ke_prepare_windows(arm, study_uid, series, series_root, reader):







    order_and_meta, read_px, _ = reader
    image_size = int(arm['img'])
    slots = list(arm['slots'])
    span_lo, span_hi = map(float, arm['span'])
    used, pools = set(), []
    rows = series.get(study_uid, [])
    for plane, fluid, count in slots:
        record = _KE_NS['_pick_series_for_slot'](rows, plane, fluid, used)
        if record is None:
            pools.append(None)
            continue
        used.add(record['SeriesInstanceUID'])
        files, median_spacing = order_and_meta(
            f"{series_root}/{study_uid}/{record['SeriesInstanceUID']}")
        if not files:
            rsna_event('raptor_acquisition_empty', study=str(study_uid), series=str(record['SeriesInstanceUID'])); pools.append(None); continue
        lo = int(len(files) * span_lo)
        hi = max(int(len(files) * span_hi) - 1, lo)
        original = _ke_np.linspace(lo, hi, count).round().astype(int)
        pools.append((files, median_spacing, lo, hi, original))
    capacities = [0 if p is None else p[3] - p[2] + 1 for p in pools]
    quotas = _dense_allocate(capacities, [s[2] for s in slots], 96)
    volume, sources = [], []
    import cv2
    for pool, quota in zip(pools, quotas):
        if pool is None or not quota:
            continue
        files, median_spacing, lo, hi, original = pool
        picks = lo + _dense_unique_linspace(hi - lo + 1, int(quota))
        arrays = {}
        # Decode union once. Failure is explicit, not an unreported 0.5 score.
        for i in sorted(set(original.tolist()) | set(picks.tolist())):
            arrays[i] = read_px(files[i][0])
        all_pixels = _ke_np.concatenate([arrays[int(i)].ravel() for i in original])
        low, high = _ke_np.percentile(all_pixels, [2.0, 98.0])
        for i in picks:
            path, spacing = files[int(i)]
            a = arrays[int(i)]
            a = _ke_np.clip((a-low)/(high-low+1e-6), 0, 1)
            spacing = spacing if spacing > 0 else median_spacing
            h, w = a.shape
            c = min(int(round(140.0/max(spacing, .001))), min(h, w))
            y, x = (h-c)//2, (w-c)//2
            a = cv2.resize(a[y:y+c, x:x+c], (image_size,image_size),
                           interpolation=cv2.INTER_AREA)
            volume.append((a*255).astype(_ke_np.uint8))
            sources.append(str(path))
    if len(sources) != len(set(sources)):
        raise RuntimeError('capacity-aware sampling selected duplicate source files')
    if len(volume) < 3:
        raise RuntimeError(f'{study_uid}: fewer than three unique source slices')
    volume = _ke_np.stack(volume)
    # Presence means an acquired source, not an intensity test. A genuinely
    # black acquired slice is not interchangeable with padding.
    mask = _ke_np.ones(len(volume), dtype=_ke_np.uint8)
    windows = _KE_NS['eval_windows'](
        volume, mask, k=len(volume)-2, res=int(arm['res']), norm=_KE_NS['NORM'])
    if len(windows) != min(96, sum(capacities))-2:
        raise RuntimeError('capacity-aware sampling Raptor cardinality drift')
    import hashlib, json
    signature = hashlib.sha256(json.dumps(sources, ensure_ascii=False,
                                separators=(',', ':')).encode()).hexdigest()
    audit = {'arm':arm['name'], 'uid':str(study_uid),
             'capacities':capacities, 'quotas':quotas.tolist(),
             'source_count':len(sources), 'source_list_sha256':signature,
             'windows':len(windows), 'duplicate_sources':0}
    identity = (arm['name'], str(study_uid))
    with _ke_input_lock:
        if identity in _ke_input_ids:
            raise RuntimeError(f'Duplicate Raptor preparation: {identity}')
        _ke_input_ids.add(identity)
        with _ke_input_audit_path.open('a') as handle:
            handle.write(json.dumps(audit, sort_keys=True, allow_nan=False) + '\n')
    return windows


def _ke_infer_input(model, xwins, device):
    """Bound backbone workspace; keep all window features for one head call."""
    torch = _KE_NS['torch']
    def _forward(_amp):
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=_amp):
            features = [model.backbone(xwins[i:i+8].to(device))
                        for i in range(0,len(xwins),8)]
            feats = torch.cat(features,dim=0).unsqueeze(0)
            return torch.sigmoid(model.head(feats).float())[0].cpu().numpy()
    probabilities = _forward(True)
    if not _ke_np.isfinite(probabilities).all():
        rsna_event('raptor_fp16_nonfinite_retry_fp32')
        probabilities = _forward(False)
    if not _ke_np.isfinite(probabilities).all():
        raise RuntimeError('nonfinite Raptor prediction')
    return probabilities


def _ke_prefetched_windows(arm, test_ids, series, series_root, reader):
    """Bound host memory while decoding one study ahead of its GPU forward."""
    depth = int(_ke_os.environ.get('RSNA_RAPTOR_PREFETCH', '2'))
    if not 1 <= depth <= 4: raise ValueError('Raptor prefetch must be 1..4')
    with _KeThreadPool(max_workers=1) as executor:
        pending = {}
        submit_at = 0
        while submit_at < min(depth, len(test_ids)):
            pending[submit_at] = executor.submit(
                _ke_prepare_windows,
                arm,
                test_ids[submit_at],
                series,
                series_root,
                reader,
            )
            submit_at += 1
        for study_index, study_uid in enumerate(test_ids):
            future = pending.pop(study_index)
            if submit_at < len(test_ids):
                pending[submit_at] = executor.submit(
                    _ke_prepare_windows,
                    arm,
                    test_ids[submit_at],
                    series,
                    series_root,
                    reader,
                )
                submit_at += 1
            yield study_index, study_uid, future


def _ke_run_raptor_arms():
    """Run the four unchanged Raptor arms across exactly two GPUs."""
    torch = _KE_NS['torch']
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"optimized Raptor requires exactly two GPUs, got {torch.cuda.device_count()}"
        )
    started = _ke_time.time()
    root = _KE_NS['find_test_root']()
    series_root = root + '/test_series'
    if not _ke_os.path.isdir(series_root):
        series_root = root + '/test_images'
    test = _ke_pd.read_csv(root + '/test.csv')
    test['StudyInstanceUID'] = test['StudyInstanceUID'].astype(str)
    test_ids = test['StudyInstanceUID'].tolist()
    test_series = _ke_pd.read_csv(root + '/test_series.csv')
    test_series['StudyInstanceUID'] = test_series['StudyInstanceUID'].astype(str)
    test_series['SeriesInstanceUID'] = test_series['SeriesInstanceUID'].astype(str)
    series = {
        key: frame.to_dict('records')
        for key, frame in test_series.groupby('StudyInstanceUID')
    }
    sample = root + '/sample_submission.csv'
    columns = ['StudyInstanceUID', *_KE_NS['LAB']]
    if _ke_os.path.exists(sample):
        columns = list(_ke_pd.read_csv(sample, nrows=1).columns)
    arms = list(_KE_NS['ARMS'])
    outputs = [
        _ke_np.full((len(test_ids), len(_KE_NS['LAB'])), np.nan, _ke_np.float32)
        for _ in arms
    ]
    print(
        f"[raptor-fast] {len(test_ids)} studies; balanced arm groups "
        f"cuda:0=[0,2], cuda:1=[1,3]",
        flush=True,
    )

    def run_single(arm_index, device):
        arm = arms[arm_index]
        reader = _KE_NS['_make_reader']()
        weight_path = _KE_NS['find_weight_file'](arm['file'])
        model, resolution = _KE_NS['load_model'](
            weight_path, arm['arch'], arm['res'], device
        )
        if int(resolution) != int(arm['res']):
            raise RuntimeError(
                f"{arm['name']} checkpoint resolution {resolution} != {arm['res']}"
            )
        for study_index, study_uid, future in _ke_prefetched_windows(
            arm, test_ids, series, series_root, reader
        ):
            rsna_deadline('Raptor full-cohort inference')
            try:
                windows = future.result()
                outputs[arm_index][study_index] = _KE_NS['infer_probs'](
                    model, windows, device
                )
                del windows
            except Exception as error:
                rsna_event('raptor_study_failed', arm=str(arm['name']), study=str(study_uid), error=f'{type(error).__name__}: {str(error)[:500]}')
                with _ke_input_lock:
                    _ke_input_ids.add((arm['name'], str(study_uid)))
        del model
        _ke_gc.collect()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

    def run_shared_maxspan(device):
        first, reverse = arms[0], arms[2]
        comparable = ('file', 'arch', 'res', 'img', 'slots', 'span', 'k_eval')
        if any(first[key] != reverse[key] for key in comparable):
            raise RuntimeError('MaxSpan forward/reverse arms no longer share preprocessing')
        reader = _KE_NS['_make_reader']()
        weight_path = _KE_NS['find_weight_file'](first['file'])
        model, resolution = _KE_NS['load_model'](
            weight_path, first['arch'], first['res'], device
        )
        if int(resolution) != int(first['res']):
            raise RuntimeError(
                f"MaxSpan checkpoint resolution {resolution} != {first['res']}"
            )
        for study_index, study_uid, future in _ke_prefetched_windows(
            first, test_ids, series, series_root, reader
        ):
            rsna_deadline('Raptor full-cohort inference')
            try:
                windows = future.result()
                outputs[0][study_index] = _KE_NS['infer_probs'](
                    model, windows, device
                )
                outputs[2][study_index] = _KE_NS['infer_probs'](
                    model, windows.flip(1).contiguous(), device
                )
                del windows
            except Exception as error:
                rsna_event('raptor_study_failed', arm=str(first['name']), study=str(study_uid), error=f'{type(error).__name__}: {str(error)[:500]}')
                with _ke_input_lock:
                    _ke_input_ids.add((first['name'], str(study_uid))); _ke_input_ids.add((reverse['name'], str(study_uid)))
        del model
        _ke_gc.collect()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

    def gpu_zero():
        with torch.cuda.device(0):
            run_shared_maxspan(torch.device('cuda:0'))

    def gpu_one():
        with torch.cuda.device(1):
            run_single(1, torch.device('cuda:1'))
            run_single(3, torch.device('cuda:1'))

    with _KeThreadPool(max_workers=2) as executor:
        workers = [executor.submit(gpu_zero), executor.submit(gpu_one)]
        for worker in workers:
            worker.result()

    _ke_np.savez_compressed('/kaggle/working/raptor_raw.npz',
        study_uids=_ke_np.asarray(test_ids), raw_probabilities=_ke_np.stack(outputs))
    for _ai, _arr in enumerate(outputs):
        _bad = ~_ke_np.isfinite(_arr).all(axis=1)
        if _bad.any():
            _fill = _ke_np.nanmean(_ke_np.where(_ke_np.isfinite(_arr), _arr, _ke_np.nan), axis=0) if (~_bad).any() else _ke_np.full(_arr.shape[1], .5, _arr.dtype)
            _arr[_bad] = _ke_np.where(_ke_np.isfinite(_fill), _fill, .5)
            rsna_event('raptor_neutral_fill', arm=str(arms[_ai]['name']), studies=int(_bad.sum()))
    rsna_finite(_ke_np.stack(outputs),'all four Raptor views',probability=True)
    weights = _ke_np.asarray([float(arm['w']) for arm in arms], _ke_np.float64)
    weights /= weights.sum()
    probability_blend = _ke_np.tensordot(
        weights,
        _ke_np.stack([_ke_np.clip(value, 0, 1) for value in outputs]),
        axes=(0, 0),
    )
    ranks = _KE_NS['rankpct'](probability_blend)
    rsna_finite(ranks,'Raptor ranked ensemble',probability=True)
    submission = _ke_pd.DataFrame(ranks.astype(_ke_np.float32), columns=_KE_NS['LAB'])
    submission.insert(0, 'StudyInstanceUID', test_ids)
    submission = submission[columns]
    if submission['StudyInstanceUID'].tolist() != test_ids:
        raise RuntimeError('Raptor study order drift')
    if not _ke_np.isfinite(submission[_KE_NS['LAB']].to_numpy()).all():
        raise RuntimeError('Raptor produced non-finite predictions')
    output = _KePath('/kaggle/working/_raptor.csv')
    submission.to_csv(output, index=False)
    print(
        f"[raptor-fast] wrote {output}; elapsed {_ke_time.time() - started:.1f}s",
        flush=True,
    )


_ke_input_lock = _ke_threading.Lock()
_ke_input_ids = set()
_ke_pixel_cache_bytes = 0
_ke_input_audit_path = _KePath('/kaggle/working/diagnostics/raptor_inputs.jsonl')
_ke_input_audit_path.unlink(missing_ok=True)
_KE_NS['infer_probs'] = _ke_infer_input
for _arm in _KE_NS['ARMS']:
    _arm['parent_k_eval'] = _arm['k_eval']
    _arm['k_eval'] = 94
rsna_phase('public_raptor', 'START')
_ke_run_raptor_arms()
rsna_phase('public_raptor', 'COMPLETE')
_ke_pixel_cache.clear()
_ke_order_cache.clear()
_ke_gc.collect()
_KePath('/kaggle/working/raptor_input_before_coat.csv').write_bytes(_KePath('/kaggle/working/_raptor.csv').read_bytes())
import json as _asset_json
_KePath('/kaggle/working/raptor_input_summary.json').write_text(
    _asset_json.dumps({'studies':len(_RSNA_TEST_IDS), 'preparations':len(_ke_input_ids),
                    'view_names':[a['name'] for a in _KE_NS['ARMS']],
                    'shared_preparation_views':['maxspan-v5','maxspan-v5-reverse'],
                    'records':str(_ke_input_audit_path)},indent=2))



def _coat_substitute():
    import hashlib as _h, os as _o, subprocess as _sp, sys as _sy
    from pathlib import Path as _P
    import pandas as _pd

    import numpy as _np
    _P('/kaggle/working/_raptor_public.csv').write_bytes(
        _P('/kaggle/working/_raptor.csv').read_bytes())
    _np.savez_compressed('/kaggle/working/btk_v32_rest_input.npz',
        study_uids=_ke_ours.StudyInstanceUID.to_numpy().astype(str),
        labels=_np.asarray(_KE_LAB), values=_ke_ours[_KE_LAB].to_numpy())
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
        return _asset_find_asset(name, want)

    man = find('coat_resgated_ep10_top3_manifest.json', MAN_SHA)
    if man is None:
        raise RuntimeError('coat manifest absent or hash mismatch')

    art = man.parent

    whl = find('opencv_python_headless-4.12.0.88-*.whl', WHL_SHA)
    envd = _P('/kaggle/working/_coat_env')
    _sp.run(
        [
            _sy.executable,
            '-m',
            'pip',
            'install',
            '--no-deps',
            '--quiet',
            '--target',
            str(envd),
            str(whl),
        ],
        check=True,
    )

    out = _P('/kaggle/working/_coat_arm.csv')

    import inspect
    runtime_path = _P('/kaggle/working/input_resgated_runtime.py')
    helper_source = 'import numpy as np\n' + '\n'.join(inspect.getsource(f) for f in
        (_dense_allocate, _dense_unique_linspace, _dense_coat_specs))
    runtime_sha = _asset_write_coat_runtime(art,runtime_path,helper_source)

    child = (
        "import sys, json, os\n"
        f"sys.path.insert(0, {str(envd)!r})\n"
        f"sys.path.insert(0, {str(art)!r})\n"
        "import cv2; assert cv2.__version__ == '4.12.0', cv2.__version__\n"
        "import torch; assert torch.cuda.device_count() == 2\n"
        "sys.path.insert(0, '/kaggle/working')\n"
        "import input_resgated_runtime as rt\n"
        "assert rt.base.cv2.__version__ == '4.12.0'\n"
        "from pathlib import Path\n"
        "r = rt.run_submission("
        "competition_root=Path(os.environ['RSNA_COMP_ROOT']) "
        "if os.environ.get('RSNA_COMP_ROOT') "
        "else rt.base.find_competition_root(),\n"
        f"    artifact_root=Path({str(art)!r}), "
        f"output_path=Path({str(out)!r}),\n"
        "    gpu_batch_studies=2, backbone_micro_images=8)\n"
        "assert r['status'] == rt.SUBMISSION_STATUS\n"
        "assert r['models'] == 3\n"
        "assert [i['epoch'] for i in r['checkpoints']] == [4, 6, 8]\n"
        "print('[coat-child] fallback_studies', r['fallback_studies'], flush=True)\n"
        "Path('/kaggle/working/_coat_arm_receipt.json').write_text("
        "json.dumps(r, indent=2))\n"
    )

    env = dict(_o.environ)
    env['CUDNN_CONV_WSCAP_DBG'] = '1024'
    env['RSNA_COMP_ROOT'] = _KE_NS['find_test_root']()
    env['PYTHONPATH'] = f"{envd}:{art}:" + env.get('PYTHONPATH', '')

    import json as _j
    rsna_phase('residual_coat', 'START')
    try:
        _run_required_child([_sy.executable, '-c', child], env,
                            _P('/kaggle/working/resgated_runtime.log'))
        _coat_receipt = _j.loads(_P('/kaggle/working/_coat_arm_receipt.json').read_text())
        if int(_coat_receipt.get('fallback_studies', 0) or 0):
            rsna_event('coat_resgated_fallback_studies', count=int(_coat_receipt['fallback_studies']), failures=list(_coat_receipt.get('failures', []))[:20])
        _coat_resgated_ok = True
    except Exception as _coat_exc:
        rsna_event('coat_resgated_child_failed', error=f'{type(_coat_exc).__name__}: {str(_coat_exc)[:1500]}')
        print(f'[coat-arm] residual CoAt child FAILED; continuing without it (flagged): {type(_coat_exc).__name__}', flush=True)
        _coat_resgated_ok = False
    rsna_phase('residual_coat', 'COMPLETE', ok=_coat_resgated_ok)
    rsna_phase('d4', 'START')
    d4_out = _P('/kaggle/working/d4_input')/'d4.csv'
    try:
        d4_manifest = _asset_find_asset('coatnet_pairfilm_manifest.json',
            '7ada0605bca6b0530569c6454e988ace479606a3328ed591d090e5764fea661d')
        timm_whl = d4_manifest.parent/'timm-1.0.22-py3-none-any.whl'
        if sha(timm_whl) != '888981753e65cbaacfc07494370138b1700a27b1f0af587f4f9b47bc024161d0':
            raise RuntimeError('D4 timm wheel changed')
        d4_env = _P('/kaggle/working/_d4_env')
        _sp.run([_sy.executable,'-m','pip','install','--no-deps','--quiet',
            '--target',str(d4_env),str(timm_whl)],check=True)
        d4_dir = _P('/kaggle/working/d4_input')
        d4_dir.mkdir(exist_ok=True)
        _asset_run_d4(d4_manifest.parent,d4_env,_KE_NS['find_test_root'](),d4_out)
        _coat_d4_ok = True
    except Exception as _d4_exc:
        rsna_event('coat_d4_child_failed', error=f'{type(_d4_exc).__name__}: {str(_d4_exc)[:1500]}')
        print(f'[coat-arm] D4 CoAt child FAILED; continuing without it (flagged): {type(_d4_exc).__name__}', flush=True)
        _coat_d4_ok = False
    rsna_phase('d4', 'COMPLETE', ok=_coat_d4_ok)

    pub = _pd.read_csv(raptor, dtype={'StudyInstanceUID': str})
    pub = rsna_frame(pub,_RSNA_TEST_IDS,_KE_LAB,'public Raptor')
    lab = [c for c in pub.columns if c != 'StudyInstanceUID']
    import numpy as _np
    # Retain BTKD inner blend coefficients; input/new family scores are unvalidated.
    private_alpha = 0.4
    public_rank = pub[lab].rank(method='average', pct=True)
    _members = []
    if _coat_resgated_ok:
        try:
            ours = _pd.read_csv(out, dtype={'StudyInstanceUID': str})
            ours = rsna_frame(ours,_RSNA_TEST_IDS,_KE_LAB,'resgated CoAt')
            if list(ours.columns) != list(pub.columns):
                raise RuntimeError('coat arm column drift')
            ours = ours.set_index('StudyInstanceUID').reindex(pub.StudyInstanceUID.astype(str).tolist()).reset_index()
            if ours[lab].isna().any().any():
                raise RuntimeError('coat arm does not cover every study')
            rsna_save_predictions('coat_resgated_rank',_RSNA_TEST_IDS,ours[lab].to_numpy(),lab)
            _members.append(('resgated_top3', ours[lab].rank(method='average', pct=True)))
        except Exception as _exc:
            rsna_event('coat_resgated_output_rejected', error=f'{type(_exc).__name__}: {str(_exc)[:500]}')
    if _coat_d4_ok:
        try:
            d4 = _pd.read_csv(d4_out,dtype={'StudyInstanceUID':str})
            d4 = rsna_frame(d4,_RSNA_TEST_IDS,_KE_LAB,'D4 CoAt')
            rsna_save_predictions('coat_d4_rank',_RSNA_TEST_IDS,d4[lab].to_numpy(),lab)
            if d4.StudyInstanceUID.duplicated().any() or list(d4.columns)!=list(pub.columns):
                raise RuntimeError('D4 schema/UID drift')
            d4 = d4.set_index('StudyInstanceUID').reindex(pub.StudyInstanceUID.astype(str))
            if d4[lab].isna().any().any():
                raise RuntimeError('D4 missing predictions')
            _members.append(('d4_swa3', d4[lab].reset_index(drop=True).rank(method='average',pct=True)))
        except Exception as _exc:
            rsna_event('coat_d4_output_rejected', error=f'{type(_exc).__name__}: {str(_exc)[:500]}')
    if not _members:
        rsna_event('coat_family_unavailable', note='public Raptor kept as the CoAt/Raptor input (flagged)')
        print('[coat-arm] no CoAt family member available; public Raptor retained (flagged)', flush=True)
        return 0
    if len(_members) == 2:
        private_rank = _asset_half_rank_mix(_members[0][1], _members[1][1])
    else:
        rsna_event('coat_family_partial', member=_members[0][0])
        private_rank = _members[0][1]
    family = pub.copy()
    family[lab] = private_rank
    family.to_csv('/kaggle/working/_coat_family_rank.csv', index=False)
    hybrid = pub.copy()
    hybrid[lab] = (1.0 - private_alpha) * public_rank + private_alpha * private_rank
    if not _np.isfinite(hybrid[lab].to_numpy(_np.float64)).all():
        raise RuntimeError('CoAt/Raptor hybrid contains non-finite values')
    tmp = raptor.with_name('.raptor_coat_hybrid.csv')
    hybrid.to_csv(tmp, index=False)
    _o.replace(tmp, raptor)
    raptor.with_name('_coat_raptor_blend_receipt.json').write_text(_j.dumps({
        'contract': 'public_raptor_private_residual_coat_v32_no_inner_rerank_v1', 'inner_rerank': False, 'private_alpha': private_alpha,
        'within_coat': {m: 0.5 for m, _ in _members} if len(_members) == 2 else {_members[0][0]: 1.0}, 'public_raptor_alpha': 1.0 - private_alpha,
        'study_count': len(pub), 'finding_specific_weights': False, 'members': [m for m, _ in _members]}, indent=2, sort_keys=True) + '\n')
    return len(pub)

_coat_n = _coat_substitute()
print(f'[coat-arm] residual top3 / D4 SWA3 equal rank family; BTKD32 no-inner-rerank; {_coat_n} studies',flush=True)

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
_coatnet_weight.update({
    'ACL': 0.75,
    'Medial Meniscus': 0.80,
    'Lateral Meniscus': 1.00,
    'Lateral OA': 0.75,
    'Fracture': 0.75,
})
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

#### CELL 24
_GOLD58_IDS_B64 = 'WyIxLjIuODI2LjAuMS4zNjgwMDQzLjguNDk4LjEwMDk1Njg3NzQ3Mjk1NDEwMzk2NTEwNTM4NTIwNTk0NjQ5MTQ5IiwgIjEuMi44MjYuMC4xLjM2ODAwNDMuOC40OTguMTAxNzA4OTg2MTU4Njc2NzMwMjg2OTY1MDUyNDg4MzkwMjgyNjkiLCAiMS4yLjgyNi4wLjEuMzY4MDA0My44LjQ5OC4xMDMwNjE1OTExMzMyNDgxMTUzODcwMzc4ODA4MDgzNjc1MjA1MiIsICIxLjIuODI2LjAuMS4zNjgwMDQzLjguNDk4LjExMjg3OTM3NzI5MTk2OTU4NDI2NTM4MDg3NDM5MTAyMDE3NTgwIiwgIjEuMi44MjYuMC4xLjM2ODAwNDMuOC40OTguMTEzODIwMjEzOTM4MDMzODk5NTE5NjQwMDU5ODMwMDIyMDkyMzgiLCAiMS4yLjgyNi4wLjEuMzY4MDA0My44LjQ5OC4xMTU0ODA0NTcxNTI2NDE1MTYzMjE1MzA0MDA4OTg4MjcwMTkzNSIsICIxLjIuODI2LjAuMS4zNjgwMDQzLjguNDk4LjExNTU3NjIwNTU5MTkxNDY5MDY5MTMwODI3OTU5MDk4MzM1ODQwIiwgIjEuMi44MjYuMC4xLjM2ODAwNDMuOC40OTguMTE3NzEzOTM4MjQ1MTk4OTI3OTcxMTQ3NzM0MDg1ODM5NzY3NTYiLCAiMS4yLjgyNi4wLjEuMzY4MDA0My44LjQ5OC4xMTg1MTQxMjkyMzAxNjA0NDk0ODEwMTY5ODAxNTk3NDgxMDYwNCIsICIxLjIuODI2LjAuMS4zNjgwMDQzLjguNDk4LjExOTE1OTM3OTgyNjg0OTg4MDczNjQ0MjA5NjA2OTA3MTY5NTgxIiwgIjEuMi44MjYuMC4xLjM2ODAwNDMuOC40OTguMTI0NDgwNzk2NDYzNTk4OTIyNTI0NDEyMDgyNTg4MzY1NTY5NDUiLCAiMS4yLjgyNi4wLjEuMzY4MDA0My44LjQ5OC4xMjUwNTAzNTQyNDA5MzYwNDI2OTUxNTMyODkzMTQ4ODc3MDgxOSIsICIxLjIuODI2LjAuMS4zNjgwMDQzLjguNDk4LjEyNjA2NjU3MjI2NTY4NTU4MzQwNzk3MTkzMTY3NDg4MTExOTczIiwgIjEuMi44MjYuMC4xLjM2ODAwNDMuOC40OTguMTI4MDEzMDg4NDQzOTg2MTQ2ODc5MDQ0NDc2MzM0MzIxOTc0OTIiLCAiMS4yLjgyNi4wLjEuMzY4MDA0My44LjQ5OC4xMjk3ODUxMDE1NzIwMjg1MjIwMjc3Njg5OTkxMDUyOTE3NDgwMyIsICIxLjIuODI2LjAuMS4zNjgwMDQzLjguNDk4LjEzMjY3NzgwMzU2MjQ1MTIwMDUyNDExNTE3MDUzMzIyODc0ODkxIiwgIjEuMi44MjYuMC4xLjM2ODAwNDMuOC40OTguMTMzMzUxMjk4ODE3Mzc3MzE0MTAwODEwMDI2Mjc3MjkyMDA5MDMiLCAiMS4yLjgyNi4wLjEuMzY4MDA0My44LjQ5OC4xNTU5MzYzODg5NzI5MjA1NzM1Njg2NDA2MDQ2NjEyMDI1MzMwOSIsICIxLjIuODI2LjAuMS4zNjgwMDQzLjguNDk4LjE2MDYwMTE5Mzg5MDYwNDk3MTM2MjMxMjE3NzE4OTIxNDgyMTkyIiwgIjEuMi44MjYuMC4xLjM2ODAwNDMuOC40OTguMTc4NDQ1NDY3NjU5MDczMjE2NDk5OTcwOTQ4Njc3OTExMDI3MTEiLCAiMS4yLjgyNi4wLjEuMzY4MDA0My44LjQ5OC4xODM5MjUwOTQ5NzE3MDYxNjk4Mzk3NzMxOTUyODAzNjU3MzM3OCIsICIxLjIuODI2LjAuMS4zNjgwMDQzLjguNDk4LjIyMTA5NzM5OTYyMjI0MzA5NDE4ODc0NTM4OTk0NDM2OTAzNDA0IiwgIjEuMi44MjYuMC4xLjM2ODAwNDMuOC40OTguMjU2OTU5NjYxODYxMjkzOTUwNDk2ODc3NDcxNTY1NzA0NjY2NDUiLCAiMS4yLjgyNi4wLjEuMzY4MDA0My44LjQ5OC4yNjc5MDcwMjM3OTUwNjE5MDkzNjgzNDQ0NzIwMzQ0ODg4MjQ2NSIsICIxLjIuODI2LjAuMS4zNjgwMDQzLjguNDk4LjI3NDM3MjYzODQzNDQ2ODc5OTMyMjgxODk3NDQ2MTAwMTM0OTI0IiwgIjEuMi44MjYuMC4xLjM2ODAwNDMuOC40OTguMjg5MjUzNDU4NTk0OTgzNTEyMDM0Nzc2NDI3NDE5MDg0NTI2MDgiLCAiMS4yLjgyNi4wLjEuMzY4MDA0My44LjQ5OC4yOTc2NDg2ODA5MTIzODI4NzA3MjE2NjgyMzUyMjg1MzQxOTU1MCIsICIxLjIuODI2LjAuMS4zNjgwMDQzLjguNDk4LjMwMjQ2MDc5NzE4NDcxNTUyOTcyMTMwNTcyNDQ0MzgzMDc5OTExIiwgIjEuMi44MjYuMC4xLjM2ODAwNDMuOC40OTguMzIzMjE4MzA3NzY3Mzk2ODk2NDU3MDA1NTUwNTU5NTU3MjU5NDUiLCAiMS4yLjgyNi4wLjEuMzY4MDA0My44LjQ5OC4zNzA0MDkxMDE5NjQ1OTA1NDU0MzMxMDMzNTA2NDE2NzIxMjc0MiIsICIxLjIuODI2LjAuMS4zNjgwMDQzLjguNDk4LjM3MzkyMTI3MzA3ODY3MjM4NTI0MjUxODY0NzUxNzEzMTQzNjgzIiwgIjEuMi44MjYuMC4xLjM2ODAwNDMuOC40OTguMzk1MDA1NTMzNzI1MTcyOTA4MjM4MTU2MDY4Mjk1MTgzMDU1MDAiLCAiMS4yLjgyNi4wLjEuMzY4MDA0My44LjQ5OC40MTYwMDQ5ODc4MzM4NDg2NDExMTE3OTE3NTUxODYxNDgzNzEyNSIsICIxLjIuODI2LjAuMS4zNjgwMDQzLjguNDk4LjQ3OTIxNzUzNDgwNTkyNTk1MTk4MDUyNDA3ODUwNTY4Njc3MTg3IiwgIjEuMi44MjYuMC4xLjM2ODAwNDMuOC40OTguNDgzNDg2MTUzNDk0MTg2MTU0Njk4NzU4MDE0MzkzNDg0MjQyNzQiLCAiMS4yLjgyNi4wLjEuMzY4MDA0My44LjQ5OC40ODk0NjU4MDk0NjY2NTAzMTg1MjM1NTAwNTI5NDczNDEwMTEzMiIsICIxLjIuODI2LjAuMS4zNjgwMDQzLjguNDk4LjU0OTc3NTgxMzIzOTMxMjc3ODE3MDc0NzQxMTgyNjcwMDgwNDUwIiwgIjEuMi44MjYuMC4xLjM2ODAwNDMuOC40OTguNTY1MTI1NjQ0NjQzMDE1NDQ5NjA0NDY0MTkxNDMxMzU0NDczNjMiLCAiMS4yLjgyNi4wLjEuMzY4MDA0My44LjQ5OC41OTQ4Mzk5OTA2NzgxNjE1Mzc1OTc4NTc4OTY1NDc2NjgyNjcxMCIsICIxLjIuODI2LjAuMS4zNjgwMDQzLjguNDk4LjYyNDY1NTk1Mzc2NDg5MjExMjc0MjI1MzEyNDUzMjE2NTU5Mzk1IiwgIjEuMi44MjYuMC4xLjM2ODAwNDMuOC40OTguNjI1NDkzNTQ2Nzc2Mzg0MDM4NDU5MDQ1NTYxNDk4NjgzNjcyMzYiLCAiMS4yLjgyNi4wLjEuMzY4MDA0My44LjQ5OC42NDQwODYwOTI1NjE2MzEyNzI3ODQzNTQ4NDIxNzY4MzI3MjkxMCIsICIxLjIuODI2LjAuMS4zNjgwMDQzLjguNDk4LjY0NzAzNTA2NzcyMTY3Nzk4NDY5MDQ4NDYwNzkxNDcyNDY1MDM5IiwgIjEuMi44MjYuMC4xLjM2ODAwNDMuOC40OTguNjU3MDg5MDUxMTg2MzMzMzk3NzExODE4NTc3ODEwNjM3ODQzMjciLCAiMS4yLjgyNi4wLjEuMzY4MDA0My44LjQ5OC42NzE4ODEyMTU0NDA2MzcyMzgzNzY2OTk4MzU5NzQ1Mzk0MTc3NCIsICIxLjIuODI2LjAuMS4zNjgwMDQzLjguNDk4LjY5MzkyMzQ4Mzg1Mjc0MTI1Mzg1MjkwMDE1NDA0MTQ0NjM5MDAyIiwgIjEuMi44MjYuMC4xLjM2ODAwNDMuOC40OTguNzI4NTMzMzMyMjAwNDM3OTQ5MDQ4NTYxMzg1NjEwOTUxNzE5MjEiLCAiMS4yLjgyNi4wLjEuMzY4MDA0My44LjQ5OC43MzUyNzUzMDY4Njg1MzkxMTEyNDQzMTU0OTMxNzAzMjY2MjIyMCIsICIxLjIuODI2LjAuMS4zNjgwMDQzLjguNDk4LjczOTI2NDQzNzI5Nzg2MTY1NjI4ODQ4ODQzNzA3NTMyODM5OTk1IiwgIjEuMi44MjYuMC4xLjM2ODAwNDMuOC40OTguNzUxODc0MzQyNDgzNTY3NzQyNzc1MjY5ODUzMjkzNDYxMjUxOTAiLCAiMS4yLjgyNi4wLjEuMzY4MDA0My44LjQ5OC43NzM2MjcxODI5ODU1MDI3NjY3OTM1MDg1NTk2MzQ1MTg1NTAwMyIsICIxLjIuODI2LjAuMS4zNjgwMDQzLjguNDk4Ljc4NTEyMjE1NTE5MTc3ODUwOTIzMjc5MjM1MzQ2OTY4Njc2ODI4IiwgIjEuMi44MjYuMC4xLjM2ODAwNDMuOC40OTguODIxNjY5NDM1NTI3NjQ0MzkxMzgzMzM1MDQ0NTYxMzk4OTAyNTQiLCAiMS4yLjgyNi4wLjEuMzY4MDA0My44LjQ5OC44Njk2ODYwMDIzOTcyNDMxMDY3ODkwNTMxMTI0NDk0NTQ2NDAzNyIsICIxLjIuODI2LjAuMS4zNjgwMDQzLjguNDk4Ljg4MDc3NDE4NjM5MzAxMTc0NDA5OTI2NzgxMzI5NjEzNTcwNDM1IiwgIjEuMi44MjYuMC4xLjM2ODAwNDMuOC40OTguOTAyODM1NjUzODEwNDIwODE3Njg1ODc4OTQ1OTY5NzA1NTI3NjciLCAiMS4yLjgyNi4wLjEuMzY4MDA0My44LjQ5OC45NDQzMzc1MzQ3MTg5MDMwNjEwODA4OTU1Nzc2MTIzMTczOTMxMiIsICIxLjIuODI2LjAuMS4zNjgwMDQzLjguNDk4Ljk3Mjc0NzIwMjU3NjM0NTg0MDcxNTAwNjQ5Mjc1MjE3NTIxNjYyIl0='
_GOLD58_CELLS_B64 = {"2": "aW1wb3J0IG51bXB5IGFzIG5wCgpkZWYgX2RlbnNlX2FsbG9jYXRlKGNhcGFjaXRpZXMsIHByb3BvcnRpb25zLCB0YXJnZXQpOgogICAgY2FwYWNpdGllcyA9IG5wLmFzYXJyYXkoY2FwYWNpdGllcywgZHR5cGU9bnAuaW50NjQpCiAgICBwcm9wb3J0aW9ucyA9IG5wLmFzYXJyYXkocHJvcG9ydGlvbnMsIGR0eXBlPW5wLmZsb2F0NjQpCiAgICBpZiBjYXBhY2l0aWVzLm5kaW0gIT0gMSBvciBjYXBhY2l0aWVzLnNoYXBlICE9IHByb3BvcnRpb25zLnNoYXBlIG9yIG5wLmFueShjYXBhY2l0aWVzIDwgMCkgb3IgKG5vdCBucC5pc2Zpbml0ZShwcm9wb3J0aW9ucykuYWxsKCkpIG9yIG5wLmFueShwcm9wb3J0aW9ucyA8PSAwKSBvciAoaW50KHRhcmdldCkgPCAxKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCdpbnZhbGlkIGNhcGFjaXR5LWF3YXJlIHNhbXBsaW5nIGFsbG9jYXRpb24nKQogICAgaWRlYWwgPSBpbnQodGFyZ2V0KSAqIHByb3BvcnRpb25zIC8gcHJvcG9ydGlvbnMuc3VtKCkKICAgIGluaXRpYWwgPSBucC5mbG9vcihpZGVhbCkuYXN0eXBlKG5wLmludDY0KQogICAgZm9yIGkgaW4gbnAuYXJnc29ydCgtKGlkZWFsIC0gaW5pdGlhbCksIGtpbmQ9J3N0YWJsZScpWzppbnQodGFyZ2V0KSAtIGludChpbml0aWFsLnN1bSgpKV06CiAgICAgICAgaW5pdGlhbFtpXSArPSAxCiAgICBxdW90YXMgPSBucC5taW5pbXVtKGluaXRpYWwsIGNhcGFjaXRpZXMpCiAgICB3YW50ZWQgPSBtaW4oaW50KHRhcmdldCksIGludChjYXBhY2l0aWVzLnN1bSgpKSkKICAgIGRlZmljaXQgPSBucC56ZXJvcyhsZW4ocXVvdGFzKSwgZHR5cGU9bnAuZmxvYXQ2NCkKICAgIHdoaWxlIGludChxdW90YXMuc3VtKCkpIDwgd2FudGVkOgogICAgICAgIGRvbm9ycyA9IHF1b3RhcyA8IGNhcGFjaXRpZXMKICAgICAgICB3ZWlnaHRzID0gbnAud2hlcmUoZG9ub3JzLCBwcm9wb3J0aW9ucywgMC4wKQogICAgICAgIGRlZmljaXRbZG9ub3JzXSArPSB3ZWlnaHRzW2Rvbm9yc10gLyB3ZWlnaHRzLnN1bSgpCiAgICAgICAgY2hvc2VuID0gaW50KG5wLmFyZ21heChucC53aGVyZShkb25vcnMsIGRlZmljaXQsIC1ucC5pbmYpKSkKICAgICAgICBxdW90YXNbY2hvc2VuXSArPSAxCiAgICAgICAgZGVmaWNpdFtjaG9zZW5dIC09IDEKICAgIGFzc2VydCBpbnQocXVvdGFzLnN1bSgpKSA9PSB3YW50ZWQgYW5kIG5wLmFsbChxdW90YXMgPD0gY2FwYWNpdGllcykKICAgIHJldHVybiBxdW90YXMKCmRlZiBfZGVuc2VfdW5pcXVlX2xpbnNwYWNlKGNhcGFjaXR5LCBjb3VudCk6CiAgICBpZiBub3QgMCA8PSBjb3VudCA8PSBjYXBhY2l0eToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCdjYW5ub3QgY3JlYXRlIHVuaXF1ZSBwaWNrcyBiZXlvbmQgcGh5c2ljYWwgY2FwYWNpdHknKQogICAgcmVzdWx0ID0gbnAubGluc3BhY2UoMCwgY2FwYWNpdHkgLSAxLCBjb3VudCkucm91bmQoKS5hc3R5cGUobnAuaW50NjQpCiAgICBpZiBjb3VudCBhbmQgKHJlc3VsdC5taW4oKSA8IDAgb3IgcmVzdWx0Lm1heCgpID49IGNhcGFjaXR5KToKICAgICAgICByYWlzZSBBc3NlcnRpb25FcnJvcignb3V0LW9mLXJhbmdlIHNvdXJjZScpCiAgICBhc3NlcnQgbGVuKG5wLnVuaXF1ZShyZXN1bHQpKSA9PSBjb3VudAogICAgcmV0dXJuIHJlc3VsdAoKZGVmIF9kZW5zZV9jb2F0X3NwZWNzKHN0dWR5KToKICAgIGltcG9ydCByYXB0b3JfbGlnaHQyMjQgYXMgc2FtcGxlcgogICAgY2FwYWNpdGllcyA9IFtdCiAgICBmb3Igcm93IGluIHN0dWR5LnNlcmllc19yb3dzOgogICAgICAgIGlmIGludChyb3cpIDwgMDoKICAgICAgICAgICAgY2FwYWNpdGllcy5hcHBlbmQoMCkKICAgICAgICBlbHNlOgogICAgICAgICAgICBjZW50ZXJzLCBfLCBfLCBfID0gc2FtcGxlci5fY2FuZGlkYXRlX2NlbnRlcnMoaW50KHJvdyksIHN0dWR5Lm9mZnNldHMsIHN0dWR5LnZhbGlkLCBzdHVkeS5zb3VyY2VfZGVwdGgpCiAgICAgICAgICAgIGNhcGFjaXRpZXMuYXBwZW5kKGxlbihjZW50ZXJzKSkKICAgIGlmIHN1bShjYXBhY2l0aWVzKSA8IDE6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcignbm8gdXNhYmxlIENvQXQgdHJpcGxldHMnKQogICAgcXVvdGFzID0gX2RlbnNlX2FsbG9jYXRlKGNhcGFjaXRpZXMsICgyMCwgMTIsIDEyLCA4LCAxNiwgOCksIDk0KQogICAgc3BlY3MgPSBzYW1wbGVyLnNhbXBsZV9ldmFsX3dpbmRvd3Moc3R1ZHkuc2VyaWVzX3Jvd3MsIHN0dWR5Lm9mZnNldHMsIHN0dWR5LnZhbGlkLCBzdHVkeS5zb3VyY2VfZGVwdGgsIGNvdW50PU5vbmUsIHNsb3RfYnVkZ2V0cz1ucC5tYXhpbXVtKHF1b3RhcywgMSkudG9saXN0KCkpCiAgICBpZiBsZW4oc3BlY3MpICE9IG1pbig5NCwgc3VtKGNhcGFjaXRpZXMpKToKICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoJ2NhcGFjaXR5LWF3YXJlIHNhbXBsaW5nIHdpbmRvdy1jb3VudCBkcmlmdCcpCiAgICBrZXlzID0gWyhzLnNlcmllc19yb3csIHMuY2VudGVyX2xvY2FsKSBmb3IgcyBpbiBzcGVjc10KICAgIGlmIGxlbihrZXlzKSAhPSBsZW4oc2V0KGtleXMpKToKICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoJ2NhcGFjaXR5LWF3YXJlIHNhbXBsaW5nIHJlcGVhdGVkIGEgbmF0aXZlIGNlbnRlcicpCiAgICBmb3IgcyBpbiBzcGVjczoKICAgICAgICBzdGFydCA9IGludChzdHVkeS5vZmZzZXRzW3Muc2VyaWVzX3Jvd10pCiAgICAgICAgaWYgbm90IG5wLmFsbChzdHVkeS52YWxpZFtzdGFydCArIG5wLmFzYXJyYXkocy5sb2NhbF9pbmRpY2VzKV0pOgogICAgICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoJ2NhcGFjaXR5LWF3YXJlIHNhbXBsaW5nIGludmFsaWQgc291cmNlIGNoYW5uZWwnKQogICAgcmV0dXJuIHNwZWNzCklORkVSUkVEOTZfV0lEVEhTID0gKDI3LCAyMSwgMTgsIDEyLCAxOCkKCmRlZiBfZGVuc2VfcXVvdGFzKGNhcGFjaXRpZXMpIC0+IG5wLm5kYXJyYXk6CiAgICBjYXBhY2l0aWVzID0gbnAuYXNhcnJheShjYXBhY2l0aWVzLCBkdHlwZT1ucC5pbnQ2NCkKICAgIGJhc2UgPSBucC5hc2FycmF5KElORkVSUkVEOTZfV0lEVEhTLCBkdHlwZT1ucC5pbnQ2NCkKICAgIGlmIGNhcGFjaXRpZXMuc2hhcGUgIT0gKDUsKSBvciBib29sKChjYXBhY2l0aWVzIDwgMCkuYW55KCkpOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoJ2V4cGVjdGVkIGZpdmUgbm9ubmVnYXRpdmUgc291cmNlIGNhcGFjaXRpZXMnKQogICAgcXVvdGFzID0gbnAubWluaW11bShiYXNlLCBjYXBhY2l0aWVzKQogICAgdGFyZ2V0ID0gbWluKDk2LCBpbnQoY2FwYWNpdGllcy5zdW0oKSkpCiAgICBkZWZpY2l0ID0gbnAuemVyb3MoNSwgZHR5cGU9bnAuZmxvYXQ2NCkKICAgIHdoaWxlIGludChxdW90YXMuc3VtKCkpIDwgdGFyZ2V0OgogICAgICAgIGRvbm9ycyA9IHF1b3RhcyA8IGNhcGFjaXRpZXMKICAgICAgICB3ZWlnaHRzID0gbnAud2hlcmUoZG9ub3JzLCBiYXNlLCAwKS5hc3R5cGUobnAuZmxvYXQ2NCkKICAgICAgICBkZWZpY2l0W2Rvbm9yc10gKz0gd2VpZ2h0c1tkb25vcnNdIC8gd2VpZ2h0cy5zdW0oKQogICAgICAgIHNlbGVjdGVkID0gaW50KG5wLmFyZ21heChucC53aGVyZShkb25vcnMsIGRlZmljaXQsIC1ucC5pbmYpKSkKICAgICAgICBxdW90YXNbc2VsZWN0ZWRdICs9IDEKICAgICAgICBkZWZpY2l0W3NlbGVjdGVkXSAtPSAxCiAgICBpZiBpbnQocXVvdGFzLnN1bSgpKSAhPSB0YXJnZXQgb3IgYm9vbCgocXVvdGFzID4gY2FwYWNpdGllcykuYW55KCkpOgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcignYWRkaXRpb25hbCBxdW90YSBhbGxvY2F0aW9uIGNoYW5nZWQnKQogICAgcmV0dXJuIHF1b3RhcwoKZGVmIF9kZW5zZV9zdGFjayhyZWZlcmVuY2UsIG9mZnNldHMsIHZhbGlkLCBzb3VyY2VfZGVwdGgpOgogICAgZnJvbSB0eXBlcyBpbXBvcnQgU2ltcGxlTmFtZXNwYWNlCiAgICBmcm9tIHJhcHRvcl9nbG9iYWxfc3RhY2tfc21hcnQzMzYgaW1wb3J0IEZpeGVkR2xvYmFsU3RhY2sKICAgIGlmIHR1cGxlKHJlZmVyZW5jZS5zbG90X3dpZHRocykgIT0gdHVwbGUoSU5GRVJSRUQ5Nl9XSURUSFMpOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoJ2NhcGFjaXR5LWF3YXJlIHNhbXBsaW5nIGlzIHBpbm5lZCB0byB0aGUgR2xvYmFsOTYgcmVmZXJlbmNlIGxheW91dCcpCiAgICBwb29scyA9IFtdCiAgICBhdWRpdCA9IFtdCiAgICBiZWdpbiA9IDAKICAgIGZvciBzbG90LCB3aWR0aCBpbiBlbnVtZXJhdGUocmVmZXJlbmNlLnNsb3Rfd2lkdGhzLCAxKToKICAgICAgICBlbmQgPSBiZWdpbiArIHdpZHRoCiAgICAgICAgb2xkID0gcmVmZXJlbmNlLmdsb2JhbF9pbmRpY2VzW2JlZ2luOmVuZF0KICAgICAgICByb3cgPSBpbnQocmVmZXJlbmNlLnNvdXJjZV9zZXJpZXNfcm93c1tiZWdpbl0pCiAgICAgICAgaW5mbyA9IHsnc2xvdCc6IHNsb3QsICdvcmlnaW5hbF93aWR0aCc6IHdpZHRoLCAnc291cmNlX3Jvdyc6IHJvd30KICAgICAgICBhdWRpdC5hcHBlbmQoaW5mbykKICAgICAgICBpZiByb3cgPCAwIG9yIGJvb2woKG9sZCA8IDApLmFsbCgpKToKICAgICAgICAgICAgcG9vbHMuYXBwZW5kKChucC5lbXB0eSgwLCBkdHlwZT1ucC5pbnQ2NCksIG5wLmVtcHR5KDAsIGR0eXBlPW5wLmZsb2F0MzIpKSkKICAgICAgICAgICAgYmVnaW4gPSBlbmQKICAgICAgICAgICAgY29udGludWUKICAgICAgICBzdGFydCwgc3RvcCA9IG1hcChpbnQsIG9mZnNldHNbcm93OnJvdyArIDJdKQogICAgICAgIGlmIG5vdCAwIDw9IHN0YXJ0IDwgc3RvcCA8PSBsZW4odmFsaWQpID09IGxlbihzb3VyY2VfZGVwdGgpOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCdzb3VyY2Ugb2Zmc2V0IGRyaWZ0JykKICAgICAgICBkZXB0aCA9IG5wLmFzYXJyYXkoc291cmNlX2RlcHRoW3N0YXJ0OnN0b3BdLCBkdHlwZT1ucC5mbG9hdDMyKQogICAgICAgIGtlZXAgPSBucC5hc2FycmF5KHZhbGlkW3N0YXJ0OnN0b3BdLCBkdHlwZT1ib29sKSAmIG5wLmlzZmluaXRlKGRlcHRoKSAmIChkZXB0aCA+PSAwLjAyKSAmIChkZXB0aCA8PSAwLjk4KQogICAgICAgIGVsaWdpYmxlID0gbnAuZmxhdG5vbnplcm8oa2VlcCkuYXN0eXBlKG5wLmludDY0KSArIHN0YXJ0CiAgICAgICAgcmV2ZXJzZSA9IGJvb2wob2xkWzBdID4gb2xkWy0xXSBvciAob2xkWzBdID09IG9sZFstMV0gYW5kIG5wLmZsb2F0MzIoc291cmNlX2RlcHRoW29sZFswXV0pICE9IHJlZmVyZW5jZS5jYW5vbmljYWxfZGVwdGhzW2JlZ2luXSkpCiAgICAgICAgaWYgcmV2ZXJzZToKICAgICAgICAgICAgZWxpZ2libGUgPSBlbGlnaWJsZVs6Oi0xXS5jb3B5KCkKICAgICAgICBkZXB0aCA9IG5wLmFzYXJyYXkoc291cmNlX2RlcHRoW2VsaWdpYmxlXSwgZHR5cGU9bnAuZmxvYXQzMikKICAgICAgICBpZiByZXZlcnNlOgogICAgICAgICAgICBkZXB0aCA9ICgxLjAgLSBkZXB0aCkuYXN0eXBlKG5wLmZsb2F0MzIpCiAgICAgICAgcG9vbHMuYXBwZW5kKChlbGlnaWJsZSwgZGVwdGgpKQogICAgICAgIGJlZ2luID0gZW5kCiAgICBxdW90YXMgPSBfZGVuc2VfcXVvdGFzKFtsZW4ocG9vbFswXSkgZm9yIHBvb2wgaW4gcG9vbHNdKQogICAgaWYgaW50KHF1b3Rhcy5zdW0oKSkgPCAzOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoJ3N0dWR5IGhhcyBmZXdlciB0aGFuIHRocmVlIHVzYWJsZSB1bmlxdWUgc291cmNlIHNsaWNlcycpCiAgICBwYXJ0cyA9IHtrZXk6IFtdIGZvciBrZXkgaW4gKCdnbG9iYWxfaW5kaWNlcycsICdub21pbmFsX3Nsb3RzJywgJ2Nhbm9uaWNhbF9kZXB0aHMnLCAnbm9taW5hbF9zdGVwcycsICdzb3VyY2Vfc2VyaWVzX3Jvd3MnLCAnc291cmNlX3Nsb3RfaWRzJyl9CiAgICBmb3Igc2xvdCwgKChlbGlnaWJsZSwgZGVwdGgpLCBxdW90YSwgaW5mbykgaW4gZW51bWVyYXRlKHppcChwb29scywgcXVvdGFzLCBhdWRpdCksIDEpOgogICAgICAgIHF1b3RhID0gaW50KHF1b3RhKQogICAgICAgIGluZm8udXBkYXRlKGVsaWdpYmxlPWxlbihlbGlnaWJsZSksIGFsbG9jYXRlZF93aWR0aD1xdW90YSwgYWRkaXRpb25hbD1tYXgoMCwgcXVvdGEgLSBpbmZvWydvcmlnaW5hbF93aWR0aCddKSkKICAgICAgICBpbmZvWydyZWFzb24nXSA9ICdtaXNzaW5nX25vX3BhZGRpbmcnIGlmIG5vdCBxdW90YSBlbHNlICdzaG9ydF9zbG90X2FsbF91bmlxdWUnIGlmIGxlbihlbGlnaWJsZSkgPCBpbmZvWydvcmlnaW5hbF93aWR0aCddIGVsc2UgJ3VuaWZvcm1fbmF0aXZlX2NlbnRlcnMnIGlmIGluZm9bJ2FkZGl0aW9uYWwnXSBlbHNlICd1bmlmb3JtX25hdGl2ZV9jZW50ZXJzJwogICAgICAgIHBvc2l0aW9ucyA9IG5wLmxpbnNwYWNlKDAsIGxlbihlbGlnaWJsZSkgLSAxLCBxdW90YSkucm91bmQoKS5hc3R5cGUobnAuaW50NjQpCiAgICAgICAgaWYgbGVuKHBvc2l0aW9ucykgIT0gcXVvdGEgb3Igbm90IGJvb2woKG5wLmRpZmYocG9zaXRpb25zKSA+IDApLmFsbCgpKToKICAgICAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKCdjYXBhY2l0eS1hd2FyZSBzYW1wbGluZyBzbG90IHNlbGVjdGlvbiByZXBlYXRlZCBhIHNvdXJjZScpCiAgICAgICAgbWFzayA9IHJlZmVyZW5jZS5ub21pbmFsX3Nsb3RzID09IHNsb3QKICAgICAgICBzb3VyY2Vfc2xvdCA9IGludChyZWZlcmVuY2Uuc291cmNlX3Nsb3RfaWRzW21hc2tdWzBdKQogICAgICAgIHBhcnRzWydnbG9iYWxfaW5kaWNlcyddLmFwcGVuZChlbGlnaWJsZVtwb3NpdGlvbnNdKQogICAgICAgIHBhcnRzWydjYW5vbmljYWxfZGVwdGhzJ10uYXBwZW5kKGRlcHRoW3Bvc2l0aW9uc10pCiAgICAgICAgcGFydHNbJ25vbWluYWxfc2xvdHMnXS5hcHBlbmQobnAuZnVsbChxdW90YSwgc2xvdCwgZHR5cGU9bnAuaW50OCkpCiAgICAgICAgcGFydHNbJ25vbWluYWxfc3RlcHMnXS5hcHBlbmQobnAuZnVsbChxdW90YSwgMC45NiAvIG1heChxdW90YSAtIDEsIDEpLCBkdHlwZT1ucC5mbG9hdDMyKSkKICAgICAgICBwYXJ0c1snc291cmNlX3Nlcmllc19yb3dzJ10uYXBwZW5kKG5wLmZ1bGwocXVvdGEsIGluZm9bJ3NvdXJjZV9yb3cnXSwgZHR5cGU9bnAuaW50MzIpKQogICAgICAgIHBhcnRzWydzb3VyY2Vfc2xvdF9pZHMnXS5hcHBlbmQobnAuZnVsbChxdW90YSwgc291cmNlX3Nsb3QsIGR0eXBlPW5wLmludDgpKQogICAgcmVzdWx0ID0gRml4ZWRHbG9iYWxTdGFjaygqKntrZXk6IG5wLmNvbmNhdGVuYXRlKHZhbHVlKSBmb3Iga2V5LCB2YWx1ZSBpbiBwYXJ0cy5pdGVtcygpfSwgc2xvdF93aWR0aHM9dHVwbGUoKGludCh2YWx1ZSkgZm9yIHZhbHVlIGluIHF1b3RhcykpKQogICAgaWYgbGVuKG5wLnVuaXF1ZShyZXN1bHQuZ2xvYmFsX2luZGljZXMpKSAhPSBsZW4ocmVzdWx0Lmdsb2JhbF9pbmRpY2VzKSBvciBib29sKChyZXN1bHQuZ2xvYmFsX2luZGljZXMgPCAwKS5hbnkoKSk6CiAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKCdjYXBhY2l0eS1hd2FyZSBzYW1wbGluZyBvdXRwdXQgY29udGFpbnMgYSByZXBlYXRlZC9taXNzaW5nIHNvdXJjZScpCiAgICByZXR1cm4gKHJlc3VsdCwgYXVkaXQpCg==", "3": "ZnJvbSBfX2Z1dHVyZV9fIGltcG9ydCBhbm5vdGF0aW9ucwppbXBvcnQgdGltZSBhcyBfc3RhcnR1cF90aW1lCl9TVEFSVFVQX1QwID0gX3N0YXJ0dXBfdGltZS5wZXJmX2NvdW50ZXIoKQppbXBvcnQgb3MKZm9yIF92IGluICgnT01QX05VTV9USFJFQURTJywgJ09QRU5CTEFTX05VTV9USFJFQURTJywgJ01LTF9OVU1fVEhSRUFEUycpOgogICAgb3MuZW52aXJvbi5zZXRkZWZhdWx0KF92LCAnNCcpCmltcG9ydCBnYwppbXBvcnQgaGFzaGxpYgppbXBvcnQganNvbgppbXBvcnQgcmUKaW1wb3J0IHRpbWUKaW1wb3J0IHRyYWNlYmFjawppbXBvcnQgdGhyZWFkaW5nCmZyb20gY29uY3VycmVudC5mdXR1cmVzIGltcG9ydCBUaHJlYWRQb29sRXhlY3V0b3IKZnJvbSBwYXRobGliIGltcG9ydCBQYXRoCmltcG9ydCBudW1weSBhcyBucAppbXBvcnQgcGFuZGFzIGFzIHBkCmltcG9ydCBweWRpY29tCmltcG9ydCB0b3JjaAppbXBvcnQgdG9yY2gubm4gYXMgbm4KaW1wb3J0IHRvcmNoLm5uLmZ1bmN0aW9uYWwgYXMgRgoKZGVmIF9jdWRhX2V4ZWN1dGlvbl9wcm9iZShpbmRleCk6CiAgICBkZXYgPSB0b3JjaC5kZXZpY2UoZidjdWRhOntpbmRleH0nKQogICAgdHJ5OgogICAgICAgIG1ham9yLCBtaW5vciA9IHRvcmNoLmN1ZGEuZ2V0X2RldmljZV9jYXBhYmlsaXR5KGluZGV4KQogICAgICAgIHByb2JlID0gbm4uQ29udjJkKDMsIDQsIGtlcm5lbF9zaXplPTMsIHBhZGRpbmc9MSkuZXZhbCgpLnRvKGRldikKICAgICAgICB3aXRoIHRvcmNoLmluZmVyZW5jZV9tb2RlKCk6CiAgICAgICAgICAgIG91dCA9IHByb2JlKHRvcmNoLnplcm9zKCgxLCAzLCAxNiwgMTYpLCBkZXZpY2U9ZGV2KSkKICAgICAgICAgICAgaWYgdHVwbGUob3V0LnNoYXBlKSAhPSAoMSwgNCwgMTYsIDE2KToKICAgICAgICAgICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcihmJ3VuZXhwZWN0ZWQgQ1VEQSBwcm9iZSBzaGFwZSB7dHVwbGUob3V0LnNoYXBlKX0nKQogICAgICAgIHRvcmNoLmN1ZGEuc3luY2hyb25pemUoaW5kZXgpCiAgICAgICAgcHJpbnQoZidjdWRhOntpbmRleH0gcHJvYmUgUEFTUyAoY29tcHV0ZSB7bWFqb3J9LnttaW5vcn0pJykKICAgICAgICBkZWwgcHJvYmUsIG91dAogICAgICAgIHRvcmNoLmN1ZGEuZW1wdHlfY2FjaGUoKQogICAgICAgIHJldHVybiBUcnVlCiAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGV4YzoKICAgICAgICBwcmludChmJ2N1ZGE6e2luZGV4fSBwcm9iZSBGQUlMICh7dHlwZShleGMpLl9fbmFtZV9ffToge2V4Y30pOyByZWplY3RpbmcgZGV2aWNlJykKICAgICAgICB0cnk6CiAgICAgICAgICAgIHRvcmNoLmN1ZGEuZW1wdHlfY2FjaGUoKQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgIHBhc3MKICAgICAgICByZXR1cm4gRmFsc2UKREVWUyA9IFtdCmlmIHRvcmNoLmN1ZGEuaXNfYXZhaWxhYmxlKCk6CiAgICBERVZTID0gW3RvcmNoLmRldmljZShmJ2N1ZGE6e2l9JykgZm9yIGkgaW4gcmFuZ2UodG9yY2guY3VkYS5kZXZpY2VfY291bnQoKSkgaWYgX2N1ZGFfZXhlY3V0aW9uX3Byb2JlKGkpXQppZiBsZW4oREVWUykgIT0gMjoKICAgIHJhaXNlIFJ1bnRpbWVFcnJvcignVGhpcyBjb21wbGV0ZSBlbnNlbWJsZSByZXF1aXJlcyB0d28gd29ya2luZyBHUFVzOyBubyBDUFUvc2luZ2xlLUdQVSBwYXJ0aWFsIGZhbGxiYWNrJykKaWYgYW55KCdUNCcgbm90IGluIHRvcmNoLmN1ZGEuZ2V0X2RldmljZV9uYW1lKGQpIGZvciBkIGluIERFVlMpOgogICAgcmFpc2UgUnVudGltZUVycm9yKCdTZWxlY3QgdGhlIHJlZmVyZW5jZSBUNCB4MiBhY2NlbGVyYXRvcicpCnByaW50KGYnZGV2aWNlczoge1tzdHIoZCkgZm9yIGQgaW4gREVWU119JykKX1NUQVJUVVBfR1BVX0lNUE9SVF9TID0gX3N0YXJ0dXBfdGltZS5wZXJmX2NvdW50ZXIoKSAtIF9TVEFSVFVQX1QwCnByaW50KGYnW3N0YXJ0dXBdIGltcG9ydHMrR1BVIHByZWZsaWdodDoge19TVEFSVFVQX0dQVV9JTVBPUlRfUzouMmZ9cycsIGZsdXNoPVRydWUpClQwID0gdGltZS50aW1lKCkKU0VFRCA9IDIwMjYKbnAucmFuZG9tLnNlZWQoU0VFRCkKdG9yY2gubWFudWFsX3NlZWQoU0VFRCkKVEFSR0VUUyA9IFsnQUNMJywgJ01DTCcsICdNZWRpYWwgTWVuaXNjdXMnLCAnTGF0ZXJhbCBNZW5pc2N1cycsICdNZWRpYWwgT0EnLCAnTGF0ZXJhbCBPQScsICdQRiBPQScsICdFZmZ1c2lvbicsICdTeW5vdml0aXMnLCAiQmFrZXIncyIsICdDb250dXNpb24nLCAnRnJhY3R1cmUnXQpDUk9QX01NID0gMTMwLjAKQ0FDSEVfSU1HID0gMzM2CkdST1VQID0gMwpOX0dST1VQX01BWCA9IDEKQ0FDSEVfRlJBQ1RJT04gPSAwLjQ1CkNBQ0hFX0JVREdFVF9NQVhfR0IgPSAyNC4wCkNBQ0hFX0JVREdFVF9HQiA9IDEyLjAKVEVTVF9TSEFSRSA9IDAuMwpIRFJfVEhSRUFEUyA9IDE2ClBJWF9USFJFQURTID0gMTIKT1JERVJfVEhSRUFEUyA9IDMyCk9SREVSX0JVREdFVF9TID0gNTQwMApSVU5TID0gW3snbmFtZSc6ICdyMjI0JywgJ2ltZyc6IDIyNH0sIHsnbmFtZSc6ICdyMzM2JywgJ2ltZyc6IDMzNn1dCkVQT0NIUyA9IDEwCkJBVENIX1NUVURJRVMgPSA4CkFVR19ST1RfREVHID0gOC4wCkFVR19TQ0FMRSA9IDAuMDgKQVVHX1NISUZUID0gMC4wNQpBVUdfSU5URU5TSVRZID0gMC4xCkxBVF9NSU5fT0ZGU0VUX01NID0gMjAuMApTTElDRV9CQU5EID0gKDAuMiwgMC44KQpSVUxFU19OQVRJVkUgPSB7J29yZGVyJzogJ25vcm1hbCcsICdsYXQnOiAnY2VudHJlJywgJ3Nsb3RfZmFsbGJhY2snOiBGYWxzZSwgJ2RlY29kZV9maWxsJzogJ25lYXJlc3QnfQpSVUxFU19MRUdBQ1kgPSB7J29yZGVyJzogJ2RvbWluYW50X2F4aXMnLCAnbGF0JzogJ2Nvcm5lcl94JywgJ3Nsb3RfZmFsbGJhY2snOiBUcnVlLCAnZGVjb2RlX2ZpbGwnOiAnemVybyd9ClJVTEVTID0gZGljdChSVUxFU19OQVRJVkUpCkxFR0FDWV9MQVRfT0ZGU0VUX01NID0gNS4wCkxSX0hFQUQgPSAwLjAwMQpMUl9CQUNLQk9ORSA9IDhlLTA2ClVORlJFRVpFX0xBU1QgPSA2CldFSUdIVF9ERUNBWSA9IDAuMDIKRVZBTF9CQVRDSCA9IDgKVElNRV9CVURHRVQgPSA4LjAgKiAzNjAwClNMT1RTX1JFQ09WRVJFRCA9IFsoJ1NBR19GTFVJRF9GUycsICdTYWdpdHRhbCcsIFRydWUsIFRydWUpLCAoJ0NPUl9GTFVJRF9GUycsICdDb3JvbmFsJywgVHJ1ZSwgVHJ1ZSksICgnQVhfRkxVSURfRlMnLCAnQXhpYWwnLCBUcnVlLCBUcnVlKSwgKCdTQUdfRkxVSURfTk9GUycsICdTYWdpdHRhbCcsIFRydWUsIEZhbHNlKSwgKCdDT1JfVDEnLCAnQ29yb25hbCcsIEZhbHNlLCBGYWxzZSksICgnU0FHX1QxJywgJ1NhZ2l0dGFsJywgRmFsc2UsIEZhbHNlKV0KU0xPVFNfUFVCTElDID0gWygnU0FHX0ZMVUlEJywgJ1NhZ2l0dGFsJywgTm9uZSwgVHJ1ZSksICgnQ09SX0ZMVUlEJywgJ0Nvcm9uYWwnLCBOb25lLCBUcnVlKSwgKCdBWF9GTFVJRCcsICdBeGlhbCcsIE5vbmUsIFRydWUpLCAoJ1NBR19TVFJVQ1QnLCAnU2FnaXR0YWwnLCBOb25lLCBGYWxzZSksICgnQ09SX1NUUlVDVCcsICdDb3JvbmFsJywgTm9uZSwgRmFsc2UpLCAoJ0FYX1NUUlVDVCcsICdBeGlhbCcsIE5vbmUsIEZhbHNlKV0KU0xPVF9TQ0hFTUUgPSBvcy5lbnZpcm9uLmdldCgnU0xPVF9TQ0hFTUUnLCAncmVjb3ZlcmVkJykKU0xPVFMgPSBTTE9UU19QVUJMSUMgaWYgU0xPVF9TQ0hFTUUgPT0gJ3B1YmxpYycgZWxzZSBTTE9UU19SRUNPVkVSRUQKTl9TTE9UID0gbGVuKFNMT1RTKQpQT09MX1BBUlRTID0geydjbHNfbWVhbic6IDIsICdjbHNfbWVhbl9mb2NhbCc6IDN9ClNMT1RfUFJJT1JfVEFCTEUgPSB7J0FDTCc6ICgwLCAzLCA1KSwgJ01DTCc6ICgxLCA0KSwgJ01lZGlhbCBNZW5pc2N1cyc6ICgwLCAxLCAzLCA0KSwgJ0xhdGVyYWwgTWVuaXNjdXMnOiAoMCwgMSwgMywgNCksICdNZWRpYWwgT0EnOiAoMSwgNCwgNSksICdMYXRlcmFsIE9BJzogKDEsIDQsIDUpLCAnUEYgT0EnOiAoMCwgMiwgNSksICdFZmZ1c2lvbic6ICgwLCAyKSwgJ1N5bm92aXRpcyc6ICgwLCAyKSwgIkJha2VyJ3MiOiAoMCwpLCAnQ29udHVzaW9uJzogKDAsIDEsIDIpLCAnRnJhY3R1cmUnOiAoMCwgMSwgMiwgNCwgNSl9ClNMT1RfUFJJT1JfU1RSRU5HVEggPSAwLjU1CkZBVFNBVF9PUFRTID0geydGUycsICdGQVRTQVQnLCAnRkFUX1NBVCcsICdGU0FUJ30KX1NFUCA9IHJlLmNvbXBpbGUoJ1tfXFwtLl0nKQpfRkFUU0FUX1JYID0gcmUuY29tcGlsZSgnXFxiZnNcXGJ8ZmF0c2F0fGZhdCBzYXR8XFxic3RpclxcYnxcXGJzcGFpclxcYnxcXGJzcGlyXFxifFxcYndlXFxifHdhdGVyIGV4Y2l0fFxcYnRpcm1cXGJ8XFxic3RpbmdcXGJ8XFxiZmF0c3VwXFxifHNtYXJ0IGZhdHxcXGJ3YXRlclxcYicpICAjIDIwMjYtMDktMTc6IEdFIERpeG9uIHdhdGVyIGltYWdlcyAoJ1NNQVJUIEZBVCcsICdXYXRlcjogU01BUlQgRkFUJywgc2VxIEZTRWZ3KSBhcmUgZmF0LXN1cHByZXNzZWQ7IDM3LzQ0MDcgdHJhaW5pbmcgc3R1ZGllcyBoYWQgbm8gZmF0LXNhdCBzZXJpZXMgZGV0ZWN0ZWQgYmVmb3JlCl9UMV9SWCA9IHJlLmNvbXBpbGUoJ1xcYnQxXFxifFxcYnQxd1xcYicpCl9UMl9SWCA9IHJlLmNvbXBpbGUoJ1xcYnQyXFxifFxcYnQyd1xcYicpCl9QRF9SWCA9IHJlLmNvbXBpbGUoJ1xcYnBkXFxifFxcYnBkd1xcYnxwcm90b258XFxiZHBcXGJ8ZGVucycpCgojIFJ1bnRpbWUgaW50ZWdyaXR5OiBubyBwYXJ0aWFsIGVuc2VtYmxlIGlzIGV2ZXIgcHVibGlzaGVkIGFzIHN1Ym1pc3Npb24uY3N2LgppbXBvcnQgb3MsIGpzb24sIGhhc2hsaWIsIHRpbWUsIHRocmVhZGluZywgdGVtcGZpbGUKZnJvbSBwYXRobGliIGltcG9ydCBQYXRoCmltcG9ydCBudW1weSBhcyBucAppbXBvcnQgcGFuZGFzIGFzIHBkCgpfUlNOQV9BVURJVF9MT0NLID0gdGhyZWFkaW5nLlJMb2NrKCkKX1JTTkFfQVVESVQgPSB7J2NvbnRyYWN0JzogJ2J0a2Rfc3BlZWR5X3Y1NThfaW5wdXRfdW5pZm9ybScsICdldmVudHMnOiBbXSwgJ3BoYXNlcyc6IHt9fQpfUlNOQV9PUkRFUl9NRU1PID0ge30KX1JTTkFfSEVBREVSU19NRU1PID0ge30KX1JTTkFfQ0FDSEVfRklMRVMgPSBbXQpfUlNOQV9URVNUX0lEUyA9IE5vbmUKX1JTTkFfTEFCRUxTID0gWydBQ0wnLCdNQ0wnLCdNZWRpYWwgTWVuaXNjdXMnLCdMYXRlcmFsIE1lbmlzY3VzJywnTWVkaWFsIE9BJywnTGF0ZXJhbCBPQScsJ1BGIE9BJywnRWZmdXNpb24nLCdTeW5vdml0aXMnLCJCYWtlcidzIiwnQ29udHVzaW9uJywnRnJhY3R1cmUnXQoKZGVmIHJzbmFfc2hhKHBhdGgpOgogICAgaCA9IGhhc2hsaWIuc2hhMjU2KCkKICAgIHdpdGggUGF0aChwYXRoKS5vcGVuKCdyYicpIGFzIGY6CiAgICAgICAgZm9yIGIgaW4gaXRlcihsYW1iZGE6IGYucmVhZCg4IDw8IDIwKSwgYicnKToKICAgICAgICAgICAgaC51cGRhdGUoYikKICAgIHJldHVybiBoLmhleGRpZ2VzdCgpCgpkZWYgcnNuYV9qc29uKHBhdGgsIHZhbHVlKToKICAgIHBhdGggPSBQYXRoKHBhdGgpOyBwYXRoLnBhcmVudC5ta2RpcihwYXJlbnRzPVRydWUsIGV4aXN0X29rPVRydWUpCiAgICB0bXAgPSBwYXRoLndpdGhfbmFtZSgnLicgKyBwYXRoLm5hbWUgKyAnLnRtcCcpCiAgICB3aXRoIHRtcC5vcGVuKCd3JykgYXMgZjoKICAgICAgICBqc29uLmR1bXAodmFsdWUsIGYsIGluZGVudD0yLCBzb3J0X2tleXM9VHJ1ZSwgYWxsb3dfbmFuPUZhbHNlLCBkZWZhdWx0PXN0cikKICAgICAgICBmLndyaXRlKCdcbicpOyBmLmZsdXNoKCk7IG9zLmZzeW5jKGYuZmlsZW5vKCkpCiAgICBvcy5yZXBsYWNlKHRtcCwgcGF0aCkKCmRlZiByc25hX2V2ZW50KGtpbmQsICoqZGV0YWlscyk6CiAgICB3aXRoIF9SU05BX0FVRElUX0xPQ0s6CiAgICAgICAgX1JTTkFfQVVESVRbJ2V2ZW50cyddLmFwcGVuZCh7J2tpbmQnOmtpbmQsICoqZGV0YWlsc30pCgpkZWYgcnNuYV9maW5pdGUodmFsdWVzLCB0YWcsIHByb2JhYmlsaXR5PUZhbHNlKToKICAgIHggPSBucC5hc2FycmF5KHZhbHVlcykKICAgIGlmIG5vdCBucC5pc3N1YmR0eXBlKHguZHR5cGUsIG5wLm51bWJlcikgb3Igbm90IG5wLmlzZmluaXRlKHgpLmFsbCgpOgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcihmJ3t0YWd9OiBub25maW5pdGUvbm9uLW51bWVyaWMgdmFsdWVzIEJFRk9SRSByYW5raW5nJykKICAgIGlmIHByb2JhYmlsaXR5IGFuZCB4LnNpemUgYW5kICh4Lm1pbigpIDwgMCBvciB4Lm1heCgpID4gMSk6CiAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKGYne3RhZ306IHByb2JhYmlsaXR5IG91dHNpZGUgWzAsMV0nKQogICAgcmV0dXJuIHgKCmRlZiByc25hX3JhbmswMSh2YWx1ZXMpOgogICAgIiIiUGFyZW50IGVuZHBvaW50IHNjYWxlOyB0aWVkIHZhbHVlcyBzaGFyZSB0aGVpciBhdmVyYWdlIHJhbmsuIiIiCiAgICB4ID0gbnAuYXNhcnJheShyc25hX2Zpbml0ZSh2YWx1ZXMsICdyYW5rMDEnKSwgZHR5cGU9bnAuZmxvYXQ2NCkKICAgIGlmIHgubmRpbSAhPSAyIG9yIG5vdCBsZW4oeCk6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigncmFuazAxIHJlcXVpcmVzIGEgbm9uZW1wdHkgW3N0dWR5LGZpbmRpbmddIG1hdHJpeCcpCiAgICBpZiBsZW4oeCkgPT0gMToKICAgICAgICByZXR1cm4gbnAuZnVsbF9saWtlKHgsIC41KQogICAgcmV0dXJuIChwZC5EYXRhRnJhbWUoeCkucmFuayhtZXRob2Q9J2F2ZXJhZ2UnKS50b19udW1weSgpIC0gMSkgLyAobGVuKHgpLTEpCgpkZWYgcnNuYV9yYW5rcGN0KHZhbHVlcyk6CiAgICB4ID0gbnAuYXNhcnJheShyc25hX2Zpbml0ZSh2YWx1ZXMsICdyYW5rcGN0JyksIGR0eXBlPW5wLmZsb2F0NjQpCiAgICBpZiB4Lm5kaW0gIT0gMiBvciBub3QgbGVuKHgpOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoJ3JhbmtwY3QgcmVxdWlyZXMgYSBub25lbXB0eSBtYXRyaXgnKQogICAgcmV0dXJuIHBkLkRhdGFGcmFtZSh4KS5yYW5rKG1ldGhvZD0nYXZlcmFnZScsIHBjdD1UcnVlKS50b19udW1weShucC5mbG9hdDY0KQoKZGVmIHJzbmFfZnJhbWUoZnJhbWUsIGlkcywgbGFiZWxzLCB0YWcpOgogICAgaWRzID0gW3N0cih1KSBmb3IgdSBpbiBpZHNdOyBsYWJlbHMgPSBsaXN0KGxhYmVscykKICAgIGlmIGxlbihpZHMpICE9IGxlbihzZXQoaWRzKSk6CiAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKGYne3RhZ306IGR1cGxpY2F0ZSBleHBlY3RlZCBVSUQnKQogICAgaWYgZnJhbWUuY29sdW1ucy50b2xpc3QoKSAhPSBbJ1N0dWR5SW5zdGFuY2VVSUQnLCAqbGFiZWxzXToKICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoZid7dGFnfTogbGFiZWwgb3JkZXIvc2NoZW1hIG1pc21hdGNoJykKICAgIGZyYW1lID0gZnJhbWUuY29weSgpOyBmcmFtZVsnU3R1ZHlJbnN0YW5jZVVJRCddID0gZnJhbWVbJ1N0dWR5SW5zdGFuY2VVSUQnXS5hc3R5cGUoc3RyKQogICAgaWYgZnJhbWUuU3R1ZHlJbnN0YW5jZVVJRC5kdXBsaWNhdGVkKCkuYW55KCkgb3Igc2V0KGZyYW1lLlN0dWR5SW5zdGFuY2VVSUQpICE9IHNldChpZHMpOgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcihmJ3t0YWd9OiBtaXNzaW5nL2V4dHJhL2R1cGxpY2F0ZSBVSUQnKQogICAgZnJhbWUgPSBmcmFtZS5zZXRfaW5kZXgoJ1N0dWR5SW5zdGFuY2VVSUQnKS5sb2NbaWRzXS5yZXNldF9pbmRleCgpCiAgICByc25hX2Zpbml0ZShmcmFtZVtsYWJlbHNdLnRvX251bXB5KCksIHRhZywgcHJvYmFiaWxpdHk9VHJ1ZSkKICAgIHJldHVybiBmcmFtZQoKZGVmIHJzbmFfc2F2ZV9wcmVkaWN0aW9ucyh0YWcsIGlkcywgdmFsdWVzLCBsYWJlbHM9Tm9uZSk6CiAgICB4ID0gcnNuYV9maW5pdGUodmFsdWVzLCB0YWcpCiAgICB0YXJnZXQgPSBQYXRoKCcva2FnZ2xlL3dvcmtpbmcvZGlhZ25vc3RpY3MnKTsgdGFyZ2V0Lm1rZGlyKHBhcmVudHM9VHJ1ZSwgZXhpc3Rfb2s9VHJ1ZSkKICAgIHBhdGggPSB0YXJnZXQvKHRhZysnLm5weicpCiAgICBucC5zYXZlel9jb21wcmVzc2VkKHBhdGgsIHN0dWR5X3VpZHM9bnAuYXNhcnJheShpZHMsZHR5cGU9c3RyKSwgbGFiZWxzPW5wLmFzYXJyYXkobGFiZWxzIG9yIF9SU05BX0xBQkVMUyxkdHlwZT1zdHIpLCB2YWx1ZXM9eCkKICAgIHdpdGggX1JTTkFfQVVESVRfTE9DSzoKICAgICAgICBfUlNOQV9BVURJVFsncGhhc2VzJ11bdGFnXT17J3BhdGgnOnN0cihwYXRoKSwnc2hhMjU2Jzpyc25hX3NoYShwYXRoKSwnc2hhcGUnOmxpc3QoeC5zaGFwZSl9CgpkZWYgcnNuYV9zdHJpY3RfbG9hZChtb2RlbCwgc3RhdGUsIHRhZyk6CiAgICAiIiJBIHJhbmRvbSBmcm96ZW4gZW5jb2RlciBpcyBuZXZlciBhbiBhY2NlcHRhYmxlIG1pc3Npbmctc3RhdGUgZmFsbGJhY2suIiIiCiAgICBleHBlY3RlZCA9IG1vZGVsLnN0YXRlX2RpY3QoKQogICAgbWlzc2luZyA9IHNvcnRlZChzZXQoZXhwZWN0ZWQpLXNldChzdGF0ZSkpOyBleHRyYSA9IHNvcnRlZChzZXQoc3RhdGUpLXNldChleHBlY3RlZCkpCiAgICBiYWRfc2hhcGUgPSBbayBmb3IgayBpbiBzZXQoZXhwZWN0ZWQpJnNldChzdGF0ZSkgaWYgdHVwbGUoZXhwZWN0ZWRba10uc2hhcGUpIT10dXBsZShzdGF0ZVtrXS5zaGFwZSldCiAgICBpZiBtaXNzaW5nIG9yIGV4dHJhIG9yIGJhZF9zaGFwZToKICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoZid7dGFnfTogaW5jb21wbGV0ZS9taXNtYXRjaGVkIGNoZWNrcG9pbnQ7IG1pc3Npbmc9e21pc3NpbmdbOjIwXX0sIGV4dHJhPXtleHRyYVs6MjBdfSwgc2hhcGVzPXtiYWRfc2hhcGVbOjIwXX0uIEEgcGFydGlhbCBlbmNvZGVyIG5lZWRzIGl0cyBleGFjdCBwaW5uZWQgc291cmNlIHdlaWdodHM7IHJhbmRvbSBpbml0aWFsaXphdGlvbiBpcyBmb3JiaWRkZW4uJykKICAgIGZvciBrLHQgaW4gc3RhdGUuaXRlbXMoKToKICAgICAgICBpZiB0LmlzX2Zsb2F0aW5nX3BvaW50KCkgYW5kIG5vdCBib29sKHQuaXNmaW5pdGUoKS5hbGwoKSk6CiAgICAgICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcihmJ3t0YWd9OiBub25maW5pdGUgY2hlY2twb2ludCB0ZW5zb3Ige2t9JykKICAgIHJldHVybiBtb2RlbC5sb2FkX3N0YXRlX2RpY3Qoc3RhdGUsIHN0cmljdD1UcnVlKQoKZGVmIHJzbmFfZGVhZGxpbmUodGFnKToKICAgIGlmIHRpbWUudGltZSgpIC0gVDAgPiBUSU1FX0JVREdFVDoKICAgICAgICByYWlzZSBUaW1lb3V0RXJyb3IoZid7dGFnfTogZnVsbC1ydW4gYnVkZ2V0IGV4Y2VlZGVkOyBubyBpbmNvbXBsZXRlIG91dHB1dCBtYXkgYmUgc3VibWl0dGVkJykKCmRlZiByc25hX2FycmF5KHNoYXBlLCB0YWcpOgogICAgIiIiTGFyZ2UgdGVtcG9yYXJ5IHBpeGVscyBiZWxvbmcgaW4gc2NyYXRjaCBzcGFjZSwgbm90IHNhdmVkIG5vdGVib29rIG91dHB1dHMuIiIiCiAgICBzaXplID0gaW50KG5wLnByb2Qoc2hhcGUpKQogICAgaWYgc2l6ZSA8PSBmbG9hdChvcy5lbnZpcm9uLmdldCgnUlNOQV9DQUNIRV9SQU1fR0lCJywgJzEnKSkgKiAoMSA8PCAzMCk6CiAgICAgICAgcmV0dXJuIG5wLnplcm9zKHNoYXBlLCBucC51aW50OCkKICAgIGRpcmVjdG9yeSA9IFBhdGgob3MuZW52aXJvbi5nZXQoJ1JTTkFfUElYRUxfU0NSQVRDSCcsICcva2FnZ2xlL3RlbXAvcnNuYV9waXhlbHNfdjU1OScpKQogICAgdHJ5OgogICAgICAgIGRpcmVjdG9yeS5ta2RpcihwYXJlbnRzPVRydWUsIGV4aXN0X29rPVRydWUpCiAgICBleGNlcHQgT1NFcnJvciBhcyBzY3JhdGNoX2Vycm9yOiAgIyAva2FnZ2xlL3RlbXAgaXMgbm90IGd1YXJhbnRlZWQ7IC90bXAgaXMgbm9uLXBlcnNpc3RlZCBzY3JhdGNoIHdpdGggfjEgVGlCIGZyZWUKICAgICAgICByc25hX2V2ZW50KCdzY3JhdGNoX2ZhbGxiYWNrJywgdGFnPXRhZywgcmVxdWVzdGVkPXN0cihkaXJlY3RvcnkpLCBlcnJvcj1mJ3t0eXBlKHNjcmF0Y2hfZXJyb3IpLl9fbmFtZV9ffToge3NjcmF0Y2hfZXJyb3J9JykKICAgICAgICBkaXJlY3RvcnkgPSBQYXRoKCcvdG1wL3JzbmFfcGl4ZWxzX3Y1NTknKQogICAgICAgIGRpcmVjdG9yeS5ta2RpcihwYXJlbnRzPVRydWUsIGV4aXN0X29rPVRydWUpCiAgICBpbXBvcnQgc2h1dGlsCiAgICBmcmVlID0gc2h1dGlsLmRpc2tfdXNhZ2UoZGlyZWN0b3J5KS5mcmVlCiAgICByc25hX2V2ZW50KCdzY3JhdGNoX2FsbG9jYXRpb24nLCB0YWc9dGFnLCBieXRlcz1zaXplLCBkaXJlY3Rvcnk9c3RyKGRpcmVjdG9yeSksIGZyZWVfYnl0ZXM9aW50KGZyZWUpKQogICAgaWYgZnJlZSA8IHNpemUgKyAoMSA8PCAzMCkgYW5kIHN0cihkaXJlY3RvcnkpICE9ICcvdG1wL3JzbmFfcGl4ZWxzX3Y1NTknOgogICAgICAgIHJzbmFfZXZlbnQoJ3NjcmF0Y2hfZmFsbGJhY2snLCB0YWc9dGFnLCByZXF1ZXN0ZWQ9c3RyKGRpcmVjdG9yeSksIGVycm9yPWYnaW5zdWZmaWNpZW50IGZyZWUgc3BhY2UgKHtmcmVlfSBieXRlcyknKQogICAgICAgIGRpcmVjdG9yeSA9IFBhdGgoJy90bXAvcnNuYV9waXhlbHNfdjU1OScpOyBkaXJlY3RvcnkubWtkaXIocGFyZW50cz1UcnVlLCBleGlzdF9vaz1UcnVlKTsgZnJlZSA9IHNodXRpbC5kaXNrX3VzYWdlKGRpcmVjdG9yeSkuZnJlZQogICAgaWYgZnJlZSA8IHNpemUgKyAoMSA8PCAzMCk6CiAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKGYne3RhZ306IG5vdCBlbm91Z2ggc2NyYXRjaCBkaXNrIGZvciBjb21wbGV0ZSBjYWNoZSAoe3NpemV9IGJ5dGVzOyB7ZnJlZX0gZnJlZSknKQogICAgZmQsIHBhdGggPSB0ZW1wZmlsZS5ta3N0ZW1wKHByZWZpeD0ncGl4ZWxzXycsIHN1ZmZpeD0nLm5weScsIGRpcj1kaXJlY3RvcnkpCiAgICBvcy5jbG9zZShmZCkKICAgIHRyeToKICAgICAgICBhID0gbnAubGliLmZvcm1hdC5vcGVuX21lbW1hcChwYXRoLCBtb2RlPSd3KycsIGR0eXBlPW5wLnVpbnQ4LCBzaGFwZT10dXBsZShzaGFwZSkpCiAgICAgICAgYVs6XSA9IDAKICAgIGV4Y2VwdCBCYXNlRXhjZXB0aW9uOgogICAgICAgIFBhdGgocGF0aCkudW5saW5rKG1pc3Npbmdfb2s9VHJ1ZSkKICAgICAgICByYWlzZQogICAgX1JTTkFfQ0FDSEVfRklMRVMuYXBwZW5kKHBhdGgpCiAgICByZXR1cm4gYQoKZGVmIHJzbmFfcmVsZWFzZV9waXhlbHMoYXJyYXkpOgogICAgaWYgaXNpbnN0YW5jZShhcnJheSxucC5tZW1tYXApOgogICAgICAgIHBhdGg9c3RyKGFycmF5LmZpbGVuYW1lKQogICAgICAgIGFycmF5LmZsdXNoKCkKICAgICAgICAjIFBPU0lYIHVubGluayByZWxlYXNlcyB0aGUgZmlsZSBhZnRlciB0aGUgbGFzdCB2aWV3IGlzIGNsb3NlZC4KICAgICAgICBQYXRoKHBhdGgpLnVubGluayhtaXNzaW5nX29rPVRydWUpCiAgICAgICAgaWYgcGF0aCBpbiBfUlNOQV9DQUNIRV9GSUxFUzogX1JTTkFfQ0FDSEVfRklMRVMucmVtb3ZlKHBhdGgpCgpkZWYgcnNuYV9pbml0aWFsaXplKGlkcyk6CiAgICBnbG9iYWwgX1JTTkFfVEVTVF9JRFMKICAgIF9SU05BX1RFU1RfSURTPVtzdHIodSkgZm9yIHUgaW4gaWRzXQogICAgaWYgbm90IF9SU05BX1RFU1RfSURTIG9yIGxlbihzZXQoX1JTTkFfVEVTVF9JRFMpKSE9bGVuKF9SU05BX1RFU1RfSURTKToKICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoJ2VtcHR5IG9yIGR1cGxpY2F0ZSBjb21wZXRpdGlvbiBzdHVkeSBJRHMnKQogICAgd29yaz1QYXRoKCcva2FnZ2xlL3dvcmtpbmcnKTsgd29yay5ta2RpcihwYXJlbnRzPVRydWUsZXhpc3Rfb2s9VHJ1ZSkKICAgIG9zLmNoZGlyKHdvcmspCiAgICBmb3IgZiBpbiBbJ3N1Ym1pc3Npb24uY3N2JywnX3BpcGVsaW5lX3N0YWdlLmNzdicsJ2J0a2RfdjU1OV9jb21wbGV0ZS5qc29uJywnYnRrZF92NTU5X2NvbXBsZXRlLmpzb24nXToKICAgICAgICAod29yay9mKS51bmxpbmsobWlzc2luZ19vaz1UcnVlKQogICAgKHdvcmsvJ2RpYWdub3N0aWNzJykubWtkaXIoZXhpc3Rfb2s9VHJ1ZSkKICAgIF9SU05BX0FVRElUWydjb2hvcnQnXT1sZW4oX1JTTkFfVEVTVF9JRFMpCiAgICBfUlNOQV9BVURJVFsnZW52aXJvbm1lbnQnXT17J3B5dGhvbic6X19pbXBvcnRfXygnc3lzJykudmVyc2lvbiwndG9yY2gnOnN0cih0b3JjaC5fX3ZlcnNpb25fXyksJ2N1ZGEnOnRvcmNoLnZlcnNpb24uY3VkYSwnZ3B1X25hbWVzJzpbdG9yY2guY3VkYS5nZXRfZGV2aWNlX25hbWUoaSkgZm9yIGkgaW4gcmFuZ2UodG9yY2guY3VkYS5kZXZpY2VfY291bnQoKSldfQogICAgcnNuYV9qc29uKHdvcmsvJ2RpYWdub3N0aWNzL3J1bnRpbWVfc3RhcnRlZC5qc29uJyxfUlNOQV9BVURJVCkKCmRlZiByc25hX2Fzc2V0X3ByZWZsaWdodCgpOgogICAgIiIiTG9jYXRlIHJlcXVpcmVkIGFkZGl0aW9ucyBiZWZvcmUgc3BlbmRpbmcgdGltZSBvbiB0aGUgcHVibGljIHBhcmVudC4iIiIKICAgIG5hbWVzPVsncmFwdG9yX2Z0X2NvYXRuZXRfdjVfZnVsbF9zd2EucHQnLCdyYXB0b3JfZnRfY29hdG5ldF92MTBfZnVsbC5wdCcsJ3JhcHRvcl9mdF9jb2F0bmV0X3Y4X2Z1bGxfc3dhLnB0J10KICAgIHJlY2VpcHQ9e25hbWU6c3RyKF9hc3NldF9maW5kX2Fzc2V0KG5hbWUpKSBmb3IgbmFtZSBpbiBuYW1lc30KICAgIGZvciBuYW1lLGRpZ2VzdCBpbiBbCiAgICAgICAgKCdjb2F0X3Jlc2dhdGVkX2VwMTBfdG9wM19tYW5pZmVzdC5qc29uJywnOTg1MTFhOGZkZWI5ZGEwZTZlNzBjNzhkMDEzZmY2MzZlMTQ3NmYzMWM4MGI1ZGMxMzRkMjk0YjE4YzNmMjg0ZScpLAogICAgICAgICgnY29hdG5ldF9wYWlyZmlsbV9tYW5pZmVzdC5qc29uJywnN2FkYTA2MDViY2E2YjA1MzA1NjljNjQ1NGU5ODhhY2U0Nzk2MDZhMzMyOGVkNTkxZDA5MGU1NzY0ZmVhNjYxZCcpLAogICAgICAgICgnb3BlbmN2X3B5dGhvbl9oZWFkbGVzcy00LjEyLjAuODgtKi53aGwnLCcyMzZjOGRmNTRhOTBmNGQwMjA3NmU2ZjljMWNjNzYzZDc5NDU0MmU4ODZjNTc2YTZmZWU0NmVjOGZmNzVhN2E5JyldOgogICAgICAgIHJlY2VpcHRbbmFtZV09eydwYXRoJzpzdHIoX2Fzc2V0X2ZpbmRfYXNzZXQobmFtZSxkaWdlc3QpKSwnc2hhMjU2JzpkaWdlc3R9CiAgICByb290cz1bUGF0aChwKSBmb3IgcCBpbiBbJy9rYWdnbGUvaW5wdXQvZGF0YXNldHMvbWF0dGlhYW5nZWxpL2tuZWUtbXJpLWZvbGQtd2VpZ2h0cycsJy9rYWdnbGUvaW5wdXQva25lZS1tcmktZm9sZC13ZWlnaHRzJ10gaWYgKFBhdGgocCkvJ21fZjAucHQnKS5pc19maWxlKCldCiAgICByb290cz1saXN0KHtzdHIocC5yZXNvbHZlKCkpOnAgZm9yIHAgaW4gcm9vdHN9LnZhbHVlcygpKQogICAgaWYgbGVuKHJvb3RzKSE9MTpyYWlzZSBSdW50aW1lRXJyb3IoZidBNTogZXhwZWN0ZWQgb25lIGNvbXBsZXRlIGF0dGFjaGVkIHJvb3QsIGZvdW5kIHtyb290c30nKQogICAgbWlzc2luZz1bc3RyKHJvb3RzWzBdL2YnbV9me2l9LnB0JykgZm9yIGkgaW4gcmFuZ2UoNSkgaWYgbm90IChyb290c1swXS9mJ21fZntpfS5wdCcpLmlzX2ZpbGUoKV0KICAgIGlmIG1pc3Npbmc6cmFpc2UgRmlsZU5vdEZvdW5kRXJyb3IoZidBNSBtaXNzaW5nIGZvbGRzOiB7bWlzc2luZ30nKQogICAgcmVjZWlwdFsnQTVfcm9vdCddPXN0cihyb290c1swXSk7cmVjZWlwdFsndGVuc29yX2xvYWRpbmdfYW5kX2ZpbmdlcnByaW50cyddPSd2YWxpZGF0ZWQgYnkgbWFuZGF0b3J5IGNvbXBvbmVudCBsb2FkZXJzIGJlZm9yZSBwcmVkaWN0aW9ucycKICAgIGQ0X21hbmlmZXN0X3BhdGggPSBfYXNzZXRfZmluZF9hc3NldCgnY29hdG5ldF9wYWlyZmlsbV9tYW5pZmVzdC5qc29uJywKICAgICAgICAnN2FkYTA2MDViY2E2YjA1MzA1NjljNjQ1NGU5ODhhY2U0Nzk2MDZhMzMyOGVkNTkxZDA5MGU1NzY0ZmVhNjYxZCcpCiAgICB0aW1tX3doZWVsID0gZDRfbWFuaWZlc3RfcGF0aC5wYXJlbnQgLyAndGltbS0xLjAuMjItcHkzLW5vbmUtYW55LndobCcKICAgIGlmIG5vdCB0aW1tX3doZWVsLmlzX2ZpbGUoKSBvciByc25hX3NoYSh0aW1tX3doZWVsKSAhPSAnODg4OTgxNzUzZTY1Y2JhYWNmYzA3NDk0MzcwMTM4YjE3MDBhMjdiMWYwYWY1ODdmNGY5YjQ3YmMwMjQxNjFkMCc6CiAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKCdENCBwaW5uZWQgdGltbSB3aGVlbCBtaXNzaW5nIG9yIGNoYW5nZWQnKQogICAgcmVjZWlwdFsnRDRfdGltbV93aGVlbF9zaGEyNTYnXSA9IHJzbmFfc2hhKHRpbW1fd2hlZWwpCiAgICByc25hX2pzb24oJy9rYWdnbGUvd29ya2luZy9kaWFnbm9zdGljcy9hc3NldF9wcmVmbGlnaHQuanNvbicscmVjZWlwdCkKCgpkZWYgcnNuYV9waGFzZShzdGFnZSwgc3RhdHVzPSdTVEFSVCcsICoqZXh0cmEpOgogICAgIiIiTG9jYWwgcGhhc2UgZXZpZGVuY2U7IHRoaXMgZG9lcyBub3QgZXhwb3NlIEthZ2dsZSdzIGhpZGRlbiByZXJ1biBsb2dzLiIiIgogICAgaW1wb3J0IGpzb24sIG9zLCByZXNvdXJjZSwgdGltZQogICAgZnJvbSBwYXRobGliIGltcG9ydCBQYXRoCiAgICB3b3JrID0gUGF0aCgnL2thZ2dsZS93b3JraW5nL2RpYWdub3N0aWNzJykKICAgIHdvcmsubWtkaXIocGFyZW50cz1UcnVlLCBleGlzdF9vaz1UcnVlKQogICAgY3VycmVudCA9IHsnc3RhZ2UnOiBzdGFnZSwgJ3N0YXR1cyc6IHN0YXR1cywKICAgICAgICAgICAgICAgJ2VsYXBzZWRfc2Vjb25kcyc6IHRpbWUudGltZSgpLVQwLAogICAgICAgICAgICAgICAnbWF4X3Jzc19raWInOiBpbnQocmVzb3VyY2UuZ2V0cnVzYWdlKHJlc291cmNlLlJVU0FHRV9TRUxGKS5ydV9tYXhyc3MpLCAqKmV4dHJhfQogICAgZm9yIHBhdGggaW4gWycvc3lzL2ZzL2Nncm91cC9tZW1vcnkuY3VycmVudCcsICcvc3lzL2ZzL2Nncm91cC9tZW1vcnkubWF4J106CiAgICAgICAgdHJ5OiBjdXJyZW50W3BhdGgucnNwbGl0KCcvJywgMSlbLTFdXSA9IFBhdGgocGF0aCkucmVhZF90ZXh0KCkuc3RyaXAoKQogICAgICAgIGV4Y2VwdCBPU0Vycm9yOiBwYXNzCiAgICBpZiB0b3JjaC5jdWRhLmlzX2luaXRpYWxpemVkKCk6CiAgICAgICAgY3VycmVudFsnZ3B1cyddID0gW3snaWQnOiBpLCAnYWxsb2NhdGVkX2J5dGVzJzogdG9yY2guY3VkYS5tZW1vcnlfYWxsb2NhdGVkKGkpLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgJ3Jlc2VydmVkX2J5dGVzJzogdG9yY2guY3VkYS5tZW1vcnlfcmVzZXJ2ZWQoaSl9CiAgICAgICAgICAgICAgICAgICAgICAgICAgIGZvciBpIGluIHJhbmdlKHRvcmNoLmN1ZGEuZGV2aWNlX2NvdW50KCkpXQogICAgcHJpbnQoJ1twaGFzZV0gJytqc29uLmR1bXBzKGN1cnJlbnQsIHNvcnRfa2V5cz1UcnVlKSwgZmx1c2g9VHJ1ZSkKICAgIHJzbmFfanNvbih3b3JrLydjdXJyZW50X3BoYXNlLmpzb24nLCBjdXJyZW50KQogICAgd2l0aCAod29yay8ncGhhc2VfZXZlbnRzLmpzb25sJykub3BlbignYScpIGFzIGY6CiAgICAgICAgZi53cml0ZShqc29uLmR1bXBzKGN1cnJlbnQsIHNvcnRfa2V5cz1UcnVlKSsnXG4nKQogICAgcmV0dXJuIGN1cnJlbnQKCiMgUmVjb3JkIHVuY2F1Z2h0IGNlbGwgZXJyb3JzIGxvY2FsbHk7IEthZ2dsZSBtYXkgc3RpbGwgd2l0aGhvbGQgaGlkZGVuLXJ1biBsb2dzLgpkZWYgX3JzbmFfY2FwdHVyZV9jZWxsX2Vycm9yKHJlc3VsdCk6CiAgICBlcnJvciA9IGdldGF0dHIocmVzdWx0LCAnZXJyb3JfaW5fZXhlYycsIE5vbmUpIG9yIGdldGF0dHIocmVzdWx0LCAnZXJyb3JfYmVmb3JlX2V4ZWMnLCBOb25lKQogICAgaWYgZXJyb3IgaXMgTm9uZToKICAgICAgICByZXR1cm4KICAgIHRyeToKICAgICAgICBwYXRoID0gUGF0aCgnL2thZ2dsZS93b3JraW5nL2RpYWdub3N0aWNzL2N1cnJlbnRfcGhhc2UuanNvbicpCiAgICAgICAgcHJldmlvdXMgPSBqc29uLmxvYWRzKHBhdGgucmVhZF90ZXh0KCkpIGlmIHBhdGguaXNfZmlsZSgpIGVsc2Uge30KICAgICAgICByc25hX3BoYXNlKHByZXZpb3VzLmdldCgnc3RhZ2UnLCAndW5rbm93bicpLCAnRkFJTEVEJywgZXJyb3JfdHlwZT10eXBlKGVycm9yKS5fX25hbWVfXywgZXJyb3I9c3RyKGVycm9yKSkKICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgbG9nZ2luZ19lcnJvcjoKICAgICAgICBwcmludCgnW2RpYWdub3N0aWMtd3JpdGUtZmFpbGVkXScsIHR5cGUobG9nZ2luZ19lcnJvcikuX19uYW1lX18sIGZsdXNoPVRydWUpCnRyeToKICAgIF9yc25hX2lwID0gZ2V0X2lweXRob24oKQogICAgaWYgX3JzbmFfaXAgaXMgbm90IE5vbmU6CiAgICAgICAgX3JzbmFfaXAuZXZlbnRzLnJlZ2lzdGVyKCdwb3N0X3J1bl9jZWxsJywgX3JzbmFfY2FwdHVyZV9jZWxsX2Vycm9yKQpleGNlcHQgTmFtZUVycm9yOgogICAgcGFzcwo=", "4": "ZGVmIGxvZyhtc2cpOgogICAgcHJpbnQoZidbe3RpbWUudGltZSgpIC0gVDA6Ny4xZn1zXSB7bXNnfScsIGZsdXNoPVRydWUpCgpkZWYgX29uZV9kaXJlY3QodGFnLCBjYW5kaWRhdGVzLCByZXF1aXJlZCk6CiAgICBtYXRjaGVzID0gW10KICAgIGZvciBjYW5kaWRhdGUgaW4gbWFwKFBhdGgsIGNhbmRpZGF0ZXMpOgogICAgICAgIGlmIGNhbmRpZGF0ZS5pc19kaXIoKSBhbmQgYWxsKChjYW5kaWRhdGUgLyBpdGVtKS5leGlzdHMoKSBmb3IgaXRlbSBpbiByZXF1aXJlZCk6CiAgICAgICAgICAgIG1hdGNoZXMuYXBwZW5kKGNhbmRpZGF0ZSkKICAgIHVuaXF1ZSA9IFtdCiAgICBmb3IgbWF0Y2ggaW4gbWF0Y2hlczoKICAgICAgICBpZiBzdHIobWF0Y2gucmVzb2x2ZSgpKSBub3QgaW4ge3N0cihpdGVtLnJlc29sdmUoKSkgZm9yIGl0ZW0gaW4gdW5pcXVlfToKICAgICAgICAgICAgdW5pcXVlLmFwcGVuZChtYXRjaCkKICAgIGlmIGxlbih1bmlxdWUpICE9IDE6CiAgICAgICAgcmFpc2UgRmlsZU5vdEZvdW5kRXJyb3IoCiAgICAgICAgICAgIGYne3RhZ306IGV4cGVjdGVkIGV4YWN0bHkgb25lIGRpcmVjdCBtb3VudGVkIGFydGlmYWN0LCBmb3VuZCAnCiAgICAgICAgICAgIGYne1tzdHIoaXRlbSkgZm9yIGl0ZW0gaW4gdW5pcXVlXX07IGNoZWNrZWQge1tzdHIoUGF0aChpdGVtKSkgZm9yIGl0ZW0gaW4gY2FuZGlkYXRlc119JwogICAgICAgICkKICAgIHJldHVybiB1bmlxdWVbMF0KCmRlZiBmaW5kX3Jvb3QoKToKICAgIHJldHVybiBfb25lX2RpcmVjdCgnY29tcGV0aXRpb24nLCBbCiAgICAgICAgb3MuZW52aXJvbi5nZXQoJ1JTTkFfQ09NUF9ST09UJywgJy9rYWdnbGUvaW5wdXQvY29tcGV0aXRpb25zL3JzbmEta25lZS1hYm5vcm1hbGl0eS1kZXRlY3Rpb24nKSwKICAgICAgICAnL2thZ2dsZS9pbnB1dC9yc25hLWtuZWUtYWJub3JtYWxpdHktZGV0ZWN0aW9uJywKICAgIF0sIFsndGVzdC5jc3YnLCAndGVzdF9zZXJpZXMuY3N2JywgJ3Rlc3Rfc2VyaWVzJ10pCgpkZWYgZmluZF9kaW5vdjIodmFyaWFudD0nc21hbGwnKToKICAgIGlmIHZhcmlhbnQgIT0gJ3NtYWxsJzoKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYndW5waW5uZWQgRElOT3YyIHZhcmlhbnQ6IHt2YXJpYW50fScpCiAgICByZXR1cm4gX29uZV9kaXJlY3QoJ0RJTk92Mi1TIG1vZGVsJywgWwogICAgICAgICcva2FnZ2xlL2lucHV0L21vZGVscy9tZXRhcmVzZWFyY2gvZGlub3YyL1B5VG9yY2gvc21hbGwvMScsCiAgICAgICAgJy9rYWdnbGUvaW5wdXQvbW9kZWxzL21ldGFyZXNlYXJjaC9kaW5vdjIvcHl0b3JjaC9zbWFsbC8xJywKICAgICAgICAnL2thZ2dsZS9pbnB1dC9kaW5vdjIvUHlUb3JjaC9zbWFsbC8xJywKICAgICAgICAnL2thZ2dsZS9pbnB1dC9kaW5vdjIvcHl0b3JjaC9zbWFsbC8xJywKICAgIF0sIFsnY29uZmlnLmpzb24nLCAncHl0b3JjaF9tb2RlbC5iaW4nXSkKUk9PVCA9IGZpbmRfcm9vdCgpCkRJTk9WMl9TT1VSQ0UgPSBmaW5kX2Rpbm92Mignc21hbGwnKQpsb2coZidpbnB1dCByb290OiB7Uk9PVH0nKQpfU1RBUlRVUF9QQVRIX1MgPSBfc3RhcnR1cF90aW1lLnBlcmZfY291bnRlcigpIC0gX1NUQVJUVVBfVDAgLSBfU1RBUlRVUF9HUFVfSU1QT1JUX1MKcHJpbnQoZidbc3RhcnR1cF0gY29tcGV0aXRpb24tcm9vdCBwcmVmbGlnaHQ6IHtfU1RBUlRVUF9QQVRIX1M6LjJmfXMnLCBmbHVzaD1UcnVlKQpJTUcgPSBDQUNIRV9JTUcKCmRlZiBhdmFpbGFibGVfZ2IoKToKICAgIHRyeToKICAgICAgICB3aXRoIG9wZW4oJy9wcm9jL21lbWluZm8nKSBhcyBmaDoKICAgICAgICAgICAgaW5mbyA9IHtrLnN0cmlwKCk6IHYgZm9yIGssIHYgaW4gKGwuc3BsaXQoJzonLCAxKSBmb3IgbCBpbiBmaCBpZiAnOicgaW4gbCl9CiAgICAgICAgcmV0dXJuIGludChpbmZvWydNZW1BdmFpbGFibGUnXS5zcGxpdCgpWzBdKSAvIDEwMjQgKiogMgogICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICByZXR1cm4gQ0FDSEVfQlVER0VUX0dCIC8gQ0FDSEVfRlJBQ1RJT04KCmRlZiBwbGFuX2NhY2hlKG5fc3R1ZHksIG5fdGVzdD0wKToKICAgIGF2YWlsID0gYXZhaWxhYmxlX2diKCkKICAgIGJ1ZGdldCA9IG1pbihhdmFpbCAqIENBQ0hFX0ZSQUNUSU9OLCBDQUNIRV9CVURHRVRfTUFYX0dCKQogICAgbl90b3RhbCA9IG5fc3R1ZHkgKyBtYXgobl90ZXN0LCBpbnQoVEVTVF9TSEFSRSAqIG5fc3R1ZHkpKQogICAgcGVyX3NsaWNlID0gbl90b3RhbCAqIE5fU0xPVCAqIElNRyAqIElNRwogICAgYWZmb3JkID0gaW50KGJ1ZGdldCAqIDEwMjQgKiogMyAvLyBtYXgocGVyX3NsaWNlLCAxKSkKICAgIGdyb3VwcyA9IG1heCgxLCBtaW4oTl9HUk9VUF9NQVgsIGFmZm9yZCAvLyBHUk9VUCkpCiAgICBsb2coZidtZW1vcnk6IHthdmFpbDouMWZ9IEdCIGF2YWlsYWJsZSwge2J1ZGdldDouMWZ9IEdCIHRvIHRoZSBjYWNoZTsgc2l6aW5nIGZvciB7bl9zdHVkeX0gdHJhaW4gKyB7bl90b3RhbCAtIG5fc3R1ZHl9IHRlc3Qgc3R1ZGllcyAtPiB7Z3JvdXBzfSBncm91cChzKSBvZiB7R1JPVVB9ID0ge2dyb3VwcyAqIEdST1VQfSBzbGljZXMgcGVyIHNsb3QnICsgKGYnICh3YW50ZWQge05fR1JPVVBfTUFYfSknIGlmIGdyb3VwcyA8IE5fR1JPVVBfTUFYIGVsc2UgJycpKQogICAgcmV0dXJuIGdyb3VwcwpOX0dST1VQID0gcGxhbl9jYWNoZShsZW4ocGQucmVhZF9jc3YoUk9PVCAvICd0cmFpbi5jc3YnKSksIGxlbihwZC5yZWFkX2NzdihST09UIC8gJ3Rlc3QuY3N2JykpKQpDQUNIRV9TTElDRVMgPSBHUk9VUCAqIE5fR1JPVVAKbG9nKGYnY2FjaGUgbGF5b3V0OiB7Tl9HUk9VUH0gZ3JvdXBzIHgge0dST1VQfSBzbGljZXMgPSB7Q0FDSEVfU0xJQ0VTfSBwZXIgc2xvdCcpCnJzbmFfaW5pdGlhbGl6ZShwZC5yZWFkX2NzdihST09ULyd0ZXN0LmNzdicsZHR5cGU9eydTdHVkeUluc3RhbmNlVUlEJzpzdHJ9KS5TdHVkeUluc3RhbmNlVUlELnRvbGlzdCgpKQpvcy5lbnZpcm9uWydSU05BX0NPTVBfUk9PVCddPXN0cihST09UKQpyc25hX2Fzc2V0X3ByZWZsaWdodCgpCgojIFJlc29sdmUgdW5zdXBwb3J0ZWQgQTUgbWV0YWRhdGEgYmVmb3JlIGxvbmcgbW9kZWwgaW5mZXJlbmNlLCB3aXRob3V0IGludmVudGluZyBsYWJlbHMuCl9tZXRhID0gcGQucmVhZF9jc3YoUk9PVC8ndGVzdF9zZXJpZXMuY3N2JywgZHR5cGU9eydTdHVkeUluc3RhbmNlVUlEJzpzdHJ9KQpfcmVxdWlyZWQgPSB7J1N0dWR5SW5zdGFuY2VVSUQnLCdBbmF0b21pY2FsX1BsYW5lJywnRmF0X1N1cHByZXNzaW9uJ30KaWYgbm90IF9yZXF1aXJlZC5pc3N1YnNldChfbWV0YS5jb2x1bW5zKToKICAgIHJhaXNlIFJ1bnRpbWVFcnJvcignTWlzc2luZyByZXF1aXJlZCBzZXJpZXMgbWV0YWRhdGEgY29sdW1uczogJytzdHIoc29ydGVkKF9yZXF1aXJlZC1zZXQoX21ldGEuY29sdW1ucykpKSkKX2E1X21hdGNoZXMgPSBfbWV0YS5BbmF0b21pY2FsX1BsYW5lLmlzaW4oWydTYWdpdHRhbCcsJ0Nvcm9uYWwnLCdBeGlhbCddKSAmIF9tZXRhLkZhdF9TdXBwcmVzc2lvbi5pc2luKFswLDFdKQpfYTVfaWRzID0gc2V0KF9tZXRhLmxvY1tfYTVfbWF0Y2hlcywnU3R1ZHlJbnN0YW5jZVVJRCddKQpfYTVfdW5hdmFpbGFibGUgPSBbdSBmb3IgdSBpbiBfUlNOQV9URVNUX0lEUyBpZiB1IG5vdCBpbiBfYTVfaWRzXQpyc25hX2pzb24oJy9rYWdnbGUvd29ya2luZy9kaWFnbm9zdGljcy9hNV9tZXRhZGF0YV9hcHBsaWNhYmlsaXR5Lmpzb24nLCB7CiAgICAnc3R1ZGllcyc6bGVuKF9SU05BX1RFU1RfSURTKSwgJ25vX21hdGNoaW5nX2FjcXVpc2l0aW9uX3VpZHMnOl9hNV91bmF2YWlsYWJsZSwKICAgICdwb2xpY3knOidiYXNlLW9ubHkgY29udHJpYnV0aW9uIGZvciBleHBsaWNpdCBBNSBhYnNlbmNlOyBtb2RlbCBmYWlsdXJlcyByZW1haW4gZmF0YWwnfSkKcnNuYV9waGFzZSgnaW5wdXRfbWV0YWRhdGEnLCAnQ09NUExFVEUnLCBhNV93aXRob3V0X21hdGNoaW5nX2FjcXVpc2l0aW9uPWxlbihfYTVfdW5hdmFpbGFibGUpLAogICAgICAgICAgIG1pc3NpbmdfZmx1aWRfZmxhZ3M9aW50KF9tZXRhLkZsdWlkX1NlbnNpdGl2ZS5pc25hKCkuc3VtKCkpIGlmICdGbHVpZF9TZW5zaXRpdmUnIGluIF9tZXRhIGVsc2UgbGVuKF9tZXRhKSkKZGVsIF9tZXRhLCBfYTVfbWF0Y2hlcywgX2E1X2lkcywgX2E1X3VuYXZhaWxhYmxlCg==", "5": "SERSX1RBR1MgPSBbJ1Nlcmllc0Rlc2NyaXB0aW9uJywgJ1NlcXVlbmNlTmFtZScsICdTY2FuT3B0aW9ucycsICdTY2FubmluZ1NlcXVlbmNlJywgJ1JlcGV0aXRpb25UaW1lJywgJ0VjaG9UaW1lJywgJ0xhdGVyYWxpdHknLCAnUGl4ZWxTcGFjaW5nJywgJ1Jvd3MnLCAnQ29sdW1ucycsICdSZXNjYWxlU2xvcGUnLCAnUmVzY2FsZUludGVyY2VwdCcsICdJbWFnZVBvc2l0aW9uUGF0aWVudCcsICdJbWFnZU9yaWVudGF0aW9uUGF0aWVudCddCgpkZWYgX2hkcl92ZWMocywgbik6CiAgICBpZiBub3QgaXNpbnN0YW5jZShzLCBzdHIpOgogICAgICAgIHJldHVybiBOb25lCiAgICB0cnk6CiAgICAgICAgdiA9IFtmbG9hdCh4KSBmb3IgeCBpbiBzLnNwbGl0KCd8JyldCiAgICBleGNlcHQgVmFsdWVFcnJvcjoKICAgICAgICByZXR1cm4gTm9uZQogICAgcmV0dXJuIG5wLmFycmF5KHYpIGlmIGxlbih2KSA+PSBuIGVsc2UgTm9uZQoKZGVmIHNpZGVfZnJvbV9nZW9tZXRyeShoKToKICAgIGN4ID0ge30KICAgIGZvciByIGluIGguaXRlcnR1cGxlcyhpbmRleD1GYWxzZSk6CiAgICAgICAgaXBwID0gX2hkcl92ZWMoZ2V0YXR0cihyLCAnSW1hZ2VQb3NpdGlvblBhdGllbnQnLCBOb25lKSwgMykKICAgICAgICBpb3AgPSBfaGRyX3ZlYyhnZXRhdHRyKHIsICdJbWFnZU9yaWVudGF0aW9uUGF0aWVudCcsIE5vbmUpLCA2KQogICAgICAgIHBzID0gX2hkcl92ZWMoZ2V0YXR0cihyLCAnUGl4ZWxTcGFjaW5nJywgTm9uZSksIDIpCiAgICAgICAgcm93cywgY29scyA9IChnZXRhdHRyKHIsICdSb3dzJywgTm9uZSksIGdldGF0dHIociwgJ0NvbHVtbnMnLCBOb25lKSkKICAgICAgICBpZiBpcHAgaXMgTm9uZSBvciBpb3AgaXMgTm9uZSBvciBwcyBpcyBOb25lIG9yIChub3Qgcm93cykgb3IgKG5vdCBjb2xzKToKICAgICAgICAgICAgY29udGludWUKICAgICAgICB0cnk6CiAgICAgICAgICAgIGMgPSBpcHBbOjNdICsgaW9wWzozXSAqIHBzWzFdICogZmxvYXQoY29scykgLyAyICsgaW9wWzM6Nl0gKiBwc1swXSAqIGZsb2F0KHJvd3MpIC8gMgogICAgICAgIGV4Y2VwdCAoVHlwZUVycm9yLCBWYWx1ZUVycm9yKToKICAgICAgICAgICAgY29udGludWUKICAgICAgICBjeC5zZXRkZWZhdWx0KHIuU3R1ZHlJbnN0YW5jZVVJRCwgW10pLmFwcGVuZChmbG9hdChjWzBdKSkKICAgIG91dCA9IHt9CiAgICBmb3Igc3QsIHhzIGluIGN4Lml0ZW1zKCk6CiAgICAgICAgbSA9IGZsb2F0KG5wLm1lZGlhbih4cykpCiAgICAgICAgb3V0W3N0XSA9IE5vbmUgaWYgYWJzKG0pIDwgTEFUX01JTl9PRkZTRVRfTU0gZWxzZSAnUicgaWYgbSA8IDAgZWxzZSAnTCcKICAgIHJldHVybiBvdXQKCmRlZiBzaWRlX2Zyb21fY29ybmVyX3goaCk6CiAgICBvdXQgPSB7fQogICAgZm9yIHN0LCBnIGluIGguZ3JvdXBieSgnU3R1ZHlJbnN0YW5jZVVJRCcpOgogICAgICAgIHhzID0gW10KICAgICAgICBmb3IgciBpbiBnLml0ZXJ0dXBsZXMoaW5kZXg9RmFsc2UpOgogICAgICAgICAgICBpcHAgPSBfaGRyX3ZlYyhnZXRhdHRyKHIsICdJbWFnZVBvc2l0aW9uUGF0aWVudCcsIE5vbmUpLCAzKQogICAgICAgICAgICBpZiBpcHAgaXMgbm90IE5vbmUgYW5kIG5wLmlzZmluaXRlKGlwcCkuYWxsKCk6CiAgICAgICAgICAgICAgICB4cy5hcHBlbmQoZmxvYXQoaXBwWzBdKSkKICAgICAgICBpZiBub3QgeHM6CiAgICAgICAgICAgIG91dFtzdF0gPSBOb25lCiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgeCA9IGZsb2F0KG5wLm1lZGlhbih4cykpCiAgICAgICAgb3V0W3N0XSA9IE5vbmUgaWYgYWJzKHgpIDwgTEVHQUNZX0xBVF9PRkZTRVRfTU0gZWxzZSAnUicgaWYgeCA8IDAgZWxzZSAnTCcKICAgIHJldHVybiBvdXQKCmRlZiBsYXRfb2YoaCwgdGFnPScnKToKICAgIGdlbyA9IHNpZGVfZnJvbV9jb3JuZXJfeChoKSBpZiBSVUxFU1snbGF0J10gPT0gJ2Nvcm5lcl94JyBlbHNlIHNpZGVfZnJvbV9nZW9tZXRyeShoKQogICAgZCwgbl90YWcsIG5fZ2VvLCBuX25vbmUsIG5fZGlzYWdyZWUgPSAoe30sIDAsIDAsIDAsIDApCiAgICBmb3Igc3QsIGcgaW4gaC5ncm91cGJ5KCdTdHVkeUluc3RhbmNlVUlEJyk6CiAgICAgICAgdiA9IFtzdHIoeCkuc3RyaXAoKS51cHBlcigpIGZvciB4IGluIGdbJ0xhdGVyYWxpdHknXS5kcm9wbmEoKV0KICAgICAgICBpZiBSVUxFU1snbGF0J10gPT0gJ2Nvcm5lcl94JyBhbmQgJ0ltYWdlTGF0ZXJhbGl0eScgaW4gZy5jb2x1bW5zOgogICAgICAgICAgICB2ICs9IFtzdHIoeCkuc3RyaXAoKS51cHBlcigpIGZvciB4IGluIGdbJ0ltYWdlTGF0ZXJhbGl0eSddLmRyb3BuYSgpXQogICAgICAgIHYgPSBbeFswXSBmb3IgeCBpbiB2IGlmIHggYW5kIHhbMF0gaW4gKCdMJywgJ1InKV0KICAgICAgICBzaWRlID0gdlswXSBpZiB2IGVsc2UgTm9uZQogICAgICAgIGlmIHNpZGUgaXMgbm90IE5vbmU6CiAgICAgICAgICAgIG5fdGFnICs9IDEKICAgICAgICAgICAgaWYgZ2VvLmdldChzdCkgaXMgbm90IE5vbmUgYW5kIGdlb1tzdF0gIT0gc2lkZToKICAgICAgICAgICAgICAgIG5fZGlzYWdyZWUgKz0gMQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIHNpZGUgPSBnZW8uZ2V0KHN0KQogICAgICAgICAgICBuX2dlbyArPSBzaWRlIGlzIG5vdCBOb25lCiAgICAgICAgICAgIG5fbm9uZSArPSBzaWRlIGlzIE5vbmUKICAgICAgICBkW3N0XSA9IHNpZGUKICAgIGxvZyhmJ3t0YWd9bGF0ZXJhbGl0eToge25fdGFnfSBmcm9tIHRoZSB0YWcsIHtuX2dlb30gZnJvbSBnZW9tZXRyeSwge25fbm9uZX0gdW5yZXNvbHZlZDsgdGFnIGFuZCBnZW9tZXRyeSBkaXNhZ3JlZSBvbiB7bl9kaXNhZ3JlZX0gKHtuX2Rpc2FncmVlIC8gbWF4KG5fdGFnLCAxKTouMSV9IG9mIHRoZSB0YWdnZWQpJykKICAgIHJldHVybiBkCgpkZWYgcHJvYmUoaXRlbSk6CiAgICBzcGxpdCwgc3R1ZHksIHNlcmllcywgcGF0aCA9IGl0ZW0KICAgIHJvdyA9IHsnc3BsaXQnOiBzcGxpdCwgJ1N0dWR5SW5zdGFuY2VVSUQnOiBzdHVkeSwgJ1Nlcmllc0luc3RhbmNlVUlEJzogc2VyaWVzLCAnZGlyJzogcGF0aH0KICAgIHRyeToKICAgICAgICBmaWxlcyA9IHNvcnRlZCgoZS5uYW1lIGZvciBlIGluIG9zLnNjYW5kaXIocGF0aCkgaWYgZS5uYW1lLmVuZHN3aXRoKCcuZGNtJykpKQogICAgICAgIHJvd1snZmlsZXMnXSA9IGZpbGVzCiAgICAgICAgcm93WyduX3NsaWNlcyddID0gbGVuKGZpbGVzKQogICAgICAgIGlmIG5vdCBmaWxlczoKICAgICAgICAgICAgcmV0dXJuIHJvdwogICAgICAgIGRzID0gcHlkaWNvbS5kY21yZWFkKG9zLnBhdGguam9pbihwYXRoLCBmaWxlc1tsZW4oZmlsZXMpIC8vIDJdKSwgc3RvcF9iZWZvcmVfcGl4ZWxzPVRydWUsIGZvcmNlPVRydWUpCiAgICAgICAgZm9yIHQgaW4gSERSX1RBR1M6CiAgICAgICAgICAgIHYgPSBnZXRhdHRyKGRzLCB0LCBOb25lKQogICAgICAgICAgICBpZiB2IGlzIE5vbmU6CiAgICAgICAgICAgICAgICByb3dbdF0gPSBOb25lCiAgICAgICAgICAgIGVsaWYgaXNpbnN0YW5jZSh2LCAobGlzdCwgdHVwbGUpKSBvciB0eXBlKHYpLl9fbmFtZV9fID09ICdNdWx0aVZhbHVlJzoKICAgICAgICAgICAgICAgIHJvd1t0XSA9ICd8Jy5qb2luKChzdHIoeCkgZm9yIHggaW4gdikpCiAgICAgICAgICAgIGVsc2U6CiAgICAgICAgICAgICAgICByb3dbdF0gPSBzdHIodikKICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZXhjOgogICAgICAgIHJvd1snZXJyJ10gPSBzdHIoZXhjKVs6MTIwXQogICAgcmV0dXJuIHJvdwoKZGVmIHdhbGsoc3BsaXQpOgogICAga2V5PShzdHIoUk9PVC5yZXNvbHZlKCkpLHN0cihzcGxpdCkpCiAgICBpZiBrZXkgaW4gX1JTTkFfSEVBREVSU19NRU1POgogICAgICAgIHJldHVybiBfUlNOQV9IRUFERVJTX01FTU9ba2V5XS5jb3B5KGRlZXA9VHJ1ZSkKICAgIGJhc2UgPSBST09UIC8gc3BsaXQKICAgIGl0ZW1zID0gW10KICAgIGlmIG5vdCBiYXNlLmlzX2RpcigpOgogICAgICAgIHJldHVybiBwZC5EYXRhRnJhbWUoY29sdW1ucz1bJ3NwbGl0JywgJ1N0dWR5SW5zdGFuY2VVSUQnLCAnU2VyaWVzSW5zdGFuY2VVSUQnLCAnZGlyJywgJ2ZpbGVzJywgJ25fc2xpY2VzJ10gKyBIRFJfVEFHUykKICAgIGZvciBzdHVkeSBpbiBvcy5zY2FuZGlyKGJhc2UpOgogICAgICAgIGlmIHN0dWR5LmlzX2RpcigpOgogICAgICAgICAgICBmb3Igc2VyaWVzIGluIG9zLnNjYW5kaXIoc3R1ZHkucGF0aCk6CiAgICAgICAgICAgICAgICBpZiBzZXJpZXMuaXNfZGlyKCk6CiAgICAgICAgICAgICAgICAgICAgaXRlbXMuYXBwZW5kKChzcGxpdCwgc3R1ZHkubmFtZSwgc2VyaWVzLm5hbWUsIHNlcmllcy5wYXRoKSkKICAgIHdpdGggVGhyZWFkUG9vbEV4ZWN1dG9yKG1heF93b3JrZXJzPUhEUl9USFJFQURTKSBhcyBwb29sOgogICAgICAgIHJvd3MgPSBsaXN0KHBvb2wubWFwKHByb2JlLCBpdGVtcykpCiAgICByZXN1bHQ9cGQuRGF0YUZyYW1lKHJvd3MpCiAgICBfUlNOQV9IRUFERVJTX01FTU9ba2V5XT1yZXN1bHQuY29weShkZWVwPVRydWUpCiAgICByZXR1cm4gcmVzdWx0CgpkZWYgYW5ub3RhdGUoZGYpOgogICAgZGVzYyA9IGRmWydTZXJpZXNEZXNjcmlwdGlvbiddLmZpbGxuYSgnJykgKyAnICcgKyBkZlsnU2VxdWVuY2VOYW1lJ10uZmlsbG5hKCcnKQogICAgZGVzYyA9IGRlc2Muc3RyLmxvd2VyKCkuc3RyLnJlcGxhY2UoX1NFUCwgJyAnLCByZWdleD1UcnVlKQogICAgb3B0cyA9IGRmWydTY2FuT3B0aW9ucyddLmZpbGxuYSgnJykuc3RyLnVwcGVyKCkuc3RyLnNwbGl0KCd8JykKICAgIG9wdHNfZnMgPSBvcHRzLmFwcGx5KGxhbWJkYSB0czogYW55KCh0LnN0cmlwKCkgaW4gRkFUU0FUX09QVFMgZm9yIHQgaW4gdHMpKSkKICAgIGRmWydmYXRzYXQnXSA9IGRlc2Muc3RyLmNvbnRhaW5zKF9GQVRTQVRfUlgpIHwgb3B0c19mcwogICAgdHIgPSBwZC50b19udW1lcmljKGRmWydSZXBldGl0aW9uVGltZSddLCBlcnJvcnM9J2NvZXJjZScpCiAgICB0ZSA9IHBkLnRvX251bWVyaWMoZGZbJ0VjaG9UaW1lJ10sIGVycm9ycz0nY29lcmNlJykKICAgIGdyZSA9IGRmWydTY2FubmluZ1NlcXVlbmNlJ10uZmlsbG5hKCcnKS5zdHIudXBwZXIoKS5zdHIuY29udGFpbnMoJ0dSJykKICAgIHQxLCB0MiwgcGR3ID0gKGRlc2Muc3RyLmNvbnRhaW5zKF9UMV9SWCksIGRlc2Muc3RyLmNvbnRhaW5zKF9UMl9SWCksIGRlc2Muc3RyLmNvbnRhaW5zKF9QRF9SWCkpCiAgICBkZlsnd2VpZ2h0J10gPSBucC53aGVyZSh0MSAmIH50MiAmIH5wZHcsICdUMScsIG5wLndoZXJlKHQyICYgfnBkdywgJ1QyJywgbnAud2hlcmUocGR3LCAnUEQnLCBucC53aGVyZShncmUsICdHUkUnLCBucC53aGVyZSh0ciA8IDgwMCwgJ1QxJywgbnAud2hlcmUodGUgPiA2MCwgJ1QyJywgbnAud2hlcmUodHIgPj0gODAwLCAnUEQnLCAnVU5LJykpKSkpKSkKICAgIGRmWydmbHVpZCddID0gbnAuaXNpbihkZlsnd2VpZ2h0J10sIFsnUEQnLCAnVDInXSkKICAgIGRmWydweCddID0gcGQudG9fbnVtZXJpYyhkZlsnUGl4ZWxTcGFjaW5nJ10uZmlsbG5hKCcnKS5zdHIuc3BsaXQoJ3wnKS5zdHJbMF0ucmVwbGFjZSgnJywgbnAubmFuKSwgZXJyb3JzPSdjb2VyY2UnKQogICAgcmV0dXJuIGRmCg==", "6": "ZGVmIHBpY2tfc2xvdHMoc2VyaWVzX2RmLCBwbGFuZV9tYXApOgogICAgc2VyaWVzX2RmID0gc2VyaWVzX2RmLmNvcHkoKQogICAgc2VyaWVzX2RmWydwbGFuZSddID0gc2VyaWVzX2RmWydTZXJpZXNJbnN0YW5jZVVJRCddLm1hcChwbGFuZV9tYXApCiAgICBvdXQgPSB7fQogICAgZm9yIHN0dWR5LCBnIGluIHNlcmllc19kZi5ncm91cGJ5KCdTdHVkeUluc3RhbmNlVUlEJyk6CiAgICAgICAgY2hvc2VuID0ge30KICAgICAgICBmb3IgbmFtZSwgcGxhbmUsIGZsdWlkLCBmcyBpbiBTTE9UUzoKICAgICAgICAgICAgc2VsID0gKGdbJ3BsYW5lJ10gPT0gcGxhbmUpICYgKGdbJ2ZhdHNhdCddID09IGZzKQogICAgICAgICAgICBpZiBmbHVpZCBpcyBub3QgTm9uZToKICAgICAgICAgICAgICAgIHNlbCAmPSBnWydmbHVpZCddID09IGZsdWlkCiAgICAgICAgICAgIGNhbmQgPSBnW3NlbF0KICAgICAgICAgICAgaWYgbGVuKGNhbmQpID09IDAgYW5kIFJVTEVTWydzbG90X2ZhbGxiYWNrJ10gYW5kIChmbHVpZCBpcyBGYWxzZSk6CiAgICAgICAgICAgICAgICBjYW5kID0gZ1soZ1sncGxhbmUnXSA9PSBwbGFuZSkgJiB+Z1snZmF0c2F0J11dCiAgICAgICAgICAgIGlmIGxlbihjYW5kKToKICAgICAgICAgICAgICAgIGNob3NlbltuYW1lXSA9IGNhbmQuc29ydF92YWx1ZXMoJ25fc2xpY2VzJywgYXNjZW5kaW5nPUZhbHNlKS5pbG9jWzBdCiAgICAgICAgb3V0W3N0dWR5XSA9IGNob3NlbgogICAgcmV0dXJuIG91dAo=", "7": "T1JERVJfVEFHUyA9IFsoMzIsIDUwKSwgKDMyLCA1NSksICgzMiwgMTkpXQpERUNPREVfRkFJTEVEID0gW10KCgpkZWYgX25hdHVyYWxfa2V5KG5hbWUpOgogICAgcmV0dXJuIHR1cGxlKChpbnQoeCkgaWYgeC5pc2RpZ2l0KCkgZWxzZSB4Lmxvd2VyKCkgZm9yIHggaW4gcmUuc3BsaXQoJyhcXGQrKScsIHN0cihuYW1lKSkpKQoKZGVmIF9vcmRlcl9kb21pbmFudF9heGlzKHJlYyk6CiAgICBmaWxlcywgZCA9IChyZWNbJ2ZpbGVzJ10sIHJlY1snZGlyJ10pCiAgICByb3dzID0gW10KICAgIGZvciBwb3MsIGYgaW4gZW51bWVyYXRlKGZpbGVzKToKICAgICAgICBpcHAgPSBpbnN0ID0gTm9uZQogICAgICAgIHRyeToKICAgICAgICAgICAgZHMgPSBweWRpY29tLmRjbXJlYWQob3MucGF0aC5qb2luKGQsIGYpLCBmb3JjZT1UcnVlLCBzdG9wX2JlZm9yZV9waXhlbHM9VHJ1ZSwgc3BlY2lmaWNfdGFncz1bJ0ltYWdlUG9zaXRpb25QYXRpZW50JywgJ0luc3RhbmNlTnVtYmVyJ10pCiAgICAgICAgICAgIHJhdyA9IGdldGF0dHIoZHMsICdJbWFnZVBvc2l0aW9uUGF0aWVudCcsIE5vbmUpCiAgICAgICAgICAgIGlmIHJhdyBpcyBub3QgTm9uZSBhbmQgbGVuKHJhdykgPj0gMzoKICAgICAgICAgICAgICAgIGMgPSBucC5hc2FycmF5KHJhd1s6M10sIGR0eXBlPW5wLmZsb2F0NjQpCiAgICAgICAgICAgICAgICBpZiBucC5pc2Zpbml0ZShjKS5hbGwoKToKICAgICAgICAgICAgICAgICAgICBpcHAgPSBjCiAgICAgICAgICAgIG4gPSBnZXRhdHRyKGRzLCAnSW5zdGFuY2VOdW1iZXInLCBOb25lKQogICAgICAgICAgICBpZiBuIGlzIG5vdCBOb25lOgogICAgICAgICAgICAgICAgaW5zdCA9IGZsb2F0KG4pCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgcGFzcwogICAgICAgIHJvd3MuYXBwZW5kKChmLCBpcHAsIGluc3QsIHBvcykpCiAgICBwbGFjZWQgPSBbciBmb3IgciBpbiByb3dzIGlmIHJbMV0gaXMgbm90IE5vbmVdCiAgICBuZWVkID0gbWF4KDIsIGludCgwLjggKiBsZW4ocm93cykpKQogICAgaWYgbGVuKHBsYWNlZCkgPj0gbmVlZDoKICAgICAgICB4eXogPSBucC5zdGFjayhbclsxXSBmb3IgciBpbiBwbGFjZWRdKQogICAgICAgIGF4aXMgPSBpbnQobnAuYXJnbWF4KG5wLnB0cCh4eXosIGF4aXM9MCkpKQogICAgICAgIHNwYXJlID0gZmxvYXQobnAubmFubWVkaWFuKHh5els6LCBheGlzXSkpCiAgICAgICAgcm93cy5zb3J0KGtleT1sYW1iZGEgcjogKGZsb2F0KHJbMV1bYXhpc10pIGlmIHJbMV0gaXMgbm90IE5vbmUgZWxzZSBzcGFyZSwgclsyXSBpZiByWzJdIGlzIG5vdCBOb25lIGVsc2UgZmxvYXQoJ2luZicpLCByWzNdKSkKICAgIGVsaWYgc3VtKChyWzJdIGlzIG5vdCBOb25lIGZvciByIGluIHJvd3MpKSA+PSBuZWVkOgogICAgICAgIHJvd3Muc29ydChrZXk9bGFtYmRhIHI6IChyWzJdIGlmIHJbMl0gaXMgbm90IE5vbmUgZWxzZSBmbG9hdCgnaW5mJyksIHJbM10pKQogICAgZWxzZToKICAgICAgICByb3dzLnNvcnQoa2V5PWxhbWJkYSByOiBfbmF0dXJhbF9rZXkoclswXSkpCiAgICByZXR1cm4gKFtyWzBdIGZvciByIGluIHJvd3NdLCBUcnVlKQoKZGVmIG9yZGVyX3NsaWNlcyhyZWMpOgogICAgaWYgUlVMRVNbJ29yZGVyJ10gPT0gJ2RvbWluYW50X2F4aXMnOgogICAgICAgIHJldHVybiBfb3JkZXJfZG9taW5hbnRfYXhpcyhyZWMpCiAgICBmaWxlcywgZCA9IChyZWNbJ2ZpbGVzJ10sIHJlY1snZGlyJ10pCiAgICBrZXllZCA9IFtdCiAgICBmb3IgZiBpbiBmaWxlczoKICAgICAgICBrID0gTm9uZQogICAgICAgIGRzID0gTm9uZQogICAgICAgIHRyeToKICAgICAgICAgICAgZHMgPSBweWRpY29tLmRjbXJlYWQob3MucGF0aC5qb2luKGQsIGYpLCBmb3JjZT1UcnVlLCBzdG9wX2JlZm9yZV9waXhlbHM9VHJ1ZSwgc3BlY2lmaWNfdGFncz1PUkRFUl9UQUdTKQogICAgICAgICAgICBpb3AgPSBucC5hc2FycmF5KGRzLkltYWdlT3JpZW50YXRpb25QYXRpZW50LCBkdHlwZT1mbG9hdCkKICAgICAgICAgICAgaXBwID0gbnAuYXNhcnJheShkcy5JbWFnZVBvc2l0aW9uUGF0aWVudCwgZHR5cGU9ZmxvYXQpCiAgICAgICAgICAgIGsgPSBmbG9hdChucC5kb3QoaXBwLCBucC5jcm9zcyhpb3BbOjNdLCBpb3BbMzpdKSkpCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgayA9IGZsb2F0KGRzLkluc3RhbmNlTnVtYmVyKQogICAgICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICAgICAgayA9IE5vbmUKICAgICAgICBrZXllZC5hcHBlbmQoKGssIGYpKQogICAgaWYgYW55KChrIGlzIE5vbmUgZm9yIGssIF8gaW4ga2V5ZWQpKToKICAgICAgICByZXR1cm4gKHNvcnRlZChmaWxlcyxrZXk9X25hdHVyYWxfa2V5KSwgRmFsc2UpCiAgICByZXR1cm4gKFtmIGZvciBfLCBmIGluIHNvcnRlZChrZXllZCwga2V5PWxhbWJkYSB0OiB0WzBdKV0sIFRydWUpCgpkZWYgcmVhZF9zbG90KHJlYywgbl9zbGljZT1Ob25lLCBvdXRfc2l6ZT1Ob25lKToKICAgICIiIlNhbWUgaGVhbHRoeSBwaXhlbHMgYXMgQlRLRDsgZmFpbGVkIHNsaWNlcyBjYW5ub3QgZXJhc2UgZ29vZCBuYXRpdmUgcGxhbmVzLiIiIgogICAgbl9zbGljZSA9IEdST1VQIGlmIG5fc2xpY2UgaXMgTm9uZSBlbHNlIGludChuX3NsaWNlKQogICAgb3V0X3NpemUgPSBJTUcgaWYgb3V0X3NpemUgaXMgTm9uZSBlbHNlIGludChvdXRfc2l6ZSkKICAgIGZpbGVzLCBkLCBweCA9IHJlYy5nZXQoJ29yZGVyZWQnKSBvciByZWNbJ2ZpbGVzJ10sIHJlY1snZGlyJ10sIHJlY1sncHgnXQogICAgaWYgbm90IGZpbGVzOgogICAgICAgIHJzbmFfZXZlbnQoJ3Nsb3Rfbm9fZmlsZXMnLCBzZXJpZXM9c3RyKHJlYy5nZXQoJ1Nlcmllc0luc3RhbmNlVUlEJywgZCkpLCBkaXI9c3RyKGQpKQogICAgICAgIHJldHVybiBOb25lCiAgICBsbywgaGkgPSBpbnQoU0xJQ0VfQkFORFswXSoobGVuKGZpbGVzKS0xKSksIGludChTTElDRV9CQU5EWzFdKihsZW4oZmlsZXMpLTEpKQogICAgaWR4ID0gbnAudW5pcXVlKG5wLmxpbnNwYWNlKGxvLGhpLG5fc2xpY2UpLmFzdHlwZShpbnQpKSBpZiBoaT5sbyBlbHNlIG5wLmFycmF5KFtsZW4oZmlsZXMpLy8yXSkKICAgIHdoaWxlIGxlbihpZHgpPG5fc2xpY2U6IGlkeD1ucC5hcHBlbmQoaWR4LGlkeFstMV0pCiAgICBkZWNvZGVkLCBlcnJvcnMgPSB7fSwgW10KICAgIGZvciBpIGluIHNvcnRlZChzZXQoaW50KHYpIGZvciB2IGluIGlkeFs6bl9zbGljZV0pKToKICAgICAgICBwYXRoPW9zLnBhdGguam9pbihkLGZpbGVzW2ldKQogICAgICAgIHRyeToKICAgICAgICAgICAgZHM9cHlkaWNvbS5kY21yZWFkKHBhdGgsZm9yY2U9VHJ1ZSkKICAgICAgICAgICAgYT1kcy5waXhlbF9hcnJheS5hc3R5cGUobnAuZmxvYXQzMikKICAgICAgICAgICAgYT1hKmZsb2F0KGdldGF0dHIoZHMsJ1Jlc2NhbGVTbG9wZScsMSkgb3IgMSkrZmxvYXQoZ2V0YXR0cihkcywnUmVzY2FsZUludGVyY2VwdCcsMCkgb3IgMCkKICAgICAgICAgICAgaWYgYS5uZGltIT0yIG9yIG5vdCBucC5pc2Zpbml0ZShhKS5hbGwoKTogcmFpc2UgVmFsdWVFcnJvcignZXhwZWN0ZWQgZmluaXRlIDItRCBwaXhlbHMnKQogICAgICAgICAgICBkZWNvZGVkW2ldPWEKICAgICAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGV4YzoKICAgICAgICAgICAgZGVjb2RlZFtpXT1Ob25lOyBlcnJvcnMuYXBwZW5kKHsncGF0aCc6cGF0aCwnZXJyb3InOmYne3R5cGUoZXhjKS5fX25hbWVfX306IHtleGN9J30pCiAgICBwbGFuZXM9W2RlY29kZWRbaW50KGkpXSBmb3IgaSBpbiBpZHhbOm5fc2xpY2VdXQogICAgZ290PVtpIGZvciBpLHAgaW4gZW51bWVyYXRlKHBsYW5lcykgaWYgcCBpcyBub3QgTm9uZV0KICAgIGlmIGVycm9yczoKICAgICAgICBERUNPREVfRkFJTEVELmFwcGVuZChyZWMuZ2V0KCdTZXJpZXNJbnN0YW5jZVVJRCcsZCkpCiAgICAgICAgcnNuYV9ldmVudCgncGFydGlhbF9zbGljZV9kZWNvZGUnLHNlcmllcz1zdHIocmVjLmdldCgnU2VyaWVzSW5zdGFuY2VVSUQnLGQpKSxlcnJvcnM9ZXJyb3JzKQogICAgaWYgbm90IGdvdDoKICAgICAgICByc25hX2V2ZW50KCdzbG90X2FsbF9kZWNvZGVfZmFpbGVkJywgc2VyaWVzPXN0cihyZWMuZ2V0KCdTZXJpZXNJbnN0YW5jZVVJRCcsIGQpKSwgZGlyPXN0cihkKSwgZXJyb3JzPWVycm9yc1s6NV0pCiAgICAgICAgcmV0dXJuIE5vbmUKICAgIGZyb20gY29sbGVjdGlvbnMgaW1wb3J0IENvdW50ZXIKICAgIHNoYXBlPUNvdW50ZXIocGxhbmVzW2ldLnNoYXBlIGZvciBpIGluIGdvdCkubW9zdF9jb21tb24oMSlbMF1bMF0KICAgIG9kZD1baSBmb3IgaSBpbiBnb3QgaWYgcGxhbmVzW2ldLnNoYXBlIT1zaGFwZV0KICAgIGlmIG9kZDoKICAgICAgICByc25hX2V2ZW50KCdzbG90X3NoYXBlX21pc21hdGNoJywgc2VyaWVzPXN0cihyZWMuZ2V0KCdTZXJpZXNJbnN0YW5jZVVJRCcsIGQpKSwgZGlyPXN0cihkKSwgbWFqb3JpdHk9W2ludCh2KSBmb3IgdiBpbiBzaGFwZV0sIGRyb3BwZWQ9bGVuKG9kZCkpCiAgICAgICAgZm9yIGkgaW4gb2RkOiBwbGFuZXNbaV09Tm9uZQogICAgICAgIGdvdD1baSBmb3IgaSBpbiBnb3QgaWYgcGxhbmVzW2ldIGlzIG5vdCBOb25lXQogICAgZm9yIGkscCBpbiBlbnVtZXJhdGUocGxhbmVzKToKICAgICAgICBpZiBwIGlzIE5vbmU6CiAgICAgICAgICAgIHBsYW5lc1tpXT1ucC56ZXJvcyhzaGFwZSxucC5mbG9hdDMyKSBpZiBSVUxFU1snZGVjb2RlX2ZpbGwnXT09J3plcm8nIGVsc2UgcGxhbmVzW21pbihnb3Qsa2V5PWxhbWJkYSBqOmFicyhqLWkpKV0KICAgIHZvbD1ucC5zdGFjayhwbGFuZXMpCiAgICBpZiBweCBhbmQgbnAuaXNmaW5pdGUocHgpIGFuZCBweD4wOgogICAgICAgIHdhbnQ9aW50KHJvdW5kKENST1BfTU0vcHgpKTsgaCx3PXNoYXBlCiAgICAgICAgaWYgMTY8d2FudDxtaW4oaCx3KToKICAgICAgICAgICAgY3ksY3g9aC8vMix3Ly8yOyBoYWxmPXdhbnQvLzIKICAgICAgICAgICAgdm9sPXZvbFs6LG1heCgwLGN5LWhhbGYpOmN5K2hhbGYsbWF4KDAsY3gtaGFsZik6Y3graGFsZl0KICAgIGxvX3YsaGlfdj1ucC5wZXJjZW50aWxlKHZvbCxbMSw5OV0pCiAgICB2b2w9bnAuY2xpcCgodm9sLWxvX3YpL21heChoaV92LWxvX3YsMWUtNiksMCwxKQogICAgdD10b3JjaC5mcm9tX251bXB5KG5wLmFzY29udGlndW91c2FycmF5KHZvbCkpLnVuc3F1ZWV6ZSgwKQogICAgdD1GLmludGVycG9sYXRlKHQsc2l6ZT0ob3V0X3NpemUsb3V0X3NpemUpLG1vZGU9J2JpbGluZWFyJyxhbGlnbl9jb3JuZXJzPUZhbHNlKQogICAgcmV0dXJuICh0LnNxdWVlemUoMCkqMjU1KS5yb3VuZCgpLmNsYW1wKDAsMjU1KS50byh0b3JjaC51aW50OCkK", "8": "ZGVmIG5vcm1hbGlzZV9sYXRlcmFsaXR5KGltZywgcGxhbmUsIGxhdCk6CiAgICBpZiBsYXQgIT0gJ1InOgogICAgICAgIHJldHVybiBpbWcKICAgIGlmIHBsYW5lIGluICgnQ29yb25hbCcsICdBeGlhbCcpOgogICAgICAgIHJldHVybiB0b3JjaC5mbGlwKGltZywgZGltcz1bLTFdKQogICAgcmV0dXJuIHRvcmNoLmZsaXAoaW1nLCBkaW1zPVswXSkK", "9": "ZGVmIGJ1aWxkX2NhY2hlKHNsb3RfbWFwLCBwbGFuZV9tYXAsIGxhdF9tYXAsIHRhZyk6CiAgICBzdHVkaWVzID0gbGlzdChfUlNOQV9URVNUX0lEUykgaWYgX1JTTkFfVEVTVF9JRFMgaXMgbm90IE5vbmUgZWxzZSBzb3J0ZWQoc2xvdF9tYXApCiAgICBpZiBzZXQoc3R1ZGllcykhPXNldChzbG90X21hcCk6CiAgICAgICAgcnNuYV9ldmVudCgnY292ZXJhZ2VfbWlzbWF0Y2gnLHRhZz10YWcsbWlzc2luZz1bc3RyKHMpIGZvciBzIGluIHN0dWRpZXMgaWYgcyBub3QgaW4gc2xvdF9tYXBdWzoyMF0sZXh0cmE9bGVuKHNldChzbG90X21hcCktc2V0KHN0dWRpZXMpKSkKICAgICAgICBzbG90X21hcD17czpzbG90X21hcC5nZXQocyx7fSkgZm9yIHMgaW4gc3R1ZGllc30KICAgIHNpZHg9e3M6aSBmb3IgaSxzIGluIGVudW1lcmF0ZShzdHVkaWVzKX0KICAgIGNhY2hlPXJzbmFfYXJyYXkoKGxlbihzdHVkaWVzKSxOX1NMT1QsQ0FDSEVfU0xJQ0VTLElNRyxJTUcpLHRhZykKICAgIG1hc2s9bnAuemVyb3MoKGxlbihzdHVkaWVzKSxOX1NMT1QpLG5wLmZsb2F0MzIpCiAgICBqb2JzPVsoc3QsayxwbGFuZSxzbG90X21hcFtzdF1bbmFtZV0pIGZvciBzdCBpbiBzdHVkaWVzIGZvciBrLChuYW1lLHBsYW5lLF8sXykgaW4gZW51bWVyYXRlKFNMT1RTKSBpZiBuYW1lIGluIHNsb3RfbWFwW3N0XV0KICAgIGxvZyhmJ3t0YWd9OiB7bGVuKHN0dWRpZXMpfSBzdHVkaWVzLCB7bGVuKGpvYnMpfSBzbG90LXNlcmllcywge2NhY2hlLm5ieXRlcy8yKiozMDouMmZ9IEdpQiBwaXhlbHMnKQogICAgbWlzc2luZz1bXTsgcmV1c2VkPTAKICAgIGZvciBqb2IgaW4gam9iczoKICAgICAgICByZWM9am9iWzNdCiAgICAgICAga2V5PShzdHIoUGF0aChyZWNbJ2RpciddKS5yZXNvbHZlKCkpLFJVTEVTWydvcmRlciddLHR1cGxlKHNvcnRlZChyZWNbJ2ZpbGVzJ10pKSkKICAgICAgICBzYXZlZD1fUlNOQV9PUkRFUl9NRU1PLmdldChrZXkpCiAgICAgICAgaWYgc2F2ZWQgaXMgbm90IE5vbmU6CiAgICAgICAgICAgIHJlY1snb3JkZXJlZCddPWxpc3Qoc2F2ZWRbMF0pO3JldXNlZCs9MQogICAgICAgIGVsc2U6IG1pc3NpbmcuYXBwZW5kKChqb2Isa2V5KSkKICAgIGRlZiBvcmRlcmVkKGl0ZW0pOgogICAgICAgIGpvYixrZXk9aXRlbQogICAgICAgIHRyeToKICAgICAgICAgICAgZmlsZXMsZ29vZD1vcmRlcl9zbGljZXMoam9iWzNdKQogICAgICAgICAgICBpZiBsZW4oZmlsZXMpIT1sZW4oam9iWzNdWydmaWxlcyddKSBvciBzZXQoZmlsZXMpIT1zZXQoam9iWzNdWydmaWxlcyddKToKICAgICAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoZidvcmRlcmluZyBsb3N0L2FkZGVkIGZpbGVzICh7bGVuKGZpbGVzKX0gdnMge2xlbihqb2JbM11bImZpbGVzIl0pfSknKQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZXhjOgogICAgICAgICAgICByc25hX2V2ZW50KCdvcmRlcmluZ19mYWxsYmFjaycsdGFnPXRhZyxzZXJpZXM9c3RyKGpvYlszXS5nZXQoJ1Nlcmllc0luc3RhbmNlVUlEJykpLGVycm9yPWYne3R5cGUoZXhjKS5fX25hbWVfX306IHtleGN9JykKICAgICAgICAgICAgcmV0dXJuIGtleSxzb3J0ZWQoam9iWzNdWydmaWxlcyddKSxGYWxzZQogICAgICAgIHJldHVybiBrZXksZmlsZXMsYm9vbChnb29kKQogICAgd2l0aCBUaHJlYWRQb29sRXhlY3V0b3IobWF4X3dvcmtlcnM9T1JERVJfVEhSRUFEUykgYXMgcG9vbDoKICAgICAgICBmb3Igc3RhcnQgaW4gcmFuZ2UoMCxsZW4obWlzc2luZyksMjU2KToKICAgICAgICAgICAgcnNuYV9kZWFkbGluZSh0YWcrJyBvcmRlcmluZycpCiAgICAgICAgICAgIGJsb2NrPW1pc3Npbmdbc3RhcnQ6c3RhcnQrMjU2XQogICAgICAgICAgICBmb3IgKGpvYixfKSwoa2V5LGZpbGVzLGdvb2QpIGluIHppcChibG9jayxwb29sLm1hcChvcmRlcmVkLGJsb2NrKSk6CiAgICAgICAgICAgICAgICBqb2JbM11bJ29yZGVyZWQnXT1maWxlcztfUlNOQV9PUkRFUl9NRU1PW2tleV09KHR1cGxlKGZpbGVzKSxnb29kKQogICAgICAgICAgICAgICAgaWYgbm90IGdvb2Q6IHJzbmFfZXZlbnQoJ2RldGVybWluaXN0aWNfb3JkZXJfZmFsbGJhY2snLHNlcmllcz1zdHIoam9iWzNdLmdldCgnU2VyaWVzSW5zdGFuY2VVSUQnKSkpCiAgICBkb25lPTA7IHVuZmlsbGVkPTAKICAgIGRlZiBfdG9sZXJhbnRfcmVhZF9zbG90KGopOgogICAgICAgIHRyeTogcmV0dXJuIHJlYWRfc2xvdChqWzNdLENBQ0hFX1NMSUNFUyxJTUcpCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBleGM6CiAgICAgICAgICAgIHJzbmFfZXZlbnQoJ3Nsb3RfcmVhZF9lcnJvcicsdGFnPXRhZyxzdHVkeT1zdHIoalswXSksc2VyaWVzPXN0cihqWzNdLmdldCgnU2VyaWVzSW5zdGFuY2VVSUQnKSksZXJyb3I9Zid7dHlwZShleGMpLl9fbmFtZV9ffToge2V4Y30nKTsgcmV0dXJuIE5vbmUKICAgIHdpdGggVGhyZWFkUG9vbEV4ZWN1dG9yKG1heF93b3JrZXJzPVBJWF9USFJFQURTKSBhcyBwb29sOgogICAgICAgIGZvciBzdGFydCBpbiByYW5nZSgwLGxlbihqb2JzKSwxMjgpOgogICAgICAgICAgICByc25hX2RlYWRsaW5lKHRhZysnIGRlY29kaW5nJykKICAgICAgICAgICAgYmxvY2s9am9ic1tzdGFydDpzdGFydCsxMjhdCiAgICAgICAgICAgIGZvciAoc3QsayxwbGFuZSxyZWMpLGltZyBpbiB6aXAoYmxvY2sscG9vbC5tYXAoX3RvbGVyYW50X3JlYWRfc2xvdCxibG9jaykpOgogICAgICAgICAgICAgICAgZG9uZSs9MQogICAgICAgICAgICAgICAgaWYgaW1nIGlzIE5vbmU6CiAgICAgICAgICAgICAgICAgICAgdW5maWxsZWQrPTE7IHJzbmFfZXZlbnQoJ3Nsb3RfdW5maWxsZWQnLHRhZz10YWcsc3R1ZHk9c3RyKHN0KSxzZXJpZXM9c3RyKHJlYy5nZXQoJ1Nlcmllc0luc3RhbmNlVUlEJykpLHNsb3Q9aW50KGspKTsgY29udGludWUKICAgICAgICAgICAgICAgIGNhY2hlW3NpZHhbc3RdLGtdPW5vcm1hbGlzZV9sYXRlcmFsaXR5KGltZyxwbGFuZSxsYXRfbWFwLmdldChzdCkpLm51bXB5KCkKICAgICAgICAgICAgICAgIG1hc2tbc2lkeFtzdF0sa109MQogICAgaWYgdW5maWxsZWQ6IGxvZyhmJ3t0YWd9OiB7dW5maWxsZWR9IHNsb3QocykgbGVmdCBlbXB0eSBhbmQgbWFza2VkIChmbGFnZ2VkKScpCiAgICBfZW1wdHk9W3N0dWRpZXNbaV0gZm9yIGkgaW4gbnAuZmxhdG5vbnplcm8obWFzay5zdW0oMSk9PTApXQogICAgaWYgX2VtcHR5OgogICAgICAgIHJzbmFfZXZlbnQoJ2VtcHR5X3N0dWR5X3Jvd3MnLHRhZz10YWcsY291bnQ9bGVuKF9lbXB0eSksc3R1ZGllcz1bc3RyKHMpIGZvciBzIGluIF9lbXB0eV0pCiAgICAgICAgbG9nKGYne3RhZ306IHtsZW4oX2VtcHR5KX0gc3R1ZHkoaWVzKSB3aXRoIG5vIHNlcmllcyBtYXRjaGluZyB0aGlzIGxheW91dCBrZXB0IGFzIGFsbC16ZXJvIG1hc2tlZCByb3dzIChwYXJlbnQgcG9saWN5KToge19lbXB0eVs6NV19JykKICAgIGlmIGRvbmUhPWxlbihqb2JzKToKICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoZid7dGFnfTogaW5jb21wbGV0ZSBjYWNoZTsge2RvbmV9L3tsZW4oam9icyl9IHNsb3RzJykKICAgIGlmIGlzaW5zdGFuY2UoY2FjaGUsbnAubWVtbWFwKTogY2FjaGUuZmx1c2goKQogICAgcnNuYV9ldmVudCgnY2FjaGVfY29tcGxldGUnLHRhZz10YWcsc3R1ZGllcz1sZW4oc3R1ZGllcyksc2xvdF9qb2JzPWxlbihqb2JzKSxmaWxsZWQ9ZG9uZSxvcmRlcl9jYWNoZV9oaXRzPXJldXNlZCxieXRlcz1pbnQoY2FjaGUubmJ5dGVzKSkKICAgIGxvZyhmJ3t0YWd9OiBDT01QTEVURSB7ZG9uZX0ve2xlbihqb2JzKX07IHJldXNlZCBvcmRlcmluZyBmb3Ige3JldXNlZH0gc2VyaWVzJykKICAgIHJldHVybiBzdHVkaWVzLGNhY2hlLG1hc2sK", "10": "Y2xhc3MgU2xvdEhlYWQobm4uTW9kdWxlKToKCiAgICBkZWYgX19pbml0X18oc2VsZiwgZGltLCBuX3Nsb3QsIG5fb3V0LCBoaWRkZW49MjU2LCBwPTAuMiwgcHJpb3I9RmFsc2UpOgogICAgICAgIHN1cGVyKCkuX19pbml0X18oKQogICAgICAgIHNlbGYucHJvaiA9IG5uLlNlcXVlbnRpYWwobm4uTGF5ZXJOb3JtKGRpbSksIG5uLkxpbmVhcihkaW0sIGhpZGRlbiksIG5uLkdFTFUoKSkKICAgICAgICBzZWxmLnNsb3RfZW1iID0gbm4uUGFyYW1ldGVyKHRvcmNoLnJhbmRuKG5fc2xvdCwgaGlkZGVuKSAqIDAuMDIpCiAgICAgICAgc2VsZi5xdWVyeSA9IG5uLlBhcmFtZXRlcih0b3JjaC5yYW5kbihuX291dCwgaGlkZGVuKSAqIDAuMDIpCiAgICAgICAgc2VsZi5kcm9wID0gbm4uRHJvcG91dChwKQogICAgICAgIHNlbGYub3V0ID0gbm4uTGluZWFyKGhpZGRlbiwgbl9vdXQpCiAgICAgICAgc2VsZi5oaWRkZW4gPSBoaWRkZW4KICAgICAgICBwXyA9IHRvcmNoLnplcm9zKG5fb3V0LCBuX3Nsb3QpCiAgICAgICAgaWYgcHJpb3IgYW5kIG5fc2xvdCA9PSBsZW4oU0xPVFMpIGFuZCAobl9vdXQgPT0gbGVuKFRBUkdFVFMpKToKICAgICAgICAgICAgZm9yIHQsIHNsb3RzIGluIFNMT1RfUFJJT1JfVEFCTEUuaXRlbXMoKToKICAgICAgICAgICAgICAgIGlmIHQgaW4gVEFSR0VUUzoKICAgICAgICAgICAgICAgICAgICBwX1tUQVJHRVRTLmluZGV4KHQpLCBsaXN0KHNsb3RzKV0gPSBTTE9UX1BSSU9SX1NUUkVOR1RICiAgICAgICAgc2VsZi5wcmlvciA9IHByaW9yCiAgICAgICAgaWYgcHJpb3I6CiAgICAgICAgICAgIHNlbGYucmVnaXN0ZXJfYnVmZmVyKCdzbG90X3ByaW9yJywgcF8pCgogICAgZGVmIGZvcndhcmQoc2VsZiwgeCwgbWFzayk6CiAgICAgICAgaCA9IHNlbGYucHJvaih4KSArIHNlbGYuc2xvdF9lbWIKICAgICAgICBhdHQgPSB0b3JjaC5laW5zdW0oJ2JzaCxvaC0+Ym9zJywgaCwgc2VsZi5xdWVyeSkgLyBzZWxmLmhpZGRlbiAqKiAwLjUKICAgICAgICBpZiBzZWxmLnByaW9yOgogICAgICAgICAgICBhdHQgPSBhdHQgKyBzZWxmLnNsb3RfcHJpb3IudW5zcXVlZXplKDApCiAgICAgICAgYXR0ID0gYXR0Lm1hc2tlZF9maWxsKG1hc2sudW5zcXVlZXplKDEpIDwgMC41LCAtMTAwMDAuMCkuc29mdG1heCgtMSkKICAgICAgICBjdHggPSBzZWxmLmRyb3AodG9yY2guZWluc3VtKCdib3MsYnNoLT5ib2gnLCBhdHQsIGgpKQogICAgICAgIHJldHVybiAoY3R4ICogc2VsZi5vdXQud2VpZ2h0LnVuc3F1ZWV6ZSgwKSkuc3VtKC0xKSArIHNlbGYub3V0LmJpYXMK", "11": "Y2xhc3MgTW9kZWwobm4uTW9kdWxlKToKCiAgICBkZWYgX19pbml0X18oc2VsZiwgYmFja2JvbmUsIGRpbSwgcG9vbD0nY2xzX21lYW4nLCBwcmlvcj1GYWxzZSk6CiAgICAgICAgc3VwZXIoKS5fX2luaXRfXygpCiAgICAgICAgc2VsZi5iYWNrYm9uZSA9IGJhY2tib25lCiAgICAgICAgc2VsZi5wb29sID0gcG9vbAogICAgICAgIHNlbGYuaGVhZCA9IFNsb3RIZWFkKGRpbSAqIFBPT0xfUEFSVFNbcG9vbF0sIE5fU0xPVCwgbGVuKFRBUkdFVFMpLCBwcmlvcj1wcmlvcikKICAgICAgICBzZWxmLnJlZ2lzdGVyX2J1ZmZlcignbWVhbicsIHRvcmNoLnRlbnNvcihbMC40ODUsIDAuNDU2LCAwLjQwNl0pLnZpZXcoMSwgMywgMSwgMSkpCiAgICAgICAgc2VsZi5yZWdpc3Rlcl9idWZmZXIoJ3N0ZCcsIHRvcmNoLnRlbnNvcihbMC4yMjksIDAuMjI0LCAwLjIyNV0pLnZpZXcoMSwgMywgMSwgMSkpCgogICAgZGVmIGZvcndhcmQoc2VsZiwgaW1ncywgbWFzaywgaW1nX3NpemU9Tm9uZSk6CiAgICAgICAgQiwgUyA9IGltZ3Muc2hhcGVbOjJdCiAgICAgICAgeCA9IGltZ3MucmVzaGFwZShCICogUywgKmltZ3Muc2hhcGVbMjpdKS5mbG9hdCgpLmRpdl8oMjU1LjApCiAgICAgICAgaWYgaW1nX3NpemUgaXMgbm90IE5vbmUgYW5kIGltZ19zaXplICE9IHguc2hhcGVbLTFdOgogICAgICAgICAgICB4ID0gRi5pbnRlcnBvbGF0ZSh4LCBzaXplPShpbWdfc2l6ZSwgaW1nX3NpemUpLCBtb2RlPSdiaWxpbmVhcicsIGFsaWduX2Nvcm5lcnM9RmFsc2UpCiAgICAgICAgeCA9ICh4IC0gc2VsZi5tZWFuKSAvIHNlbGYuc3RkCiAgICAgICAgb3V0ID0gc2VsZi5iYWNrYm9uZShwaXhlbF92YWx1ZXM9eCkubGFzdF9oaWRkZW5fc3RhdGUKICAgICAgICBwYXRjaCA9IG91dFs6LCAxOl0KICAgICAgICBwYXJ0cyA9IFtvdXRbOiwgMF0sIHBhdGNoLm1lYW4oMSldCiAgICAgICAgaWYgc2VsZi5wb29sID09ICdjbHNfbWVhbl9mb2NhbCc6CiAgICAgICAgICAgIGsgPSBtYXgoMSwgcGF0Y2guc2hhcGVbMV0gLy8gOCkKICAgICAgICAgICAgcGFydHMuYXBwZW5kKHBhdGNoLnRvcGsoaywgZGltPTEpLnZhbHVlcy5tZWFuKDEpKQogICAgICAgIGZlYXQgPSB0b3JjaC5jYXQocGFydHMsIGRpbT0xKS5yZXNoYXBlKEIsIFMsIC0xKQogICAgICAgIHJldHVybiBzZWxmLmhlYWQoZmVhdCwgbWFzaykK", "12": "ZGVmIGJ1aWxkX21vZGVsKHVuZnJlZXplX2xhc3QsIHNvdXJjZT1Ob25lLCB2YXJpYW50PSdzbWFsbCcsIHBvb2w9J2Nsc19tZWFuJywgcHJpb3I9RmFsc2UpOgogICAgZnJvbSB0cmFuc2Zvcm1lcnMgaW1wb3J0IEF1dG9Nb2RlbAogICAgcCA9IHNvdXJjZSBpZiBzb3VyY2UgaXMgbm90IE5vbmUgZWxzZSBESU5PVjJfU09VUkNFCiAgICBpZiBwIGlzIE5vbmU6CiAgICAgICAgcmFpc2UgRmlsZU5vdEZvdW5kRXJyb3IoJ0RJTk92MiB3ZWlnaHRzIG5vdCBhdHRhY2hlZCcpCiAgICBiYiA9IEF1dG9Nb2RlbC5mcm9tX3ByZXRyYWluZWQoc3RyKHApKQogICAgbl9sYXllciA9IGxlbihiYi5lbmNvZGVyLmxheWVyKQogICAgZm9yIHBybSBpbiBiYi5wYXJhbWV0ZXJzKCk6CiAgICAgICAgcHJtLnJlcXVpcmVzX2dyYWQgPSBGYWxzZQogICAgZm9yIGJsayBpbiBiYi5lbmNvZGVyLmxheWVyW21heCgwLCBuX2xheWVyIC0gdW5mcmVlemVfbGFzdCk6XToKICAgICAgICBmb3IgcHJtIGluIGJsay5wYXJhbWV0ZXJzKCk6CiAgICAgICAgICAgIHBybS5yZXF1aXJlc19ncmFkID0gVHJ1ZQogICAgZm9yIHBybSBpbiBiYi5sYXllcm5vcm0ucGFyYW1ldGVycygpOgogICAgICAgIHBybS5yZXF1aXJlc19ncmFkID0gVHJ1ZQogICAgZGltID0gYmIuY29uZmlnLmhpZGRlbl9zaXplCiAgICB0cmFpbmFibGUgPSBzdW0oKHAubnVtZWwoKSBmb3IgcCBpbiBiYi5wYXJhbWV0ZXJzKCkgaWYgcC5yZXF1aXJlc19ncmFkKSkKICAgIGxvZyhmJ2JhY2tib25lOiB7bl9sYXllcn0gYmxvY2tzLCBsYXN0IHt1bmZyZWV6ZV9sYXN0fSB0cmFpbmFibGUgKHt0cmFpbmFibGUgLyAxMDAwMDAwLjA6LjFmfU0gcGFyYW1zKSwgZmVhdHVyZSBkaW0ge2RpbSAqIFBPT0xfUEFSVFNbcG9vbF19JykKICAgIHJldHVybiBNb2RlbChiYiwgZGltLCBwb29sPXBvb2wsIHByaW9yPXByaW9yKQpfc3BlZWRfb3JpZ2luYWxfYnVpbGRfbW9kZWwgPSBidWlsZF9tb2RlbAoiIiJSZXVzZSBhIENQVSBhcmNoaXRlY3R1cmUgdGVtcGxhdGU7IGV2ZXJ5IG1lbWJlciBzdGlsbCBzdHJpY3QtbG9hZHMgYW5kIGZpbmdlcnByaW50cy4iIiIKaW1wb3J0IGNvcHkgYXMgX3NwZWVkX2NvcHkKaW1wb3J0IHRocmVhZGluZyBhcyBfc3BlZWRfZGlub190aHJlYWRpbmcKCl9zcGVlZF9kaW5vX3RlbXBsYXRlcyA9IHt9Cl9zcGVlZF9kaW5vX2xvY2sgPSBfc3BlZWRfZGlub190aHJlYWRpbmcuUkxvY2soKQoKCmRlZiBidWlsZF9tb2RlbCh1bmZyZWV6ZV9sYXN0LCBzb3VyY2U9Tm9uZSwgdmFyaWFudD0nc21hbGwnLCBwb29sPSdjbHNfbWVhbicsIHByaW9yPUZhbHNlKToKICAgIGtleSA9IChpbnQodW5mcmVlemVfbGFzdCksIE5vbmUgaWYgc291cmNlIGlzIE5vbmUgZWxzZSBzdHIoc291cmNlKSwgdmFyaWFudCwKICAgICAgICAgICBwb29sLCBib29sKHByaW9yKSwgTl9TTE9ULCB0dXBsZShUQVJHRVRTKSkKICAgIHdpdGggX3NwZWVkX2Rpbm9fbG9jazoKICAgICAgICBpZiBrZXkgbm90IGluIF9zcGVlZF9kaW5vX3RlbXBsYXRlczoKICAgICAgICAgICAgdGVtcGxhdGUgPSBfc3BlZWRfb3JpZ2luYWxfYnVpbGRfbW9kZWwodW5mcmVlemVfbGFzdCwgc291cmNlLCB2YXJpYW50LCBwb29sLCBwcmlvcikKICAgICAgICAgICAgYXNzZXJ0IGFsbChwLmRldmljZS50eXBlID09ICdjcHUnIGZvciBwIGluIHRlbXBsYXRlLnBhcmFtZXRlcnMoKSkKICAgICAgICAgICAgX3NwZWVkX2Rpbm9fdGVtcGxhdGVzW2tleV0gPSB0ZW1wbGF0ZQogICAgICAgICMgTm8gbXV0YWJsZSBzdGF0ZSBvciBHUFUgdGVuc29ycyBzaGFyZWQgYmV0d2VlbiBpbmRlcGVuZGVudGx5IHRyYWluZWQgbW9kZWxzLgogICAgICAgIHJldHVybiBfc3BlZWRfY29weS5kZWVwY29weShfc3BlZWRfZGlub190ZW1wbGF0ZXNba2V5XSkK", "13": "U0hBUkVEX0RJTk9fUFJFRklYX0xBWUVSUyA9IDYKU0hBUkVEX0RJTk9fUFJFRklYX1NIQTI1NiA9ICcwYTU1Yjg5M2JkZTk3MWM4NjRlYTVhZWVhNDQzZjA3NWIxN2MwODRkMTNhYTM2ZjMwYmQxYzc4NjNmNjU1Y2Y5JwpkZWYgX3NoYXJlZF9wcmVmaXhfc3RhdGVfc2hhMjU2KHN0YXRlKToKICAgIGRpZ2VzdCA9IGhhc2hsaWIuc2hhMjU2KCkKICAgIHByZWZpeGVzID0gKAogICAgICAgICdiYWNrYm9uZS5lbWJlZGRpbmdzLicsCiAgICAgICAgKigKICAgICAgICAgICAgZidiYWNrYm9uZS5lbmNvZGVyLmxheWVyLntpbmRleH0uJwogICAgICAgICAgICBmb3IgaW5kZXggaW4gcmFuZ2UoCiAgICAgICAgICAgICAgICBTSEFSRURfRElOT19QUkVGSVhfTEFZRVJTCiAgICAgICAgICAgICkKICAgICAgICApLAogICAgKQogICAga2V5cyA9IHNvcnRlZCgKICAgICAgICBrZXkKICAgICAgICBmb3Iga2V5IGluIHN0YXRlCiAgICAgICAgaWYga2V5LnN0YXJ0c3dpdGgocHJlZml4ZXMpCiAgICApCiAgICBpZiBsZW4oa2V5cykgIT0gMTEzOgogICAgICAgIHJhaXNlIFdlaWdodHNFcnJvcigKICAgICAgICAgICAgJ3NoYXJlZCBESU5PdjIgcHJlZml4IGtleS1jb3VudCBkcmlmdDogJwogICAgICAgICAgICBmJ3tsZW4oa2V5cyl9ICE9IDExMycKICAgICAgICApCiAgICBmb3Iga2V5IGluIGtleXM6CiAgICAgICAgdGVuc29yID0gc3RhdGVba2V5XS5kZXRhY2goKS5jcHUoKS5jb250aWd1b3VzKCkKICAgICAgICBkaWdlc3QudXBkYXRlKGtleS5lbmNvZGUoKSkKICAgICAgICBkaWdlc3QudXBkYXRlKHN0cih0ZW5zb3IuZHR5cGUpLmVuY29kZSgpKQogICAgICAgIGRpZ2VzdC51cGRhdGUoc3RyKHR1cGxlKHRlbnNvci5zaGFwZSkpLmVuY29kZSgpKQogICAgICAgIGRpZ2VzdC51cGRhdGUodGVuc29yLm51bXB5KCkudG9ieXRlcygpKQogICAgcmV0dXJuIGRpZ2VzdC5oZXhkaWdlc3QoKQoKZGVmIF9zaGFyZWRfcHJlZml4X2VsaWdpYmxlKG1lbWJlcnMpOgogICAgaWYgbm90IG1lbWJlcnMgb3Igbm90IERFVlM6CiAgICAgICAgcmV0dXJuIEZhbHNlCiAgICByZXR1cm4gYWxsKAogICAgICAgICdzdGF0ZScgbm90IGluIG1lbWJlcgogICAgICAgIGFuZCBpbnQobWVtYmVyWydjb25maWcnXVsndW5mcmVlemVfbGFzdCddKQogICAgICAgID09IFNIQVJFRF9ESU5PX1BSRUZJWF9MQVlFUlMKICAgICAgICBhbmQgbWVtYmVyWydjb25maWcnXVsndmFyaWFudCddID09ICdzbWFsbCcKICAgICAgICBhbmQgbWVtYmVyWydjb25maWcnXS5nZXQoJ3Bvb2wnLCAnY2xzX21lYW4nKQogICAgICAgIGluIFBPT0xfUEFSVFMKICAgICAgICBmb3IgbWVtYmVyIGluIG1lbWJlcnMKICAgICkKCmRlZiBfc2hhcmVkX2Rpbm9fcHJlZml4KG1vZGVsLCBpbWFnZXMsIGltZ19zaXplKToKICAgIGJhdGNoLCBzbG90cyA9IGltYWdlcy5zaGFwZVs6Ml0KICAgIHZhbHVlcyA9IGltYWdlcy5yZXNoYXBlKAogICAgICAgIGJhdGNoICogc2xvdHMsCiAgICAgICAgKmltYWdlcy5zaGFwZVsyOl0sCiAgICApLmZsb2F0KCkuZGl2XygyNTUuMCkKICAgIGlmICgKICAgICAgICBpbWdfc2l6ZSBpcyBub3QgTm9uZQogICAgICAgIGFuZCBpbWdfc2l6ZSAhPSB2YWx1ZXMuc2hhcGVbLTFdCiAgICApOgogICAgICAgIHZhbHVlcyA9IEYuaW50ZXJwb2xhdGUoCiAgICAgICAgICAgIHZhbHVlcywKICAgICAgICAgICAgc2l6ZT0oaW1nX3NpemUsIGltZ19zaXplKSwKICAgICAgICAgICAgbW9kZT0nYmlsaW5lYXInLAogICAgICAgICAgICBhbGlnbl9jb3JuZXJzPUZhbHNlLAogICAgICAgICkKICAgIHZhbHVlcyA9ICgKICAgICAgICB2YWx1ZXMgLSBtb2RlbC5tZWFuCiAgICApIC8gbW9kZWwuc3RkCiAgICBoaWRkZW4gPSBtb2RlbC5iYWNrYm9uZS5lbWJlZGRpbmdzKHZhbHVlcykKICAgIGZvciBsYXllciBpbiBtb2RlbC5iYWNrYm9uZS5lbmNvZGVyLmxheWVyWwogICAgICAgIDpTSEFSRURfRElOT19QUkVGSVhfTEFZRVJTCiAgICBdOgogICAgICAgIGhpZGRlbiA9IGxheWVyKGhpZGRlbikKICAgICAgICBoaWRkZW4gPSBoaWRkZW5bMF0gaWYgaXNpbnN0YW5jZShoaWRkZW4sICh0dXBsZSwgbGlzdCkpIGVsc2UgaGlkZGVuCiAgICByZXR1cm4gaGlkZGVuCgpkZWYgX3NoYXJlZF9kaW5vX3RhaWwoCiAgICBtb2RlbCwKICAgIGhpZGRlbiwKICAgIGJhdGNoLAogICAgc2xvdHMsCiAgICBzbG90X21hc2ssCik6CiAgICB2YWx1ZSA9IGhpZGRlbgogICAgZm9yIGxheWVyIGluIG1vZGVsLmJhY2tib25lLmVuY29kZXIubGF5ZXJbCiAgICAgICAgU0hBUkVEX0RJTk9fUFJFRklYX0xBWUVSUzoKICAgIF06CiAgICAgICAgdmFsdWUgPSBsYXllcih2YWx1ZSkKICAgICAgICB2YWx1ZSA9IHZhbHVlWzBdIGlmIGlzaW5zdGFuY2UodmFsdWUsICh0dXBsZSwgbGlzdCkpIGVsc2UgdmFsdWUKICAgIHZhbHVlID0gbW9kZWwuYmFja2JvbmUubGF5ZXJub3JtKHZhbHVlKQogICAgcGF0Y2ggPSB2YWx1ZVs6LCAxOl0KICAgIHBhcnRzID0gWwogICAgICAgIHZhbHVlWzosIDBdLAogICAgICAgIHBhdGNoLm1lYW4oMSksCiAgICBdCiAgICBpZiBtb2RlbC5wb29sID09ICdjbHNfbWVhbl9mb2NhbCc6CiAgICAgICAgY291bnQgPSBtYXgoCiAgICAgICAgICAgIDEsCiAgICAgICAgICAgIHBhdGNoLnNoYXBlWzFdIC8vIDgsCiAgICAgICAgKQogICAgICAgIHBhcnRzLmFwcGVuZCgKICAgICAgICAgICAgcGF0Y2gudG9waygKICAgICAgICAgICAgICAgIGNvdW50LAogICAgICAgICAgICAgICAgZGltPTEsCiAgICAgICAgICAgICkudmFsdWVzLm1lYW4oMSkKICAgICAgICApCiAgICBmZWF0dXJlID0gdG9yY2guY2F0KAogICAgICAgIHBhcnRzLAogICAgICAgIGRpbT0xLAogICAgKS5yZXNoYXBlKAogICAgICAgIGJhdGNoLAogICAgICAgIHNsb3RzLAogICAgICAgIC0xLAogICAgKQogICAgcmV0dXJuIG1vZGVsLmhlYWQoZmVhdHVyZSwgc2xvdF9tYXNrKQoKQHRvcmNoLm5vX2dyYWQoKQpkZWYgX3ByZWRpY3Rfc2hhcmVkX2Rpbm9fbWVtYmVycygKICAgIGVudHJpZXMsCiAgICBjYWNoZSwKICAgIG1hc2ssCiAgICBpZHgsCiAgICBkZXYsCiAgICBpbWdfc2l6ZSwKICAgIGdyb3VwLAogICAgc3RhcnRzLAopOgogICAgdGFyZ2V0X2lkeCA9IHsKICAgICAgICB0YXJnZXQ6IGluZGV4CiAgICAgICAgZm9yIGluZGV4LCB0YXJnZXQgaW4gZW51bWVyYXRlKFRBUkdFVFMpCiAgICB9CiAgICBzdGF0ZXMgPSB7fQogICAgZm9yIG1lbWJlciwgbW9kZWwsIGppdHRlciBpbiBlbnRyaWVzOgogICAgICAgIGdlbmVyYXRvciA9IHRvcmNoLkdlbmVyYXRvcihkZXZpY2U9ZGV2KQogICAgICAgIGppdHRlcl9zZWVkID0gKAogICAgICAgICAgICBTRUVECiAgICAgICAgICAgICsgaW50KAogICAgICAgICAgICAgICAgaGFzaGxpYi5zaGEyNTYoCiAgICAgICAgICAgICAgICAgICAgc3RyKG1lbWJlclsnaWQnXSkuZW5jb2RlKCkKICAgICAgICAgICAgICAgICkuaGV4ZGlnZXN0KClbOjhdLAogICAgICAgICAgICAgICAgMTYsCiAgICAgICAgICAgICkKICAgICAgICApCiAgICAgICAgZ2VuZXJhdG9yLm1hbnVhbF9zZWVkKAogICAgICAgICAgICBpbnQoaml0dGVyX3NlZWQpICUgKDIgKiogNjMgLSAxKQogICAgICAgICkKICAgICAgICBzdGF0ZXNbbWVtYmVyWydpZCddXSA9IHsKICAgICAgICAgICAgJ291dCc6IFtdLAogICAgICAgICAgICAncHVibGljJzogW10sCiAgICAgICAgICAgICdzb2Z0JzogW10sCiAgICAgICAgICAgICdnZW5lcmF0b3InOiBnZW5lcmF0b3IsCiAgICAgICAgICAgICdqaXR0ZXInOiBqaXR0ZXIsCiAgICAgICAgfQogICAgICAgIG1vZGVsLmV2YWwoKQoKICAgIHJlZmVyZW5jZSA9IGVudHJpZXNbMF1bMV0KICAgIGZvciBiYXRjaF9zdGFydCBpbiByYW5nZSgKICAgICAgICAwLAogICAgICAgIGxlbihpZHgpLAogICAgICAgIEVWQUxfQkFUQ0gsCiAgICApOgogICAgICAgIHNlbGVjdGVkID0gaWR4WwogICAgICAgICAgICBiYXRjaF9zdGFydDoKICAgICAgICAgICAgYmF0Y2hfc3RhcnQgKyBFVkFMX0JBVENICiAgICAgICAgXQogICAgICAgIHNsb3RfbWFzayA9IHRvcmNoLmZyb21fbnVtcHkoCiAgICAgICAgICAgIG1hc2tbc2VsZWN0ZWRdCiAgICAgICAgKS50byhkZXYpCiAgICAgICAgYmF0Y2ggPSBsZW4oc2VsZWN0ZWQpCiAgICAgICAgc2xvdHMgPSBjYWNoZS5zaGFwZVsxXQogICAgICAgIG1lbWJlcl93aW5kb3dzID0gewogICAgICAgICAgICBtZW1iZXJbJ2lkJ106ICgKICAgICAgICAgICAgICAgIFtdLAogICAgICAgICAgICAgICAgW10sCiAgICAgICAgICAgICAgICBbXSwKICAgICAgICAgICAgKQogICAgICAgICAgICBmb3IgbWVtYmVyLCBfLCBfIGluIGVudHJpZXMKICAgICAgICB9CgogICAgICAgIGZvciBzdGFydCBpbiBzdGFydHM6CiAgICAgICAgICAgIHJvd3MgPSB0b3JjaC5mcm9tX251bXB5KAogICAgICAgICAgICAgICAgbnAuYXNjb250aWd1b3VzYXJyYXkoCiAgICAgICAgICAgICAgICAgICAgY2FjaGVbCiAgICAgICAgICAgICAgICAgICAgICAgIHNlbGVjdGVkLAogICAgICAgICAgICAgICAgICAgICAgICA6LAogICAgICAgICAgICAgICAgICAgICAgICBzdGFydDpzdGFydCArIGdyb3VwLAogICAgICAgICAgICAgICAgICAgIF0KICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgKS50byhkZXYpCiAgICAgICAgICAgIGRlZiBfZGlub19zaGFyZWRfZm9yd2FyZChfYW1wKToKICAgICAgICAgICAgICAgIHdpdGggdG9yY2guYXV0b2Nhc3QoJ2N1ZGEnLCBlbmFibGVkPV9hbXApOgogICAgICAgICAgICAgICAgICAgIF9jb21tb24gPSBfc2hhcmVkX2Rpbm9fcHJlZml4KHJlZmVyZW5jZSwgcm93cywgaW1nX3NpemUpCiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIHttZW1iZXJbJ2lkJ106IF9zaGFyZWRfZGlub190YWlsKG1vZGVsLCBfY29tbW9uLCBiYXRjaCwgc2xvdHMsIHNsb3RfbWFzaykuZmxvYXQoKSBmb3IgbWVtYmVyLCBtb2RlbCwgXyBpbiBlbnRyaWVzfQogICAgICAgICAgICBvcmlnaW5hbF9sb2dpdHMgPSBfZGlub19zaGFyZWRfZm9yd2FyZChkZXYudHlwZSA9PSAnY3VkYScpCiAgICAgICAgICAgIGlmIGRldi50eXBlID09ICdjdWRhJyBhbmQgbm90IGFsbChib29sKHRvcmNoLmlzZmluaXRlKHYpLmFsbCgpKSBmb3IgdiBpbiBvcmlnaW5hbF9sb2dpdHMudmFsdWVzKCkpOgogICAgICAgICAgICAgICAgcnNuYV9ldmVudCgnZGlub19mcDE2X25vbmZpbml0ZV9yZXRyeV9mcDMyJywgd2luZG93X3N0YXJ0PWludChzdGFydCkpCiAgICAgICAgICAgICAgICBvcmlnaW5hbF9sb2dpdHMgPSBfZGlub19zaGFyZWRfZm9yd2FyZChGYWxzZSkKCiAgICAgICAgICAgIGZvciBtZW1iZXIsIG1vZGVsLCBqaXR0ZXIgaW4gZW50cmllczoKICAgICAgICAgICAgICAgIG1lbWJlcl9pZCA9IG1lbWJlclsnaWQnXQogICAgICAgICAgICAgICAgdmlld19sb2dpdHMgPSBbCiAgICAgICAgICAgICAgICAgICAgb3JpZ2luYWxfbG9naXRzW21lbWJlcl9pZF0KICAgICAgICAgICAgICAgIF0KICAgICAgICAgICAgICAgIGlmIGppdHRlcjoKICAgICAgICAgICAgICAgICAgICBqaXR0ZXJlZCA9IGF1Z21lbnQoCiAgICAgICAgICAgICAgICAgICAgICAgIHJvd3MsCiAgICAgICAgICAgICAgICAgICAgICAgIGdlbmVyYXRvcj1zdGF0ZXNbCiAgICAgICAgICAgICAgICAgICAgICAgICAgICBtZW1iZXJfaWQKICAgICAgICAgICAgICAgICAgICAgICAgXVsnZ2VuZXJhdG9yJ10sCiAgICAgICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgICAgIHdpdGggdG9yY2guYXV0b2Nhc3QoCiAgICAgICAgICAgICAgICAgICAgICAgICdjdWRhJywKICAgICAgICAgICAgICAgICAgICAgICAgZW5hYmxlZD1kZXYudHlwZSA9PSAnY3VkYScsCiAgICAgICAgICAgICAgICAgICAgKToKICAgICAgICAgICAgICAgICAgICAgICAgaml0dGVyX2hpZGRlbiA9ICgKICAgICAgICAgICAgICAgICAgICAgICAgICAgIF9zaGFyZWRfZGlub19wcmVmaXgoCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcmVmZXJlbmNlLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGppdHRlcmVkLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGltZ19zaXplLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAgICAgICAgIHZpZXdfbG9naXRzLmFwcGVuZCgKICAgICAgICAgICAgICAgICAgICAgICAgICAgIF9zaGFyZWRfZGlub190YWlsKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG1vZGVsLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGppdHRlcl9oaWRkZW4sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgYmF0Y2gsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgc2xvdHMsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgc2xvdF9tYXNrLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgKS5mbG9hdCgpCiAgICAgICAgICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgICAgICBpZiBkZXYudHlwZSA9PSAnY3VkYScgYW5kIG5vdCBib29sKHRvcmNoLmlzZmluaXRlKHZpZXdfbG9naXRzWy0xXSkuYWxsKCkpOgogICAgICAgICAgICAgICAgICAgICAgICByc25hX2V2ZW50KCdkaW5vX2ppdHRlcl9mcDE2X25vbmZpbml0ZV9kcm9wcGVkJywgbWVtYmVyPXN0cihtZW1iZXJfaWQpKQogICAgICAgICAgICAgICAgICAgICAgICB2aWV3X2xvZ2l0cy5wb3AoKQogICAgICAgICAgICAgICAgdmlld19wcm9icyA9IFsKICAgICAgICAgICAgICAgICAgICB0b3JjaC5zaWdtb2lkKHZhbHVlKQogICAgICAgICAgICAgICAgICAgIGZvciB2YWx1ZSBpbiB2aWV3X2xvZ2l0cwogICAgICAgICAgICAgICAgXQogICAgICAgICAgICAgICAgcHJvYmFiaWxpdGllcywgbG9naXRzLCBvcmlnaW5hbHMgPSAoCiAgICAgICAgICAgICAgICAgICAgbWVtYmVyX3dpbmRvd3NbbWVtYmVyX2lkXQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgbG9naXRzLmFwcGVuZCgKICAgICAgICAgICAgICAgICAgICB0b3JjaC5zdGFjaygKICAgICAgICAgICAgICAgICAgICAgICAgdmlld19sb2dpdHMKICAgICAgICAgICAgICAgICAgICApLm1lYW4oMCkKICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIHByb2JhYmlsaXRpZXMuYXBwZW5kKAogICAgICAgICAgICAgICAgICAgIHRvcmNoLnN0YWNrKAogICAgICAgICAgICAgICAgICAgICAgICB2aWV3X3Byb2JzCiAgICAgICAgICAgICAgICAgICAgKS5tZWFuKDApCiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICBvcmlnaW5hbHMuYXBwZW5kKAogICAgICAgICAgICAgICAgICAgIHZpZXdfcHJvYnNbMF0KICAgICAgICAgICAgICAgICkKCiAgICAgICAgZm9yIG1lbWJlciwgXywgXyBpbiBlbnRyaWVzOgogICAgICAgICAgICBtZW1iZXJfaWQgPSBtZW1iZXJbJ2lkJ10KICAgICAgICAgICAgd2luX3Byb2JzLCB3aW5fbG9naXRzLCB3aW5fb3JpZ2luYWxzID0gKAogICAgICAgICAgICAgICAgbWVtYmVyX3dpbmRvd3NbbWVtYmVyX2lkXQogICAgICAgICAgICApCiAgICAgICAgICAgIHByb2JhYmlsaXRpZXMgPSB0b3JjaC5zdGFjayh3aW5fcHJvYnMpCiAgICAgICAgICAgIGxvZ2l0cyA9IHRvcmNoLnN0YWNrKHdpbl9sb2dpdHMpCiAgICAgICAgICAgIG9yaWdpbmFscyA9IHRvcmNoLnN0YWNrKHdpbl9vcmlnaW5hbHMpCiAgICAgICAgICAgIHZhbHVlID0gKAogICAgICAgICAgICAgICAgdG9yY2guc2lnbW9pZChsb2dpdHMubWVhbigwKSkKICAgICAgICAgICAgICAgIGlmIFRUQV9QT09MID09ICdsb2dpdCcKICAgICAgICAgICAgICAgIGVsc2UgcHJvYmFiaWxpdGllcy5tZWFuKDApCiAgICAgICAgICAgICkKICAgICAgICAgICAgdmFsdWUgPSBhcHBseV90YXJnZXRfd2luZG93X3Bvb2woCiAgICAgICAgICAgICAgICB2YWx1ZSwKICAgICAgICAgICAgICAgIHByb2JhYmlsaXRpZXMsCiAgICAgICAgICAgICAgICBsb2dpdHMsCiAgICAgICAgICAgICAgICBvcmlnaW5hbHMsCiAgICAgICAgICAgICAgICBUVEFfVEFSR0VUX1BPT0wsCiAgICAgICAgICAgICAgICB0YXJnZXRfaWR4LAogICAgICAgICAgICApCiAgICAgICAgICAgIHN0YXRlc1ttZW1iZXJfaWRdWydvdXQnXS5hcHBlbmQoCiAgICAgICAgICAgICAgICB2YWx1ZS5jcHUoKS5udW1weSgpCiAgICAgICAgICAgICkKICAgICAgICAgICAgcHVibGljX3ZhbHVlID0gYXBwbHlfdGFyZ2V0X3dpbmRvd19wb29sKAogICAgICAgICAgICAgICAgb3JpZ2luYWxzLm1lYW4oMCksCiAgICAgICAgICAgICAgICBvcmlnaW5hbHMsCiAgICAgICAgICAgICAgICBsb2dpdHMsCiAgICAgICAgICAgICAgICBvcmlnaW5hbHMsCiAgICAgICAgICAgICAgICBQVUJMSUNfRlJPTlRJRVJfVEFSR0VUX1BPT0wsCiAgICAgICAgICAgICAgICB0YXJnZXRfaWR4LAogICAgICAgICAgICApCiAgICAgICAgICAgIHN0YXRlc1ttZW1iZXJfaWRdWydwdWJsaWMnXS5hcHBlbmQoCiAgICAgICAgICAgICAgICBwdWJsaWNfdmFsdWUuY3B1KCkubnVtcHkoKQogICAgICAgICAgICApCiAgICAgICAgICAgIHN0YXRlc1ttZW1iZXJfaWRdWydzb2Z0J10uYXBwZW5kKAogICAgICAgICAgICAgICAgbGVnYWN5X2ZvbGRfc29mdF93aW5kb3dfcG9vbCgKICAgICAgICAgICAgICAgICAgICBvcmlnaW5hbHMsCiAgICAgICAgICAgICAgICAgICAgdGFyZ2V0X2lkeCwKICAgICAgICAgICAgICAgICkuY3B1KCkubnVtcHkoKQogICAgICAgICAgICApCgogICAgcmV0dXJuIHsKICAgICAgICBtZW1iZXJbJ2lkJ106ICgKICAgICAgICAgICAgbnAuY29uY2F0ZW5hdGUoc3RhdGVzW21lbWJlclsnaWQnXV1bJ291dCddKSwKICAgICAgICAgICAgbnAuY29uY2F0ZW5hdGUoc3RhdGVzW21lbWJlclsnaWQnXV1bJ3B1YmxpYyddKSwKICAgICAgICAgICAgbnAuY29uY2F0ZW5hdGUoc3RhdGVzW21lbWJlclsnaWQnXV1bJ3NvZnQnXSksCiAgICAgICAgKQogICAgICAgIGZvciBtZW1iZXIsIF8sIF8gaW4gZW50cmllcwogICAgfQoKQHRvcmNoLmluZmVyZW5jZV9tb2RlKCkKZGVmIF9jaGVja19zaGFyZWRfZGlub19wYXRoKG1vZGVsLGRldixpbWdfc2l6ZSx0YWcpOgogICAgbW9kZWwuZXZhbCgpCiAgICBnPXRvcmNoLkdlbmVyYXRvcigpLm1hbnVhbF9zZWVkKFNFRUQpCiAgICBpbT10b3JjaC5yYW5kaW50KDAsMjU2LCgyLE5fU0xPVCxHUk9VUCxpbWdfc2l6ZSxpbWdfc2l6ZSksZ2VuZXJhdG9yPWcsZHR5cGU9dG9yY2gudWludDgpLnRvKGRldikKICAgIG1hc2s9dG9yY2gub25lcygyLE5fU0xPVCxkZXZpY2U9ZGV2KTttYXNrWzEsLTFdPTAKICAgIGRpcmVjdD1tb2RlbChpbSxtYXNrLGltZ19zaXplKS5mbG9hdCgpCiAgICBzcGxpdD1fc2hhcmVkX2Rpbm9fdGFpbChtb2RlbCxfc2hhcmVkX2Rpbm9fcHJlZml4KG1vZGVsLGltLGltZ19zaXplKSwyLE5fU0xPVCxtYXNrKS5mbG9hdCgpCiAgICBpZiBub3QgdG9yY2guaXNmaW5pdGUoc3BsaXQpLmFsbCgpIG9yIG5vdCB0b3JjaC5hbGxjbG9zZShzcGxpdCxkaXJlY3QscnRvbD0xZS01LGF0b2w9MmUtNSk6CiAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKGYne3RhZ306IHNoYXJlZC1wcmVmaXggZXhlY3V0aW9uIGRpZmZlcnMgZnJvbSBmdWxsIGZvcndhcmQnKQogICAgcnNuYV9ldmVudCgnZGlub19zaGFyZWRfcGF0aF9wYXJpdHknLG1lbWJlcj10YWcsbWF4X2Ficz1mbG9hdCgoc3BsaXQtZGlyZWN0KS5hYnMoKS5tYXgoKSkpCgpkZWYgX3J1bl9zaGFyZWRfZGlub19ncm91cCgKICAgIHBhdGgsCiAgICBtZW1iZXJzLAogICAgY2FjaGUsCiAgICBtYXNrLAogICAgaWR4LAogICAgc3RhcnRzLAopOgogICAgb3JkZXJlZCA9IHNvcnRlZCgKICAgICAgICBtZW1iZXJzLAogICAgICAgIGtleT1sYW1iZGEgbWVtYmVyOiAtKAogICAgICAgICAgICBtZW1iZXIuZ2V0KCdob2xkb3V0Jykgb3IgMAogICAgICAgICksCiAgICApCiAgICAjIHJ1bl9kaW5vdjIgcmV0YWlucyBvbmx5IHN1Ym1pc3Npb25fcHVibGljXzA4OTkuY3N2LCB3aG9zZSBtZW1iZXIKICAgICMgcHJlZGljdGlvbnMgYXJlIGJ1aWx0IGV4Y2x1c2l2ZWx5IGZyb20gdGhlIHVuYXVnbWVudGVkIHZpZXdzLgogICAgIyBKaXR0ZXJpbmcgaGVyZSBhZmZlY3RlZCBvbmx5IHRoZSBpbnRlcm1lZGlhdGUgc3VibWlzc2lvbi5jc3YgdGhhdCB0aGUKICAgICMgcmV0YWluZWQgcHVibGljLWZyb250aWVyIGZpbGUgcmVwbGFjZXMgYXQgdGhlIGVuZCBvZiB0aGlzIHN0YWdlLgogICAgcGxhbnMgPSBbCiAgICAgICAgKAogICAgICAgICAgICBtZW1iZXIsCiAgICAgICAgICAgIEZhbHNlLAogICAgICAgICkKICAgICAgICBmb3IgbWVtYmVyIGluIG9yZGVyZWQKICAgIF0KICAgIGFzc2lnbm1lbnRzID0gWwogICAgICAgIHBsYW5zW2luZGV4OjpsZW4oREVWUyldCiAgICAgICAgZm9yIGluZGV4IGluIHJhbmdlKGxlbihERVZTKSkKICAgIF0KICAgIHJlc3VsdHMgPSB7fQogICAgcmVzdWx0X2xvY2sgPSB0aHJlYWRpbmcuTG9jaygpCgogICAgZGVmIHdvcmtlcihkZXYsIHBsYW5zX2Zvcl9kZXZpY2UpOgogICAgICAgIGdsb2JhbCBfRElOT1YyX01BVENIRURfTUVNQkVSUwogICAgICAgIGxvYWRlZCA9IFtdCiAgICAgICAgZm9yIG1lbWJlciwgaml0dGVyIGluIHBsYW5zX2Zvcl9kZXZpY2U6CiAgICAgICAgICAgIHdpdGggQlVJTERfTE9DSzoKICAgICAgICAgICAgICAgIGNoZWNrcG9pbnQgPSB0b3JjaC5sb2FkKAogICAgICAgICAgICAgICAgICAgIFBhdGgocGF0aCkgLyBtZW1iZXJbJ2ZpbGUnXSwKICAgICAgICAgICAgICAgICAgICBtYXBfbG9jYXRpb249J2NwdScsCiAgICAgICAgICAgICAgICAgICAgd2VpZ2h0c19vbmx5PUZhbHNlLAogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgc3RhdGUgPSBjaGVja3BvaW50Wydtb2RlbCddCiAgICAgICAgICAgICAgICBwcmVmaXhfc2hhMjU2ID0gKAogICAgICAgICAgICAgICAgICAgIF9zaGFyZWRfcHJlZml4X3N0YXRlX3NoYTI1NigKICAgICAgICAgICAgICAgICAgICAgICAgc3RhdGUKICAgICAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICBpZiAoCiAgICAgICAgICAgICAgICAgICAgcHJlZml4X3NoYTI1NgogICAgICAgICAgICAgICAgICAgICE9IFNIQVJFRF9ESU5PX1BSRUZJWF9TSEEyNTYKICAgICAgICAgICAgICAgICk6CiAgICAgICAgICAgICAgICAgICAgcmFpc2UgV2VpZ2h0c0Vycm9yKAogICAgICAgICAgICAgICAgICAgICAgICBmInttZW1iZXJbJ2lkJ119OiBmcm96ZW4gcHJlZml4ICIKICAgICAgICAgICAgICAgICAgICAgICAgJ2hhc2ggbWlzbWF0Y2g6ICcKICAgICAgICAgICAgICAgICAgICAgICAgZid7cHJlZml4X3NoYTI1Nn0nCiAgICAgICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgbW9kZWwgPSBidWlsZF9tb2RlbCgKICAgICAgICAgICAgICAgICAgICBpbnQoCiAgICAgICAgICAgICAgICAgICAgICAgIG1lbWJlclsnY29uZmlnJ11bCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAndW5mcmVlemVfbGFzdCcKICAgICAgICAgICAgICAgICAgICAgICAgXQogICAgICAgICAgICAgICAgICAgICksCiAgICAgICAgICAgICAgICAgICAgdmFyaWFudD1tZW1iZXJbJ2NvbmZpZyddWwogICAgICAgICAgICAgICAgICAgICAgICAndmFyaWFudCcKICAgICAgICAgICAgICAgICAgICBdLAogICAgICAgICAgICAgICAgICAgIHBvb2w9bWVtYmVyWydjb25maWcnXS5nZXQoCiAgICAgICAgICAgICAgICAgICAgICAgICdwb29sJywKICAgICAgICAgICAgICAgICAgICAgICAgJ2Nsc19tZWFuJywKICAgICAgICAgICAgICAgICAgICApLAogICAgICAgICAgICAgICAgICAgIHByaW9yPWJvb2woCiAgICAgICAgICAgICAgICAgICAgICAgIG1lbWJlclsnY29uZmlnJ10uZ2V0KAogICAgICAgICAgICAgICAgICAgICAgICAgICAgJ3ByaW9yJywKICAgICAgICAgICAgICAgICAgICAgICAgICAgIEZhbHNlLAogICAgICAgICAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICAgICAgKSwKICAgICAgICAgICAgICAgICkudG8oZGV2KQogICAgICAgICAgICAgICAgbW9kZWwubG9hZF9zdGF0ZV9kaWN0KHN0YXRlKQogICAgICAgICAgICAgICAgY2hlY2tfZmluZ2VycHJpbnQoCiAgICAgICAgICAgICAgICAgICAgbW9kZWwsCiAgICAgICAgICAgICAgICAgICAgZGV2LAogICAgICAgICAgICAgICAgICAgIElNRywKICAgICAgICAgICAgICAgICAgICBjaGVja3BvaW50LmdldCgnZmluZ2VycHJpbnQnKSwKICAgICAgICAgICAgICAgICAgICB0YWc9ZiJ7bWVtYmVyWydpZCddfTogIiwKICAgICAgICAgICAgICAgICkKICAgICAgICAgICAgICAgIF9jaGVja19zaGFyZWRfZGlub19wYXRoKG1vZGVsLCBkZXYsIElNRywgbWVtYmVyWydpZCddKQogICAgICAgICAgICAgICAgX0RJTk9WMl9NQVRDSEVEX01FTUJFUlMgKz0gMQogICAgICAgICAgICAgICAgZGVsIGNoZWNrcG9pbnQsIHN0YXRlCiAgICAgICAgICAgIGxvYWRlZC5hcHBlbmQoCiAgICAgICAgICAgICAgICAobWVtYmVyLCBtb2RlbCwgaml0dGVyKQogICAgICAgICAgICApCgogICAgICAgIHByZWRpY3RlZCA9IF9wcmVkaWN0X3NoYXJlZF9kaW5vX21lbWJlcnMoCiAgICAgICAgICAgIGxvYWRlZCwKICAgICAgICAgICAgY2FjaGUsCiAgICAgICAgICAgIG1hc2ssCiAgICAgICAgICAgIGlkeCwKICAgICAgICAgICAgZGV2LAogICAgICAgICAgICBJTUcsCiAgICAgICAgICAgIEdST1VQLAogICAgICAgICAgICBzdGFydHMsCiAgICAgICAgKQogICAgICAgIHdpdGggcmVzdWx0X2xvY2s6CiAgICAgICAgICAgIHJlc3VsdHMudXBkYXRlKHByZWRpY3RlZCkKICAgICAgICBkZWwgbG9hZGVkCiAgICAgICAgZ2MuY29sbGVjdCgpCiAgICAgICAgd2l0aCB0b3JjaC5jdWRhLmRldmljZShkZXYpOgogICAgICAgICAgICB0b3JjaC5jdWRhLmVtcHR5X2NhY2hlKCkKCiAgICB3aXRoIFRocmVhZFBvb2xFeGVjdXRvcihtYXhfd29ya2Vycz1sZW4oREVWUykpIGFzIHBvb2w6CiAgICAgICAgZnV0dXJlcz1bcG9vbC5zdWJtaXQod29ya2VyLGRldixwbGFuKSBmb3IgZGV2LHBsYW4gaW4gemlwKERFVlMsYXNzaWdubWVudHMpXQogICAgICAgIGZvciBmdXR1cmUgaW4gZnV0dXJlczogZnV0dXJlLnJlc3VsdCgpCiAgICBpZiBsZW4ocmVzdWx0cykgIT0gbGVuKG9yZGVyZWQpOgogICAgICAgIHJhaXNlIFdlaWdodHNFcnJvcigKICAgICAgICAgICAgJ3NoYXJlZCBESU5PdjIgd29ya2VyIGRpZCBub3QgcmV0dXJuICcKICAgICAgICAgICAgZidhbGwgbWVtYmVyczoge2xlbihyZXN1bHRzKX0gLyAnCiAgICAgICAgICAgIGYne2xlbihvcmRlcmVkKX0nCiAgICAgICAgKQogICAgbG9nKAogICAgICAgICdzaGFyZWQgZmlyc3QgNiBESU5PdjIgYmxvY2tzIGZvciAnCiAgICAgICAgZid7bGVuKG9yZGVyZWQpfSBtZW1iZXJzOyBvcmlnaW5hbCB2aWV3cyAnCiAgICAgICAgZidjb21wdXRlZCBvbmNlIHBlciB7bGVuKERFVlMpfSBkZXZpY2UocyknCiAgICApCiAgICByZXR1cm4gWwogICAgICAgICgKICAgICAgICAgICAgbWVtYmVyLAogICAgICAgICAgICByZXN1bHRzW21lbWJlclsnaWQnXV0sCiAgICAgICAgICAgIGppdHRlciwKICAgICAgICApCiAgICAgICAgZm9yIG1lbWJlciwgaml0dGVyIGluIHBsYW5zCiAgICBdCgpGSU5HRVJQUklOVF9UT0wgPSAwLjAwMgpFWFBFQ1RFRF9ESU5PVjJfTUVNQkVSUyA9IDIwCl9ESU5PVjJfTUFUQ0hFRF9NRU1CRVJTID0gMAoKZGVmIGZpbmdlcnByaW50KG1vZGVsLCBkZXYsIGltZ19zaXplLCBuX3Nsb3Q9Tm9uZSwgZ3JvdXA9Tm9uZSwgc2VlZD1Ob25lKToKICAgIG5fc2xvdCA9IE5fU0xPVCBpZiBuX3Nsb3QgaXMgTm9uZSBlbHNlIG5fc2xvdAogICAgZ3JvdXAgPSBHUk9VUCBpZiBncm91cCBpcyBOb25lIGVsc2UgZ3JvdXAKICAgIHNlZWQgPSBTRUVEIGlmIHNlZWQgaXMgTm9uZSBlbHNlIHNlZWQKICAgIGcgPSB0b3JjaC5HZW5lcmF0b3IoKS5tYW51YWxfc2VlZChzZWVkKQogICAgaW1ncyA9IHRvcmNoLnJhbmRpbnQoMCwgMjU2LCAoMiwgbl9zbG90LCBncm91cCwgaW1nX3NpemUsIGltZ19zaXplKSwgZ2VuZXJhdG9yPWcsIGR0eXBlPXRvcmNoLnVpbnQ4KS50byhkZXYpCiAgICBtYXNrID0gdG9yY2gub25lcygyLCBuX3Nsb3QsIGRldmljZT1kZXYpCiAgICBtYXNrWzEsIC0xXSA9IDAuMAogICAgd2FzX3RyYWluaW5nID0gbW9kZWwudHJhaW5pbmcKICAgIG1vZGVsLmV2YWwoKQogICAgd2l0aCB0b3JjaC5ub19ncmFkKCk6CiAgICAgICAgb3V0ID0gbW9kZWwoaW1ncywgbWFzaywgaW1nX3NpemUpLmZsb2F0KCkuY3B1KCkubnVtcHkoKQogICAgaWYgd2FzX3RyYWluaW5nOgogICAgICAgIG1vZGVsLnRyYWluKCkKICAgIHJldHVybiBvdXQKCmRlZiBjaGVja19maW5nZXJwcmludChtb2RlbCwgZGV2LCBpbWdfc2l6ZSwgZXhwZWN0ZWQsIHRvbD1GSU5HRVJQUklOVF9UT0wsIHRhZz0nJyk6CiAgICBnb3QgPSBmaW5nZXJwcmludChtb2RlbCwgZGV2LCBpbWdfc2l6ZSkKICAgIGV4cCA9IG5wLmFzYXJyYXkoZXhwZWN0ZWQsIG5wLmZsb2F0MzIpCiAgICBpZiBub3QgbnAuaXNmaW5pdGUoZ290KS5hbGwoKSBvciBub3QgbnAuaXNmaW5pdGUoZXhwKS5hbGwoKToKICAgICAgICByYWlzZSBXZWlnaHRzRXJyb3IoZid7dGFnfTogbm9uZmluaXRlIGZpbmdlcnByaW50JykKICAgIGlmIGdvdC5zaGFwZSAhPSBleHAuc2hhcGU6CiAgICAgICAgcmFpc2UgV2VpZ2h0c0Vycm9yKGYne3RhZ31maW5nZXJwcmludCBzaGFwZSB7Z290LnNoYXBlfSAhPSBzdG9yZWQge2V4cC5zaGFwZX06IHRoZSBhcmNoaXRlY3R1cmUgaXMgbm90IHRoZSBvbmUgdGhlc2Ugd2VpZ2h0cyB3ZXJlIGZpdHRlZCB0bycpCiAgICBkID0gZmxvYXQobnAuYWJzKGdvdCAtIGV4cCkubWF4KCkpCiAgICBpZiBkID4gdG9sOgogICAgICAgIHJhaXNlIFdlaWdodHNFcnJvcihmJ3t0YWd9ZmluZ2VycHJpbnQgZGlmZmVycyBieSB7ZDouNGd9ICh0b2xlcmFuY2Uge3RvbDpnfSkuIFRoZSB3ZWlnaHRzIGxvYWQgYnV0IGRvIG5vdCBjb21wdXRlIHdoYXQgdGhleSBjb21wdXRlZCB3aGVuIGZpdHRlZCAtIHByZXByb2Nlc3NpbmcsIHJlc29sdXRpb24gb3IgYXJjaGl0ZWN0dXJlIGhhcyBtb3ZlZCBiZXR3ZWVuIHRoZSB0d28gcnVucy4nKQogICAgbG9nKGYne3RhZ31maW5nZXJwcmludCBtYXRjaGVzIHdpdGhpbiB7ZDouMmd9JykKICAgIHJldHVybiBkCgpjbGFzcyBXZWlnaHRzRXJyb3IoUnVudGltZUVycm9yKToKICAgIHBhc3MKCmRlZiBmaW5kX3dlaWdodHMobmFtZT0nbWFuaWZlc3QuanNvbicpOgogICAgaW1wb3J0IGpzb24KICAgIHJvb3RzID0gWwogICAgICAgIFBhdGgoJy9rYWdnbGUvaW5wdXQvZGF0YXNldHMvcGlsa3dhbmcvcnNuYS1rbmVlLXdlaWdodHMnKSwKICAgICAgICBQYXRoKCcva2FnZ2xlL2lucHV0L3JzbmEta25lZS13ZWlnaHRzJyksCiAgICBdCiAgICB2YWxpZCA9IFtdCiAgICBmb3Igcm9vdCBpbiByb290czoKICAgICAgICBwYXRoID0gcm9vdCAvIG5hbWUKICAgICAgICBpZiBub3QgcGF0aC5pc19maWxlKCk6CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgdHJ5OgogICAgICAgICAgICBtYW4gPSBqc29uLmxvYWRzKHBhdGgucmVhZF90ZXh0KCkpCiAgICAgICAgZXhjZXB0IChPU0Vycm9yLCBWYWx1ZUVycm9yKSBhcyBleGM6CiAgICAgICAgICAgIHJhaXNlIFdlaWdodHNFcnJvcihmJ2ludmFsaWQgcGlubmVkIHdlaWdodHMgbWFuaWZlc3QgYXQge3BhdGh9OiB7ZXhjfScpIGZyb20gZXhjCiAgICAgICAgaWYgbm90IGlzaW5zdGFuY2UobWFuLmdldCgnbWVtYmVycycpLCBsaXN0KSBvciBsZW4obWFuWydtZW1iZXJzJ10pICE9IEVYUEVDVEVEX0RJTk9WMl9NRU1CRVJTOgogICAgICAgICAgICByYWlzZSBXZWlnaHRzRXJyb3IoZid7cGF0aH0gbXVzdCBsaXN0IGV4YWN0bHkge0VYUEVDVEVEX0RJTk9WMl9NRU1CRVJTfSBtZW1iZXJzJykKICAgICAgICBpZiBsZW4oe20uZ2V0KCdpZCcpIGZvciBtIGluIG1hblsnbWVtYmVycyddfSkgIT0gRVhQRUNURURfRElOT1YyX01FTUJFUlM6CiAgICAgICAgICAgIHJhaXNlIFdlaWdodHNFcnJvcihmJ3twYXRofSBoYXMgZHVwbGljYXRlIG9yIG1pc3NpbmcgbWVtYmVyIGlkcycpCiAgICAgICAgbWlzc2luZyA9IFttWydmaWxlJ10gZm9yIG0gaW4gbWFuWydtZW1iZXJzJ10gaWYgbm90IChyb290IC8gbVsnZmlsZSddKS5pc19maWxlKCldCiAgICAgICAgaWYgbWlzc2luZzoKICAgICAgICAgICAgcmFpc2UgV2VpZ2h0c0Vycm9yKAogICAgICAgICAgICAgICAgZid7cm9vdH0gbGlzdHMge2xlbihtYW5bIm1lbWJlcnMiXSl9IG1lbWJlcnMgYnV0IG1pc3NlcyB7bWlzc2luZ1swXSFyfScKICAgICAgICAgICAgKQogICAgICAgIHZhbGlkLmFwcGVuZChyb290KQogICAgaWYgbGVuKHZhbGlkKSAhPSAxOgogICAgICAgIHJhaXNlIFdlaWdodHNFcnJvcihmJ2V4cGVjdGVkIG9uZSBwaW5uZWQgcnNuYS1rbmVlLXdlaWdodHMgcm9vdCwgZm91bmQge3ZhbGlkfScpCiAgICByZXR1cm4gdmFsaWRbMF0KVFRBX09WRVJMQVAgPSBUcnVlClRUQV9QT09MID0gJ3Byb2InClBVQkxJQ19GUk9OVElFUl9UQVJHRVRfUE9PTCA9IHsnRnJhY3R1cmUnOiAnbWF4JywgJ0NvbnR1c2lvbic6ICdtYXgnLCAnTWVkaWFsIE1lbmlzY3VzJzogJ21heCcsICdMYXRlcmFsIE1lbmlzY3VzJzogJ21heCcsICdBQ0wnOiAndG9wMicsICdNQ0wnOiAndG9wMicsICJCYWtlcidzIjogJ21heCd9ClRUQV9UQVJHRVRfUE9PTCA9IHsqKlBVQkxJQ19GUk9OVElFUl9UQVJHRVRfUE9PTCwgJ1N5bm92aXRpcyc6ICdvcmlnaW5hbF9tZWFuJ30KIyBOby1leHRyYS1wYXNzIGRpdmVyc2l0eSBicmFuY2g6IHNtb290aCBmb2NhbCBwb29saW5nIGlzIGV2YWx1YXRlZCBmcm9tCiMgdGhlIHNhbWUgbm8taml0dGVyIHB1YmxpYy1tZW1iZXIgd2luZG93cyBhbHJlYWR5IHVzZWQgYnkgdGhlIHBhcmVudC4KTEVHQUNZX0ZPTERfU09GVFBPT0xfQkVUQSA9IHsKICAgICdBQ0wnOiA2LjAsICdNQ0wnOiA2LjAsCiAgICAnTWVkaWFsIE1lbmlzY3VzJzogOC4wLCAnTGF0ZXJhbCBNZW5pc2N1cyc6IDguMCwKICAgICJCYWtlcidzIjogOC4wLCAnQ29udHVzaW9uJzogOC4wLCAnRnJhY3R1cmUnOiAxMC4wLAp9CkxFR0FDWV9GT0xEX1NPRlRQT09MX0FMUEhBID0gewogICAgJ0FDTCc6IDAuMjAsICdNQ0wnOiAwLjIwLAogICAgJ01lZGlhbCBNZW5pc2N1cyc6IDAuMjUsICdMYXRlcmFsIE1lbmlzY3VzJzogMC4yNSwKICAgICJCYWtlcidzIjogMC4yMCwgJ0NvbnR1c2lvbic6IDAuMjAsICdGcmFjdHVyZSc6IDAuMTUsCn0KTEVHQUNZX01FTUJFUl9XRUlHSFRfQllfVEFSR0VUID0geydMYXRlcmFsIE1lbmlzY3VzJzogMTUuMCwgJ01lZGlhbCBPQSc6IDIuNSwgJ0xhdGVyYWwgT0EnOiAxNS4wLCAnQ29udHVzaW9uJzogNS4wfQoKZGVmIHdpbmRvd19zdGFydHMobl9zbGljZSwgZ3JvdXAsIG92ZXJsYXA9Tm9uZSk6CiAgICBvdmVybGFwID0gVFRBX09WRVJMQVAgaWYgb3ZlcmxhcCBpcyBOb25lIGVsc2Ugb3ZlcmxhcAogICAgaWYgb3ZlcmxhcCBhbmQgbl9zbGljZSA+PSBncm91cDoKICAgICAgICByZXR1cm4gbGlzdChyYW5nZShuX3NsaWNlIC0gZ3JvdXAgKyAxKSkKICAgIHJldHVybiBbZyAqIGdyb3VwIGZvciBnIGluIHJhbmdlKG1heChuX3NsaWNlIC8vIGdyb3VwLCAxKSldCgpkZWYgYXBwbHlfdGFyZ2V0X3dpbmRvd19wb29sKHZhbHVlcywgcHJvYnMsIGxvZ2l0cywgb3JpZ2luYWxfcHJvYnMsIG1hcHBpbmcsIHRhcmdldF9pZHgpOgogICAgZm9yIHRhcmdldCwgbW9kZSBpbiBtYXBwaW5nLml0ZW1zKCk6CiAgICAgICAgaiA9IHRhcmdldF9pZHhbdGFyZ2V0XQogICAgICAgIGlmIG1vZGUgPT0gJ21heCc6CiAgICAgICAgICAgIHZhbHVlc1s6LCBqXSA9IHByb2JzWzosIDosIGpdLm1heCgwKS52YWx1ZXMKICAgICAgICBlbGlmIG1vZGUgPT0gJ21lYW4nOgogICAgICAgICAgICB2YWx1ZXNbOiwgal0gPSBwcm9ic1s6LCA6LCBqXS5tZWFuKDApCiAgICAgICAgZWxpZiBtb2RlID09ICdsb2dpdF9tZWFuJzoKICAgICAgICAgICAgdmFsdWVzWzosIGpdID0gdG9yY2guc2lnbW9pZChsb2dpdHNbOiwgOiwgal0ubWVhbigwKSkKICAgICAgICBlbGlmIG1vZGUgPT0gJ29yaWdpbmFsX21lYW4nOgogICAgICAgICAgICB2YWx1ZXNbOiwgal0gPSBvcmlnaW5hbF9wcm9ic1s6LCA6LCBqXS5tZWFuKDApCiAgICAgICAgZWxpZiBtb2RlIGluICgndG9wMicsICd0b3AzJyk6CiAgICAgICAgICAgIGsgPSBtaW4oaW50KG1vZGVbMzpdKSwgcHJvYnMuc2hhcGVbMF0pCiAgICAgICAgICAgIHZhbHVlc1s6LCBqXSA9IHByb2JzWzosIDosIGpdLnRvcGsoaywgZGltPTApLnZhbHVlcy5tZWFuKDApCiAgICAgICAgZWxzZToKICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmJ3Vua25vd24gVFRBIHBvb2xpbmcgbW9kZSBmb3Ige3RhcmdldH06IHttb2RlfScpCiAgICByZXR1cm4gdmFsdWVzCgpkZWYgbGVnYWN5X2ZvbGRfc29mdF93aW5kb3dfcG9vbChvcmlnaW5hbF9wcm9icywgdGFyZ2V0X2lkeCk6CiAgICB2YWx1ZXMgPSBvcmlnaW5hbF9wcm9icy5tZWFuKDApLmNsb25lKCkKICAgIGZvciB0YXJnZXQsIGJldGEgaW4gTEVHQUNZX0ZPTERfU09GVFBPT0xfQkVUQS5pdGVtcygpOgogICAgICAgIGogPSB0YXJnZXRfaWR4W3RhcmdldF0KICAgICAgICB4ID0gb3JpZ2luYWxfcHJvYnNbOiwgOiwgal0KICAgICAgICB3ZWlnaHQgPSB0b3JjaC5zb2Z0bWF4KGZsb2F0KGJldGEpICogeCwgZGltPTApCiAgICAgICAgdmFsdWVzWzosIGpdID0gKHdlaWdodCAqIHgpLnN1bSgwKQogICAgcmV0dXJuIHZhbHVlcwoKQHRvcmNoLm5vX2dyYWQoKQpkZWYgcHJlZGljdF9tZW1iZXIobW9kZWwsIGNhY2hlLCBtYXNrLCBpZHgsIGRldiwgaW1nX3NpemUsIGdyb3VwPU5vbmUsIHBvb2w9Tm9uZSwgc3RhcnRzPU5vbmUsIGppdHRlcj1GYWxzZSwgaml0dGVyX3NlZWQ9U0VFRCwgcmV0dXJuX3B1YmxpY19mcm9udGllcj1GYWxzZSk6CiAgICBncm91cCA9IEdST1VQIGlmIGdyb3VwIGlzIE5vbmUgZWxzZSBncm91cAogICAgcG9vbCA9IFRUQV9QT09MIGlmIHBvb2wgaXMgTm9uZSBlbHNlIHBvb2wKICAgIHN0YXJ0cyA9IHdpbmRvd19zdGFydHMoY2FjaGUuc2hhcGVbMl0sIGdyb3VwKSBpZiBzdGFydHMgaXMgTm9uZSBlbHNlIGxpc3Qoc3RhcnRzKQogICAgaWYgbm90IHN0YXJ0czoKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCdwcmVkaWN0X21lbWJlciB3YXMgZ2l2ZW4gbm8gd2luZG93cyB0byBhdmVyYWdlIG92ZXInKQogICAgdGFyZ2V0X2lkeCA9IHt0OiBqIGZvciBqLCB0IGluIGVudW1lcmF0ZShUQVJHRVRTKX0KICAgIHVua25vd24gPSAoc2V0KFRUQV9UQVJHRVRfUE9PTCkgfCBzZXQoUFVCTElDX0ZST05USUVSX1RBUkdFVF9QT09MKSkgLSBzZXQodGFyZ2V0X2lkeCkKICAgIGlmIHVua25vd246CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmJ3Vua25vd24gdGFyZ2V0KHMpIGluIFRUQV9UQVJHRVRfUE9PTDoge3Vua25vd259JykKICAgIGppdHRlcl9nZW4gPSB0b3JjaC5HZW5lcmF0b3IoZGV2aWNlPWRldikKICAgIGppdHRlcl9nZW4ubWFudWFsX3NlZWQoaW50KGppdHRlcl9zZWVkKSAlICgyICoqIDYzIC0gMSkpCiAgICBtb2RlbC5ldmFsKCkKICAgIG91dCwgcHVibGljX2Zyb250aWVyX291dCwgcHVibGljX3NvZnRfb3V0ID0gKFtdLCBbXSwgW10pCiAgICBmb3IgYiBpbiByYW5nZSgwLCBsZW4oaWR4KSwgRVZBTF9CQVRDSCk6CiAgICAgICAgc2VsID0gaWR4W2I6YiArIEVWQUxfQkFUQ0hdCiAgICAgICAgbSA9IHRvcmNoLmZyb21fbnVtcHkobWFza1tzZWxdKS50byhkZXYpCiAgICAgICAgd2luX3Byb2JzLCB3aW5fbG9naXRzLCB3aW5fb3JpZ2luYWxfcHJvYnMgPSAoW10sIFtdLCBbXSkKICAgICAgICBmb3Igc3QgaW4gc3RhcnRzOgogICAgICAgICAgICByb3dzID0gdG9yY2guZnJvbV9udW1weShucC5hc2NvbnRpZ3VvdXNhcnJheShjYWNoZVtzZWwsIDosIHN0OnN0ICsgZ3JvdXBdKSkudG8oZGV2KQogICAgICAgICAgICB2aWV3cyA9IFtyb3dzXSArIChbYXVnbWVudChyb3dzLCBnZW5lcmF0b3I9aml0dGVyX2dlbildIGlmIGppdHRlciBlbHNlIFtdKQogICAgICAgICAgICB2aWV3X3Byb2JzLCB2aWV3X2xvZ2l0cyA9IChbXSwgW10pCiAgICAgICAgICAgIGZvciB2aWV3IGluIHZpZXdzOgogICAgICAgICAgICAgICAgd2l0aCB0b3JjaC5hdXRvY2FzdCgnY3VkYScsIGVuYWJsZWQ9ZGV2LnR5cGUgPT0gJ2N1ZGEnKToKICAgICAgICAgICAgICAgICAgICB6ID0gbW9kZWwodmlldywgbSwgaW1nX3NpemUpLmZsb2F0KCkKICAgICAgICAgICAgICAgIGlmIGRldi50eXBlID09ICdjdWRhJyBhbmQgbm90IGJvb2wodG9yY2guaXNmaW5pdGUoeikuYWxsKCkpOgogICAgICAgICAgICAgICAgICAgIHJzbmFfZXZlbnQoJ2Rpbm9fbGVnYWN5X2ZwMTZfbm9uZmluaXRlX3JldHJ5X2ZwMzInKQogICAgICAgICAgICAgICAgICAgIHdpdGggdG9yY2guYXV0b2Nhc3QoJ2N1ZGEnLCBlbmFibGVkPUZhbHNlKToKICAgICAgICAgICAgICAgICAgICAgICAgeiA9IG1vZGVsKHZpZXcsIG0sIGltZ19zaXplKS5mbG9hdCgpCiAgICAgICAgICAgICAgICB2aWV3X2xvZ2l0cy5hcHBlbmQoeikKICAgICAgICAgICAgICAgIHZpZXdfcHJvYnMuYXBwZW5kKHRvcmNoLnNpZ21vaWQoeikpCiAgICAgICAgICAgIHdpbl9sb2dpdHMuYXBwZW5kKHRvcmNoLnN0YWNrKHZpZXdfbG9naXRzKS5tZWFuKDApKQogICAgICAgICAgICB3aW5fcHJvYnMuYXBwZW5kKHRvcmNoLnN0YWNrKHZpZXdfcHJvYnMpLm1lYW4oMCkpCiAgICAgICAgICAgIHdpbl9vcmlnaW5hbF9wcm9icy5hcHBlbmQodmlld19wcm9ic1swXSkKICAgICAgICBwcm9icyA9IHRvcmNoLnN0YWNrKHdpbl9wcm9icykKICAgICAgICBsb2dpdHMgPSB0b3JjaC5zdGFjayh3aW5fbG9naXRzKQogICAgICAgIG9yaWdpbmFsX3Byb2JzID0gdG9yY2guc3RhY2sod2luX29yaWdpbmFsX3Byb2JzKQogICAgICAgIHYgPSB0b3JjaC5zaWdtb2lkKGxvZ2l0cy5tZWFuKDApKSBpZiBwb29sID09ICdsb2dpdCcgZWxzZSBwcm9icy5tZWFuKDApCiAgICAgICAgdiA9IGFwcGx5X3RhcmdldF93aW5kb3dfcG9vbCh2LCBwcm9icywgbG9naXRzLCBvcmlnaW5hbF9wcm9icywgVFRBX1RBUkdFVF9QT09MLCB0YXJnZXRfaWR4KQogICAgICAgIG91dC5hcHBlbmQodi5jcHUoKS5udW1weSgpKQogICAgICAgIGlmIHJldHVybl9wdWJsaWNfZnJvbnRpZXI6CiAgICAgICAgICAgIHB1YmxpY192ID0gYXBwbHlfdGFyZ2V0X3dpbmRvd19wb29sKG9yaWdpbmFsX3Byb2JzLm1lYW4oMCksIG9yaWdpbmFsX3Byb2JzLCBsb2dpdHMsIG9yaWdpbmFsX3Byb2JzLCBQVUJMSUNfRlJPTlRJRVJfVEFSR0VUX1BPT0wsIHRhcmdldF9pZHgpCiAgICAgICAgICAgIHB1YmxpY19mcm9udGllcl9vdXQuYXBwZW5kKHB1YmxpY192LmNwdSgpLm51bXB5KCkpCiAgICAgICAgICAgIHB1YmxpY19zb2Z0ID0gbGVnYWN5X2ZvbGRfc29mdF93aW5kb3dfcG9vbChvcmlnaW5hbF9wcm9icywgdGFyZ2V0X2lkeCkKICAgICAgICAgICAgcHVibGljX3NvZnRfb3V0LmFwcGVuZChwdWJsaWNfc29mdC5jcHUoKS5udW1weSgpKQogICAgcHJpbWFyeSA9IG5wLmNvbmNhdGVuYXRlKG91dCkgaWYgb3V0IGVsc2UgbnAuemVyb3MoKDAsIGxlbihUQVJHRVRTKSksIG5wLmZsb2F0MzIpCiAgICBpZiBub3QgcmV0dXJuX3B1YmxpY19mcm9udGllcjoKICAgICAgICByZXR1cm4gcHJpbWFyeQogICAgcHVibGljX2Zyb250aWVyID0gbnAuY29uY2F0ZW5hdGUocHVibGljX2Zyb250aWVyX291dCkgaWYgcHVibGljX2Zyb250aWVyX291dCBlbHNlIG5wLnplcm9zKCgwLCBsZW4oVEFSR0VUUykpLCBucC5mbG9hdDMyKQogICAgcHVibGljX3NvZnQgPSBucC5jb25jYXRlbmF0ZShwdWJsaWNfc29mdF9vdXQpIGlmIHB1YmxpY19zb2Z0X291dCBlbHNlIG5wLnplcm9zKCgwLCBsZW4oVEFSR0VUUykpLCBucC5mbG9hdDMyKQogICAgcmV0dXJuIChwcmltYXJ5LCBwdWJsaWNfZnJvbnRpZXIsIHB1YmxpY19zb2Z0KQpCVUlMRF9MT0NLID0gdGhyZWFkaW5nLkxvY2soKQpTVEFURV9MT0NLID0gdGhyZWFkaW5nLkxvY2soKQpMRUdBQ1lfQlVORExFX0ZJTEUgPSAncnNuYV8yMDI2MDgwN192MS5wdCcKTEVHQUNZX1dFSUdIVCA9IDAuNQoKZGVmIGZpbmRfbGVnYWN5X2J1bmRsZSgpOgogICAgY2FuZGlkYXRlcyA9IFsKICAgICAgICBQYXRoKCcva2FnZ2xlL2lucHV0L2RhdGFzZXRzL3BpbGt3YW5nL3JzbmEta25lZS13ZWlnaHRzJykgLyBMRUdBQ1lfQlVORExFX0ZJTEUsCiAgICAgICAgUGF0aCgnL2thZ2dsZS9pbnB1dC9yc25hLWtuZWUtd2VpZ2h0cycpIC8gTEVHQUNZX0JVTkRMRV9GSUxFLAogICAgXQogICAgaGl0cyA9IFtwYXRoIGZvciBwYXRoIGluIGNhbmRpZGF0ZXMgaWYgcGF0aC5pc19maWxlKCldCiAgICBpZiBsZW4oaGl0cykgPiAxOgogICAgICAgIHJhaXNlIFdlaWdodHNFcnJvcihmJ2FtYmlndW91cyBsZWdhY3kgYnVuZGxlOiB7aGl0c30nKQogICAgcmV0dXJuIGhpdHNbMF0gaWYgaGl0cyBlbHNlIE5vbmUKCmRlZiBsZWdhY3lfZ3JvdXBfbWVtYmVycygpOgogICAgcmV0dXJuIHt9CgpkZWYgX3J1bl9tZW1iZXIocGF0aCwgbSwgZGV2LCBDdGUsIE10ZSwgaWR4LCBzdGFydHMsIGppdHRlcik6CiAgICB0MCA9IHRpbWUudGltZSgpCiAgICB3aXRoIEJVSUxEX0xPQ0s6CiAgICAgICAgaWYgJ3N0YXRlJyBpbiBtOgogICAgICAgICAgICByYWlzZSBXZWlnaHRzRXJyb3IoZiJ7bVsnaWQnXX06IGlubGluZSBsZWdhY3kgc3RhdGUgaXMgZm9yYmlkZGVuIikKICAgICAgICBjayA9IHRvcmNoLmxvYWQoUGF0aChwYXRoKSAvIG1bJ2ZpbGUnXSwgbWFwX2xvY2F0aW9uPSdjcHUnLCB3ZWlnaHRzX29ubHk9RmFsc2UpCiAgICAgICAgc3RhdGUsIGZwID0gKGNrWydtb2RlbCddLCBjay5nZXQoJ2ZpbmdlcnByaW50JykpCiAgICAgICAgaWYgZnAgaXMgTm9uZToKICAgICAgICAgICAgcmFpc2UgV2VpZ2h0c0Vycm9yKGYie21bJ2lkJ119OiBzdG9yZWQgZmluZ2VycHJpbnQgaXMgcmVxdWlyZWQiKQogICAgICAgIG1vZGVsID0gYnVpbGRfbW9kZWwoaW50KG1bJ2NvbmZpZyddWyd1bmZyZWV6ZV9sYXN0J10pLCB2YXJpYW50PW1bJ2NvbmZpZyddWyd2YXJpYW50J10sIHBvb2w9bVsnY29uZmlnJ10uZ2V0KCdwb29sJywgJ2Nsc19tZWFuJyksIHByaW9yPWJvb2wobVsnY29uZmlnJ10uZ2V0KCdwcmlvcicsIEZhbHNlKSkpLnRvKGRldikKICAgICAgICBtb2RlbC5sb2FkX3N0YXRlX2RpY3Qoc3RhdGUpCiAgICAgICAgY2hlY2tfZmluZ2VycHJpbnQobW9kZWwsIGRldiwgSU1HLCBmcCwgdGFnPWYie21bJ2lkJ119OiAiKQogICAgICAgIGdsb2JhbCBfRElOT1YyX01BVENIRURfTUVNQkVSUwogICAgICAgIF9ESU5PVjJfTUFUQ0hFRF9NRU1CRVJTICs9IDEKICAgIHRfcmVhZHkgPSB0aW1lLnRpbWUoKQogICAgaml0dGVyX3NlZWQgPSBTRUVEICsgaW50KGhhc2hsaWIuc2hhMjU2KHN0cihtWydpZCddKS5lbmNvZGUoKSkuaGV4ZGlnZXN0KClbOjhdLCAxNikKICAgIHB1YmxpY19tZW1iZXIgPSAnc3RhdGUnIG5vdCBpbiBtCiAgICBwcmVkaWN0ZWQgPSBwcmVkaWN0X21lbWJlcihtb2RlbCwgQ3RlLCBNdGUsIGlkeCwgZGV2LCBJTUcsIHN0YXJ0cz1zdGFydHMsIGppdHRlcj1qaXR0ZXIsIGppdHRlcl9zZWVkPWppdHRlcl9zZWVkLCByZXR1cm5fcHVibGljX2Zyb250aWVyPXB1YmxpY19tZW1iZXIpCiAgICBpZiBwdWJsaWNfbWVtYmVyOgogICAgICAgIHAsIHB1YmxpY19wLCBwdWJsaWNfc29mdCA9IHByZWRpY3RlZAogICAgZWxzZToKICAgICAgICBwLCBwdWJsaWNfcCwgcHVibGljX3NvZnQgPSAocHJlZGljdGVkLCBOb25lLCBOb25lKQogICAgdF9kb25lID0gdGltZS50aW1lKCkKICAgIGRlbCBtb2RlbCwgc3RhdGUKICAgIGdjLmNvbGxlY3QoKQogICAgaWYgZGV2LnR5cGUgPT0gJ2N1ZGEnOgogICAgICAgIHdpdGggdG9yY2guY3VkYS5kZXZpY2UoZGV2KToKICAgICAgICAgICAgdG9yY2guY3VkYS5lbXB0eV9jYWNoZSgpCiAgICBwYXNzZXMgPSBsZW4oc3RhcnRzKSAqICgyIGlmIGppdHRlciBlbHNlIDEpCiAgICByZXR1cm4gKHAsIHB1YmxpY19wLCBwdWJsaWNfc29mdCwgKHRfcmVhZHkgLSB0MCwgKHRfZG9uZSAtIHRfcmVhZHkpIC8gbWF4KHBhc3NlcywgMSkpKQoKZGVmIF9jb21iaW5lKHBlcl9tZW1iZXIpOgogICAgYWxsX2lkcyA9IHNvcnRlZCh7cyBmb3IgbSBpbiBwZXJfbWVtYmVyIGZvciBzIGluIG1bJ2lkcyddfSkKICAgIHBvcyA9IHtzOiBpIGZvciBpLCBzIGluIGVudW1lcmF0ZShhbGxfaWRzKX0KICAgIGFjYyA9IG5wLnplcm9zKChsZW4oYWxsX2lkcyksIGxlbihUQVJHRVRTKSksIG5wLmZsb2F0NjQpCiAgICB0b3QgPSBucC56ZXJvcyhsZW4oVEFSR0VUUyksIG5wLmZsb2F0NjQpCiAgICBmb3IgbSBpbiBwZXJfbWVtYmVyOgogICAgICAgIHRhcmdldF93ZWlnaHQgPSBtLmdldCgndGFyZ2V0X3dlaWdodCcpCiAgICAgICAgdyA9IG5wLmFzYXJyYXkodGFyZ2V0X3dlaWdodCBpZiB0YXJnZXRfd2VpZ2h0IGlzIG5vdCBOb25lIGVsc2UgW2Zsb2F0KG0uZ2V0KCd3ZWlnaHQnLCAxLjApKV0gKiBsZW4oVEFSR0VUUyksIGR0eXBlPW5wLmZsb2F0NjQpCiAgICAgICAgaWYgdy5zaGFwZSAhPSAobGVuKFRBUkdFVFMpLCkgb3IgbnAuYW55KHcgPCAwKToKICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmImludmFsaWQgdGFyZ2V0IHdlaWdodHMgZm9yIHttLmdldCgnaWQnKX06IHt3fSIpCiAgICAgICAgcnNuYV9maW5pdGUobVsncHJlZCddLHN0cihtWydpZCddKSsnIHJldGFpbmVkIHJhdycscHJvYmFiaWxpdHk9VHJ1ZSkKICAgICAgICByID0gcGQuRGF0YUZyYW1lKG1bJ3ByZWQnXSkucmFuayhwY3Q9VHJ1ZSkudG9fbnVtcHkoKQogICAgICAgIGFjY1tbcG9zW3NdIGZvciBzIGluIG1bJ2lkcyddXV0gKz0gciAqIHdbTm9uZSwgOl0KICAgICAgICB0b3QgKz0gdwogICAgaWYgbnAuYW55KHRvdCA8PSAwKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYnYXQgbGVhc3Qgb25lIHRhcmdldCBoYXMgbm8gZW5zZW1ibGUgdm90ZToge3RvdH0nKQogICAgcmV0dXJuIChhbGxfaWRzLCBhY2MgLyB0b3RbTm9uZSwgOl0pCgpkZWYgY29tYmluZV9wdWJsaWNfbWVtYmVyc19ieV9mb2xkKHBlcl9tZW1iZXIsIHByZWRfa2V5PSdwcmVkJyk6CiAgICAjIFJhdy1hdmVyYWdlIHRoZSBmb3VyIG1lbWJlcnMgd2l0aGluIGVhY2ggZm9sZCwgcmFuayBlYWNoIGZvbGQsCiAgICAjIHRoZW4gZ2l2ZSBhbGwgZml2ZSBmb2xkcyBlcXVhbCB3ZWlnaHQuCiAgICBhbGxfaWRzID0gc29ydGVkKHtzdHVkeSBmb3IgbWVtYmVyIGluIHBlcl9tZW1iZXIgZm9yIHN0dWR5IGluIG1lbWJlclsnaWRzJ119KQogICAgcG9zaXRpb24gPSB7c3R1ZHk6IGkgZm9yIGksIHN0dWR5IGluIGVudW1lcmF0ZShhbGxfaWRzKX0KICAgIGdyb3VwcyA9IHt9CiAgICBmb3IgaSwgbWVtYmVyIGluIGVudW1lcmF0ZShwZXJfbWVtYmVyKToKICAgICAgICBmb2xkID0gbWVtYmVyLmdldCgnZm9sZCcpCiAgICAgICAga2V5ID0gZidmb2xkX3tmb2xkfScgaWYgZm9sZCBpcyBub3QgTm9uZSBlbHNlIGYnbWVtYmVyX3tpfScKICAgICAgICBncm91cHMuc2V0ZGVmYXVsdChrZXksIFtdKS5hcHBlbmQobWVtYmVyKQogICAgZm9sZF9yYW5rcywgZGlhZ25vc3RpY3MgPSAoW10sIFtdKQogICAgZm9yIGtleSwgbWVtYmVyc19pbl9mb2xkIGluIHNvcnRlZChncm91cHMuaXRlbXMoKSk6CiAgICAgICAgbWF0cmljZXMgPSBbXQogICAgICAgIGZvciBtZW1iZXIgaW4gbWVtYmVyc19pbl9mb2xkOgogICAgICAgICAgICB2YWx1ZXMgPSBucC5mdWxsKChsZW4oYWxsX2lkcyksIGxlbihUQVJHRVRTKSksIG5wLm5hbiwgbnAuZmxvYXQ2NCkKICAgICAgICAgICAgdmFsdWVzW1twb3NpdGlvbltzdHVkeV0gZm9yIHN0dWR5IGluIG1lbWJlclsnaWRzJ11dXSA9IG5wLmFzYXJyYXkobWVtYmVyW3ByZWRfa2V5XSwgbnAuZmxvYXQ2NCkKICAgICAgICAgICAgaWYgbnAuaXNuYW4odmFsdWVzKS5hbnkoKToKICAgICAgICAgICAgICAgIHJhaXNlIFdlaWdodHNFcnJvcihmInttZW1iZXIuZ2V0KCdpZCcpfTogaW5jb21wbGV0ZSB7cHJlZF9rZXl9IGNvdmVyYWdlIikKICAgICAgICAgICAgbWF0cmljZXMuYXBwZW5kKHZhbHVlcykKICAgICAgICByYXdfZm9sZF9tZWFuID0gbnAubWVhbihtYXRyaWNlcywgYXhpcz0wKQogICAgICAgIGZvbGRfcmFua3MuYXBwZW5kKHBkLkRhdGFGcmFtZShyYXdfZm9sZF9tZWFuKS5yYW5rKG1ldGhvZD0nYXZlcmFnZScsIHBjdD1UcnVlKS50b19udW1weShucC5mbG9hdDY0KSkKICAgICAgICBkaWFnbm9zdGljcy5hcHBlbmQoeydlbnNlbWJsZV9ncm91cCc6IGtleSwgJ21lbWJlcnMnOiBsZW4obWVtYmVyc19pbl9mb2xkKX0pCiAgICBpZiBsZW4oZm9sZF9yYW5rcykgIT0gNToKICAgICAgICByYWlzZSBXZWlnaHRzRXJyb3IoZidsZWdhY3kgYnJhbmNoIHJlcXVpcmVzIGZpdmUgZm9sZHMsIGZvdW5kIHtsZW4oZm9sZF9yYW5rcyl9JykKICAgIHJldHVybiBhbGxfaWRzLCBucC5tZWFuKGZvbGRfcmFua3MsIGF4aXM9MCksIHBkLkRhdGFGcmFtZShkaWFnbm9zdGljcykKCmRlZiBibGVuZF9sZWdhY3lfZnJvbnRpZXJfYW5kX3NvZnQoZnJvbnRpZXJfcmFuaywgc29mdF9yYW5rKToKICAgIG91dHB1dCA9IG5wLmFzYXJyYXkoZnJvbnRpZXJfcmFuaywgbnAuZmxvYXQ2NCkuY29weSgpCiAgICBmb3IgaiwgdGFyZ2V0IGluIGVudW1lcmF0ZShUQVJHRVRTKToKICAgICAgICBhbHBoYSA9IGZsb2F0KExFR0FDWV9GT0xEX1NPRlRQT09MX0FMUEhBLmdldCh0YXJnZXQsIDAuMCkpCiAgICAgICAgaWYgYWxwaGE6CiAgICAgICAgICAgIG91dHB1dFs6LCBqXSA9ICgxLjAgLSBhbHBoYSkgKiBmcm9udGllcl9yYW5rWzosIGpdICsgYWxwaGEgKiBzb2Z0X3JhbmtbOiwgal0KICAgIHJldHVybiBvdXRwdXQKCmRlZiBpbmZlcl9mcm9tX3BhY2thZ2UocGF0aCwgZGV2PU5vbmUpOgogICAgbWFuID0ganNvbi5sb2FkcygoUGF0aChwYXRoKSAvICdtYW5pZmVzdC5qc29uJykucmVhZF90ZXh0KCkpCiAgICBtZW1iZXJzID0gbWFuWydtZW1iZXJzJ10KICAgIGlmIGxlbihtZW1iZXJzKSAhPSBFWFBFQ1RFRF9ESU5PVjJfTUVNQkVSUzoKICAgICAgICByYWlzZSBXZWlnaHRzRXJyb3IoZidleHBlY3RlZCB7RVhQRUNURURfRElOT1YyX01FTUJFUlN9IERJTk92MiBtZW1iZXJzLCBmb3VuZCB7bGVuKG1lbWJlcnMpfScpCiAgICBsb2coZid3ZWlnaHRzIHBhY2thZ2U6IHtsZW4obWVtYmVycyl9IG1lbWJlcihzKSBmcm9tIHtwYXRofTsge2xlbihERVZTKX0gZGV2aWNlKHMpJykKICAgIHRlc3RfZGYgPSBwZC5yZWFkX2NzdihST09UIC8gJ3Rlc3QuY3N2JykKICAgIHRlc3Rfc2VyaWVzID0gcGQucmVhZF9jc3YoUk9PVCAvICd0ZXN0X3Nlcmllcy5jc3YnKQogICAgcGxhbmVfbWFwID0gZGljdCh6aXAodGVzdF9zZXJpZXNbJ1Nlcmllc0luc3RhbmNlVUlEJ10sIHRlc3Rfc2VyaWVzWydBbmF0b21pY2FsX1BsYW5lJ10pKQogICAgaHRlID0gYW5ub3RhdGUod2FsaygndGVzdF9zZXJpZXMnKSkKICAgIGxvZyhmJ3Rlc3QgaGVhZGVyIHBhc3M6IHtsZW4oaHRlKX0gc2VyaWVzJykKICAgIGdyb3VwcyA9IHt9CiAgICBmb3IgbSBpbiBtZW1iZXJzOgogICAgICAgIGdyb3Vwcy5zZXRkZWZhdWx0KG1bJ3BpeGVsX2dyb3VwJ10sIFtdKS5hcHBlbmQobSkKICAgIGdyb3Vwcy51cGRhdGUobGVnYWN5X2dyb3VwX21lbWJlcnMoKSkKICAgIHBlcl9tZW1iZXIsIHB1YmxpY19mcm9udGllcl9tZW1iZXJzID0gKFtdLCBbXSkKICAgIGZhaWx1cmVzID0gW10KICAgIGFib3J0ID0gdGhyZWFkaW5nLkV2ZW50KCkKICAgIGVzdCA9IHsnZml4ZWQnOiBOb25lLCAnd2luJzogTm9uZX0KCiAgICBkZWYgYmFuayhtLCBpZHMsIHByZWQsIHN0YXJ0cywgaml0dGVyLCBwdWJsaWNfcHJlZD1Ob25lLCBwdWJsaWNfc29mdD1Ob25lKToKICAgICAgICBpZiBub3QgbnAuaXNmaW5pdGUocHJlZCkuYWxsKCk6CiAgICAgICAgICAgIHJzbmFfZXZlbnQoJ2Rpbm9fbWVtYmVyX25vbmZpbml0ZV9uZXV0cmFsJywgbWVtYmVyPXN0cihtWydpZCddKSwgcm93cz1pbnQoKH5ucC5pc2Zpbml0ZShwcmVkKSkucmVzaGFwZShsZW4ocHJlZCksIC0xKS5hbnkoYXhpcz0xKS5zdW0oKSkpCiAgICAgICAgICAgIHByZWQgPSBucC53aGVyZShucC5pc2Zpbml0ZShwcmVkKSwgcHJlZCwgMC41KQogICAgICAgIGlmIHB1YmxpY19wcmVkIGlzIG5vdCBOb25lIGFuZCBub3QgbnAuaXNmaW5pdGUocHVibGljX3ByZWQpLmFsbCgpOgogICAgICAgICAgICByc25hX2V2ZW50KCdkaW5vX21lbWJlcl9wdWJsaWNfbm9uZmluaXRlX25ldXRyYWwnLCBtZW1iZXI9c3RyKG1bJ2lkJ10pKQogICAgICAgICAgICBwdWJsaWNfcHJlZCA9IG5wLndoZXJlKG5wLmlzZmluaXRlKHB1YmxpY19wcmVkKSwgcHVibGljX3ByZWQsIDAuNSkKICAgICAgICByc25hX2Zpbml0ZShwcmVkLCBtWydpZCddKycgRElOTyByYXcnLCBwcm9iYWJpbGl0eT1UcnVlKQogICAgICAgIGlmIHB1YmxpY19wcmVkIGlzIG5vdCBOb25lOiByc25hX2Zpbml0ZShwdWJsaWNfcHJlZCxtWydpZCddKycgRElOTyByZXRhaW5lZCByYXcnLHByb2JhYmlsaXR5PVRydWUpCiAgICAgICAgcnNuYV9zYXZlX3ByZWRpY3Rpb25zKCdkaW5vXycrc3RyKG1bJ2lkJ10pLGlkcyxwdWJsaWNfcHJlZCBpZiBwdWJsaWNfcHJlZCBpcyBub3QgTm9uZSBlbHNlIHByZWQsVEFSR0VUUykKICAgICAgICBpZiBmbG9hdChucC5zdGQocHJlZCkpIDwgMWUtMDk6CiAgICAgICAgICAgIHJhaXNlIFdlaWdodHNFcnJvcihmInttWydpZCddfTogZGVnZW5lcmF0ZSBwcmVkaWN0aW9ucyIpCiAgICAgICAgd2l0aCBTVEFURV9MT0NLOgogICAgICAgICAgICBwZXJfbWVtYmVyLmFwcGVuZCh7J2lkJzogbVsnaWQnXSwgJ2ZvbGQnOiBtLmdldCgnZm9sZCcpLCAnaWRzJzogaWRzLCAncHJlZCc6IHByZWQsICd3ZWlnaHQnOiBtLmdldCgnd2VpZ2h0JywgMS4wKSwgJ3RhcmdldF93ZWlnaHQnOiBtLmdldCgndGFyZ2V0X3dlaWdodCcpLCAnaG9sZG91dCc6IG0uZ2V0KCdob2xkb3V0Jyl9KQogICAgICAgICAgICBpZiBwdWJsaWNfcHJlZCBpcyBub3QgTm9uZSBhbmQgbGVuKHN0YXJ0cykgPT0gbGVuKHN0YXJ0c19mdWxsKToKICAgICAgICAgICAgICAgIGlmIGZsb2F0KG5wLnN0ZChwdWJsaWNfcHJlZCkpIDwgMWUtMDk6CiAgICAgICAgICAgICAgICAgICAgcmFpc2UgV2VpZ2h0c0Vycm9yKGYie21bJ2lkJ119OiBkZWdlbmVyYXRlIHB1YmxpYy1mcm9udGllciBwcmVkaWN0aW9uIikKICAgICAgICAgICAgICAgIHB1YmxpY19mcm9udGllcl9tZW1iZXJzLmFwcGVuZCh7J2lkJzogbVsnaWQnXSwgJ2ZvbGQnOiBtLmdldCgnZm9sZCcpLCAnaWRzJzogaWRzLCAncHJlZCc6IHB1YmxpY19wcmVkLCAnc29mdF9wcmVkJzogcHVibGljX3NvZnR9KQogICAgICAgICAgICBlbGlmIHB1YmxpY19wcmVkIGlzIG5vdCBOb25lOgogICAgICAgICAgICAgICAgcmFpc2UgV2VpZ2h0c0Vycm9yKGYie21bJ2lkJ119OiBpbmNvbXBsZXRlIHB1YmxpYy1mcm9udGllciB3aW5kb3dzIikKICAgICAgICAgICAgbG9nKGYiICBiYW5rZWQge21bJ2lkJ119IGZvbGQge20uZ2V0KCdmb2xkJywgJz8nKX0gKHtsZW4oc3RhcnRzKX0gd2luZG93KHMpOyB7bGVuKHBlcl9tZW1iZXIpfSBtZW1iZXIocykiKQogICAgZm9yIGdpLCAoa2V5LCBnbSkgaW4gZW51bWVyYXRlKGdyb3Vwcy5pdGVtcygpLCAxKToKICAgICAgICBjZmcgPSBqc29uLmxvYWRzKGtleSkKICAgICAgICBhZG9wdF9jb25maWdfZ2xvYmFscyhjZmcpCiAgICAgICAgbG9nKGYiZGVjb2RlIGdyb3VwIHtnaX0ve2xlbihncm91cHMpfToge2NmZ1snaW1nJ119cHggeCB7Y2ZnWydzbGljZXMnXX0gc2xpY2VzLCBjcm9wIHtjZmdbJ2Nyb3BfbW0nXX0gbW0gLT4ge2xlbihnbSl9IG1lbWJlcihzKSIpCiAgICAgICAgc3RfdGUsIEN0ZSwgTXRlID0gYnVpbGRfY2FjaGUocGlja19zbG90cyhodGUsIHBsYW5lX21hcCksIHBsYW5lX21hcCwgbGF0X29mKGh0ZSwgJ3Rlc3QgJyksIGYndGVzdCBne2dpfScpCiAgICAgICAgaWR4ID0gbnAuYXJhbmdlKGxlbihzdF90ZSkpCiAgICAgICAgc3RhcnRzX2Z1bGwgPSB3aW5kb3dfc3RhcnRzKEN0ZS5zaGFwZVsyXSwgR1JPVVApCiAgICAgICAgaWYgbGVuKGdyb3VwcykgPT0gMSBhbmQgX3NoYXJlZF9wcmVmaXhfZWxpZ2libGUoZ20pOgogICAgICAgICAgICBzaGFyZWRfcmVzdWx0cyA9IF9ydW5fc2hhcmVkX2Rpbm9fZ3JvdXAoCiAgICAgICAgICAgICAgICBwYXRoLAogICAgICAgICAgICAgICAgZ20sCiAgICAgICAgICAgICAgICBDdGUsCiAgICAgICAgICAgICAgICBNdGUsCiAgICAgICAgICAgICAgICBpZHgsCiAgICAgICAgICAgICAgICBzdGFydHNfZnVsbCwKICAgICAgICAgICAgKQogICAgICAgICAgICBmb3IgbWVtYmVyLCBwcmVkaWN0ZWQsIGppdHRlciBpbiBzaGFyZWRfcmVzdWx0czoKICAgICAgICAgICAgICAgIHByZWRpY3Rpb24sIHB1YmxpYywgc29mdCA9IHByZWRpY3RlZAogICAgICAgICAgICAgICAgYmFuaygKICAgICAgICAgICAgICAgICAgICBtZW1iZXIsCiAgICAgICAgICAgICAgICAgICAgc3RfdGUsCiAgICAgICAgICAgICAgICAgICAgcHJlZGljdGlvbiwKICAgICAgICAgICAgICAgICAgICBzdGFydHNfZnVsbCwKICAgICAgICAgICAgICAgICAgICBqaXR0ZXIsCiAgICAgICAgICAgICAgICAgICAgcHVibGljLAogICAgICAgICAgICAgICAgICAgIHNvZnQsCiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgIHJzbmFfcmVsZWFzZV9waXhlbHMoQ3RlKQogICAgICAgICAgICBkZWwgQ3RlLCBNdGUsIHNoYXJlZF9yZXN1bHRzCiAgICAgICAgICAgIGdjLmNvbGxlY3QoKQogICAgICAgICAgICBjb250aW51ZQogICAgICAgIHBlbmRpbmcgPSBzb3J0ZWQoZ20sIGtleT1sYW1iZGEgbTogLShtLmdldCgnaG9sZG91dCcpIG9yIDApKQogICAgICAgIGxlZnRfYWZ0ZXIgPSBzdW0oKGxlbihnKSBmb3IgaiwgKF8sIGcpIGluIGVudW1lcmF0ZShncm91cHMuaXRlbXMoKSwgMSkgaWYgaiA+IGdpKSkKCiAgICAgICAgZGVmIHBvcF9uZXh0KCk6CiAgICAgICAgICAgIHdpdGggU1RBVEVfTE9DSzoKICAgICAgICAgICAgICAgIGlmIG5vdCBwZW5kaW5nOgogICAgICAgICAgICAgICAgICAgIHJldHVybiAoTm9uZSwgTm9uZSwgRmFsc2UpCiAgICAgICAgICAgICAgICByc25hX2RlYWRsaW5lKCdESU5PIGNvbXBsZXRlLW1lbWJlciBpbmZlcmVuY2UnKQogICAgICAgICAgICAgICAgc3RhcnRzLCBqaXQgPSBzdGFydHNfZnVsbCwgRmFsc2UKICAgICAgICAgICAgICAgIHJldHVybiAocGVuZGluZy5wb3AoMCksIHN0YXJ0cywgaml0KQoKICAgICAgICBkZWYgd29ya2VyKGRldik6CiAgICAgICAgICAgIHdoaWxlIG5vdCBhYm9ydC5pc19zZXQoKToKICAgICAgICAgICAgICAgIG0sIHN0YXJ0cywgaml0ID0gcG9wX25leHQoKQogICAgICAgICAgICAgICAgaWYgbSBpcyBOb25lOgogICAgICAgICAgICAgICAgICAgIHJldHVybgogICAgICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgICAgIHAsIHB1YmxpY19wLCBwdWJsaWNfc29mdCwgKGZzLCB3cykgPSBfcnVuX21lbWJlcihwYXRoLCBtLCBkZXYsIEN0ZSwgTXRlLCBpZHgsIHN0YXJ0cywgaml0KQogICAgICAgICAgICAgICAgICAgIHdpdGggU1RBVEVfTE9DSzoKICAgICAgICAgICAgICAgICAgICAgICAgZXN0WydmaXhlZCddLCBlc3RbJ3dpbiddID0gKGZzLCB3cykKICAgICAgICAgICAgICAgICAgICBiYW5rKG0sIHN0X3RlLCBwLCBzdGFydHMsIGppdCwgcHVibGljX3AsIHB1YmxpY19zb2Z0KQogICAgICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBleGM6CiAgICAgICAgICAgICAgICAgICAgd2l0aCBTVEFURV9MT0NLOgogICAgICAgICAgICAgICAgICAgICAgICBmYWlsdXJlcy5hcHBlbmQoKG1bJ2lkJ10sIHN0cihkZXYpLCB0eXBlKGV4YykuX19uYW1lX18sIHN0cihleGMpKSkKICAgICAgICAgICAgICAgICAgICBhYm9ydC5zZXQoKQogICAgICAgICAgICAgICAgICAgIHJldHVybgogICAgICAgIHRocmVhZHMgPSBbdGhyZWFkaW5nLlRocmVhZCh0YXJnZXQ9d29ya2VyLCBhcmdzPShkLCkpIGZvciBkIGluIERFVlNdCiAgICAgICAgZm9yIHQgaW4gdGhyZWFkczoKICAgICAgICAgICAgdC5zdGFydCgpCiAgICAgICAgZm9yIHQgaW4gdGhyZWFkczoKICAgICAgICAgICAgdC5qb2luKCkKICAgICAgICBpZiBmYWlsdXJlczoKICAgICAgICAgICAgcmFpc2UgV2VpZ2h0c0Vycm9yKGYnRElOT3YyIG1lbWJlciBmYWlsdXJlOyBubyBmYWxsYmFjayBwZXJtaXR0ZWQ6IHtmYWlsdXJlc1swXX0nKQogICAgICAgIHJzbmFfcmVsZWFzZV9waXhlbHMoQ3RlKQogICAgICAgIGRlbCBDdGUsIE10ZQogICAgICAgIGdjLmNvbGxlY3QoKQogICAgaWYgX0RJTk9WMl9NQVRDSEVEX01FTUJFUlMgIT0gRVhQRUNURURfRElOT1YyX01FTUJFUlM6CiAgICAgICAgcmFpc2UgV2VpZ2h0c0Vycm9yKGYnZmluZ2VycHJpbnQgZ2F0ZSBmYWlsZWQ6IHtfRElOT1YyX01BVENIRURfTUVNQkVSU30gLyB7RVhQRUNURURfRElOT1YyX01FTUJFUlN9JykKICAgIGlmIGxlbihwdWJsaWNfZnJvbnRpZXJfbWVtYmVycykgIT0gRVhQRUNURURfRElOT1YyX01FTUJFUlM6CiAgICAgICAgcmFpc2UgV2VpZ2h0c0Vycm9yKGYncHVibGljLWZyb250aWVyIGluZmVyZW5jZSBpbmNvbXBsZXRlOiB7bGVuKHB1YmxpY19mcm9udGllcl9tZW1iZXJzKX0gLyB7RVhQRUNURURfRElOT1YyX01FTUJFUlN9IG1lbWJlcnMnKQogICAgbG9nKGYnRElOT3YyIGZhaWwtY2xvc2VkIGdhdGUgUEFTUzoge19ESU5PVjJfTUFUQ0hFRF9NRU1CRVJTfS97RVhQRUNURURfRElOT1YyX01FTUJFUlN9IGZpbmdlcnByaW50cyBhbmQgbWVtYmVycycpCiAgICBmcm9udGllcl9pZHMsIGZyb250aWVyX2FjYyA9IF9jb21iaW5lKHB1YmxpY19mcm9udGllcl9tZW1iZXJzKQogICAgc3ViID0gd3JpdGVfc3VibWlzc2lvbihmcm9udGllcl9hY2MsIGZyb250aWVyX2lkcywgdGVzdF9kZiwgJ19waXBlbGluZV9zdGFnZS5jc3YnKQogICAgbG9nKGYnc3VibWlzc2lvbi5jc3YgPSBleGFjdCBuby1qaXR0ZXIgcHVibGljLWZyb250aWVyIHJhbmsgbWVhbiBvZiB7bGVuKHB1YmxpY19mcm9udGllcl9tZW1iZXJzKX0gbWVtYmVyKHMpOyB7c3ViLnNoYXBlfScpCiAgICByZXR1cm4gc3ViCgpkZWYgYWRvcHRfY29uZmlnX2dsb2JhbHMoY2ZnKToKICAgIGdsb2JhbCBJTUcsIENBQ0hFX0lNRywgR1JPVVAsIENBQ0hFX1NMSUNFUywgTl9HUk9VUCwgQ1JPUF9NTSwgU0xJQ0VfQkFORCwgUlVMRVMKICAgIENBQ0hFX0lNRyA9IElNRyA9IGludChjZmdbJ2ltZyddKQogICAgR1JPVVAgPSBpbnQoY2ZnWydncm91cCddKQogICAgQ0FDSEVfU0xJQ0VTID0gaW50KGNmZ1snc2xpY2VzJ10pCiAgICBOX0dST1VQID0gbWF4KENBQ0hFX1NMSUNFUyAvLyBHUk9VUCwgMSkKICAgIENST1BfTU0gPSBmbG9hdChjZmdbJ2Nyb3BfbW0nXSkKICAgIFNMSUNFX0JBTkQgPSB0dXBsZSgoZmxvYXQoeCkgZm9yIHggaW4gY2ZnWydiYW5kJ10pKQogICAgcnVsZXMgPSBjZmcuZ2V0KCdydWxlcycpIG9yIFJVTEVTX05BVElWRQogICAgdW5rbm93biA9IHtrOiB2IGZvciBrLCB2IGluIHJ1bGVzLml0ZW1zKCkgaWYgayBub3QgaW4gUlVMRVNfTkFUSVZFIG9yIHYgbm90IGluIChSVUxFU19OQVRJVkVba10sIFJVTEVTX0xFR0FDWVtrXSl9CiAgICBpZiB1bmtub3duOgogICAgICAgIHJhaXNlIFdlaWdodHNFcnJvcihmJ3RoZSBtZW1iZXJzIHJlY29yZCBwaXhlbCBydWxlcyB0aGlzIHBpcGVsaW5lIGNhbm5vdCByZXByb2R1Y2U6IHt1bmtub3dufScpCiAgICBSVUxFUyA9IHsqKlJVTEVTX05BVElWRSwgKipydWxlc30KICAgIGlmIFtzWzBdIGZvciBzIGluIFNMT1RTXSAhPSBsaXN0KGNmZ1snc2xvdHMnXSk6CiAgICAgICAgcmFpc2UgV2VpZ2h0c0Vycm9yKGYidGhlIG1lbWJlcnMgd2VyZSBmaXR0ZWQgb24gc2xvdHMge2NmZ1snc2xvdHMnXX0gYW5kIHRoaXMgcGlwZWxpbmUgZGVmaW5lcyB7W3NbMF0gZm9yIHMgaW4gU0xPVFNdfTsgYSB3ZWlnaHQgd291bGQgYmUgcmVhZCBhZ2FpbnN0IHRoZSB3cm9uZyBzbG90IikK", "14": "ZGVmIHdyaXRlX3N1Ym1pc3Npb24ocHJlZCwgc3R1ZGllcywgdGVzdF9kZiwgcGF0aCk6CiAgICBzdWIgPSBwZC5EYXRhRnJhbWUocGQuRGF0YUZyYW1lKHByZWQpLnJhbmsocGN0PVRydWUpLnZhbHVlcywgY29sdW1ucz1UQVJHRVRTKQogICAgc3ViLmluc2VydCgwLCAnU3R1ZHlJbnN0YW5jZVVJRCcsIHN0dWRpZXMpCiAgICBzdWIgPSB0ZXN0X2RmW1snU3R1ZHlJbnN0YW5jZVVJRCddXS5tZXJnZShzdWIsIG9uPSdTdHVkeUluc3RhbmNlVUlEJywgaG93PSdsZWZ0JykKICAgIGlmIHN1YltUQVJHRVRTXS5pc25hKCkuYW55KCkuYW55KCk6CiAgICAgICAgcnNuYV9ldmVudCgnc3VibWlzc2lvbl9uZXV0cmFsX2ZpbGwnLCBzdHVkaWVzPWludChzdWJbVEFSR0VUU10uaXNuYSgpLmFueShheGlzPTEpLnN1bSgpKSkKICAgICAgICBzdWJbVEFSR0VUU10gPSBzdWJbVEFSR0VUU10uZmlsbG5hKDAuNSkKICAgIHN1Yi50b19jc3YocGF0aCwgaW5kZXg9RmFsc2UpCiAgICByZXR1cm4gc3ViCgpkZWYgX3YzN192YWxpZGF0ZV9zdWJtaXNzaW9uKHBhdGgsIHRlc3RfZGYsIHRhZyk6CiAgICBmcmFtZSA9IHBkLnJlYWRfY3N2KHBhdGgpCiAgICBleHBlY3RlZCA9IFsnU3R1ZHlJbnN0YW5jZVVJRCddICsgVEFSR0VUUwogICAgaWYgbGlzdChmcmFtZS5jb2x1bW5zKSAhPSBleHBlY3RlZDoKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYne3RhZ306IGNvbHVtbnMgZGlmZmVyIGZyb20gdGhlIGNvbXBldGl0aW9uIGNvbnRyYWN0JykKICAgIGlmIGxlbihmcmFtZSkgIT0gbGVuKHRlc3RfZGYpIG9yIG5vdCBmcmFtZVsnU3R1ZHlJbnN0YW5jZVVJRCddLmlzX3VuaXF1ZToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYne3RhZ306IHJvdyBjb3VudCBvciBTdHVkeUluc3RhbmNlVUlEIHVuaXF1ZW5lc3MgZmFpbGVkJykKICAgIGlmIHNldChmcmFtZVsnU3R1ZHlJbnN0YW5jZVVJRCddLmFzdHlwZShzdHIpKSAhPSBzZXQodGVzdF9kZlsnU3R1ZHlJbnN0YW5jZVVJRCddLmFzdHlwZShzdHIpKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYne3RhZ306IFN0dWR5SW5zdGFuY2VVSUQgc2V0IGRpZmZlcnMgZnJvbSB0ZXN0LmNzdicpCiAgICB2YWx1ZXMgPSBmcmFtZVtUQVJHRVRTXS50b19udW1weShucC5mbG9hdDY0KQogICAgaWYgbm90IG5wLmlzZmluaXRlKHZhbHVlcykuYWxsKCk6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmJ3t0YWd9OiBub24tZmluaXRlIHByZWRpY3Rpb24nKQogICAgcmV0dXJuIGZyYW1lCgpkZWYgbWFpbigpOgogICAgcGtnID0gZmluZF93ZWlnaHRzKCkKICAgIGlmIHBrZyBpcyBOb25lOgogICAgICAgIHJhaXNlIFdlaWdodHNFcnJvcigncmVxdWlyZWQgcHVibGljIGNoZWNrcG9pbnQgbWFuaWZlc3Qgbm90IGZvdW5kJykKICAgIGluZmVyX2Zyb21fcGFja2FnZShwa2csIERFVlNbMF0pCiAgICBfdjM3X3ZhbGlkYXRlX3N1Ym1pc3Npb24oJ19waXBlbGluZV9zdGFnZS5jc3YnLCBwZC5yZWFkX2NzdihST09UIC8gJ3Rlc3QuY3N2JyksICdwdWJsaWMgZnJvbnRpZXInKQo=", "15": "cnNuYV9waGFzZSgnZGlub3YyJywgJ1NUQVJUJykKbWFpbigpCmxvZygnZG9uZScpCgpfc3BlZWRfZGlub190ZW1wbGF0ZXMuY2xlYXIoKQpnYy5jb2xsZWN0KCkKCnJzbmFfcGhhc2UoJ2Rpbm92MicsICdDT01QTEVURScpCg==", "16": "aWYgZ2xvYmFscygpLmdldCgnX0RJTk9WMl9NQVRDSEVEX01FTUJFUlMnKSAhPSAyMDoKICAgIHJhaXNlIFJ1bnRpbWVFcnJvcignRElOT3YyIDIwLzIwIGZpbmdlcnByaW50IGdhdGUgZGlkIG5vdCBwYXNzJykKcnNuYV9waGFzZSgnYTUnLCAnU1RBUlQnKQpfQTVfU0FWRUQgPSBkaWN0KGdsb2JhbHMoKSkKaW1wb3J0IGdjLCBvcywgdGltZSwgd2FybmluZ3MKZnJvbSBjb25jdXJyZW50LmZ1dHVyZXMgaW1wb3J0IFRocmVhZFBvb2xFeGVjdXRvciBhcyBBNURlY29kZVBvb2wsIGFzX2NvbXBsZXRlZApmcm9tIHBhdGhsaWIgaW1wb3J0IFBhdGgKaW1wb3J0IGN2MgppbXBvcnQgbnVtcHkgYXMgbnAKaW1wb3J0IHBhbmRhcyBhcyBwZAppbXBvcnQgcHlkaWNvbQppbXBvcnQgdGltbQppbXBvcnQgdG9yY2gKaW1wb3J0IHRvcmNoLm5uIGFzIG5uCmltcG9ydCB0b3JjaC5ubi5mdW5jdGlvbmFsIGFzIEYKd2FybmluZ3MuZmlsdGVyd2FybmluZ3MoJ2lnbm9yZScpCmN2Mi5zZXROdW1UaHJlYWRzKDEpCkNST1BfTU0gPSAxMzAuMApTSVpFID0gMzM2ClNMSUNFX0JBTkQgPSAoMC4xMiwgMC44OCkKTl9TTElDRSA9IDE2CklOVEVOU0lUWSA9ICdzbGljZScKU0xPVFMgPSBbKCdTYWdpdHRhbCcsIDEpLCAoJ1NhZ2l0dGFsJywgMCksICgnQ29yb25hbCcsIDEpLCAoJ0Nvcm9uYWwnLCAwKSwgKCdBeGlhbCcsIDEpLCAoJ0F4aWFsJywgMCldCk5fU0xPVCA9IGxlbihTTE9UUykKTEFCRUxTID0gWydBQ0wnLCAnTUNMJywgJ01lZGlhbCBNZW5pc2N1cycsICdMYXRlcmFsIE1lbmlzY3VzJywgJ01lZGlhbCBPQScsICdMYXRlcmFsIE9BJywgJ1BGIE9BJywgJ0VmZnVzaW9uJywgJ1N5bm92aXRpcycsICJCYWtlcidzIiwgJ0NvbnR1c2lvbicsICdGcmFjdHVyZSddCgpkZWYgX29uZV9hNV9yb290KHRhZywgY2FuZGlkYXRlcywgbWFya2VyKToKICAgIGhpdHMgPSBbUGF0aChwYXRoKSBmb3IgcGF0aCBpbiBjYW5kaWRhdGVzIGlmIFBhdGgocGF0aCkuaXNfZGlyKCkgYW5kIChQYXRoKHBhdGgpIC8gbWFya2VyKS5leGlzdHMoKV0KICAgIGlmIGxlbihoaXRzKSAhPSAxOgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcihmJ3t0YWd9OiBleHBlY3RlZCBvbmUgZGlyZWN0IHJvb3QsIGZvdW5kIHtoaXRzfScpCiAgICByZXR1cm4gaGl0c1swXQoKQ09NUCA9IFBhdGgoUk9PVCkKQ0tQVCA9IF9vbmVfYTVfcm9vdCgnQTUgd2VpZ2h0cycsIFsKICAgICcva2FnZ2xlL2lucHV0L2RhdGFzZXRzL21hdHRpYWFuZ2VsaS9rbmVlLW1yaS1mb2xkLXdlaWdodHMnLAogICAgJy9rYWdnbGUvaW5wdXQva25lZS1tcmktZm9sZC13ZWlnaHRzJywKXSwgJ21fZjAucHQnKQphc3NlcnQgQ09NUCBpcyBub3QgTm9uZSwgJ2NvbXBldGl0aW9uIGRhdGEgbm90IGF0dGFjaGVkJwphc3NlcnQgQ0tQVCBpcyBub3QgTm9uZSwgJ2ZvbGQgd2VpZ2h0cyBub3QgYXR0YWNoZWQnCmFzc2VydCAoQ09NUCAvICdzYW1wbGVfc3VibWlzc2lvbi5jc3YnKS5leGlzdHMoKSwgZidubyBjb21wZXRpdGlvbiBkYXRhIGF0IHtDT01QfScKYXNzZXJ0IGxpc3QoQ0tQVC5nbG9iKCcqX2YqLnB0JykpLCBmJ25vIGNoZWNrcG9pbnRzIGF0IHtDS1BUfScKREVWID0gJ2N1ZGEnIGlmIHRvcmNoLmN1ZGEuaXNfYXZhaWxhYmxlKCkgZWxzZSAnY3B1Jwo=", "17": "U0VSSUVTX1JPT1QgPSBDT01QIC8gJ3Rlc3Rfc2VyaWVzJwppZiBub3QgU0VSSUVTX1JPT1QuaXNfZGlyKCk6CiAgICByYWlzZSBGaWxlTm90Rm91bmRFcnJvcignTWlzc2luZyB0ZXN0X3NlcmllczsgdHJhaW4tc2VyaWVzIGZhbGxiYWNrIGlzIGZvcmJpZGRlbicpCnByaW50KCdzZXJpZXMgcm9vdDonLCBTRVJJRVNfUk9PVCkKCmRlZiBvcmRlcmVkX2ZpbGVzKHNkaXIsIGNhcD02NCk6CiAgICAjIFByZXNlcnZlIHRoZSBwYXJlbnQncyBJbnN0YW5jZU51bWJlciBvcmRlcmluZzsgcmVtb3ZlIHByZS1zb3J0IHRydW5jYXRpb24uCiAgICAjIFRoZSB1bnVzZWQgY2FwIGFyZ3VtZW50IHJlbWFpbnMgb25seSBmb3IgY2FsbCBjb21wYXRpYmlsaXR5LgogICAga2V5ZWQ9W107IHVucmVhZGFibGU9W10KICAgIGZvciBmIGluIHNvcnRlZChzZGlyLmdsb2IoJyouZGNtJykpOgogICAgICAgIHRyeToKICAgICAgICAgICAgZHM9cHlkaWNvbS5kY21yZWFkKHN0cihmKSxzdG9wX2JlZm9yZV9waXhlbHM9VHJ1ZSkKICAgICAgICAgICAga2V5PWludChkcy5JbnN0YW5jZU51bWJlcikKICAgICAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGV4YzoKICAgICAgICAgICAgdW5yZWFkYWJsZS5hcHBlbmQoc3RyKGYpKTsgY29udGludWUKICAgICAgICBrZXllZC5hcHBlbmQoKGtleSxzdHIoZikpKQogICAgaWYgdW5yZWFkYWJsZToKICAgICAgICByc25hX2V2ZW50KCdhNV9vcmRlcl91bnJlYWRhYmxlX2Ryb3BwZWQnLHNlcmllcz1zdHIoc2RpciksdW5yZWFkYWJsZT1sZW4odW5yZWFkYWJsZSkscmVhZGFibGU9bGVuKGtleWVkKSxleGFtcGxlcz11bnJlYWRhYmxlWzozXSkKICAgIHJldHVybiBbZiBmb3IgXyxmIGluIHNvcnRlZChrZXllZCldCgpkZWYgc2VyaWVzX3NpZGUocGF0aCk6CiAgICB0cnk6CiAgICAgICAgcmV0dXJuIGZsb2F0KHB5ZGljb20uZGNtcmVhZChwYXRoLCBzdG9wX2JlZm9yZV9waXhlbHM9VHJ1ZSkuSW1hZ2VQb3NpdGlvblBhdGllbnRbMF0pCiAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgIHJldHVybiAwLjAKCmRlZiByZWFkX2Nyb3AocGF0aCk6CiAgICB0cnk6CiAgICAgICAgZHMgPSBweWRpY29tLmRjbXJlYWQocGF0aCkKICAgICAgICBhcnIgPSBkcy5waXhlbF9hcnJheS5hc3R5cGUobnAuZmxvYXQzMikKICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgcmV0dXJuIE5vbmUKICAgIHRyeToKICAgICAgICBwcyA9IGZsb2F0KGRzLlBpeGVsU3BhY2luZ1swXSkKICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgcHMgPSBDUk9QX01NIC8gbWF4KGFyci5zaGFwZSkKICAgIGhhbGYgPSBpbnQocm91bmQoQ1JPUF9NTSAvIHBzIC8gMikpCiAgICBjeSwgY3ggPSAoYXJyLnNoYXBlWzBdIC8vIDIsIGFyci5zaGFwZVsxXSAvLyAyKQogICAgeTAsIHkxID0gKG1heCgwLCBjeSAtIGhhbGYpLCBtaW4oYXJyLnNoYXBlWzBdLCBjeSArIGhhbGYpKQogICAgeDAsIHgxID0gKG1heCgwLCBjeCAtIGhhbGYpLCBtaW4oYXJyLnNoYXBlWzFdLCBjeCArIGhhbGYpKQogICAgY3JvcCA9IGFyclt5MDp5MSwgeDA6eDFdCiAgICByZXR1cm4gTm9uZSBpZiBjcm9wLnNpemUgPT0gMCBlbHNlIGNyb3AKCmRlZiB3aW5kb3coY3JvcCwgbG8sIGhpLCBmbGlwKToKICAgIGMgPSBucC5jbGlwKChjcm9wIC0gbG8pIC8gbWF4KGhpIC0gbG8sIDFlLTA2KSwgMCwgMSkKICAgIGltZyA9IGN2Mi5yZXNpemUoYywgKFNJWkUsIFNJWkUpLCBpbnRlcnBvbGF0aW9uPWN2Mi5JTlRFUl9BUkVBKQogICAgcmV0dXJuIGltZ1s6LCA6Oi0xXS5jb3B5KCkgaWYgZmxpcCBlbHNlIGltZwoKZGVmIHJlbmRlcihwYXRoLCBmbGlwKToKICAgIGNyb3AgPSByZWFkX2Nyb3AocGF0aCkKICAgIGlmIGNyb3AgaXMgTm9uZToKICAgICAgICByZXR1cm4gTm9uZQogICAgbG8sIGhpID0gbnAucGVyY2VudGlsZShjcm9wWzo6NCwgOjo0XSwgWzEsIDk5XSkKICAgIHJldHVybiB3aW5kb3coY3JvcCwgbG8sIGhpLCBmbGlwKQoKZGVmIGJ1aWxkX3N0dWR5KGFyZ3MpOgogICAgaWR4LHN0dWR5LHJlY3M9YXJncwogICAgb3V0PW5wLnplcm9zKChOX1NMT1QsTl9TTElDRSxTSVpFLFNJWkUpLG5wLnVpbnQ4KTttYXNrPW5wLnplcm9zKE5fU0xPVCxucC51aW50OCkKICAgIHJvd3M9cGQuRGF0YUZyYW1lKHJlY3MpCiAgICBpZiBub3QgbGVuKHJvd3MpOgogICAgICAgIHJzbmFfZXZlbnQoJ2E1X25vX21ldGFkYXRhJyxzdHVkeT1zdHIoc3R1ZHkpKTsgcmV0dXJuIGlkeCxvdXQsbWFzawogICAgZm9yIHNfaSwocGxhbmUsZnMpIGluIGVudW1lcmF0ZShTTE9UUyk6CiAgICAgICAgc3ViPXJvd3NbKHJvd3MuQW5hdG9taWNhbF9QbGFuZT09cGxhbmUpJihyb3dzLkZhdF9TdXBwcmVzc2lvbj09ZnMpXQogICAgICAgIGlmIHN1Yi5lbXB0eTogY29udGludWUgICMgQWN0dWFsIGFjcXVpc2l0aW9uIGFic2VuY2UsIG5vdCBhIGRlY29kaW5nIGZhaWx1cmUuCiAgICAgICAgc2VyaWVzPXN0cihzdWIuaWxvY1swXS5TZXJpZXNJbnN0YW5jZVVJRCkKICAgICAgICBmaWxlcz1vcmRlcmVkX2ZpbGVzKFNFUklFU19ST09UL3N0cihzdHVkeSkvc2VyaWVzKQogICAgICAgIGlmIG5vdCBmaWxlczoKICAgICAgICAgICAgcnNuYV9ldmVudCgnYTVfc2xvdF9ub19maWxlcycsc3R1ZHk9c3RyKHN0dWR5KSxzZXJpZXM9c2VyaWVzKTsgY29udGludWUKICAgICAgICBmbGlwPXBsYW5lIT0nU2FnaXR0YWwnIGFuZCBzZXJpZXNfc2lkZShmaWxlc1swXSk8MAogICAgICAgIGkwPWludChyb3VuZChTTElDRV9CQU5EWzBdKihsZW4oZmlsZXMpLTEpKSk7aTE9aW50KHJvdW5kKFNMSUNFX0JBTkRbMV0qKGxlbihmaWxlcyktMSkpKQogICAgICAgIGF2YWlsPWxpc3QocmFuZ2UoaTAsaTErMSkpCiAgICAgICAgaWYgbGVuKGF2YWlsKT49Tl9TTElDRToKICAgICAgICAgICAgcGlja3M9W2F2YWlsW2ludChyb3VuZCh0KSldIGZvciB0IGluIG5wLmxpbnNwYWNlKDAsbGVuKGF2YWlsKS0xLE5fU0xJQ0UpXTtvZmY9MAogICAgICAgIGVsc2U6cGlja3Msb2ZmPWF2YWlsLChOX1NMSUNFLWxlbihhdmFpbCkpLy8yCiAgICAgICAgcmVuZGVyZWQ9MDtlcnJvcnM9W10KICAgICAgICBpZiBJTlRFTlNJVFk9PSdzZXJpZXMnOgogICAgICAgICAgICBjcm9wcz1bcmVhZF9jcm9wKGZpbGVzW3BdKSBmb3IgcCBpbiBwaWNrc107Z290PVt4IGZvciB4IGluIGNyb3BzIGlmIHggaXMgbm90IE5vbmVdCiAgICAgICAgICAgIGlmIG5vdCBnb3Q6CiAgICAgICAgICAgICAgICByc25hX2V2ZW50KCdhNV9zbG90X2FsbF9jcm9wc19mYWlsZWQnLHN0dWR5PXN0cihzdHVkeSksc2VyaWVzPXNlcmllcyk7IGNvbnRpbnVlCiAgICAgICAgICAgIHNhbXA9bnAuY29uY2F0ZW5hdGUoW3hbOjo0LDo6NF0ucmF2ZWwoKSBmb3IgeCBpbiBnb3RdKTtsb18saGlfPW5wLnBlcmNlbnRpbGUoc2FtcCxbMSw5OV0pCiAgICAgICAgICAgIGZvciBjLHggaW4gZW51bWVyYXRlKGNyb3BzKToKICAgICAgICAgICAgICAgIGlmIHggaXMgTm9uZTp4PXJlYWRfY3JvcChmaWxlc1ttaW4obGVuKGZpbGVzKS0xLHBpY2tzW2NdKzEpXSkKICAgICAgICAgICAgICAgIGlmIHggaXMgTm9uZTplcnJvcnMuYXBwZW5kKGZpbGVzW3BpY2tzW2NdXSk7Y29udGludWUKICAgICAgICAgICAgICAgIGltZz13aW5kb3coeCxsb18saGlfLGZsaXApCiAgICAgICAgICAgICAgICByc25hX2Zpbml0ZShpbWcsJ0E1IHJlbmRlcmVkIHBpeGVscycpCiAgICAgICAgICAgICAgICBvdXRbc19pLG9mZitjXT0oaW1nKjI1NSkuYXN0eXBlKG5wLnVpbnQ4KTtyZW5kZXJlZCs9MQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIGZvciBjLHAgaW4gZW51bWVyYXRlKHBpY2tzKToKICAgICAgICAgICAgICAgIGltZz1yZW5kZXIoZmlsZXNbcF0sZmxpcCkKICAgICAgICAgICAgICAgIGlmIGltZyBpcyBOb25lOmltZz1yZW5kZXIoZmlsZXNbbWluKGxlbihmaWxlcyktMSxwKzEpXSxmbGlwKQogICAgICAgICAgICAgICAgaWYgaW1nIGlzIE5vbmU6ZXJyb3JzLmFwcGVuZChmaWxlc1twXSk7Y29udGludWUKICAgICAgICAgICAgICAgIHJzbmFfZmluaXRlKGltZywnQTUgcmVuZGVyZWQgcGl4ZWxzJykKICAgICAgICAgICAgICAgIG91dFtzX2ksb2ZmK2NdPShpbWcqMjU1KS5hc3R5cGUobnAudWludDgpO3JlbmRlcmVkKz0xCiAgICAgICAgaWYgZXJyb3JzIG9yIHJlbmRlcmVkIT1sZW4ocGlja3MpOgogICAgICAgICAgICByc25hX2V2ZW50KCdhNV9kZWNvZGVfZmFpbHVyZScsc3R1ZHk9c3RyKHN0dWR5KSxzZXJpZXM9c2VyaWVzLGZhaWxlZD1lcnJvcnMscmVuZGVyZWQ9aW50KHJlbmRlcmVkKSx3YW50ZWQ9aW50KGxlbihwaWNrcykpKQogICAgICAgICAgICBpZiBub3QgcmVuZGVyZWQ6IGNvbnRpbnVlCiAgICAgICAgbWFza1tzX2ldPXJlbmRlcmVkCiAgICBpZiBub3QgbWFzay5hbnkoKToKICAgICAgICByc25hX2V2ZW50KCdhNV9lbXB0eV9zdHVkeScsc3R1ZHk9c3RyKHN0dWR5KSkKICAgICAgICBwcmludChmJ1thNV0ge3N0dWR5fTogbm8gYWNxdWlyZWQgc2xvdHM7IGFsbC16ZXJvIG1hc2sga2VwdCAocGFyZW50IHBvbGljeSknLCBmbHVzaD1UcnVlKQogICAgcmV0dXJuIGlkeCxvdXQsbWFzawpzdWJfZGYgPSBwZC5yZWFkX2NzdihDT01QIC8gJ3NhbXBsZV9zdWJtaXNzaW9uLmNzdicsZHR5cGU9eydTdHVkeUluc3RhbmNlVUlEJzpzdHJ9KQpzZXJfY3N2ID0gcGQucmVhZF9jc3YoQ09NUCAvICd0ZXN0X3Nlcmllcy5jc3YnLGR0eXBlPXsnU3R1ZHlJbnN0YW5jZVVJRCc6c3RyLCdTZXJpZXNJbnN0YW5jZVVJRCc6c3RyfSkKaWYgc2V0KHN1Yl9kZi5TdHVkeUluc3RhbmNlVUlEKSE9c2V0KF9SU05BX1RFU1RfSURTKToKICAgIHJhaXNlIFJ1bnRpbWVFcnJvcignQTUgdGVzdC9zYW1wbGUgVUlEIG1pc21hdGNoJykKc2VyX2NzdiA9IHNlcl9jc3YubG9jWzosIH5zZXJfY3N2LmNvbHVtbnMuZHVwbGljYXRlZCgpXQpzdHVkaWVzID0gc3ViX2RmLlN0dWR5SW5zdGFuY2VVSUQudG9saXN0KCkKYnkgPSB7czogZy50b19kaWN0KCdyZWNvcmRzJykgZm9yIHMsIGcgaW4gc2VyX2NzdltzZXJfY3N2LlN0dWR5SW5zdGFuY2VVSUQuaXNpbihzZXQoc3R1ZGllcykpXS5ncm91cGJ5KCdTdHVkeUluc3RhbmNlVUlEJyl9CnByaW50KGYne2xlbihzdHVkaWVzKTosfSB0ZXN0IHN0dWRpZXMsIHtsZW4oYnkpOix9IHdpdGggc2VyaWVzIG1ldGFkYXRhJykK", "18": "Tl9TTE9UX1RZUEVTLCBNQVNLX0lEWCA9ICg2LCAwKQoKZGVmIHNlZ21lbnRfc29mdG1heChzY29yZXMsIHNpZHgsIEIpOgogICAgVCwgSyA9IHNjb3Jlcy5zaGFwZQogICAgaWR4ID0gc2lkeC51bnNxdWVlemUoMSkuZXhwYW5kKC0xLCBLKQogICAgbSA9IHRvcmNoLmZ1bGwoKEIsIEspLCBmbG9hdCgnLWluZicpLCBkZXZpY2U9c2NvcmVzLmRldmljZSwgZHR5cGU9c2NvcmVzLmR0eXBlKQogICAgbSA9IG0uc2NhdHRlcl9yZWR1Y2UoMCwgaWR4LCBzY29yZXMsIHJlZHVjZT0nYW1heCcsIGluY2x1ZGVfc2VsZj1UcnVlKQogICAgZSA9IChzY29yZXMgLSBtW3NpZHhdKS5leHAoKQogICAgcyA9IHRvcmNoLnplcm9zKEIsIEssIGRldmljZT1zY29yZXMuZGV2aWNlLCBkdHlwZT1zY29yZXMuZHR5cGUpLmluZGV4X2FkZF8oMCwgc2lkeCwgZSkKICAgIHJldHVybiBlIC8gc1tzaWR4XS5jbGFtcChtaW49MWUtMDYpCgpjbGFzcyBNZWFuTWF4UG9vbChubi5Nb2R1bGUpOgoKICAgIGRlZiBmb3J3YXJkKHNlbGYsIGYsIHNpZHgsIEIsIHNsb3Q9Tm9uZSwgcmV0dXJuX2F0dG49RmFsc2UpOgogICAgICAgIEQgPSBmLnNoYXBlWzFdCiAgICAgICAgY250ID0gdG9yY2guemVyb3MoQiwgZGV2aWNlPWYuZGV2aWNlLCBkdHlwZT1mLmR0eXBlKS5pbmRleF9hZGRfKDAsIHNpZHgsIHRvcmNoLm9uZXMoZi5zaGFwZVswXSwgZGV2aWNlPWYuZGV2aWNlLCBkdHlwZT1mLmR0eXBlKSkKICAgICAgICBtZWFuID0gdG9yY2guemVyb3MoQiwgRCwgZGV2aWNlPWYuZGV2aWNlLCBkdHlwZT1mLmR0eXBlKS5pbmRleF9hZGRfKDAsIHNpZHgsIGYpCiAgICAgICAgbWVhbiA9IG1lYW4gLyBjbnQuY2xhbXAobWluPTEpLnVuc3F1ZWV6ZSgxKQogICAgICAgIG14ID0gdG9yY2guZnVsbCgoQiwgRCksIC0xMDAwMC4wLCBkZXZpY2U9Zi5kZXZpY2UsIGR0eXBlPWYuZHR5cGUpCiAgICAgICAgbXggPSBteC5zY2F0dGVyX3JlZHVjZSgwLCBzaWR4LnVuc3F1ZWV6ZSgxKS5leHBhbmQoLTEsIEQpLCBmLCByZWR1Y2U9J2FtYXgnLCBpbmNsdWRlX3NlbGY9VHJ1ZSkKICAgICAgICByZXR1cm4gKHRvcmNoLmNhdChbbWVhbiwgbXhdLCAxKSwgTm9uZSkKCmNsYXNzIExhYmVsQXR0ZW50aW9uUG9vbChubi5Nb2R1bGUpOgoKICAgIGRlZiBfX2luaXRfXyhzZWxmLCBkLCBuX2xhYmVscz0xMiwgbl9oZWFkcz00LCBzbG90X2JpYXM9VHJ1ZSk6CiAgICAgICAgc3VwZXIoKS5fX2luaXRfXygpCiAgICAgICAgc2VsZi5kLCBzZWxmLmssIHNlbGYuaCA9IChkLCBuX2xhYmVscywgbl9oZWFkcykKICAgICAgICBzZWxmLnEgPSBubi5QYXJhbWV0ZXIodG9yY2gucmFuZG4obl9sYWJlbHMsIGQpICogMC4wMikKICAgICAgICBzZWxmLmtleSwgc2VsZi52YWwgPSAobm4uTGluZWFyKGQsIGQpLCBubi5MaW5lYXIoZCwgZCkpCiAgICAgICAgc2VsZi5zbG90X2JpYXMgPSBubi5QYXJhbWV0ZXIodG9yY2guemVyb3Mobl9sYWJlbHMsIE5fU0xPVF9UWVBFUyArIDEpKSBpZiBzbG90X2JpYXMgZWxzZSBOb25lCgogICAgZGVmIGZvcndhcmQoc2VsZiwgZiwgc2lkeCwgQiwgc2xvdD1Ob25lLCByZXR1cm5fYXR0bj1GYWxzZSk6CiAgICAgICAgc2NvcmVzID0gc2VsZi5rZXkoZikgQCBzZWxmLnEudCgpIC8gc2VsZi5kICoqIDAuNQogICAgICAgIGlmIHNlbGYuc2xvdF9iaWFzIGlzIG5vdCBOb25lIGFuZCBzbG90IGlzIG5vdCBOb25lOgogICAgICAgICAgICBzY29yZXMgPSBzY29yZXMgKyBzZWxmLnNsb3RfYmlhcy50KClbc2xvdF0KICAgICAgICBhID0gc2VnbWVudF9zb2Z0bWF4KHNjb3Jlcywgc2lkeCwgQikKICAgICAgICBvdXQgPSB0b3JjaC56ZXJvcyhCLCBzZWxmLmssIHNlbGYuZCwgZGV2aWNlPWYuZGV2aWNlLCBkdHlwZT1mLmR0eXBlKQogICAgICAgIG91dCA9IG91dC5pbmRleF9hZGRfKDAsIHNpZHgsIGEudW5zcXVlZXplKC0xKSAqIHNlbGYudmFsKGYpLnVuc3F1ZWV6ZSgxKSkKICAgICAgICByZXR1cm4gKG91dCwgYSkKCmNsYXNzIFRva2VuWEF0dG5Qb29sKG5uLk1vZHVsZSk6CgogICAgZGVmIF9faW5pdF9fKHNlbGYsIGQsIG5fbGFiZWxzPTEyLCBuX2hlYWRzPTYsIGRyb3BvdXQ9MC4yKToKICAgICAgICBzdXBlcigpLl9faW5pdF9fKCkKICAgICAgICBzZWxmLmQsIHNlbGYuayA9IChkLCBuX2xhYmVscykKICAgICAgICBzZWxmLnEgPSBubi5QYXJhbWV0ZXIodG9yY2gucmFuZG4obl9sYWJlbHMsIGQpICogMC4wMikKICAgICAgICBzZWxmLnNsb3RfZW1iID0gbm4uRW1iZWRkaW5nKE5fU0xPVF9UWVBFUyArIDEsIGQsIHBhZGRpbmdfaWR4PTApCiAgICAgICAgc2VsZi5rdl9ub3JtID0gbm4uTGF5ZXJOb3JtKGQpCiAgICAgICAgc2VsZi5hdHRuID0gbm4uTXVsdGloZWFkQXR0ZW50aW9uKGQsIG5faGVhZHMsIGRyb3BvdXQ9ZHJvcG91dCwgYmF0Y2hfZmlyc3Q9VHJ1ZSkKCiAgICBkZWYgZm9yd2FyZChzZWxmLCB0b2ssIHNpZHgsIEIsIHNsb3Q9Tm9uZSwgcmV0dXJuX2F0dG49RmFsc2UpOgogICAgICAgIFQsIE4sIEQgPSB0b2suc2hhcGUKICAgICAgICBjbnQgPSB0b3JjaC5iaW5jb3VudChzaWR4LCBtaW5sZW5ndGg9QikKICAgICAgICBTID0gaW50KGNudC5tYXgoKS5pdGVtKCkpCiAgICAgICAgc3RhcnRzID0gdG9yY2guY3Vtc3VtKGNudCwgMCkgLSBjbnQKICAgICAgICBwb3MgPSB0b3JjaC5hcmFuZ2UoVCwgZGV2aWNlPXRvay5kZXZpY2UpIC0gc3RhcnRzW3NpZHhdCiAgICAgICAga3YgPSB0b2sgKyBzZWxmLnNsb3RfZW1iKHNsb3QpLnVuc3F1ZWV6ZSgxKQogICAgICAgIHBhZCA9IHRvay5uZXdfemVyb3MoQiwgUywgTiwgRCkKICAgICAgICBwYWRbc2lkeCwgcG9zXSA9IGt2CiAgICAgICAga2VlcCA9IHRvcmNoLnplcm9zKEIsIFMsIGR0eXBlPXRvcmNoLmJvb2wsIGRldmljZT10b2suZGV2aWNlKQogICAgICAgIGtlZXBbc2lkeCwgcG9zXSA9IFRydWUKICAgICAgICBrcG0gPSB+a2VlcC5yZXBlYXRfaW50ZXJsZWF2ZShOLCBkaW09MSkKICAgICAgICBwYWQgPSBzZWxmLmt2X25vcm0ocGFkLnJlc2hhcGUoQiwgUyAqIE4sIEQpKQogICAgICAgIHEgPSBzZWxmLnEudW5zcXVlZXplKDApLmV4cGFuZChCLCAtMSwgLTEpCiAgICAgICAgYXR0LCB3ID0gc2VsZi5hdHRuKHEsIHBhZCwgcGFkLCBrZXlfcGFkZGluZ19tYXNrPWtwbSwgbmVlZF93ZWlnaHRzPXJldHVybl9hdHRuLCBhdmVyYWdlX2F0dG5fd2VpZ2h0cz1UcnVlKQogICAgICAgIGNscyA9IHRva1s6LCAwXQogICAgICAgIG1lYW4gPSB0b3JjaC56ZXJvcyhCLCBELCBkZXZpY2U9dG9rLmRldmljZSwgZHR5cGU9dG9rLmR0eXBlKS5pbmRleF9hZGRfKDAsIHNpZHgsIGNscykgLyBjbnQuY2xhbXAobWluPTEpLnVuc3F1ZWV6ZSgxKQogICAgICAgIG14ID0gdG9yY2guZnVsbCgoQiwgRCksIC0xMDAwMC4wLCBkZXZpY2U9dG9rLmRldmljZSwgZHR5cGU9dG9rLmR0eXBlKQogICAgICAgIG14ID0gbXguc2NhdHRlcl9yZWR1Y2UoMCwgc2lkeC51bnNxdWVlemUoMSkuZXhwYW5kKC0xLCBEKSwgY2xzLCByZWR1Y2U9J2FtYXgnLCBpbmNsdWRlX3NlbGY9VHJ1ZSkKICAgICAgICBiYXNlID0gdG9yY2guY2F0KFttZWFuLCBteF0sIDEpLnVuc3F1ZWV6ZSgxKS5leHBhbmQoLTEsIHNlbGYuaywgLTEpCiAgICAgICAgcmV0dXJuICh0b3JjaC5jYXQoW2F0dCwgYmFzZV0sIC0xKSwgdykKCmNsYXNzIFZpVFNsb3RUb2tlbihubi5Nb2R1bGUpOgoKICAgIGRlZiBfX2luaXRfXyhzZWxmLCB2aXQsIG5fY2F0LCBkaW09Tm9uZSk6CiAgICAgICAgc3VwZXIoKS5fX2luaXRfXygpCiAgICAgICAgc2VsZi52aXQgPSB2aXQKICAgICAgICBkID0gZGltIG9yIHZpdC5lbWJlZF9kaW0KICAgICAgICBzZWxmLnRvayA9IG5uLkVtYmVkZGluZyhuX2NhdCArIDEsIGQsIHBhZGRpbmdfaWR4PU1BU0tfSURYKQogICAgICAgIHNlbGYubnVtX2ZlYXR1cmVzID0gdml0Lm51bV9mZWF0dXJlcwogICAgICAgIHNlbGYuX29yaWdfcHJlZml4ID0gZ2V0YXR0cih2aXQsICdudW1fcHJlZml4X3Rva2VucycsIDEpCiAgICAgICAgdml0Lm51bV9wcmVmaXhfdG9rZW5zID0gc2VsZi5fb3JpZ19wcmVmaXggKyAxCiAgICAgICAgZm9yIGJsayBpbiB2aXQuYmxvY2tzOgogICAgICAgICAgICBhID0gZ2V0YXR0cihibGssICdhdHRuJywgTm9uZSkKICAgICAgICAgICAgaWYgYSBpcyBub3QgTm9uZSBhbmQgaGFzYXR0cihhLCAnbnVtX3ByZWZpeF90b2tlbnMnKToKICAgICAgICAgICAgICAgIGEubnVtX3ByZWZpeF90b2tlbnMgPSBhLm51bV9wcmVmaXhfdG9rZW5zICsgMQoKICAgIEBzdGF0aWNtZXRob2QKICAgIGRlZiBfbWF5YmUobW9kLCB4KToKICAgICAgICByZXR1cm4geCBpZiBtb2QgaXMgTm9uZSBlbHNlIG1vZCh4KQoKICAgIGRlZiBmb3J3YXJkX2ZlYXR1cmVzKHNlbGYsIHgsIGNhdCk6CiAgICAgICAgdiA9IHNlbGYudml0CiAgICAgICAgeCA9IHYucGF0Y2hfZW1iZWQoeCkKICAgICAgICBwb3MgPSB2Ll9wb3NfZW1iZWQoeCkKICAgICAgICByb3BlID0gTm9uZQogICAgICAgIGlmIGlzaW5zdGFuY2UocG9zLCB0dXBsZSk6CiAgICAgICAgICAgIHgsIHJvcGUgPSBwb3MKICAgICAgICBlbHNlOgogICAgICAgICAgICB4ID0gcG9zCiAgICAgICAgeCA9IHNlbGYuX21heWJlKGdldGF0dHIodiwgJ3BhdGNoX2Ryb3AnLCBOb25lKSwgeCkKICAgICAgICB4ID0gc2VsZi5fbWF5YmUoZ2V0YXR0cih2LCAnbm9ybV9wcmUnLCBOb25lKSwgeCkKICAgICAgICBucHQgPSBzZWxmLl9vcmlnX3ByZWZpeAogICAgICAgIHRvayA9IHNlbGYudG9rKGNhdCkudW5zcXVlZXplKDEpCiAgICAgICAgeCA9IHRvcmNoLmNhdChbeFs6LCA6bnB0XSwgdG9rLCB4WzosIG5wdDpdXSwgZGltPTEpCiAgICAgICAgaWYgcm9wZSBpcyBub3QgTm9uZToKICAgICAgICAgICAgaWYgZ2V0YXR0cih2LCAncm9wZV9taXhlZCcsIEZhbHNlKToKICAgICAgICAgICAgICAgIGZvciBpLCBibGsgaW4gZW51bWVyYXRlKHYuYmxvY2tzKToKICAgICAgICAgICAgICAgICAgICB4ID0gYmxrKHgsIHJvcGU9cm9wZVtpXSkKICAgICAgICAgICAgZWxzZToKICAgICAgICAgICAgICAgIGZvciBibGsgaW4gdi5ibG9ja3M6CiAgICAgICAgICAgICAgICAgICAgeCA9IGJsayh4LCByb3BlPXJvcGUpCiAgICAgICAgZWxzZToKICAgICAgICAgICAgeCA9IHYuYmxvY2tzKHgpCiAgICAgICAgcmV0dXJuIHYubm9ybSh4KQoKICAgIGRlZiBmb3J3YXJkX2hlYWQoc2VsZiwgeCwgcHJlX2xvZ2l0cz1UcnVlKToKICAgICAgICByZXR1cm4gc2VsZi52aXQuZm9yd2FyZF9oZWFkKHgsIHByZV9sb2dpdHM9cHJlX2xvZ2l0cykKSU1BR0VORVRfTUVBTiA9ICgwLjQ4NSwgMC40NTYsIDAuNDA2KQpJTUFHRU5FVF9TVEQgPSAoMC4yMjksIDAuMjI0LCAwLjIyNSkKCmNsYXNzIF9HYXRlZERlcHRoQmxvY2sobm4uTW9kdWxlKToKCiAgICBkZWYgX19pbml0X18oc2VsZiwgbl9zbGljZSwgZHJvcG91dD0wLjAsIGxzX2luaXQ9MC4xKToKICAgICAgICBzdXBlcigpLl9faW5pdF9fKCkKICAgICAgICBzZWxmLm5vcm0gPSBubi5Hcm91cE5vcm0oMSwgbl9zbGljZSkKICAgICAgICBzZWxmLnYgPSBubi5Db252MmQobl9zbGljZSwgbl9zbGljZSwgMSkKICAgICAgICBzZWxmLmcgPSBubi5Db252MmQobl9zbGljZSwgbl9zbGljZSwgMSkKICAgICAgICBzZWxmLm91dCA9IG5uLkNvbnYyZChuX3NsaWNlLCBuX3NsaWNlLCAxKQogICAgICAgIHNlbGYuZ2FtbWEgPSBubi5QYXJhbWV0ZXIodG9yY2guZnVsbCgobl9zbGljZSwgMSwgMSksIGxzX2luaXQpKQogICAgICAgIHNlbGYuZHJvcCA9IG5uLkRyb3BvdXQyZChkcm9wb3V0KSBpZiBkcm9wb3V0IGVsc2Ugbm4uSWRlbnRpdHkoKQoKICAgIGRlZiBmb3J3YXJkKHNlbGYsIHgpOgogICAgICAgIHogPSBzZWxmLm5vcm0oeCkKICAgICAgICByZXR1cm4geCArIHNlbGYuZ2FtbWEgKiBzZWxmLmRyb3Aoc2VsZi5vdXQoc2VsZi52KHopICogRi5zaWx1KHNlbGYuZyh6KSkpKQoKY2xhc3MgRGVwdGhDb21wcmVzcyhubi5Nb2R1bGUpOgoKICAgIGRlZiBfX2luaXRfXyhzZWxmLCBuX3NsaWNlPTE2LCBvdXRfY2g9MywgZGVwdGg9MSwgZHJvcG91dD0wLjAsIGxzX2luaXQ9MC4xLCBpbWFnZW5ldD1UcnVlLCBwcm9qX25vaXNlPTAuMjUpOgogICAgICAgIHN1cGVyKCkuX19pbml0X18oKQogICAgICAgIHNlbGYuaW1hZ2VuZXQgPSBpbWFnZW5ldAogICAgICAgIHNlbGYuYmxvY2tzID0gbm4uTW9kdWxlTGlzdChbX0dhdGVkRGVwdGhCbG9jayhuX3NsaWNlLCBkcm9wb3V0LCBsc19pbml0KSBmb3IgXyBpbiByYW5nZShkZXB0aCldKQogICAgICAgIHNlbGYucHJvaiA9IG5uLkNvbnYyZChuX3NsaWNlLCBvdXRfY2gsIDEsIGJpYXM9VHJ1ZSkKICAgICAgICBpZiBpbWFnZW5ldDoKICAgICAgICAgICAgc2VsZi5yZWdpc3Rlcl9idWZmZXIoJ211JywgdG9yY2gudGVuc29yKElNQUdFTkVUX01FQU4pLnZpZXcoMSwgLTEsIDEsIDEpKQogICAgICAgICAgICBzZWxmLnJlZ2lzdGVyX2J1ZmZlcignc2QnLCB0b3JjaC50ZW5zb3IoSU1BR0VORVRfU1REKS52aWV3KDEsIC0xLCAxLCAxKSkKCiAgICBkZWYgZm9yd2FyZChzZWxmLCB4KToKICAgICAgICBrZWVwID0gKHguYW1heChkaW09MSwga2VlcGRpbT1UcnVlKSA+IDApLnRvKHguZHR5cGUpCiAgICAgICAgeiA9IHgKICAgICAgICBmb3IgYiBpbiBzZWxmLmJsb2NrczoKICAgICAgICAgICAgeiA9IGIoeikKICAgICAgICB6ID0gc2VsZi5wcm9qKHopCiAgICAgICAgaWYgc2VsZi5pbWFnZW5ldDoKICAgICAgICAgICAgeiA9ICh6IC0gc2VsZi5tdS50byh6LmR0eXBlKSkgLyBzZWxmLnNkLnRvKHouZHR5cGUpCiAgICAgICAgcmV0dXJuIHogKiBrZWVwCk5fUExBTkUsIE5fQ09OVFJBU1QgPSAoMywgMikKX1BMQU5FX09GID0gbGFtYmRhIHM6IHRvcmNoLmNsYW1wKHMgLSAxLCAwLCA1KSAvLyAyCl9DT05UUkFTVF9PRiA9IGxhbWJkYSBzOiB0b3JjaC5jbGFtcChzIC0gMSwgMCwgNSkgJSAyCgpjbGFzcyBTbG90RGVwdGhNaXhlcihubi5Nb2R1bGUpOgoKICAgIGRlZiBfX2luaXRfXyhzZWxmLCBuX3NsaWNlPTE2LCBrc2l6ZT01LCBhbHBoYV9tYXg9MC4yNSk6CiAgICAgICAgc3VwZXIoKS5fX2luaXRfXygpCiAgICAgICAgc2VsZi5uX3NsaWNlLCBzZWxmLmtzaXplLCBzZWxmLnIgPSAobl9zbGljZSwga3NpemUsIGtzaXplIC8vIDIpCiAgICAgICAgc2VsZi5hbHBoYV9tYXggPSBhbHBoYV9tYXgKICAgICAgICBiID0gdG9yY2gudGVuc29yKFsxLjAsIDQuMCwgNi4wLCA0LjAsIDEuMF0pCiAgICAgICAgc2VsZi5yZWdpc3Rlcl9idWZmZXIoJ2Jhc2UnLCBiLmxvZygpW3NlbGYucjpdKQogICAgICAgIG5fdSA9IHNlbGYuciArIDEKICAgICAgICBzZWxmLnNoYXJlZCA9IG5uLlBhcmFtZXRlcih0b3JjaC56ZXJvcyhuX3UpKQogICAgICAgIHNlbGYucGxhbmVfayA9IG5uLlBhcmFtZXRlcih0b3JjaC56ZXJvcyhOX1BMQU5FLCBuX3UpKQogICAgICAgIHNlbGYuY29udHJhc3RfayA9IG5uLlBhcmFtZXRlcih0b3JjaC56ZXJvcyhOX0NPTlRSQVNULCBuX3UpKQogICAgICAgIHNlbGYuZzAgPSBubi5QYXJhbWV0ZXIodG9yY2guemVyb3MoKCkpKQogICAgICAgIHNlbGYuZ2F0ZV9wID0gbm4uUGFyYW1ldGVyKHRvcmNoLnplcm9zKE5fUExBTkUpKQogICAgICAgIHNlbGYuZ2F0ZV9jID0gbm4uUGFyYW1ldGVyKHRvcmNoLnplcm9zKE5fQ09OVFJBU1QpKQogICAgICAgIGlkeCA9IHRvcmNoLmFyYW5nZShuX3NsaWNlKQogICAgICAgIHNlbGYucmVnaXN0ZXJfYnVmZmVyKCdvZmYnLCBpZHhbTm9uZSwgOl0gLSBpZHhbOiwgTm9uZV0pCgogICAgZGVmIGtlcm5lbChzZWxmLCBzbG90KToKICAgICAgICBwLCBjID0gKF9QTEFORV9PRihzbG90KSwgX0NPTlRSQVNUX09GKHNsb3QpKQogICAgICAgIGhhbGYgPSBzZWxmLmJhc2UgKyBzZWxmLnNoYXJlZCArIHNlbGYucGxhbmVfa1twXSArIHNlbGYuY29udHJhc3Rfa1tjXQogICAgICAgIGZ1bGwgPSB0b3JjaC5jYXQoW2hhbGYuZmxpcCgtMSlbLi4uLCA6c2VsZi5yXSwgaGFsZl0sIGRpbT0tMSkKICAgICAgICByZXR1cm4gRi5zb2Z0bWF4KGZ1bGwsIGRpbT0tMSkKCiAgICBkZWYgYWxwaGEoc2VsZiwgc2xvdCk6CiAgICAgICAgcCwgYyA9IChfUExBTkVfT0Yoc2xvdCksIF9DT05UUkFTVF9PRihzbG90KSkKICAgICAgICByZXR1cm4gc2VsZi5hbHBoYV9tYXggKiB0b3JjaC50YW5oKHNlbGYuZzAgKyBzZWxmLmdhdGVfcFtwXSArIHNlbGYuZ2F0ZV9jW2NdKQoKICAgIGRlZiBmb3J3YXJkKHNlbGYsIHgsIHNsb3QsIHZtYXNrKToKICAgICAgICBULCBTLCBILCBXID0geC5zaGFwZQogICAgICAgIGlmIHZtYXNrIGlzIE5vbmU6CiAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoJ3N0ZW09bWl4ZXIgcmVxdWlyZXMgdGhlIHBhZGRpbmcgbWFzaycpCiAgICAgICAgayA9IHNlbGYua2VybmVsKHNsb3QpCiAgICAgICAgdiA9IHZtYXNrLnRvKGsuZHR5cGUpCiAgICAgICAgZCA9IHNlbGYub2ZmICsgc2VsZi5yCiAgICAgICAgaW5iID0gKGQgPj0gMCkgJiAoZCA8IHNlbGYua3NpemUpCiAgICAgICAga2sgPSBrWzosIGQuY2xhbXAoMCwgc2VsZi5rc2l6ZSAtIDEpXSAqIGluYgogICAgICAgIE0gPSBrayAqIHZbOiwgTm9uZSwgOl0KICAgICAgICBkZW4gPSBNLnN1bSgtMSwga2VlcGRpbT1UcnVlKQogICAgICAgIGV5ZSA9IHRvcmNoLmV5ZShTLCBkZXZpY2U9eC5kZXZpY2UsIGR0eXBlPU0uZHR5cGUpLmV4cGFuZChULCBTLCBTKQogICAgICAgIG9rID0gKGRlbiA+IDFlLTA2KSAmIHZbOiwgOiwgTm9uZV0uYm9vbCgpCiAgICAgICAgTSA9IHRvcmNoLndoZXJlKG9rLCBNIC8gZGVuLmNsYW1wKG1pbj0xZS0wNiksIGV5ZSkKICAgICAgICBhID0gc2VsZi5hbHBoYShzbG90KVs6LCBOb25lLCBOb25lXQogICAgICAgIEFvcCA9ICgoMS4wIC0gYSkgKiBleWUgKyBhICogTSkudG8oeC5kdHlwZSkKICAgICAgICBpZiB4LmlzX2NvbnRpZ3VvdXMobWVtb3J5X2Zvcm1hdD10b3JjaC5jaGFubmVsc19sYXN0KSBhbmQgKG5vdCB4LmlzX2NvbnRpZ3VvdXMoKSk6CiAgICAgICAgICAgIHkgPSB0b3JjaC5ibW0oeC5wZXJtdXRlKDAsIDIsIDMsIDEpLnJlc2hhcGUoVCwgSCAqIFcsIFMpLCBBb3AudHJhbnNwb3NlKDEsIDIpKQogICAgICAgICAgICByZXR1cm4geS5yZXNoYXBlKFQsIEgsIFcsIFMpLnBlcm11dGUoMCwgMywgMSwgMikKICAgICAgICByZXR1cm4gdG9yY2guYm1tKEFvcCwgeC5yZXNoYXBlKFQsIFMsIEggKiBXKSkucmVzaGFwZShULCBTLCBILCBXKQoKZGVmIF9zZWdfbWVhbl9tYXgodiwgc2lkeCwgQik6CiAgICBEID0gdi5zaGFwZVsxXQogICAgY250ID0gdG9yY2guemVyb3MoQiwgZGV2aWNlPXYuZGV2aWNlLCBkdHlwZT12LmR0eXBlKS5pbmRleF9hZGRfKDAsIHNpZHgsIHRvcmNoLm9uZXModi5zaGFwZVswXSwgZGV2aWNlPXYuZGV2aWNlLCBkdHlwZT12LmR0eXBlKSkKICAgIG1lYW4gPSB0b3JjaC56ZXJvcyhCLCBELCBkZXZpY2U9di5kZXZpY2UsIGR0eXBlPXYuZHR5cGUpLmluZGV4X2FkZF8oMCwgc2lkeCwgdikKICAgIG1lYW4gPSBtZWFuIC8gY250LmNsYW1wKG1pbj0xKS51bnNxdWVlemUoMSkKICAgIG14ID0gdG9yY2guZnVsbCgoQiwgRCksIC0xMDAwMC4wLCBkZXZpY2U9di5kZXZpY2UsIGR0eXBlPXYuZHR5cGUpCiAgICBteCA9IG14LnNjYXR0ZXJfcmVkdWNlKDAsIHNpZHgudW5zcXVlZXplKDEpLmV4cGFuZCgtMSwgRCksIHYsIHJlZHVjZT0nYW1heCcsIGluY2x1ZGVfc2VsZj1UcnVlKQogICAgcmV0dXJuIHRvcmNoLmNhdChbbWVhbiwgbXhdLCAxKQoKZGVmIF9wYWRfa3YoeCwgc2lkeCwgQiwgbm9ybSk6CiAgICBULCBQLCBEID0geC5zaGFwZQogICAgY250ID0gdG9yY2guYmluY291bnQoc2lkeCwgbWlubGVuZ3RoPUIpCiAgICBTID0gaW50KGNudC5tYXgoKS5pdGVtKCkpCiAgICBzdGFydHMgPSB0b3JjaC5jdW1zdW0oY250LCAwKSAtIGNudAogICAgcG9zID0gdG9yY2guYXJhbmdlKFQsIGRldmljZT14LmRldmljZSkgLSBzdGFydHNbc2lkeF0KICAgIHBhZCA9IHgubmV3X3plcm9zKEIsIFMsIFAsIEQpCiAgICBwYWRbc2lkeCwgcG9zXSA9IHgKICAgIGtlZXAgPSB0b3JjaC56ZXJvcyhCLCBTLCBkdHlwZT10b3JjaC5ib29sLCBkZXZpY2U9eC5kZXZpY2UpCiAgICBrZWVwW3NpZHgsIHBvc10gPSBUcnVlCiAgICByZXR1cm4gKG5vcm0ocGFkLnJlc2hhcGUoQiwgUyAqIFAsIEQpKSwgfmtlZXAucmVwZWF0X2ludGVybGVhdmUoUCwgZGltPTEpKQoKY2xhc3MgX0dhdGVkRGVsdGEobm4uTW9kdWxlKToKCiAgICBkZWYgX19pbml0X18oc2VsZiwgZCwgbl9sYWJlbHMsIG5faGVhZHMsIGRyb3BvdXQpOgogICAgICAgIHN1cGVyKCkuX19pbml0X18oKQogICAgICAgIHNlbGYucSA9IG5uLlBhcmFtZXRlcih0b3JjaC5yYW5kbihuX2xhYmVscywgZCkgKiAwLjAyKQogICAgICAgIHNlbGYua3Zfbm9ybSA9IG5uLkxheWVyTm9ybShkKQogICAgICAgIHNlbGYuYXR0biA9IG5uLk11bHRpaGVhZEF0dGVudGlvbihkLCBuX2hlYWRzLCBkcm9wb3V0PWRyb3BvdXQsIGJhdGNoX2ZpcnN0PVRydWUpCiAgICAgICAgc2VsZi5kX25vcm0gPSBubi5MYXllck5vcm0oZCkKICAgICAgICBzZWxmLmR3ID0gbm4uUGFyYW1ldGVyKHRvcmNoLnJhbmRuKG5fbGFiZWxzLCBkKSAqICgxLjAgLyBkICoqIDAuNSkpCiAgICAgICAgc2VsZi5kYiA9IG5uLlBhcmFtZXRlcih0b3JjaC56ZXJvcyhuX2xhYmVscykpCiAgICAgICAgc2VsZi5nYXRlID0gbm4uUGFyYW1ldGVyKHRvcmNoLnplcm9zKG5fbGFiZWxzKSkKCiAgICBkZWYgZGVsdGEoc2VsZiwgcGF0LCBzaWR4LCBCLCByZXR1cm5fYXR0bik6CiAgICAgICAga3YsIGtwbSA9IF9wYWRfa3YocGF0LCBzaWR4LCBCLCBzZWxmLmt2X25vcm0pCiAgICAgICAgcSA9IHNlbGYucS51bnNxdWVlemUoMCkuZXhwYW5kKEIsIC0xLCAtMSkKICAgICAgICBhdHQsIHcgPSBzZWxmLmF0dG4ocSwga3YsIGt2LCBrZXlfcGFkZGluZ19tYXNrPWtwbSwgbmVlZF93ZWlnaHRzPXJldHVybl9hdHRuLCBhdmVyYWdlX2F0dG5fd2VpZ2h0cz1UcnVlKQogICAgICAgIHJldHVybiAoKHNlbGYuZF9ub3JtKGF0dCkgKiBzZWxmLmR3KS5zdW0oLTEpICsgc2VsZi5kYiwgdykKCmNsYXNzIFRva2VuUmVzaWR1YWxQb29sKF9HYXRlZERlbHRhKToKCiAgICBkZWYgX19pbml0X18oc2VsZiwgZCwgbl9sYWJlbHM9MTIsIG5faGVhZHM9NiwgcGU9NjQsIGRyb3BvdXQ9MC4yKToKICAgICAgICBzdXBlcigpLl9faW5pdF9fKGQsIG5fbGFiZWxzLCBuX2hlYWRzLCBkcm9wb3V0KQogICAgICAgIHNlbGYuYmFzZSA9IG5uLlNlcXVlbnRpYWwobm4uTGF5ZXJOb3JtKDIgKiBkICsgcGUpLCBubi5Ecm9wb3V0KGRyb3BvdXQpLCBubi5MaW5lYXIoMiAqIGQgKyBwZSwgbl9sYWJlbHMpKQoKICAgIGRlZiBmb3J3YXJkKHNlbGYsIHRvaywgc2xvdCwgc2lkeCwgQiwgcHJlcywgcmV0dXJuX2F0dG49RmFsc2UpOgogICAgICAgIGJhc2UgPSBzZWxmLmJhc2UodG9yY2guY2F0KFtfc2VnX21lYW5fbWF4KHRva1s6LCAxOl0ubWVhbigxKSwgc2lkeCwgQiksIHByZXNdLCAxKSkKICAgICAgICBkXywgdyA9IHNlbGYuZGVsdGEodG9rWzosIDE6XSwgc2lkeCwgQiwgcmV0dXJuX2F0dG4pCiAgICAgICAgcmV0dXJuIChiYXNlICsgc2VsZi5nYXRlICogZF8sIHcpCgpjbGFzcyBDb2RleFJlc2lkdWFsUG9vbChfR2F0ZWREZWx0YSk6CgogICAgZGVmIF9faW5pdF9fKHNlbGYsIGQsIG5fbGFiZWxzPTEyLCBuX2hlYWRzPTYsIHBlPTY0LCBkcm9wb3V0PTAuMik6CiAgICAgICAgc3VwZXIoKS5fX2luaXRfXyhkLCBuX2xhYmVscywgbl9oZWFkcywgZHJvcG91dCkKICAgICAgICBzZWxmLmJhc2UgPSBubi5TZXF1ZW50aWFsKG5uLkxheWVyTm9ybSgyICogZCArIHBlKSwgbm4uRHJvcG91dChkcm9wb3V0KSwgbm4uTGluZWFyKDIgKiBkICsgcGUsIG5fbGFiZWxzKSkKCiAgICBkZWYgZm9yd2FyZChzZWxmLCB0b2ssIHNsb3QsIHNpZHgsIEIsIHByZXMsIHJldHVybl9hdHRuPUZhbHNlKToKICAgICAgICBiYXNlID0gc2VsZi5iYXNlKHRvcmNoLmNhdChbX3NlZ19tZWFuX21heCh0b2tbOiwgMF0sIHNpZHgsIEIpLCBwcmVzXSwgMSkpCiAgICAgICAgZF8sIHcgPSBzZWxmLmRlbHRhKHRva1s6LCAxOl0sIHNpZHgsIEIsIHJldHVybl9hdHRuKQogICAgICAgIHJldHVybiAoYmFzZSArIHNlbGYuZ2F0ZSAqIGRfLCB3KQoKY2xhc3MgQ2xzQWRkUG9vbChubi5Nb2R1bGUpOgoKICAgIGRlZiBfX2luaXRfXyhzZWxmLCBkLCBuX2xhYmVscz0xMiwgcGU9NjQsIGRyb3BvdXQ9MC4yKToKICAgICAgICBzdXBlcigpLl9faW5pdF9fKCkKICAgICAgICBzZWxmLm5ldCA9IG5uLlNlcXVlbnRpYWwobm4uTGF5ZXJOb3JtKDQgKiBkICsgcGUpLCBubi5Ecm9wb3V0KGRyb3BvdXQpLCBubi5MaW5lYXIoNCAqIGQgKyBwZSwgbl9sYWJlbHMpKQoKICAgIGRlZiBmb3J3YXJkKHNlbGYsIHRvaywgc2xvdCwgc2lkeCwgQiwgcHJlcywgcmV0dXJuX2F0dG49RmFsc2UpOgogICAgICAgIHJldHVybiAoc2VsZi5uZXQodG9yY2guY2F0KFtfc2VnX21lYW5fbWF4KHRva1s6LCAxOl0ubWVhbigxKSwgc2lkeCwgQiksIF9zZWdfbWVhbl9tYXgodG9rWzosIDBdLCBzaWR4LCBCKSwgcHJlc10sIDEpKSwgTm9uZSkKCmNsYXNzIFJlYWRvdXQobm4uTW9kdWxlKToKCiAgICBkZWYgX19pbml0X18oc2VsZiwgcG9vbCwgZCwgbl9sYWJlbHM9MTIsIHBlPTY0KToKICAgICAgICBzdXBlcigpLl9faW5pdF9fKCkKICAgICAgICBzZWxmLnBvb2xfa2luZCwgc2VsZi5rID0gKHBvb2wsIG5fbGFiZWxzKQogICAgICAgIHNlbGYucHJlc19lbWIgPSBubi5FbWJlZGRpbmcoTl9TTE9UX1RZUEVTICsgMSwgcGUsIHBhZGRpbmdfaWR4PTApCiAgICAgICAgaWYgcG9vbCBpbiAoJ3hyZXMnLCAnY2xzYWRkJywgJ3hjb2RleCcpOgogICAgICAgICAgICBzZWxmLnBvb2wgPSB7J3hyZXMnOiBUb2tlblJlc2lkdWFsUG9vbCwgJ2Nsc2FkZCc6IENsc0FkZFBvb2wsICd4Y29kZXgnOiBDb2RleFJlc2lkdWFsUG9vbH1bcG9vbF0oZCwgbl9sYWJlbHMsIHBlPXBlKQogICAgICAgIGVsaWYgcG9vbCBpbiAoJ2F0dG4nLCAneGF0dG4nKToKICAgICAgICAgICAgaWYgcG9vbCA9PSAneGF0dG4nOgogICAgICAgICAgICAgICAgc2VsZi5wb29sID0gVG9rZW5YQXR0blBvb2woZCwgbl9sYWJlbHMpCiAgICAgICAgICAgICAgICB3ZCA9IDMgKiBkICsgcGUKICAgICAgICAgICAgZWxzZToKICAgICAgICAgICAgICAgIHNlbGYucG9vbCA9IExhYmVsQXR0ZW50aW9uUG9vbChkLCBuX2xhYmVscykKICAgICAgICAgICAgICAgIHdkID0gZCArIHBlCiAgICAgICAgICAgIHNlbGYubm9ybSA9IG5uLkxheWVyTm9ybSh3ZCkKICAgICAgICAgICAgc2VsZi53ID0gbm4uUGFyYW1ldGVyKHRvcmNoLnJhbmRuKG5fbGFiZWxzLCB3ZCkgKiAoMS4wIC8gd2QgKiogMC41KSkKICAgICAgICAgICAgc2VsZi5iID0gbm4uUGFyYW1ldGVyKHRvcmNoLnplcm9zKG5fbGFiZWxzKSkKICAgICAgICBlbHNlOgogICAgICAgICAgICBzZWxmLnBvb2wgPSBNZWFuTWF4UG9vbCgpCiAgICAgICAgICAgIHNlbGYubmV0ID0gbm4uU2VxdWVudGlhbChubi5MYXllck5vcm0oMiAqIGQgKyBwZSksIG5uLkRyb3BvdXQoMC4yKSwgbm4uTGluZWFyKDIgKiBkICsgcGUsIG5fbGFiZWxzKSkKICAgICAgICBzZWxmLmRyb3AgPSBubi5Ecm9wb3V0KDAuMikKCiAgICBkZWYgZm9yd2FyZChzZWxmLCBmLCBzbG90LCBzaWR4LCBCLCByZXR1cm5fYXR0bj1GYWxzZSk6CiAgICAgICAgcGUgPSBzZWxmLnByZXNfZW1iKHNsb3QpCiAgICAgICAgcHJlcyA9IHRvcmNoLnplcm9zKEIsIHBlLnNoYXBlWzFdLCBkZXZpY2U9Zi5kZXZpY2UsIGR0eXBlPWYuZHR5cGUpLmluZGV4X2FkZF8oMCwgc2lkeCwgcGUpCiAgICAgICAgaWYgc2VsZi5wb29sX2tpbmQgaW4gKCd4cmVzJywgJ2Nsc2FkZCcsICd4Y29kZXgnKToKICAgICAgICAgICAgcmV0dXJuIHNlbGYucG9vbChmLCBzbG90LCBzaWR4LCBCLCBwcmVzKVswXQogICAgICAgIHBvb2xlZCwgYXR0biA9IHNlbGYucG9vbChmLCBzaWR4LCBCLCBzbG90PXNsb3QsIHJldHVybl9hdHRuPXJldHVybl9hdHRuKQogICAgICAgIGlmIHNlbGYucG9vbF9raW5kIGluICgnYXR0bicsICd4YXR0bicpOgogICAgICAgICAgICB4ID0gdG9yY2guY2F0KFtwb29sZWQsIHByZXMudW5zcXVlZXplKDEpLmV4cGFuZCgtMSwgc2VsZi5rLCAtMSldLCAtMSkKICAgICAgICAgICAgeCA9IHNlbGYuZHJvcChzZWxmLm5vcm0oeCkpCiAgICAgICAgICAgIHJldHVybiAoeCAqIHNlbGYudykuc3VtKC0xKSArIHNlbGYuYgogICAgICAgIHJldHVybiBzZWxmLm5ldCh0b3JjaC5jYXQoW3Bvb2xlZCwgcHJlc10sIDEpKQoKY2xhc3MgTmV0KG5uLk1vZHVsZSk6CgogICAgZGVmIF9faW5pdF9fKHNlbGYsIGVuYywgY29uZCwgbl9tZXRhPTAsIHBvb2w9J21lYW5fbWF4Jywgc3RlbT0nbmF0aXZlJywgbl9zbGljZT0xNik6CiAgICAgICAgc3VwZXIoKS5fX2luaXRfXygpCiAgICAgICAgc2VsZi5lbmMsIHNlbGYuY29uZCA9IChlbmMsIGNvbmQpCiAgICAgICAgc2VsZi5jb21wcmVzcyA9IERlcHRoQ29tcHJlc3Mobl9zbGljZSwgMykgaWYgc3RlbSA9PSAnY29tcHJlc3MnIGVsc2UgTm9uZQogICAgICAgIHNlbGYubWl4ZXIgPSBTbG90RGVwdGhNaXhlcihuX3NsaWNlKSBpZiBzdGVtID09ICdtaXhlcicgZWxzZSBOb25lCiAgICAgICAgc2VsZi50b2tlbnMgPSBwb29sIGluICgneGF0dG4nLCAneHJlcycsICdjbHNhZGQnLCAneGNvZGV4JykKICAgICAgICBEID0gZW5jLm51bV9mZWF0dXJlcwogICAgICAgIHNlbGYubWV0YV9tbHAgPSBubi5TZXF1ZW50aWFsKG5uLkxheWVyTm9ybShuX21ldGEpLCBubi5MaW5lYXIobl9tZXRhLCAxMjgpLCBubi5HRUxVKCksIG5uLkxpbmVhcigxMjgsIEQpKSBpZiBuX21ldGEgPiAwIGVsc2UgTm9uZQogICAgICAgIHNlbGYucmVhZG91dCA9IFJlYWRvdXQocG9vbCwgRCkKICAgICAgICBpZiBjb25kID09ICdwb3N0JzoKICAgICAgICAgICAgc2VsZi5zbG90X2VtYiA9IG5uLkVtYmVkZGluZyhOX1NMT1RfVFlQRVMgKyAxLCBELCBwYWRkaW5nX2lkeD1NQVNLX0lEWCkKCiAgICBkZWYgZm9yd2FyZChzZWxmLCBpbSwgc2xvdCwgc21ldGEsIHNpZHgsIEIsIHZtPU5vbmUpOgogICAgICAgIGlmIHNlbGYubWl4ZXIgaXMgbm90IE5vbmU6CiAgICAgICAgICAgIGltID0gc2VsZi5taXhlcihpbSwgc2xvdCwgdm0pCiAgICAgICAgaWYgc2VsZi5jb21wcmVzcyBpcyBub3QgTm9uZToKICAgICAgICAgICAgaW0gPSBzZWxmLmNvbXByZXNzKGltKQogICAgICAgIGYgPSBzZWxmLmVuYy5mb3J3YXJkX2ZlYXR1cmVzKGltLCBzbG90KSBpZiBzZWxmLmNvbmQgPT0gJ3Rva2VuJyBlbHNlIHNlbGYuZW5jLmZvcndhcmRfZmVhdHVyZXMoaW0pCiAgICAgICAgaWYgc2VsZi50b2tlbnM6CiAgICAgICAgICAgIGlubmVyID0gZ2V0YXR0cihzZWxmLmVuYywgJ3ZpdCcsIHNlbGYuZW5jKQogICAgICAgICAgICBvcmlnID0gZ2V0YXR0cihzZWxmLmVuYywgJ19vcmlnX3ByZWZpeCcsIGdldGF0dHIoaW5uZXIsICdudW1fcHJlZml4X3Rva2VucycsIDEpKQogICAgICAgICAgICBmID0gdG9yY2guY2F0KFtmWzosIDoxXSwgZls6LCBvcmlnOl1dLCAxKQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIGYgPSBzZWxmLmVuYy5mb3J3YXJkX2hlYWQoZiwgcHJlX2xvZ2l0cz1UcnVlKQogICAgICAgICAgICBpZiBmLmRpbSgpID4gMjoKICAgICAgICAgICAgICAgIGYgPSBmLmZsYXR0ZW4oMSkKICAgICAgICBleCA9IChsYW1iZGEgdjogdi51bnNxdWVlemUoMSkpIGlmIHNlbGYudG9rZW5zIGVsc2UgbGFtYmRhIHY6IHYKICAgICAgICBpZiBzZWxmLmNvbmQgPT0gJ3Bvc3QnOgogICAgICAgICAgICBmID0gZiArIGV4KHNlbGYuc2xvdF9lbWIoc2xvdCkpCiAgICAgICAgaWYgc2VsZi5tZXRhX21scCBpcyBub3QgTm9uZSBhbmQgc21ldGEuc2hhcGVbMV0gPiAwOgogICAgICAgICAgICBtdCA9IHNlbGYubWV0YV9tbHAoc21ldGEpCiAgICAgICAgICAgIGYgPSB0b3JjaC5jYXQoW2YsIG10LnVuc3F1ZWV6ZSgxKV0sIDEpIGlmIHNlbGYudG9rZW5zIGVsc2UgZiArIG10CiAgICAgICAgcmV0dXJuIHNlbGYucmVhZG91dChmLCBzbG90LCBzaWR4LCBCKQptb2RlbHM9W10KX0E1X0xPQURfUkVDRUlQVD1bXQpfYTVfcGF0aHM9W0NLUFQvZidtX2Z7aX0ucHQnIGZvciBpIGluIHJhbmdlKDUpXQppZiBub3QgYWxsKHAuaXNfZmlsZSgpIGZvciBwIGluIF9hNV9wYXRocyk6CiAgICByYWlzZSBSdW50aW1lRXJyb3IoZidBNSByZXF1aXJlcyBtX2YwLnB0Li5tX2Y0LnB0OyBtaXNzaW5nIHtbc3RyKHApIGZvciBwIGluIF9hNV9wYXRocyBpZiBub3QgcC5pc19maWxlKCldfScpCl9hNV9jb25maWc9Tm9uZQpmb3IgY2twdF9wYXRoIGluIF9hNV9wYXRoczoKICAgIHo9dG9yY2gubG9hZChja3B0X3BhdGgsbWFwX2xvY2F0aW9uPSdjcHUnLHdlaWdodHNfb25seT1GYWxzZSkKICAgIGNmZz16WydjZmcnXTtmb2xkPWludCh6Wydmb2xkJ10pCiAgICBpZiBja3B0X3BhdGgubmFtZSE9ZidtX2Z7Zm9sZH0ucHQnOiByYWlzZSBSdW50aW1lRXJyb3IoJ0E1IGZvbGQvZmlsZSBpZGVudGl0eSBtaXNtYXRjaCcpCiAgICBmaWVsZHM9eydiYWNrYm9uZSc6Y2ZnWydiYWNrYm9uZSddLCdjb25kJzpjZmdbJ2NvbmQnXSwncG9vbCc6Y2ZnWydwb29sJ10sJ2ltZyc6aW50KGNmZ1snaW1nJ10pLCduX3NsaWNlJzppbnQoY2ZnLmdldCgnbl9zbGljZScsMTYpKSwnbl9tZXRhJzppbnQoY2ZnLmdldCgnbl9tZXRhJywwKSksJ21ldGEnOmNmZy5nZXQoJ21ldGEnLCdub25lJyksJ3N0ZW0nOmNmZy5nZXQoJ3N0ZW0nLCduYXRpdmUnKSwnbm9ybSc6Y2ZnLmdldCgnbm9ybScsJ25vbmUnKX0KICAgIGlmIF9hNV9jb25maWcgaXMgTm9uZTpfYTVfY29uZmlnPWZpZWxkcwogICAgaWYgZmllbGRzIT1fYTVfY29uZmlnOnJhaXNlIFJ1bnRpbWVFcnJvcihmJ0E1IHBlci1mb2xkIGlucHV0L21vZGVsIGNvbnRyYWN0IGRpZmZlcnM6IHtja3B0X3BhdGh9JykKICAgIGlmIGZpZWxkc1snaW1nJ10hPVNJWkUgb3IgZmllbGRzWyduX3NsaWNlJ10hPU5fU0xJQ0Ugb3IgZmllbGRzWyduX21ldGEnXSE9MDoKICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoZidBNSBjaGVja3BvaW50IGlucHV0IGNvbnRyYWN0IGluY29tcGF0aWJsZSB3aXRoIGxvYWRlcjoge2ZpZWxkc30nKQogICAgaWYgJ2xhYmVscycgaW4geiBhbmQgbGlzdCh6WydsYWJlbHMnXSkhPWxpc3QoTEFCRUxTKTpyYWlzZSBSdW50aW1lRXJyb3IoJ0E1IGxhYmVsIG9yZGVyIGRpZmZlcnMnKQogICAgX3N0ZW09ZmllbGRzWydzdGVtJ107X2luPTMgaWYgX3N0ZW09PSdjb21wcmVzcycgZWxzZSBmaWVsZHNbJ25fc2xpY2UnXQogICAgZW5jPXRpbW0uY3JlYXRlX21vZGVsKGNmZ1snYmFja2JvbmUnXSxwcmV0cmFpbmVkPUZhbHNlLG51bV9jbGFzc2VzPTAsaW5fY2hhbnM9X2luLCoqKHsnaW1nX3NpemUnOmNmZ1snaW1nJ119IGlmICd2aXRfJyBpbiBjZmdbJ2JhY2tib25lJ10gZWxzZSB7fSkpCiAgICBpZiBjZmdbJ2NvbmQnXT09J3Rva2VuJzplbmM9VmlUU2xvdFRva2VuKGVuYyxOX1NMT1RfVFlQRVMpCiAgICBtPU5ldChlbmMsY2ZnWydjb25kJ10sY2ZnLmdldCgnbl9tZXRhJywwKSxjZmdbJ3Bvb2wnXSxzdGVtPV9zdGVtLG5fc2xpY2U9Y2ZnLmdldCgnbl9zbGljZScsMTYpKQogICAgcnNuYV9zdHJpY3RfbG9hZChtLHpbJ3N0YXRlX2RpY3QnXSxmJ0E1IGZvbGQge2ZvbGR9JykKICAgIG1vZGVscy5hcHBlbmQobS5ldmFsKCkpO19BNV9MT0FEX1JFQ0VJUFQuYXBwZW5kKHsnZm9sZCc6Zm9sZCwnZmlsZSc6c3RyKGNrcHRfcGF0aCksJ3NoYTI1Nic6cnNuYV9zaGEoY2twdF9wYXRoKSwndGVuc29yX2tleXMnOmxlbih6WydzdGF0ZV9kaWN0J10pLCdzdHJpY3QnOlRydWUsJ2NvbmZpZyc6ZmllbGRzfSkKICAgIHByaW50KGYnQTUgc3RyaWN0LWxvYWRlZCB7Y2twdF9wYXRoLm5hbWV9OyBhbGwgZW5jb2RlciBhbmQgaGVhZCB0ZW5zb3JzIHByZXNlbnQnKQpDRkc9ZGljdChjZmcpCnJzbmFfanNvbignL2thZ2dsZS93b3JraW5nL2RpYWdub3N0aWNzL2E1X2xvYWRzLmpzb24nLF9BNV9MT0FEX1JFQ0VJUFQpCg==", "19": "QU1QX1BSRUYgPSAnYmYxNicgICMgQlRLRCBwcmVjaXNpb24gcmV0YWluZWQsIGluY2x1ZGluZyBUNCBleGVjdXRpb247IG5vIHVudmFsaWRhdGVkIEFNUCBjaGFuZ2UuCgpkZWYgYW1wX2ZvcihkZXYpOgogICAgaWYgbm90IHN0cihkZXYpLnN0YXJ0c3dpdGgoJ2N1ZGEnKToKICAgICAgICByZXR1cm4gKHRvcmNoLmZsb2F0MzIsIEZhbHNlKQogICAgY2MgPSB0b3JjaC5jdWRhLmdldF9kZXZpY2VfY2FwYWJpbGl0eShkZXYpCiAgICBpZiBBTVBfUFJFRiA9PSAnYmYxNic6CiAgICAgICAgcmV0dXJuICh0b3JjaC5iZmxvYXQxNiwgVHJ1ZSkKICAgIGlmIEFNUF9QUkVGID09ICdmcDE2JzoKICAgICAgICByZXR1cm4gKHRvcmNoLmZsb2F0MTYsIFRydWUpCiAgICBpZiBBTVBfUFJFRiA9PSAnZnAzMic6CiAgICAgICAgcmV0dXJuICh0b3JjaC5mbG9hdDMyLCBGYWxzZSkKICAgIHJldHVybiAodG9yY2guYmZsb2F0MTYgaWYgY2MgPj0gKDgsIDApIGVsc2UgdG9yY2guZmxvYXQxNiwgVHJ1ZSkKQU1QX0RULCBBTVBfT04gPSBhbXBfZm9yKERFVikKV09SS0VSUyA9IG1heCgxLCBtaW4oNCwgb3MuY3B1X2NvdW50KCkgb3IgNCkpCkNIVU5LID0gNDgKTUlDUk8gPSA4CmltcG9ydCBjb3B5IGFzIF9zcGVlZF9hNV9jb3B5CmZyb20gY29uY3VycmVudC5mdXR1cmVzIGltcG9ydCBUaHJlYWRQb29sRXhlY3V0b3IgYXMgX1NwZWVkQTVQb29sCmlmIHRvcmNoLmN1ZGEuZGV2aWNlX2NvdW50KCkgIT0gMjoKICAgIHJhaXNlIFJ1bnRpbWVFcnJvcignQTUgb3B0aW1pemVkIHBhdGggcmVxdWlyZXMgZXhhY3RseSB0d28gR1BVcycpCl9zcGVlZF9hNV9zZWNvbmQgPSBbX3NwZWVkX2E1X2NvcHkuZGVlcGNvcHkobSkudG8oJ2N1ZGE6MScpLmV2YWwoKSBmb3IgbSBpbiBtb2RlbHNdCl9zcGVlZF9hNV9tb2RlbHMgPSBbW20udG8oJ2N1ZGE6MCcpLmV2YWwoKSBmb3IgbSBpbiBtb2RlbHNdLCBfc3BlZWRfYTVfc2Vjb25kXQpkZWwgX3NwZWVkX2E1X3NlY29uZAptb2RlbHMgPSBfc3BlZWRfYTVfbW9kZWxzWzBdCl9zcGVlZF9hNV9wb29sID0gX1NwZWVkQTVQb29sKG1heF93b3JrZXJzPTIpCnByaW50KGYiZGV2aWNlIHtERVZ9IHwgYW1wIHtzdHIoQU1QX0RUKS5zcGxpdCgnLicpWy0xXX0gKG9uPXtBTVBfT059KSB8IHdvcmtlcnMge1dPUktFUlN9IHwgY2h1bmsge0NIVU5LfSB8IG1pY3JvIHtNSUNST30iKQoKZGVmIF9ub3JtXyhpbSk6CiAgICBrID0gQ0ZHLmdldCgnbm9ybScsICdub25lJykKICAgIGlmIGsgPT0gJ3pzY29yZSc6CiAgICAgICAgbSA9IChpbSA+IDApLmZsb2F0KCkKICAgICAgICBuID0gbS5zdW0oZGltPSgxLCAyLCAzKSwga2VlcGRpbT1UcnVlKS5jbGFtcChtaW49MS4wKQogICAgICAgIG11ID0gKGltICogbSkuc3VtKGRpbT0oMSwgMiwgMyksIGtlZXBkaW09VHJ1ZSkgLyBuCiAgICAgICAgdmFyID0gKCgoaW0gLSBtdSkgKiBtKSAqKiAyKS5zdW0oZGltPSgxLCAyLCAzKSwga2VlcGRpbT1UcnVlKSAvIG4KICAgICAgICByZXR1cm4gKGltIC0gbXUpIC8gKHZhci5zcXJ0KCkgKyAxZS0wNikgKiBtCiAgICBpZiBrID09ICdpbWFnZW5ldCc6CiAgICAgICAgbSA9IChpbSA+IDApLmZsb2F0KCkKICAgICAgICByZXR1cm4gKGltIC0gMC40ODUpIC8gMC4yMjkgKiBtCiAgICByZXR1cm4gaW0KCkB0b3JjaC5ub19ncmFkKCkKZGVmIF9zcGVlZF9hNV9taWNybyhpbWFnZXMsIG1hc2tzLCAqLCBkZXYsIG1vZGVsX2dyb3VwLCBhbXBfZHR5cGUpOgogICAgIiIiUHJlZGljdCBhY3F1aXJlZCByb3dzIG9ubHk7IE5hTiBpcyBhIHNlbnRpbmVsIGV4Y2x1c2l2ZWx5IGZvciBhYnNlbnQgaW5wdXRzLiIiIgogICAgYWN0aXZlID0gbnAuZmxhdG5vbnplcm8oKG1hc2tzID4gMCkuYW55KGF4aXM9MSkpCiAgICBvdXQgPSBucC5mdWxsKChsZW4obW9kZWxfZ3JvdXApLCBsZW4obWFza3MpLCBsZW4oTEFCRUxTKSksIG5wLm5hbiwgbnAuZmxvYXQzMikKICAgIGlmIG5vdCBsZW4oYWN0aXZlKToKICAgICAgICByZXR1cm4gb3V0CiAgICBpbXMsIHNsb3RzLCBzaWR4LCB2bXMgPSBbXSwgW10sIFtdLCBbXQogICAgZm9yIGNvbXBhY3RfaW5kZXgsIG9yaWdpbmFsX2luZGV4IGluIGVudW1lcmF0ZShhY3RpdmUpOgogICAgICAgIHByZXNlbnQgPSBucC5mbGF0bm9uemVybyhtYXNrc1tvcmlnaW5hbF9pbmRleF0gPiAwKQogICAgICAgIGJsayA9IGltYWdlc1tvcmlnaW5hbF9pbmRleF1bcHJlc2VudF0KICAgICAgICBpbXMuYXBwZW5kKHRvcmNoLmZyb21fbnVtcHkoYmxrKSkKICAgICAgICB2bXMuYXBwZW5kKHRvcmNoLmZyb21fbnVtcHkoYmxrLnJlc2hhcGUoYmxrLnNoYXBlWzBdLCBibGsuc2hhcGVbMV0sIC0xKS5tYXgoMikgPiAwKSkKICAgICAgICBzbG90cy5hcHBlbmQodG9yY2guZnJvbV9udW1weShwcmVzZW50ICsgMSkubG9uZygpKQogICAgICAgIHNpZHguYXBwZW5kKHRvcmNoLmZ1bGwoKGxlbihwcmVzZW50KSwpLCBjb21wYWN0X2luZGV4LCBkdHlwZT10b3JjaC5sb25nKSkKICAgIGltID0gX25vcm1fKHRvcmNoLmNhdChpbXMpLnRvKGRldiwgbm9uX2Jsb2NraW5nPVRydWUpLmZsb2F0KCkuZGl2XygyNTUuMCkpCiAgICBzbCA9IHRvcmNoLmNhdChzbG90cykudG8oZGV2KQogICAgc2kgPSB0b3JjaC5jYXQoc2lkeCkudG8oZGV2KQogICAgdm0gPSB0b3JjaC5jYXQodm1zKS50byhkZXYpCiAgICBzbSA9IHRvcmNoLnplcm9zKGxlbihzbCksIENGRy5nZXQoJ25fbWV0YScsIDApLCBkZXZpY2U9ZGV2KQogICAgcGVyID0gdG9yY2guemVyb3MobGVuKG1vZGVsX2dyb3VwKSwgbGVuKGFjdGl2ZSksIGxlbihMQUJFTFMpLCBkZXZpY2U9ZGV2LCBkdHlwZT10b3JjaC5mbG9hdDMyKQogICAgZGVmIF9hNV9mb3J3YXJkKGVuYWJsZWQpOgogICAgICAgIHdpdGggdG9yY2guYXV0b2Nhc3QoJ2N1ZGEnIGlmIHN0cihkZXYpLnN0YXJ0c3dpdGgoJ2N1ZGEnKSBlbHNlICdjcHUnLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgZHR5cGU9YW1wX2R0eXBlLCBlbmFibGVkPWVuYWJsZWQpOgogICAgICAgICAgICBmb3IgZm9sZF9pbmRleCwgbW9kZWwgaW4gZW51bWVyYXRlKG1vZGVsX2dyb3VwKToKICAgICAgICAgICAgICAgIHBlcltmb2xkX2luZGV4XSA9IHRvcmNoLnNpZ21vaWQobW9kZWwoaW0sIHNsLCBzbSwgc2ksIGxlbihhY3RpdmUpLCB2bT12bSkuZmxvYXQoKSkKICAgICAgICByZXR1cm4gcGVyLmNwdSgpLm51bXB5KCkKICAgIGdvdCA9IF9hNV9mb3J3YXJkKEFNUF9PTikKICAgIGlmIEFNUF9PTiBhbmQgbm90IG5wLmlzZmluaXRlKGdvdCkuYWxsKCk6CiAgICAgICAgcnNuYV9ldmVudCgnYTVfZnAxNl9ub25maW5pdGVfcmV0cnlfZnAzMicsIHJvd3M9aW50KGxlbihhY3RpdmUpKSkKICAgICAgICBnb3QgPSBfYTVfZm9yd2FyZChGYWxzZSkKICAgIF9iYWQgPSB+bnAuaXNmaW5pdGUoZ290KQogICAgaWYgX2JhZC5hbnkoKToKICAgICAgICByc25hX2V2ZW50KCdhNV9ub25maW5pdGVfbmV1dHJhbCcsIHJvd3M9aW50KF9iYWQuYW55KGF4aXM9KDAsIDIpKS5zdW0oKSkpCiAgICAgICAgZ290ID0gbnAud2hlcmUoX2JhZCwgMC41LCBnb3QpCiAgICByc25hX2Zpbml0ZShnb3QsICdBNSBhY3F1aXJlZC1yb3cgcHJvYmFiaWxpdGllcycsIHByb2JhYmlsaXR5PVRydWUpCiAgICBvdXRbOiwgYWN0aXZlXSA9IGdvdAogICAgcmV0dXJuIG91dAoKIiIiVHdvIEdQVSBvd25lcnMsIHByZXNlcnZpbmcgZXZlcnkgb3JpZ2luYWwgZWlnaHQtc3R1ZHkgbWljcm9iYXRjaC4iIiIKCgpkZWYgc3BlZWRfYTVfcmFuZ2VzKG4sIG1pY3JvPTgsIGRldmljZXM9Mik6CiAgICBpZiBuIDwgMCBvciBtaWNybyA8IDEgb3IgZGV2aWNlcyA8IDE6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigobiwgbWljcm8sIGRldmljZXMpKQogICAgcmV0dXJuIFtbKHN0YXJ0LCBtaW4oc3RhcnQgKyBtaWNybywgbikpCiAgICAgICAgICAgICBmb3IgaSwgc3RhcnQgaW4gZW51bWVyYXRlKHJhbmdlKDAsIG4sIG1pY3JvKSkgaWYgaSAlIGRldmljZXMgPT0gZF0KICAgICAgICAgICAgZm9yIGQgaW4gcmFuZ2UoZGV2aWNlcyldCgoKZGVmIHNwZWVkX2E1X3ByZWRpY3QoaW1hZ2VzLCBtYXNrcywgbW9kZWxzX2J5X2RldmljZSwgZXhlY3V0b3IsIGR0eXBlPU5vbmUpOgogICAgIiIiTm8gc2hhcmQtbG9jYWwgcmFua3M7IG91dHB1dCByZW1haW5zIFtmb2xkLCBvcmlnaW5hbCBzdHVkeSwgZmluZGluZ10uIiIiCiAgICBjb3VudCA9IGxlbihtYXNrcykKICAgIGFzc2VydCBsZW4oaW1hZ2VzKSA9PSBjb3VudAogICAgb3V0cHV0ID0gbnAuZnVsbCgobGVuKG1vZGVsc19ieV9kZXZpY2VbMF0pLCBjb3VudCwgbGVuKExBQkVMUykpLCBucC5uYW4sIG5wLmZsb2F0MzIpCiAgICBwbGFucyA9IHNwZWVkX2E1X3Jhbmdlcyhjb3VudCwgTUlDUk8sIGxlbihtb2RlbHNfYnlfZGV2aWNlKSkKCiAgICBkZWYgb3duZXIoZGV2aWNlX2luZGV4LCByYW5nZXMpOgogICAgICAgIGRldmljZSA9IHRvcmNoLmRldmljZShmImN1ZGE6e2RldmljZV9pbmRleH0iKQogICAgICAgIHJlc3VsdCA9IFtdCiAgICAgICAgd2l0aCB0b3JjaC5jdWRhLmRldmljZShkZXZpY2UpOgogICAgICAgICAgICBmb3Igc3RhcnQsIHN0b3AgaW4gcmFuZ2VzOgogICAgICAgICAgICAgICAgdmFsdWUgPSBfc3BlZWRfYTVfbWljcm8oaW1hZ2VzW3N0YXJ0OnN0b3BdLCBtYXNrc1tzdGFydDpzdG9wXSwKICAgICAgICAgICAgICAgICAgICBkZXY9ZGV2aWNlLCBtb2RlbF9ncm91cD1tb2RlbHNfYnlfZGV2aWNlW2RldmljZV9pbmRleF0sCiAgICAgICAgICAgICAgICAgICAgYW1wX2R0eXBlPUFNUF9EVCBpZiBkdHlwZSBpcyBOb25lIGVsc2UgZHR5cGUpCiAgICAgICAgICAgICAgICByZXN1bHQuYXBwZW5kKChzdGFydCwgc3RvcCwgdmFsdWUpKQogICAgICAgIHJldHVybiByZXN1bHQKCiAgICB0YXNrcyA9IFtleGVjdXRvci5zdWJtaXQob3duZXIsIGQsIHJhbmdlcykgZm9yIGQsIHJhbmdlcyBpbiBlbnVtZXJhdGUocGxhbnMpIGlmIHJhbmdlc10KICAgIGZvciBmdXR1cmUgaW4gdGFza3M6CiAgICAgICAgZm9yIHN0YXJ0LCBzdG9wLCB2YWx1ZSBpbiBmdXR1cmUucmVzdWx0KCk6CiAgICAgICAgICAgIG91dHB1dFs6LCBzdGFydDpzdG9wXSA9IHZhbHVlCiAgICByZXR1cm4gb3V0cHV0CgpkZWYgcHJlZGljdChpbWFnZXMsIG1hc2tzKToKICAgIHJldHVybiBzcGVlZF9hNV9wcmVkaWN0KGltYWdlcywgbWFza3MsIF9zcGVlZF9hNV9tb2RlbHMsIF9zcGVlZF9hNV9wb29sKQoKZGVmIF9hNV92YWxpZGF0ZV9hcHBsaWNhYmlsaXR5KHJhdywgZWxpZ2libGUpOgogICAgIiIiTmV2ZXIgY29uZnVzZSBhbiBhYnNlbnQgaW5wdXQgd2l0aCBmYWlsZWQgaW5mZXJlbmNlIG9uIGFuIGFjcXVpcmVkIGlucHV0LiIiIgogICAgeCA9IG5wLmFzYXJyYXkocmF3KQogICAgbSA9IG5wLmFzYXJyYXkoZWxpZ2libGUsIGR0eXBlPWJvb2wpCiAgICBpZiB4Lm5kaW0gIT0gMyBvciB4LnNoYXBlWzFdICE9IGxlbihtKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCdBNSBlbGlnaWJpbGl0eS9yYXcgc2hhcGUgbWlzbWF0Y2gnKQogICAgaWYgbS5hbnkoKToKICAgICAgICByc25hX2Zpbml0ZSh4WzosIG1dLCAnQTUgYWNxdWlyZWQtcm93IGNvbXBsZXRlIHRlbnNvcicsIHByb2JhYmlsaXR5PVRydWUpCiAgICBpZiAofm0pLmFueSgpIGFuZCBub3QgbnAuaXNuYW4oeFs6LCB+bV0pLmFsbCgpOgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcignQTUgYWJzZW50IHJvd3MgbXVzdCByZXRhaW4gZXhwbGljaXQgTmFOIHNlbnRpbmVscycpCiAgICByZXR1cm4gbQoKZGVmIF9hNV9yYW5rX2F2YWlsYWJsZShyYXcsIGVsaWdpYmxlKToKICAgIGVsaWdpYmxlID0gX2E1X3ZhbGlkYXRlX2FwcGxpY2FiaWxpdHkocmF3LCBlbGlnaWJsZSkKICAgIHJlc3VsdCA9IG5wLmZ1bGwoKGxlbihlbGlnaWJsZSksIHJhdy5zaGFwZVstMV0pLCAuNSwgZHR5cGU9bnAuZmxvYXQ2NCkKICAgIGlmIGVsaWdpYmxlLmFueSgpOgogICAgICAgIHJlc3VsdFtlbGlnaWJsZV0gPSAwLjAKICAgICAgICBmb3IgZm9sZCBpbiByYXc6CiAgICAgICAgICAgIHJlc3VsdFtlbGlnaWJsZV0gKz0gcnNuYV9yYW5rMDEoZm9sZFtlbGlnaWJsZV0pCiAgICAgICAgcmVzdWx0W2VsaWdpYmxlXSAvPSByYXcuc2hhcGVbMF0KICAgIHJldHVybiByZXN1bHQKCmRlZiBfYTVfYmxlbmRfYXZhaWxhYmxlKHBhcmVudCwgYTVfdmFsdWVzLCBlbGlnaWJsZSwgd2VpZ2h0KToKICAgICIiIlBhcmVudC1vbmx5IG9uIG1ldGFkYXRhLXByb3ZlbiBhYnNlbmNlOyBubyBzaWxlbnQgZmFsbGJhY2sgb24gbW9kZWwgZXJyb3IuIiIiCiAgICBiYXNlID0gbnAuYXNhcnJheShwYXJlbnQsIGR0eXBlPW5wLmZsb2F0NjQpCiAgICBhNSA9IG5wLmFzYXJyYXkoYTVfdmFsdWVzLCBkdHlwZT1ucC5mbG9hdDY0KQogICAgbWFzayA9IG5wLmFzYXJyYXkoZWxpZ2libGUsIGR0eXBlPWJvb2wpCiAgICBpZiBiYXNlLnNoYXBlICE9IGE1LnNoYXBlIG9yIG1hc2suc2hhcGUgIT0gKGxlbihiYXNlKSwpOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoJ0E1IG1lcmdlIHNoYXBlIG1pc21hdGNoJykKICAgIHJzbmFfZmluaXRlKGJhc2UsICdBNSBwYXJlbnQgcHJvYmFiaWxpdGllcycsIHByb2JhYmlsaXR5PVRydWUpCiAgICBpZiBub3QgbWFzay5hbnkoKSBvciB3ZWlnaHQgPT0gMDoKICAgICAgICByZXR1cm4gYmFzZS5jb3B5KCkKICAgIHJzbmFfZmluaXRlKGE1W21hc2tdLCAnQTUgYXBwbGljYWJsZSByYW5rcycsIHByb2JhYmlsaXR5PVRydWUpCiAgICBiYXNlX3JhbmsgPSBwZC5EYXRhRnJhbWUoYmFzZSkucmFuayhtZXRob2Q9J2F2ZXJhZ2UnLCBwY3Q9VHJ1ZSkudG9fbnVtcHkoKQogICAgYTVfcmFuayA9IHBkLkRhdGFGcmFtZShhNVttYXNrXSkucmFuayhtZXRob2Q9J2F2ZXJhZ2UnLCBwY3Q9VHJ1ZSkudG9fbnVtcHkoKQogICAgcmVzdWx0ID0gYmFzZV9yYW5rLmNvcHkoKQogICAgcmVzdWx0W21hc2tdID0gKDEtd2VpZ2h0KSpiYXNlX3JhbmtbbWFza10gKyB3ZWlnaHQqYTVfcmFuawogICAgcnNuYV9maW5pdGUocmVzdWx0LCAnQTUgYXBwbGljYWJpbGl0eS1hd2FyZSBtZXJnZScsIHByb2JhYmlsaXR5PVRydWUpCiAgICByZXR1cm4gcmVzdWx0CgojIE1hY3JvIFJPQy1BVUMgZGVwZW5kcyBvbiBvcmRlcmluZywgc28gY29tYmluZSBmb2xkIG9yZGVyaW5ncyByYXRoZXIKIyB0aGFuIGFsbG93aW5nIGEgZm9sZCdzIHByb2JhYmlsaXR5IHNjYWxlIHRvIGRvbWluYXRlIHRoZSBtZWFuLgpwcmVkcyA9IG5wLmZ1bGwoKGxlbihtb2RlbHMpLCBsZW4oc3R1ZGllcyksIGxlbihMQUJFTFMpKSwgbnAubmFuLCBucC5mbG9hdDMyKQpfYTVfZWxpZ2libGUgPSBucC56ZXJvcyhsZW4oc3R1ZGllcyksIGR0eXBlPWJvb2wpCnQwLCBkb25lID0gKHRpbWUudGltZSgpLCAwKQpkZWYgc3VibWl0X3N0dWR5X2Jsb2NrKGV4ZWN1dG9yLCBzdGFydCk6CiAgICAiIiJLZWVwIG9uZSBib3VuZGVkIENQVSBkZWNvZGUgYmxvY2sgYWhlYWQgb2YgR1BVIGluZmVyZW5jZS4iIiIKICAgIGJsb2NrID0gc3R1ZGllc1tzdGFydDpzdGFydCArIENIVU5LXQogICAgZnV0dXJlcyA9IFsKICAgICAgICBleGVjdXRvci5zdWJtaXQoCiAgICAgICAgICAgIGJ1aWxkX3N0dWR5LAogICAgICAgICAgICAoaW5kZXgsIHN0dWR5LCBieS5nZXQoc3R1ZHksIFtdKSksCiAgICAgICAgKQogICAgICAgIGZvciBpbmRleCwgc3R1ZHkgaW4gZW51bWVyYXRlKGJsb2NrKQogICAgXQogICAgcmV0dXJuIHN0YXJ0LCBibG9jaywgZnV0dXJlcwoKIyBDUFUgZGVjb2RlIHRocmVhZHMgYXZvaWQgZm9ya2luZyBhIHByb2Nlc3MgYWZ0ZXIgQ1VEQS9tb2RlbCBpbml0aWFsaXphdGlvbi4Kd2l0aCBBNURlY29kZVBvb2wobWF4X3dvcmtlcnM9V09SS0VSUykgYXMgZXg6CiAgICBwZW5kaW5nX2Jsb2NrID0gc3VibWl0X3N0dWR5X2Jsb2NrKGV4LCAwKQogICAgd2hpbGUgcGVuZGluZ19ibG9jayBpcyBub3QgTm9uZToKICAgICAgICByc25hX2RlYWRsaW5lKCdBNSBmdWxsLWNvaG9ydCBpbmZlcmVuY2UnKQogICAgICAgIGMwLCBibG9jaywgZnV0cyA9IHBlbmRpbmdfYmxvY2sKICAgICAgICBuZXh0X3N0YXJ0ID0gYzAgKyBsZW4oYmxvY2spCiAgICAgICAgcGVuZGluZ19ibG9jayA9ICgKICAgICAgICAgICAgc3VibWl0X3N0dWR5X2Jsb2NrKGV4LCBuZXh0X3N0YXJ0KQogICAgICAgICAgICBpZiBuZXh0X3N0YXJ0IDwgbGVuKHN0dWRpZXMpCiAgICAgICAgICAgIGVsc2UgTm9uZQogICAgICAgICkKICAgICAgICBpbWdzID0gbnAuemVyb3MoKGxlbihibG9jayksIE5fU0xPVCwgTl9TTElDRSwgU0laRSwgU0laRSksIG5wLnVpbnQ4KQogICAgICAgIG1za3MgPSBucC56ZXJvcygobGVuKGJsb2NrKSwgTl9TTE9UKSwgbnAudWludDgpCiAgICAgICAgX2Z1dF9pbmRleCA9IHtmOiBpIGZvciBpLCBmIGluIGVudW1lcmF0ZShmdXRzKX0KICAgICAgICBmb3IgZiBpbiBhc19jb21wbGV0ZWQoZnV0cyk6CiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIGksIGEsIGsgPSBmLnJlc3VsdCgpCiAgICAgICAgICAgICAgICBpbWdzW2ldLCBtc2tzW2ldID0gKGEsIGspCiAgICAgICAgICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZToKICAgICAgICAgICAgICAgIGkgPSBfZnV0X2luZGV4W2ZdCiAgICAgICAgICAgICAgICByc25hX2V2ZW50KCdhNV9zdHVkeV9mYWlsZWQnLCBzdHVkeT1zdHIoYmxvY2tbaV0pLCBlcnJvcj1mJ3t0eXBlKGUpLl9fbmFtZV9ffToge3N0cihlKVs6NTAwXX0nKQogICAgICAgICAgICAgICAgcHJpbnQoZidbYTVdIHtibG9ja1tpXX06IGJ1aWxkIGZhaWxlZCwgdHJlYXRlZCBhcyBBNS1hYnNlbnQgKGZsYWdnZWQpOiB7dHlwZShlKS5fX25hbWVfX306IHtzdHIoZSlbOjIwMF19JywgZmx1c2g9VHJ1ZSkKICAgICAgICAgICAgICAgIGltZ3NbaV0gPSAwOyBtc2tzW2ldID0gMAogICAgICAgIF9hNV9lbGlnaWJsZVtjMDpjMCArIGxlbihibG9jayldID0gKG1za3MgPiAwKS5hbnkoYXhpcz0xKQogICAgICAgIHByZWRzWzosIGMwOmMwICsgbGVuKGJsb2NrKV0gPSBwcmVkaWN0KGltZ3MsIG1za3MpCiAgICAgICAgX2E1X3ZhbGlkYXRlX2FwcGxpY2FiaWxpdHkocHJlZHNbOiwgYzA6YzAgKyBsZW4oYmxvY2spXSwgX2E1X2VsaWdpYmxlW2MwOmMwICsgbGVuKGJsb2NrKV0pCiAgICAgICAgZG9uZSArPSBsZW4oYmxvY2spCiAgICAgICAgZWwgPSB0aW1lLnRpbWUoKSAtIHQwCiAgICAgICAgcHJpbnQoZicgIHtkb25lOix9L3tsZW4oc3R1ZGllcyk6LH0gIHtlbCAvIDYwOi4xZn1tICBldGEge2VsIC8gZG9uZSAqIChsZW4oc3R1ZGllcykgLSBkb25lKSAvIDYwOi4xZn1tJywgZmx1c2g9VHJ1ZSkKICAgICAgICBkZWwgaW1ncywgbXNrcwogICAgICAgIGdjLmNvbGxlY3QoKQpwcmludChmJ1xuaW5mZXJlbmNlIGRvbmUgaW4geyh0aW1lLnRpbWUoKSAtIHQwKSAvIDYwOi4xZn0gbWluJykKbnAuc2F2ZXpfY29tcHJlc3NlZCgnL2thZ2dsZS93b3JraW5nL3NwZWVkX2E1X3Jhdy5ucHonLAogICAgc3R1ZHlfdWlkcz1ucC5hc2FycmF5KHN0dWRpZXMpLCByYXdfcHJvYmFiaWxpdGllcz1wcmVkcywgYXBwbGljYWJsZT1fYTVfZWxpZ2libGUpCl9zcGVlZF9hNV9wb29sLnNodXRkb3duKHdhaXQ9VHJ1ZSkKZGVsIF9zcGVlZF9hNV9tb2RlbHMsIG1vZGVscywgbSwgZW5jLCB6CmdjLmNvbGxlY3QoKQpmb3IgX3NwZWVkX2RldmljZSBpbiByYW5nZSgyKToKICAgIHdpdGggdG9yY2guY3VkYS5kZXZpY2UoX3NwZWVkX2RldmljZSk6CiAgICAgICAgdG9yY2guY3VkYS5lbXB0eV9jYWNoZSgpCkE1X1cgPSAwLjUyCkE1X0xBQkVMUyA9IGxpc3QoTEFCRUxTKQpfYTVfb2sgPSBfYTVfdmFsaWRhdGVfYXBwbGljYWJpbGl0eShwcmVkcywgX2E1X2VsaWdpYmxlKQpfYTVfcmFua19tZWFuID0gX2E1X3JhbmtfYXZhaWxhYmxlKHByZWRzLCBfYTVfZWxpZ2libGUpCkE1X1BSRURTID0gZGljdCh6aXAoc3ViX2RmWydTdHVkeUluc3RhbmNlVUlEJ10uYXN0eXBlKHN0ciksIF9hNV9yYW5rX21lYW4uYXN0eXBlKG5wLmZsb2F0MzIpKSkKQTVfQVZBSUxBQkxFID0gZGljdCh6aXAoc3ViX2RmWydTdHVkeUluc3RhbmNlVUlEJ10uYXN0eXBlKHN0ciksIF9hNV9lbGlnaWJsZS50b2xpc3QoKSkpCnJzbmFfc2F2ZV9wcmVkaWN0aW9ucygnYTVfcmFua19tZWFuJywgc3R1ZGllcywgX2E1X3JhbmtfbWVhbiwgTEFCRUxTKQpyc25hX2pzb24oJy9rYWdnbGUvd29ya2luZy9kaWFnbm9zdGljcy9hNV9hcHBsaWNhYmlsaXR5Lmpzb24nLCB7CiAgICAndW5hdmFpbGFibGVfdWlkcyc6W3UgZm9yIHUsIG9rIGluIEE1X0FWQUlMQUJMRS5pdGVtcygpIGlmIG5vdCBva10sCiAgICAnYXZhaWxhYmxlX2NvdW50JzppbnQoX2E1X2VsaWdpYmxlLnN1bSgpKSwgJ21vZGVsX2ZhaWx1cmVfZmFsbGJhY2tfYWxsb3dlZCc6RmFsc2UsCiAgICAncmFua19wbGFjZWhvbGRlcl9mb3JfdW5hdmFpbGFibGUnOjAuNSwgJ3BsYWNlaG9sZGVyX3VzZWRfaW5fYmxlbmQnOkZhbHNlfSkKcnNuYV9waGFzZSgnYTUnLCAnQ09NUExFVEUnLCBhcHBsaWNhYmxlPWludChfYTVfZWxpZ2libGUuc3VtKCkpLCBhYnNlbnQ9aW50KCh+X2E1X2VsaWdpYmxlKS5zdW0oKSkpCmZvciBfYTVrLCBfYTV2IGluIF9BNV9TQVZFRC5pdGVtcygpOgogICAgZ2xvYmFscygpW19hNWtdID0gX2E1dgpkZWwgX0E1X1NBVkVELCBfYTVrLCBfYTV2Cg==", "20": "X2E1X3N1YiA9IHBkLnJlYWRfY3N2KCcva2FnZ2xlL3dvcmtpbmcvX3BpcGVsaW5lX3N0YWdlLmNzdicsIGR0eXBlPXsnU3R1ZHlJbnN0YW5jZVVJRCc6c3RyfSkKYXNzZXJ0IF9hNV9zdWIuY29sdW1ucy50b2xpc3QoKVsxOl0gPT0gQTVfTEFCRUxTLCAnc3VibWlzc2lvbiBzY2hlbWEgZHJpZnQnCmlmIEE1X1cgPiAwOgogICAgX2E1X3VpZF9vcmRlciA9IF9hNV9zdWIuU3R1ZHlJbnN0YW5jZVVJRC5hc3R5cGUoc3RyKS50b2xpc3QoKQogICAgX2E1X291cnMgPSBucC5zdGFjayhbQTVfUFJFRFNbdV0gZm9yIHUgaW4gX2E1X3VpZF9vcmRlcl0pCiAgICBfYTVfcHJlc2VuY2UgPSBucC5hc2FycmF5KFtBNV9BVkFJTEFCTEVbdV0gZm9yIHUgaW4gX2E1X3VpZF9vcmRlcl0sIGR0eXBlPWJvb2wpCiAgICBfYTVfc3ViW0E1X0xBQkVMU10gPSBfYTVfYmxlbmRfYXZhaWxhYmxlKAogICAgICAgIF9hNV9zdWJbQTVfTEFCRUxTXS50b19udW1weSgpLCBfYTVfb3VycywgX2E1X3ByZXNlbmNlLCBBNV9XKQogICAgYXNzZXJ0IG5wLmlzZmluaXRlKF9hNV9zdWJbQTVfTEFCRUxTXS50b19udW1weSgpKS5hbGwoKQogICAgX2E1X3N1Yi50b19jc3YoJy9rYWdnbGUvd29ya2luZy9fcGlwZWxpbmVfc3RhZ2UuY3N2JywgaW5kZXg9RmFsc2UpCg==", "21": "ZnJvbSBfX2Z1dHVyZV9fIGltcG9ydCBhbm5vdGF0aW9ucwppbXBvcnQgYmFzZTY0IGFzIF9yYWRfYjY0CmltcG9ydCB6bGliIGFzIF9yYWRfemxpYgpfUkFEX0NBTF9QQVlMT0FEID0gJ2VOcnRtazF2SThjUmh2OUtzSmRjS0tFL3E2dHpjNHo0WkNNQmNqUVdockNSRFNHMlpFamFJRUdRLzU3bjdSbFJRM0tHNGpxTEpBY0RTNG83MDlOZFhSOXZ2VlU5LzN6MzArM04vYnZmZlJ1dWF3Z3hsbTdlcTJlUGVmZnJwVjh2L1Y5ZXV2THJhRDNrNmlsVzZ6bjEyNnZZZCtVNmVLbXg5UmlhRjdPU3grWDF3ZUU2N0s3U2RTZ3BWU2F1dWFlY1V4cjNydHAxTEswSHl5Vjd5OUdteS9FNnBCaFRMNjFadDJnV3grV3RTZXc2Sm1Pb001QUh1dFhwK3JvOCtacmxzbnZKT2ZEZGZSTCtLbDR6ZDJicWxsTm1nYTdydnJ2YTJPekdORnV5bkd4cG1ueHJTLzA2OWhDdHB0NlQxZExMT1FWc1RiSjFmV3VuRzJxWEFYOE5pVSs3Vks4cmRxdk5VODl1d1F3anBXeWVMZFRDblZ5ODR1YTlwdGhTanpuSFhMakRwWnhiTGU0V1BSQVFDVCtMclNWY0dHZEx6TlgwWEtoZXNDMmVWNXVaNjdsUVBEbG1TelVoUzIrNjFFTHRvVE95V0dqZFB1ZGM3M2Z2bmo3Yy9IZzdFbHB5ellYamR3SVozMm03WDNJTlowY3J0dnRjODMzdWE2ZnlvVVlVR2Y0SDEzckpwWCtHSzVhYTVWYmVXTzl5ZS8wM2JMaTk3aThTcGFTQ2w0OW5jNGdkeEkycDFkWnlWbVFuZkw1NmpCSC9qNzBHcVNxNmR5TVJPTEhRR3ZDcGNTbmtZc3llb2hORXJuVmpiYW1WanM0N3dPcEJhN0JBc2E2MVNYdWxEZlNJd0FiQWtRcURtU0szOFd3SW1LSzFrRkozZDExQ3BGcVIxbVBIbGoxcFdWQUhlRGZ5VTJoazB2R29nejBHcUNCYkttRk14OXhJV0VBYlE4STRQVXZtQXBsUUNldWlqN0dOWEdNazN5aEJzRklmRW52VkU2SEZZTUhEdEZ1THpLd3B5YTlnaXNhWmR1QWtjeUF2bXcxUk9qbWQ3T25CWXl0cE9CcUpqVHlLNEt6VDBQaTB0UUpZc2xjMG5HcWNOWlY3ZEVhUnZmY1Nld2c5aDRwNDFpd08rNENXTWtORzEzMjZWQm1IRlVvQjdWb2FrekUrSkF0WU4vTmtuUzlsTEorOVM1cVI1NlEya0xBNHFnVHVwbEdObVVKQWtiWGlHYnJVMEhDSXNndFlhR09VVlhiYXlOZXlrTGZwR3Y4bHVsQmZnaUZFbXg2MXhtM0xUbHJPUG9hNTFOZzdJeUppVDR0V3hyRTBabW5KUm5oYXdac3lMZ2pYSUM5TVJ1MHRSQWdJbWtKN2JVeUhLazFlVWdJZXd0UmpYR0RKcU8xbU5KckhmTlV0eTFIUlcyU1NvVjNGQVZJQS8vaHJVWEtBTnhoYlJiRWtncUFGS2lDQzQzcU9FWHNQVGFLdEtDZGtxeDMva29NWXN1UCttaDRIckhvUXRxU0l3L2g0RE03RXBSTDV6UmlyQWNIaVVEajJTVWxyOWZGQXNFbEhCWC96V2dKQ1Fsdys4M1Frc3c4UHQ5K1R5MGhtT0JRQmg5WFFBcjdmeGpSUTJJa0dUVC93OGNBaUJLd1JjMWp3Y01oK1dLdE1nRlNoTmFER0pXUkFlY2hOU09NVWxkWFR5bk5JVVBBYUVLS2tDdjFjTEZ6eGFvd0JPbVg4dlU4UGoxdm9XYTVKVFBGY0dONFFYbStQcFpQakE4UVFZWXA5UWFiOXJaaUlBdUxYOEFBOGYvalg0a0hnSStCWEN0QkVlT1RGdk5QYXBJeUM5MllBRU9tSHlwSmNDU2lGT3BRaTcxMk03M0lMOEtpQlg4SHo2MnBETnNBSDhNSy9yTVIrdE5TUklSSndBa0lJWThSbi9mRDJrQjJWOUk3Qk1YK0tTSi9YSnROQWJ3RmdsbVFaYnUvOTBDWmNNaEFlWkt1ZEtBdmxTQkx3TndQMWFBODhCWXBQZEpSa2JBREdlQnpXTjJJT3Z5U1ZFTHh5VTBMeDBKSFFIc0ZwRVFSc0NCOWlPMXFUM0lPSEdVak1EcWVjelg0QmlJYnZDamtpTEQ2dDZHMjM5cCtnQU1MMUtRdUFkYUxMZGlkRFY2WTZ1UFIrOSszTFQ4S3JsWWd6QXNqZzdvaytWSmFraW12Z0FqbjVxVUZtdFRHSktYR3hTYWRnMlV1TGtXb2s0RXZHSjBIZHNyK0RtcFZsR3NVTVpreHR1VEk4dkRBVmFJc25TSURiSzBEaFZTcEFGbHU0aVJ0Z3JMWVdSYUFyQURPY0ZEQWZFckVkd0JTOGRXcE5xSHVwVzNvN25JdWtUbXhsbUFTcTRMLzUxRVNyem5xSlRTa2dDenNsVlNNbmNla2I4NjU5bGpleVl0dHhLMG9YOWs1MG4zaDJrRHFvYXBUb1d0OFM5L3lPM3RUWHl0ZUx1KzE5cmtQVmlLa0libExuekpoSlVoWUVOVUpDdGRmV0trRUZlOVdUc1dpbjZhenBxSkZJZkV3TnlMTmVzK1BaWEZWM2grdElPWGpiNW16SVJuQ0xhV0t1aG5QdGpzQ1ZQQU93ZGtoUWhCU01mV0xWcnVUZ3dFTS9XWHQxRllndjVBTjRJc3lCaUhyT1NzY0NCaklvbWtzZ1pDUEZuOWdycURyRTlVWERRQ1NQZnM1Ulh5ZXVRbDNnd2NFSWRuNUp6QURocEczNHM0dUY1TlEyNGVoMUdXSVNrRW1ZQTZpRmlOZXpZQWJrS05HSmhJaytsOVhOWEszcjdBZ0xUNFluUlV1SGljSlMwTlJMTEcwS0RyRjFJYmthTXpqMk1vZUt6QUgva1RvOUtzbldKZThnRFpVdXhzMWlQaTBDYXlBQk1kTElUNHFRQ2JmRU5mQ2lBc3VqSXNqOUtNVVExa1h3WFVCRlpzcEhabTlaai9kQi9TclZ5MXFPSmc5QXBLQWxtU1FOcHI2U2ppYm5OZ1ZvcVpRQ3I4Z2xsZ3NUaUZBT0NGU2VEY1lXdUtPQ2haUVhpNWRQeG91RjVPbzZvZ1ZUYXhEd25WRThLZlFGSTZab213Z2Ivb0k0VkcyQWUxTWRzVFFDd1F1WnA1WlJPWlBjZHVkV0hyWlJmd2U0Y1BFMWN2OUwvRkR1d1J3R05ZT3R0R2xsTVJJdFMxSzNzRkRRREF3UU12Z3BmSXp5SWVRajJ5VEZPaG9tR0xzQlZQdEFBU0dMcWlYS25DaEFTVy80SUxVVEdsZnRRbFVWbDhGRGppZmJLWUl6TEtYbWNvNGFuTlBLNlpWbDhNaFZCcytENVlnVG84SFh1SUVEUURKRjR3SEVZTDVQdkVFeG40UEl0N3gwSnVtRGVVZ2p4SGVGSVpEdCs2eE41bUFMQ2M2cE1qVFo3b3k4RW9tQ1VBMDlNVFJ2dnVDTHh5RkMrc0VXQ0dpcWpFK0hJQUo0ZzM5Qmk1c2FkU045TUpaYUZXTGJwdWJCTkJ2bG13clBBT05KV1MxRHBhSWdieUt1WlNiS2c3cVp6Z05Lc3Fud0lYOFJBT3ZJTmRnZlJnMGpQTkpCV0UrNXhBUTlxRGFBUGg0bktJbmFnakFMOW90ajI2VStzQW5iVUVSVEYwUVlqSFY4ajhPRUErbW9odEg5U0xMdW9VSWg2dUFXSEk2eUF3NTR1RXNxTDZ6dWZNQUF0M3lXKzZ4U0ZMbUt3RW1ZZ0lQbFhIWWJQQW8zQXZLb3ZDay8wS1I1bTNmbUtrcEZ6Mlczbmc1OGg3S3pxVlZIcVZZVXBmR2xBSkh2UTJkTjdRS21Yb1FJVGdnSFV6OUhHWmhTdHk1c21ZSGVTTDRwRmR2YlhsWFpjU0pWN0dDSEtQcHNlUEVtVkFPbTQxV3hidVczWStxa3JoWWhpc2JyU0JxYkdpR2RBa0J3ZDVnek5laHhHVVZoenU1SkNLYTJWQXlMMnBmWWdZZ1V0Y1R3dTNSS1p6TzJKdWJsaGZDRGJPMVFxeXB1OFNkVFdPMDUwMVo2aUFBSDY4ZzIxbUVDdmpYOGJaWjd5dnBWaXFxOUJsSENuaGo3REZ1aVhtOHFFd2dyQWtLOWhwZmJ1VS9uTi9pNGw3SS9xUUVucFI0cVZIVXhnRGpibFd1TTJVWkZoT3B6cCthcGI4aTR1YThnbnlIcnFQTUhzODM5Z0s3Z2FhWUREdElMcU5UV09GOGt4VllkTXhsVkR3RTY4L0Y4RFIyQ2dwQ1Nra3prRXZLWTN4OUdvUnFqcG44WkxqNjJUeUVvUlZWMWR4clYyR3ROT01vc1BBanFTQzZwbTZ5UktSSkZNNXduaXhiaGg2dUxsOUZkakVsc1V2WDdXeFI2MVQxZ0dvaE9rUWFlajI3TXhpUitEUWpBbmRRNVNnTDRZYitWTUtUMlVuY0dLNHNlNWhmZVI3SnI2bnlnYlRKY1BHSzdRU3BYT29CSmt1TVhsRlRZUVgwYVJzT2JZRVFuMjJCTnJEUWRLd21aMUREYVpOY0FOeHVtZGtHczNwWklKY2FDY3hCdVBTbEd6aVRmK1VOWklncW1MTVRXUzEvWFlOQnhGb3dINUZjaFpiN3VUNUVRcWVydlVLK1JjMTVwUDU1bUJwS3JhNldqdTVNQ2Jad2lLT1NTL0hGdVp5VlJCRmVlSW9WcmMzNW9OblZ0MUFScEtEYUhGMkJPMXozRExJTE9yT1ZaZGhJRzUyb2p5RnhzVmExazZCcStzT1M3QUEwdXBJcXBpQ3Q5T204KzFTc0lvSTQvWllGS1pxOXRkd25oQ3pyVlJwb0N4cHFTbyswdWExRE5MQ09NYzNYMTl2R1NaWjl6bmZFQU1UVXB5bVNHcVlXL2twWFU2RldWNitwYWg5T0dMdmVhVGtvcE9WMWQzVTlvV2ZBaE1peUs0NmZSZ1BlTDlMVFNEQXRVamo1Y0ZONEQ4UzV6bWFDam1hVGpKL0t2eGIzTWFGckhWSTI3UVMydlJmY01DbWVxY1VYWStOWDY2NTJva3lqMTlPRUMzU2FGcmRXeWV5Sk14bzdnUHJBTm4xN0dqUTVSZ3RxdnJlZ05qenIxaFlPT3hBVGVvczRiL0lJdEdSeEVGWkM0YU5ydVZMMVpiR3g4b2pwRTVtb0xpR05hdmF6QitiYXhQb1hyL2dLNTZ6akkxVkdianRHOG5BM1hnM1NwRmwyZ3RxenFzTS9vR2diNUF4ajR3K1hEMWc0OE1CRnJNQW5SK1RSeE1jaTErOGgrdUNDMWUxbm1DN1hFV2hmMnpFZEl3NVNzQ2UzVGNjMThYaWJzVS9QS2hKaGQ3YTM4MHM2N1JMRXZRUVdVdzlLS2ptcHM3cWVmVmV1Yi9JWlFvTGJLMHF4bkhSRlA3cWxPeThCVVJDN0Z5MkxEUGs3TjFGMVZFbTFsclpiU21TV1ZTb05rNjNEUVhndklvaEFqWWNqWFU3Yi96STh1N1JWdlBpUkNoU3FWamtpVTZZUWpyVDFJbVZBQU4vV29LRXA2MlYyYVFHQWRBdVFLaFNXNXFvdXlJZFE0ak5lNU5TNkk4djEwaExJZXFJSU1xcGxEbjcxbzBkWWw0ZkVGa1JaNExtaEZKTlVTUXFaQ05sQlQyZklXbkt6SjlYV3dHNDhiT3pHWldFSWZmY1Z4L3EyM3hBemlCVjgzZ2YxY0U0Lyttdkk4L0VGVjNRVm1QVzBhNnJCN1pEMzE0cGxvVGxsT1NsRllxWDNYNHU3VEpnNmpteExSV3hpdDFGYVB0QXJLb0hmSGViM2p3UG1OR0RxZnZTK0l2OU5zMTRWdjZwOXJmeWZ0K0hHaUZxbUpJUVNVKytybGJlZ1BCcUR1bWxpOXpvT0dTcHRlYzhPNkdBV3F0YUFna09ncVozUDFXaFFRMDhhajJLR09OdUVJcWRaZXR6dUxGQjZxUW5paWQ4SGtaVG5seUdzUEduQTRsWTVRdTE0NTZXbmJva0c5SXVwd1UwTm9QaEhUd1JSVXF5blpVMEhPYlY4UVV5WnVvL2xiNnRGaHNkeGI3YkNvVTlxV2U4dkx4QndqMmxhVnpnUytiSVpHTWZlZTFRYWxkbDg3Z0JaNWpqcjNZNHJxMlhhZjNMYXRvaEh3eHpicUJLdHY0ZDhiSG5oMWVVcWZWUngwN2pjNHpmenlTcmo4SUdWOU5NRlg0bUJLcnE2RlhjNEp6MTU0LzM3Mzd1NysrZmJ4dyszUHo5TjdlZzBYMG1IQ2VIZkhiSHJOaHJDSXBkWFJBL0xSUkM0V1I5c1UvWnFKQjQ2WEpzSm91d1UxclBUK2lsNnVLSHFOQWRiSjJJblVKcDB3VldCVjd3OE51bGRVN3VNeWFqWnFoMk42VWZlb002eW0xekxHaXpGNlQ2QW5OUVppSE8vS1pNaWpaTU9WMi95S1FGS3YwVFNYcTBIYjV0ZUU5RjRITHA1MFFLSjNPWDY0ZWRaN2llKysrUExyZDd0MzM5eis1ZTdteDkvODhRdCtmODJkeDVmLy9PbXI2ZThmdnYvKzQ5UGR3ejAvZjMvejE5dkgzejd4Njh1SCsrZnBLaFArL1Bqdy9QRGg0Y2Z2K0h6ODZmNUprOTkvOTNUN2VIZXJzZmZmL2ZubWgvSDN5NGZIOGZlTHY5L3g5NnViNStsN3ZxOWYwd2o5bXNmOCtISDZmaG5EcjNrTXZ6UkdHM3A4K1BpelZ0M3ZhWHkveXNqb3g1c1B6eDhmYnhuKzdjdVd2N205djN2NjhQRnBzZkg5cGNXd0xjMW95RUkzZi83SC9jUGY3cDd2bmhaNmV2LytYLzhHWUllM3hnPT0nCiMgQlRLRCBSYWQgZ3JhcGg6IHByZXNlcnZlIGl0cyBFMTMvRTExIGNhbGlicmF0ZWQgZmVhdHVyZSBjb250cmFjdDsgcmVtb3ZlIHVudXNlZCB0d2luIHdvcmsuCiMKIyBUaGUgcGlubmVkIHJlZmVyZW5jZSBSYWQgZmFtaWx5IGlzIGZ1c2VkIHdpdGggY29ycmVjdC1jb250cmFjdCBFMTMsIHRoZW4KIyB0aGUgc2FtZSBFMTMgaGVhZHMgcnVuIG9uIHRoZSBFMTEgbGF5b3V0IGF0IDAuMTUuIE5vIHR3aW4vbGVnYWN5IHdyYXBwZXIKIyBmb2xsb3dzIGl0LCBtYXRjaGluZyB0aGUgYnJhbmNoIHRoYXQgcHJvZHVjZWQgVjQ4J3MgdmlzaWJsZSBzdWJtaXNzaW9uLgoKaW1wb3J0IGNvbnRleHRsaWIgYXMgX3JhZF9jb250ZXh0bGliCmltcG9ydCBnYyBhcyBfcmFkX2djCmltcG9ydCBoYXNobGliIGFzIF9yYWRfaGFzaGxpYgppbXBvcnQganNvbiBhcyBfcmFkX2pzb24KaW1wb3J0IG9zIGFzIF9yYWRfb3MKaW1wb3J0IHJlIGFzIF9yYWRfcmUKaW1wb3J0IHRpbWUgYXMgX3JhZF90aW1lCmZyb20gY29uY3VycmVudC5mdXR1cmVzIGltcG9ydCBUaHJlYWRQb29sRXhlY3V0b3IgYXMgX1JhZFRocmVhZFBvb2wKZnJvbSBwYXRobGliIGltcG9ydCBQYXRoIGFzIF9SYWRQYXRoCgppbXBvcnQgbnVtcHkgYXMgX3JhZF9ucAppbXBvcnQgcGFuZGFzIGFzIF9yYWRfcGQKaW1wb3J0IHB5ZGljb20gYXMgX3JhZF9weWRpY29tCmltcG9ydCB0b3JjaCBhcyBfcmFkX3RvcmNoCmltcG9ydCB0b3JjaC5ubiBhcyBfcmFkX25uCmltcG9ydCB0b3JjaC5ubi5mdW5jdGlvbmFsIGFzIF9yYWRfRgpmcm9tIHRvcmNodmlzaW9uLm1vZGVscyBpbXBvcnQgcmVzbmV0NTAgYXMgX3JhZF9yZXNuZXQ1MAoKX1JBRF9MQUJFTFMgPSBbCiAgICAnQUNMJywgJ01DTCcsICdNZWRpYWwgTWVuaXNjdXMnLCAnTGF0ZXJhbCBNZW5pc2N1cycsICdNZWRpYWwgT0EnLAogICAgJ0xhdGVyYWwgT0EnLCAnUEYgT0EnLCAnRWZmdXNpb24nLCAnU3lub3ZpdGlzJywgIkJha2VyJ3MiLAogICAgJ0NvbnR1c2lvbicsICdGcmFjdHVyZScsCl0KX1JBRF9BTFBIQSA9IDAuNTUKX1JBRF9FWENMVURFID0gKCJCYWtlcidzIiwgJ0ZyYWN0dXJlJykKX1JBRF9IRUFEU19TSEEyNTYgPSAnNTRmNjU3ODI2YjM0NThhN2JhM2Q0NjJlMTk4YmEzODA3MzJmMmIxMzYyNDYxODIzMTI3MDQ5Mjk4NzRhOWEyYycKX1JBRF9SRUZFUkVOQ0VfSEVBRFNfU0hBMjU2ID0gJzBmNDY1NjQ5Nzk5ZWNmYmNjYWFjMTc2Nzg0NDYzOWU3Y2VkNDRlMWJjOWJhYmRlNmU0YmFjN2M1ZDliODllYWEnCl9SQURfRU5DT0RFUl9TSEEyNTYgPSAnMDg2MjlmN2U3YmQzZTI5YjhlZTk1MjJjYTNmNjVjZTRkMDEwYTdkZGY3NGYwZWEzYzdlM2YzZDBiYmFiMDczNCcKX1JBRF9FMTNfSEVBRFNfU0hBMjU2ID0gJ2FkOWYxOWFmNzNiZmRmNGU0OTI2M2MwZTQ1MDYwZGMzY2IyMzllMTE5NTAzOWIyNmRjOGMwYTNhNmJjZDFhOGEnCl9SQURfRTEzX01FTUJFUl9XRUlHSFQgPSAwLjUwCl9SQURfVjQ4X1NFQ09ORF9BTFBIQSA9IDAuMjAKX1JBRF9UV0lOX0FMVF9XRUlHSFQgPSAwLjUwMDAwMQpfUkFEX1RPS0VOX0RJTSwgX1JBRF9IRUFEX0RJTSA9IDIwNDgsIDUxMgoKX1JBRF9FMTFfU0xPVFMgPSBbCiAgICAoJ1NBR19OT0ZTJywgJ1NhZ2l0dGFsJywgTm9uZSwgRmFsc2UpLAogICAgKCdDT1JfTk9GUycsICdDb3JvbmFsJywgTm9uZSwgRmFsc2UpLAogICAgKCdBWF9OT0ZTJywgJ0F4aWFsJywgTm9uZSwgRmFsc2UpLAogICAgKCdTQUdfRlMnLCAnU2FnaXR0YWwnLCBOb25lLCBUcnVlKSwKXQpfUkFEX0UxMV9DUk9QX01NID0gMTMwLjAKX1JBRF9FMTFfQ0FDSEVfU0xJQ0VTID0gOApfUkFEX0UxMV9JTUcgPSAyMjQKCl9SQURfRTEzX1NMT1RTID0gWwogICAgKCdTQUdfRlMnLCAnU2FnaXR0YWwnLCBOb25lLCBUcnVlKSwKICAgICgnQ09SX0ZTJywgJ0Nvcm9uYWwnLCBOb25lLCBUcnVlKSwKICAgICgnQVhfRlMnLCAnQXhpYWwnLCBOb25lLCBUcnVlKSwKICAgICgnU0FHX05PRlMnLCAnU2FnaXR0YWwnLCBOb25lLCBGYWxzZSksCl0KX1JBRF9FMTNfQ1JPUF9NTSA9IDEzMC4wCl9SQURfRTEzX0NBQ0hFX1NMSUNFUyA9IDgKX1JBRF9FMTNfSU1HID0gMjI0CgojIE91ciBpbmRlcGVuZGVudGx5IHRyYWluZWQgZml2ZS1mb2xkIGZhbWlseS4gIEl0cyBwcmVwcm9jZXNzaW5nIGFuZCBlc3RpbWF0b3IKIyBhcmUgcHJlc2VydmVkIGZyb20gVjM1OiBuYXRpdmUgRElDT00gZ2VvbWV0cnkvZmF0LXNhdCBoYW5kbGluZyBhbmQgYSBtZWFuIG9mCiMgcGVyLWZvbGQgcGVyY2VudGlsZSByYW5rcyAocmF0aGVyIHRoYW4gdjE1J3MgcmFuayBvZiB0aGUgcHJvYmFiaWxpdHkgbWVhbikuCl9PVVJfTl9TTE9ULCBfT1VSX05fU0xJQ0UsIF9PVVJfSU1HID0gMywgOCwgMjI0CgojIEV4YWN0IFY0MC9FMTAgdGVzdCByZXByZXNlbnRhdGlvbjogdGhyZWUgZmF0LXN1cHByZXNzZWQgcGxhbmVzLCBlaWdodAojIGFjcXVpcmVkIHNsaWNlcyBwZXIgcGxhbmUsIGZ1bGwgZnJhbWUsIGxlZ2FjeSBvcmRlcmluZy9sYXRlcmFsaXR5L2ZpbGwuClNMT1RTID0gWwogICAgKCdTQUdfRlMnLCAnU2FnaXR0YWwnLCBOb25lLCBUcnVlKSwKICAgICgnQ09SX0ZTJywgJ0Nvcm9uYWwnLCBOb25lLCBUcnVlKSwKICAgICgnQVhfRlMnLCAnQXhpYWwnLCBOb25lLCBUcnVlKSwKXQpOX1NMT1QgPSBsZW4oU0xPVFMpCkNBQ0hFX1NMSUNFUyA9IDgKSU1HID0gQ0FDSEVfSU1HID0gMjI0CkNST1BfTU0gPSAxMF8wMDAuMApTTElDRV9CQU5EID0gKDAuMiwgMC44KQpSVUxFUyA9IGRpY3QoUlVMRVNfTEVHQUNZKQpUSU1FX0JVREdFVCA9IDguMCAqIDM2MDAKCgpkZWYgX3JhZF9sb2cobWVzc2FnZSk6CiAgICBwcmludChmJ1tSYWQtZHVhbDVdIHttZXNzYWdlfScsIGZsdXNoPVRydWUpCgoKZGVmIF9yYWRfc2hhMjU2KHBhdGgsIGNodW5rPTggPDwgMjApOgogICAgZGlnZXN0ID0gX3JhZF9oYXNobGliLnNoYTI1NigpCiAgICB3aXRoIG9wZW4ocGF0aCwgJ3JiJykgYXMgaGFuZGxlOgogICAgICAgIGZvciBibG9jayBpbiBpdGVyKGxhbWJkYTogaGFuZGxlLnJlYWQoY2h1bmspLCBiJycpOgogICAgICAgICAgICBkaWdlc3QudXBkYXRlKGJsb2NrKQogICAgcmV0dXJuIGRpZ2VzdC5oZXhkaWdlc3QoKQoKCmRlZiBfcmFkX2ZpbmRfZmlsZShuYW1lLCBleHBlY3RlZF9zaGE9Tm9uZSwgZXhwbGljaXRfZW52PU5vbmUpOgogICAgaWYgZXhwbGljaXRfZW52IGFuZCBfcmFkX29zLmVudmlyb24uZ2V0KGV4cGxpY2l0X2Vudik6CiAgICAgICAgY2FuZGlkYXRlcyA9IFtfUmFkUGF0aChfcmFkX29zLmVudmlyb25bZXhwbGljaXRfZW52XSldCiAgICBlbHNlOgogICAgICAgIGJ5X25hbWUgPSB7CiAgICAgICAgICAgICdSZXNOZXQ1MC5wdCc6IFsKICAgICAgICAgICAgICAgICcva2FnZ2xlL2lucHV0L2RhdGFzZXRzL21hcndhbm1hdGgvcmVzbmV0LTUwLXJhZGltYWdlbmV0LW1hcndhbi9SZXNOZXQ1MC5wdCcsCiAgICAgICAgICAgICAgICAnL2thZ2dsZS9pbnB1dC9yZXNuZXQtNTAtcmFkaW1hZ2VuZXQtbWFyd2FuL1Jlc05ldDUwLnB0JywKICAgICAgICAgICAgXSwKICAgICAgICAgICAgJ3Y1Ml9yYWRpbWFnZW5ldF9oZWFkcy5wdCc6IFsKICAgICAgICAgICAgICAgICcva2FnZ2xlL2lucHV0L2RhdGFzZXRzL3BydnNpeWFuL3JzbmEta25lZS12NTItcmFkaW1hZ2VuZXQtaGVhZHMtMjAyNjA4MTIvdjUyX3JhZGltYWdlbmV0X2hlYWRzLnB0JywKICAgICAgICAgICAgICAgICcva2FnZ2xlL2lucHV0L3JzbmEta25lZS12NTItcmFkaW1hZ2VuZXQtaGVhZHMtMjAyNjA4MTIvdjUyX3JhZGltYWdlbmV0X2hlYWRzLnB0JywKICAgICAgICAgICAgICAgICcva2FnZ2xlL2lucHV0L2RhdGFzZXRzL2FudG9pbmVnZzEvcnNuYS1rbmVlLWU5LXJhZGltYWdlbmV0LWhlYWRzLXYxNS92NTJfcmFkaW1hZ2VuZXRfaGVhZHMucHQnLAogICAgICAgICAgICAgICAgJy9rYWdnbGUvaW5wdXQvcnNuYS1rbmVlLWU5LXJhZGltYWdlbmV0LWhlYWRzLXYxNS92NTJfcmFkaW1hZ2VuZXRfaGVhZHMucHQnLAogICAgICAgICAgICBdLAogICAgICAgICAgICAndjUyX2UxMV9oZWFkcy5wdCc6IFsKICAgICAgICAgICAgICAgICcva2FnZ2xlL2lucHV0L25vdGVib29rcy9zb2ZpYWFuamVuamUvcnNuYS1rbmVlLWUxMy10cmFpbi9yc25hX3JhZF9lMTEvdjUyX2UxMV9oZWFkcy5wdCcsCiAgICAgICAgICAgICAgICAnL2thZ2dsZS9pbnB1dC9yc25hLWtuZWUtZTEzLXRyYWluL3JzbmFfcmFkX2UxMS92NTJfZTExX2hlYWRzLnB0JywKICAgICAgICAgICAgICAgICcva2FnZ2xlL2lucHV0L25vdGVib29rcy9zb2ZpYWFuamVuamUvcnNuYS1rbmVlLWUxMS10cmFpbi9yc25hX3JhZF9lMTEvdjUyX2UxMV9oZWFkcy5wdCcsCiAgICAgICAgICAgICAgICAnL2thZ2dsZS9pbnB1dC9yc25hLWtuZWUtZTExLXRyYWluL3JzbmFfcmFkX2UxMS92NTJfZTExX2hlYWRzLnB0JywKICAgICAgICAgICAgICAgICcva2FnZ2xlL2lucHV0L2RhdGFzZXRzL2FudG9pbmVnZzEvcnNuYS1rbmVlLWUxMS1kaXZlcnNlLWhlYWRzLXYyMC92NTJfZTExX2hlYWRzLnB0JywKICAgICAgICAgICAgICAgICcva2FnZ2xlL2lucHV0L3JzbmEta25lZS1lMTEtZGl2ZXJzZS1oZWFkcy12MjAvdjUyX2UxMV9oZWFkcy5wdCcsCiAgICAgICAgICAgIF0sCiAgICAgICAgfQogICAgICAgIGlmIG5hbWUgbm90IGluIGJ5X25hbWU6CiAgICAgICAgICAgIHJhaXNlIEZpbGVOb3RGb3VuZEVycm9yKGYndW5waW5uZWQgUmFkIGFydGlmYWN0IG5hbWU6IHtuYW1lfScpCiAgICAgICAgY2FuZGlkYXRlcyA9IFtfUmFkUGF0aChwYXRoKSBmb3IgcGF0aCBpbiBieV9uYW1lW25hbWVdXQogICAgZXhpc3RpbmcgPSBbcGF0aCBmb3IgcGF0aCBpbiBjYW5kaWRhdGVzIGlmIHBhdGguaXNfZmlsZSgpXQogICAgdmFsaWQgPSBbCiAgICAgICAgcGF0aCBmb3IgcGF0aCBpbiBleGlzdGluZwogICAgICAgIGlmIGV4cGVjdGVkX3NoYSBpcyBOb25lIG9yIF9yYWRfc2hhMjU2KHBhdGgpID09IGV4cGVjdGVkX3NoYQogICAgXQogICAgaWYgbGVuKHZhbGlkKSAhPSAxOgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcigKICAgICAgICAgICAgZidleHBlY3RlZCBvbmUgdmVyaWZpZWQgUmFkIGFydGlmYWN0IHtuYW1lfSwgZm91bmQge3ZhbGlkfTsgZXhpc3Rpbmc9e2V4aXN0aW5nfScKICAgICAgICApCiAgICByZXR1cm4gdmFsaWRbMF0KCgpjbGFzcyBfUmFkRW5jb2RlcihfcmFkX25uLk1vZHVsZSk6CiAgICBkZWYgX19pbml0X18oc2VsZik6CiAgICAgICAgc3VwZXIoKS5fX2luaXRfXygpCiAgICAgICAgc2VsZi5iYWNrYm9uZSA9IF9yYWRfbm4uU2VxdWVudGlhbCgKICAgICAgICAgICAgKmxpc3QoX3JhZF9yZXNuZXQ1MCh3ZWlnaHRzPU5vbmUpLmNoaWxkcmVuKCkpWzotMl0KICAgICAgICApCgogICAgZGVmIGZvcndhcmQoc2VsZiwgaW1hZ2UpOgogICAgICAgIHJldHVybiBzZWxmLmJhY2tib25lKGltYWdlKS5tZWFuKGRpbT0oMiwgMykpCgoKY2xhc3MgX1JhZEhlYWQoX3JhZF9ubi5Nb2R1bGUpOgogICAgZGVmIF9faW5pdF9fKHNlbGYpOgogICAgICAgIHN1cGVyKCkuX19pbml0X18oKQogICAgICAgIHNlbGYucHJvamVjdCA9IF9yYWRfbm4uU2VxdWVudGlhbCgKICAgICAgICAgICAgX3JhZF9ubi5MYXllck5vcm0oX1JBRF9UT0tFTl9ESU0pLAogICAgICAgICAgICBfcmFkX25uLkxpbmVhcihfUkFEX1RPS0VOX0RJTSwgX1JBRF9IRUFEX0RJTSksCiAgICAgICAgICAgIF9yYWRfbm4uR0VMVSgpLAogICAgICAgICkKICAgICAgICBzZWxmLnBsYW5lID0gX3JhZF9ubi5QYXJhbWV0ZXIoX3JhZF90b3JjaC5yYW5kbihOX1NMT1QsIF9SQURfSEVBRF9ESU0pICogLjAxKQogICAgICAgIHNlbGYucG9zaXRpb24gPSBfcmFkX25uLlBhcmFtZXRlcihfcmFkX3RvcmNoLnJhbmRuKENBQ0hFX1NMSUNFUywgX1JBRF9IRUFEX0RJTSkgKiAuMDEpCiAgICAgICAgc2VsZi5xdWVyeSA9IF9yYWRfbm4uUGFyYW1ldGVyKF9yYWRfdG9yY2gucmFuZG4obGVuKF9SQURfTEFCRUxTKSwgX1JBRF9IRUFEX0RJTSkgKiAuMDIpCiAgICAgICAgc2VsZi5hdHRuID0gX3JhZF9ubi5NdWx0aWhlYWRBdHRlbnRpb24oCiAgICAgICAgICAgIF9SQURfSEVBRF9ESU0sIDgsIGRyb3BvdXQ9LjEwLCBiYXRjaF9maXJzdD1UcnVlCiAgICAgICAgKQogICAgICAgIHNlbGYuZnVzZSA9IF9yYWRfbm4uU2VxdWVudGlhbCgKICAgICAgICAgICAgX3JhZF9ubi5MYXllck5vcm0oX1JBRF9IRUFEX0RJTSAqIDQpLAogICAgICAgICAgICBfcmFkX25uLkxpbmVhcihfUkFEX0hFQURfRElNICogNCwgX1JBRF9IRUFEX0RJTSksCiAgICAgICAgICAgIF9yYWRfbm4uR0VMVSgpLAogICAgICAgICAgICBfcmFkX25uLkRyb3BvdXQoLjE1KSwKICAgICAgICApCiAgICAgICAgc2VsZi53ZWlnaHQgPSBfcmFkX25uLlBhcmFtZXRlcigKICAgICAgICAgICAgX3JhZF90b3JjaC5yYW5kbihsZW4oX1JBRF9MQUJFTFMpLCBfUkFEX0hFQURfRElNKSAqIC4wMgogICAgICAgICkKICAgICAgICBzZWxmLmJpYXMgPSBfcmFkX25uLlBhcmFtZXRlcihfcmFkX3RvcmNoLnplcm9zKGxlbihfUkFEX0xBQkVMUykpKQoKICAgIGRlZiBmb3J3YXJkKHNlbGYsIGZlYXR1cmUsIG1hc2spOgogICAgICAgIHRva2VuID0gc2VsZi5wcm9qZWN0KGZlYXR1cmUuZmxvYXQoKSkKICAgICAgICB0b2tlbiA9IHRva2VuLnZpZXcobGVuKHRva2VuKSwgc2VsZi5wbGFuZS5zaGFwZVswXSwgc2VsZi5wb3NpdGlvbi5zaGFwZVswXSwgc2VsZi5wbGFuZS5zaGFwZVstMV0pCiAgICAgICAgdG9rZW4gPSB0b2tlbiArIHNlbGYucGxhbmVbTm9uZSwgOiwgTm9uZV0gKyBzZWxmLnBvc2l0aW9uW05vbmUsIE5vbmVdCiAgICAgICAgdG9rZW4gPSB0b2tlbi5mbGF0dGVuKDEsIDIpCiAgICAgICAga2V5X3BhZGRpbmcgPSBtYXNrIDw9IDAKICAgICAgICBhbGxfZW1wdHkgPSBrZXlfcGFkZGluZy5hbGwoMSkKICAgICAgICBpZiBhbGxfZW1wdHkuYW55KCk6CiAgICAgICAgICAgIGtleV9wYWRkaW5nID0ga2V5X3BhZGRpbmcuY2xvbmUoKQogICAgICAgICAgICBrZXlfcGFkZGluZ1thbGxfZW1wdHksIDBdID0gRmFsc2UKICAgICAgICBxdWVyeSA9IHNlbGYucXVlcnkudW5zcXVlZXplKDApLmV4cGFuZChsZW4odG9rZW4pLCAtMSwgLTEpCiAgICAgICAgYXR0ZW5kZWQgPSBxdWVyeSArIHNlbGYuYXR0bigKICAgICAgICAgICAgcXVlcnksIHRva2VuLCB0b2tlbiwga2V5X3BhZGRpbmdfbWFzaz1rZXlfcGFkZGluZywgbmVlZF93ZWlnaHRzPUZhbHNlCiAgICAgICAgKVswXQogICAgICAgIGRlbm9taW5hdG9yID0gbWFzay5zdW0oMSwga2VlcGRpbT1UcnVlKS5jbGFtcF9taW4oMSkudW5zcXVlZXplKC0xKQogICAgICAgIG1lYW4gPSAodG9rZW4gKiBtYXNrLnVuc3F1ZWV6ZSgtMSkpLnN1bSgxLCBrZWVwZGltPVRydWUpIC8gZGVub21pbmF0b3IKICAgICAgICBtZWFuID0gbWVhbi5leHBhbmQoLTEsIGxlbihfUkFEX0xBQkVMUyksIC0xKQogICAgICAgIGZ1c2VkID0gc2VsZi5mdXNlKF9yYWRfdG9yY2guY2F0KAogICAgICAgICAgICBbYXR0ZW5kZWQsIG1lYW4sIF9yYWRfdG9yY2guYWJzKGF0dGVuZGVkIC0gbWVhbiksIGF0dGVuZGVkICogbWVhbl0sIGRpbT0tMQogICAgICAgICkpCiAgICAgICAgcmV0dXJuIChmdXNlZCAqIHNlbGYud2VpZ2h0LnVuc3F1ZWV6ZSgwKSkuc3VtKC0xKSArIHNlbGYuYmlhcwoKCmRlZiBfcmFkX2xvYWRfcHVibGljX2hlYWRzKGRldmljZSwgZXhwZWN0ZWRfc2hhKToKICAgIGhlYWRzX3BhdGggPSBfcmFkX2ZpbmRfZmlsZSgndjUyX3JhZGltYWdlbmV0X2hlYWRzLnB0JywgZXhwZWN0ZWRfc2hhKQogICAgcGF5bG9hZCA9IF9yYWRfdG9yY2gubG9hZChoZWFkc19wYXRoLCBtYXBfbG9jYXRpb249J2NwdScsIHdlaWdodHNfb25seT1UcnVlKQogICAgZXhwZWN0ZWQgPSB7CiAgICAgICAgJ3ZlcnNpb24nOiAndjUyLXJhZGltYWdlbmV0LXJlc25ldDUwLW9mZmljaWFsLTEnLAogICAgICAgICd0YXJnZXRzJzogX1JBRF9MQUJFTFMsCiAgICAgICAgJ2VuY29kZXJfc2hhMjU2JzogX1JBRF9FTkNPREVSX1NIQTI1NiwKICAgICAgICAnZW5jb2Rlcl9zb3VyY2VfY29tbWl0JzogJzBjZTE2ZjczNzVkYjQyMzZlNjQ2ODI5ZDFlY2E2MWNkYjQyODIxMzMnLAogICAgICAgICdpbWcnOiAyMjQsCiAgICAgICAgJ3NsaWNlc19wZXJfcGxhbmUnOiA4LAogICAgICAgICdmZWF0dXJlJzogJ2dsb2JhbF9hdmVyYWdlX3Bvb2wnLAogICAgfQogICAgZm9yIGtleSwgdmFsdWUgaW4gZXhwZWN0ZWQuaXRlbXMoKToKICAgICAgICBpZiBwYXlsb2FkLmdldChrZXkpICE9IHZhbHVlOgogICAgICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoZidwdWJsaWMtdjE1IGhlYWQgY29udHJhY3QgZHJpZnQgZm9yIHtrZXl9JykKICAgIGZvbGRzID0gcGF5bG9hZC5nZXQoJ2ZvbGRzJykKICAgIGlmIG5vdCBpc2luc3RhbmNlKGZvbGRzLCBsaXN0KSBvciBsZW4oZm9sZHMpICE9IDU6CiAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKCdwdWJsaWMtdjE1IGJ1bmRsZSByZXF1aXJlcyBleGFjdGx5IGZpdmUgaGVhZHMnKQogICAgaWYgc29ydGVkKGludChyZWNvcmQuZ2V0KCdmb2xkJywgLTEpKSBmb3IgcmVjb3JkIGluIGZvbGRzKSAhPSBsaXN0KHJhbmdlKDUpKToKICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoJ3B1YmxpYy12MTUgZm9sZCBpZGVudGl0eSBkcmlmdCcpCiAgICBoZWFkcyA9IFtdCiAgICBmb3IgcmVjb3JkIGluIGZvbGRzOgogICAgICAgIGhlYWQgPSBfUmFkSGVhZCgpLnRvKGRldmljZSkuZXZhbCgpCiAgICAgICAgaGVhZC5sb2FkX3N0YXRlX2RpY3QocmVjb3JkWydzdGF0ZV9kaWN0J10sIHN0cmljdD1UcnVlKQogICAgICAgIGhlYWRzLmFwcGVuZChoZWFkKQogICAgcmV0dXJuIGhlYWRzLCBzdHIoaGVhZHNfcGF0aCkKCgpkZWYgX3JhZF9sb2FkX2UxM19oZWFkcyhkZXZpY2UpOgogICAgIyBWNDggdXNlZCBhbiB1bnF1YWxpZmllZCBmaWxlbmFtZSBzaGFyZWQgYnkgRTExIGFuZCBFMTMuIFJlc29sdmUgdGhlCiAgICAjIGludGVuZGVkIEUxMyBidW5kbGUgYnkgY29udGVudCBhbmQgdmFsaWRhdGUgaXRzIGNvbXBsZXRlIHBpeGVsIGNvbnRyYWN0LgogICAgaGVhZHNfcGF0aCA9IF9yYWRfZmluZF9maWxlKCd2NTJfZTExX2hlYWRzLnB0JywgX1JBRF9FMTNfSEVBRFNfU0hBMjU2KQogICAgcGF5bG9hZCA9IF9yYWRfdG9yY2gubG9hZChoZWFkc19wYXRoLCBtYXBfbG9jYXRpb249J2NwdScsIHdlaWdodHNfb25seT1GYWxzZSkKICAgIGV4cGVjdGVkID0gewogICAgICAgICd2ZXJzaW9uJzogJ2UxMS1yYWRpbWFnZW5ldC1yZXNuZXQ1MC1kaXZlcnNlLTEnLAogICAgICAgICd0YXJnZXRzJzogX1JBRF9MQUJFTFMsCiAgICAgICAgJ2VuY29kZXJfc2hhMjU2JzogX1JBRF9FTkNPREVSX1NIQTI1NiwKICAgICAgICAnc2xvdHMnOiBbbGlzdChzbG90KSBmb3Igc2xvdCBpbiBfUkFEX0UxM19TTE9UU10sCiAgICAgICAgJ2Nyb3BfbW0nOiBfUkFEX0UxM19DUk9QX01NLAogICAgICAgICdpbWcnOiBfUkFEX0UxM19JTUcsCiAgICAgICAgJ3NsaWNlc19wZXJfcGxhbmUnOiBfUkFEX0UxM19DQUNIRV9TTElDRVMsCiAgICAgICAgJ2ZlYXR1cmUnOiAnZ2xvYmFsX2F2ZXJhZ2VfcG9vbCcsCiAgICB9CiAgICBmb3Iga2V5LCB2YWx1ZSBpbiBleHBlY3RlZC5pdGVtcygpOgogICAgICAgIGlmIHBheWxvYWQuZ2V0KGtleSkgIT0gdmFsdWU6CiAgICAgICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcihmJ0UxMyBoZWFkIGNvbnRyYWN0IGRyaWZ0IGZvciB7a2V5fScpCiAgICBmb2xkcyA9IHBheWxvYWQuZ2V0KCdmb2xkcycpCiAgICBpZiBub3QgaXNpbnN0YW5jZShmb2xkcywgbGlzdCkgb3IgbGVuKGZvbGRzKSAhPSA1OgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcignRTEzIGJ1bmRsZSByZXF1aXJlcyBleGFjdGx5IGZpdmUgaGVhZHMnKQogICAgaWYgc29ydGVkKGludChyZWNvcmQuZ2V0KCdmb2xkJywgLTEpKSBmb3IgcmVjb3JkIGluIGZvbGRzKSAhPSBsaXN0KHJhbmdlKDUpKToKICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoJ0UxMyBmb2xkIGlkZW50aXR5IGRyaWZ0JykKICAgIGhlYWRzID0gW10KICAgIGZvciByZWNvcmQgaW4gZm9sZHM6CiAgICAgICAgaGVhZCA9IF9SYWRIZWFkKCkudG8oZGV2aWNlKS5ldmFsKCkKICAgICAgICBoZWFkLmxvYWRfc3RhdGVfZGljdChyZWNvcmRbJ3N0YXRlX2RpY3QnXSwgc3RyaWN0PVRydWUpCiAgICAgICAgaGVhZHMuYXBwZW5kKGhlYWQpCiAgICByZXR1cm4gaGVhZHMsIHN0cihoZWFkc19wYXRoKQoKCmRlZiBfcmFkX2xvYWRfbW9kZWxzKGRldmljZSk6CiAgICAiIiJUd28gcGVyc2lzdGVudCBlbmNvZGVyIHJlcGxpY2FzOyB1bnVzZWQgYWx0ZXJuYXRpdmUgaGVhZHMgYXJlIG5vdCBsb2FkZWQuIiIiCiAgICBpbXBvcnQgY29weQogICAgcGF0aCA9IF9yYWRfZmluZF9maWxlKCdSZXNOZXQ1MC5wdCcsIF9SQURfRU5DT0RFUl9TSEEyNTYsIGV4cGxpY2l0X2Vudj0nUlNOQV9SQURfV0VJR0hUX1BBVEgnKQogICAgZW5jb2RlciA9IF9SYWRFbmNvZGVyKCkuZXZhbCgpCiAgICByc25hX3N0cmljdF9sb2FkKGVuY29kZXIsIF9yYWRfdG9yY2gubG9hZChwYXRoLG1hcF9sb2NhdGlvbj0nY3B1Jyx3ZWlnaHRzX29ubHk9VHJ1ZSksJ1JhZCBlbmNvZGVyJykKICAgIGlmIHN1bShwLm51bWVsKCkgZm9yIHAgaW4gZW5jb2Rlci5wYXJhbWV0ZXJzKCkpIT0yMzUwODAzMjoKICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoJ1JhZCBlbmNvZGVyIHBhcmFtZXRlciBjb3VudCBjaGFuZ2VkJykKICAgIHJlcGxpY2FzPVtlbmNvZGVyLnRvKCdjdWRhOjAnKSwgY29weS5kZWVwY29weShlbmNvZGVyKS50bygnY3VkYToxJyldCiAgICBmb3IgbSBpbiByZXBsaWNhczoKICAgICAgICBtLnJlcXVpcmVzX2dyYWRfKEZhbHNlKQogICAgaGVhZHMsaHBhdGg9X3JhZF9sb2FkX3B1YmxpY19oZWFkcyhkZXZpY2UsX1JBRF9SRUZFUkVOQ0VfSEVBRFNfU0hBMjU2KQogICAgcmV0dXJuIHJlcGxpY2FzLGhlYWRzLHN0cihwYXRoKSxocGF0aAoKCkBfcmFkX3RvcmNoLmluZmVyZW5jZV9tb2RlKCkKZGVmIF9yYWRfZW5jb2RlKGVuY29kZXIsIHBpeGVscywgc2xvdF9tYXNrLCBkZXZpY2UpOgogICAgIiIiUHJlc2VydmUgcGFyZW50J3MgPD05NiBpbWFnZXMvZGV2aWNlIGNodW5rcyB3aXRob3V0IERhdGFQYXJhbGxlbCByZXBsaWNhdGlvbi4iIiIKICAgIG4sc2xvdHMsc2xpY2VzLGhlaWdodCx3aWR0aD1waXhlbHMuc2hhcGUKICAgIHRva2VuX21hc2s9X3JhZF9ucC5yZXBlYXQoc2xvdF9tYXNrWzosOixOb25lXSxzbGljZXMsYXhpcz0yKS5yZXNoYXBlKG4sLTEpCiAgICBfcmFkX2VtcHR5PWludCgodG9rZW5fbWFzay5zdW0oMSk9PTApLnN1bSgpKQogICAgaWYgX3JhZF9lbXB0eToKICAgICAgICBwcmludChmJ1tyYWRdIHtfcmFkX2VtcHR5fSBzdHVkeShpZXMpIHdpdGggbm8gc2xvdCBldmlkZW5jZSBlbmNvZGVkIGFzIGFsbC16ZXJvIHJvd3MgKHBhcmVudCBwb2xpY3kpJywgZmx1c2g9VHJ1ZSkKICAgIHZhbGlkPV9yYWRfbnAuZmxhdG5vbnplcm8odG9rZW5fbWFzay5yZXNoYXBlKC0xKT4wKQogICAgZmxhdD1waXhlbHMucmVzaGFwZSgtMSxoZWlnaHQsd2lkdGgpCiAgICBmZWF0dXJlcz1fcmFkX25wLnplcm9zKChuLHNsb3RzKnNsaWNlcyxfUkFEX1RPS0VOX0RJTSksX3JhZF9ucC5mbG9hdDE2KQogICAgZGVmIHJ1bihpbmRpY2VzLGQpOgogICAgICAgIGRldj1fcmFkX3RvcmNoLmRldmljZShmJ2N1ZGE6e2R9JykKICAgICAgICBkZWYgX3JhZF9mb3J3YXJkKF9hbXApOgogICAgICAgICAgICB3aXRoIF9yYWRfdG9yY2guY3VkYS5kZXZpY2UoZGV2KSwgX3JhZF90b3JjaC5pbmZlcmVuY2VfbW9kZSgpLCBfcmFkX3RvcmNoLmF1dG9jYXN0KCdjdWRhJyxkdHlwZT1fcmFkX3RvcmNoLmZsb2F0MTYsZW5hYmxlZD1fYW1wKToKICAgICAgICAgICAgICAgIGltYWdlPV9yYWRfdG9yY2guZnJvbV9udW1weShmbGF0W2luZGljZXNdKS50byhkZXYpLmZsb2F0KCkuZGl2XygxMjcuNSkuc3ViXygxLikKICAgICAgICAgICAgICAgIGltYWdlPWltYWdlLnVuc3F1ZWV6ZSgxKS5leHBhbmQoLTEsMywtMSwtMSkuY29udGlndW91cygpCiAgICAgICAgICAgICAgICByZXR1cm4gZW5jb2RlcltkXShpbWFnZSkuZmxvYXQoKS5jcHUoKS5udW1weSgpCiAgICAgICAgdmFsdWU9X3JhZF9mb3J3YXJkKFRydWUpCiAgICAgICAgaGFsZj12YWx1ZS5hc3R5cGUoX3JhZF9ucC5mbG9hdDE2KQogICAgICAgIGlmIG5vdCAoX3JhZF9ucC5pc2Zpbml0ZSh2YWx1ZSkuYWxsKCkgYW5kIF9yYWRfbnAuaXNmaW5pdGUoaGFsZikuYWxsKCkpOgogICAgICAgICAgICByc25hX2V2ZW50KCdyYWRfZnAxNl9ub25maW5pdGVfcmV0cnlfZnAzMicsdG9rZW5zPWludChsZW4oaW5kaWNlcykpKQogICAgICAgICAgICB2YWx1ZT1fcmFkX2ZvcndhcmQoRmFsc2UpOyBoYWxmPV9yYWRfbnAuY2xpcCh2YWx1ZSwtNjUwMDAsNjUwMDApLmFzdHlwZShfcmFkX25wLmZsb2F0MTYpCiAgICAgICAgaWYgbm90IF9yYWRfbnAuaXNmaW5pdGUoaGFsZikuYWxsKCk6CiAgICAgICAgICAgIHJzbmFfZXZlbnQoJ3JhZF9ub25maW5pdGVfemVyb2VkJyx0b2tlbnM9aW50KCh+X3JhZF9ucC5pc2Zpbml0ZShoYWxmKS5yZXNoYXBlKGxlbihoYWxmKSwtMSkuYWxsKGF4aXM9MSkpLnN1bSgpKSkKICAgICAgICAgICAgaGFsZj1fcmFkX25wLndoZXJlKF9yYWRfbnAuaXNmaW5pdGUoaGFsZiksaGFsZiwwKS5hc3R5cGUoX3JhZF9ucC5mbG9hdDE2KQogICAgICAgIHJldHVybiBpbmRpY2VzLGhhbGYKICAgIHdpdGggX1JhZFRocmVhZFBvb2wobWF4X3dvcmtlcnM9MikgYXMgcG9vbDoKICAgICAgICBmb3Igc3RhcnQgaW4gcmFuZ2UoMCxsZW4odmFsaWQpLDE5Mik6CiAgICAgICAgICAgIHJzbmFfZGVhZGxpbmUoJ1JhZCBmdWxsLWNvaG9ydCBpbmZlcmVuY2UnKQogICAgICAgICAgICBibG9jaz12YWxpZFtzdGFydDpzdGFydCsxOTJdOyBjdXQ9KGxlbihibG9jaykrMSkvLzIKICAgICAgICAgICAgam9icz1bcG9vbC5zdWJtaXQocnVuLGlpLGQpIGZvciBkLGlpIGluIGVudW1lcmF0ZSgoYmxvY2tbOmN1dF0sYmxvY2tbY3V0Ol0pKSBpZiBsZW4oaWkpXQogICAgICAgICAgICBmb3Igam9iIGluIGpvYnM6CiAgICAgICAgICAgICAgICBpbmRpY2VzLHZhbHVlPWpvYi5yZXN1bHQoKTsgZmVhdHVyZXMucmVzaGFwZSgtMSxfUkFEX1RPS0VOX0RJTSlbaW5kaWNlc109dmFsdWUKICAgIHJldHVybiBmZWF0dXJlcyx0b2tlbl9tYXNrLmFzdHlwZShfcmFkX25wLmZsb2F0MzIpCgoKQF9yYWRfdG9yY2guaW5mZXJlbmNlX21vZGUoKQpkZWYgX3JhZF9wcmVkaWN0X2hlYWQoaGVhZCwgZmVhdHVyZXMsIG1hc2tzLCBkZXZpY2UsIGJhdGNoPTY0KToKICAgIHByZWRpY3Rpb25zID0gW10KICAgIGZvciBzdGFydCBpbiByYW5nZSgwLCBsZW4oZmVhdHVyZXMpLCBiYXRjaCk6CiAgICAgICAgaW1hZ2UgPSBfcmFkX3RvcmNoLmZyb21fbnVtcHkoZmVhdHVyZXNbc3RhcnQ6c3RhcnQgKyBiYXRjaF0pLnRvKGRldmljZSkKICAgICAgICBtYXNrID0gX3JhZF90b3JjaC5mcm9tX251bXB5KG1hc2tzW3N0YXJ0OnN0YXJ0ICsgYmF0Y2hdKS50byhkZXZpY2UpCiAgICAgICAgZGVmIF9oZWFkX2ZvcndhcmQoX2FtcCk6CiAgICAgICAgICAgIGFtcCA9IChfcmFkX3RvcmNoLmF1dG9jYXN0KCdjdWRhJywgZW5hYmxlZD1fYW1wKQogICAgICAgICAgICAgICAgICAgaWYgZGV2aWNlLnR5cGUgPT0gJ2N1ZGEnIGVsc2UgX3JhZF9jb250ZXh0bGliLm51bGxjb250ZXh0KCkpCiAgICAgICAgICAgIHdpdGggYW1wOgogICAgICAgICAgICAgICAgcmV0dXJuIF9yYWRfdG9yY2guc2lnbW9pZChoZWFkKGltYWdlIGlmIF9hbXAgZWxzZSBpbWFnZS5mbG9hdCgpLCBtYXNrKSkuZmxvYXQoKS5jcHUoKQogICAgICAgIGJsb2NrID0gX2hlYWRfZm9yd2FyZChUcnVlKQogICAgICAgIGlmIG5vdCBib29sKF9yYWRfdG9yY2guaXNmaW5pdGUoYmxvY2spLmFsbCgpKToKICAgICAgICAgICAgcnNuYV9ldmVudCgncmFkX2hlYWRfZnAxNl9ub25maW5pdGVfcmV0cnlfZnAzMicsIHJvd3M9aW50KGxlbihpbWFnZSkpKQogICAgICAgICAgICBibG9jayA9IF9oZWFkX2ZvcndhcmQoRmFsc2UpCiAgICAgICAgaWYgbm90IGJvb2woX3JhZF90b3JjaC5pc2Zpbml0ZShibG9jaykuYWxsKCkpOgogICAgICAgICAgICByc25hX2V2ZW50KCdyYWRfaGVhZF9ub25maW5pdGVfbmV1dHJhbCcsIHJvd3M9aW50KCh+X3JhZF90b3JjaC5pc2Zpbml0ZShibG9jaykpLmFueShkaW09MSkuc3VtKCkpKQogICAgICAgICAgICBibG9jayA9IF9yYWRfdG9yY2gud2hlcmUoX3JhZF90b3JjaC5pc2Zpbml0ZShibG9jayksIGJsb2NrLCBfcmFkX3RvcmNoLmZ1bGxfbGlrZShibG9jaywgMC41KSkKICAgICAgICBwcmVkaWN0aW9ucy5hcHBlbmQoYmxvY2spCiAgICByZXR1cm4gX3JhZF90b3JjaC5jYXQocHJlZGljdGlvbnMpLm51bXB5KCkKCgpkZWYgX3JhZF9yYW5rX2NvbHVtbnModmFsdWVzKToKICAgIHJldHVybiByc25hX3JhbmtwY3QodmFsdWVzKQoKCmRlZiBfcmFkX3ZhbGlkYXRlKGZyYW1lLCBleHBlY3RlZF9pZHMpOgogICAgaWYgZnJhbWUuY29sdW1ucy50b2xpc3QoKSAhPSBbJ1N0dWR5SW5zdGFuY2VVSUQnLCAqX1JBRF9MQUJFTFNdOgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcignVjM2IHN1Ym1pc3Npb24gc2NoZW1hIGRyaWZ0JykKICAgIGlkcyA9IGZyYW1lWydTdHVkeUluc3RhbmNlVUlEJ10uYXN0eXBlKHN0cikudG9saXN0KCkKICAgIGlmIGlkcyAhPSBsaXN0KG1hcChzdHIsIGV4cGVjdGVkX2lkcykpIG9yIGxlbihpZHMpICE9IGxlbihzZXQoaWRzKSk6CiAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKCdWMzYgc3VibWlzc2lvbiBzdHVkeSBpZGVudGl0eS9vcmRlciBkcmlmdCcpCiAgICB2YWx1ZXMgPSBmcmFtZVtfUkFEX0xBQkVMU10udG9fbnVtcHkoX3JhZF9ucC5mbG9hdDY0KQogICAgaWYgbm90IF9yYWRfbnAuaXNmaW5pdGUodmFsdWVzKS5hbGwoKSBvciB2YWx1ZXMubWluKCkgPCAwIG9yIHZhbHVlcy5tYXgoKSA+IDE6CiAgICAgICAgcnNuYV9ldmVudCgncmFkX2ludmFsaWRfdmFsdWVzX3JlcGFpcmVkJywgbm9uZmluaXRlPWludCgofl9yYWRfbnAuaXNmaW5pdGUodmFsdWVzKSkuc3VtKCkpLCBvdXRfb2ZfcmFuZ2U9aW50KCgodmFsdWVzIDwgMCkgfCAodmFsdWVzID4gMSkpLnN1bSgpKSkKICAgICAgICBmcmFtZVtfUkFEX0xBQkVMU10gPSBfcmFkX25wLmNsaXAoX3JhZF9ucC53aGVyZShfcmFkX25wLmlzZmluaXRlKHZhbHVlcyksIHZhbHVlcywgMC41KSwgMCwgMSkKCgpjbGFzcyBfUmFkTGF5b3V0UHJlZmV0Y2g6CiAgICAiIiJPbmUgQ1BVIGxheW91dCBidWlsZGVyIG92ZXJsYXBzIEdQVSB3b3JrOyBHUFUgaGVhZCBzaGFwZXMgYXJlIG1vZGVsLW93bmVkLiIiIgogICAgZGVmIF9faW5pdF9fKHNlbGYsIHByZXBhcmUsIGNvbmZpZ3VyYXRpb25zLCByZWxlYXNlKToKICAgICAgICBmcm9tIGNvbmN1cnJlbnQuZnV0dXJlcyBpbXBvcnQgVGhyZWFkUG9vbEV4ZWN1dG9yCiAgICAgICAgc2VsZi5wcmVwYXJlID0gcHJlcGFyZQogICAgICAgIHNlbGYuY29uZmlndXJhdGlvbnMgPSBsaXN0KGNvbmZpZ3VyYXRpb25zKQogICAgICAgIHNlbGYucmVsZWFzZSA9IHJlbGVhc2UKICAgICAgICBzZWxmLmV4ZWN1dG9yID0gVGhyZWFkUG9vbEV4ZWN1dG9yKG1heF93b3JrZXJzPTEpCiAgICAgICAgc2VsZi5wZW5kaW5nID0gTm9uZQogICAgICAgIHNlbGYubmV4dF9pbmRleCA9IDAKICAgICAgICBzZWxmLmFjcXVpcmVkID0gTm9uZQoKICAgIGRlZiBfc3VibWl0KHNlbGYpOgogICAgICAgIGlmIHNlbGYubmV4dF9pbmRleCA8IGxlbihzZWxmLmNvbmZpZ3VyYXRpb25zKToKICAgICAgICAgICAgY29uZmlnID0gc2VsZi5jb25maWd1cmF0aW9uc1tzZWxmLm5leHRfaW5kZXhdCiAgICAgICAgICAgIHNlbGYubmV4dF9pbmRleCArPSAxCiAgICAgICAgICAgIHNlbGYucGVuZGluZyA9IHNlbGYuZXhlY3V0b3Iuc3VibWl0KHNlbGYucHJlcGFyZSwgKmNvbmZpZykKCiAgICBkZWYgX19lbnRlcl9fKHNlbGYpOgogICAgICAgIHNlbGYuX3N1Ym1pdCgpCiAgICAgICAgcmV0dXJuIHNlbGYKCiAgICBkZWYgdGFrZShzZWxmKToKICAgICAgICBpZiBzZWxmLmFjcXVpcmVkIGlzIG5vdCBOb25lOgogICAgICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoJ3JlbGVhc2UgdGhlIGN1cnJlbnQgUmFkIGxheW91dCBiZWZvcmUgdGFraW5nIGFub3RoZXInKQogICAgICAgIGlmIHNlbGYucGVuZGluZyBpcyBOb25lOgogICAgICAgICAgICByYWlzZSBTdG9wSXRlcmF0aW9uKCdhbGwgUmFkIGxheW91dHMgY29uc3VtZWQnKQogICAgICAgIGZ1dHVyZSwgc2VsZi5wZW5kaW5nID0gc2VsZi5wZW5kaW5nLCBOb25lCiAgICAgICAgdHJ5OgogICAgICAgICAgICBzZWxmLmFjcXVpcmVkID0gZnV0dXJlLnJlc3VsdCgpCiAgICAgICAgZXhjZXB0IEJhc2VFeGNlcHRpb246CiAgICAgICAgICAgIHNlbGYuZXhlY3V0b3Iuc2h1dGRvd24od2FpdD1UcnVlLCBjYW5jZWxfZnV0dXJlcz1UcnVlKQogICAgICAgICAgICByYWlzZQogICAgICAgIHNlbGYuX3N1Ym1pdCgpCiAgICAgICAgcmV0dXJuIHNlbGYuYWNxdWlyZWQKCiAgICBkZWYgcmVsZWFzZV9jdXJyZW50KHNlbGYpOgogICAgICAgIGlmIHNlbGYuYWNxdWlyZWQgaXMgbm90IE5vbmU6CiAgICAgICAgICAgIHNlbGYucmVsZWFzZShzZWxmLmFjcXVpcmVkWzFdKSAgIyAoc3R1ZHkgSURzLCBwaXhlbHMsIG1hc2tzKQogICAgICAgICAgICBzZWxmLmFjcXVpcmVkID0gTm9uZQoKICAgIGRlZiBfX2V4aXRfXyhzZWxmLCBleGNfdHlwZSwgZXhjLCB0Yik6CiAgICAgICAgc2VsZi5yZWxlYXNlX2N1cnJlbnQoKQogICAgICAgIHBlbmRpbmcsIHNlbGYucGVuZGluZyA9IHNlbGYucGVuZGluZywgTm9uZQogICAgICAgIHNlbGYuZXhlY3V0b3Iuc2h1dGRvd24od2FpdD1UcnVlLCBjYW5jZWxfZnV0dXJlcz1UcnVlKQogICAgICAgIGlmIHBlbmRpbmcgaXMgbm90IE5vbmUgYW5kIG5vdCBwZW5kaW5nLmNhbmNlbGxlZCgpOgogICAgICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICBleHRyYSA9IHBlbmRpbmcucmVzdWx0KCkKICAgICAgICAgICAgICAgIHNlbGYucmVsZWFzZShleHRyYVsxXSkKICAgICAgICAgICAgZXhjZXB0IEJhc2VFeGNlcHRpb246CiAgICAgICAgICAgICAgICBpZiBleGNfdHlwZSBpcyBOb25lOgogICAgICAgICAgICAgICAgICAgIHJhaXNlCiAgICAgICAgcmV0dXJuIEZhbHNlCgoKZGVmIF9yYWRfbWFpbigpOgogICAgIiIiQlRLRCBSYWQvY2FsaWJyYXRvciBncmFwaCB1bmNoYW5nZWQ7IGRpYWdub3N0aWMtb25seSB0d2luIHdvcmsgcmVtb3ZlZC4KCiAgICBCVEtEJ3MgRTEzLW9uLUUxMSBzZWNvbmQgaW5wdXQgaXMgaW50ZW50aW9uYWwgaW4gaXRzIGZpdHRlZCBjYWxpYnJhdG9yLgogICAgS2VlcCB0aGF0IHBhaXIgdG9nZXRoZXI7IGRvIG5vdCBjbGFpbSBzdGFuZGFsb25lIEUxMyB0cmFpbi9zZXJ2ZSBwYXJpdHkuCiAgICAiIiIKICAgIHN0YXJ0ZWQ9X3JhZF90aW1lLnRpbWUoKTt3b3JrPV9SYWRQYXRoKCcva2FnZ2xlL3dvcmtpbmcnKTtwcmltYXJ5PXdvcmsvJ19waXBlbGluZV9zdGFnZS5jc3YnCiAgICB0ZXN0PV9yYWRfcGQucmVhZF9jc3YoUk9PVC8ndGVzdC5jc3YnLGR0eXBlPXsnU3R1ZHlJbnN0YW5jZVVJRCc6c3RyfSkKICAgIGlkcz10ZXN0LlN0dWR5SW5zdGFuY2VVSUQudG9saXN0KCkKICAgIGJhc2VsaW5lPXJzbmFfZnJhbWUoX3JhZF9wZC5yZWFkX2NzdihwcmltYXJ5LGR0eXBlPXsnU3R1ZHlJbnN0YW5jZVVJRCc6c3RyfSksaWRzLF9SQURfTEFCRUxTLCdSYWQgcGFyZW50JykKICAgIGRldmljZT1fcmFkX3RvcmNoLmRldmljZSgnY3VkYTowJykKICAgIGVuY29kZXIscmVmZXJlbmNlX2hlYWRzLGVuY29kZXJfcGF0aCxyZWZlcmVuY2VfaGVhZHNfcGF0aD1fcmFkX2xvYWRfbW9kZWxzKGRldmljZSkKICAgIHRzZXI9X3JhZF9wZC5yZWFkX2NzdihST09ULyd0ZXN0X3Nlcmllcy5jc3YnLGR0eXBlPXsnU3R1ZHlJbnN0YW5jZVVJRCc6c3RyLCdTZXJpZXNJbnN0YW5jZVVJRCc6c3RyfSkKICAgIHBsYW5lPWRpY3QoemlwKHRzZXIuU2VyaWVzSW5zdGFuY2VVSUQsdHNlci5BbmF0b21pY2FsX1BsYW5lKSkKICAgICMgT25lIGFubm90YXRpb24gcGFzcy4gQWxsIHRocmVlIGxheW91dHMgdXNlIHRoZSBpZGVudGljYWwgbGVnYWN5IGFubm90YXRpb24gcnVsZXMuCiAgICBoZWFkZXJzPWFubm90YXRlKHdhbGsoJ3Rlc3Rfc2VyaWVzJykpCiAgICBzaWRlX21hcD1sYXRfb2YoaGVhZGVycywncmFkIGNvbW1vbicpCiAgICBkZWYgcHJlcGFyZV9sYXlvdXQobGF5b3V0LGNyb3AsdGFnKToKICAgICAgICBnbG9iYWxzKCkudXBkYXRlKFNMT1RTPWxpc3QobGF5b3V0KSxOX1NMT1Q9bGVuKGxheW91dCksQ0FDSEVfU0xJQ0VTPTgsSU1HPTIyNCxDQUNIRV9JTUc9MjI0LENST1BfTU09ZmxvYXQoY3JvcCksUlVMRVM9ZGljdChSVUxFU19MRUdBQ1kpKQogICAgICAgIHZhbHVlPWJ1aWxkX2NhY2hlKHBpY2tfc2xvdHMoaGVhZGVycyxwbGFuZSkscGxhbmUsc2lkZV9tYXAsdGFnKQogICAgICAgIGlmIHZhbHVlWzBdIT1pZHM6CiAgICAgICAgICAgIHJzbmFfcmVsZWFzZV9waXhlbHModmFsdWVbMV0pCiAgICAgICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcignUmFkIGNhY2hlIG9yZGVyIGNoYW5nZWQnKQogICAgICAgIHJldHVybiB2YWx1ZQogICAgZGVmIGVuY29kZV9sYXlvdXQocHJlZmV0Y2gpOgogICAgICAgIHN0dWRpZXMscGl4ZWxzLG1hc2s9cHJlZmV0Y2gudGFrZSgpCiAgICAgICAgdHJ5OgogICAgICAgICAgICBmLHRtPV9yYWRfZW5jb2RlKGVuY29kZXIscGl4ZWxzLG1hc2ssZGV2aWNlKQogICAgICAgICAgICByZXR1cm4gZix0bSxpbnQodG0uc3VtKCkpCiAgICAgICAgZmluYWxseToKICAgICAgICAgICAgcHJlZmV0Y2gucmVsZWFzZV9jdXJyZW50KCkKICAgIGRlZiBwcmVkaWN0X2hlYWRzKGhlYWRzLGYsbWFzayx0YWcpOgogICAgICAgIGlmIGxlbihoZWFkcykhPTU6IHJhaXNlIFJ1bnRpbWVFcnJvcih0YWcrJzogZml2ZSBoZWFkcyByZXF1aXJlZCcpCiAgICAgICAgcHJvYnM9X3JhZF9ucC5zdGFjayhbX3JhZF9wcmVkaWN0X2hlYWQoaCxmLG1hc2ssZGV2aWNlKSBmb3IgaCBpbiBoZWFkc10pCiAgICAgICAgcnNuYV9maW5pdGUocHJvYnMsdGFnLHByb2JhYmlsaXR5PVRydWUpO3JzbmFfc2F2ZV9wcmVkaWN0aW9ucyh0YWcsaWRzLHByb2JzLF9SQURfTEFCRUxTKQogICAgICAgIHJldHVybiBwcm9icy5tZWFuKDApCiAgICBmaXJzdD1bKCdTQUdfRlMnLCdTYWdpdHRhbCcsTm9uZSxUcnVlKSwoJ0NPUl9GUycsJ0Nvcm9uYWwnLE5vbmUsVHJ1ZSksKCdBWF9GUycsJ0F4aWFsJyxOb25lLFRydWUpXQogICAgIyBFMTMgaGFzIGZvdXIgc2xvdHM7IGNvbnN0cnVjdCBpdHMgbW9kZWwgYmVmb3JlIHRoZSBwcm9kdWNlciBtdXRhdGVzIGdsb2JhbHMuCiAgICBnbG9iYWxzKCkudXBkYXRlKFNMT1RTPWxpc3QoX1JBRF9FMTNfU0xPVFMpLE5fU0xPVD00LENBQ0hFX1NMSUNFUz04KQogICAgZTEzX2hlYWRzLGUxM19wYXRoPV9yYWRfbG9hZF9lMTNfaGVhZHMoZGV2aWNlKQogICAgY29uZmlndXJhdGlvbnM9WyhmaXJzdCwxMDAwMC4sJ3JhZF9lMTAnKSwoX1JBRF9FMTNfU0xPVFMsMTMwLiwncmFkX2UxMycpLChfUkFEX0UxMV9TTE9UUywxMzAuLCdyYWRfbGVnYWN5X3NlY29uZCcpXQogICAgd2l0aCBfUmFkTGF5b3V0UHJlZmV0Y2gocHJlcGFyZV9sYXlvdXQsY29uZmlndXJhdGlvbnMscnNuYV9yZWxlYXNlX3BpeGVscykgYXMgcHJlZmV0Y2g6CiAgICAgICAgZixtLHRva2VuczA9ZW5jb2RlX2xheW91dChwcmVmZXRjaCkKICAgICAgICByZWZlcmVuY2VfcmFuaz1fcmFkX3JhbmtfY29sdW1ucyhwcmVkaWN0X2hlYWRzKHJlZmVyZW5jZV9oZWFkcyxmLG0sJ3JhZF9lMTBfcmF3JykpCiAgICAgICAgZGVsIGYsbSxyZWZlcmVuY2VfaGVhZHMKICAgICAgICBmLG0sdG9rZW5zMT1lbmNvZGVfbGF5b3V0KHByZWZldGNoKQogICAgICAgIGUxM19yYW5rPV9yYWRfcmFua19jb2x1bW5zKHByZWRpY3RfaGVhZHMoZTEzX2hlYWRzLGYsbSwncmFkX2UxM19yYXcnKSkKICAgICAgICByZWZlcmVuY2VfcmFuaz1fcmFkX3JhbmtfY29sdW1ucygoMS1fUkFEX0UxM19NRU1CRVJfV0VJR0hUKSpyZWZlcmVuY2VfcmFuaytfUkFEX0UxM19NRU1CRVJfV0VJR0hUKmUxM19yYW5rKQogICAgICAgIGRlbCBmLG0sZTEzX3JhbmsKICAgICAgICBiYXNlbGluZV9yYW5rPV9yYWRfcmFua19jb2x1bW5zKGJhc2VsaW5lW19SQURfTEFCRUxTXS50b19udW1weSgpKQogICAgICAgIGNhbmRpZGF0ZT1iYXNlbGluZS5jb3B5KCkKICAgICAgICBmb3IgaixsYWJlbCBpbiBlbnVtZXJhdGUoX1JBRF9MQUJFTFMpOgogICAgICAgICAgICBpZiBsYWJlbCBub3QgaW4gX1JBRF9FWENMVURFOgogICAgICAgICAgICAgICAgY2FuZGlkYXRlW2xhYmVsXT0oMS1fUkFEX0FMUEhBKSpiYXNlbGluZV9yYW5rWzosal0rX1JBRF9BTFBIQSpyZWZlcmVuY2VfcmFua1s6LGpdCiAgICAgICAgX3JhZF92YWxpZGF0ZShjYW5kaWRhdGUsaWRzKQogICAgICAgIGYsbSx0b2tlbnMyPWVuY29kZV9sYXlvdXQocHJlZmV0Y2gpCiAgICAgICAgc2Vjb25kX3Jhbms9X3JhZF9yYW5rX2NvbHVtbnMocHJlZGljdF9oZWFkcyhlMTNfaGVhZHMsZixtLCdyYWRfbGVnYWN5X3NlY29uZF9yYXcnKSkKICAgICAgICBkZWwgZixtLGUxM19oZWFkcwogICAgYnJhbmNoPWNhbmRpZGF0ZS5jb3B5KCkKICAgIGJyYW5jaFtfUkFEX0xBQkVMU109X3JhZF9yYW5rX2NvbHVtbnMoKDEtX1JBRF9WNDhfU0VDT05EX0FMUEhBKSpfcmFkX3JhbmtfY29sdW1ucyhjYW5kaWRhdGVbX1JBRF9MQUJFTFNdLnRvX251bXB5KCkpK19SQURfVjQ4X1NFQ09ORF9BTFBIQSpzZWNvbmRfcmFuaykKICAgIGNhbD1fcmFkX2pzb24ubG9hZHMoX3JhZF96bGliLmRlY29tcHJlc3MoX3JhZF9iNjQuYjY0ZGVjb2RlKF9SQURfQ0FMX1BBWUxPQUQpKS5kZWNvZGUoKSkKICAgIHByb3RvY29sPV9yYWRfcGQuRGF0YUZyYW1lKGluZGV4PV9yYWRfcGQuSW5kZXgoaWRzLG5hbWU9J1N0dWR5SW5zdGFuY2VVSUQnKSkKICAgIHByb3RvY29sWyduX3NlcmllcyddPXRzZXIuZ3JvdXBieSgnU3R1ZHlJbnN0YW5jZVVJRCcpLnNpemUoKS5yZWluZGV4KGlkcykuZmlsbG5hKDApCiAgICBmb3IgcGwgaW4gKCdTYWdpdHRhbCcsJ0Nvcm9uYWwnLCdBeGlhbCcpOgogICAgICAgIHBhcnQ9dHNlclt0c2VyLkFuYXRvbWljYWxfUGxhbmUuYXN0eXBlKHN0cik9PXBsXQogICAgICAgIHByb3RvY29sW2Ynbl97cGxbOjNdfSddPXBhcnQuZ3JvdXBieSgnU3R1ZHlJbnN0YW5jZVVJRCcpLnNpemUoKS5yZWluZGV4KGlkcykuZmlsbG5hKDApCiAgICBmb3IgZmxhZyBpbiAoJ0ZhdF9TdXBwcmVzc2lvbicsJ0ZsdWlkX1NlbnNpdGl2ZScpOgogICAgICAgIG1hcmtlZD10c2VyW19yYWRfcGQudG9fbnVtZXJpYyh0c2VyW2ZsYWddLGVycm9ycz0nY29lcmNlJykuZmlsbG5hKDApPjBdCiAgICAgICAgcHJvdG9jb2xbZmxhZ1s6M11dPW1hcmtlZC5ncm91cGJ5KCdTdHVkeUluc3RhbmNlVUlEJykuc2l6ZSgpLnJlaW5kZXgoaWRzKS5maWxsbmEoMCkKICAgICAgICBmb3IgcGwgaW4gKCdTYWdpdHRhbCcsJ0Nvcm9uYWwnLCdBeGlhbCcpOgogICAgICAgICAgICBwYXJ0PW1hcmtlZFttYXJrZWQuQW5hdG9taWNhbF9QbGFuZS5hc3R5cGUoc3RyKT09cGxdCiAgICAgICAgICAgIHByb3RvY29sW2Yne2ZsYWdbOjNdfV97cGxbOjNdfSddPXBhcnQuZ3JvdXBieSgnU3R1ZHlJbnN0YW5jZVVJRCcpLnNpemUoKS5yZWluZGV4KGlkcykuZmlsbG5hKDApCiAgICBpZiBsaXN0KHByb3RvY29sLmNvbHVtbnMpIT1saXN0KGNhbFsncHJvdG9jb2xfY29sdW1ucyddKTogcmFpc2UgUnVudGltZUVycm9yKCdjYWxpYnJhdG9yIHByb3RvY29sIGRyaWZ0JykKICAgIG1lYW49KGJhc2VsaW5lX3JhbmsrcmVmZXJlbmNlX3Jhbmsrc2Vjb25kX3JhbmspLzMuCiAgICBibG9ja3M9W2Jhc2VsaW5lX3JhbmsscmVmZXJlbmNlX3Jhbmssc2Vjb25kX3JhbmsscmVmZXJlbmNlX3JhbmstYmFzZWxpbmVfcmFuayxzZWNvbmRfcmFuay1iYXNlbGluZV9yYW5rLG1lYW5dCiAgICBmb3IgZ3JvdXAgaW4gY2FsWydncm91cHMnXToKICAgICAgICBibG9ja3MuYXBwZW5kKG1lYW5bOixbX1JBRF9MQUJFTFMuaW5kZXgodCkgZm9yIHQgaW4gZ3JvdXBdXS5tZWFuKGF4aXM9MSxrZWVwZGltcz1UcnVlKSkKICAgIGJsb2Nrcy5hcHBlbmQocHJvdG9jb2wudG9fbnVtcHkoX3JhZF9ucC5mbG9hdDY0KSkKICAgIHg9X3JhZF9ucC5jb25jYXRlbmF0ZShibG9ja3MsYXhpcz0xKTsgY2VudGVyPV9yYWRfbnAuYXNhcnJheShjYWxbJ21lYW4nXSk7c3ByZWFkPV9yYWRfbnAuYXNhcnJheShjYWxbJ3NjYWxlJ10pO2NvZWY9X3JhZF9ucC5hc2FycmF5KGNhbFsnY29lZiddKTtiaWFzPV9yYWRfbnAuYXNhcnJheShjYWxbJ2ludGVyY2VwdCddKQogICAgaWYgeC5zaGFwZVsxXSE9Y29lZi5zaGFwZVsxXSBvciBub3QgYm9vbCgoc3ByZWFkPjApLmFsbCgpKTogcmFpc2UgUnVudGltZUVycm9yKCdjYWxpYnJhdG9yIHNoYXBlL3NjYWxlIGRyaWZ0JykKICAgIGFkanVzdGVkPV9yYWRfcmFua19jb2x1bW5zKCgoeC1jZW50ZXIpL3NwcmVhZClAY29lZi5UK2JpYXMpCiAgICB2YWx1ZXM9YnJhbmNoW19SQURfTEFCRUxTXS50b19udW1weShfcmFkX25wLmZsb2F0NjQpLmNvcHkoKQogICAgZm9yIGosbGFiZWwgaW4gZW51bWVyYXRlKF9SQURfTEFCRUxTKToKICAgICAgICBpZiBsYWJlbCBpbiBzZXQoY2FsWydnYXRlJ10pOiB2YWx1ZXNbOixqXT0uNip2YWx1ZXNbOixqXSsuNCphZGp1c3RlZFs6LGpdCiAgICBmaW5hbD1icmFuY2guY29weSgpO2ZpbmFsW19SQURfTEFCRUxTXT1fcmFkX3JhbmtfY29sdW1ucyh2YWx1ZXMpCiAgICBfcmFkX3ZhbGlkYXRlKGZpbmFsLGlkcyk7cnNuYV9zYXZlX3ByZWRpY3Rpb25zKCdidGtkX3BhcmVudF9hZnRlcl9yYWQnLGlkcyxmaW5hbFtfUkFEX0xBQkVMU10udG9fbnVtcHkoKSxfUkFEX0xBQkVMUykKICAgIHRtcD1wcmltYXJ5LndpdGhfc3VmZml4KCcudG1wJyk7ZmluYWwudG9fY3N2KHRtcCxpbmRleD1GYWxzZSk7X3JhZF9vcy5yZXBsYWNlKHRtcCxwcmltYXJ5KQogICAgZ2xvYmFscygpWydWMThfQ0FMSUJSQVRPUl9BUFBMSUVEJ109VHJ1ZQogICAgcnNuYV9qc29uKHdvcmsvJ3Y1NThfcmFkX3JlY2VpcHQuanNvbicseydzdGF0dXMnOidDT01QTEVURScsJ2dyYXBoJzonQlRLRCBsZWdhY3kgUmFkIHBsdXMgaXRzIG1hdGNoZWQgY2FsaWJyYXRvciBwcmVzZXJ2ZWQnLCdyZWZlcmVuY2VfaGVhZHNfc2hhMjU2JzpfUkFEX1JFRkVSRU5DRV9IRUFEU19TSEEyNTYsJ2UxM19zaGEyNTYnOl9SQURfRTEzX0hFQURTX1NIQTI1NiwnZW5jb2Rlcl9zaGEyNTYnOl9SQURfRU5DT0RFUl9TSEEyNTYsJ2xlZ2FjeV9jcm9zc19sYXlvdXRfcGFzc19wcmVzZXJ2ZWQnOlRydWUsJ3N0YW5kYWxvbmVfY3Jvc3NfbGF5b3V0X3RyYWluX3Bhcml0eV9jbGFpbWVkJzpGYWxzZSwndW51c2VkX2FsdGVybmF0ZV9mYW1pbHlfcmVtb3ZlZCc6VHJ1ZSwnZW5jb2Rlcl9yZXBsaWNhcyc6MiwnbGF5b3V0X2NwdV9wcmVmZXRjaCc6VHJ1ZSwnbW9kZWxfb3duZWRfaGVhZF9zaGFwZSc6VHJ1ZSwndG9rZW5zJzpbdG9rZW5zMCx0b2tlbnMxLHRva2VuczJdLCdzZWNvbmRzJzpfcmFkX3RpbWUudGltZSgpLXN0YXJ0ZWR9KQogICAgZGVsIGVuY29kZXI7X3JhZF9nYy5jb2xsZWN0KCkKICAgIGZvciBkIGluIHJhbmdlKDIpOgogICAgICAgIHdpdGggX3JhZF90b3JjaC5jdWRhLmRldmljZShkKTogX3JhZF90b3JjaC5jdWRhLmVtcHR5X2NhY2hlKCkKICAgIF9yYWRfbG9nKCdCVEtEIHJlZmVyZW5jZSBSYWQvY2FsaWJyYXRvciBjb21wbGV0ZTsgdW51c2VkIGFsdGVybmF0ZS1oZWFkIGJyYW5jaCByZW1vdmVkJykKCgpyc25hX3BoYXNlKCdyYWRpbWFnZW5ldCcsICdTVEFSVCcpCl9yYWRfbWFpbigpCnJzbmFfcGhhc2UoJ3JhZGltYWdlbmV0JywgJ0NPTVBMRVRFJykK", "22": "aWYgZ2xvYmFscygpLmdldCgnX0RJTk9WMl9NQVRDSEVEX01FTUJFUlMnKSAhPSAyMDoKICAgIHJhaXNlIFJ1bnRpbWVFcnJvcignRElOT3YyIDIwLzIwIGZpbmdlcnByaW50IGdhdGUgZGlkIG5vdCBwYXNzJykKaW1wb3J0IGdjIGFzIF9rZV9nYwppbXBvcnQgb3MgYXMgX2tlX29zCmltcG9ydCB0aW1lIGFzIF9rZV90aW1lCmZyb20gY29uY3VycmVudC5mdXR1cmVzIGltcG9ydCBUaHJlYWRQb29sRXhlY3V0b3IgYXMgX0tlVGhyZWFkUG9vbAppbXBvcnQgbnVtcHkgYXMgX2tlX25wCmltcG9ydCBwYW5kYXMgYXMgX2tlX3BkCmZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aCBhcyBfS2VQYXRoCgpfa2VfcHJpbWFyeSA9IF9LZVBhdGgoJy9rYWdnbGUvd29ya2luZy9fcGlwZWxpbmVfc3RhZ2UuY3N2JykKX2tlX291cnMgPSBfa2VfcGQucmVhZF9jc3YoX2tlX3ByaW1hcnksIGR0eXBlPXsnU3R1ZHlJbnN0YW5jZVVJRCc6IHN0cn0pCl9LRV9MQUIgPSBbYyBmb3IgYyBpbiBfa2Vfb3Vycy5jb2x1bW5zIGlmIGMgIT0gJ1N0dWR5SW5zdGFuY2VVSUQnXQoKX0tFX1NSQyA9IHInJycKaW1wb3J0IG9zLCBnbG9iLCB0aW1lLCBnYywgaGFzaGxpYgpvcy5lbnZpcm9uLnNldGRlZmF1bHQoJ0hGX0hVQl9PRkZMSU5FJywgJzEnKQpvcy5lbnZpcm9uLnNldGRlZmF1bHQoJ1RSQU5TRk9STUVSU19PRkZMSU5FJywgJzEnKQpvcy5lbnZpcm9uLnNldGRlZmF1bHQoJ0hGX0hVQl9ESVNBQkxFX1RFTEVNRVRSWScsICcxJykKaW1wb3J0IG51bXB5IGFzIG5wCmltcG9ydCB0b3JjaCwgdG9yY2gubm4gYXMgbm4sIHRvcmNoLm5uLmZ1bmN0aW9uYWwgYXMgRgppbXBvcnQgdGltbQp0b3JjaC5iYWNrZW5kcy5jdWRubi5iZW5jaG1hcmsgPSBGYWxzZSAgIyByYWdnZWQgOTQtd2luZG93IGNodW5rczogbm8gcmVwZWF0ZWQgYXV0b3R1bmluZwp0b3JjaC5iYWNrZW5kcy5jdWRhLm1hdG11bC5hbGxvd190ZjMyID0gVHJ1ZQpJTUcgPSAzMzYKQ1JPUF9NTSA9IDE0MC4wClNQQU5fTE8sIFNQQU5fSEkgPSAwLjAyLCAwLjk4ClNMT1RTID0gWygiU2FnaXR0YWwiLCAxLCAxOCksICgiU2FnaXR0YWwiLCAwLCAxNCksCiAgICAgICAgICgiQ29yb25hbCIsIDEsIDEyKSwgKCJDb3JvbmFsIiwgMCwgOCksICgiQXhpYWwiLCAtMSwgMTIpXQpNQVhTID0gc3VtKHNsb3RbMl0gZm9yIHNsb3QgaW4gU0xPVFMpCktfRVZBTCA9IDYyCk5PUk0gPSAiaW1hZ2VuZXQiCkxBQiA9IFsiQUNMIiwgIk1DTCIsICJNZWRpYWwgTWVuaXNjdXMiLCAiTGF0ZXJhbCBNZW5pc2N1cyIsICJNZWRpYWwgT0EiLAogICAgICAgIkxhdGVyYWwgT0EiLCAiUEYgT0EiLCAiRWZmdXNpb24iLCAiU3lub3ZpdGlzIiwgIkJha2VyJ3MiLAogICAgICAgIkNvbnR1c2lvbiIsICJGcmFjdHVyZSJdCl9NRUFOID0gdG9yY2gudGVuc29yKFswLjQ4NSwgMC40NTYsIDAuNDA2XSkudmlldygzLCAxLCAxKQpfU1REID0gdG9yY2gudGVuc29yKFswLjIyOSwgMC4yMjQsIDAuMjI1XSkudmlldygzLCAxLCAxKQpfU0xPVFM2NCA9IFsoIlNhZ2l0dGFsIiwgMSwgMTgpLCAoIlNhZ2l0dGFsIiwgMCwgMTQpLAogICAgICAgICAgICAoIkNvcm9uYWwiLCAxLCAxMiksICgiQ29yb25hbCIsIDAsIDgpLCAoIkF4aWFsIiwgLTEsIDEyKV0KX1NMT1RTNDQgPSBbKCJTYWdpdHRhbCIsIDEsIDEyKSwgKCJTYWdpdHRhbCIsIDAsIDEwKSwKICAgICAgICAgICAgKCJDb3JvbmFsIiwgMSwgOCksICgiQ29yb25hbCIsIDAsIDYpLCAoIkF4aWFsIiwgLTEsIDgpXQpBUk1TID0gWwogICAgeyJuYW1lIjogIm1heHNwYW4tdjUiLCAiZmlsZSI6ICJyYXB0b3JfZnRfY29hdG5ldF92NV9mdWxsX3N3YS5wdCIsCiAgICAgImFyY2giOiAiY29hdG5ldF9ybWxwXzJfcndfMzg0LnN3X2luMTJrX2Z0X2luMWsiLCAicmVzIjogMzg0LAogICAgICJpbWciOiAzMzYsICJzbG90cyI6IF9TTE9UUzY0LCAic3BhbiI6ICgwLjAyLCAwLjk4KSwgImtfZXZhbCI6IDYyLAogICAgICJyZXZlcnNlIjogRmFsc2UsICJ3IjogMC42MH0sCiAgICB7Im5hbWUiOiAibmF0aXZlMzg0ZGVuc2UtdjEwIiwgImZpbGUiOiAicmFwdG9yX2Z0X2NvYXRuZXRfdjEwX2Z1bGwucHQiLAogICAgICJhcmNoIjogImNvYXRuZXRfcm1scF8yX3J3XzM4NC5zd19pbjEya19mdF9pbjFrIiwgInJlcyI6IDM4NCwKICAgICAiaW1nIjogMzg0LCAic2xvdHMiOiBfU0xPVFM2NCwgInNwYW4iOiAoMC4wMiwgMC45OCksICJrX2V2YWwiOiA2MiwKICAgICAicmV2ZXJzZSI6IEZhbHNlLCAidyI6IDAuMTB9LAogICAgeyJuYW1lIjogIm1heHNwYW4tdjUtcmV2ZXJzZSIsICJmaWxlIjogInJhcHRvcl9mdF9jb2F0bmV0X3Y1X2Z1bGxfc3dhLnB0IiwKICAgICAiYXJjaCI6ICJjb2F0bmV0X3JtbHBfMl9yd18zODQuc3dfaW4xMmtfZnRfaW4xayIsICJyZXMiOiAzODQsCiAgICAgImltZyI6IDMzNiwgInNsb3RzIjogX1NMT1RTNjQsICJzcGFuIjogKDAuMDIsIDAuOTgpLCAia19ldmFsIjogNjIsCiAgICAgInJldmVyc2UiOiBUcnVlLCAidyI6IDAuMTB9LAogICAgeyJuYW1lIjogIm5hdGl2ZTM4NC12OCIsICJmaWxlIjogInJhcHRvcl9mdF9jb2F0bmV0X3Y4X2Z1bGxfc3dhLnB0IiwKICAgICAiYXJjaCI6ICJjb2F0bmV0X3JtbHBfMl9yd18zODQuc3dfaW4xMmtfZnRfaW4xayIsICJyZXMiOiAzODQsCiAgICAgImltZyI6IDM4NCwgInNsb3RzIjogX1NMT1RTNDQsICJzcGFuIjogKDAuMDYsIDAuOTQpLCAia19ldmFsIjogNDIsCiAgICAgInJldmVyc2UiOiBGYWxzZSwgInciOiAwLjIwfSwKXQoKZGVmIGJ1aWxkX2JhY2tib25lKGFyY2gsIHByZXRyYWluZWQ9RmFsc2UpOgogICAgaHlicmlkID0gYXJjaC5zdGFydHN3aXRoKCgnbWF4dml0JywgJ21heHh2aXQnLCAnY29hdG5ldCcsICdjb2F0XycsICdjb252bmV4dCcpKQogICAgaXNfdml0ID0gbm90IGh5YnJpZCBhbmQgYW55KChrIGluIGFyY2ggZm9yIGsgaW4gKCd2aXQnLCAnZGVpdCcsICdkaW5vdjInLCAnZXZhJywgJ2JlaXQnKSkpCiAgICBrdyA9IGRpY3QocHJldHJhaW5lZD1wcmV0cmFpbmVkLCBudW1fY2xhc3Nlcz0wLCBpbl9jaGFucz0zKQogICAgaWYgaXNfdml0OgogICAgICAgIGt3LnVwZGF0ZShnbG9iYWxfcG9vbD0ndG9rZW4nLCBkeW5hbWljX2ltZ19zaXplPVRydWUpCiAgICBlbHNlOgogICAgICAgIGt3LnVwZGF0ZShnbG9iYWxfcG9vbD0nYXZnJykKICAgIHJldHVybiB0aW1tLmNyZWF0ZV9tb2RlbChhcmNoLCAqKmt3KQoKY2xhc3MgUmFwdG9yQ2xhc3NpZmllcihubi5Nb2R1bGUpOgoKICAgIGRlZiBfX2luaXRfXyhzZWxmLCBiYWNrYm9uZSwgRl9kaW09NzY4LCBuPTEyLCBkcm9wPTAuMik6CiAgICAgICAgc3VwZXIoKS5fX2luaXRfXygpCiAgICAgICAgc2VsZi5iYWNrYm9uZSA9IGJhY2tib25lCiAgICAgICAgc2VsZi5ub3JtID0gbm4uTGF5ZXJOb3JtKEZfZGltKQogICAgICAgIHNlbGYuYXR0ID0gbm4uU2VxdWVudGlhbChubi5MaW5lYXIoRl9kaW0sIDI1NiksIG5uLlRhbmgoKSwgbm4uRHJvcG91dChkcm9wKSwgbm4uTGluZWFyKDI1NiwgbikpCiAgICAgICAgc2VsZi5jbHNXID0gbm4uUGFyYW1ldGVyKHRvcmNoLnplcm9zKG4sIEZfZGltKSkKICAgICAgICBzZWxmLmNsc2IgPSBubi5QYXJhbWV0ZXIodG9yY2guemVyb3MobikpCiAgICAgICAgbm4uaW5pdC50cnVuY19ub3JtYWxfKHNlbGYuY2xzVywgc3RkPTAuMDIpCiAgICAgICAgc2VsZi5uID0gbgoKICAgIGRlZiBlbmNvZGUoc2VsZiwgeCk6CiAgICAgICAgQiwgSyA9IHguc2hhcGVbOjJdCiAgICAgICAgZiA9IHNlbGYuYmFja2JvbmUoeC5mbGF0dGVuKDAsIDEpKQogICAgICAgIHJldHVybiBmLnZpZXcoQiwgSywgLTEpCgogICAgZGVmIGhlYWQoc2VsZiwgZmVhdHMpOgogICAgICAgIGggPSBzZWxmLm5vcm0oZmVhdHMpCiAgICAgICAgYSA9IHNlbGYuYXR0KGgpCiAgICAgICAgYSA9IHRvcmNoLnNvZnRtYXgoYSwgZGltPTEpCiAgICAgICAgcG9vbGVkID0gdG9yY2guZWluc3VtKCdia24sYmtmLT5ibmYnLCBhLCBoKQogICAgICAgIGxvZ2l0cyA9IChwb29sZWQgKiBzZWxmLmNsc1cpLnN1bSgtMSkgKyBzZWxmLmNsc2IKICAgICAgICByZXR1cm4gbG9naXRzCgogICAgZGVmIGZvcndhcmQoc2VsZiwgeCk6CiAgICAgICAgcmV0dXJuIHNlbGYuaGVhZChzZWxmLmVuY29kZSh4KSkKCmRlZiBsb2FkX21vZGVsKHB0X3BhdGgsIGFyY2hfZGVmYXVsdCwgcmVzX2RlZmF1bHQsIGRldmljZSwgbmdwdT0xKToKICAgIGNrID0gdG9yY2gubG9hZChwdF9wYXRoLCBtYXBfbG9jYXRpb249J2NwdScsIHdlaWdodHNfb25seT1GYWxzZSkKICAgIGFyY2ggPSBjay5nZXQoJ2FyY2gnLCBhcmNoX2RlZmF1bHQpCiAgICBpZiBhcmNoICE9IGFyY2hfZGVmYXVsdDoKICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoZidSYXB0b3IgYXJjaGl0ZWN0dXJlIGRyaWZ0OiB7YXJjaH0gIT0ge2FyY2hfZGVmYXVsdH0nKQogICAgY2tfcmVzID0gaW50KGNrLmdldCgncmVzJywgcmVzX2RlZmF1bHQpKQogICAgYmIgPSBidWlsZF9iYWNrYm9uZShhcmNoLCBwcmV0cmFpbmVkPUZhbHNlKQogICAgbW9kZWwgPSBSYXB0b3JDbGFzc2lmaWVyKGJiLCBGX2RpbT1iYi5udW1fZmVhdHVyZXMpCiAgICByc25hX3N0cmljdF9sb2FkKG1vZGVsLGNrWydtb2RlbCddLHN0cihwdF9wYXRoKSkKICAgIHJzbmFfZXZlbnQoJ3JhcHRvcl9jaGVja3BvaW50JyxmaWxlPXN0cihwdF9wYXRoKSxzaGEyNTY9cnNuYV9zaGEocHRfcGF0aCksYXJjaD1hcmNoLHJlcz1ja19yZXMpCiAgICBtb2RlbC5ldmFsKCkudG8oZGV2aWNlKQogICAgZGVsIGNrCiAgICBnYy5jb2xsZWN0KCkKICAgIHJldHVybiAobW9kZWwsIGNrX3JlcykKCgoKZGVmIF9ldmFsX2NlbnRlcnMobWFzaywgRCwgayk6CiAgICB2YWxpZCA9IG5wLndoZXJlKG1hc2sgPiAwKVswXQogICAgaWYgbGVuKHZhbGlkKSA8IDM6CiAgICAgICAgdmFsaWQgPSBucC5hcmFuZ2UobWluKDMsIEQpKQogICAgbG8sIGhpID0gKGludCh2YWxpZC5taW4oKSksIGludCh2YWxpZC5tYXgoKSkpCiAgICBjcyA9IFtjIGZvciBjIGluIHJhbmdlKGxvICsgMSwgaGkpIGlmIGMgLSAxID49IGxvIGFuZCBjICsgMSA8PSBoaV0KICAgIGlmIG5vdCBjczoKICAgICAgICBjcyA9IFttYXgoMSwgbWluKChsbyArIGhpKSAvLyAyLCBEIC0gMikpXQogICAgaWR4ID0gbnAubGluc3BhY2UoMCwgbGVuKGNzKSAtIDEsIGspLnJvdW5kKCkuYXN0eXBlKGludCkKICAgIHJldHVybiBbY3NbaV0gZm9yIGkgaW4gaWR4XQoKZGVmIGV2YWxfd2luZG93cyh2b2wsIG1hc2ssIGssIHJlcywgbm9ybT1OT1JNKToKICAgICIiIlJlc2l6ZSBlYWNoIHNlbGVjdGVkIHNvdXJjZSBwbGFuZSBvbmNlLCB0aGVuIGdhdGhlciB0aGUgdW5jaGFuZ2VkIFJHQiB0cmlwbGV0cy4iIiIKICAgIHZvbHVtZSA9IG5wLmFzYXJyYXkodm9sKQogICAgZGVwdGggPSBpbnQodm9sdW1lLnNoYXBlWzBdKQogICAgY2VudGVycyA9IG5wLmFzYXJyYXkoX2V2YWxfY2VudGVycyhtYXNrLCBkZXB0aCwgayksIGR0eXBlPW5wLmludDY0KQogICAgY2VudGVycyA9IG5wLmNsaXAoY2VudGVycywgMSwgZGVwdGggLSAyKQogICAgaWYgdHVwbGUodm9sdW1lLnNoYXBlWy0yOl0pID09IChyZXMsIHJlcyk6CiAgICAgICAgIyBOYXRpdmUzODQgbmVlZHMgbm8gaW50ZXJwb2xhdGlvbjsgdGhlIGV4aXN0aW5nIHNtYWxsIHBlci10cmlwbGV0IGNvcHkKICAgICAgICAjIGF2b2lkcyBhIHNsb3dlciBsYXJnZSBhZHZhbmNlZC1pbmRleCBnYXRoZXIgYW5kIHByZXNlcnZlcyBpdHMgZXhhY3QgbWF0aC4KICAgICAgICB3aW5zID0gbnAuZW1wdHkoKGxlbihjZW50ZXJzKSwgMywgcmVzLCByZXMpLCBucC5mbG9hdDMyKQogICAgICAgIGZvciBqLCBjZW50ZXIgaW4gZW51bWVyYXRlKGNlbnRlcnMpOgogICAgICAgICAgICB3aW5zW2pdID0gbnAuc3RhY2soW3ZvbHVtZVtjZW50ZXIgLSAxXSwgdm9sdW1lW2NlbnRlcl0sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdm9sdW1lW2NlbnRlciArIDFdXSwgYXhpcz0wKS5hc3R5cGUobnAuZmxvYXQzMikgLyAyNTUuMAogICAgICAgIHggPSB0b3JjaC5mcm9tX251bXB5KHdpbnMpCiAgICAgICAgaWYgbm9ybSA9PSAnaW1hZ2VuZXQnOgogICAgICAgICAgICB4ID0gKHggLSBfTUVBTikgLyBfU1RECiAgICAgICAgcmV0dXJuIHgKICAgIHRyaXBsZXRzID0gY2VudGVyc1s6LCBOb25lXSArIG5wLmFzYXJyYXkoWy0xLCAwLCAxXSwgZHR5cGU9bnAuaW50NjQpCiAgICAjIFVzZSBvbmx5IHBsYW5lcyByZWZlcmVuY2VkIGJ5IHRoZSBvcmlnaW5hbCBldmFsLWNlbnRlciByZWNpcGUuCiAgICB1bmlxdWUsIGludmVyc2UgPSBucC51bmlxdWUodHJpcGxldHMucmVzaGFwZSgtMSksIHJldHVybl9pbnZlcnNlPVRydWUpCiAgICBzb3VyY2UgPSB2b2x1bWVbdW5pcXVlXS5hc3R5cGUobnAuZmxvYXQzMikgLyAyNTUuMAogICAgcGxhbmVzID0gdG9yY2guZnJvbV9udW1weShzb3VyY2UpCiAgICBpZiB0dXBsZShwbGFuZXMuc2hhcGVbLTI6XSkgIT0gKHJlcywgcmVzKToKICAgICAgICByZXNpemVkID0gdG9yY2guZW1wdHkoKGxlbihwbGFuZXMpLCByZXMsIHJlcyksIGR0eXBlPXRvcmNoLmZsb2F0MzIpCiAgICAgICAgZm9yIHN0YXJ0IGluIHJhbmdlKDAsIGxlbihwbGFuZXMpLCAxNik6CiAgICAgICAgICAgIHJlc2l6ZWRbc3RhcnQ6c3RhcnQrMTZdID0gRi5pbnRlcnBvbGF0ZSgKICAgICAgICAgICAgICAgIHBsYW5lc1tzdGFydDpzdGFydCsxNiwgTm9uZV0sIHNpemU9KHJlcywgcmVzKSwKICAgICAgICAgICAgICAgIG1vZGU9J2JpbGluZWFyJywgYWxpZ25fY29ybmVycz1GYWxzZSlbOiwgMF0KICAgICAgICBwbGFuZXMgPSByZXNpemVkCiAgICB4ID0gcGxhbmVzW3RvcmNoLmZyb21fbnVtcHkoaW52ZXJzZS5yZXNoYXBlKC0xLCAzKSldCiAgICBpZiBub3JtID09ICdpbWFnZW5ldCc6CiAgICAgICAgeCA9ICh4IC0gX01FQU4pIC8gX1NURAogICAgcmV0dXJuIHgKCgoKCgpkZWYgcmFua3BjdCh4KToKICAgIHJldHVybiByc25hX3JhbmswMSh4KQoKZGVmIF9tYWtlX3JlYWRlcigpOgogICAgaW1wb3J0IHB5ZGljb20sIGN2MgogICAgZnJvbSBweWRpY29tLnBpeGVsX2RhdGFfaGFuZGxlcnMudXRpbCBpbXBvcnQgYXBwbHlfbW9kYWxpdHlfbHV0CgogICAgZGVmIG9yZGVyX2FuZF9tZXRhKHNkaXIpOgogICAgICAgIGZzID0gZ2xvYi5nbG9iKHNkaXIgKyAnLyouZGNtJykKICAgICAgICByZWNzID0gW10KICAgICAgICBwc19saXN0ID0gW10KICAgICAgICBmb3IgZiBpbiBmczoKICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgaCA9IHB5ZGljb20uZGNtcmVhZChmLCBzdG9wX2JlZm9yZV9waXhlbHM9VHJ1ZSkKICAgICAgICAgICAgICAgIGlvcCA9IGdldGF0dHIoaCwgJ0ltYWdlT3JpZW50YXRpb25QYXRpZW50JywgTm9uZSkKICAgICAgICAgICAgICAgIGlwcCA9IGdldGF0dHIoaCwgJ0ltYWdlUG9zaXRpb25QYXRpZW50JywgTm9uZSkKICAgICAgICAgICAgICAgIGlmIGlvcCBpcyBub3QgTm9uZSBhbmQgaXBwIGlzIG5vdCBOb25lIGFuZCAobGVuKGlvcCkgPT0gNik6CiAgICAgICAgICAgICAgICAgICAgciA9IG5wLmFycmF5KGlvcFs6M10sIGZsb2F0KQogICAgICAgICAgICAgICAgICAgIGMgPSBucC5hcnJheShpb3BbMzpdLCBmbG9hdCkKICAgICAgICAgICAgICAgICAgICBuID0gbnAuY3Jvc3MociwgYykKICAgICAgICAgICAgICAgICAgICBwb3MgPSBmbG9hdChucC5kb3QobnAuYXJyYXkoaXBwLCBmbG9hdCksIG4pKQogICAgICAgICAgICAgICAgZWxzZToKICAgICAgICAgICAgICAgICAgICBpbnN0YW5jZSA9IGdldGF0dHIoaCwgJ0luc3RhbmNlTnVtYmVyJywgTm9uZSkKICAgICAgICAgICAgICAgICAgICBpZiBpbnN0YW5jZSBpcyBOb25lOgogICAgICAgICAgICAgICAgICAgICAgICByc25hX2V2ZW50KCdyYXB0b3Jfb3JkZXJfaW5kZXhfZmFsbGJhY2snLCBmaWxlPXN0cihmKSk7IGluc3RhbmNlID0gbGVuKHJlY3MpCiAgICAgICAgICAgICAgICAgICAgcG9zID0gZmxvYXQoaW5zdGFuY2UpCiAgICAgICAgICAgICAgICBpZiBub3QgbnAuaXNmaW5pdGUocG9zKToKICAgICAgICAgICAgICAgICAgICByc25hX2V2ZW50KCdyYXB0b3Jfb3JkZXJfbm9uZmluaXRlX2ZhbGxiYWNrJywgZmlsZT1zdHIoZikpOyBwb3MgPSBmbG9hdChsZW4ocmVjcykpCiAgICAgICAgICAgICAgICBwcyA9IGdldGF0dHIoaCwgJ1BpeGVsU3BhY2luZycsIE5vbmUpCiAgICAgICAgICAgICAgICBwcyA9IGZsb2F0KHBzWzBdKSBpZiBwcyBpcyBub3QgTm9uZSBlbHNlIDAuNQogICAgICAgICAgICAgICAgcHNfbGlzdC5hcHBlbmQocHMpCiAgICAgICAgICAgICAgICByZWNzLmFwcGVuZCgocG9zLCBmLCBwcykpCiAgICAgICAgICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZXhjOgogICAgICAgICAgICAgICAgcnNuYV9ldmVudCgncmFwdG9yX29yZGVyX3VucmVhZGFibGVfc2tpcHBlZCcsIGZpbGU9c3RyKGYpLCBlcnJvcj1mJ3t0eXBlKGV4YykuX19uYW1lX199OiB7ZXhjfScpOyBjb250aW51ZQogICAgICAgIHJlY3Muc29ydChrZXk9bGFtYmRhIHg6ICh4WzBdLCB4WzFdKSkKICAgICAgICBtZWRfcHMgPSBmbG9hdChucC5tZWRpYW4ocHNfbGlzdCkpIGlmIHBzX2xpc3QgZWxzZSAwLjUKICAgICAgICByZXR1cm4gKFsoZiwgcHMpIGZvciBfLCBmLCBwcyBpbiByZWNzXSwgbWVkX3BzKQoKICAgIGRlZiByZWFkX3B4KGYpOgogICAgICAgIGQgPSBweWRpY29tLmRjbXJlYWQoZikKICAgICAgICBhID0gYXBwbHlfbW9kYWxpdHlfbHV0KGQucGl4ZWxfYXJyYXksIGQpLmFzdHlwZShucC5mbG9hdDMyKQogICAgICAgIGlmIHN0cihnZXRhdHRyKGQsICdQaG90b21ldHJpY0ludGVycHJldGF0aW9uJywgJycpKSA9PSAnTU9OT0NIUk9NRTEnOgogICAgICAgICAgICBhID0gYS5tYXgoKSAtIGEKICAgICAgICByZXR1cm4gYQoKICAgIGRlZiBtbV9jcm9wX3Jlc2l6ZShhLCBwcyk6CiAgICAgICAgaCwgdyA9IGEuc2hhcGUKICAgICAgICBjcHggPSBpbnQocm91bmQoQ1JPUF9NTSAvIG1heChwcywgMC4wMDEpKSkKICAgICAgICBjcHggPSBtaW4oY3B4LCBtaW4oaCwgdykpCiAgICAgICAgeTAgPSAoaCAtIGNweCkgLy8gMgogICAgICAgIHgwID0gKHcgLSBjcHgpIC8vIDIKICAgICAgICBhID0gYVt5MDp5MCArIGNweCwgeDA6eDAgKyBjcHhdCiAgICAgICAgcmV0dXJuIGN2Mi5yZXNpemUoYSwgKElNRywgSU1HKSwgaW50ZXJwb2xhdGlvbj1jdjIuSU5URVJfQVJFQSkKICAgIHJldHVybiAob3JkZXJfYW5kX21ldGEsIHJlYWRfcHgsIG1tX2Nyb3BfcmVzaXplKQoKZGVmIF9waWNrX3Nlcmllc19mb3Jfc2xvdChyb3dzLCBwbGFuZSwgZmx1aWQsIHVzZWQpOgogICAgIiIiQW4gdW5rbm93biBwcm90b2NvbCBmbGFnIGlzIG5vdCBmYWxzZSBhbmQgbXVzdCBub3QgcmVhY2ggaW50KE5hTikuIiIiCiAgICBkZWYgZmxhZyh2YWx1ZSk6CiAgICAgICAgaWYgdmFsdWUgaXMgTm9uZToKICAgICAgICAgICAgcmV0dXJuIE5vbmUKICAgICAgICBpZiBpc2luc3RhbmNlKHZhbHVlLCBzdHIpIGFuZCB2YWx1ZS5zdHJpcCgpLmxvd2VyKCkgaW4gKCcnLCAnbmFuJywgJ25vbmUnLCAnPG5hPicsICd1bmtub3duJyk6CiAgICAgICAgICAgIHJldHVybiBOb25lCiAgICAgICAgdHJ5OgogICAgICAgICAgICBudW1iZXIgPSBmbG9hdCh2YWx1ZSkKICAgICAgICBleGNlcHQgKFR5cGVFcnJvciwgVmFsdWVFcnJvcik6CiAgICAgICAgICAgIHRleHQgPSBzdHIodmFsdWUpLnN0cmlwKCkubG93ZXIoKQogICAgICAgICAgICBpZiB0ZXh0IGluICgndHJ1ZScsICd5ZXMnLCAneScsICd0Jyk6CiAgICAgICAgICAgICAgICByZXR1cm4gMQogICAgICAgICAgICBpZiB0ZXh0IGluICgnZmFsc2UnLCAnbm8nLCAnbicsICdmJyk6CiAgICAgICAgICAgICAgICByZXR1cm4gMAogICAgICAgICAgICByZXR1cm4gTm9uZSAgIyB1bmtub3duIGNvZGU6IG5laXRoZXIgZmFsc2Ugbm9yIGZhdGFsOyB0aGUgcGxhbmUtbGV2ZWwgZmFsbGJhY2sgYXBwbGllcwogICAgICAgIGlmIG5vdCBucC5pc2Zpbml0ZShudW1iZXIpOgogICAgICAgICAgICByZXR1cm4gTm9uZQogICAgICAgIGlmIG51bWJlciBub3QgaW4gKDAuMCwgMS4wKToKICAgICAgICAgICAgcmV0dXJuIE5vbmUKICAgICAgICByZXR1cm4gaW50KG51bWJlcikKICAgIGNhbmRpZGF0ZXMgPSBbciBmb3IgciBpbiByb3dzIGlmIHJbJ0FuYXRvbWljYWxfUGxhbmUnXSA9PSBwbGFuZQogICAgICAgICAgICAgICAgICBhbmQgclsnU2VyaWVzSW5zdGFuY2VVSUQnXSBub3QgaW4gdXNlZF0KICAgIGlmIGZsdWlkIGluICgwLCAxKToKICAgICAgICBwcmVmZXJyZWQgPSBbciBmb3IgciBpbiBjYW5kaWRhdGVzIGlmIGZsYWcoci5nZXQoJ0ZsdWlkX1NlbnNpdGl2ZScpKSA9PSBmbHVpZF0KICAgICAgICBpZiBwcmVmZXJyZWQ6CiAgICAgICAgICAgIHJldHVybiBwcmVmZXJyZWRbMF0KICAgIHJldHVybiBjYW5kaWRhdGVzWzBdIGlmIGNhbmRpZGF0ZXMgZWxzZSBOb25lCgoKCmRlZiBmaW5kX3Rlc3Rfcm9vdCgpOgogICAgZnJvbSBwYXRobGliIGltcG9ydCBQYXRoCiAgICBmb3IgdmFsdWUgaW4gW29zLmVudmlyb24uZ2V0KCdSU05BX0NPTVBfUk9PVCcsICcnKSwKICAgICAgICAgICAgICAgICAgJy9rYWdnbGUvaW5wdXQvY29tcGV0aXRpb25zL3JzbmEta25lZS1hYm5vcm1hbGl0eS1kZXRlY3Rpb24nLAogICAgICAgICAgICAgICAgICAnL2thZ2dsZS9pbnB1dC9yc25hLWtuZWUtYWJub3JtYWxpdHktZGV0ZWN0aW9uJ106CiAgICAgICAgaWYgdmFsdWUgYW5kIChQYXRoKHZhbHVlKS8ndGVzdC5jc3YnKS5pc19maWxlKCk6CiAgICAgICAgICAgIHJldHVybiB2YWx1ZQogICAgcmFpc2UgRmlsZU5vdEZvdW5kRXJyb3IoJ2V4cGxpY2l0IGNvbXBldGl0aW9uIHJvb3QgYWJzZW50JykKCmRlZiBmaW5kX3dlaWdodF9maWxlKGZuYW1lKToKICAgIHJldHVybiBzdHIoX2Fzc2V0X2ZpbmRfYXNzZXQoZm5hbWUpKQonJycKX0tFX05TID0geydfX25hbWVfXyc6ICdfa2VfcmFwdG9yJywgJ19hc3NldF9maW5kX2Fzc2V0JzogX2Fzc2V0X2ZpbmRfYXNzZXQsICdyc25hX3JhbmswMSc6cnNuYV9yYW5rMDEsICdyc25hX3N0cmljdF9sb2FkJzpyc25hX3N0cmljdF9sb2FkLCAncnNuYV9ldmVudCc6cnNuYV9ldmVudCwgJ3JzbmFfc2hhJzpyc25hX3NoYX0KZXhlYyhjb21waWxlKF9LRV9TUkMsICc8cmFwdG9yPicsICdleGVjJyksIF9LRV9OUykKZnJvbSBjb25jdXJyZW50LmZ1dHVyZXMgaW1wb3J0IEZ1dHVyZSBhcyBfS2VGdXR1cmUKaW1wb3J0IHRocmVhZGluZyBhcyBfa2VfdGhyZWFkaW5nCgpfa2VfYmFzZV9tYWtlX3JlYWRlciA9IF9LRV9OU1snX21ha2VfcmVhZGVyJ10KX2tlX29yZGVyX2NhY2hlID0ge30KZnJvbSBjb2xsZWN0aW9ucyBpbXBvcnQgT3JkZXJlZERpY3QgYXMgX0tlT3JkZXJlZERpY3QKX2tlX3BpeGVsX2NhY2hlID0gX0tlT3JkZXJlZERpY3QoKQpfa2Vfb3JkZXJfY2FjaGVfbG9jayA9IF9rZV90aHJlYWRpbmcuTG9jaygpCgoKZGVmIF9rZV9tYWtlX2NhY2hlZF9yZWFkZXIoKToKICAgICIiIlNoYXJlIGltbXV0YWJsZSBESUNPTSBvcmRlcmluZyBtZXRhZGF0YSBhY3Jvc3MgUmFwdG9yIHZpZXdzLiIiIgogICAgYmFzZV9vcmRlciwgcmVhZF9waXhlbCwgY3JvcF9yZXNpemUgPSBfa2VfYmFzZV9tYWtlX3JlYWRlcigpCgogICAgZGVmIGNhY2hlZF9vcmRlcihzZXJpZXNfZGlyKToKICAgICAgICBrZXkgPSBzdHIoc2VyaWVzX2RpcikKICAgICAgICBvd25lciA9IEZhbHNlCiAgICAgICAgd2l0aCBfa2Vfb3JkZXJfY2FjaGVfbG9jazoKICAgICAgICAgICAgZnV0dXJlID0gX2tlX29yZGVyX2NhY2hlLmdldChrZXkpCiAgICAgICAgICAgIGlmIGZ1dHVyZSBpcyBOb25lOgogICAgICAgICAgICAgICAgZnV0dXJlID0gX0tlRnV0dXJlKCkKICAgICAgICAgICAgICAgIF9rZV9vcmRlcl9jYWNoZVtrZXldID0gZnV0dXJlCiAgICAgICAgICAgICAgICBvd25lciA9IFRydWUKICAgICAgICBpZiBvd25lcjoKICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgZnV0dXJlLnNldF9yZXN1bHQoYmFzZV9vcmRlcihzZXJpZXNfZGlyKSkKICAgICAgICAgICAgZXhjZXB0IEJhc2VFeGNlcHRpb24gYXMgZXJyb3I6CiAgICAgICAgICAgICAgICBmdXR1cmUuc2V0X2V4Y2VwdGlvbihlcnJvcikKICAgICAgICAgICAgICAgIHdpdGggX2tlX29yZGVyX2NhY2hlX2xvY2s6CiAgICAgICAgICAgICAgICAgICAgX2tlX29yZGVyX2NhY2hlLnBvcChrZXksIE5vbmUpCiAgICAgICAgICAgICAgICByYWlzZQogICAgICAgIHJldHVybiBmdXR1cmUucmVzdWx0KCkKCiAgICBkZWYgY2FjaGVkX3BpeGVsKHBhdGgpOgogICAgICAgIGdsb2JhbCBfa2VfcGl4ZWxfY2FjaGVfYnl0ZXMKICAgICAgICBmcm9tIGNvbGxlY3Rpb25zIGltcG9ydCBPcmRlcmVkRGljdAogICAgICAgIGtleT1zdHIocGF0aCkKICAgICAgICB3aXRoIF9rZV9vcmRlcl9jYWNoZV9sb2NrOgogICAgICAgICAgICBnb3Q9X2tlX3BpeGVsX2NhY2hlLmdldChrZXkpCiAgICAgICAgICAgIGlmIGdvdCBpcyBub3QgTm9uZToKICAgICAgICAgICAgICAgIF9rZV9waXhlbF9jYWNoZS5tb3ZlX3RvX2VuZChrZXkpO3JldHVybiBnb3QKICAgICAgICB2YWx1ZT1yZWFkX3BpeGVsKHBhdGgpCiAgICAgICAgaWYgdmFsdWUubmRpbSE9MiBvciBub3QgX2tlX25wLmlzZmluaXRlKHZhbHVlKS5hbGwoKToKICAgICAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKGYnUmFwdG9yIGludmFsaWQgcGl4ZWxzIHtwYXRofScpCiAgICAgICAgdmFsdWUuc2V0ZmxhZ3Mod3JpdGU9RmFsc2UpCiAgICAgICAgd2l0aCBfa2Vfb3JkZXJfY2FjaGVfbG9jazoKICAgICAgICAgICAgcHJldmlvdXMgPSBfa2VfcGl4ZWxfY2FjaGUucG9wKGtleSwgTm9uZSkKICAgICAgICAgICAgaWYgcHJldmlvdXMgaXMgbm90IE5vbmU6CiAgICAgICAgICAgICAgICBfa2VfcGl4ZWxfY2FjaGVfYnl0ZXMgLT0gcHJldmlvdXMubmJ5dGVzCiAgICAgICAgICAgIF9rZV9waXhlbF9jYWNoZVtrZXldID0gdmFsdWUKICAgICAgICAgICAgX2tlX3BpeGVsX2NhY2hlX2J5dGVzICs9IHZhbHVlLm5ieXRlcwogICAgICAgICAgICB3aGlsZSBfa2VfcGl4ZWxfY2FjaGVfYnl0ZXMgPiAoMTkyIDw8IDIwKToKICAgICAgICAgICAgICAgIF8sIHJlbW92ZWQgPSBfa2VfcGl4ZWxfY2FjaGUucG9waXRlbShsYXN0PUZhbHNlKQogICAgICAgICAgICAgICAgX2tlX3BpeGVsX2NhY2hlX2J5dGVzIC09IHJlbW92ZWQubmJ5dGVzCiAgICAgICAgcmV0dXJuIHZhbHVlCiAgICByZXR1cm4gY2FjaGVkX29yZGVyLCBjYWNoZWRfcGl4ZWwsIGNyb3BfcmVzaXplCgoKX0tFX05TWydfbWFrZV9yZWFkZXInXSA9IF9rZV9tYWtlX2NhY2hlZF9yZWFkZXIKaWYgX2tlX29zLmVudmlyb24uZ2V0KCdSU05BX0NPTVBfUk9PVCcpOgogICAgX0tFX05TWydmaW5kX3Rlc3Rfcm9vdCddID0gbGFtYmRhOiBfa2Vfb3MuZW52aXJvblsnUlNOQV9DT01QX1JPT1QnXQoKCmRlZiBfa2VfcHJlcGFyZV93aW5kb3dzKGFybSwgc3R1ZHlfdWlkLCBzZXJpZXMsIHNlcmllc19yb290LCByZWFkZXIpOgoKCgoKCgoKICAgIG9yZGVyX2FuZF9tZXRhLCByZWFkX3B4LCBfID0gcmVhZGVyCiAgICBpbWFnZV9zaXplID0gaW50KGFybVsnaW1nJ10pCiAgICBzbG90cyA9IGxpc3QoYXJtWydzbG90cyddKQogICAgc3Bhbl9sbywgc3Bhbl9oaSA9IG1hcChmbG9hdCwgYXJtWydzcGFuJ10pCiAgICB1c2VkLCBwb29scyA9IHNldCgpLCBbXQogICAgcm93cyA9IHNlcmllcy5nZXQoc3R1ZHlfdWlkLCBbXSkKICAgIGZvciBwbGFuZSwgZmx1aWQsIGNvdW50IGluIHNsb3RzOgogICAgICAgIHJlY29yZCA9IF9LRV9OU1snX3BpY2tfc2VyaWVzX2Zvcl9zbG90J10ocm93cywgcGxhbmUsIGZsdWlkLCB1c2VkKQogICAgICAgIGlmIHJlY29yZCBpcyBOb25lOgogICAgICAgICAgICBwb29scy5hcHBlbmQoTm9uZSkKICAgICAgICAgICAgY29udGludWUKICAgICAgICB1c2VkLmFkZChyZWNvcmRbJ1Nlcmllc0luc3RhbmNlVUlEJ10pCiAgICAgICAgZmlsZXMsIG1lZGlhbl9zcGFjaW5nID0gb3JkZXJfYW5kX21ldGEoCiAgICAgICAgICAgIGYie3Nlcmllc19yb290fS97c3R1ZHlfdWlkfS97cmVjb3JkWydTZXJpZXNJbnN0YW5jZVVJRCddfSIpCiAgICAgICAgaWYgbm90IGZpbGVzOgogICAgICAgICAgICByc25hX2V2ZW50KCdyYXB0b3JfYWNxdWlzaXRpb25fZW1wdHknLCBzdHVkeT1zdHIoc3R1ZHlfdWlkKSwgc2VyaWVzPXN0cihyZWNvcmRbJ1Nlcmllc0luc3RhbmNlVUlEJ10pKTsgcG9vbHMuYXBwZW5kKE5vbmUpOyBjb250aW51ZQogICAgICAgIGxvID0gaW50KGxlbihmaWxlcykgKiBzcGFuX2xvKQogICAgICAgIGhpID0gbWF4KGludChsZW4oZmlsZXMpICogc3Bhbl9oaSkgLSAxLCBsbykKICAgICAgICBvcmlnaW5hbCA9IF9rZV9ucC5saW5zcGFjZShsbywgaGksIGNvdW50KS5yb3VuZCgpLmFzdHlwZShpbnQpCiAgICAgICAgcG9vbHMuYXBwZW5kKChmaWxlcywgbWVkaWFuX3NwYWNpbmcsIGxvLCBoaSwgb3JpZ2luYWwpKQogICAgY2FwYWNpdGllcyA9IFswIGlmIHAgaXMgTm9uZSBlbHNlIHBbM10gLSBwWzJdICsgMSBmb3IgcCBpbiBwb29sc10KICAgIHF1b3RhcyA9IF9kZW5zZV9hbGxvY2F0ZShjYXBhY2l0aWVzLCBbc1syXSBmb3IgcyBpbiBzbG90c10sIDk2KQogICAgdm9sdW1lLCBzb3VyY2VzID0gW10sIFtdCiAgICBpbXBvcnQgY3YyCiAgICBmb3IgcG9vbCwgcXVvdGEgaW4gemlwKHBvb2xzLCBxdW90YXMpOgogICAgICAgIGlmIHBvb2wgaXMgTm9uZSBvciBub3QgcXVvdGE6CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgZmlsZXMsIG1lZGlhbl9zcGFjaW5nLCBsbywgaGksIG9yaWdpbmFsID0gcG9vbAogICAgICAgIHBpY2tzID0gbG8gKyBfZGVuc2VfdW5pcXVlX2xpbnNwYWNlKGhpIC0gbG8gKyAxLCBpbnQocXVvdGEpKQogICAgICAgIGFycmF5cyA9IHt9CiAgICAgICAgIyBEZWNvZGUgdW5pb24gb25jZS4gRmFpbHVyZSBpcyBleHBsaWNpdCwgbm90IGFuIHVucmVwb3J0ZWQgMC41IHNjb3JlLgogICAgICAgIGZvciBpIGluIHNvcnRlZChzZXQob3JpZ2luYWwudG9saXN0KCkpIHwgc2V0KHBpY2tzLnRvbGlzdCgpKSk6CiAgICAgICAgICAgIGFycmF5c1tpXSA9IHJlYWRfcHgoZmlsZXNbaV1bMF0pCiAgICAgICAgYWxsX3BpeGVscyA9IF9rZV9ucC5jb25jYXRlbmF0ZShbYXJyYXlzW2ludChpKV0ucmF2ZWwoKSBmb3IgaSBpbiBvcmlnaW5hbF0pCiAgICAgICAgbG93LCBoaWdoID0gX2tlX25wLnBlcmNlbnRpbGUoYWxsX3BpeGVscywgWzIuMCwgOTguMF0pCiAgICAgICAgZm9yIGkgaW4gcGlja3M6CiAgICAgICAgICAgIHBhdGgsIHNwYWNpbmcgPSBmaWxlc1tpbnQoaSldCiAgICAgICAgICAgIGEgPSBhcnJheXNbaW50KGkpXQogICAgICAgICAgICBhID0gX2tlX25wLmNsaXAoKGEtbG93KS8oaGlnaC1sb3crMWUtNiksIDAsIDEpCiAgICAgICAgICAgIHNwYWNpbmcgPSBzcGFjaW5nIGlmIHNwYWNpbmcgPiAwIGVsc2UgbWVkaWFuX3NwYWNpbmcKICAgICAgICAgICAgaCwgdyA9IGEuc2hhcGUKICAgICAgICAgICAgYyA9IG1pbihpbnQocm91bmQoMTQwLjAvbWF4KHNwYWNpbmcsIC4wMDEpKSksIG1pbihoLCB3KSkKICAgICAgICAgICAgeSwgeCA9IChoLWMpLy8yLCAody1jKS8vMgogICAgICAgICAgICBhID0gY3YyLnJlc2l6ZShhW3k6eStjLCB4OngrY10sIChpbWFnZV9zaXplLGltYWdlX3NpemUpLAogICAgICAgICAgICAgICAgICAgICAgICAgICBpbnRlcnBvbGF0aW9uPWN2Mi5JTlRFUl9BUkVBKQogICAgICAgICAgICB2b2x1bWUuYXBwZW5kKChhKjI1NSkuYXN0eXBlKF9rZV9ucC51aW50OCkpCiAgICAgICAgICAgIHNvdXJjZXMuYXBwZW5kKHN0cihwYXRoKSkKICAgIGlmIGxlbihzb3VyY2VzKSAhPSBsZW4oc2V0KHNvdXJjZXMpKToKICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoJ2NhcGFjaXR5LWF3YXJlIHNhbXBsaW5nIHNlbGVjdGVkIGR1cGxpY2F0ZSBzb3VyY2UgZmlsZXMnKQogICAgaWYgbGVuKHZvbHVtZSkgPCAzOgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcihmJ3tzdHVkeV91aWR9OiBmZXdlciB0aGFuIHRocmVlIHVuaXF1ZSBzb3VyY2Ugc2xpY2VzJykKICAgIHZvbHVtZSA9IF9rZV9ucC5zdGFjayh2b2x1bWUpCiAgICAjIFByZXNlbmNlIG1lYW5zIGFuIGFjcXVpcmVkIHNvdXJjZSwgbm90IGFuIGludGVuc2l0eSB0ZXN0LiBBIGdlbnVpbmVseQogICAgIyBibGFjayBhY3F1aXJlZCBzbGljZSBpcyBub3QgaW50ZXJjaGFuZ2VhYmxlIHdpdGggcGFkZGluZy4KICAgIG1hc2sgPSBfa2VfbnAub25lcyhsZW4odm9sdW1lKSwgZHR5cGU9X2tlX25wLnVpbnQ4KQogICAgd2luZG93cyA9IF9LRV9OU1snZXZhbF93aW5kb3dzJ10oCiAgICAgICAgdm9sdW1lLCBtYXNrLCBrPWxlbih2b2x1bWUpLTIsIHJlcz1pbnQoYXJtWydyZXMnXSksIG5vcm09X0tFX05TWydOT1JNJ10pCiAgICBpZiBsZW4od2luZG93cykgIT0gbWluKDk2LCBzdW0oY2FwYWNpdGllcykpLTI6CiAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKCdjYXBhY2l0eS1hd2FyZSBzYW1wbGluZyBSYXB0b3IgY2FyZGluYWxpdHkgZHJpZnQnKQogICAgaW1wb3J0IGhhc2hsaWIsIGpzb24KICAgIHNpZ25hdHVyZSA9IGhhc2hsaWIuc2hhMjU2KGpzb24uZHVtcHMoc291cmNlcywgZW5zdXJlX2FzY2lpPUZhbHNlLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHNlcGFyYXRvcnM9KCcsJywgJzonKSkuZW5jb2RlKCkpLmhleGRpZ2VzdCgpCiAgICBhdWRpdCA9IHsnYXJtJzphcm1bJ25hbWUnXSwgJ3VpZCc6c3RyKHN0dWR5X3VpZCksCiAgICAgICAgICAgICAnY2FwYWNpdGllcyc6Y2FwYWNpdGllcywgJ3F1b3Rhcyc6cXVvdGFzLnRvbGlzdCgpLAogICAgICAgICAgICAgJ3NvdXJjZV9jb3VudCc6bGVuKHNvdXJjZXMpLCAnc291cmNlX2xpc3Rfc2hhMjU2JzpzaWduYXR1cmUsCiAgICAgICAgICAgICAnd2luZG93cyc6bGVuKHdpbmRvd3MpLCAnZHVwbGljYXRlX3NvdXJjZXMnOjB9CiAgICBpZGVudGl0eSA9IChhcm1bJ25hbWUnXSwgc3RyKHN0dWR5X3VpZCkpCiAgICB3aXRoIF9rZV9pbnB1dF9sb2NrOgogICAgICAgIGlmIGlkZW50aXR5IGluIF9rZV9pbnB1dF9pZHM6CiAgICAgICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcihmJ0R1cGxpY2F0ZSBSYXB0b3IgcHJlcGFyYXRpb246IHtpZGVudGl0eX0nKQogICAgICAgIF9rZV9pbnB1dF9pZHMuYWRkKGlkZW50aXR5KQogICAgICAgIHdpdGggX2tlX2lucHV0X2F1ZGl0X3BhdGgub3BlbignYScpIGFzIGhhbmRsZToKICAgICAgICAgICAgaGFuZGxlLndyaXRlKGpzb24uZHVtcHMoYXVkaXQsIHNvcnRfa2V5cz1UcnVlLCBhbGxvd19uYW49RmFsc2UpICsgJ1xuJykKICAgIHJldHVybiB3aW5kb3dzCgoKZGVmIF9rZV9pbmZlcl9pbnB1dChtb2RlbCwgeHdpbnMsIGRldmljZSk6CiAgICAiIiJCb3VuZCBiYWNrYm9uZSB3b3Jrc3BhY2U7IGtlZXAgYWxsIHdpbmRvdyBmZWF0dXJlcyBmb3Igb25lIGhlYWQgY2FsbC4iIiIKICAgIHRvcmNoID0gX0tFX05TWyd0b3JjaCddCiAgICBkZWYgX2ZvcndhcmQoX2FtcCk6CiAgICAgICAgd2l0aCB0b3JjaC5pbmZlcmVuY2VfbW9kZSgpLCB0b3JjaC5hdXRvY2FzdCgnY3VkYScsIGR0eXBlPXRvcmNoLmZsb2F0MTYsIGVuYWJsZWQ9X2FtcCk6CiAgICAgICAgICAgIGZlYXR1cmVzID0gW21vZGVsLmJhY2tib25lKHh3aW5zW2k6aSs4XS50byhkZXZpY2UpKQogICAgICAgICAgICAgICAgICAgICAgICBmb3IgaSBpbiByYW5nZSgwLGxlbih4d2lucyksOCldCiAgICAgICAgICAgIGZlYXRzID0gdG9yY2guY2F0KGZlYXR1cmVzLGRpbT0wKS51bnNxdWVlemUoMCkKICAgICAgICAgICAgcmV0dXJuIHRvcmNoLnNpZ21vaWQobW9kZWwuaGVhZChmZWF0cykuZmxvYXQoKSlbMF0uY3B1KCkubnVtcHkoKQogICAgcHJvYmFiaWxpdGllcyA9IF9mb3J3YXJkKFRydWUpCiAgICBpZiBub3QgX2tlX25wLmlzZmluaXRlKHByb2JhYmlsaXRpZXMpLmFsbCgpOgogICAgICAgIHJzbmFfZXZlbnQoJ3JhcHRvcl9mcDE2X25vbmZpbml0ZV9yZXRyeV9mcDMyJykKICAgICAgICBwcm9iYWJpbGl0aWVzID0gX2ZvcndhcmQoRmFsc2UpCiAgICBpZiBub3QgX2tlX25wLmlzZmluaXRlKHByb2JhYmlsaXRpZXMpLmFsbCgpOgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcignbm9uZmluaXRlIFJhcHRvciBwcmVkaWN0aW9uJykKICAgIHJldHVybiBwcm9iYWJpbGl0aWVzCgoKZGVmIF9rZV9wcmVmZXRjaGVkX3dpbmRvd3MoYXJtLCB0ZXN0X2lkcywgc2VyaWVzLCBzZXJpZXNfcm9vdCwgcmVhZGVyKToKICAgICIiIkJvdW5kIGhvc3QgbWVtb3J5IHdoaWxlIGRlY29kaW5nIG9uZSBzdHVkeSBhaGVhZCBvZiBpdHMgR1BVIGZvcndhcmQuIiIiCiAgICBkZXB0aCA9IGludChfa2Vfb3MuZW52aXJvbi5nZXQoJ1JTTkFfUkFQVE9SX1BSRUZFVENIJywgJzInKSkKICAgIGlmIG5vdCAxIDw9IGRlcHRoIDw9IDQ6IHJhaXNlIFZhbHVlRXJyb3IoJ1JhcHRvciBwcmVmZXRjaCBtdXN0IGJlIDEuLjQnKQogICAgd2l0aCBfS2VUaHJlYWRQb29sKG1heF93b3JrZXJzPTEpIGFzIGV4ZWN1dG9yOgogICAgICAgIHBlbmRpbmcgPSB7fQogICAgICAgIHN1Ym1pdF9hdCA9IDAKICAgICAgICB3aGlsZSBzdWJtaXRfYXQgPCBtaW4oZGVwdGgsIGxlbih0ZXN0X2lkcykpOgogICAgICAgICAgICBwZW5kaW5nW3N1Ym1pdF9hdF0gPSBleGVjdXRvci5zdWJtaXQoCiAgICAgICAgICAgICAgICBfa2VfcHJlcGFyZV93aW5kb3dzLAogICAgICAgICAgICAgICAgYXJtLAogICAgICAgICAgICAgICAgdGVzdF9pZHNbc3VibWl0X2F0XSwKICAgICAgICAgICAgICAgIHNlcmllcywKICAgICAgICAgICAgICAgIHNlcmllc19yb290LAogICAgICAgICAgICAgICAgcmVhZGVyLAogICAgICAgICAgICApCiAgICAgICAgICAgIHN1Ym1pdF9hdCArPSAxCiAgICAgICAgZm9yIHN0dWR5X2luZGV4LCBzdHVkeV91aWQgaW4gZW51bWVyYXRlKHRlc3RfaWRzKToKICAgICAgICAgICAgZnV0dXJlID0gcGVuZGluZy5wb3Aoc3R1ZHlfaW5kZXgpCiAgICAgICAgICAgIGlmIHN1Ym1pdF9hdCA8IGxlbih0ZXN0X2lkcyk6CiAgICAgICAgICAgICAgICBwZW5kaW5nW3N1Ym1pdF9hdF0gPSBleGVjdXRvci5zdWJtaXQoCiAgICAgICAgICAgICAgICAgICAgX2tlX3ByZXBhcmVfd2luZG93cywKICAgICAgICAgICAgICAgICAgICBhcm0sCiAgICAgICAgICAgICAgICAgICAgdGVzdF9pZHNbc3VibWl0X2F0XSwKICAgICAgICAgICAgICAgICAgICBzZXJpZXMsCiAgICAgICAgICAgICAgICAgICAgc2VyaWVzX3Jvb3QsCiAgICAgICAgICAgICAgICAgICAgcmVhZGVyLAogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgc3VibWl0X2F0ICs9IDEKICAgICAgICAgICAgeWllbGQgc3R1ZHlfaW5kZXgsIHN0dWR5X3VpZCwgZnV0dXJlCgoKZGVmIF9rZV9ydW5fcmFwdG9yX2FybXMoKToKICAgICIiIlJ1biB0aGUgZm91ciB1bmNoYW5nZWQgUmFwdG9yIGFybXMgYWNyb3NzIGV4YWN0bHkgdHdvIEdQVXMuIiIiCiAgICB0b3JjaCA9IF9LRV9OU1sndG9yY2gnXQogICAgaWYgbm90IHRvcmNoLmN1ZGEuaXNfYXZhaWxhYmxlKCkgb3IgdG9yY2guY3VkYS5kZXZpY2VfY291bnQoKSAhPSAyOgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcigKICAgICAgICAgICAgZiJvcHRpbWl6ZWQgUmFwdG9yIHJlcXVpcmVzIGV4YWN0bHkgdHdvIEdQVXMsIGdvdCB7dG9yY2guY3VkYS5kZXZpY2VfY291bnQoKX0iCiAgICAgICAgKQogICAgc3RhcnRlZCA9IF9rZV90aW1lLnRpbWUoKQogICAgcm9vdCA9IF9LRV9OU1snZmluZF90ZXN0X3Jvb3QnXSgpCiAgICBzZXJpZXNfcm9vdCA9IHJvb3QgKyAnL3Rlc3Rfc2VyaWVzJwogICAgaWYgbm90IF9rZV9vcy5wYXRoLmlzZGlyKHNlcmllc19yb290KToKICAgICAgICBzZXJpZXNfcm9vdCA9IHJvb3QgKyAnL3Rlc3RfaW1hZ2VzJwogICAgdGVzdCA9IF9rZV9wZC5yZWFkX2Nzdihyb290ICsgJy90ZXN0LmNzdicpCiAgICB0ZXN0WydTdHVkeUluc3RhbmNlVUlEJ10gPSB0ZXN0WydTdHVkeUluc3RhbmNlVUlEJ10uYXN0eXBlKHN0cikKICAgIHRlc3RfaWRzID0gdGVzdFsnU3R1ZHlJbnN0YW5jZVVJRCddLnRvbGlzdCgpCiAgICB0ZXN0X3NlcmllcyA9IF9rZV9wZC5yZWFkX2Nzdihyb290ICsgJy90ZXN0X3Nlcmllcy5jc3YnKQogICAgdGVzdF9zZXJpZXNbJ1N0dWR5SW5zdGFuY2VVSUQnXSA9IHRlc3Rfc2VyaWVzWydTdHVkeUluc3RhbmNlVUlEJ10uYXN0eXBlKHN0cikKICAgIHRlc3Rfc2VyaWVzWydTZXJpZXNJbnN0YW5jZVVJRCddID0gdGVzdF9zZXJpZXNbJ1Nlcmllc0luc3RhbmNlVUlEJ10uYXN0eXBlKHN0cikKICAgIHNlcmllcyA9IHsKICAgICAgICBrZXk6IGZyYW1lLnRvX2RpY3QoJ3JlY29yZHMnKQogICAgICAgIGZvciBrZXksIGZyYW1lIGluIHRlc3Rfc2VyaWVzLmdyb3VwYnkoJ1N0dWR5SW5zdGFuY2VVSUQnKQogICAgfQogICAgc2FtcGxlID0gcm9vdCArICcvc2FtcGxlX3N1Ym1pc3Npb24uY3N2JwogICAgY29sdW1ucyA9IFsnU3R1ZHlJbnN0YW5jZVVJRCcsICpfS0VfTlNbJ0xBQiddXQogICAgaWYgX2tlX29zLnBhdGguZXhpc3RzKHNhbXBsZSk6CiAgICAgICAgY29sdW1ucyA9IGxpc3QoX2tlX3BkLnJlYWRfY3N2KHNhbXBsZSwgbnJvd3M9MSkuY29sdW1ucykKICAgIGFybXMgPSBsaXN0KF9LRV9OU1snQVJNUyddKQogICAgb3V0cHV0cyA9IFsKICAgICAgICBfa2VfbnAuZnVsbCgobGVuKHRlc3RfaWRzKSwgbGVuKF9LRV9OU1snTEFCJ10pKSwgbnAubmFuLCBfa2VfbnAuZmxvYXQzMikKICAgICAgICBmb3IgXyBpbiBhcm1zCiAgICBdCiAgICBwcmludCgKICAgICAgICBmIltyYXB0b3ItZmFzdF0ge2xlbih0ZXN0X2lkcyl9IHN0dWRpZXM7IGJhbGFuY2VkIGFybSBncm91cHMgIgogICAgICAgIGYiY3VkYTowPVswLDJdLCBjdWRhOjE9WzEsM10iLAogICAgICAgIGZsdXNoPVRydWUsCiAgICApCgogICAgZGVmIHJ1bl9zaW5nbGUoYXJtX2luZGV4LCBkZXZpY2UpOgogICAgICAgIGFybSA9IGFybXNbYXJtX2luZGV4XQogICAgICAgIHJlYWRlciA9IF9LRV9OU1snX21ha2VfcmVhZGVyJ10oKQogICAgICAgIHdlaWdodF9wYXRoID0gX0tFX05TWydmaW5kX3dlaWdodF9maWxlJ10oYXJtWydmaWxlJ10pCiAgICAgICAgbW9kZWwsIHJlc29sdXRpb24gPSBfS0VfTlNbJ2xvYWRfbW9kZWwnXSgKICAgICAgICAgICAgd2VpZ2h0X3BhdGgsIGFybVsnYXJjaCddLCBhcm1bJ3JlcyddLCBkZXZpY2UKICAgICAgICApCiAgICAgICAgaWYgaW50KHJlc29sdXRpb24pICE9IGludChhcm1bJ3JlcyddKToKICAgICAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKAogICAgICAgICAgICAgICAgZiJ7YXJtWyduYW1lJ119IGNoZWNrcG9pbnQgcmVzb2x1dGlvbiB7cmVzb2x1dGlvbn0gIT0ge2FybVsncmVzJ119IgogICAgICAgICAgICApCiAgICAgICAgZm9yIHN0dWR5X2luZGV4LCBzdHVkeV91aWQsIGZ1dHVyZSBpbiBfa2VfcHJlZmV0Y2hlZF93aW5kb3dzKAogICAgICAgICAgICBhcm0sIHRlc3RfaWRzLCBzZXJpZXMsIHNlcmllc19yb290LCByZWFkZXIKICAgICAgICApOgogICAgICAgICAgICByc25hX2RlYWRsaW5lKCdSYXB0b3IgZnVsbC1jb2hvcnQgaW5mZXJlbmNlJykKICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgd2luZG93cyA9IGZ1dHVyZS5yZXN1bHQoKQogICAgICAgICAgICAgICAgb3V0cHV0c1thcm1faW5kZXhdW3N0dWR5X2luZGV4XSA9IF9LRV9OU1snaW5mZXJfcHJvYnMnXSgKICAgICAgICAgICAgICAgICAgICBtb2RlbCwgd2luZG93cywgZGV2aWNlCiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICBkZWwgd2luZG93cwogICAgICAgICAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGVycm9yOgogICAgICAgICAgICAgICAgcnNuYV9ldmVudCgncmFwdG9yX3N0dWR5X2ZhaWxlZCcsIGFybT1zdHIoYXJtWyduYW1lJ10pLCBzdHVkeT1zdHIoc3R1ZHlfdWlkKSwgZXJyb3I9Zid7dHlwZShlcnJvcikuX19uYW1lX199OiB7c3RyKGVycm9yKVs6NTAwXX0nKQogICAgICAgICAgICAgICAgd2l0aCBfa2VfaW5wdXRfbG9jazoKICAgICAgICAgICAgICAgICAgICBfa2VfaW5wdXRfaWRzLmFkZCgoYXJtWyduYW1lJ10sIHN0cihzdHVkeV91aWQpKSkKICAgICAgICBkZWwgbW9kZWwKICAgICAgICBfa2VfZ2MuY29sbGVjdCgpCiAgICAgICAgd2l0aCB0b3JjaC5jdWRhLmRldmljZShkZXZpY2UpOgogICAgICAgICAgICB0b3JjaC5jdWRhLmVtcHR5X2NhY2hlKCkKCiAgICBkZWYgcnVuX3NoYXJlZF9tYXhzcGFuKGRldmljZSk6CiAgICAgICAgZmlyc3QsIHJldmVyc2UgPSBhcm1zWzBdLCBhcm1zWzJdCiAgICAgICAgY29tcGFyYWJsZSA9ICgnZmlsZScsICdhcmNoJywgJ3JlcycsICdpbWcnLCAnc2xvdHMnLCAnc3BhbicsICdrX2V2YWwnKQogICAgICAgIGlmIGFueShmaXJzdFtrZXldICE9IHJldmVyc2Vba2V5XSBmb3Iga2V5IGluIGNvbXBhcmFibGUpOgogICAgICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoJ01heFNwYW4gZm9yd2FyZC9yZXZlcnNlIGFybXMgbm8gbG9uZ2VyIHNoYXJlIHByZXByb2Nlc3NpbmcnKQogICAgICAgIHJlYWRlciA9IF9LRV9OU1snX21ha2VfcmVhZGVyJ10oKQogICAgICAgIHdlaWdodF9wYXRoID0gX0tFX05TWydmaW5kX3dlaWdodF9maWxlJ10oZmlyc3RbJ2ZpbGUnXSkKICAgICAgICBtb2RlbCwgcmVzb2x1dGlvbiA9IF9LRV9OU1snbG9hZF9tb2RlbCddKAogICAgICAgICAgICB3ZWlnaHRfcGF0aCwgZmlyc3RbJ2FyY2gnXSwgZmlyc3RbJ3JlcyddLCBkZXZpY2UKICAgICAgICApCiAgICAgICAgaWYgaW50KHJlc29sdXRpb24pICE9IGludChmaXJzdFsncmVzJ10pOgogICAgICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoCiAgICAgICAgICAgICAgICBmIk1heFNwYW4gY2hlY2twb2ludCByZXNvbHV0aW9uIHtyZXNvbHV0aW9ufSAhPSB7Zmlyc3RbJ3JlcyddfSIKICAgICAgICAgICAgKQogICAgICAgIGZvciBzdHVkeV9pbmRleCwgc3R1ZHlfdWlkLCBmdXR1cmUgaW4gX2tlX3ByZWZldGNoZWRfd2luZG93cygKICAgICAgICAgICAgZmlyc3QsIHRlc3RfaWRzLCBzZXJpZXMsIHNlcmllc19yb290LCByZWFkZXIKICAgICAgICApOgogICAgICAgICAgICByc25hX2RlYWRsaW5lKCdSYXB0b3IgZnVsbC1jb2hvcnQgaW5mZXJlbmNlJykKICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgd2luZG93cyA9IGZ1dHVyZS5yZXN1bHQoKQogICAgICAgICAgICAgICAgb3V0cHV0c1swXVtzdHVkeV9pbmRleF0gPSBfS0VfTlNbJ2luZmVyX3Byb2JzJ10oCiAgICAgICAgICAgICAgICAgICAgbW9kZWwsIHdpbmRvd3MsIGRldmljZQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgb3V0cHV0c1syXVtzdHVkeV9pbmRleF0gPSBfS0VfTlNbJ2luZmVyX3Byb2JzJ10oCiAgICAgICAgICAgICAgICAgICAgbW9kZWwsIHdpbmRvd3MuZmxpcCgxKS5jb250aWd1b3VzKCksIGRldmljZQogICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgZGVsIHdpbmRvd3MKICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBlcnJvcjoKICAgICAgICAgICAgICAgIHJzbmFfZXZlbnQoJ3JhcHRvcl9zdHVkeV9mYWlsZWQnLCBhcm09c3RyKGZpcnN0WyduYW1lJ10pLCBzdHVkeT1zdHIoc3R1ZHlfdWlkKSwgZXJyb3I9Zid7dHlwZShlcnJvcikuX19uYW1lX199OiB7c3RyKGVycm9yKVs6NTAwXX0nKQogICAgICAgICAgICAgICAgd2l0aCBfa2VfaW5wdXRfbG9jazoKICAgICAgICAgICAgICAgICAgICBfa2VfaW5wdXRfaWRzLmFkZCgoZmlyc3RbJ25hbWUnXSwgc3RyKHN0dWR5X3VpZCkpKTsgX2tlX2lucHV0X2lkcy5hZGQoKHJldmVyc2VbJ25hbWUnXSwgc3RyKHN0dWR5X3VpZCkpKQogICAgICAgIGRlbCBtb2RlbAogICAgICAgIF9rZV9nYy5jb2xsZWN0KCkKICAgICAgICB3aXRoIHRvcmNoLmN1ZGEuZGV2aWNlKGRldmljZSk6CiAgICAgICAgICAgIHRvcmNoLmN1ZGEuZW1wdHlfY2FjaGUoKQoKICAgIGRlZiBncHVfemVybygpOgogICAgICAgIHdpdGggdG9yY2guY3VkYS5kZXZpY2UoMCk6CiAgICAgICAgICAgIHJ1bl9zaGFyZWRfbWF4c3Bhbih0b3JjaC5kZXZpY2UoJ2N1ZGE6MCcpKQoKICAgIGRlZiBncHVfb25lKCk6CiAgICAgICAgd2l0aCB0b3JjaC5jdWRhLmRldmljZSgxKToKICAgICAgICAgICAgcnVuX3NpbmdsZSgxLCB0b3JjaC5kZXZpY2UoJ2N1ZGE6MScpKQogICAgICAgICAgICBydW5fc2luZ2xlKDMsIHRvcmNoLmRldmljZSgnY3VkYToxJykpCgogICAgd2l0aCBfS2VUaHJlYWRQb29sKG1heF93b3JrZXJzPTIpIGFzIGV4ZWN1dG9yOgogICAgICAgIHdvcmtlcnMgPSBbZXhlY3V0b3Iuc3VibWl0KGdwdV96ZXJvKSwgZXhlY3V0b3Iuc3VibWl0KGdwdV9vbmUpXQogICAgICAgIGZvciB3b3JrZXIgaW4gd29ya2VyczoKICAgICAgICAgICAgd29ya2VyLnJlc3VsdCgpCgogICAgX2tlX25wLnNhdmV6X2NvbXByZXNzZWQoJy9rYWdnbGUvd29ya2luZy9yYXB0b3JfcmF3Lm5weicsCiAgICAgICAgc3R1ZHlfdWlkcz1fa2VfbnAuYXNhcnJheSh0ZXN0X2lkcyksIHJhd19wcm9iYWJpbGl0aWVzPV9rZV9ucC5zdGFjayhvdXRwdXRzKSkKICAgIGZvciBfYWksIF9hcnIgaW4gZW51bWVyYXRlKG91dHB1dHMpOgogICAgICAgIF9iYWQgPSB+X2tlX25wLmlzZmluaXRlKF9hcnIpLmFsbChheGlzPTEpCiAgICAgICAgaWYgX2JhZC5hbnkoKToKICAgICAgICAgICAgX2ZpbGwgPSBfa2VfbnAubmFubWVhbihfa2VfbnAud2hlcmUoX2tlX25wLmlzZmluaXRlKF9hcnIpLCBfYXJyLCBfa2VfbnAubmFuKSwgYXhpcz0wKSBpZiAofl9iYWQpLmFueSgpIGVsc2UgX2tlX25wLmZ1bGwoX2Fyci5zaGFwZVsxXSwgLjUsIF9hcnIuZHR5cGUpCiAgICAgICAgICAgIF9hcnJbX2JhZF0gPSBfa2VfbnAud2hlcmUoX2tlX25wLmlzZmluaXRlKF9maWxsKSwgX2ZpbGwsIC41KQogICAgICAgICAgICByc25hX2V2ZW50KCdyYXB0b3JfbmV1dHJhbF9maWxsJywgYXJtPXN0cihhcm1zW19haV1bJ25hbWUnXSksIHN0dWRpZXM9aW50KF9iYWQuc3VtKCkpKQogICAgcnNuYV9maW5pdGUoX2tlX25wLnN0YWNrKG91dHB1dHMpLCdhbGwgZm91ciBSYXB0b3Igdmlld3MnLHByb2JhYmlsaXR5PVRydWUpCiAgICB3ZWlnaHRzID0gX2tlX25wLmFzYXJyYXkoW2Zsb2F0KGFybVsndyddKSBmb3IgYXJtIGluIGFybXNdLCBfa2VfbnAuZmxvYXQ2NCkKICAgIHdlaWdodHMgLz0gd2VpZ2h0cy5zdW0oKQogICAgcHJvYmFiaWxpdHlfYmxlbmQgPSBfa2VfbnAudGVuc29yZG90KAogICAgICAgIHdlaWdodHMsCiAgICAgICAgX2tlX25wLnN0YWNrKFtfa2VfbnAuY2xpcCh2YWx1ZSwgMCwgMSkgZm9yIHZhbHVlIGluIG91dHB1dHNdKSwKICAgICAgICBheGVzPSgwLCAwKSwKICAgICkKICAgIHJhbmtzID0gX0tFX05TWydyYW5rcGN0J10ocHJvYmFiaWxpdHlfYmxlbmQpCiAgICByc25hX2Zpbml0ZShyYW5rcywnUmFwdG9yIHJhbmtlZCBlbnNlbWJsZScscHJvYmFiaWxpdHk9VHJ1ZSkKICAgIHN1Ym1pc3Npb24gPSBfa2VfcGQuRGF0YUZyYW1lKHJhbmtzLmFzdHlwZShfa2VfbnAuZmxvYXQzMiksIGNvbHVtbnM9X0tFX05TWydMQUInXSkKICAgIHN1Ym1pc3Npb24uaW5zZXJ0KDAsICdTdHVkeUluc3RhbmNlVUlEJywgdGVzdF9pZHMpCiAgICBzdWJtaXNzaW9uID0gc3VibWlzc2lvbltjb2x1bW5zXQogICAgaWYgc3VibWlzc2lvblsnU3R1ZHlJbnN0YW5jZVVJRCddLnRvbGlzdCgpICE9IHRlc3RfaWRzOgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcignUmFwdG9yIHN0dWR5IG9yZGVyIGRyaWZ0JykKICAgIGlmIG5vdCBfa2VfbnAuaXNmaW5pdGUoc3VibWlzc2lvbltfS0VfTlNbJ0xBQiddXS50b19udW1weSgpKS5hbGwoKToKICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoJ1JhcHRvciBwcm9kdWNlZCBub24tZmluaXRlIHByZWRpY3Rpb25zJykKICAgIG91dHB1dCA9IF9LZVBhdGgoJy9rYWdnbGUvd29ya2luZy9fcmFwdG9yLmNzdicpCiAgICBzdWJtaXNzaW9uLnRvX2NzdihvdXRwdXQsIGluZGV4PUZhbHNlKQogICAgcHJpbnQoCiAgICAgICAgZiJbcmFwdG9yLWZhc3RdIHdyb3RlIHtvdXRwdXR9OyBlbGFwc2VkIHtfa2VfdGltZS50aW1lKCkgLSBzdGFydGVkOi4xZn1zIiwKICAgICAgICBmbHVzaD1UcnVlLAogICAgKQoKCl9rZV9pbnB1dF9sb2NrID0gX2tlX3RocmVhZGluZy5Mb2NrKCkKX2tlX2lucHV0X2lkcyA9IHNldCgpCl9rZV9waXhlbF9jYWNoZV9ieXRlcyA9IDAKX2tlX2lucHV0X2F1ZGl0X3BhdGggPSBfS2VQYXRoKCcva2FnZ2xlL3dvcmtpbmcvZGlhZ25vc3RpY3MvcmFwdG9yX2lucHV0cy5qc29ubCcpCl9rZV9pbnB1dF9hdWRpdF9wYXRoLnVubGluayhtaXNzaW5nX29rPVRydWUpCl9LRV9OU1snaW5mZXJfcHJvYnMnXSA9IF9rZV9pbmZlcl9pbnB1dApmb3IgX2FybSBpbiBfS0VfTlNbJ0FSTVMnXToKICAgIF9hcm1bJ3BhcmVudF9rX2V2YWwnXSA9IF9hcm1bJ2tfZXZhbCddCiAgICBfYXJtWydrX2V2YWwnXSA9IDk0CnJzbmFfcGhhc2UoJ3B1YmxpY19yYXB0b3InLCAnU1RBUlQnKQpfa2VfcnVuX3JhcHRvcl9hcm1zKCkKcnNuYV9waGFzZSgncHVibGljX3JhcHRvcicsICdDT01QTEVURScpCl9rZV9waXhlbF9jYWNoZS5jbGVhcigpCl9rZV9vcmRlcl9jYWNoZS5jbGVhcigpCl9rZV9nYy5jb2xsZWN0KCkKX0tlUGF0aCgnL2thZ2dsZS93b3JraW5nL3JhcHRvcl9pbnB1dF9iZWZvcmVfY29hdC5jc3YnKS53cml0ZV9ieXRlcyhfS2VQYXRoKCcva2FnZ2xlL3dvcmtpbmcvX3JhcHRvci5jc3YnKS5yZWFkX2J5dGVzKCkpCmltcG9ydCBqc29uIGFzIF9hc3NldF9qc29uCl9LZVBhdGgoJy9rYWdnbGUvd29ya2luZy9yYXB0b3JfaW5wdXRfc3VtbWFyeS5qc29uJykud3JpdGVfdGV4dCgKICAgIF9hc3NldF9qc29uLmR1bXBzKHsnc3R1ZGllcyc6bGVuKF9SU05BX1RFU1RfSURTKSwgJ3ByZXBhcmF0aW9ucyc6bGVuKF9rZV9pbnB1dF9pZHMpLAogICAgICAgICAgICAgICAgICAgICd2aWV3X25hbWVzJzpbYVsnbmFtZSddIGZvciBhIGluIF9LRV9OU1snQVJNUyddXSwKICAgICAgICAgICAgICAgICAgICAnc2hhcmVkX3ByZXBhcmF0aW9uX3ZpZXdzJzpbJ21heHNwYW4tdjUnLCdtYXhzcGFuLXY1LXJldmVyc2UnXSwKICAgICAgICAgICAgICAgICAgICAncmVjb3Jkcyc6c3RyKF9rZV9pbnB1dF9hdWRpdF9wYXRoKX0saW5kZW50PTIpKQoKCgpkZWYgX2NvYXRfc3Vic3RpdHV0ZSgpOgogICAgaW1wb3J0IGhhc2hsaWIgYXMgX2gsIG9zIGFzIF9vLCBzdWJwcm9jZXNzIGFzIF9zcCwgc3lzIGFzIF9zeQogICAgZnJvbSBwYXRobGliIGltcG9ydCBQYXRoIGFzIF9QCiAgICBpbXBvcnQgcGFuZGFzIGFzIF9wZAoKICAgIGltcG9ydCBudW1weSBhcyBfbnAKICAgIF9QKCcva2FnZ2xlL3dvcmtpbmcvX3JhcHRvcl9wdWJsaWMuY3N2Jykud3JpdGVfYnl0ZXMoCiAgICAgICAgX1AoJy9rYWdnbGUvd29ya2luZy9fcmFwdG9yLmNzdicpLnJlYWRfYnl0ZXMoKSkKICAgIF9ucC5zYXZlel9jb21wcmVzc2VkKCcva2FnZ2xlL3dvcmtpbmcvYnRrX3YzMl9yZXN0X2lucHV0Lm5weicsCiAgICAgICAgc3R1ZHlfdWlkcz1fa2Vfb3Vycy5TdHVkeUluc3RhbmNlVUlELnRvX251bXB5KCkuYXN0eXBlKHN0ciksCiAgICAgICAgbGFiZWxzPV9ucC5hc2FycmF5KF9LRV9MQUIpLCB2YWx1ZXM9X2tlX291cnNbX0tFX0xBQl0udG9fbnVtcHkoKSkKICAgIE1BTl9TSEEgPSAnOTg1MTFhOGZkZWI5ZGEwZTZlNzBjNzhkMDEzZmY2MzZlMTQ3NmYzMWM4MGI1ZGMxMzRkMjk0YjE4YzNmMjg0ZScKICAgIFdITF9TSEEgPSAnMjM2YzhkZjU0YTkwZjRkMDIwNzZlNmY5YzFjYzc2M2Q3OTQ1NDJlODg2YzU3NmE2ZmVlNDZlYzhmZjc1YTdhOScKICAgIHJhcHRvciA9IF9QKCcva2FnZ2xlL3dvcmtpbmcvX3JhcHRvci5jc3YnKQoKICAgIGRlZiBzaGEocCk6CiAgICAgICAgZCA9IF9oLnNoYTI1NigpCiAgICAgICAgd2l0aCBfUChwKS5vcGVuKCdyYicpIGFzIGY6CiAgICAgICAgICAgIGZvciBiIGluIGl0ZXIobGFtYmRhOiBmLnJlYWQoOCA8PCAyMCksIGInJyk6CiAgICAgICAgICAgICAgICBkLnVwZGF0ZShiKQogICAgICAgIHJldHVybiBkLmhleGRpZ2VzdCgpCgogICAgZGVmIGZpbmQobmFtZSwgd2FudCk6CiAgICAgICAgcmV0dXJuIF9hc3NldF9maW5kX2Fzc2V0KG5hbWUsIHdhbnQpCgogICAgbWFuID0gZmluZCgnY29hdF9yZXNnYXRlZF9lcDEwX3RvcDNfbWFuaWZlc3QuanNvbicsIE1BTl9TSEEpCiAgICBpZiBtYW4gaXMgTm9uZToKICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoJ2NvYXQgbWFuaWZlc3QgYWJzZW50IG9yIGhhc2ggbWlzbWF0Y2gnKQoKICAgIGFydCA9IG1hbi5wYXJlbnQKCiAgICB3aGwgPSBmaW5kKCdvcGVuY3ZfcHl0aG9uX2hlYWRsZXNzLTQuMTIuMC44OC0qLndobCcsIFdITF9TSEEpCiAgICBlbnZkID0gX1AoJy9rYWdnbGUvd29ya2luZy9fY29hdF9lbnYnKQogICAgX3NwLnJ1bigKICAgICAgICBbCiAgICAgICAgICAgIF9zeS5leGVjdXRhYmxlLAogICAgICAgICAgICAnLW0nLAogICAgICAgICAgICAncGlwJywKICAgICAgICAgICAgJ2luc3RhbGwnLAogICAgICAgICAgICAnLS1uby1kZXBzJywKICAgICAgICAgICAgJy0tcXVpZXQnLAogICAgICAgICAgICAnLS10YXJnZXQnLAogICAgICAgICAgICBzdHIoZW52ZCksCiAgICAgICAgICAgIHN0cih3aGwpLAogICAgICAgIF0sCiAgICAgICAgY2hlY2s9VHJ1ZSwKICAgICkKCiAgICBvdXQgPSBfUCgnL2thZ2dsZS93b3JraW5nL19jb2F0X2FybS5jc3YnKQoKICAgIGltcG9ydCBpbnNwZWN0CiAgICBydW50aW1lX3BhdGggPSBfUCgnL2thZ2dsZS93b3JraW5nL2lucHV0X3Jlc2dhdGVkX3J1bnRpbWUucHknKQogICAgaGVscGVyX3NvdXJjZSA9ICdpbXBvcnQgbnVtcHkgYXMgbnBcbicgKyAnXG4nLmpvaW4oaW5zcGVjdC5nZXRzb3VyY2UoZikgZm9yIGYgaW4KICAgICAgICAoX2RlbnNlX2FsbG9jYXRlLCBfZGVuc2VfdW5pcXVlX2xpbnNwYWNlLCBfZGVuc2VfY29hdF9zcGVjcykpCiAgICBydW50aW1lX3NoYSA9IF9hc3NldF93cml0ZV9jb2F0X3J1bnRpbWUoYXJ0LHJ1bnRpbWVfcGF0aCxoZWxwZXJfc291cmNlKQoKICAgIGNoaWxkID0gKAogICAgICAgICJpbXBvcnQgc3lzLCBqc29uLCBvc1xuIgogICAgICAgIGYic3lzLnBhdGguaW5zZXJ0KDAsIHtzdHIoZW52ZCkhcn0pXG4iCiAgICAgICAgZiJzeXMucGF0aC5pbnNlcnQoMCwge3N0cihhcnQpIXJ9KVxuIgogICAgICAgICJpbXBvcnQgY3YyOyBhc3NlcnQgY3YyLl9fdmVyc2lvbl9fID09ICc0LjEyLjAnLCBjdjIuX192ZXJzaW9uX19cbiIKICAgICAgICAiaW1wb3J0IHRvcmNoOyBhc3NlcnQgdG9yY2guY3VkYS5kZXZpY2VfY291bnQoKSA9PSAyXG4iCiAgICAgICAgInN5cy5wYXRoLmluc2VydCgwLCAnL2thZ2dsZS93b3JraW5nJylcbiIKICAgICAgICAiaW1wb3J0IGlucHV0X3Jlc2dhdGVkX3J1bnRpbWUgYXMgcnRcbiIKICAgICAgICAiYXNzZXJ0IHJ0LmJhc2UuY3YyLl9fdmVyc2lvbl9fID09ICc0LjEyLjAnXG4iCiAgICAgICAgImZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aFxuIgogICAgICAgICJyID0gcnQucnVuX3N1Ym1pc3Npb24oIgogICAgICAgICJjb21wZXRpdGlvbl9yb290PVBhdGgob3MuZW52aXJvblsnUlNOQV9DT01QX1JPT1QnXSkgIgogICAgICAgICJpZiBvcy5lbnZpcm9uLmdldCgnUlNOQV9DT01QX1JPT1QnKSAiCiAgICAgICAgImVsc2UgcnQuYmFzZS5maW5kX2NvbXBldGl0aW9uX3Jvb3QoKSxcbiIKICAgICAgICBmIiAgICBhcnRpZmFjdF9yb290PVBhdGgoe3N0cihhcnQpIXJ9KSwgIgogICAgICAgIGYib3V0cHV0X3BhdGg9UGF0aCh7c3RyKG91dCkhcn0pLFxuIgogICAgICAgICIgICAgZ3B1X2JhdGNoX3N0dWRpZXM9MiwgYmFja2JvbmVfbWljcm9faW1hZ2VzPTgpXG4iCiAgICAgICAgImFzc2VydCByWydzdGF0dXMnXSA9PSBydC5TVUJNSVNTSU9OX1NUQVRVU1xuIgogICAgICAgICJhc3NlcnQgclsnbW9kZWxzJ10gPT0gM1xuIgogICAgICAgICJhc3NlcnQgW2lbJ2Vwb2NoJ10gZm9yIGkgaW4gclsnY2hlY2twb2ludHMnXV0gPT0gWzQsIDYsIDhdXG4iCiAgICAgICAgInByaW50KCdbY29hdC1jaGlsZF0gZmFsbGJhY2tfc3R1ZGllcycsIHJbJ2ZhbGxiYWNrX3N0dWRpZXMnXSwgZmx1c2g9VHJ1ZSlcbiIKICAgICAgICAiUGF0aCgnL2thZ2dsZS93b3JraW5nL19jb2F0X2FybV9yZWNlaXB0Lmpzb24nKS53cml0ZV90ZXh0KCIKICAgICAgICAianNvbi5kdW1wcyhyLCBpbmRlbnQ9MikpXG4iCiAgICApCgogICAgZW52ID0gZGljdChfby5lbnZpcm9uKQogICAgZW52WydDVUROTl9DT05WX1dTQ0FQX0RCRyddID0gJzEwMjQnCiAgICBlbnZbJ1JTTkFfQ09NUF9ST09UJ10gPSBfS0VfTlNbJ2ZpbmRfdGVzdF9yb290J10oKQogICAgZW52WydQWVRIT05QQVRIJ10gPSBmIntlbnZkfTp7YXJ0fToiICsgZW52LmdldCgnUFlUSE9OUEFUSCcsICcnKQoKICAgIGltcG9ydCBqc29uIGFzIF9qCiAgICByc25hX3BoYXNlKCdyZXNpZHVhbF9jb2F0JywgJ1NUQVJUJykKICAgIHRyeToKICAgICAgICBfcnVuX3JlcXVpcmVkX2NoaWxkKFtfc3kuZXhlY3V0YWJsZSwgJy1jJywgY2hpbGRdLCBlbnYsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICBfUCgnL2thZ2dsZS93b3JraW5nL3Jlc2dhdGVkX3J1bnRpbWUubG9nJykpCiAgICAgICAgX2NvYXRfcmVjZWlwdCA9IF9qLmxvYWRzKF9QKCcva2FnZ2xlL3dvcmtpbmcvX2NvYXRfYXJtX3JlY2VpcHQuanNvbicpLnJlYWRfdGV4dCgpKQogICAgICAgIGlmIGludChfY29hdF9yZWNlaXB0LmdldCgnZmFsbGJhY2tfc3R1ZGllcycsIDApIG9yIDApOgogICAgICAgICAgICByc25hX2V2ZW50KCdjb2F0X3Jlc2dhdGVkX2ZhbGxiYWNrX3N0dWRpZXMnLCBjb3VudD1pbnQoX2NvYXRfcmVjZWlwdFsnZmFsbGJhY2tfc3R1ZGllcyddKSwgZmFpbHVyZXM9bGlzdChfY29hdF9yZWNlaXB0LmdldCgnZmFpbHVyZXMnLCBbXSkpWzoyMF0pCiAgICAgICAgX2NvYXRfcmVzZ2F0ZWRfb2sgPSBUcnVlCiAgICBleGNlcHQgRXhjZXB0aW9uIGFzIF9jb2F0X2V4YzoKICAgICAgICByc25hX2V2ZW50KCdjb2F0X3Jlc2dhdGVkX2NoaWxkX2ZhaWxlZCcsIGVycm9yPWYne3R5cGUoX2NvYXRfZXhjKS5fX25hbWVfX306IHtzdHIoX2NvYXRfZXhjKVs6MTUwMF19JykKICAgICAgICBwcmludChmJ1tjb2F0LWFybV0gcmVzaWR1YWwgQ29BdCBjaGlsZCBGQUlMRUQ7IGNvbnRpbnVpbmcgd2l0aG91dCBpdCAoZmxhZ2dlZCk6IHt0eXBlKF9jb2F0X2V4YykuX19uYW1lX199JywgZmx1c2g9VHJ1ZSkKICAgICAgICBfY29hdF9yZXNnYXRlZF9vayA9IEZhbHNlCiAgICByc25hX3BoYXNlKCdyZXNpZHVhbF9jb2F0JywgJ0NPTVBMRVRFJywgb2s9X2NvYXRfcmVzZ2F0ZWRfb2spCiAgICByc25hX3BoYXNlKCdkNCcsICdTVEFSVCcpCiAgICBkNF9vdXQgPSBfUCgnL2thZ2dsZS93b3JraW5nL2Q0X2lucHV0JykvJ2Q0LmNzdicKICAgIHRyeToKICAgICAgICBkNF9tYW5pZmVzdCA9IF9hc3NldF9maW5kX2Fzc2V0KCdjb2F0bmV0X3BhaXJmaWxtX21hbmlmZXN0Lmpzb24nLAogICAgICAgICAgICAnN2FkYTA2MDViY2E2YjA1MzA1NjljNjQ1NGU5ODhhY2U0Nzk2MDZhMzMyOGVkNTkxZDA5MGU1NzY0ZmVhNjYxZCcpCiAgICAgICAgdGltbV93aGwgPSBkNF9tYW5pZmVzdC5wYXJlbnQvJ3RpbW0tMS4wLjIyLXB5My1ub25lLWFueS53aGwnCiAgICAgICAgaWYgc2hhKHRpbW1fd2hsKSAhPSAnODg4OTgxNzUzZTY1Y2JhYWNmYzA3NDk0MzcwMTM4YjE3MDBhMjdiMWYwYWY1ODdmNGY5YjQ3YmMwMjQxNjFkMCc6CiAgICAgICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcignRDQgdGltbSB3aGVlbCBjaGFuZ2VkJykKICAgICAgICBkNF9lbnYgPSBfUCgnL2thZ2dsZS93b3JraW5nL19kNF9lbnYnKQogICAgICAgIF9zcC5ydW4oW19zeS5leGVjdXRhYmxlLCctbScsJ3BpcCcsJ2luc3RhbGwnLCctLW5vLWRlcHMnLCctLXF1aWV0JywKICAgICAgICAgICAgJy0tdGFyZ2V0JyxzdHIoZDRfZW52KSxzdHIodGltbV93aGwpXSxjaGVjaz1UcnVlKQogICAgICAgIGQ0X2RpciA9IF9QKCcva2FnZ2xlL3dvcmtpbmcvZDRfaW5wdXQnKQogICAgICAgIGQ0X2Rpci5ta2RpcihleGlzdF9vaz1UcnVlKQogICAgICAgIF9hc3NldF9ydW5fZDQoZDRfbWFuaWZlc3QucGFyZW50LGQ0X2VudixfS0VfTlNbJ2ZpbmRfdGVzdF9yb290J10oKSxkNF9vdXQpCiAgICAgICAgX2NvYXRfZDRfb2sgPSBUcnVlCiAgICBleGNlcHQgRXhjZXB0aW9uIGFzIF9kNF9leGM6CiAgICAgICAgcnNuYV9ldmVudCgnY29hdF9kNF9jaGlsZF9mYWlsZWQnLCBlcnJvcj1mJ3t0eXBlKF9kNF9leGMpLl9fbmFtZV9ffToge3N0cihfZDRfZXhjKVs6MTUwMF19JykKICAgICAgICBwcmludChmJ1tjb2F0LWFybV0gRDQgQ29BdCBjaGlsZCBGQUlMRUQ7IGNvbnRpbnVpbmcgd2l0aG91dCBpdCAoZmxhZ2dlZCk6IHt0eXBlKF9kNF9leGMpLl9fbmFtZV9ffScsIGZsdXNoPVRydWUpCiAgICAgICAgX2NvYXRfZDRfb2sgPSBGYWxzZQogICAgcnNuYV9waGFzZSgnZDQnLCAnQ09NUExFVEUnLCBvaz1fY29hdF9kNF9vaykKCiAgICBwdWIgPSBfcGQucmVhZF9jc3YocmFwdG9yLCBkdHlwZT17J1N0dWR5SW5zdGFuY2VVSUQnOiBzdHJ9KQogICAgcHViID0gcnNuYV9mcmFtZShwdWIsX1JTTkFfVEVTVF9JRFMsX0tFX0xBQiwncHVibGljIFJhcHRvcicpCiAgICBsYWIgPSBbYyBmb3IgYyBpbiBwdWIuY29sdW1ucyBpZiBjICE9ICdTdHVkeUluc3RhbmNlVUlEJ10KICAgIGltcG9ydCBudW1weSBhcyBfbnAKICAgICMgUmV0YWluIEJUS0QgaW5uZXIgYmxlbmQgY29lZmZpY2llbnRzOyBpbnB1dC9uZXcgZmFtaWx5IHNjb3JlcyBhcmUgdW52YWxpZGF0ZWQuCiAgICBwcml2YXRlX2FscGhhID0gMC40CiAgICBwdWJsaWNfcmFuayA9IHB1YltsYWJdLnJhbmsobWV0aG9kPSdhdmVyYWdlJywgcGN0PVRydWUpCiAgICBfbWVtYmVycyA9IFtdCiAgICBpZiBfY29hdF9yZXNnYXRlZF9vazoKICAgICAgICB0cnk6CiAgICAgICAgICAgIG91cnMgPSBfcGQucmVhZF9jc3Yob3V0LCBkdHlwZT17J1N0dWR5SW5zdGFuY2VVSUQnOiBzdHJ9KQogICAgICAgICAgICBvdXJzID0gcnNuYV9mcmFtZShvdXJzLF9SU05BX1RFU1RfSURTLF9LRV9MQUIsJ3Jlc2dhdGVkIENvQXQnKQogICAgICAgICAgICBpZiBsaXN0KG91cnMuY29sdW1ucykgIT0gbGlzdChwdWIuY29sdW1ucyk6CiAgICAgICAgICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoJ2NvYXQgYXJtIGNvbHVtbiBkcmlmdCcpCiAgICAgICAgICAgIG91cnMgPSBvdXJzLnNldF9pbmRleCgnU3R1ZHlJbnN0YW5jZVVJRCcpLnJlaW5kZXgocHViLlN0dWR5SW5zdGFuY2VVSUQuYXN0eXBlKHN0cikudG9saXN0KCkpLnJlc2V0X2luZGV4KCkKICAgICAgICAgICAgaWYgb3Vyc1tsYWJdLmlzbmEoKS5hbnkoKS5hbnkoKToKICAgICAgICAgICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcignY29hdCBhcm0gZG9lcyBub3QgY292ZXIgZXZlcnkgc3R1ZHknKQogICAgICAgICAgICByc25hX3NhdmVfcHJlZGljdGlvbnMoJ2NvYXRfcmVzZ2F0ZWRfcmFuaycsX1JTTkFfVEVTVF9JRFMsb3Vyc1tsYWJdLnRvX251bXB5KCksbGFiKQogICAgICAgICAgICBfbWVtYmVycy5hcHBlbmQoKCdyZXNnYXRlZF90b3AzJywgb3Vyc1tsYWJdLnJhbmsobWV0aG9kPSdhdmVyYWdlJywgcGN0PVRydWUpKSkKICAgICAgICBleGNlcHQgRXhjZXB0aW9uIGFzIF9leGM6CiAgICAgICAgICAgIHJzbmFfZXZlbnQoJ2NvYXRfcmVzZ2F0ZWRfb3V0cHV0X3JlamVjdGVkJywgZXJyb3I9Zid7dHlwZShfZXhjKS5fX25hbWVfX306IHtzdHIoX2V4YylbOjUwMF19JykKICAgIGlmIF9jb2F0X2Q0X29rOgogICAgICAgIHRyeToKICAgICAgICAgICAgZDQgPSBfcGQucmVhZF9jc3YoZDRfb3V0LGR0eXBlPXsnU3R1ZHlJbnN0YW5jZVVJRCc6c3RyfSkKICAgICAgICAgICAgZDQgPSByc25hX2ZyYW1lKGQ0LF9SU05BX1RFU1RfSURTLF9LRV9MQUIsJ0Q0IENvQXQnKQogICAgICAgICAgICByc25hX3NhdmVfcHJlZGljdGlvbnMoJ2NvYXRfZDRfcmFuaycsX1JTTkFfVEVTVF9JRFMsZDRbbGFiXS50b19udW1weSgpLGxhYikKICAgICAgICAgICAgaWYgZDQuU3R1ZHlJbnN0YW5jZVVJRC5kdXBsaWNhdGVkKCkuYW55KCkgb3IgbGlzdChkNC5jb2x1bW5zKSE9bGlzdChwdWIuY29sdW1ucyk6CiAgICAgICAgICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoJ0Q0IHNjaGVtYS9VSUQgZHJpZnQnKQogICAgICAgICAgICBkNCA9IGQ0LnNldF9pbmRleCgnU3R1ZHlJbnN0YW5jZVVJRCcpLnJlaW5kZXgocHViLlN0dWR5SW5zdGFuY2VVSUQuYXN0eXBlKHN0cikpCiAgICAgICAgICAgIGlmIGQ0W2xhYl0uaXNuYSgpLmFueSgpLmFueSgpOgogICAgICAgICAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKCdENCBtaXNzaW5nIHByZWRpY3Rpb25zJykKICAgICAgICAgICAgX21lbWJlcnMuYXBwZW5kKCgnZDRfc3dhMycsIGQ0W2xhYl0ucmVzZXRfaW5kZXgoZHJvcD1UcnVlKS5yYW5rKG1ldGhvZD0nYXZlcmFnZScscGN0PVRydWUpKSkKICAgICAgICBleGNlcHQgRXhjZXB0aW9uIGFzIF9leGM6CiAgICAgICAgICAgIHJzbmFfZXZlbnQoJ2NvYXRfZDRfb3V0cHV0X3JlamVjdGVkJywgZXJyb3I9Zid7dHlwZShfZXhjKS5fX25hbWVfX306IHtzdHIoX2V4YylbOjUwMF19JykKICAgIGlmIG5vdCBfbWVtYmVyczoKICAgICAgICByc25hX2V2ZW50KCdjb2F0X2ZhbWlseV91bmF2YWlsYWJsZScsIG5vdGU9J3B1YmxpYyBSYXB0b3Iga2VwdCBhcyB0aGUgQ29BdC9SYXB0b3IgaW5wdXQgKGZsYWdnZWQpJykKICAgICAgICBwcmludCgnW2NvYXQtYXJtXSBubyBDb0F0IGZhbWlseSBtZW1iZXIgYXZhaWxhYmxlOyBwdWJsaWMgUmFwdG9yIHJldGFpbmVkIChmbGFnZ2VkKScsIGZsdXNoPVRydWUpCiAgICAgICAgcmV0dXJuIDAKICAgIGlmIGxlbihfbWVtYmVycykgPT0gMjoKICAgICAgICBwcml2YXRlX3JhbmsgPSBfYXNzZXRfaGFsZl9yYW5rX21peChfbWVtYmVyc1swXVsxXSwgX21lbWJlcnNbMV1bMV0pCiAgICBlbHNlOgogICAgICAgIHJzbmFfZXZlbnQoJ2NvYXRfZmFtaWx5X3BhcnRpYWwnLCBtZW1iZXI9X21lbWJlcnNbMF1bMF0pCiAgICAgICAgcHJpdmF0ZV9yYW5rID0gX21lbWJlcnNbMF1bMV0KICAgIGZhbWlseSA9IHB1Yi5jb3B5KCkKICAgIGZhbWlseVtsYWJdID0gcHJpdmF0ZV9yYW5rCiAgICBmYW1pbHkudG9fY3N2KCcva2FnZ2xlL3dvcmtpbmcvX2NvYXRfZmFtaWx5X3JhbmsuY3N2JywgaW5kZXg9RmFsc2UpCiAgICBoeWJyaWQgPSBwdWIuY29weSgpCiAgICBoeWJyaWRbbGFiXSA9ICgxLjAgLSBwcml2YXRlX2FscGhhKSAqIHB1YmxpY19yYW5rICsgcHJpdmF0ZV9hbHBoYSAqIHByaXZhdGVfcmFuawogICAgaWYgbm90IF9ucC5pc2Zpbml0ZShoeWJyaWRbbGFiXS50b19udW1weShfbnAuZmxvYXQ2NCkpLmFsbCgpOgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcignQ29BdC9SYXB0b3IgaHlicmlkIGNvbnRhaW5zIG5vbi1maW5pdGUgdmFsdWVzJykKICAgIHRtcCA9IHJhcHRvci53aXRoX25hbWUoJy5yYXB0b3JfY29hdF9oeWJyaWQuY3N2JykKICAgIGh5YnJpZC50b19jc3YodG1wLCBpbmRleD1GYWxzZSkKICAgIF9vLnJlcGxhY2UodG1wLCByYXB0b3IpCiAgICByYXB0b3Iud2l0aF9uYW1lKCdfY29hdF9yYXB0b3JfYmxlbmRfcmVjZWlwdC5qc29uJykud3JpdGVfdGV4dChfai5kdW1wcyh7CiAgICAgICAgJ2NvbnRyYWN0JzogJ3B1YmxpY19yYXB0b3JfcHJpdmF0ZV9yZXNpZHVhbF9jb2F0X3YzMl9ub19pbm5lcl9yZXJhbmtfdjEnLCAnaW5uZXJfcmVyYW5rJzogRmFsc2UsICdwcml2YXRlX2FscGhhJzogcHJpdmF0ZV9hbHBoYSwKICAgICAgICAnd2l0aGluX2NvYXQnOiB7bTogMC41IGZvciBtLCBfIGluIF9tZW1iZXJzfSBpZiBsZW4oX21lbWJlcnMpID09IDIgZWxzZSB7X21lbWJlcnNbMF1bMF06IDEuMH0sICdwdWJsaWNfcmFwdG9yX2FscGhhJzogMS4wIC0gcHJpdmF0ZV9hbHBoYSwKICAgICAgICAnc3R1ZHlfY291bnQnOiBsZW4ocHViKSwgJ2ZpbmRpbmdfc3BlY2lmaWNfd2VpZ2h0cyc6IEZhbHNlLCAnbWVtYmVycyc6IFttIGZvciBtLCBfIGluIF9tZW1iZXJzXX0sIGluZGVudD0yLCBzb3J0X2tleXM9VHJ1ZSkgKyAnXG4nKQogICAgcmV0dXJuIGxlbihwdWIpCgpfY29hdF9uID0gX2NvYXRfc3Vic3RpdHV0ZSgpCnByaW50KGYnW2NvYXQtYXJtXSByZXNpZHVhbCB0b3AzIC8gRDQgU1dBMyBlcXVhbCByYW5rIGZhbWlseTsgQlRLRDMyIG5vLWlubmVyLXJlcmFuazsge19jb2F0X259IHN0dWRpZXMnLGZsdXNoPVRydWUpCgpfa2VfdGhlaXJzID0gX2tlX3BkLnJlYWRfY3N2KCcva2FnZ2xlL3dvcmtpbmcvX3JhcHRvci5jc3YnLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgIGR0eXBlPXsnU3R1ZHlJbnN0YW5jZVVJRCc6IHN0cn0pCmFzc2VydCBsaXN0KF9rZV90aGVpcnMuY29sdW1ucykgPT0gbGlzdChfa2Vfb3Vycy5jb2x1bW5zKSwgJ2NvbHVtbiBkcmlmdCcKX2tlX3RoZWlycyA9IF9rZV90aGVpcnMuc2V0X2luZGV4KCdTdHVkeUluc3RhbmNlVUlEJykucmVpbmRleCgKICAgIF9rZV9vdXJzWydTdHVkeUluc3RhbmNlVUlEJ10pLnJlc2V0X2luZGV4KCkKYXNzZXJ0IF9rZV90aGVpcnNbX0tFX0xBQl0ubm90bmEoKS5hbGwoKS5hbGwoKSwgJ3N0dWR5IGlkZW50aXR5IGRyaWZ0JwoKCl9rZV90ciA9IF9rZV9vdXJzW19LRV9MQUJdLnJhbmsobWV0aG9kPSdhdmVyYWdlJywgcGN0PVRydWUpCl9rZV9jciA9IF9rZV90aGVpcnNbX0tFX0xBQl0uY29weSgpCl9ibGVuZF90cmFuc2Zvcm1lciA9IF9rZV9vdXJzLmNvcHkoKQpfYmxlbmRfY29hdG5ldCA9IF9rZV90aGVpcnMuY29weSgpCl9ibGVuZF9sYWJlbHMgPSBsaXN0KF9LRV9MQUIpCl9ibGVuZF90ciA9IF9rZV90ci5jb3B5KCkKX2JsZW5kX2NyID0gX2tlX2NyLmNvcHkoKQpfY29hdG5ldF93ZWlnaHQgPSB7bGFiZWw6IDAuNjAgZm9yIGxhYmVsIGluIF9ibGVuZF9sYWJlbHN9Cl9jb2F0bmV0X3dlaWdodC51cGRhdGUoewogICAgJ0FDTCc6IDAuNzUsCiAgICAnTWVkaWFsIE1lbmlzY3VzJzogMC44MCwKICAgICdMYXRlcmFsIE1lbmlzY3VzJzogMS4wMCwKICAgICdMYXRlcmFsIE9BJzogMC43NSwKICAgICdGcmFjdHVyZSc6IDAuNzUsCn0pCl9ibGVuZF9vdXRwdXQgPSBfYmxlbmRfdHJhbnNmb3JtZXIuY29weSgpCmZvciBfYmxlbmRfbGFiZWwgaW4gX2JsZW5kX2xhYmVsczoKICAgIF9ibGVuZF93ID0gZmxvYXQoX2NvYXRuZXRfd2VpZ2h0W19ibGVuZF9sYWJlbF0pCiAgICBfYmxlbmRfb3V0cHV0W19ibGVuZF9sYWJlbF0gPSAoCiAgICAgICAgKDEuMCAtIF9ibGVuZF93KSAqIF9ibGVuZF90cltfYmxlbmRfbGFiZWxdCiAgICAgICAgKyBfYmxlbmRfdyAqIF9ibGVuZF9jcltfYmxlbmRfbGFiZWxdCiAgICApCl9ibGVuZF9vdXRwdXRbX2JsZW5kX2xhYmVsc10gPSBfYmxlbmRfb3V0cHV0W19ibGVuZF9sYWJlbHNdLnJhbmsoCiAgICBtZXRob2Q9J2F2ZXJhZ2UnLCBwY3Q9VHJ1ZQopCmFzc2VydCBfa2VfbnAuaXNmaW5pdGUoCiAgICBfYmxlbmRfb3V0cHV0W19ibGVuZF9sYWJlbHNdLnRvX251bXB5KF9rZV9ucC5mbG9hdDY0KQopLmFsbCgpCl9ibGVuZF9vdXRwdXQudG9fY3N2KF9rZV9wcmltYXJ5LCBpbmRleD1GYWxzZSkK"}
_GOLD58_CELL_SHAS = {"2": "5815eed58685ad17", "3": "32c4445b136e86d7", "4": "61f911b3b024f870", "5": "a773890aca786656", "6": "a554f7d0f44a197e", "7": "b0c6253db8e7fcac", "8": "ba17bd3298fc57f0", "9": "a1334e4b2174f151", "10": "6878773f8ffa7883", "11": "17434f44700a37e3", "12": "0cee81943e6bfbb9", "13": "a816bbcb91c3f864", "14": "9006955666c6c5f8", "15": "cd7a8762ef90ccf5", "16": "6fffd33cf27daf23", "17": "f1fa6aa2130ce82d", "18": "a0f7ec9a6a26310a", "19": "c6126a20146a0853", "20": "5c31acb1019dd9de", "21": "8a5b323bada53d4d", "22": "20caeab91407b936"}

# ==== GOLD58 DUMP (B-1 v9, auto-inserted 2026-09-20) ====
# 全量重放 cell2-22: shadow 输入 + RSNA_COMP_ROOT 覆写, 58 个专家标注 train study 上
# 复现完整 blend, 导出 gold58_blend.npz; finally 恢复 stage/globals/env, 原提交流程照常.
import base64, hashlib, json, os, shutil, time, traceback
from pathlib import Path
import numpy as np, pandas as pd
import torch
_GOLD58_T0 = time.time()
_GOLD58_IDS = json.loads(base64.b64decode(_GOLD58_IDS_B64).decode())
_GOLD58_SHADOW = Path('/kaggle/working/_gold_shadow')
_GOLD58_STAGE = Path('/kaggle/working/_pipeline_stage.csv')
_GOLD58_SRC = {int(i): base64.b64decode(b).decode() for i, b in _GOLD58_CELLS_B64.items()}
_GOLD58_STRIP19 = "for _a5k, _a5v in _A5_SAVED.items():\n    globals()[_a5k] = _a5v\ndel _A5_SAVED, _a5k, _a5v\n"
_GOLD58_KEEP = {'_GOLD58_G', '_GOLD58_T0', '_GOLD58_IDS', '_GOLD58_SHADOW', '_GOLD58_STAGE', '_GOLD58_SRC', '_GOLD58_KEEP', '_GOLD58_PATCH', '_GOLD58_PATCH_CELLS', '_GOLD58_STRIP19', '_GOLD58_IDS_B64', '_GOLD58_CELLS_B64', '_GOLD58_CELL_SHAS', '_gold58_restore', '_gold58_run', 'In', 'Out', 'get_ipython', 'exit', 'quit', 'base64', 'hashlib', 'json', 'os', 'shutil', 'time', 'traceback', 'np', 'pd', 'Path', 'torch'}

# 确定性补丁表: 长串在前, 防止子串误伤。env 覆写已让 ROOT 变量指 shadow;
# 补丁兜底处理字面量路径(cell4/22 的 fallback 字符串等)。cell22 用
# root+'/test.csv' 字符串拼接, 打补丁会拼出双路径 → 唯一豁免。
_GOLD58_PATCH = [
    ("'/test_series.csv'", "'/kaggle/working/_gold_shadow/test_series.csv'"),
    ("'test_series.csv'", "'/kaggle/working/_gold_shadow/test_series.csv'"),
    ("'/test.csv'", "'/kaggle/working/_gold_shadow/test.csv'"),
    ("'test.csv'", "'/kaggle/working/_gold_shadow/test.csv'"),
    ("'/sample_submission.csv'", "'/kaggle/working/_gold_shadow/sample_submission.csv'"),
    ("'sample_submission.csv'", "'/kaggle/working/_gold_shadow/sample_submission.csv'"),
    ("'/test_series'", "'/kaggle/working/_gold_shadow/test_series'"),
    ("'test_series'", "'/kaggle/working/_gold_shadow/test_series'"),
    ("'/kaggle/input/competitions/rsna-knee-abnormality-detection'", "'/kaggle/working/_gold_shadow'"),
    ("'/kaggle/input/rsna-knee-abnormality-detection'", "'/kaggle/working/_gold_shadow'"),
]
_GOLD58_PATCH_CELLS = {i: (i != 22) for i in range(2, 23)}

def _gold58_restore(snap):
    g = globals()
    for k in [k for k in g if k not in snap and k not in _GOLD58_KEEP
              and not k.startswith('__')]:
        try:
            del g[k]
        except Exception:
            pass
    for k, v in snap.items():
        if k not in _GOLD58_KEEP:
            g[k] = v

def _gold58_run():
    # dump 前的完整状态(供最终恢复)
    _pre = dict(globals())
    _ENV_SAVE = None  # 函数局部: finally 里引用, 需在 try 赋值前有绑定
    try:
        # ---- shadow 输入目录 ----
        if _GOLD58_SHADOW.exists():
            shutil.rmtree(_GOLD58_SHADOW)
        _GOLD58_SHADOW.mkdir(parents=True)
        ts = pd.read_csv(COMP / 'train_series.csv',
                         dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
        ts58 = ts[ts.StudyInstanceUID.isin(set(_GOLD58_IDS))]
        assert len(ts58) > 0, 'train_series.csv 中找不到 gold study'
        _need = {'StudyInstanceUID', 'SeriesInstanceUID', 'Anatomical_Plane',
                 'Fat_Suppression', 'Fluid_Sensitive'}
        assert _need <= set(ts58.columns), f'train_series.csv 缺列: {sorted(_need - set(ts58.columns))}'
        ts58.to_csv(_GOLD58_SHADOW / 'test_series.csv', index=False)
        shutil.copy(COMP / 'train.csv', _GOLD58_SHADOW / 'train.csv')
        cols = pd.read_csv(COMP / 'sample_submission.csv', nrows=0).columns
        labels = [c for c in cols if c != 'StudyInstanceUID']
        assert len(labels) == 12, f'标签列数异常 {len(labels)}'
        sm = pd.DataFrame({'StudyInstanceUID': _GOLD58_IDS})
        for c in labels:
            sm[c] = 0.5
        sm.to_csv(_GOLD58_SHADOW / 'sample_submission.csv', index=False)
        sm.to_csv(_GOLD58_SHADOW / 'test.csv', index=False)
        imgd = _GOLD58_SHADOW / 'test_series'
        imgd.mkdir()
        for uid in _GOLD58_IDS:
            src = COMP / 'train_series' / uid
            if not src.is_dir():
                raise FileNotFoundError(f'缺 DICOM 目录 {src}')
            (imgd / uid).symlink_to(src)
        print(f'[gold58] shadow 就绪: {len(ts58)} series 行, 58 study 链接', flush=True)

        # ---- 备份 stage + 清理臂缓存/d4_input/dinov2 pixel cache ----
        shutil.copy(_GOLD58_STAGE, str(_GOLD58_STAGE) + '.gold58_bak')
        _stale = ['speed_a5_raw.npz', 'coat_top3_part0.npz', 'coat_top3_part1.npz',
                  'coat_resgated_ep10_top3_predictions.npz',
                  'coat_resgated_ep10_top3_submission_receipt.json',
                  'raptor_raw.npz', 'btk_v32_rest_input.npz', '_coat_arm.csv',
                  '_coat_family_rank.csv', '_coat_arm_receipt.json',
                  '_coat_raptor_blend_receipt.json', '_raptor.csv', '_raptor_public.csv',
                  'raptor_input_before_coat.csv', 'raptor_input_summary.json',
                  'v558_rad_receipt.json', 'input_resgated_runtime.py', 'resgated_runtime.log']
        _rm = sum(1 for f in _stale if (Path('/kaggle/working') / f).exists()
                  and not (Path('/kaggle/working') / f).unlink())
        _d4d = Path('/kaggle/working/d4_input')
        _d4r = _d4d.exists()
        if _d4r:
            shutil.rmtree(_d4d)
        _pix = Path(os.environ.get('RSNA_PIXEL_SCRATCH', '/kaggle/temp/rsna_pixels_v559'))
        _pixr = _pix.exists()
        if _pixr:
            shutil.rmtree(_pix)
        print(f'[gold58] 清理旧臂缓存 {_rm} 个' + (', d4_input 目录' if _d4r else '')
              + (', dinov2 pixel cache' if _pixr else ''), flush=True)

        # ---- env 覆写: cell4 ROOT / cell22 find_test_root / coat+d4 子进程同源 ----
        _ENV_SAVE = os.environ.get('RSNA_COMP_ROOT')
        os.environ['RSNA_COMP_ROOT'] = str(_GOLD58_SHADOW)

        # ---- 全量重放 cell2-22 ----
        ip = get_ipython()
        for i in range(2, 23):
            src = _GOLD58_SRC[i]
            n = {}
            if _GOLD58_PATCH_CELLS.get(i, True):
                for old, new in _GOLD58_PATCH:
                    n[old] = src.count(old)
                    src = src.replace(old, new)
            if i == 19:
                # 剥掉尾部自恢复: 全量重放下 _A5_SAVED(=cell16 顶部快照)里混着
                # 第一遍残留的 3-study 全局(A5_PREDS/by/studies/...), 恢复会把它们
                # 打回去污染 20-22; 该块仅内存卫生, 模型释放由 cell19 自身完成.
                assert src.count(_GOLD58_STRIP19) == 1, 'cell19 尾部自恢复文本漂移'
                src = src.replace(_GOLD58_STRIP19, '')
            print(f'[gold58] === 重放 cell{i} ({len(src)} chars, patched='
                  f'{_GOLD58_PATCH_CELLS.get(i, True)}, patches '
                  + ','.join(f'{v}' for v in n.values()) + ') ===', flush=True)
            res = ip.run_cell(src, store_history=False)
            if not getattr(res, 'success', True):
                raise RuntimeError(f'cell{i} 重放失败: '
                                   f'{res.error_in_exec or res.error_before_exec}')

        # ---- 校验导出 ----
        out = pd.read_csv(_GOLD58_STAGE, dtype={'StudyInstanceUID': str})
        assert len(out) == 58, f'stage 行数 {len(out)} != 58'
        assert set(out.StudyInstanceUID) == set(_GOLD58_IDS), 'study 集合不符'
        pred = out[_RSNA_LABELS].to_numpy(np.float64)
        assert np.isfinite(pred).all(), '非有限预测值'
        np.savez('/kaggle/working/gold58_blend.npz', pred=pred,
                 ids=np.array(out.StudyInstanceUID.tolist()),
                 labels=np.array(_RSNA_LABELS))
        print(f'[gold58] DONE: gold58_blend.npz 用时 {time.time()-_GOLD58_T0:.0f}s, '
              f'sha={hashlib.sha256(pred.tobytes()).hexdigest()[:16]}', flush=True)
    except Exception:
        print('[gold58] !!! FAILED — 恢复现场, 原提交流程照常 !!!', flush=True)
        traceback.print_exc()
    finally:
        try:
            if _ENV_SAVE is not None:
                os.environ['RSNA_COMP_ROOT'] = _ENV_SAVE
            else:
                os.environ.pop('RSNA_COMP_ROOT', None)
            _gold58_restore(_pre)
            b = Path(str(_GOLD58_STAGE) + '.gold58_bak')
            if b.exists():
                shutil.copy(b, _GOLD58_STAGE)
                b.unlink()
            print('[gold58] 现场已恢复', flush=True)
        except Exception:
            print('[gold58] !!! 恢复现场时出错 !!!', flush=True)
            traceback.print_exc()

_gold58_run()

#### CELL 25
# Publish submission.csv only after the entire BTKD + CoAt family graph succeeds.
import platform
_release_path=Path('/kaggle/working/_pipeline_stage.csv')
_release_df=rsna_frame(pd.read_csv(_release_path,dtype={'StudyInstanceUID':str}),_RSNA_TEST_IDS,_RSNA_LABELS,'FINAL')
if _DINOV2_MATCHED_MEMBERS!=20:raise RuntimeError('final DINO member count mismatch')
if not globals().get('V18_CALIBRATOR_APPLIED'):raise RuntimeError('BTKD calibrator was not applied')
_expected_preparations={(name,uid) for name in ('maxspan-v5','native384dense-v10','native384-v8') for uid in _RSNA_TEST_IDS}
if _ke_input_ids != _expected_preparations:
    rsna_event('raptor_preparation_identities_incomplete', missing=len(_expected_preparations - _ke_input_ids), extra=len(_ke_input_ids - _expected_preparations))
rsna_save_predictions('final',_RSNA_TEST_IDS,_release_df[_RSNA_LABELS].to_numpy(),_RSNA_LABELS)
_RSNA_AUDIT.update(status='COMPLETE',dino_members=20,a5_folds=5,raptor_views=4,coats={'resgated_epochs':[4,6,8],'d4_parent_swa_epochs':[15,16],'d4_adapter_swa_epochs':[11,9,5]},family_mix=[.5,.5],inner_public_coat_mix=[.6,.4],inner_rerank=False,outer_weights=_coatnet_weight,bt_input_sha256='eb9c51cf04ddbd4e923f1a5c278e7b019036db9e67ecc20bea6661bd536bb303',v555_input_sha256='22d71a1311ef8282e26a0ca498bed57100da5152352deae7cbdaeca4ab8539c9',elapsed_seconds=time.time()-T0,score_recovery_claimed=False)
for cache_file in list(_RSNA_CACHE_FILES):Path(cache_file).unlink(missing_ok=True)
_final=Path('/kaggle/working/submission.csv');_tmp=_final.with_suffix('.csv.tmp')
with _tmp.open('w') as _out_handle:
    _release_df.to_csv(_out_handle,index=False)
    _out_handle.flush();os.fsync(_out_handle.fileno())
_RSNA_AUDIT['submission_sha256']=rsna_sha(_tmp)
os.replace(_tmp,_final)
rsna_json('/kaggle/working/btkd_v559_complete.json',_RSNA_AUDIT)
print(f'COMPLETE: {_final}, {len(_release_df)} studies, SHA256={_RSNA_AUDIT["submission_sha256"]}')

