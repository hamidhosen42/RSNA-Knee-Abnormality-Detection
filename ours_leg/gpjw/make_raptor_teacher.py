"""Teacher predictions of a public Raptor CoAtNet on the TRAIN studies (raw probabilities), for distilled readers.

Sources (each taken verbatim from a notebook that scored, then only re-pointed at the train split):
  mirror384  nartaa's 0.949 Anatomical Mirror script (cell 2 of submission_h954 = heliosli's 0.954 notebook)
             -> mirror_train_raw.csv  (the existing esfersami50/rsna-knee-mirror-teacher v2 did exactly this)
  m224       nartaa's 224-px / 0.8-crop checkpoint (raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt) with the
             anatomical mirror view, as in varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up v1 cell 9
             -> m224_train_raw.csv
Changes (same as make_mirror_teacher.py v2): read train.csv / train_series.csv / train_series/; feed the prep pool
in blocks of 96 studies (v1 of the Mirror teacher was OOM-killed when imap_unordered buffered every study); recycle
pool workers; dump raw probabilities every 400 studies and at the end; stop feeding at STOP_H hours.
A study that failed or was not reached stays at 0.5 in every column (read as "missing" by the student builder).

    python make_raptor_teacher.py --source=m224 [--owner=hosen42]  ->  kaggle/raptorteacher_m224[_owner]/
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OWNER = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--owner=")), "hosen42")
SOURCE = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--source=")), "m224")
STOP_H = 10.5

if SOURCE == "mirror384":
    cell = "".join(json.load(open(os.path.join(ROOT, "submission_h954", "rsna-knee-h954.ipynb")))["cells"][2]["source"])
    RAW, SUB = "mirror_train_raw.csv", "mirror_train_sub.csv"
elif SOURCE == "m224":
    nb = json.load(open(os.path.join(ROOT, "public_notebooks", "0.956_7v_varshithakolupuri__rsna-0-956-one-free-flip-68-ranks-up.ipynb")))
    cell = "".join(nb["cells"][9]["source"])
    assert "c08_r224_swa.pt" in cell and "[s9]" in cell
    RAW, SUB = "m224_train_raw.csv", "m224_train_sub.csv"
else:
    raise SystemExit("--source must be mirror384 or m224")
assert cell.startswith("%%writefile /tmp/raptor")
s = cell.split("\n", 1)[1]


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:80], s.count(old))
    s = s.replace(old, new)


sub('''    tsdir = ROOT + "/test_series"
    if not os.path.isdir(tsdir):
        tsdir = ROOT + "/test_images"''', '''    tsdir = ROOT + "/train_series"   # [teacher] train split
    if not os.path.isdir(tsdir):
        tsdir = ROOT + "/train_images"''')
sub('''    test = pd.read_csv(ROOT + "/test.csv"); test["StudyInstanceUID"]''',
    '''    test = pd.read_csv(ROOT + "/train.csv"); test["StudyInstanceUID"]''')
sub('''    tser = pd.read_csv(ROOT + "/test_series.csv")''', '''    tser = pd.read_csv(ROOT + "/train_series.csv")''')
sub('''    with mp.get_context("fork").Pool(nproc, initializer=_pre_init) as pool:''',
    '''    with mp.get_context("fork").Pool(nproc, initializer=_pre_init, maxtasksperchild=64) as pool:   # [teacher] recycle workers''')
sub('''        for i, (sid, vol, mask) in enumerate(pool.imap_unordered(_pre_one, test_ids, chunksize=2)):
            q.put((sid, vol, mask))''', f'''        # [teacher] feed the pool in blocks of 96 studies: imap_unordered over all 4,407 ids buffered every finished
        # study in the parent whenever the prep pool outran the GPUs (the first Mirror teacher was OOM-killed that way)
        def _blocked():
            for j in range(0, len(test_ids), 96):
                if time.time() - t0 > {STOP_H} * 3600:   # [teacher] always finish inside the session
                    print(f"[teacher] STOP feeding at study {{j}}/{{N}} after {{(time.time() - t0) / 3600:.2f}} h", flush=True)
                    return
                yield from pool.imap_unordered(_pre_one, test_ids[j:j + 96], chunksize=2)
        for i, (sid, vol, mask) in enumerate(_blocked()):
            q.put((sid, vol, mask))
            if (i + 1) % 400 == 0:   # [teacher] partial dump: rows not scored yet are still 0.5
                _part = pd.DataFrame(arm_probs[0].astype(np.float32), columns=LAB); _part.insert(0, "StudyInstanceUID", test_ids)
                _part.to_csv("/kaggle/working/{RAW}", index=False)''')
if SOURCE == "mirror384":
    sub('''    _raw.to_csv("/kaggle/working/raptor_raw.csv", index=False)''',
        f'''    _raw.to_csv("/kaggle/working/{RAW}", index=False)   # [teacher]
    print(f"[teacher] wrote {RAW} | studies left at 0.5: {{int((_raw[LAB].values == 0.5).all(axis=1).sum())}}", flush=True)''')
    sub('''    out = "/kaggle/working/raptor_sub.csv"''', f'''    out = "/kaggle/working/{SUB}"''')
else:
    sub('''    print(f"[gpu] done: {done_n[0]}/{N} studies on {len(devs)} device(s) | {time.time()-t0:.0f}s", flush=True)''',
        f'''    print(f"[gpu] done: {{done_n[0]}}/{{N}} studies on {{len(devs)}} device(s) | {{time.time()-t0:.0f}}s", flush=True)
    _raw = pd.DataFrame(arm_probs[0].astype(np.float32), columns=LAB); _raw.insert(0, "StudyInstanceUID", test_ids)   # [teacher]
    _raw.to_csv("/kaggle/working/{RAW}", index=False)   # raw probabilities (plain + mirror averaged); failed rows stay 0.5
    print(f"[teacher] wrote {RAW} | studies left at 0.5: {{int((_raw[LAB].values == 0.5).all(axis=1).sum())}}", flush=True)''')
    sub('''    out = "/kaggle/working/raptor224_sub.csv"   # [s5] 224 stage output; the blend cell writes submission.csv''',
        f'''    out = "/kaggle/working/{SUB}"   # [teacher] rank file (diagnostic only)''')
compile(s, "raptor_teacher.py", "exec")

d = os.path.join(HERE, "..", "kaggle", f"raptorteacher_{SOURCE}" + ("" if OWNER == "hosen42" else f"_{OWNER}"))
os.makedirs(d, exist_ok=True)
open(os.path.join(d, "raptor_teacher.py"), "w").write(s)
slug = f"rsna-knee-{SOURCE}-teacher"
meta = {"id": f"{OWNER}/{slug}", "title": slug, "code_file": "raptor_teacher.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f",
        "docker_image_pinning_type": "original", "dataset_sources": ["nartaa/rsna-knee-publication-swa-weights-20261007"],
        "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": []}
json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
print("wrote", d, "| raw ->", RAW)
