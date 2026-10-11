"""Teacher predictions of nartaa's Anatomical Mirror Raptor (public 0.949) on the TRAIN studies, for a distilled reader.

The Mirror scoring script is taken verbatim from heliosli's 0.954 notebook (cell 2 of submission_h954, SHA-checked
checkpoint raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt, anatomical L/R TTA), with only these changes:
  * it reads train.csv / train_series.csv / train_series/ instead of the test files;
  * it writes /kaggle/working/mirror_train_raw.csv (raw probabilities; a study that failed or was not reached stays 0.5);
  * every 400 prepared studies it also writes that file (partial results survive a stopped session);
  * it stops feeding studies after STOP_H hours so the output is always written inside the 12 h session.
The teacher was trained on all train studies, so these are in-sample predictions: they carry the teacher's own label
blend and its OAI-pretrained features (rule note: a student of this teacher is OAI-derived, a public-LB candidate only).

    python make_mirror_teacher.py [--owner=esfersami50]  ->  kaggle/mirrorteacher/
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OWNER = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--owner=")), "esfersami50")
STOP_H = 10.5

h954 = json.load(open(os.path.join(ROOT, "submission_h954", "rsna-knee-h954.ipynb")))
h954_meta = json.load(open(os.path.join(ROOT, "submission_h954", "kernel-metadata.json")))
cell = "".join(h954["cells"][2]["source"])
assert cell.startswith("%%writefile /tmp/raptor_run.py\n")
s = cell.split("\n", 1)[1]


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old, s.count(old))
    s = s.replace(old, new)


sub('''    tsdir = ROOT + "/test_series"
    if not os.path.isdir(tsdir):
        tsdir = ROOT + "/test_images"''', '''    tsdir = ROOT + "/train_series"   # [teacher] train split
    if not os.path.isdir(tsdir):
        tsdir = ROOT + "/train_images"''')
sub('''    test = pd.read_csv(ROOT + "/test.csv"); test["StudyInstanceUID"]''',
    '''    test = pd.read_csv(ROOT + "/train.csv"); test["StudyInstanceUID"]''')
sub('''    tser = pd.read_csv(ROOT + "/test_series.csv")''', '''    tser = pd.read_csv(ROOT + "/train_series.csv")''')
sub('''        for i, (sid, vol, mask) in enumerate(pool.imap_unordered(_pre_one, test_ids, chunksize=2)):
            q.put((sid, vol, mask))''', '''        # [teacher] v2: feed the pool in blocks of 96 studies.  v1's imap_unordered over all 4,407 ids buffered every
        # finished study in the parent whenever the prep pool outran the GPUs, and the parent was OOM-killed after ~300.
        def _blocked():
            for j in range(0, len(test_ids), 96):
                if time.time() - t0 > __STOP_H__ * 3600:   # [teacher] always finish inside the session
                    print(f"[teacher] STOP feeding at study {j}/{N} after {(time.time() - t0) / 3600:.2f} h", flush=True)
                    return
                yield from pool.imap_unordered(_pre_one, test_ids[j:j + 96], chunksize=2)
        for i, (sid, vol, mask) in enumerate(_blocked()):
            q.put((sid, vol, mask))
            if (i + 1) % 400 == 0:   # [teacher] partial dump: rows not scored yet are still 0.5
                _part = pd.DataFrame(arm_probs[0].astype(np.float32), columns=LAB); _part.insert(0, "StudyInstanceUID", test_ids)
                _part.to_csv("/kaggle/working/mirror_train_raw.csv", index=False)''')
sub('''    _raw.to_csv("/kaggle/working/raptor_raw.csv", index=False)''',
    '''    _raw.to_csv("/kaggle/working/mirror_train_raw.csv", index=False)   # [teacher]
    print(f"[teacher] wrote mirror_train_raw.csv | studies left at 0.5: {int((_raw[LAB].values == 0.5).all(axis=1).sum())}", flush=True)''')
sub('''    out = "/kaggle/working/raptor_sub.csv"''', '''    out = "/kaggle/working/mirror_train_sub.csv"''')
sub('''    with mp.get_context("fork").Pool(nproc, initializer=_pre_init) as pool:''',
    '''    with mp.get_context("fork").Pool(nproc, initializer=_pre_init, maxtasksperchild=64) as pool:   # [teacher] recycle workers''')
s = s.replace("__STOP_H__", repr(STOP_H))
compile(s, "mirror_teacher.py", "exec")

d = os.path.join(HERE, "..", "kaggle", "mirrorteacher" + ("" if OWNER == "esfersami50" else f"_{OWNER}"))
os.makedirs(d, exist_ok=True)
open(os.path.join(d, "mirror_teacher.py"), "w").write(s)
meta = {"id": f"{OWNER}/rsna-knee-mirror-teacher", "title": "rsna-knee-mirror-teacher", "code_file": "mirror_teacher.py",
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "machine_shape": "NvidiaTeslaT4", "docker_image": h954_meta["docker_image"],
        "docker_image_pinning_type": "original",
        "dataset_sources": ["nartaa/rsna-knee-publication-swa-weights-20261007"],
        "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": []}
json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
print("wrote", d)
