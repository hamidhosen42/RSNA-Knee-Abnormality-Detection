"""Build the CPU kernels that cache goodpjw2008's canonical 384 px series volumes for training folds 3-4.

The reader (dataset goodpjw2008/rsna-knee-2-5d-convnext-reader v1, Apache-2.0) trains from a cache of
every train series made by its own `preprocess.load_series` (0.4 mm/px, 384 px, <= 64 slices, uint8).
Raw that cache is ~100 GB, over Kaggle's 20 GB output limit, so here each slice is stored as a
lossless PNG and the train studies are split over N shard kernels by a hash of the StudyInstanceUID.
`load_series` is embedded verbatim (src/preprocess.py == the dataset's file), so the pixels are
exactly what the author trained on; only the container differs.

Per series: cache384/<series>.npz with `data` (concatenated PNG bytes) and `off` (n+1 offsets).
Per shard:  cache384_meta.csv (study, series, n, n_raw, n_bad, plane, slice_gap, error, bytes).

    python make_cache_kernels.py [n_shards=5]  ->  kaggle/gpjwcache_<i>/  (push each with the kaggle CLI)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
OWNER = "esfersami50"
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"

prep = open(os.path.join(HERE, "src", "preprocess.py")).read()
prep = prep.replace('\nif __name__ == "__main__":\n    main()', "\n")      # keep only the library part

MAIN = r'''

# ======================================================================= shard kernel (ours)
import glob, hashlib, subprocess, time
SHARD, N_SHARDS = __SHARD__, __NSHARDS__
BUDGET_S = 11.3 * 3600
MAX_BYTES = 18.5e9
T0 = time.time()


def _find(name):
    for pat in ["/kaggle/input/*/", "/kaggle/input/*/*/", "/kaggle/input/*/*/*/"]:
        for d in sorted(glob.glob(pat)):
            if os.path.exists(os.path.join(d, name)):
                return d.rstrip("/")
    raise FileNotFoundError(name)


def _install_decoders():
    try:
        import pylibjpeg  # noqa: F401
        return "present"
    except ImportError:
        pass
    assets = _find("cnxt_v0_fold0.pt")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--find-links", assets,
                    "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=False)
    return "installed from " + assets


def _encode(vol):
    bufs = [cv2.imencode(".png", s, [cv2.IMWRITE_PNG_COMPRESSION, 6])[1].tobytes() for s in vol]
    off = np.cumsum([0] + [len(b) for b in bufs]).astype(np.int64)
    return np.frombuffer(b"".join(bufs), np.uint8), off


def _shard_job(args):
    root, study, series, plane, out_dir = args
    info = {"study": study, "series": series}
    out = f"{out_dir}/{series}.npz"
    try:
        vol, extra = load_series(f"{root}/train_series/{study}/{series}", plane)
        data, off = _encode(vol)
        np.savez(out + ".tmp.npz", data=data, off=off)
        os.replace(out + ".tmp.npz", out)
        info.update(extra)
        info["bytes"] = int(len(data))
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"[:200]
    return info


if __name__ == "__main__":
    print("decoders:", _install_decoders(), flush=True)
    root = _find("train_series.csv")
    out_dir = "/kaggle/working/cache384"
    os.makedirs(out_dir, exist_ok=True)
    se = pd.read_csv(f"{root}/train_series.csv")
    pick = se.StudyInstanceUID.map(lambda s: int(hashlib.md5(str(s).encode()).hexdigest(), 16) % N_SHARDS == SHARD)
    se = se[pick].sort_values(["StudyInstanceUID", "SeriesInstanceUID"])
    tasks = [(root, a, b, p, out_dir) for a, b, p in zip(se.StudyInstanceUID, se.SeriesInstanceUID, se.Anatomical_Plane)]
    print(f"shard {SHARD}/{N_SHARDS}: {se.StudyInstanceUID.nunique()} studies, {len(tasks)} series | data {root}", flush=True)
    rows, total, stopped = [], 0, None
    with Pool(4) as p:
        for i, r in enumerate(p.imap(_shard_job, tasks, chunksize=2)):
            rows.append(r)
            total += r.get("bytes", 0)
            if i % 250 == 0:
                print(f"{i}/{len(tasks)} {total / 1e9:.2f} GB {(time.time() - T0) / 60:.0f} min", flush=True)
            if time.time() - T0 > BUDGET_S or total > MAX_BYTES:
                stopped = f"stopped at {i + 1}/{len(tasks)}: {'time' if time.time() - T0 > BUDGET_S else 'size'}"
                p.terminate()
                break
    df = pd.DataFrame(rows)
    df.to_csv("/kaggle/working/cache384_meta.csv", index=False)
    print(f"done {len(df)} series, errors {int(df['error'].notna().sum()) if 'error' in df else 0}, "
          f"{total / 1e9:.2f} GB, {(time.time() - T0) / 3600:.2f} h" + (f" | {stopped}" if stopped else ""), flush=True)
'''

for i in range(N):
    d = os.path.join(HERE, "..", "kaggle", f"gpjwcache_{i}")
    os.makedirs(d, exist_ok=True)
    src = prep + MAIN.replace("__SHARD__", str(i)).replace("__NSHARDS__", str(N))
    compile(src, "gpjwcache.py", "exec")
    open(os.path.join(d, "gpjwcache.py"), "w").write(src)
    meta = {"id": f"{OWNER}/rsna-knee-gpjw-cache-{i}", "title": f"rsna-knee-gpjw-cache-{i}", "code_file": "gpjwcache.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": False, "docker_image": DOCKER, "docker_image_pinning_type": "original",
            "dataset_sources": ["goodpjw2008/rsna-knee-2-5d-convnext-reader/1"],
            "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": []}
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
print(f"wrote {N} shard kernels")
