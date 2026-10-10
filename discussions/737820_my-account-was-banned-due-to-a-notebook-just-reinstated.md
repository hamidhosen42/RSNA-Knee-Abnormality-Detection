# My account was banned due to a notebook. Just reinstated 

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737820
- **Author:** **Jiwei Liu** (@jiweiliu, Grandmaster) · _topic author_ · 2026-08-27 16:30 UTC · ▲10 · LB rank 80
- **Topic votes:** 10 · **Comments/replies:** 5 · **Pinned:** no
- **Last comment:** 2026-09-26 00:23 UTC by Chris Deotte
- **Cloned:** 2026-10-09 09:46 UTC (topic id 737820, raw JSON `_raw/737820.json`)

---

Hello all, I'd like to share a scary experience. My account was banned 8 hours ago. I submitted a appeal and my account is just reinstated so I can publish this post.

I got this email about accoutn ban:
```
Hi Jiwei Liu,
Our automated content review system recently found that your content is not compliant with one or more of our policies. See below for more information about your content status and how to correct the issue.

Content:
Notebook: RSNA Knee M1 C9 MaxSpan Inductor cache T4 (2026-08-27 08:10)

Source of Report:Automated systems
Issue Found: Violates our Community Guidelines and/or Termsagainst Resource abuse. For further explanation of why the content and/or use of the platform is considered violative on these grounds, please refer to Kaggle’s Community Guidelines.
Result: We have unpublished the content and issued a ban on your account, unless we determine otherwise after an appeal.
```

The notebook referenced is a simple python script:
```
"""Build a reusable T4 TorchInductor cache for the M1-C9 MaxSpan graph."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tarfile
import time


REFERENCE_PARENT_MANIFEST_SHA256 = (
    "e71afd1717aa9c2751a7eafc36960c480a7f10f1da841d26d53cd331b47e51fd"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def package() -> Path:
    paths = sorted(
        path.resolve()
        for path in Path("/kaggle/input").glob("**/m0_c3_manifest.json")
        if path.is_file()
    )
    if len(paths) != 1:
        raise FileNotFoundError(
            f"expected one m0_c3_manifest.json, observed {len(paths)}"
        )
    return paths[0].parent


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import runtime: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    work = Path("/kaggle/working")
    cache = Path("/tmp/torchinductor_cache")
    cache.mkdir(parents=True, exist_ok=False)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"

    import numpy as np
    import torch

    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"cache builder requires two visible GPUs, observed "
            f"{torch.cuda.device_count()}"
        )
    gpu_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if any("T4" not in name for name in gpu_names):
        raise RuntimeError(f"cache builder requires T4 GPUs, observed {gpu_names}")

    parent = package()
    manifest_path = parent / "m0_c3_manifest.json"
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != REFERENCE_PARENT_MANIFEST_SHA256:
        raise RuntimeError(
            f"parent manifest drift: expected={REFERENCE_PARENT_MANIFEST_SHA256} "
            f"observed={manifest_hash}"
        )
    manifest = json.loads(manifest_path.read_text())
    checkpoint = parent / "maxspan_member0.pth"
    checkpoint_hash = sha256_file(checkpoint)
    expected_checkpoint_hash = manifest["files"].get(checkpoint.name)
    if checkpoint_hash != expected_checkpoint_hash:
        raise RuntimeError(
            f"checkpoint drift: expected={expected_checkpoint_hash} "
            f"observed={checkpoint_hash}"
        )

    runtime_path = parent / "maxspan_runtime.py"
    runtime_hash = sha256_file(runtime_path)
    expected_runtime_hash = manifest["files"].get(runtime_path.name)
    if runtime_hash != expected_runtime_hash:
        raise RuntimeError(
            f"runtime drift: expected={expected_runtime_hash} "
            f"observed={runtime_hash}"
        )
    runtime = import_file("m1_c9_fast_maxspan_runtime", runtime_path)

    device_audits = []
    for index in range(2):
        device = torch.device(f"cuda:{index}")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(index)
        model = runtime.load_model(checkpoint, device)
        image = torch.zeros(
            (
                1,
                runtime.EVAL_WINDOWS,
                3,
                runtime.MODEL_SIZE,
                runtime.MODEL_SIZE,
            ),
            dtype=torch.float32,
            device=device,
        )

        def invoke(candidate):
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.float16
            ):
                return candidate(image).float()

        torch.cuda.synchronize(device)
        eager_started = time.monotonic()
        eager = invoke(model)
        torch.cuda.synchronize(device)
        eager_seconds = time.monotonic() - eager_started

        compiled = torch.compile(
            model,
            backend="inductor",
            mode="default",
            fullgraph=False,
            dynamic=False,
        )
        torch.cuda.synchronize(device)
        compile_started = time.monotonic()
        compiled_first = invoke(compiled)
        torch.cuda.synchronize(device)
        compile_first_seconds = time.monotonic() - compile_started

        torch.cuda.synchronize(device)
        steady_started = time.monotonic()
        compiled_steady = invoke(compiled)
        torch.cuda.synchronize(device)
        steady_seconds = time.monotonic() - steady_started
        eager_values = eager.detach().cpu().numpy()
        first_values = compiled_first.detach().cpu().numpy()
        steady_values = compiled_steady.detach().cpu().numpy()
        device_audits.append(
            {
                "device": index,
                "name": gpu_names[index],
                "eager_seconds": eager_seconds,
                "compile_first_seconds": compile_first_seconds,
                "compiled_steady_seconds": steady_seconds,
                "max_abs_compiled_vs_eager": float(
                    np.max(np.abs(first_values - eager_values))
                ),
                "max_abs_compiled_repeat": float(
                    np.max(np.abs(steady_values - first_values))
                ),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(index),
            }
        )
        del compiled_steady, compiled_first, eager, image, compiled, model
        torch.cuda.empty_cache()

    archive = work / "m1_c9_maxspan_inductor_cache_t4.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=6) as handle:
        handle.add(cache, arcname=cache.name)
    cache_files = sorted(path for path in cache.rglob("*") if path.is_file())
    cache_bytes = sum(path.stat().st_size for path in cache_files)
    archive_hash = sha256_file(archive)
    builder_source_hash = sha256_file(Path(__file__).resolve())
    shutil.rmtree(cache)
    if cache.exists():
        raise RuntimeError(f"failed to remove raw cache directory: {cache}")
    receipt = {
        "status": "VALID_M1_C9_MAXSPAN_INDUCTOR_CACHE_T4",
        "artifact_version": 3,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder_source_sha256": builder_source_hash,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "triton": importlib.metadata.version("triton"),
        "gpu_names": gpu_names,
        "gpu_capabilities": [
            list(torch.cuda.get_device_capability(index)) for index in range(2)
        ],
        "compile": {
            "backend": "inductor",
            "mode": "default",
            "fullgraph": False,
            "dynamic": False,
            "input_shape": [
                1,
                runtime.EVAL_WINDOWS,
                3,
                runtime.MODEL_SIZE,
                runtime.MODEL_SIZE,
            ],
            "input_dtype": "torch.float32",
            "autocast_dtype": "torch.float16",
            "inference_mode": True,
        },
        "cache_runtime_path": str(cache),
        "parent_manifest_sha256": manifest_hash,
        "maxspan_runtime_sha256": runtime_hash,
        "checkpoint_name": checkpoint.name,
        "checkpoint_sha256": checkpoint_hash,
        "device_audits": device_audits,
        "cache_files": len(cache_files),
        "cache_bytes": cache_bytes,
        "archive_name": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_hash,
        "raw_cache_removed": True,
    }
    receipt_path = work / "m1_c9_maxspan_inductor_cache_t4_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
```

What's interesting is I pushed 3 versions of this kernel and only version 3 triggered the account ban somehow. Here is version 2-3 diff:
```
def main() -> None:
     work = Path("/kaggle/working")
-    cache = work / "torchinductor_cache"
+    cache = Path("/tmp/torchinductor_cache")

     archive = work / "m1_c9_maxspan_inductor_cache_t4.tar.gz"
     with tarfile.open(archive, "w:gz", compresslevel=6) as handle:
-        handle.add(cache, arcname="torchinductor_cache")
+        handle.add(cache, arcname=cache.name)

-        "artifact_version": 2,
+        "artifact_version": 3,

+        "cache_runtime_path": str(cache),
```

---

## Comments (5, threaded, oldest first)

### [1] Jiwei Liu

**Jiwei Liu** (@jiweiliu, Grandmaster) · _topic author_ · 2026-08-27 16:32 UTC · ▲3 · LB rank 80

It is a direct account ban. Not the usual warning, suspension and ban sequence. I was shocked

#### ↳ [1.1] CroDoc

**CroDoc** (@crodoc, Grandmaster) · 2026-08-27 18:28 UTC · ▲2

good thing you are back, now back to farming... kaggriculture 👨‍🌾

### [2] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-08-27 17:14 UTC · ▲4 · LB rank 641

It's interesting when an AI makes a judgement that's wrong.  I got a suspension a bit ago for a joke that the AI felt was a personal attack.   Only a week of not participating in discussions - no biggie.  

I love AI.  I think it will bring lots to the world.

But you will never see me get into an AI driven taxi :)

### [3] Komil Parmar

**Komil Parmar** (@komilparmar, Expert) · 2026-08-27 18:05 UTC · LB rank 59

Rest aside, quite a clever technique sir. I am thinking of using it on colab, but yeah will have to check colab's policy first. lol! 
Hope your account is back online in its original state ASAP!

### [4] Chris Deotte

**Chris Deotte** (@cdeotte, Grandmaster) · 2026-09-26 00:23 UTC · LB rank 93

Glad you are back! Welcome back!
