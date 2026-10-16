"""E19: add dreaddevelopment's public Fine Spacing Raptor checkpoint (raptor-knee-finespacing v1, CC0,
raptor_ft_coatnet_v9_full.pt, solo public LB 0.932) as a fifth Raptor view of the 0.945 stack.

Single change against the parent notebook (E18 by default):
  * ASSET_ROOTS gains the raptor-knee-finespacing mount;
  * ARMS gains {"name": "finespacing-v9", ...} with exactly the MaxSpan preprocessing (same img, slots,
    span, k_eval), so it reuses MaxSpan's prepared windows on cuda:0 - one extra forward per study,
    no extra DICOM decode;
  * Raptor view weights: maxspan-v5 0.60 -> 0.30, finespacing-v9 0.30 (the two strongest checkpoints of
    the same family split the MaxSpan share equally; fixed before submitting, not tuned on the LB);
    native384dense 0.10, maxspan-reverse 0.10, native384-v8 0.20 unchanged.
  * Time gate: v9 runs only if the Raptor stage starts within E19_GATE_S of the notebook start;
    otherwise the original 4-view weights are restored and the output equals the parent's.

    python build_e19_finespacing.py [--parent submission_e18] [--out submission_e19] [--owner hosen42]
"""
import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--parent", default="submission_e18")
ap.add_argument("--out", default="submission_e19")
ap.add_argument("--owner", default="hosen42")
ap.add_argument("--gate_h", type=float, default=4.5)
a = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import NB, inp, out_dir, write_json  # noqa: E402  (one place for every path: ours_leg/paths.py)
par = inp(a.parent)
nb_name = next(f for f in os.listdir(par) if f.endswith(".ipynb"))
nb = json.load(open(os.path.join(par, nb_name)))
cells = nb["cells"]


def sub(cell_idx, old, new, count=1):
    s = "".join(cells[cell_idx]["source"])
    assert s.count(old) == count, (cell_idx, old[:80], s.count(old))
    cells[cell_idx]["source"] = s.replace(old, new).splitlines(keepends=True)


# 1. asset roots
sub(4, "ASSET_ROOTS = ['/kaggle/input/datasets/dreaddevelopment/raptor-knee-maxspan', ",
    "ASSET_ROOTS = ['/kaggle/input/datasets/dreaddevelopment/raptor-knee-finespacing', "
    "'/kaggle/input/raptor-knee-finespacing', '/kaggle/input/datasets/dreaddevelopment/raptor-knee-maxspan', ")

# 2. the fifth Raptor view (index 4, so the cuda:0=[0,2] / cuda:1=[1,3] assignment is untouched)
sub(30, '''     "img": 336, "slots": _SLOTS64, "span": (0.02, 0.98), "k_eval": 62,
     "reverse": False, "w": 0.60},''', '''     "img": 336, "slots": _SLOTS64, "span": (0.02, 0.98), "k_eval": 62,
     "reverse": False, "w": 0.30},''')
sub(30, '''     "img": 384, "slots": _SLOTS44, "span": (0.06, 0.94), "k_eval": 42,
     "reverse": False, "w": 0.20},
]''', '''     "img": 384, "slots": _SLOTS44, "span": (0.06, 0.94), "k_eval": 42,
     "reverse": False, "w": 0.20},
    # E19: public Fine Spacing checkpoint (dreaddevelopment/raptor-knee-finespacing v1, CC0); reuses the
    # MaxSpan windows on cuda:0. Weight 0.30 = half of MaxSpan's former 0.60, fixed before submitting.
    {"name": "finespacing-v9", "file": "raptor_ft_coatnet_v9_full.pt",
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,
     "img": 336, "slots": _SLOTS64, "span": (0.02, 0.98), "k_eval": 62,
     "reverse": False, "w": 0.30},
]''')

# 3. run v9 on MaxSpan's prepared windows (time-gated)
sub(30, '''        model, resolution = _KE_NS['load_model'](
            weight_path, first['arch'], first['res'], device
        )
        if int(resolution) != int(first['res']):
            raise RuntimeError(
                f"MaxSpan checkpoint resolution {resolution} != {first['res']}"
            )
''', f'''        model, resolution = _KE_NS['load_model'](
            weight_path, first['arch'], first['res'], device
        )
        if int(resolution) != int(first['res']):
            raise RuntimeError(
                f"MaxSpan checkpoint resolution {{resolution}} != {{first['res']}}"
            )
        # E19: Fine Spacing v9 shares MaxSpan's prepared windows; skipped (weights restored) if late.
        fine_index = next((i for i, x in enumerate(arms) if x['name'] == 'finespacing-v9'), None)
        fine_model = None
        if fine_index is not None:
            fine = arms[fine_index]
            if any(first[key] != fine[key] for key in ('arch', 'res', 'img', 'slots', 'span', 'k_eval')):
                raise RuntimeError('finespacing-v9 no longer shares MaxSpan preprocessing')
            elapsed = _ke_time.time() - T0
            if elapsed < {a.gate_h} * 3600:
                fine_model, fine_res = _KE_NS['load_model'](
                    _KE_NS['find_weight_file'](fine['file']), fine['arch'], fine['res'], device)
                if int(fine_res) != int(fine['res']):
                    raise RuntimeError(f"finespacing-v9 checkpoint resolution {{fine_res}} != {{fine['res']}}")
                print(f"[E19] finespacing-v9 ON at {{elapsed / 3600:.2f}} h; Raptor weights "
                      f"{{[(x['name'], x['w']) for x in arms]}}", flush=True)
            else:
                arms[fine_index]['w'] = 0.0
                first['w'] = 0.60
                print(f"[E19] finespacing-v9 SKIPPED at {{elapsed / 3600:.2f}} h; parent weights restored", flush=True)
''')
sub(30, '''                outputs[2][study_index] = _KE_NS['infer_probs'](
                    model, windows.flip(1).contiguous(), device
                )
                del windows''', '''                outputs[2][study_index] = _KE_NS['infer_probs'](
                    model, windows.flip(1).contiguous(), device
                )
                if fine_model is not None:
                    outputs[fine_index][study_index] = _KE_NS['infer_probs'](
                        fine_model, windows, device
                    )
                del windows''')
sub(30, '''        del model
        _ke_gc.collect()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

    def gpu_zero():''', '''        del model, fine_model
        _ke_gc.collect()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

    def gpu_zero():''')

out = out_dir(a.out)
os.makedirs(out, exist_ok=True)
slug = "rsna-knee-" + a.out.replace("submission_", "")
write_json(os.path.join(out, slug + ".ipynb"), nb, indent=1)
meta = json.load(open(os.path.join(par, "kernel-metadata.json")))
meta.update(id=f"{a.owner}/{slug}", title=slug, code_file=slug + ".ipynb")
if "dreaddevelopment/raptor-knee-finespacing/1" not in meta["dataset_sources"]:
    meta["dataset_sources"] = meta["dataset_sources"] + ["dreaddevelopment/raptor-knee-finespacing/1"]
write_json(os.path.join(out, "kernel-metadata.json"), meta, indent=2)
compile("".join(cells[30]["source"]).split("_KE_SRC = r'''")[0], "cell30_head", "exec")
print("wrote", out, "| sources", len(meta["dataset_sources"]), "| kernels", meta["kernel_sources"])
