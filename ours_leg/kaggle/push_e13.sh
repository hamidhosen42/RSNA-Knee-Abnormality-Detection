#!/bin/zsh
# Push the E13 cache builder with knee_prep.py embedded verbatim (CPU kernels, no GPU quota).
#   ./push_e13.sh cache [LIMIT]   probe: first LIMIT train studies, kernel rsna-knee-e13-cache
#   ./push_e13.sh shard <0|1>     full build of one half, kernel rsna-knee-e13-cache-<a|b>
set -euo pipefail
HERE=${0:A:h}; E13=${HERE:h}/e13
PY=${PYTHON:-$HOME/.local/share/uv/python/cpython-3.12.14-macos-aarch64-none/bin/python3.12}
UVX=${UVX:-$(command -v uvx || echo $HOME/Library/Python/3.9/bin/uvx)}
MODE=${1:-}; ARG=${2:-0}
[[ $MODE == cache || $MODE == shard ]] || { echo "usage: $0 cache [LIMIT] | shard <0|1>" >&2; exit 2; }
DIR=$(mktemp -d)
"$PY" - "$E13/build_cache.py" "$E13/knee_prep.py" "$HERE/e13cache.kernel-metadata.json" "$DIR" "$MODE" "$ARG" <<'PYEOF'
import base64, json, sys
src, prep, meta, out, mode, arg = sys.argv[1:7]
t = open(src).read()
assert t.count("__KNEE_PREP_B64__") == 1
t = t.replace("__KNEE_PREP_B64__", base64.b64encode(open(prep, "rb").read()).decode())
m = json.load(open(meta))
if mode == "cache":
    t = t.replace('os.environ.get("E13_LIMIT", "0")', f'os.environ.get("E13_LIMIT", "{int(arg)}")')
else:
    shard = int(arg); assert shard in (0, 1)
    t = t.replace('os.environ.get("E13_SHARD", "0")', f'os.environ.get("E13_SHARD", "{shard}")')
    t = t.replace('os.environ.get("E13_N_SHARDS", "1")', 'os.environ.get("E13_N_SHARDS", "2")')
    m["id"] = m["title"] = f"hosen42/rsna-knee-e13-cache-{'ab'[shard]}"
    m["title"] = m["id"].split("/")[1]
compile(t, "build_cache.py", "exec")
open(f"{out}/build_cache.py", "w").write(t)
json.dump(m, open(f"{out}/kernel-metadata.json", "w"), indent=1)
print("prepared", m["id"])
PYEOF
export KAGGLE_API_TOKEN=$(tr -d '\n\r ' < ~/.kaggle/access_token)
[[ -n ${DRY_RUN:-} ]] && { ls -la "$DIR"; grep -n 'E13_SHARD\|E13_N_SHARDS\|E13_LIMIT' "$DIR/build_cache.py"; cat "$DIR/kernel-metadata.json"; rm -rf "$DIR"; exit 0; }
$UVX --from 'kaggle>=2.2' kaggle kernels push -p "$DIR"
rm -rf "$DIR"
