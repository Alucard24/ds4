#!/bin/sh
# ds4-server with Qwen3.8-27B, ready to try.
#
# Three profiles:
#
#   ./run-qwen-server.sh           trunk + embedded draft head, ctx 24576  (best)
#   ./run-qwen-server.sh sidecar   trunk + NVFP4 sidecar,      ctx 16384
#   ./run-qwen-server.sh long      trunk only,                 ctx 32768
#
# The first is the recommended one.  Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf is the
# same trunk with the draft head already inside it - 11.29 GiB against 10.96 +
# 1.09 for the separate sidecar - and since the draft block is Q6_K rather than
# NVFP4 it also decodes faster: 64.9 tok/s against 54.6 measured on the same
# prompt.  What those 0.76 GiB buy is context: 24576 tokens with the draft head
# active, where the sidecar combination is refused above 16384.  28672 is
# refused, so 24576 is the edge of what fits on a 16 GiB card.

set -e

M=${DS4_QWEN_MODEL_DIR:-/home/diegom/AI-Projects/llama.cpp/build/bin/models/Qwen3.8-27B}
PORT=${DS4_PORT:-8080}
PROFILE=${1:-merged}


cd "$(dirname "$0")"

[ -x ./ds4-server ] || { echo "ds4-server not built. Run: make ds4-server CUDA_ARCH=sm_120" >&2; exit 1; }
[ -f "$M/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf" ] || { echo "model not found under $M" >&2; exit 1; }

case "$PROFILE" in
merged)
    # The draft head does not fit next to a 32768 context on a 16 GiB card: the
    # engine says so and skips it, so this profile is long context without
    # drafting.  For MTP use ctx 24576 or the KV type switch, -ctk q8_0 -ctv q8_0.
    #
    # This comment sits above the command, not inside it: in sh a comment line
    # after a trailing backslash ends the command, so the options below would
    # silently vanish and the server would come up on the default port 8000 while
    # this script told you 8080.  That is what "the browser says error" was.
    printf '%s\n' "ds4-server (merged, ctx 32768, MTP off) on http://127.0.0.1:$PORT" >&2
    exec ./ds4-server \
        -m "$M/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf" \
        --ctx 32768 --host 127.0.0.1 --port "$PORT" \
        --vision "$M/mmproj-Qwen3.8-27B-BF16.gguf" \
        --mtp --mtp-draft 4 \
        --kv-disk-dir "$HOME/.ds4/server-kv" --kv-disk-space-mb 8192 \
        --trace /tmp/ds4-server.trace
    ;;
sidecar)
    printf '%s\n' "ds4-server (sidecar, ctx 16384) on http://127.0.0.1:$PORT" >&2
    exec ./ds4-server \
        -m "$M/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf" \
        --ctx 16384 --host 127.0.0.1 --port "$PORT" \
        --vision "$M/mmproj-Qwen3.8-27B-BF16.gguf" \
        --mtp-model "$M/Qwen3.8-27B-NVFP4-MTP-HIGHEST.gguf" --mtp-draft 4 \
        --kv-disk-dir "$HOME/.ds4/server-kv" --kv-disk-space-mb 8192 \
        --trace /tmp/ds4-server.trace
    ;;
long)
    printf '%s\n' "ds4-server (long, ctx 32768, no draft head) on http://127.0.0.1:$PORT" >&2
    exec ./ds4-server \
        -m "$M/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf" \
        --ctx 32768 --host 127.0.0.1 --port "$PORT" \
        --vision "$M/mmproj-Qwen3.8-27B-BF16.gguf" \
        --kv-disk-dir "$HOME/.ds4/server-kv" --kv-disk-space-mb 8192 \
        --trace /tmp/ds4-server.trace
    ;;
*)
    echo "usage: $0 [fast|long]" >&2
    exit 2
    ;;
esac
