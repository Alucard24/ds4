#!/bin/sh
# ds4-server with Qwen3.8-27B, ready to try.
#
# Three profiles:
#
#   ./run-qwen-server.sh           trunk + embedded draft head, ctx 16384  (chat)
#   ./run-qwen-server.sh sidecar   trunk + NVFP4 sidecar,      ctx 16384
#   ./run-qwen-server.sh long      trunk only,                 ctx 32768  (documents)
#
# The first is the recommended one.  Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf is the
# same trunk with the draft head already inside it - 11.29 GiB against 10.96 +
# 1.09 for the separate sidecar - and since the draft block is Q6_K rather than
# NVFP4 it also decodes faster: 64.9 tok/s against 54.6 measured on the same
# prompt at short context.
#
# It used to ask for ctx 32768, where the draft head does not fit: the engine
# skips it with a message and the profile was identical to `long`.  Measured with
# the split-KV decode in place, at ctx 16384 drafting is worth +4.5% of decode
# (46.2 against 44.3 tok/s) and costs 22% of prefill (802 against 1033 tok/s,
# the weight cache thinning around the draft head); at ctx 24576 it does not fit
# at all and both numbers get worse (37.2 and 645).  So this profile is chat and
# `long` is for documents.

set -e

M=${DS4_QWEN_MODEL_DIR:-/home/diegom/AI-Projects/llama.cpp/build/bin/models/Qwen3.8-27B}
PORT=${DS4_PORT:-8080}
PROFILE=${1:-merged}


cd "$(dirname "$0")"

[ -x ./ds4-server ] || { echo "ds4-server not built. Run: make ds4-server CUDA_ARCH=sm_120" >&2; exit 1; }
[ -f "$M/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf" ] || { echo "model not found under $M" >&2; exit 1; }

case "$PROFILE" in
merged)
    # 16384 is the largest context where the draft head fits on a 16 GiB card
    # next to the KV (at 24576 the engine warns and everything slows down).  For
    # more context without drafting use the `long` profile; -ctk q8_0 -ctv q8_0
    # roughly doubles what fits if the format's prefill cost is acceptable.
    #
    # This comment sits above the command, not inside it: in sh a comment line
    # after a trailing backslash ends the command, so the options below would
    # silently vanish and the server would come up on the default port 8000 while
    # this script told you 8080.  That is what "the browser says error" was.
    printf '%s\n' "ds4-server (merged: trunk + draft head, ctx 16384) on http://127.0.0.1:$PORT" >&2
    exec ./ds4-server \
        -m "$M/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf" \
        --ctx 16384 --host 127.0.0.1 --port "$PORT" \
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
