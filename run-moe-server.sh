#!/bin/sh
# Qwen3.8-Flash-Next MoE server (UD-IQ3_XXS / ISTA GSQ-RCO IQ3_XXS) with the
# built-in browser chat, same shape as run-qwen-server.sh for the dense model.
#
#   ./run-moe-server.sh            # UD-IQ3_XXS (default)
#   ./run-moe-server.sh ista       # ISTA GSQ-RCO IQ3_XXS
#
# Then open the printed address in the browser (the page is served by
# ds4-server itself: textarea + Send, no extra front-end needed).
set -eu

usage() {
    cat <<'EOF'
Usage: ./run-moe-server.sh [ud|ista]

  ud     Qwen3.8-Flash-Next-UD-IQ3_XXS (Unsloth dynamic, 3 shards) -- default
  ista   Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS (ISTA, 2 shards)

These are the MoE trunks: routed experts run on the CPU through the int8
kernels, which is what makes prefill fast on this machine (measured ~51-58
t/s on a 1739-token prompt versus ~5 t/s with the all-GPU expert path).

Environment overrides:
  DS4_CTX, DS4_CTK, DS4_CTV     context and matching KV types: f16/q8_0/q4_0
                                (defaults: 16384, q8_0)
  DS4_HOST, DS4_PORT            defaults: 127.0.0.1, 8080
  DS4_PREFILL_CHUNK             graph prefill chunk (default 1024; the CPU-MoE
                                prefill path covers T up to 4096)
  DS4_CPU_MOE                   1 (default) CPU experts, 0 all-GPU experts
  DS4_THREADS                   CPU threads for the expert kernels (default 16)
  DS4_CUDA_WEIGHT_CACHE_LIMIT_GB  device weight-cache budget (default 6)
  DS4_MOE_MODEL_DIR             model root (default ~/models/qwen38-flashnext-gguf)
  DS4_KV_DIR                    cache root (default ~/.ds4/server-kv-moe)
  DS4_TRACE                     default /dev/null: writing the trace to disk
                                costs about 40% of decode speed (8.5 -> 14.4 t/s
                                measured on UD-IQ3_XXS).  Set a path to enable it.
  DS4_THINK                    0 (default) or a level: none|minimal|low|medium|
                                high|xhigh|max.  Sets the default for clients
                                that send no thinking field (the llama.cpp web UI
                                sends none, so this is its switch); 1 = high.
                                An explicit field in the request still wins.
  DS4_WEBUI_DIR                 serve this directory as the web UI instead of the
                                built-in page.  Default: the prebuilt llama.cpp
                                webui when present, empty to use the built-in page.
  DS4_POWER, DS4_SYSTEM, DS4_PREFIX   as in run-qwen-server.sh
  DS4_DRY_RUN                   print the command line and exit

Notes:
  - ds4 is single-instance: stop any other ds4-server (e.g. the dense one) first.
  - If DS4_PORT is busy (llama-server often holds 8080) the script picks the
    next free port and prints it, unless you set DS4_PORT explicitly.
  - No --mtp here: with CPU experts the T=2 verify doubles CPU work and decode
    gets slower, so the draft head stays off.
EOF
}

[ "$#" -le 1 ] || { usage >&2; exit 2; }
PROFILE=${1:-ud}
case "$PROFILE" in -h|--help|help) usage; exit 0 ;; esac

ROOT=${DS4_MOE_MODEL_DIR:-$HOME/models/qwen38-flashnext-gguf}
HOST=${DS4_HOST:-127.0.0.1}
PORT=${DS4_PORT:-8080}
PORT_EXPLICIT=0
[ -n "${DS4_PORT:-}" ] && PORT_EXPLICIT=1
KV=${DS4_KV_DIR:-$HOME/.ds4/server-kv-moe}

case "$PROFILE" in
    ud)   MODEL=$ROOT/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
          DEFAULT_KV=q8_0 ;;
    ista) MODEL=$ROOT/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
          DEFAULT_KV=q8_0 ;;
    *) echo "unknown profile: $PROFILE" >&2; usage >&2; exit 2 ;;
esac

# The shipped Flash-Next vision encoder is BF16, which the vision loader does
# not accept (it wants f32/f16/q8_0), and chat does not need it.  Opt in with
# DS4_VISION=<path> if a compatible mmproj is available.
VISION=${DS4_VISION:-}
CTK=${DS4_CTK:-$DEFAULT_KV}
CTV=${DS4_CTV:-$CTK}
[ "$CTK" = "$CTV" ] || { echo "DS4_CTK and DS4_CTV must match" >&2; exit 2; }
case "$CTK" in f16|q8_0|q4_0) ;; *) echo "invalid KV type: $CTK" >&2; exit 2 ;; esac
CHUNK=${DS4_PREFILL_CHUNK:-1024}

# Thinking default for clients that send no thinking field (the web UI).
THINK=${DS4_THINK:-0}
case "$THINK" in
    0|1|off|on|none|minimal|low|medium|high|xhigh|max) export DS4_THINK_DEFAULT=$THINK ;;
    *) echo "DS4_THINK: usa 0/1 oppure none|minimal|low|medium|high|xhigh|max" >&2; exit 2 ;;
esac

# Web UI: prefer a prebuilt llama.cpp webui (same origin, served by ds4-server).
if [ -n "${DS4_WEBUI_DIR+x}" ]; then
    WEBUI=$DS4_WEBUI_DIR
else
    WEBUI=$HOME/.unsloth/llama.cpp/tools/server/public
    [ -f "$WEBUI/index.html" ] || WEBUI=""
fi
[ -z "$WEBUI" ] || [ -f "$WEBUI/index.html" ] || { echo "webui dir has no index.html: $WEBUI" >&2; exit 2; }

# CPU experts (the fast path). DS4_CPU_MOE=0 restores all-GPU experts.
case "${DS4_CPU_MOE:-1}" in
    1)
        export DS4_QWEN4_CPU_MOE=1
        export DS4_QWEN4_CUDA_STREAM_EXPERTS=1
        export DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=${DS4_CUDA_WEIGHT_CACHE_LIMIT_GB:-6}
        export DS4_THREADS=${DS4_THREADS:-16}
        MOE_NOTE="cpu-moe, ${DS4_THREADS} threads"
        ;;
    0)
        unset DS4_QWEN4_CPU_MOE DS4_QWEN4_CUDA_STREAM_EXPERTS DS4_CUDA_WEIGHT_CACHE_LIMIT_GB
        MOE_NOTE="gpu experts"
        ;;
    *) echo "DS4_CPU_MOE must be 0 or 1" >&2; exit 2 ;;
esac

cd "$(dirname "$0")"
[ -x ./ds4-server ] || { echo "Run: make ds4-server CUDA_ARCH=sm_120" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }
if [ -n "$VISION" ]; then
    [ -f "$VISION" ] || { echo "vision encoder not found: $VISION" >&2; exit 1; }
    set -- "$@" --vision "$VISION"
fi

# ds4 accepts a single instance at a time (dry runs do not start anything).
if [ -z "${DS4_DRY_RUN:-}" ] && pgrep -x ds4-server >/dev/null 2>&1; then
    echo "ds4-server is already running; stop it first (ds4 is single-instance)" >&2
    exit 2
fi

# Pick the next free port when the default one is taken (llama-server uses 8080).
PORT_IN_USE=0
if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q "[:.]$PORT "; then
    PORT_IN_USE=1
fi
if [ "$PORT_IN_USE" = 1 ]; then
    if [ "$PORT_EXPLICIT" = 1 ]; then
        echo "port $PORT is already in use; choose another with DS4_PORT" >&2
        exit 2
    fi
    for p in 8081 8082 8083 8084 8090; do
        if ! ss -ltn 2>/dev/null | grep -q "[:.]$p "; then PORT=$p; break; fi
    done
    PORT_IN_USE=0
    ss -ltn 2>/dev/null | grep -q "[:.]$PORT " && { echo "no free port in 8081-8090; set DS4_PORT" >&2; exit 2; }
fi

set -- --cuda --prefill-chunk "$CHUNK" --kv-cache-reject-different-quant
[ -n "$WEBUI" ] && set -- "$@" --webui-dir "$WEBUI"
if [ -n "${DS4_PREFIX:-}" ]; then set -- "$@" --prefix-file "$DS4_PREFIX"; fi
if [ -n "${DS4_POWER:-}" ]; then set -- "$@" --power "$DS4_POWER"; fi

if [ -n "${DS4_DRY_RUN:-}" ]; then
    printf 'ds4-server'
    for a in "$@"; do printf ' %s' "$a"; done
    printf ' -m %s -ctk %s -ctv %s -c %s --host %s --port %s --kv-disk-dir %s --kv-disk-space-mb 8192 --trace %s --system %s\n' \
        "$MODEL" "$CTK" "$CTV" "${DS4_CTX:-16384}" "$HOST" "$PORT" "$KV/$PROFILE" \
        "${DS4_TRACE:-/dev/null}" "${DS4_SYSTEM:-}"
    printf 'env: %s  (chunk %s, thinking %s, webui %s)\n' "$MOE_NOTE" "$CHUNK" \
    "$([ "$THINK" = 0 ] && echo off || echo on)" "${WEBUI:-built-in page}"
    exit 0
fi

printf '%s\n' "ds4-server MoE ($PROFILE, ctx ${DS4_CTX:-16384}, KV $CTK, $MOE_NOTE)" >&2
printf '%s\n' "  chat nel browser:  http://$HOST:$PORT/" >&2
printf '%s\n' "  (endpoint OpenAI-compatibili su http://$HOST:$PORT/v1)" >&2

exec ./ds4-server "$@" \
    -m "$MODEL" -ctk "$CTK" -ctv "$CTV" -c "${DS4_CTX:-16384}" \
    --host "$HOST" --port "$PORT" \
    --kv-disk-dir "$KV/$PROFILE" --kv-disk-space-mb 8192 \
    --trace "${DS4_TRACE:-/dev/null}" --system "${DS4_SYSTEM:-}"
