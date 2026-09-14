#!/bin/sh
# Qwen CUDA server profiles. IQ3_S remains the default; IQ3_XXS is opt-in.
set -eu

usage() {
    cat <<'EOF'
Usage: ./run-qwen-server.sh [merged|sidecar|long|q4|xxs]
  merged   IQ3_S + embedded MTP, 16384 context (default)
  sidecar  IQ3_S + NVFP4 MTP sidecar, 16384 context
  long     IQ3_S, no MTP, 32768 context
  q4       IQ3_S, no MTP, q4_0 KV, 131072 allocated context
  xxs      IQ3_XXS + embedded MTP, 49152 allocated context
Capacity is not a guarantee of full-depth throughput or MTP residency.

Environment overrides:
  DS4_CTX, DS4_CTK, DS4_CTV     context and matching KV types: f16/q8_0/q4_0
  DS4_HOST, DS4_PORT           defaults: 127.0.0.1, 8080
  DS4_POWER                   0..100 (default: engine's 100)
  DS4_SYSTEM, DS4_PREFIX       extra system text and optional example-turn file
  DS4_TRACE                   default /tmp/ds4-server.trace; /dev/null discards it
  DS4_QWEN_MODEL_DIR           model directory
  DS4_KV_DIR                   cache root (default ~/.ds4/server-kv)
  DS4_STEERING_FILE            optional 64x5120 f32 direction matrix
  DS4_STEERING_FFN/ATTN        scales (defaults with a file: 1 / 0)
Steered runs use a separate cache subdirectory. Runtime payload validation also
rejects incompatible steering. Active steering uses serial MTP verification;
it does not inherit the normal batched MTP speedup. No GGUF weights are changed.
EOF
}

[ "$#" -le 1 ] || { usage >&2; exit 2; }
PROFILE=${1:-merged}
case "$PROFILE" in -h|--help|help) usage; exit 0 ;; esac
M=${DS4_QWEN_MODEL_DIR:-/home/diegom/AI-Projects/llama.cpp/build/bin/models/Qwen3.8-27B}
HOST=${DS4_HOST:-127.0.0.1}
PORT=${DS4_PORT:-8080}
KV=${DS4_KV_DIR:-$HOME/.ds4/server-kv}
# Accumulate argv, not space-delimited strings: prompt/direction paths can contain
# whitespace or wildcard characters. Keep optional flags independent of each other.
set --
case "$PROFILE" in
    merged)
        MODEL=$M/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
        CTX=16384; DEFAULT_KV=f16
        set -- --mtp --mtp-draft 4 ;;
    sidecar)
        MODEL=$M/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf
        CTX=16384; DEFAULT_KV=f16
        set -- --mtp-model "$M/Qwen3.8-27B-NVFP4-MTP-HIGHEST.gguf" --mtp-draft 4 ;;
    long)
        MODEL=$M/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf
        CTX=32768; DEFAULT_KV=f16 ;;
    q4)
        MODEL=$M/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf
        CTX=131072; DEFAULT_KV=q4_0 ;;
    xxs)
        MODEL=$M/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf
        CTX=49152; DEFAULT_KV=f16
        set -- --mtp --mtp-draft 4 ;;
    *) echo "unknown profile: $PROFILE" >&2; usage >&2; exit 2 ;;
esac
CTK=${DS4_CTK:-$DEFAULT_KV}
CTV=${DS4_CTV:-$CTK}
[ "$CTK" = "$CTV" ] || { echo "DS4_CTK and DS4_CTV must match" >&2; exit 2; }
case "$CTK" in f16|q8_0|q4_0) ;; *) echo "invalid KV type: $CTK" >&2; exit 2 ;; esac

cd "$(dirname "$0")"
[ -x ./ds4-server ] || { echo "Run: make ds4-server CUDA_ARCH=sm_120" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }
if [ -n "${DS4_PREFIX:-}" ]; then set -- "$@" --prefix-file "$DS4_PREFIX"; fi
if [ -n "${DS4_POWER:-}" ]; then set -- "$@" --power "$DS4_POWER"; fi
if [ -n "${DS4_STEERING_FILE:-}" ]; then
    FFN=${DS4_STEERING_FFN:-1}; ATTN=${DS4_STEERING_ATTN:-0}
    direction_hash=$(sha256sum -- "$DS4_STEERING_FILE")
    direction_hash=${direction_hash%% *}
    identity=$(printf '%s\n' "$MODEL" "$direction_hash" "$FFN" "$ATTN" "$CTK" | sha256sum)
    identity=${identity%% *}
    KV=$KV/steered/$identity
    set -- "$@" --dir-steering-file "$DS4_STEERING_FILE" \
        --dir-steering-ffn "$FFN" --dir-steering-attn "$ATTN"
elif [ -n "${DS4_STEERING_FFN:-}${DS4_STEERING_ATTN:-}" ]; then
    echo "DS4_STEERING_FFN/ATTN require DS4_STEERING_FILE" >&2; exit 2
fi

printf '%s\n' "ds4-server ($PROFILE, ctx ${DS4_CTX:-$CTX}, KV $CTK) on http://$HOST:$PORT" >&2
exec ./ds4-server "$@" \
    -m "$MODEL" --ctx "${DS4_CTX:-$CTX}" --host "$HOST" --port "$PORT" \
    -ctk "$CTK" -ctv "$CTV" --kv-cache-reject-different-quant \
    --vision "$M/mmproj-Qwen3.8-27B-BF16.gguf" \
    --kv-disk-dir "$KV" --kv-disk-space-mb 8192 \
    --trace "${DS4_TRACE:-/tmp/ds4-server.trace}" --system "${DS4_SYSTEM:-}"
