#!/bin/sh
# ds4-server with Qwen3.8-27B, ready to try.
#
# Four profiles:
#
#   ./run-qwen-server.sh           trunk + embedded draft head, ctx 16384  (chat)
#   ./run-qwen-server.sh sidecar   trunk + NVFP4 sidecar,      ctx 16384
#   ./run-qwen-server.sh long      trunk only,                 ctx 32768  (documents)
#   ./run-qwen-server.sh xxs       IQ3_XXS variant, draft,     ctx 49152  (+3.7% NLL, not the default)
#   ./run-qwen-server.sh q4        trunk only, q4_0 KV,        ctx 131072 (documents that
#                                                                        do not fit)
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
#
# `q4` is the profile for context that f16 cannot hold.  Attention K/V is 64 KiB
# per token as f16, and the allocator says so itself: at 131072 tokens the f16 K
# alone asks for 4096 MiB and the engine fails with "CUDA tensor alloc failed:
# out of memory".  q4_0 stores it at about a quarter, and this profile starts and
# serves at 131072 with 15411 MiB of VRAM used on a 16 GiB card.
#
# What the format buys is memory, and its decode cost grows with depth: 3% at 2048
# (47.1-47.6 tok/s against f16's 49.1-49.2) but 36% at 32768 (23.2 against 36.1).
# That cost was chased twice and is the format's, not a defect: templating the decode
# kernel on the format (bit-identical, measured) gained nothing, and a probe that read
# a fixed scale instead of each block's own did not speed it up either.
# The paths differ: the prefill widens a chunk once into an f16 mirror and runs the
# f16 kernel, so it is unaffected (531 against 528 tok/s at 32768), while the decode
# reads the quantized cache directly and dequantizes inside the attention loop, which
# at depth is most of the token.  At 131072 there is no f16 comparison to make: that
# cache does not allocate.  The NLL costs are in the server's -ctk help.
#
# ds4-bench is not the tool to price this with.  It fails to create a session at
# 131072 with q4_0 even with 13947 MiB free ("failed to allocate Qwen CUDA tensor
# hidden", 10 MiB), so it does not allocate the way the server does; and its
# --ctx-max is divided by the KV ratio for quantized types - 8192 asked measures
# at 2048 - so its ctx_tokens column is the context actually run, not the one
# requested.
#
# `xxs` is not the default and is not meant to become one: the release trunk is
# IQ3_S, and every other profile here uses it.  This one is the same chat shape as
# `merged` - the IQ3_XXS trunk carries its draft
# head inside it too - except that the trunk is 10.44 GiB against 11.77, so the
# draft fits at 49152 where the release trunk has to stop at 16384.  Measured on this
# card with a 6000-token prompt: 995.8 tok/s of prefill and 57.1 tok/s of decode at
# 49152, against 999.0 and 57.2 at 32768 - flat.  At 65536 the engine skips the draft
# by itself (its resident budget computes 14.50 GiB against the device) and the trunk
# carries on without it.  No warning is printed here: the resident-budget warning is
# tied to the separate NVFP4 sidecar, whose 24576 threshold was measured with the
# bigger trunk, and the embedded head is not that.  The price is quality: the same
# 16-token sentence reads NLL 1.87606303 against 1.80954673, confirmed against the
# CPU reference to 0.0027 nats.  The regression pins both values.
#
# Any profile can be adjusted without editing this file:
#
#   DS4_CTX      context tokens            (default: the profile's own)
#   DS4_CTK      KV cache type for K       (default f16; also q8_0, q4_0)
#   DS4_CTV      KV cache type for V       (must match DS4_CTK)
#   DS4_TRACE    trace file                (default /tmp/ds4-server.trace; /dev/null
#                                           discards it - it holds prompts and replies)
#   DS4_HOST     bind address              (default 127.0.0.1; 0.0.0.0 for the LAN)
#   DS4_PORT     port                      (default 8080)
#   DS4_POWER    1..100 duty cycle         (default: the engine's 100)
#   DS4_QWEN_MODEL_DIR  where the GGUFs are
#
# Every profile shares one disk KV directory, so
# --kv-cache-reject-different-quant is always on: without it a checkpoint written
# by the q4 profile could be resumed by an f16 one, and the restored cache would
# not be the cache that prompt was prefilled with.

set -e

M=${DS4_QWEN_MODEL_DIR:-/home/diegom/AI-Projects/llama.cpp/build/bin/models/Qwen3.8-27B}
HOST=${DS4_HOST:-127.0.0.1}
PORT=${DS4_PORT:-8080}
CTK=${DS4_CTK:-f16}
CTV=${DS4_CTV:-f16}
# The trace holds rendered prompts and replies in clear text.  /dev/null discards it.
TRACE=${DS4_TRACE:-/tmp/ds4-server.trace}
PROFILE=${1:-merged}
KV=${DS4_KV_DIR:-$HOME/.ds4/server-kv}

usage() {
    sed -n '3,10p' "$0" | sed 's/^# \{0,1\}//'
    cat <<'EOF'
Any profile accepts:
  DS4_CTX N        context tokens (default: the profile's own)
  DS4_CTK/DS4_CTV  f16 (default), q8_0 or q4_0; K and V must match
  DS4_HOST, DS4_PORT, DS4_POWER, DS4_QWEN_MODEL_DIR, DS4_KV_DIR
EOF
}

case "$PROFILE" in
    -h|--help|help) usage; exit 0 ;;
esac

if [ "$CTK" != "$CTV" ]; then
    echo "DS4_CTK=$CTK and DS4_CTV=$CTV differ: the engine requires K and V to match" >&2
    exit 2
fi

cd "$(dirname "$0")"

[ -x ./ds4-server ] || { echo "ds4-server not built. Run: make ds4-server CUDA_ARCH=sm_120" >&2; exit 1; }
[ -f "$M/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf" ] || { echo "model not found under $M" >&2; exit 1; }

# Not `[ -n ... ] && ...`: under set -e that form aborts the script whenever the
# variable is unset, which is its normal state.
POWER_ARGS=
if [ -n "$DS4_POWER" ]; then
    POWER_ARGS="--power $DS4_POWER"
fi

case "$PROFILE" in
xxs)
    printf '%s\n' "ds4-server (xxs: IQ3_XXS trunk + draft head, ctx ${DS4_CTX:-49152}, KV $CTK) on http://$HOST:$PORT" >&2
    exec ./ds4-server \
        -m "$M/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf" \
        --ctx "${DS4_CTX:-49152}" --host "$HOST" --port "$PORT" \
        -ctk "$CTK" -ctv "$CTV" --kv-cache-reject-different-quant \
        --vision "$M/mmproj-Qwen3.8-27B-BF16.gguf" \
        --mtp --mtp-draft 4 \
        --kv-disk-dir "$KV" --kv-disk-space-mb 8192 \
        --trace "$TRACE" $POWER_ARGS
    ;;
merged)
    # 16384 is the largest context where the draft head fits on a 16 GiB card
    # next to the KV (at 24576 the engine warns and everything slows down).  For
    # more context without drafting use the `long` profile, or `q4` for contexts
    # that need a smaller cache than f16.
    #
    # This comment sits above the command, not inside it: in sh a comment line
    # after a trailing backslash ends the command, so the options below would
    # silently vanish and the server would come up on the default port 8000 while
    # this script told you 8080.  That is what "the browser says error" was.
    printf '%s\n' "ds4-server (merged: trunk + draft head, ctx ${DS4_CTX:-16384}, KV $CTK) on http://$HOST:$PORT" >&2
    exec ./ds4-server \
        -m "$M/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf" \
        --ctx "${DS4_CTX:-16384}" --host "$HOST" --port "$PORT" \
        -ctk "$CTK" -ctv "$CTV" --kv-cache-reject-different-quant \
        --vision "$M/mmproj-Qwen3.8-27B-BF16.gguf" \
        --mtp --mtp-draft 4 \
        --kv-disk-dir "$KV" --kv-disk-space-mb 8192 \
        --trace "$TRACE" $POWER_ARGS
    ;;
sidecar)
    printf '%s\n' "ds4-server (sidecar, ctx ${DS4_CTX:-16384}, KV $CTK) on http://$HOST:$PORT" >&2
    exec ./ds4-server \
        -m "$M/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf" \
        --ctx "${DS4_CTX:-16384}" --host "$HOST" --port "$PORT" \
        -ctk "$CTK" -ctv "$CTV" --kv-cache-reject-different-quant \
        --vision "$M/mmproj-Qwen3.8-27B-BF16.gguf" \
        --mtp-model "$M/Qwen3.8-27B-NVFP4-MTP-HIGHEST.gguf" --mtp-draft 4 \
        --kv-disk-dir "$KV" --kv-disk-space-mb 8192 \
        --trace "$TRACE" $POWER_ARGS
    ;;
long)
    printf '%s\n' "ds4-server (long, ctx ${DS4_CTX:-32768}, KV $CTK, no draft head) on http://$HOST:$PORT" >&2
    exec ./ds4-server \
        -m "$M/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf" \
        --ctx "${DS4_CTX:-32768}" --host "$HOST" --port "$PORT" \
        -ctk "$CTK" -ctv "$CTV" --kv-cache-reject-different-quant \
        --vision "$M/mmproj-Qwen3.8-27B-BF16.gguf" \
        --kv-disk-dir "$KV" --kv-disk-space-mb 8192 \
        --trace "$TRACE" $POWER_ARGS
    ;;
q4)
    # The largest context this card can hold: q4_0 KV, no draft head (it would
    # not fit), so this is the profile for a document that does not fit in 32768.
    if [ "$CTK" = "f16" ]; then
        CTK=q4_0; CTV=q4_0
    fi
    printf '%s\n' "ds4-server (q4: ctx ${DS4_CTX:-131072}, KV $CTK) on http://$HOST:$PORT" >&2
    exec ./ds4-server \
        -m "$M/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf" \
        --ctx "${DS4_CTX:-131072}" --host "$HOST" --port "$PORT" \
        -ctk "$CTK" -ctv "$CTV" --kv-cache-reject-different-quant \
        --vision "$M/mmproj-Qwen3.8-27B-BF16.gguf" \
        --kv-disk-dir "$KV" --kv-disk-space-mb 8192 \
        --trace "$TRACE" $POWER_ARGS
    ;;
*)
    echo "unknown profile: $PROFILE" >&2
    usage >&2
    exit 2
    ;;
esac
