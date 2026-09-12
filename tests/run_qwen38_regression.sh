#!/bin/sh
# Consolidated Qwen-only regression: CPU reference, CUDA trunk numerics, vision,
# video, MTP speculation and the server protocol tests, in one pass.
#
# Run this after any structural change, not only before a tag.  The CUDA gates
# are what a kernel change is judged by, and they cannot see the CPU build: that
# one stayed broken for half a day behind green CUDA gates because the embedded
# MTP commit called CUDA-only helpers from shared engine code (fixed in 5e1d9cd).
# A gate that passes is not the whole story.
#
# Model paths come from DS4_QWEN_REGRESSION_MODEL_DIR, and the individual files
# from DS4_QWEN_TEST_MODEL / DS4_QWEN_TEST_MTP / DS4_QWEN_TEST_MMPROJ, so this
# runs on a tree whose models live somewhere else.
set -e

MODEL_DIR=${DS4_QWEN_REGRESSION_MODEL_DIR:-/home/diegom/AI-Projects/llama.cpp/build/bin/models/Qwen3.8-27B}
MODEL=${DS4_QWEN_TEST_MODEL:-$MODEL_DIR/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf}
MTP=${DS4_QWEN_TEST_MTP:-$MODEL_DIR/Qwen3.8-27B-NVFP4-MTP-HIGHEST.gguf}
MMPROJ=${DS4_QWEN_TEST_MMPROJ:-$MODEL_DIR/mmproj-Qwen3.8-27B-BF16.gguf}
GGML_CPU=${DS4_TEST_GGML_CPU:-/home/diegom/AI-Projects/llama.cpp/build/bin/libggml-cpu.so}
PROMPT='The capital of France is Paris. The largest ocean on Earth is the Pacific Ocean.'

cd "$(dirname "$0")/.."

echo "===== all five binaries ====="
make CUDA_ARCH=sm_120

echo "===== CPU reference build and tests ====="
make clean >/dev/null 2>&1
make -B test-qwen38-cpu

echo "===== mixed-IQ CUDA formats, batch16 MMQ ====="
make test-qwen38-cuda DS4_TEST_GGML_CPU="$GGML_CPU"

echo "===== CUDA trunk numerics, session, DSV4 replay ====="
make tests/test_qwen38_session_cuda CUDA_ARCH=sm_120
DS4_TEST_QWEN38_CUDA=1 ./tests/test_qwen38_session_cuda "$MODEL" "$PROMPT"

echo "===== the same gate with the q8_0 KV cache ====="
# DS4_KV_Q8 and DS4_KV_Q4 are the hooks the test binaries use, since they take no
# engine options.  What is asserted is the number itself, which is the thing that
# would move if a quantized path regressed.  Note what this test can and cannot
# say: over 16 tokens the three formats are 1.80954673, 1.80900178 and 1.82242633,
# and the difference between the first two is inside its noise - the format
# comparison that means something is the recall test, where all three answer four
# for four with identical text.
# Two explicit invocations, not one clever loop: setting an environment variable
# to the empty string still defines it, so a conditional assignment turned both
# hooks on at once and q4_0's number was reported for q8_0.
nll_q8=$(DS4_KV_Q8=1 DS4_TEST_QWEN38_CUDA=1 ./tests/test_qwen38_session_cuda "$MODEL" "$PROMPT" 2>&1 | grep '^MEAN_NLL' || true)
if [ "$nll_q8" = "MEAN_NLL 1.80900178 TOKENS 16" ]; then
    echo "q8_0 KV NLL as recorded: $nll_q8"
else
    echo "q8_0 KV NLL changed from the recorded 1.80900178: $nll_q8"
    exit 1
fi
nll_q4=$(DS4_KV_Q4=1 DS4_TEST_QWEN38_CUDA=1 ./tests/test_qwen38_session_cuda "$MODEL" "$PROMPT" 2>&1 | grep '^MEAN_NLL' || true)
if [ "$nll_q4" = "MEAN_NLL 1.82242633 TOKENS 16" ]; then
    echo "q4_0 KV NLL as recorded: $nll_q4"
else
    echo "q4_0 KV NLL changed from the recorded 1.82242633: $nll_q4"
    exit 1
fi

echo "===== the prefill, which both gates above miss ====="
# The ds4 binary from the top of this script is gone by now: the CPU reference
# step runs `make clean`.  Build it again rather than depend on ordering.
make ds4 CUDA_ARCH=sm_120
# Those two numbers are decode numbers: the session test evaluates the prompt one
# token at a time, so it never runs the chunk kernel a real request's prefill
# uses.  That kernel read half the rows for a day - the quantized twin kept the
# eight-rows-per-block indexing after the f16 one moved to sixteen - and every
# gate above stayed green, because on this prompt the greedy continuation is
# identical either way.  So the check is the one that saw it: the same prefill in
# all three formats, compared to f16 rather than to a fixed number, so it stays
# valid through numerics-neutral work.  The tolerances are calibrated by both
# sides: on this sixteen-token prompt the correct kernel measures 0.10 (q8_0) and
# 0.23 (q4_0) against f16, and the half-rows kernel measured 5.37 and 5.39, so
# the line sits between them with a factor of five on each side.
prefill_logit() {
    fmt=$1
    opt=""
    [ "$fmt" != "f16" ] && opt="-ctk $fmt -ctv $fmt"
    ./ds4 -m "$MODEL" -c 4096 --raw --temp 0 $opt \
        --dump-logprobs "/tmp/ds4-prefill-$fmt.json" --logprobs-top-k 1 \
        -p "$PROMPT" -n 1 >/dev/null 2>&1
    sed -n 's/.*"logit":\([-0-9.]*\).*/\1/p' "/tmp/ds4-prefill-$fmt.json" | head -1
}
l_f16=$(prefill_logit f16)
l_q8=$(prefill_logit q8_0)
l_q4=$(prefill_logit q4_0)
if [ -z "$l_f16" ] || [ -z "$l_q8" ] || [ -z "$l_q4" ]; then
    echo "prefill logit check could not read a first-token logit: f16='$l_f16' q8_0='$l_q8' q4_0='$l_q4'"
    exit 1
fi
awk -v a="$l_f16" -v b="$l_q8" -v c="$l_q4" 'BEGIN {
    d8 = (b > a ? b - a : a - b);
    d4 = (c > a ? c - a : a - c);
    printf "prefill first-token logit: f16 %s, q8_0 %s (delta %.4f, limit 0.50), q4_0 %s (delta %.4f, limit 0.80)\n", a, b, d8, c, d4;
    if (d8 > 0.50 || d4 > 0.80) {
        print "a quantized prefill has drifted from f16 by more than the quantization explains";
        exit 1;
    }
}'

echo "===== tokenizer vectors ====="
make test-tokenizer-vectors 2>/dev/null || echo "(tokenizer vectors target not present; covered by run_qwen38_cpu.sh)"

echo "===== Qwen3-VL image and temporal frame merge ====="
make tests/test_qwen3vl_vision CUDA_ARCH=sm_120
./tests/test_qwen3vl_vision "$MODEL" "$MMPROJ" -   # '-' materialises the embedded fixture

echo "===== Qwen3-VL session, payload replay, video ====="
make tests/test_qwen3vl_session CUDA_ARCH=sm_120
./tests/test_qwen3vl_session "$MODEL" "$MMPROJ" -

echo "===== video container input (ffmpeg) ====="
make tests/test_qwen3vl_video CUDA_ARCH=sm_120
./tests/test_qwen3vl_video "$MODEL" "$MMPROJ"

echo "===== MTP draft and speculative decoding ====="
make tests/test_qwen38_mtp CUDA_ARCH=sm_120
./tests/test_qwen38_mtp "$MODEL" "$MTP" "$PROMPT"

echo "===== server protocol tests ====="
make ds4_test CUDA_ARCH=sm_120
./ds4_test --server

echo "===== ALL QWEN REGRESSIONS PASSED ====="

# The CPU target above compiles the shared objects without GPU support and make
# does not track flags per object: leave the tree in a CUDA state, with every
# binary rebuilt, because a bare 'make ds4' would relink CUDA objects that are
# not there any more.
make clean >/dev/null 2>&1
make ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4_test CUDA_ARCH=sm_120 >/dev/null 2>&1
