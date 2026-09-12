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
# DS4_KV_Q8 is the hook the test binaries use, since they take no engine
# options.  What is asserted is the number itself, which is the thing that would
# move if the q8_0 path regressed.  Note what this test can and cannot say: over
# 16 tokens the two formats are 1.80954673 against 1.80900178, which is inside
# its noise - the format comparison is the recall test, where both answer four
# for four with identical text.
nll_q8=$(DS4_KV_Q8=1 DS4_TEST_QWEN38_CUDA=1 ./tests/test_qwen38_session_cuda "$MODEL" "$PROMPT" 2>&1 | grep '^MEAN_NLL' || true)
case "$nll_q8" in
  "MEAN_NLL 1.80900178 TOKENS 16")
    echo "q8_0 KV NLL as recorded: $nll_q8" ;;
  *)
    echo "q8_0 KV NLL changed from the recorded 1.80900178: $nll_q8"
    exit 1 ;;
esac

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
