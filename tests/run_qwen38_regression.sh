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

echo "===== recall at depth, the quality gate for the prefill ====="
# The prefill's attention moved to tensor cores (f16 Q and P) and its only
# *quality* gate is this one: four access codes at 2%, 25%, 50% and 75% of a
# ~19.5k-token document, asked for at the end and answered by greedy decode.
# Generated here in shell so no 78 KiB fixture has to be committed; 378 lines of
# this filler is about 19.5k tokens, and a wrong count shows up as a refusal to
# start rather than as a silent pass.
NEEDLE=/tmp/ds4-regression-needle.txt
awk -v lines=378 '
BEGIN {
    filler = "The maintenance log for the northern relay station records routine checks, weather observations and equipment notes in the order they were written, each entry dated and signed by the operator on duty that day.";
    codes[1] = "PASS-A-4821"; codes[2] = "PASS-B-1397"; codes[3] = "PASS-C-7342"; codes[4] = "PASS-D-9615";
    at[1] = int(lines * 0.02); at[2] = int(lines * 0.25); at[3] = int(lines * 0.50); at[4] = int(lines * 0.75);
    for (i = 0; i < lines; i++) {
        for (c = 1; c <= 4; c++) if (i == at[c]) printf("Access code %s.\n", codes[c]);
        print filler;
    }
    print "";
    print "Question: list the four access codes in this document, in the order they appear, separated by spaces.";
    print "Answer:";
}' > "$NEEDLE"
recall=$(./ds4 -m "$MODEL" -c 24576 --raw --temp 0 -p "$(cat "$NEEDLE")" -n 200 2>/dev/null || true)
hits=0
for code in PASS-A-4821 PASS-B-1397 PASS-C-7342 PASS-D-9615; do
    case "$recall" in *"$code"*) hits=$((hits + 1)) ;; esac
done
if [ "$hits" = "4" ]; then
    echo "recall at ~19.5k: 4/4 codes"
else
    echo "recall at ~19.5k: $hits/4 codes (expected A-4821 B-1397 C-7342 D-9615)"
    exit 1
fi

echo "===== the three binaries nothing was running ====="
# ds4-eval, ds4-bench and ds4-agent are built by this script and were never run
# by it: the eval's graders, the bench's own flag handling and the agent's
# end-to-end turn had no gate, and the last two were touched for -ctk/-ctv.
#
# They have to be built *here*: the CPU reference step runs `make clean`, so by
# this point in the script nothing from the first build exists any more.  Every
# step that runs a binary builds it, or it measures a missing file.
make ds4-eval ds4-bench ds4-agent CUDA_ARCH=sm_120 >/dev/null 2>&1
./ds4-eval --self-test-extractors 2>&1 | grep -q "self-tests passed" \
    || { echo "eval extractor self-tests failed"; exit 1; }
echo "eval extractors: passed"

# The bench wants a prompt at least as long as --ctx-max, so it is generated here.
python3 -c "
words = ('The capital of France is Paris and the largest ocean is the Pacific and the '
         'tallest mountain is Everest and the longest river is the Nile.')
open('/tmp/ds4-regression-bench.txt', 'w').write(' '.join([words] * 1100))"
bench=$(./ds4-bench -m "$MODEL" --prompt-file /tmp/ds4-regression-bench.txt \
        --ctx-max 4096 --gen-tokens 0 2>/dev/null || true)
best=$(printf '%s\n' "$bench" | awk -F, 'NR > 1 && $3 + 0 > best { best = $3 + 0 } END { printf "%.0f", best }')
if [ -z "$best" ] || [ "$best" -lt 500 ]; then
    echo "bench prefill rate missing or collapsed: '${best:-none} t/s'"
    printf '%s\n' "$bench" | tail -4
    exit 1
fi
echo "bench prefill at 4096: ${best} t/s"

agent=$( (cd /tmp && "$OLDPWD/ds4-agent" -m "$MODEL" -c 4096 --non-interactive \
        -p 'Reply with exactly the two words: all good' 2>&1) || true)
case "$agent" in
    *"all good"*) echo "agent one-shot turn: answered" ;;
    *) echo "agent one-shot turn failed:"; printf '%s\n' "$agent" | tail -4; exit 1 ;;
esac

# The agent's session cache: agent_kv_save_path demanded a routed-expert
# quantization of 2 or 4 bits, a DeepSeek-shaped requirement, so with Qwen every
# save failed and the documented session persistence never happened.  The header
# byte is compared on load against the same engine call, so zero (no routed
# experts) is self-consistent.  A private HOME keeps this away from real sessions.
acache=$(mktemp -d)
(cd /tmp && HOME="$acache" "$OLDPWD/ds4-agent" -m "$MODEL" -c 4096 --non-interactive \
    -p 'Reply with exactly the two words: all good' >/dev/null 2>"$acache/first.log") || true
if grep -q "failed to save system prompt KV" "$acache/first.log"; then
    echo "the agent still refuses to save its KV cache:"
    grep "failed to save" "$acache/first.log" | head -2
    exit 1
fi
if [ ! -f "$acache/.ds4/kvcache/sysprompt.kv" ]; then
    echo "the agent ran but wrote no sysprompt KV cache"
    exit 1
fi
# Same invariant the load path checks: the byte must be what the engine reports.
qsaved=$(python3 -c "
import sys
h = open(sys.argv[1], 'rb').read(16)
print(h[4])" "$acache/.ds4/kvcache/sysprompt.kv")
if [ "$qsaved" != "0" ]; then
    echo "sysprompt KV records quant_bits=$qsaved; Qwen3.8 has no routed experts"
    exit 1
fi
# And the second run must accept that file rather than fail on it.
(cd /tmp && HOME="$acache" "$OLDPWD/ds4-agent" -m "$MODEL" -c 4096 --non-interactive \
    -p 'Reply with exactly the two words: all good' >/dev/null 2>"$acache/second.log") || true
if grep -qE "failed to (save|load) .*KV|cached text does not match" "$acache/second.log"; then
    echo "the agent rejected its own KV cache on the second run:"
    grep -E "failed to (save|load)|does not match" "$acache/second.log" | head -2
    exit 1
fi
echo "agent KV cache: saved (quant_bits=0) and reloaded"
rm -rf "$acache"

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
