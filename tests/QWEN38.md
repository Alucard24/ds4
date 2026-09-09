# Qwen3.8-27B native text and vision checks (Linux)

This documents the CPU text reference and native CUDA text plus Qwen3-VL
image path. Competitive CUDA performance, video, full chat/tool template
parity, Metal execution and recurrent disk-cache serialization are not complete.
Unsupported cache operations fail explicitly; they must not save a
DeepSeek-shaped payload.

## No-model kernel tests

```sh
make test-qwen38-cpu
SANITIZE=1 tests/run_qwen38_cpu.sh
```

These test the actual private functions in `ds4.c`: text IMRoPE split-half
pairing/unchanged tail, eight steps of raw convolution history, causal gated
GQA head assignment, and four GDN steps against an independent double-precision
recurrence. The Linux target is also a prerequisite of `make test`.

## Optional ggml dequant comparison

No ggml library is linked into ds4. The following **test-only** comparison uses
an already-built local llama.cpp shared library:

```sh
L=/home/diegom/AI-Projects/llama.cpp
M="$L/build/bin/models/Qwen3.8-27B/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf"
tests/run_qwen38_cpu.sh "$L/build/bin/libggml-cpu.so" "$M"
```

For each supported IQ type plus Q2_K/Q4_K: 64 synthetic four-block rows with
finite scales; first and last rows of every matching real tensor; float-input
dots checked against ggml-dequantized weights. This is not a test of ggml's
activation-quantized matmul kernels. Current target covers 402 real tensors.

The external-library sanitizer run has reported 104 bytes leaked during
library/loader initialization after `dlclose`; standalone kernel and engine
session sanitizer runs without that library are clean. Do not report the
external-library run as sanitizer-clean.

## CUDA mixed-IQ MMVQ

The explicit CUDA target checks all nine packed formats used by the language
GGUF at K=5120. Constant activations give a near-exact format/layout oracle;
varied activations include expected Q8_1 activation-quantization error. IQ2_S
also checks the exact get-row path needed by the token embedding. A 16-row MMQ
case covers the eight formats supported by tiled prefill; IQ1_M is deliberately
row-sliced through its validated MMVQ path.

```sh
make clean
make tests/test_qwen38_cuda CUDA_ARCH=sm_120
make test-qwen38-cuda CUDA_ARCH=sm_120 \
  DS4_TEST_GGML_CPU="$L/build/bin/libggml-cpu.so"
```

The external ggml CPU library supplies only independent host dequantization.
Weights and activations are copied to the RTX GPU; ds4's vendored MIT MMVQ runs
the matvec.

## CUDA text graph

The CUDA correctness path runs the complete 64-layer model: native mixed-IQ
MMVQ, depthwise convolution and GDN recurrence, gated GQA, SwiGLU FFN and
output logits. GDN state remains FP32. The 16 global-attention layers use FP16
K/V (64 KiB per context token), matching the practical memory class of normal
llama.cpp CUDA inference. At context 32768 the measured process peak was
14100 MiB on the RTX 5070 Ti, including the approximately 10.95 GiB model.

```sh
make clean
make cuda CUDA_ARCH=sm_120
make test-qwen38-cuda-session CUDA_ARCH=sm_120 \
  DS4_TEST_QWEN38_MODEL="$Q/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf"
```

The session gate checks token-at-a-time eval, no-op sync, changed-prefix
rebuild, shortening plus extension, finite logits, and explicit rejection of
an unsupported recurrent payload. No-op sync remains bit-exact; rebuilds
across MMVQ/MMQ batch shapes must preserve argmax within the numerical gate
below. On the canonical six-token sentence its CUDA mean NLL was
2.43385090 versus 2.43708414 for the FP32 CPU-state oracle, and both generated
` Paris` greedily.

CUDA prefill is layer-major in bounded chunks of at most 128 tokens. Batches up
to eight use MMVQ; larger batches use native packed-weight MMQ, with the lone
IQ1_M matrix row-sliced through MMVQ. `DS4_QWEN38_PREFILL_CHUNK=1..128` and
`DS4_QWEN38_PREFILL_SEQUENTIAL=1` are diagnostic comparison controls, not
alternate release modes. MMQ and MMVQ change floating-point reduction order;
the rebuild test therefore requires stable argmax and a bounded 0.5 maximum
logit delta rather than claiming bit identity across batch shapes.

## CUDA Qwen3-VL vision

The Qwen3-VL path loads the llama.cpp-compatible
`mmproj-Qwen3.8-27B-BF16.gguf` sidecar without linking llama.cpp. It validates
the fixed `qwen3vl_merger` shape (27 layers, width 1152, 16 heads, FFN 4304,
2x2 spatial merge, no deepstack), performs bicubic dynamic-resolution
preprocessing, and executes patch projection, resized learned position
embeddings, bidirectional MRoPE attention, GELU FFNs, post norm, and the
5120-wide merger on CUDA. The BF16 matrices remain packed in the auxiliary
model mapping; FP32 biases, norms and activations preserve the GGUF graph.
Language-model prefill replaces `<|image_pad|>` rows with the projected vectors
and uses Qwen's compressed 2D MRoPE positions while retaining raw token indices
for the causal KV cache.

```sh
make tests/test_qwen3vl_image
./tests/test_qwen3vl_image

make test-qwen3vl-vision CUDA_ARCH=sm_120 \
  DS4_TEST_QWEN38_MODEL="$Q/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf" \
  DS4_TEST_QWEN38_MMPROJ="$Q/mmproj-Qwen3.8-27B-BF16.gguf" \
  DS4_TEST_QWEN38_IMAGE=/path/to/image.png

make test-qwen3vl-session CUDA_ARCH=sm_120 \
  DS4_TEST_QWEN38_MODEL="$Q/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf" \
  DS4_TEST_QWEN38_MMPROJ="$Q/mmproj-Qwen3.8-27B-BF16.gguf" \
  DS4_TEST_QWEN38_IMAGE=/path/to/white224.png
```

The public-session test transfers image ownership into the multimodal prompt,
checks exact no-op sync reuse, continues autoregressive decoding through the
same recurrent/GA state, requires the greedy result to identify the white
fixture, and exercises a two-image prompt with separate MRoPE spans.

`ds4-server` selects Qwen's `<|im_start|>` chat rendering and native
`<tool_call><function=...>` syntax for model id 4. Its model aliases are
`qwen3.8-27b`, `qwen3.8-27b-chat`, and `qwen3.8-27b-reasoner`. CUDA image smoke
tests pass through `/v1/chat/completions`, `/v1/responses`, and Anthropic
`/v1/messages`; a second image-bearing turn reuses the live multimodal prefix
(75 cached tokens in the checked 224x224 case). Server parser/rendering unit
tests are included in `./ds4_test --server`.

Against llama.cpp's `llama-mtmd-debug`, a synthetic 224x224 white image yields
49 vectors. llama.cpp reports sum `110.311356` and first values
`0.5204, -0.1312, -0.6193`; ds4 reports `106.937073` and
`0.517695, -0.130709, -0.619747`. At 256x256, llama.cpp reports sum
`78.989944` and ds4 `71.992472`; first-row values remain within about 0.008.
The remaining aggregate difference has not been isolated; full embedding parity
is therefore not claimed and the sum is not used as an exact-bit gate. End-to-end greedy smoke tests identify a blank white image
and describe the 512x507 Earth fixture as Earth with Africa and Madagascar.

Dynamic preprocessing uses llama.cpp's normal 8..4096 merged-token limits.
The current vision attention kernel is bounded in memory but intentionally
simple and scales quadratically; large high-resolution images still need a
tiled/flash-attention optimization. One-process timings on the RTX 5070 Ti were
0.716 s for the complete 224x224 encoder test (49 output tokens), 1.315 s for
the 512x507 Earth fixture resized to 512x512 (256 tokens), and 3.565 s for a
768x768 image (576 tokens), including process startup and sidecar mapping. The
public 192-context session test peaked at 13026 MiB. On a 16GB GPU, reduce
language context when loading the additional approximately 0.87 GiB sidecar if
allocation pressure is high.

A warm-process text benchmark can be built and run with:

```sh
make tests/test_qwen38_cuda_perf CUDA_ARCH=sm_120
./tests/test_qwen38_cuda_perf "$Q/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf" \
  /tmp/prompt.txt 128
```

On the RTX 5070 Ti, a 494-token local prompt measured 203.926 prefill tok/s and
33.359 decode tok/s at the resulting context. This is a real improvement over
token-replay prefill but remains well below the recorded llama.cpp baselines
(pp512 1628.69 tok/s, tg128 53.24 tok/s), so performance work is not complete.

## Real-model session and continuation checks

Build CPU only (run `make clean` first if switching backends):

```sh
make cpu
cc -O2 -std=c99 -I. tests/test_qwen38_session.c \
  ds4_cpu.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o \
  -lm -pthread -o /tmp/qwen-session
/tmp/qwen-session "$M" 'The capital of France is Paris.'
```

This checks every final logit after token-by-token eval, no-op sync, changed
prefix/rebuild, and shortening followed by extension. Expected maximum error:
**zero** for the same CPU build. Also tests safe rejection of unsupported cache
save/load. It uses one loaded model and one live session.

Independent llama.cpp C-API probe (same raw text, no BOS/template, sequential
CPU decode, CPU weights, KQV and operation offload disabled):

```sh
cc -O2 -I"$L/include" -I"$L/ggml/include" tests/qwen38_llama_reference.c \
  -L"$L/build/bin" -Wl,-rpath,"$L/build/bin" -lllama -lm -o /tmp/qwen-ref
/tmp/qwen-ref "$M" 'The capital of France is Paris.'
```

Compare `LP position token_id logprob` lines, requiring identical IDs first.
On the checked build/model: mean NLL over 6 tokens is 2.43708414 (ds4) versus
2.42675939 (llama). This is a smoke check, not a statistically robust quality
gate or proof of universal logit parity.

For a longer NLL-only check without repeated session rebuilds:

```sh
T='The quick brown fox jumps over the lazy dog. In a small village, a baker makes fresh bread every morning.'
DS4_NLL_DUMP=1 ./ds4 --cpu --raw -m "$M" -p "$T" --temp 0 -n 1
/tmp/qwen-ref "$M" "$T"
```

Observed 22 scored tokens: NLL 1.73504823 vs 1.72583734, mean absolute logprob
difference 0.02385070, maximum 0.09440743. CPU prefill currently replays tokens
one at a time and is slow. Never run two large model processes concurrently.

Full-session sanitizer build without replacing normal engine objects:

```sh
cc -O1 -g -march=native -std=c99 -D_GNU_SOURCE -DDS4_NO_GPU \
  -fsanitize=address,undefined -fno-omit-frame-pointer -c ds4.c -o /tmp/qwen-asan.o
cc -O1 -g -std=c99 -I. -fsanitize=address,undefined -fno-omit-frame-pointer \
  tests/test_qwen38_session.c /tmp/qwen-asan.o ds4_image.o ds4_distributed.o \
  ds4_tp.o ds4_ssd.o ds4_layer_pack.o -lm -pthread -o /tmp/qwen-session-asan
/tmp/qwen-session-asan "$M" 'The quick'
```
