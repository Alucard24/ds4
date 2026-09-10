# Qwen3.8-27B native text and vision checks (Linux)

This documents the CPU text reference and native CUDA text plus Qwen3-VL
image path. Competitive CUDA performance, video, and full chat/tool template
parity are not complete. Metal and ROCm execution are outside the current
hardware-validated scope; this work targets Qwen on CUDA.

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
14216 MiB on the RTX 5070 Ti with the 256-token prefill workspace, including
the approximately 10.95 GiB model.

```sh
make clean
make cuda CUDA_ARCH=sm_120
make test-qwen38-cuda-session CUDA_ARCH=sm_120 \
  DS4_TEST_QWEN38_MODEL="$Q/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf"
```

The session gate checks token-at-a-time eval, no-op sync, changed-prefix
rebuild, shortening plus extension, finite logits, and a complete DSV4
save/load round trip. It verifies both the restored logits and an exactly
reproduced continuation, proving that GDN recurrence, convolution history and
GA KV were restored rather than only the visible token history. Truncated
payload sizes are rejected before mutating the live session. No-op sync remains
bit-exact; rebuilds across MMVQ/MMQ batch shapes must preserve argmax within the
numerical gate below. On the canonical six-token sentence its CUDA mean NLL was
2.43385090 versus 2.43708414 for the FP32 CPU-state oracle, and both generated
` Paris` greedily.

CUDA prefill is layer-major in bounded chunks of at most 256 tokens. Batches up
to eight use MMVQ; larger batches use native packed-weight MMQ, with the lone
IQ1_M matrix row-sliced through MMVQ. `DS4_QWEN38_PREFILL_CHUNK=1..256` and
`DS4_QWEN38_PREFILL_SEQUENTIAL=1` are diagnostic comparison controls, not
alternate release modes. MMQ and MMVQ change floating-point reduction order;
the rebuild test therefore requires stable argmax and a bounded 0.5 maximum
logit delta rather than claiming bit identity across batch shapes.

## CUDA Qwen3-VL vision

The Qwen3-VL path loads the llama.cpp-compatible
`mmproj-Qwen3.8-27B-BF16.gguf` sidecar without linking llama.cpp. It validates
the fixed `qwen3vl_merger` shape (27 layers, width 1152, 16 heads, FFN 4304,
2x2 spatial merge, no deepstack), performs aspect-preserving dynamic-resolution
preprocessing with Pillow-compatible bicubic filtering and centered black
padding, and executes patch projection, resized learned position embeddings,
bidirectional MRoPE attention, GELU FFNs, post norm, and the 5120-wide merger
on CUDA. The F32 patch matrices use the graph's FP16 im2col/GEMM boundary;
BF16 transformer matrices remain packed in the auxiliary model mapping, while
FP32 biases, norms and activations preserve the GGUF graph.
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

The preprocessing test covers square, padded non-square, minimum-size and a
nonuniform 300x200 Pillow-resize checksum in merger patch order.

The public-session test transfers image ownership into the multimodal prompt,
checks exact no-op sync reuse, continues autoregressive decoding through the
same recurrent/GA state, requires the greedy result to identify the white
fixture, and exercises a two-image prompt with separate MRoPE spans. It also
round-trips a multimodal disk payload and verifies exact continuation plus
restoration of the compressed MRoPE frontier and image fingerprint identity.

`ds4-server` selects Qwen's `<|im_start|>` chat rendering and native
`<tool_call><function=...>` syntax for model id 4. Its model aliases are
`qwen3.8-27b`, `qwen3.8-27b-chat`, and `qwen3.8-27b-reasoner`. CUDA image smoke
tests pass through `/v1/chat/completions`, `/v1/responses`, and Anthropic
`/v1/messages`; a second image-bearing turn reuses the live multimodal prefix
(75 cached tokens in the checked 224x224 case). Server parser/rendering unit
tests are included in `./ds4_test --server`.

The parity investigation found two concrete front-end differences. llama.cpp's
patch convolution emits FP16 im2col/GEMM results even though the two sidecar
weights are stored as F32, and its Qwen preprocessor preserves aspect ratio with
`PAD_CEIL` black bars instead of stretching to the aligned canvas. ds4 now
matches those rules, including Pillow's separable 22-bit fixed-point bicubic
resize, learned-position interpolation operation order, vision RoPE frequency
construction, and CUDA layer-norm reduction. For both a 224x224 white fixture
and a 512x507 Earth PNG, the complete patch-plus-position tensor is bit-exact
against llama.cpp; the white fixture also remains bit-exact through the first
layer's QKV projection.

The remaining difference starts in attention. ds4 uses bounded FP32 online
softmax, while normal llama.cpp MTMD enables lower-precision FlashAttention and
its non-Flash CUDA path normally permits TF32 GEMMs. Against an independent
FP64 calculation of the first white-image attention layer, ds4 has mean absolute
error `2.00e-8` and relative L2 error `3.27e-7`; ordinary non-Flash llama.cpp
has `6.07e-6` and `9.98e-5`. Running llama.cpp with
`NVIDIA_TF32_OVERRIDE=0` brings its first-layer result within mean absolute
error `2.58e-8` of ds4, confirming that this is a precision-policy difference,
not an attention-layout bug.

Over the complete 224x224 white embedding, ds4 versus non-Flash/no-TF32
llama.cpp has mean absolute error `0.001350` and relative L2 error `0.00516`
(ds4 sum `108.206112`, llama.cpp sum `107.587742`). Normal FlashAttention
llama.cpp reports sum `110.311216` and differs from its own no-TF32 result by
mean absolute error `0.003280` and relative L2 error `0.01320`. On the padded
512x507 Earth PNG, ds4 versus non-Flash/no-TF32 llama.cpp has mean absolute
error `0.001581` and relative L2 error `0.00401`. Nonlinear BF16 boundaries
amplify tiny attention differences through later layers, so full embedding
parity is still not claimed and raw sums are not exact-bit gates. Use lossless
PNG fixtures for numerical checks: ds4's Iris JPEG decoder and llama.cpp's
image decoder need not produce identical source pixels. End-to-end greedy smoke
tests identify a blank white image and describe the Earth fixture as Earth with
Africa and Madagascar.

Dynamic preprocessing uses llama.cpp's normal 8..4096 merged-token limits.
The vision attention kernel remains bounded in memory and scales quadratically;
large high-resolution images will eventually benefit from tiled or flash
attention. A one-warp-per-query rewrite removed the old per-key block barriers
without changing any output bits. Within one loaded engine on the RTX 5070 Ti,
median warm encode time fell from 27.2 to 9.9 ms for 224x224 (49 output tokens),
from 613 to 124 ms for the padded 512x507 Earth fixture (256 tokens), and from
2.89 s to 0.480 s for 768x768 (576 tokens). These calls include image decode,
preprocessing, temporary allocation and the complete encoder, but exclude model
open; first-encode times were 0.456, 0.504 and 0.866 s respectively. The public
192-context session test peaked at 13026 MiB. On a 16GB GPU, reduce language
context when loading the additional approximately 0.87 GiB sidecar if
allocation pressure is high.

A warm-process text benchmark can be built and run with:

```sh
make tests/test_qwen38_cuda_perf CUDA_ARCH=sm_120
./tests/test_qwen38_cuda_perf "$Q/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf" \
  /tmp/prompt.txt 128
```

The benchmark reports both the first (`COLD_PREFILL`) sync and a same-session
rebuild (`PREFILL`), because the first sync also populates ds4's lazy CUDA model
cache. On the RTX 5070 Ti, three runs of a 538-token local prompt had median
load 0.214 s, cold prefill 248.3 tok/s, steady-state prefill 799.2 tok/s, and
128-token decode 39.95 tok/s. Before this pass, the same prompt measured cold
prefill 220.9 tok/s, steady-state prefill 689.8 tok/s, and decode 32.57 tok/s.
At 2012 prompt tokens, cold/steady prefill measured 417.1/623.9 tok/s and decode
at the resulting context measured 27.75 tok/s, versus 330.3/460.4/18.73 before.
The gain comes from a barrier-free bounded GA attention schedule, a 256-token
prefill chunk, removal of redundant dense-MMVQ output cleanup, and branchless
Q4_K scale unpacking. The recorded normal llama.cpp baselines remain faster
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
prefix/rebuild, shortening followed by extension, and disk-payload restore.
Expected maximum error is **zero** for the same CPU build, including the first
continuation after restore. It uses one loaded model and one live session.

Qwen DSV4 payloads persist checkpoint tokens, host logits, FP32 GDN recurrent
and convolution state, all live GA K/V rows, the multimodal logical RoPE
frontier, and bounded image identities. CUDA writes its already-quantized F16
GA cache directly; the CPU oracle writes F32. The element-size field allows the
loader to convert either representation when changing backend. Payload size is
roughly 149 MiB of fixed recurrent/logit state plus 64 KiB per token on CUDA
(128 KiB per token for CPU checkpoints), so normal KV-store limits still apply.
The outer KVC header accepts quant tag zero only for dense Qwen model id 4;
DeepSeek/GLM entries still require their routed-expert 2/4-bit tag. A checked
server restart stored a 75-token Qwen checkpoint as a 155.26 MiB file, restored
it in 35.8 ms, and reported all 75 prompt tokens from `disk-text`.

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
