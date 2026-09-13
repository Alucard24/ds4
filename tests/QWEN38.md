# Qwen3.8-27B native text and vision checks (Linux)

This documents the completed CPU text reference and native CUDA text plus
Qwen3-VL image path. The supported hardware scope is Qwen on CUDA; Metal,
ROCm and video remain intentionally outside the hardware-validated MVP rather
than being partially implemented release paths. Performance is measured against
normal llama.cpp and numerical parity is claimed only by the gates below.

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
llama.cpp CUDA inference. At context 32768 the final measured process peak was
14214 MiB on the RTX 5070 Ti with the prefill workspace, including the
approximately 10.95 GiB model. The workspace has since grown from 256 to 512
tokens per chunk, which adds about 10 MiB of scratch.

```sh
make clean
make cuda CUDA_ARCH=sm_120
make test-qwen38-cuda-session CUDA_ARCH=sm_120 \
  DS4_TEST_QWEN38_MODEL="$Q/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf"
```

The session gate checks token-at-a-time eval, no-op sync, changed-prefix
rebuild, shortening plus extension, finite logits, the fixed CUDA reference
mean NLL `1.81334038` over its 16-token regression sentence, and a complete DSV4
save/load round trip. It verifies both the restored logits and an exactly
reproduced continuation, proving that GDN recurrence, convolution history and
GA KV were restored rather than only the visible token history. Truncated
payload sizes are rejected before mutating the live session. No-op sync remains
bit-exact; rebuilds across MMVQ/MMQ batch shapes must preserve argmax within the
numerical gate below. On the canonical six-token sentence its CUDA mean NLL was
2.43385090 versus 2.43708414 for the FP32 CPU-state oracle, and both generated
` Paris` greedily.

CUDA prefill is layer-major in bounded chunks of at most 512 tokens. Batches up
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
`<tool_call><function=...>` syntax for model id 4. Its system tool instructions,
normalized OpenAI function-schema objects, whitespace trimming, historical
thinking wrappers, grouped `<tool_response>` turns, and assistant generation
prefix match the GGUF `tokenizer.chat_template`. OpenAI and Anthropic SSE project Qwen
tool calls incrementally without exposing the sampled XML; parameter values are
buffered until `</parameter>` because Qwen does not declare whether a value is a
JSON literal or a string in its opening tag. Multiple calls and tool tags split
across stream updates are covered by `./ds4_test --server`.

The model aliases are `qwen3.8-27b`, `qwen3.8-27b-chat`, and
`qwen3.8-27b-reasoner`. CUDA image smoke tests pass through
`/v1/chat/completions`, `/v1/responses`, and Anthropic `/v1/messages`; a second
image-bearing turn reuses the live multimodal prefix (75 cached tokens in the
checked 224x224 case). The CUDA session gate compares five llama.cpp-derived tokenizer vectors:
multilingual UTF-8, decomposed combining characters, emoji ZWJ/modifier
sequences, CJK/Indic/Thai/Korean/Cyrillic text, code punctuation and Qwen chat,
thinking, tool and vision special tokens. Rendered special-token strings use
`ds4_tokenize_rendered_chat()`; ordinary user text intentionally remains on
`ds4_tokenize_text()`. The server test also hashes a complete historical
thinking/tool-response prompt and requires exact equality with llama.cpp
`/apply-template`, in addition to the incremental stream boundary tests.

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
Vision attention remains bounded in memory and quadratic in work. Its kernel now
groups sixteen query warps per block and stages 32-key K/V tiles in shared
memory. Every query retains the original key order, FP32 online softmax and
warp-reduction order; no N² score buffer, TF32, FP16 Q/K/V, or FlashAttention
path was introduced.

The first one-warp-per-query rewrite removed the old per-key block barriers and
reduced median warm full-encoder time from 27.2 to 9.9 ms for 224x224, 613 to
124 ms for the padded 512x507 Earth fixture, and 2.89 s to 0.480 s for 768x768.
In a fresh paired run for the tiled follow-up, the corresponding one-warp versus
tiled medians were 9.482 vs 9.323 ms, 121.270 vs 104.165 ms, and 575.284 vs
409.293 ms. GPU clock state makes absolute runs noisy, so these paired medians
and exact matching sums (`108.206111801`, `-12235.925095321`, and
`-2166.399592827`) are the retained gate. Calls include image decode,
preprocessing, temporary allocation and the complete encoder, but exclude model
open. The public 192-context session test peaked at 13026 MiB. On a 16GB GPU,
reduce language context when loading the additional approximately 0.87 GiB
sidecar if allocation pressure is high. A 2048x2048 RGB fixture exercised the
maximum 4096 merged image tokens without an allocation failure and completed in
17.75 s, confirming bounded memory behavior; the upper limit remains
computationally expensive despite tiling.

A warm-process text benchmark can be built and run with:

```sh
make tests/test_qwen38_cuda_perf CUDA_ARCH=sm_120
./tests/test_qwen38_cuda_perf "$Q/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf" \
  /tmp/prompt.txt 128
```

The benchmark reports both the first (`COLD_PREFILL`) sync and a same-session
rebuild (`PREFILL`).

That lazy population was measured later and turned out to be the whole prefill
gap: the Qwen branch returned before the generic eager tensor-span preparation
used by the other families, so a first prompt faulted every weight range in at
about 23 GB/s inside its own prefill windows. The branch now runs that preload,
which is why the earlier "cold prefill" numbers below are superseded: with the
weights resident there is no cold/steady split, and a 350-token first prompt
measures 817-969 tok/s (noisy, that figure is dominated by the per-session
setup described above) with a 2000-token prompt at 1179-1181 tok/s on the RTX
5070 Ti (the block shape settled at 512 threads).

The largest single win since the first stable point is in the GA attention
prefill, which was bound by the number of requests it sent to the cache: every
warp pulled its own copy of the key vector and its own slice of the value vector
straight from the cache, once per key, so the cost grew as queries x keys
without bound (69.6/208.0/344.9/470.1 ms for the four 512-token chunks of a
2000-token prompt). Staging a 32-key tile of keys and values in shared memory,
loaded once per block, brought that to 16.8/50.9/86.1/108.1 ms - a factor of
four to four and a half - and took the 2000-token prefill from 789 to 1180
tok/s. Decode is untouched: at one query there is nothing to share a tile with,
so that path keeps the old kernel shape and the launcher picks by token count.

What is left outside the instrumented windows in a chunk - about 113 ms of a
   380 ms chunk - looked like host cost and turned out not to be. Two hypotheses
   were tested and refuted. Batching the token embedding (a first draft of the
   chunk made one host call per token, 512 of them) into a single call changed
   the outside-window time from 786 to 778 ms and was reverted. And halving the
   chunk size, which doubles the number of chunks and therefore doubles the
   number of host calls per prompt without changing any per-token work, costs
   nothing measurable:

     chunk 512 -> 1139 tok/s      128 -> 1053 tok/s
     chunk 256 -> 1143 tok/s       64 ->  862 tok/s

   Since there are about a thousand host-issued operations per chunk whatever
   the chunk size, that flat 512/256 pair bounds the per-call cost at a few
   microseconds. What the outside-window time actually is, then, is the long
   tail of small GPU operations no window covers - norms, residual adds, rope,
   the gating, the convolution, the cache writes - and that tail is exactly what
   starves when the host is busy: with a qemu VM holding 97% of a core those
   segments stretched from 113 ms to 786 ms while the big GDN and GA kernels
   stayed put.
   The lever for that is fewer host-issued operations per chunk, i.e. CUDA
   graphs over the layer loop, which is a project and not an afternoon - an
   earlier attempt at graphs for the FFN was measured and reverted.

Two experiments toward this measured nothing and are worth recording, because
together they are what identified the real limit. Removing an eight-fold
redundancy in the score computation (all eight warps of a block were computing
the same 256-dim score from the same lanes) changed the time by less than a
percent: those warps were sharing the same L1 lines, so the redundancy cost no
traffic and was really hiding its own latency. Interleaving two keys per
iteration changed nothing either. Only then did the request count turn out to be
the thing.
The kernel itself was never slow - 0.64 ms per 5120x17408 gate matmul, 48
TMAC/s, the rate llama.cpp reaches; the earlier numbers came from runs that paid
the upload inside the measurement. See `tests/QWEN38_PREFILL.md`.

Recorded measurements before that change, kept for the trail: load 0.207 s, cold
prefill 252.0 tok/s, steady-state prefill 802.7 tok/s, 128-token decode
40.17 tok/s on a 538-token prompt; at 2012 tokens, cold/steady 422.8/624.5 tok/s
with decode 27.43 tok/s. The decode figures still hold; the prefill ones were
measuring the upload.
The gain comes from a barrier-free bounded GA attention schedule, the prefill
chunk, removal of redundant dense-MMVQ output cleanup, and branchless Q4_K scale
unpacking. Two later changes moved the prefill again, both numerics-neutral and
both verified against the gate: the Qwen weight preload at engine open, and
keeping the GDN recurrent state in a 64 KiB shared tile for the whole chunk
instead of reading and writing it in global memory once per token (103 GiB of
traffic per chunk). The recurrence fell from 204.8 to 141.9 ms per 350-token
chunk and the chunk total from 334.8 to 271.5 ms, then to 97.4 and 227.5 ms by
processing two state rows per iteration in that kernel (each row needs two
warp-wide shuffle reductions, and rows are independent), and to 62.7 and
192.5 ms by running 256 threads instead of 128, so eight warps walk sixteen
state rows each instead of four warps walking thirty-two, and to 52.6 and
182.5 ms at 512 threads (sixteen warps, eight rows each).  A 1024-thread version
(thirty-two warps, four rows each) measured 179.7-180.8 ms against 182.4-184.5
in one pass, but an interleaved A/B of the two binaries under sustained load
spread over 179.6-254.5 ms on both, so the difference was inside the noise and
the simpler block shape was kept.  Only the first 128
threads own a q/k/o element there; threads 128..255 feed zeros into those three
reductions, and adding 0.0f is exact, so every reduction keeps its order and its
value.  A four-row variant was not repeatable (212 ms on one run, 290 on the
next) and was dropped.  Trunk NLL unchanged at 1.81334038 through all of it, and
decode picked up the shorter chain as well (43.1 tok/s greedy, 55.6 with MTP). The recorded normal llama.cpp baselines remain faster
(pp512 1628.69 tok/s, tg128 53.24 tok/s); matching llama.cpp is not claimed.
The MTP speculative round now reaches 1.30x on this prompt (41.4 -> 53.7 tok/s)
with the stream identical to one-token greedy.

The deep-context prefill attention was the next item.  Two wrong models came
first and are worth recording: splitting each row's key range in the style of the
decode split, which measured nothing (the prefill already has 464 rows of
parallelism, so cutting the keys duplicates partial softmax states instead of
shortening a chain), and an occupancy theory, killed by rebuilding at tile sizes
8, 16 and 32, which all gave the same 86 ms.  What settled it was an ablation
mask compiled into the kernel - staging only, then the K dot, the warp
reduction, the softmax and the V accumulate subtracted one at a time - together
with a standalone reproduction of the same loop body: the body runs at
0.246 ns per (row, head, key) and the kernel at 0.275, so the kernel was already
at the hardware's issue limit for its own instruction stream, and that stream was
112 instructions per key for 34 of arithmetic.  (`redux.sync.add.f32`, the
hardware float reduction that would have collapsed the five-shuffle tree, exists
only on the datacenter Blackwell parts, not on sm_120.)  Three changes followed,
each verified bit-identical against the recorded logprobs of a 2000-token prompt
(max delta 0.000000) and against the gate:

  - the tile copy became 16 bytes per thread per pass (`uint4`) instead of 64
two-byte loads with dependent two-byte shared stores, which the disassembly
showed were neither vectorised nor overlapped: the copy alone was 44 ms of the
deepest chunk's 121;
  - the online softmax moved off lane 0 onto every lane, and the warp reduction
became a butterfly so that no broadcast is needed: same tree, same order, same
bits, but no divergence handling (BSSY/BSYNC, WARPSYNC) and no shuffles;
  - two adjacent query rows now share one warp and ride the same key and value
registers, so the eight shared loads, eight half-to-float conversions and the
address arithmetic per tensor are paid once for both rows.  Adjacent rows differ
by one position, so the loop walks the union of their ranges and masks the extra
key with -infinity, and the running state starts from a finite floor so that
expf(-inf - -inf) cannot be reached before a row has started.

GA attention core per prefill chunk (512/512/512/464 rows, phase events, ms):

    69.6 / 208.0 / 344.9 / 470.1   before the shared tile
    18.7 /  53.4 /  87.7 / 109.2   after the shared tile
     8.9 /  23.8 /  38.3 /  47.7   after the three changes above

The deepest chunk went from 313.2 to 251.0 ms and the 2000-token prompt from 1123
to 1222 tok/s (llama.cpp: 1353).  What is left in a chunk is no longer attention:
the GDN recurrence is 73.4 ms, the output head 58.0, the projections 53.9 and the
GA attention 47.7.

The GDN recurrence was next, and it is a serial scan: one block per head walks
the chunk's tokens in order, because token t+1 needs token t's state.  The same
ablation method answered what the loop's time was made of: of 73.7 ms on the
deepest chunk, the state row loop was 37.2, the state-independent scalars 12.8,
the output normalization 7.8, and the remaining 17.9 was the barrier skeleton --
seven `__syncthreads` per token, each exposing about 250 cycles, with one block
per SM and nothing else resident to overlap them with.  That is 39 ms of the
73.7 spent in barrier latency alone.  A 32-warp version of the recurrence
testing the opposite hypothesis (that the row loop wanted a shorter chain) is
slower, 80.0/80.8/80.2 against 73.7 ms, and was reverted: the row loop is not a
chain waiting to be shortened, and more warps make each barrier dearer.

So everything in the token loop that does not depend on the state moved out of
it; the scalars are computed by a parallel kernel over the 464x48 (token, head)
items and the output normalization by another, and the recurrence keeps two
barriers per token: one after the normalized q/k are staged, one after the state
rows are updated (every warp reads every row on the next token, so that one is
structural).  Each moved computation keeps its tree -- the 128-wide sums are the
same 32-element warp groups with the same all-zero groups feeding them, and the
per-head inverse norm is the same reduction -- so the results are bit-identical.
The warp sum also became a butterfly here, as in the GA attention: the total
lands on every lane, which removes the broadcast that followed each of the four
reductions in a two-row iteration.

    73.4  ->  53.4  ->  48.8 ms      GDN core, deepest chunk
   313.2  ->  251.0  -> 227.2 ms     chunk total
    1123  ->  1222  ->  1287 tok/s   the 2000-token prompt (llama.cpp: 1353)

All of it is numerics-neutral: the four greedy logits of that prompt stay
bit-identical (max delta 0.000000), the gate reads 1.80954673, the MTP trunk is
still bit-identical over 48 tokens, and the single-token decode phases are
unchanged at 18.0-18.2 ms with 2.4 ms outside the windows.  The largest single
item in a deep chunk is now the output head at 57.7 ms, ahead of the GDN at 48.8,
the projections at 54.2 together and the GA attention at 47.8.

The attention's dot product went to the tensor cores.  The last untouched item of
the prefill, worth 48 of the chunk's 227 ms, is the QK^T: measured in isolation the
body costs 0.246 ns per (key, row) and the scalar dot plus its butterfly is 0.1357
of it, against **0.0191** for the same dot as HMMA m16n8k16 - 7.1x cheaper.

The shape that takes it is flash-attention-2, after a first attempt failed for a
reason worth keeping: keeping the whole kernel and adding a ninth warp for the MMAs
spent the register budget (126 registers, 43 KiB shared, one resident block per SM
instead of three) and measured 20% *slower*.  Register allocation is uniform across
a kernel's warps, so the score warp's half fragment and the staged query cost every
warp.  The working shape is five warps per block - one on QK with the online
softmax, four on the PV for 64 output dimensions each - with the k loop outside the
n loop so only four registers of the Q fragment are live at a time, and 79 registers
against the scalar kernel's 79.

    GA attention core   47.6-48.9 -> 28.1-31.4 ms
    chunk total        229-230 -> 209-215 ms
    prefill end to end 1296 -> 1332 tok/s

The remaining 48 ms of attention cost is the number the tensor cores buy, and it is
a **declared numerics change**: the query and P are rounded to f16, which moves the
first-token logit by at most 0.17 against the scalar kernel where `-ctk q4_0`
already accepts 0.25.  Verified rather than assumed: the recall at 19583 tokens with
four access codes answers four for four, the decode NLL gate is untouched at
1.80954673 (the decode has its own kernel and the KV is written by the prepare
kernel), and the prefill gate across the three formats still passes with deltas of
0.0117 (q8_0) and 0.3198 (q4_0).

Two bugs of the first FA-2 are worth remembering because of how they hid.  The PV
always runs over 32 keys, so the tile's tail past the last valid one had to be
zeroed: a masked key contributes 0 * V, and with uninitialized shared memory that is
0 * NaN, which is why the first tile produced NaNs and why the error *fell* with
prompt length (5.7 logits at 16 tokens, 0.9 at 96).  And the k16 A fragment had its
two row pairs swapped - PTX wants (row+8, col) before (row, col+8) - which moves
half of the k extent to the wrong row group and costs about half the dot in the
score.  The finder was a new diagnostic, `DS4_QWEN38_GA_DUMP`, which prints the
first GA layer's attention rows: it turned "1-5 logits off" into "NaN in most
dimensions, row 0 identical".

The phase attribution of a chunk was wrong for as long as it existed, in two ways
that together hid where the time actually goes.  The collector attributes an
interval to the group of the mark that *closes* it, so the FFN's mark has to come
after the down projection and the residual rather than before the SwiGLU: with it
before, those 59 ms were filed under `head` and groups 3 and 7 stayed at zero.  And
the extra line subtracted a running total from per-chunk numbers, so every chunk
after the first printed a negative time for the post-attention norm.

With both fixed, a 336 ms chunk at 464 rows adds up with about 3 ms unaccounted:

| item | ms | share |
|---|---|---|
| **FFN up** | 53.3 | 16% |
| **FFN gate** | 60.1 | 18% |
| **FFN down + SwiGLU + residual** | 59.2 | 18% |
| attention core (GDN 50.5 + GA 29.9) | 80.4 | 24% |
| projections (qkv and o) | 57.7 | 17% |
| attention output | 19.6 | 6% |
| output head and final norm | ~3 | 1% |

So the chunk is **FFN-dominated at 51%**, and the head is not a large item at all:
the 57.6 ms bucket I had been calling "the head" was the FFN's own down projection,
which also means the 20.3 TFLOP/s I attributed to the head earlier measured the FFN
with head-shaped arithmetic, a number to discard.  The FFN's gate runs at 0.64 ms
per layer, 48 TMAC/s, which is the rate recorded for llama.cpp in this same
document: the prefill's matmuls are at the achievable rate, and that is the end of
the prefill's road rather than a missing optimization.

Splitting the GDN's state rows across more blocks was measured and **did not
transfer**: the standalone reproduction of the row loop gained 24% with four row
groups, and in the real kernel the same change left the GDN core at 51.1-51.6 ms
against 48.8, with the prefill falling from 1341 to 1246 tok/s.  The row loop is
about 37 of the kernel's 50 ms and the rest - the state load, four barriers per
token, the scalars - does not change, while 192 blocks no longer fit in one wave
(it is two resident blocks per SM by registers) and the tail of the second wave
eats the gain.  Reverted.  The lesson is the one the MMA spike taught the same day:
a microbenchmark measures the body, and the body is not the kernel.

The quantized KV cache could not be saved to disk at all, and the bug sat two
functions away from the code that had sized the arena correctly all along.  The
session arena asks for 576 bytes per position per layer for a q4_0 cache (32 blocks
of 18 bytes), 1088 for q8_0 and 2048 for f16; the payload sizing computed
saved_tokens * KV_DIM * kv_element_bytes and passed 2 for any GPU session, so it
demanded 2048 bytes per position from a tensor holding 576.  Every save then failed
with "session tensor is smaller than the payload" - in the server, where a cache
store is skipped with a log line rather than fatal, and in ds4-bench, where the
snapshot dies and the harness stops measuring at the frontier it failed on.  Nothing
had noticed because the release trunk runs f16 KV, where both numbers agree.

The load side was worse in principle: it chose its branch by element width, which is
2 both for f16 and for the block formats, so a quantized payload would have been
written raw into the block tensor.  That path was unreachable precisely because the
save never produced a file.  Both directions now go through one helper,
session_qwen38_kv_pos_bytes(), which is also what the arena calls, so the three
numbers cannot drift apart again.  Measured after the fix: the server writes cold
checkpoints again (173.63 MiB, 100.3 ms) and the next start restores them with a
cache hit (46.3 ms against the 1047 ms prefill it replaces); the bench runs every
frontier with q4_0 instead of stopping at 4096; and the f16 payloads are
byte-for-byte what they were - the difference between the two formats for 2048
tokens is exactly 2048 * 16 * 2 * (2048 - 576) bytes.

The decode cost of a quantized cache grows with depth, which the first measurements
at 2048 did not show: 3% there (47.1-47.6 against 49.1-49.2 tok/s) but 36% at 32768
(23.2 against 36.1), because the f16 mirror is rebuilt per chunk.  Prefill is
unaffected (531 against 528).  At 131072 there is no comparison to make: an f16
cache of that size does not allocate.

One ds4-bench finding from the same session was not a bug: --ctx-max appeared to be
divided by the KV ratio (8192 asked, 2048 measured), but the run had died at the
4096 frontier and the last surviving row had been mistaken for the whole curve.  The
defect was the redirection in the measurement, not the harness.

A second trunk quantization mix now runs: Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf,
10.44 GiB against 11.77.  Its name is a recipe, not a type - it carries nine
quantizations and six of them were unexecutable (IQ3_XXS, IQ2_S, IQ2_XS, IQ4_XS,
IQ1_M, IQ1_S = 59% of its tensors).  Five were already accepted by the loader's type
gate and already named in both CUDA dispatch tables, and the vendored MMQ/MMVQ
kernels implement all five; the only missing type was IQ1_S, and the vendored
kernels implement it too.  So the engine work was: the CPU dequantizer and vec_dot
for IQ1_S (ported from the same table the IQ1_M path already uses, iq1s_grid),
DS4_TENSOR_IQ1_S in the type gate, and the type in three dispatch lists - the
geometry table, the dense matmul switch and ds4_mmq_quant_dense_vec.  Two things
came out of it.  The loader's type table declared iq1_s as 110 bytes per block, the
size of IQ3_S, where the format is 50; nothing had ever read it, and any
implementation would have read the tensor at the wrong offsets.  And IQ1_S could not
take the MMA path even though the vendored kernels have tiles for it: the ds4-side
glue has no ds4_mmq_dense_impl<GGML_TYPE_IQ1_S> to link, which shows up as a link
error, so the format rides the same eight-row chunking as IQ1_M
(qwen38_gpu_matvec_rows), which is 0.05 GiB of tensors and costs nothing measurable.

Verification, in the order it was done: the CPU dequantizer is line-for-line the
llama.cpp reference (same dl, same delta, same grid index); the session test passes
on the file from the CPU path with LOGIT_MAX_ERROR 0; and the two paths agree on the
same sentence to 0.0027 nats (CPU 1.8733425 against CUDA 1.87606303), which is what
validates IQ1_S itself on both engines.  It does not validate the eight-row chunking:
that path was already exercised by the release trunk, whose single IQ1_M tensor
(blk.13.ffn_gate, 19.5 MB) goes through it in every prefill and has been covered by
the prefill gates all along.  Nor is IQ1_S alone in the "new" trunk's type list: the
release trunk already carries IQ3_XXS, IQ4_XS, IQ2_S, IQ2_XS and IQ2_XXS, so those
kernels were in production use before this change and nothing about them was
touched.  The release trunk is
untouched: the same gate still reads 1.80954673 bit for bit.  The pin was made
model-aware (DS4_TEST_QWEN38_EXPECT_NLL) so both values are gated.

The trade, measured: NLL 1.80954673 -> 1.87606303 (+3.7%) for -11% of model size;
decode 55.75 -> 58.55 tok/s and prefill 1095.3 -> 996.8 at the same context.  The
engine's resident-budget warning fires on this trunk even though its text describes
the separate NVFP4 sidecar; the measurement it predicts would collapse does not.

Two ds4-bench findings, recorded because the profile work ran into them and neither
is something to write a number on top of.  At ctx 131072 with q4_0 the bench cannot
create a session even with 13947 MiB of VRAM free ("failed to allocate Qwen CUDA
tensor hidden", 10 MiB) while ds4-server runs the same 131072/q4_0 shape in 15411
MiB, so the two do not allocate alike.  And for a quantized KV the bench's --ctx-max
is divided by the KV ratio - 8192 asked measures at 2048, 32768 asked at 8192 - so
the ctx_tokens column is the context actually run; one case (65536 asked with q4_0)
printed 65536 instead, so the rule is not self-consistent.  The same change replaced
a stale claim in the profile script: "1253 tok/s of prefill and 58.2 tok/s of decode
at 131072" was not reproducible, and the measurements that were taken say the format
costs a few percent of decode at equal context (47.1-47.6 against 49.1-49.2 tok/s at
2048) and that depth, not the format, is what moves throughput (15.3 tok/s at 65536).

Three of the five binaries had no runtime gate at all: the script built ds4-eval,
ds4-bench and ds4-agent and never ran them, and the last two had just been given
their `-ctk`/`-ctv` flags.  They have one now, and it costs twenty seconds: the eval's
grader self-tests, a prefill-only bench at 4096 (about 1400 tok/s, and a collapse to
nothing if the flags break), and an agent one-shot turn checked for its answer.

Two things came out of writing it.  Every step has to build what it uses, because the
CPU reference step runs `make clean` and by then nothing from the first build exists -
the first version of this step failed on a missing `ds4-eval` after passing by hand,
which is the same trap the `ds4` binary set earlier in the same script.  And the agent
turn reports

    ds4-agent: failed to save system prompt KV: unsupported routed quantization for KV save

which was a real bug of its own: `agent_kv_save_path` insisted the engine's *routed
expert* quantization is 2 or 4 bits, a DeepSeek-shaped requirement, so with Qwen the
agent's documented session save never happened.  The byte is a checkpoint identity
field: it is written into the header and compared on load against the same engine
call (`ds4_engine_routed_quant_bits`), which reports zero for a model with no routed
experts.  Zero was therefore already self-consistent on the load side; only the save
refused it, and for DeepSeek the check never fired, so removing it is a no-op there.
The agent now saves `sysprompt.kv` (2048 tokens, 292 MB of payload - the GDN recurrent
state is 201 MB of it) and reloads it on the next run, which the regression checks
together with the header byte itself.

The server's landing page now streams: it asks for `stream: true` and paints both
`reasoning_content` and `content` as they arrive, because this model thinks for a long
time before it answers and an empty box reads as a failure.

The quantized twin got the same treatment and the naive form of it **lost**, which
is worth recording because the reason is structural: q8_0's attention core went from
72.6 to **122.6 ms** (q4_0 72.7 to 112.1) when the dequantization was moved into
each row block's staging, even with a tile of f16 - half the shared a float tile
needs.  Reverted.

The FA-2 body is so cheap that the *staging* becomes the wall: dequantizing a value
costs four to six instructions, a 32-key tile is 16384 of them, and 160 threads
means about **600 instructions per thread per tile** against a body of 240 (QK) and
154 (PV).  In the scalar kernel the body was around 4600 instructions per warp per
tile, so the dequantization hid inside it.  Making the body twenty times cheaper did
not make it faster; it exposed what was always underneath.

And it is paid **twenty-nine times over**: every row block of a head re-stages and
re-dequantizes the same keys.  So the fix is not a faster dequantization but a
single one - dequantize the chunk's key range once into a scratch buffer, per layer,
and let every block read f16 from there.  That is a small kernel plus a transient
buffer (4 KiB per position per layer: 78 MiB at 19.5k, 512 MiB at 131k) reused layer
by layer, and it was done: a widening kernel writes an f16 mirror of the cache with
the cache's own layout, so **the f16 prefill kernel reads it unchanged** and no
quantized attention kernel exists any more.

    quantized attention core   73.0 / 73.4 ms -> 27.9 / 28.0
    f16 attention core         28.1 ms  (unchanged)
    prefill q8_0 / q4_0        1297 / 1272 -> 1415 / 1398 tok/s

The quantized prefill attention is now the same number as the f16 one: the format
costs nothing in the prefill any more.  One correction to the reasoning above: the
widening is **not** lossless.  Eight bits of code behind an f16 scale do not fit in
eleven bits of mantissa once multiplied, so rounding the product to f16 is a
fresh ~1e-3 relative error on the values - which shows up exactly where it should,
in the prefill gate's deltas against f16, which moved from 0.0117 to 0.0422 (q8_0)
and from 0.3198 to 0.3252 (q4_0).  Both stay far inside the gate's limits (0.50 and
0.80), and the recall at 19.9k with q4_0 answers four for four.

The prefill's host launch path was investigated as a project of its own
(`TODO-7e760186`: a CUDA graph over the layer loop) and the premise did not
survive measurement, so it was closed.  What was found, in order:

- `QWEN38PHASE-WALL`'s `outside_windows` is 111-119 ms of a ~342 ms chunk, and it
  is **not** host idle time.  Moving the FFN's phase mark from mid-block to the
  end of the block (a throwaway diagnostic) moved 57 ms between buckets without
  changing the total, and the same is true of the output head's bucket, so what
  lives in that 111 ms is the *unmarked* small work: norms, residuals, rope, the
  convolution, the KV writes.
- Host CPU contention does not move the prefill at all.  The engine is CPU-heavy
  - 3.76 s of CPU over a 3.87 s wall, 97% of a core - but that work overlaps the
  GPU.  The A/B, with the process pinned to one core:

  | load on that core | chunk wall | GPU windows | outside |
  |---|---|---|---|
  | none | 341.8 | 230.0 | 111.8 |
  | one spinner | 346.1 | 227.1 | 119.0 |
  | two spinners (33% of a core) | 344.7 | 231.9 | 112.8 |

  Two spinners on the engine's own core is *harsher* than the VM's 97% of a core
  that the project was motivated by, and the number does not move: the host
  launch cost is not on the critical path here.  A graph removes exactly that
  cost, which is why the project has nothing to take.
- The phase instrumentation itself costs 1.5% (1320/1321 tok/s with it,
  1337/1344 without), so it is not what the numbers above are measuring.

The one thing left unexplained is the 786 ms `outside_windows` recorded in the
original session under VM load.  It is not reproducible with CPU contention - the
harsher version of it above leaves the number where it was - so if it was real it
was something else: the VM sharing the GPU, a driver or kernel state, or a
different kind of load.  Recorded so that nobody re-derives it from scratch.

The prefill chunk kernel was finally checked against the **CPU reference**, which
is the one comparison that had never been made: the gates above are decode gates,
and the prefill fingerprints compare the GPU against its own past.  Building the
CPU side in a separate worktree (`git worktree add /tmp/ds4-cpu HEAD`, then
`make cpu` there) keeps the CUDA build untouched, and the two binaries take the
same prompt through `--dump-logprobs`.

    prompt      cpu                                 gpu                                 argmax   logit deltas
    16 tokens   561, 78802, 15822, 303              561, 78802, 15822, 303               same     0.014 - 0.038
    96 tokens   561 (18.739616)                     561 (18.846823)                      same     0.020 - 0.107

The second prompt is the interesting one: 96 tokens is one chunk and three 32-key
tiles, so it covers the tile loop, the staging that repeats per tile, and the
per-row masking that the two-rows-per-warp form introduced - which is exactly
where a mistake would live.  Same argmax, and differences of the order a
64-layer model accumulates between two implementations.

This is a deliberate check, not a regression step: the CPU prefill replays tokens
and runs at about 0.1 tok/s (96 tokens took 12 minutes of wall and 105 of CPU),
so a gate that runs it would dominate the suite.  The 16-token version is about
two minutes and can be repeated when the chunk kernel changes.

It also gives the quantized path a real footing: the CPU has no quantized KV, so
`-ctk`/`-ctv` can only be compared against the f16 GPU - but that f16 GPU is now
the one validated here, and the quantized rates sit 0.009 (q8_0) and 0.25 (q4_0)
away from it on the same prefill.

The attention's remaining 48 ms per deep chunk is its dot product: measured in
isolation (a standalone reproduction of the release body, same geometry), the
body costs 0.246 ns per (key, row) and the scalar dot plus its butterfly
reduction is **0.1357 ns** of that.  The same dot as tensor cores - one warp on
16 query rows, HMMA m16n8k16, the Q fragment held in registers across the whole
key range - costs **0.0191 ns**: **7.1x cheaper**.

| QK^T form | ns per (key, row) |
|---|---|
| scalar: 8 LDS.U16 + 8 conversions + 8 FMA + 5-shuffle butterfly | 0.1357 |
| HMMA m16n8k16, Q in registers | **0.0191** |

Extrapolated with the rest of the body unchanged, that is 0.246 -> 0.13 ns, about
**1.9x on the attention core, +7% on the prefill** - and the prefill is the only
place it shows, because the decode path keeps its own kernel and the KV contents
are written by the prepare kernel, so the NLL gate is untouched by construction.

Two obstacles the spike made explicit, both structural and neither fatal:

1. **The accumulator does not fit.** In the MMA layout a warp computing 16 rows
   against 256 output dimensions holds 128 f32 accumulators per thread, plus 64
   registers of Q fragment.  The output therefore has to be split across warps.
2. **The output cannot be split by dimension alone**, because a score needs all
   256 key dimensions: the warp that computes the scores must be the one that
   owns them, so the split has to be by *keys* into the PV, with the P tile
   staged in shared memory - the FA-2 shape.  One warp does QK plus the online
   softmax, writes P (16x8 f16, 256 bytes) to shared, and two to four warps each
   run the PV for their slice of the 256 output dimensions with their own
   accumulators, rescaling on the max and denominator the first warp broadcasts.

That is a kernel rewrite of a few hundred lines, not a change to the existing
one, and the numbers above are what it is worth: recorded in `TODO-52b82ef4` with
the fragment layouts and the two decisions, so it starts from a measurement
instead of an estimate.

The full FA-2 shape was then written and measured: **1.56x faster** (GA core
30.3/30.4/30.4 ms against 47.3/47.4/47.4, chunk 209 against 226) with the budget
respected - 79 registers against the scalar kernel's 79, 33.9 KiB shared, three
blocks per SM - which is what the staged attempt had failed to do.  Structurally
it works: five warps, one doing QK with HMMA m16n8k16 and the online softmax, four
doing the PV for 64 output dimensions each, the k loop outer and the n loop inner
so only four registers of the Q fragment are live at a time.

It is **not numerically right** and was not promoted: the first-token logits sit
1 to 5 away from the scalar kernel's, in a way that does not grow with prompt
length (5.7 at 16 tokens, 0.8 at 96), which is the signature of a near miss in a
fragment layout rather than of a wrong idea.  Seven bugs were found and fixed on
the way - the B fragment's key index and its packed pair in both the QK and the
PV, reads past the chunk in the query fragment, a `__syncthreads` inside a branch
(only one warp reached it), the missing attention scale, and a shared denominator
that was not rescaled with the register one.  What is left is the one thing that
was never checked: **the spike's fragment layouts were verified to build and to
time, never to compute the right numbers**, and a mirrored layout produces exactly
this symptom - scores that are correlated with the right ones but not equal.

So the first step next time is a numerical unit test of the two MMAs against a
scalar product, not another kernel.

The obvious shortcut was tried first - keep the whole kernel, add a ninth warp for
the MMAs, hand the scores over through shared memory - and it is a **measured
loss**: 57.6/57.8 ms against 47.8/48.2 for the scalar kernel on the deepest chunk,
20% slower, and reverted.  The reason is not the tensor cores, which do what the
spike said; it is what the shortcut does to the rest of the kernel:

| kernel | registers | shared | blocks per SM |
|---|---|---|---|
| scalar (release) | 79 | 32 KiB | **3** |
| staged MMA | **126** | **43 KiB** | **1** |

Register allocation is uniform across a kernel, so the score warp's 16x8 half
fragment plus the staged query and the score board cut the resident blocks from
three to one: a third of the latency hiding, which eats a dot that got seven times
cheaper.  The spike's extrapolation was optimistic for exactly this reason - it
measured the dot in a latency-dominated configuration of one warp per block and
said nothing about the register and shared budget the real shape has to live in.

So the viable form is the full FA-2 after all, with the output split across warps
so that no warp holds a 16x256 accumulator, and with the register count as a
first-class constraint rather than a consequence: under about 64 registers to keep
three blocks resident.  Anything that keeps the 16-row score fragment in one warp's
registers will land where this did.  There was also a real bug in the attempt worth
recording: the staged query is read by the score warp before the key loop, so it
needs a barrier right after the staging, and there was none.

A final experiment split Q8_1 activation quantization from MMVQ so the Qwen
attention and FFN projections could share one quantized row. Three 538-token
runs produced 798.1/798.4/787.3 warm prefill tok/s and
40.14/38.92/39.56 decode tok/s, with no repeatable improvement over the
retained path, so the API and forward changes were reverted rather than adding
permanent scratch/state complexity.

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

## Qwen3-VL video (frame sequences)

llama.cpp's own video path shells out to `ffmpeg`; ds4 stays dependency-free and
takes an ordered frame list instead. A frame sequence rides one temporal merge
pass: adjacent frames are paired for the temporal convolution slices and the
last frame is paired with itself when the count is odd, so the temporal grid is
`ceil(frames/2)` and MRoPE advances by that count. Total image tokens across all
frame groups stay under 4096 and every frame must have the same dimensions.

```sh
# CLI: 2..128 ordered PNG/JPEG frames
./ds4 -m "$M" -p "" --vision mmproj.gguf
ds4> /video f0.png f1.png f2.png
```

The API is `ds4_engine_vision_encode_frame_files()` /
`ds4_engine_vision_encode_frame_memory()`, and the prompt carries
`<|video_pad|>` instead of `<|image_pad|>` (layout
`DS4_VISION_LAYOUT_QWEN3VL_VIDEO`). A frame sequence fingerprints as one unit
over `(frame count, dimensions, per-frame fingerprints)`, so an image checkpoint
can never be reused for a video or for a different frame list.

Gates:

```sh
make tests/test_qwen3vl_vision tests/test_qwen3vl_session CUDA_ARCH=sm_120
./tests/test_qwen3vl_vision "$M" mmproj.gguf /tmp/white224.png
./tests/test_qwen3vl_session "$M" mmproj.gguf /tmp/white224.png
```

The still-image checksum must stay `108.206111801`; the temporal gate checks
that one still fed as two identical frames reproduces the still embedding bit
for bit and that an odd three-frame sequence equals the still embedding twice.
The session gate additionally covers video sync, no-op sync, fingerprint
mismatch rejection and a DSV4 payload round trip with exact continuation logits.

## Qwen3.8 MTP draft head and speculative decoding

The draft head lives in a separate GGUF. `Qwen3.8-27B-NVFP4-MTP-HIGHEST.gguf`
holds a 64-block trunk ds4 never executes plus `blk.64` (one dense
global-attention block) and `blk.64.nextn.{eh_proj,enorm,hnorm,shared_head_norm}`.
ds4 binds only those tensors, preloads their merged byte span (0.79 GiB) and
reuses the trunk's token embedding and LM head, which is what
`shared_head_norm` implies. Running the sidecar itself is not possible on 16 GB:
the file is 23 GB.

```sh
./ds4 -m "$M" --mtp-model Qwen3.8-27B-NVFP4-MTP-HIGHEST.gguf --raw \
     -p 'The capital of France is Paris.' --temp 0 -n 64
```

Draft row `r` pairs the trunk hidden left by position `r-1` with the token
committed at `r` — the same shift llama.cpp's `qwen35` MTP graph uses — so the
row's logits predict exactly what the trunk head just predicted. The draft block
keeps its own K/V rows and never writes trunk state.

### Speculative rounds

`ds4_session_qwen38_spec_step()` commits the pending token through the ordinary
token path, drafts up to four further tokens and verifies all of them with one
batched trunk pass. Chunk row `i` holds the trunk's argmax for the token after
position `p+i`, so draft `i+1` is accepted exactly when that argmax is the
drafted token; a rejected draft ends the round there and the chunk's argmax at
that row is the correcting token. A partial accept rewinds the GDN convolution
and recurrent state to the snapshot taken before the verify chunk and replays
the accepted prefix plus the correcting token.

Measured on RTX 5070 Ti, 16 GB, greedy, 48-token run:

| drafts per round | tok/round | tok/s | vs one-token |
|---|---|---|---|
| 3 | 2.33 | 47.6–48.2 | 1.15–1.16x |
| **4** | **2.87** | **48.9–49.9** | **1.20x** |
| 5 | 2.95 | 45.8 | 1.10x (and drifted at token 38/39) |
| 6 | 3.25 | 45.2 | 1.09x |

Every extra draft re-reads the draft block and the full 248k-vocabulary LM head,
so decode is bandwidth-bound and four drafts is the measured optimum. Draft
acceptance is 34/39 (87.2%) against the trunk argmax.

Correctness gate:

```sh
make tests/test_qwen38_mtp CUDA_ARCH=sm_120
./tests/test_qwen38_mtp "$M" Qwen3.8-27B-NVFP4-MTP-HIGHEST.gguf 'The capital of France is Paris. The largest ocean on Earth is the Pacific Ocean.'
```

It asserts that the speculative stream reproduces one-token greedy decoding for
the whole reference run, that rounds really commit more than one token, and that
the sidecar leaves the trunk path bit-identical (same tokens and same final
logits with and without the sidecar). The token-for-token equality is measured
per run, not guaranteed by construction: acceptance only ever confirms the
batched trunk's own argmax, so a near-tie between two logits can still flip when
the chunk and token reduction orders disagree.

## Video containers (ffmpeg, optional)

A container is an input convenience, never a second preparation path: ffmpeg
decodes to raw RGB and everything after that is the frame-list path. The
decoder is `ffprobe`/`ffmpeg` from PATH, exec'd without a shell, so there is no
link-time dependency and no shell quoting of a file name. When the tools are
missing the call fails with a message that names them and the frame-list entry
points keep working.

```sh
# one container: frames are sampled evenly over the duration
./ds4 -m "$M" --vision mmproj.gguf -c 16384
ds4> /video clip.mp4

# sampling policy (default 32, range 2..128)
DS4_VIDEO_FRAMES=16 ./ds4 -m "$M" --vision mmproj.gguf -c 16384
```

Policy and identity, both deliberate:

- `max_frames` evenly spaced samples over the whole duration, realised with
  ffmpeg's `fps` filter, so the same file always yields the same frames. A
  different value is a different view of the video, not a different file.
- Frame fingerprints hash the decoded pixels (domain-separated, per frame), not
  the container bytes, so a cache identity describes what the model saw. A
  container and its extracted PNG frames therefore have different identities
  even though they produce identical embeddings.

Gate:

```sh
make test-qwen3vl-video CUDA_ARCH=sm_120
```

It generates its own fixture with ffmpeg (`testsrc`, 224x224, 2s), then checks
that the requested frame count is honoured, that two decodes give identical
pixels and fingerprints, that different frames never share a fingerprint, that
four frames merge into two temporal groups, that the identity survives a second
decode, and — the point of the whole path — that the container's embedding is
**bit-identical** to the same frames passed as PNG files. Skips itself when
ffmpeg is not in PATH.

## Q6_K and the self-contained MTP file

`Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` is the trunk with the draft head inside it:
the same 851 trunk tensors plus block 64 and its `nextn.*`, 11.29 GiB against
10.96 for the trunk alone.  The engine used to refuse that layout by design and
require the separate NVFP4 sidecar, which costs 1.09 GiB resident; the draft
block here is Q6_K (eight tensors) and F32 (seven norms), and Q6_K turned out to
be declared in the loader's type enum but never executable - missing from the
geometry table, from both MMQ and MMVQ dispatchers, from the batched and
single-row matvec switches and from the public matmul entry.  The vendored MMQ
and MMVQ kernels had carried Q6_K all along; only the ds4-side wiring was
absent, so the fix was to name it in the five places a quant type has to appear.

Served with `--mtp` the draft head is bound from the same file and the embedding
table and LM head stay the trunk's, exactly as `--mtp-model` does for a sidecar.
Greedy output over 40 tokens is identical to the trunk run without drafting, the
trunk NLL gate is unchanged at 1.81334038, and decode measures 64.9 tok/s
against 54.6 for the NVFP4 sidecar on the same prompt - the Q6_K draft is both
smaller and quicker.

The context that fits follows from the size: 24576 tokens with the draft head
active (28672 is refused by the residency guard, which prints the arithmetic),
against 16384 with the sidecar.

## KV cache quantization: -ctk and -ctv

The attention KV cache costs 2048 bytes per position per layer in each of K and
V: 64 KiB per token, so 32768 tokens is the whole context a 16 GiB card can hold
in f16. `-ctk q8_0 -ctv q8_0` stores 32 values per block behind an fp16 scale,
1088 bytes per position, 34 KiB per token, about **66k tokens** of context.

| KV | bytes per token | context that fits | MEAN_NLL |
|---|---|---|---|
| f16 (default) | 64 KiB | ~36k (measured 32768) | 1.81334038 |
| q8_0 | 34 KiB | ~66k (measured 65536 loads) | 1.81927471 (+0.33%) |

Recall was measured rather than assumed: one prompt of 19530 tokens carrying four
access codes at 2%, 25%, 50% and 75% of the context, asked for at the end.

    f16    code 32: PASS-A-4821  code 160: PASS-B-1397  code 325: PASS-C-7342  code 490: PASS-D-9615
    q8_0   code 32: PASS-A-4821  code 160: PASS-B-1397  code 325: PASS-C-7342  code 490: PASS-D-9615

Four for four in both, same answers. That bounds catastrophic loss, not subtle
degradation - the 0.33% NLL is the measure for that - but it is the test that
would have caught a broken quantization.

The price today is prefill speed: the same prompt of 19530 tokens took 53 s in
f16 and 1 min 26 s in q8_0, a factor of about 1.8 that shows up in the prefill
rate (514 against 282 tokens/s).

Staging a shared tile in the q8_0 prefill twin was the obvious candidate and it
was tried early, on its own: 8704 bytes per side against 16384 in f16, one
contiguous run per (key, head) rather than a strided gather. It measured nothing,
1 min 24 s against 1 min 26 s, and was reverted. That result was correct and the
conclusion drawn from it was incomplete: the tile removes traffic, and traffic
was not the limit *while the per-key chain was still long*, because dequantizing
a q8_0 value adds a byte load, a scale load, a conversion and a multiply inside
that chain against one load and a conversion in f16. Once the chain itself got
shorter - two query rows sharing one dequantization, and the online softmax off
lane 0 and onto every lane - the traffic the tile removes became visible, and the
tile came back as part of the same rewrite.

Measured on the deepest prefill chunk of a 2000-token prompt, GA attention core
in ms, before and after:

| KV | before | after | prefill before | prefill after |
|---|---|---|---|---|
| f16 | 47.6 | 47.9 | 1314 | 1324 t/s |
| q8_0 | **252.3** | **74.1** | 941 | **1269 t/s** |
| q4_0 | **223.4** | **72.6** | 893 | **1266 t/s** |

So the quantized prefill attention was 5.3x the f16 one and is now 1.5x, and the
prefill rate gained 35% (q8_0) and 42% (q4_0) to within 4% of the f16 one.  The
last step was the interesting one: the dequantization itself was per value *inside
the key loop*, which is why the tile alone had never helped.  The tile now holds
**dequantized floats**, written once by the staging pass and read by the same
readers the f16 kernel uses, so the dequantization is paid once per value instead
of once per (value, row).  Sixteen keys of dequantized floats are 32 KiB for both
tensors, the same shared budget the f16 kernel uses.  The expression is unchanged
- scale times code - so the emitted values are unchanged too, bit for bit (max
logit delta 0.000000 over the same prefill, gate 1.80954673 on the f16 trunk).

One bug is worth recording because of where it hid. When the f16 kernel moved to
two rows per warp, the shared grid became `(n+15)/16` while the quantized twin
kept `blockIdx.y * 8`: it computed **half the rows** and nothing noticed, because
the gates are decode gates - the session test evaluates the prompt one token at a
time and never runs the chunk kernel - and on the test prompt the greedy
continuation is identical either way. The regression now has a prefill gate: the
same prompt in all three formats, compared to f16 rather than to a fixed number
so it survives numerics-neutral work, with the tolerances calibrated on both
sides (correct kernel 0.10 and 0.23, half-rows kernel 5.37 and 5.39). It was
verified to fail on the broken build, because a test that cannot fail measures
nothing.

The kernels are duplicated rather than unified because the build uses
`--use_fast_math`: an earlier attempt threaded a runtime flag through the shared
read helpers and moved the release path to 1.80874846. The f16 kernels are byte
for byte untouched and qwen38_ga_*_q8 are new code with their own baseline.

`-ctk` and `-ctv` must name the same type - K and V share one layout and one pair
of kernels - and q4_0 is refused until its twin exists. The disk KV payload
refuses to write or read while it does not record the format, because a q8_0
cache reloaded as f16 would decode garbage. `DS4_KV_Q8=1` is the hook the test
binaries use, since they take no engine options.

## Deep decode: split-KV, measured and behind a switch

A decode token at a 21.510-token context took 190 ms, of which 195-202 of a 217 ms
token at 24k is the GA attention: one block per head walking the whole key range,
12.5 ms per layer, about 730 cycles per key with only 24 to 48 warps of work on 70
SMs. Cutting that range into N parts, each block leaving a partial online-softmax
state for a small combine kernel, measures:

| N | tok/s | ms/token | gain |
|---|---|---|---|
| 1 (baseline) | 5.25 | 190 | 1.0x |
| 2 | 9.13 | 109.5 | 1.74x |
| 4 | 15.86 | 63.0 | 3.02x |
| 8 | 23.97 | 41.7 | 4.57x |
| 16 | 27.84 | 35.9 | 5.3x |
| 64 | 30.36 | 32.9 | 5.8x |
| 128 | 30.45 | 32.8 | 5.8x |

So **5.8x**, saturating around 32-36 ms per token, against 38.8 tok/s for llama.cpp
at the same depth on the same file. Recall does not suffer: four access codes at
2/25/50/75% of a 19530-token prompt come back four for four, identical to f16
without the split and to q8_0.

  trunk NLL, decode with split-KV  1.80954673   (the release path: this is the
                                                 association the engine now uses,
                                                 against 1.81334038 before it)
  trunk NLL, q8_0 KV               1.80900178   (16 tokens: inside this test's
                                                 noise against f16, see below)
  trunk NLL, q4_0 KV               1.82242633   (+0.7%: the first format whose
                                                 difference is outside the noise)

Four hypotheses have been measured and dropped on the way: the per-key load
latency (a shared tile gives 3.95 against 4.43 tok/s), the partials in scratch
(N=1 costs 189 ms against 190 for the single-row shape, so scratch and combine
are free), bandwidth (45 GB/s against the ~900 the card has), and the instruction
count (removing the eight-fold score redundancy changes nothing: 27.84 against
28.02 at N=16). **About 30 ms per token of the remaining 33 is unexplained**, and
that is the next thing to measure rather than guess.

Writing the de-duplicated kernel found a silent bug in the version being
measured: it guarded with tid >= 128 while the block has 256 threads, so it
accumulated only half of the 256 output dimensions and left the rest of the
scratch holding the previous token's partials - attention-shaped numbers, so the
NLL read a plausible 1.80977761. Only the layout change, which moved a number
that had to stay still, exposed it.

It is the decode path now, at 64 parts, and not a switch: two shapes that
disagree in the last bits cannot both be "the" engine. The single-row shape it
replaced was better only before it was made correct - at 4k it measured 20.7
tok/s against 46.7 for the split, at 21.5k 5.25 against 30.4.
