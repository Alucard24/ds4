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
measures 753-763 tok/s with a 2000-token prompt at 770-776 tok/s on the RTX
5070 Ti.
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
state rows each instead of four warps walking thirty-two.  Only the first 128
threads own a q/k/o element there; threads 128..255 feed zeros into those three
reductions, and adding 0.0f is exact, so every reduction keeps its order and its
value.  A four-row variant was not repeatable (212 ms on one run, 290 on the
next) and was dropped.  Trunk NLL unchanged at 1.81334038 through all of it, and
decode picked up the shorter chain as well (42.6 tok/s greedy, 55.0 with MTP). The recorded normal llama.cpp baselines remain faster
(pp512 1628.69 tok/s, tg128 53.24 tok/s); matching llama.cpp is not claimed.
The MTP speculative round now reaches 1.30x on this prompt (41.4 -> 53.7 tok/s)
with the stream identical to one-token greedy.

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
