# Qwen3.8-27B support — working status

Moved here from `misc/`, which is not tracked by git: this document records the
support state and the measured gates, so it belongs next to the other Qwen
notes rather than in an ignored directory. The closing sections below predate
the video, container and MTP work; `tests/QWEN38.md` is the current gate list.

Branch: `feat/qwen38-27b`. This ignored/local note replaces the earlier draft:
claims about gated configs, invalid head sizes, embedded MTP in the IQ3_S file,
lossless requantization, model ID 2, and a validated-but-chaotic CPU forward
were incorrect or unverified. Local GGUFs and llama.cpp source are the evidence.

## Scope and target

- Native C text **and single-image vision** MVP; CUDA first on RTX 5070 Ti 16GB.
- Preserve the existing mixed-IQ file; no Q4_K requantization.
- Shared engine/session/frontends, not a second runner or a llama.cpp dependency.
- llama.cpp MIT tables/kernels may be adapted with attribution.
- The user's normal baseline is default llama-mtmd-cli, not the experimental
  options in exec_qwen3_8_27b_max_perf.txt. Speculation is not assumed enabled.
- CUDA text/vision, template/tool semantics and recurrent disk-cache codec are
  implemented on this branch. The CPU path remains the independent numerical
  reference rather than the production performance target.

## Actual files and shape

Directory: `/home/diegom/AI-Projects/llama.cpp/build/bin/models/Qwen3.8-27B/`.

- `Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf`: architecture `qwen35`, 851 tensors,
  approximately 11.7 GB / 10.95 GiB. No `blk.64+` MTP block.
- `mmproj-Qwen3.8-27B-BF16.gguf`: 931,146,528 bytes, architecture `clip`,
  projector `qwen3vl_merger`, 334 tensors (224 F32, 110 BF16); observed hidden
  1152 and FFN 4304. Exact deepstack/encoder integration still needs inspection.
- Embedded chat template plus local `chat_template.jinja`; XML tool calls.

LM: hidden 5120, FFN 17408, 64 blocks, vocab 248320, RMS epsilon 1e-6,
context 262144. Family ID **2**, cache/model variant ID **4** (0..3 occupied).

- 48 GDN blocks: `(layer+1)%4 != 0`, 16 Q/K heads, 48 V heads, head dim 128,
  inner 6144, convolution kernel 4 across 10240 QKV channels.
- 16 GA blocks: 24 Q heads, 4 KV heads, head dim 256. Q projection is 12288
  because it contains per-head query **and gate**; output width is 6144.
  These widths need not equal the residual hidden width 5120.
- Partial IMRoPE: 64 rotary dimensions, sections [11,11,10,0], base 1e7.
  Text positions are (pos,pos,pos,0), using split-half pairs (i,i+32), not
  adjacent pairs. Vision will require separate spatial position handling.
- All blocks: pre-attention RMS, attention output + residual, post-attention
  RMS, dense SwiGLU FFN + residual. Untied embedding/output matrices.

Quant histogram: F32 353, BF16 96, Q2_K 13, Q4_K 39, IQ2_XXS 5, IQ2_XS 9,
IQ3_XXS 78, IQ3_S 144, IQ2_S 17, IQ4_XS 96, IQ1_M 1.

## Reference computation

Reference: local llama.cpp `src/models/qwen35.cpp`, `models.h`, CPU RoPE ops,
GDN kernels and GGML quant decoders.

GDN, each token/head:

1. RMS input, project QKV, z, alpha, beta.
2. Depthwise causal convolution + SiLU; history stores **raw projections**.
3. L2-normalize Q/K with epsilon; Q/K head for V head h is h%16.
4. Let `decay=exp(ssm_a * softplus(alpha+dt))`, `b=sigmoid(beta)`.
   With S represented key-by-value: `Sbar=decay*S`,
   `delta=b*(v-Sbar^T*k)`, `S=Sbar+k*delta^T`, `o=S^T*q/sqrt(128)`.
5. Per-head RMS(o) * SiLU(z), output projection, residual and FFN.

GA: Q/K RMS, partial IMRoPE, causal softmax over **positions for one mapped
KV head** (`kv_head=query_head/6`), sigmoid gate, output projection.

## Implemented / checked

- Phase 1a: six IQ formats, tables, CPU dequant/dot.
- Phase 1b: pinned qwen35 config and weight binding; `--inspect` works.
- Phase 1c: native pre-tokenizer and Qwen chat template. Five llama.cpp-derived
  vectors cover multilingual/combining/ZWJ text, code punctuation, and chat,
  thinking, tool and vision specials. Complete historical tool rendering is
  checked by an exact `/apply-template` prompt hash. Fixed the
  single-multibyte-whitespace progress bug.
- Phase 1d: CPU forward, GDN history/state, GA KV, FFN/logits, greedy CLI,
  session sync/eval/reset/free. Prefill is sequential token replay.
- Added explicit rejection of unsupported recurrent payload save/load and
  distributed/TP/sliced/MTP/steering combinations. GPU inference still rejects.
- Fixed real bugs: wrong convolution history, incorrect GA head mixing,
  adjacent RoPE pairs, missing GA output projection, plus earlier residual
  aliasing, scratch overflow and IQ2_XXS dequant scaling fixes.
- CPU memory estimate now counts Qwen's actual GA KV and GDN state instead of
  reporting zero DeepSeek-shaped KV. FP32 GA cache: 128 KiB/token, about 4 GiB
  at 32768 context, plus ~144 MiB recurrent state and ~5.6 MiB conv history.

### Results (bounded smoke checks, not comprehensive quality certification)

See `tests/QWEN38.md` for reproducible commands.

- RoPE, conv history, gated GQA and multi-step GDN unit checks pass, including
  ASan/UBSan. Full CPU session test on `The quick` also passes ASan/UBSan.
- Nine quant types match ggml dequantization on 64 synthetic four-block rows
  per type, plus first/last rows of all **402** quantized tensors in this GGUF.
  Dot products checked against float inputs and dequantized ggml weights.
- Session eval vs full sync/rewrite/shorten+extend: max final-logit error **0**.
- Raw greedy `The capital of France is` -> ` Paris.\nThe capital of` (6 tokens).
  Both ds4 and llama.cpp rank token 11751 (` Paris`) first after the prompt.
- Raw next-token NLL, identical IDs, no BOS/template, llama CPU operation
  offload disabled:
  - `The capital of France is Paris.`: 6 scored tokens, ds4 **2.43708414**,
    llama.cpp **2.42675939**.
  - `The quick brown fox jumps over the lazy dog. In a small village, a baker
    makes fresh bread every morning.`: 22 scored tokens, ds4 **1.73504823**,
    llama.cpp **1.72583734**; mean |delta logprob| **0.02385070**, max **0.09440743**.
- `make cpu`, Q4_K/MXFP4 dot tests, session-state/TP-command unit tests, Linux
  memory test, extractor self-tests, CPU-built `ds4_test --server`, and
  DeepSeek `--inspect` pass. Full `make test`, DeepSeek q1..q4 generation gates,
  CUDA/Metal/SSD integration and performance gates have **not** been run here.
- Optional ASan quant run with dlopen/dlclose of the external ggml library
  reports a 104-byte leak in loader/library initialization. Standalone kernel
  and full-session sanitizer runs without that library are clean.

## Completion scope and final gates

The CUDA text-and-single-image MVP is implemented. Native packed MMVQ/MMQ,
bounded 256-token prefill, GDN/GA inference, FP16 GA KV, Qwen3-VL preprocessing
and encoder, multimodal MRoPE, recurrent DSV4/KVC state and Qwen tool protocols
all have dedicated regression gates. The 32k text context and maximum 4096
merged-image-token cases were measured on the 16GB target GPU. The loader pins
the supported language and projector shapes and rejects unsupported execution
modes instead of silently accepting related Qwen variants.

The closing regression consists of the mixed-IQ oracle, CUDA NLL/session/replay,
server template/tool protocol, image preprocessing, vision checksum and public
multimodal session tests. Metal, ROCm and universal quality equivalence remain
explicitly outside this hardware-scoped MVP. The normal llama.cpp result remains
the performance reference, not a required bitwise or throughput-equivalence
claim.

Video and the MTP draft head were added after that MVP and are covered by their
own gates (`tests/QWEN38.md`): a frame sequence rides one temporal merge pass
with its own `<|video_pad|>` layout and a frame-sequence fingerprint, and the
`Qwen3.8-27B-NVFP4-MTP-HIGHEST` sidecar runs as a draft head over the IQ3_S
trunk with verified greedy speculation (1.20x measured, stream equal to
one-token greedy over the gate's run). The sidecar never changes the trunk
path: the same greedy run with and without it produces bit-identical tokens and
final logits, and no GGUF is ever written or requantized.

Preliminary CUDA llama-bench reference (one repetition, no vision): pp512
**1628.69 t/s**, tg128 **53.24 t/s**, ngl=-1, build `00ed45216 (10870)`.
Repeat robustly before using it as an optimization acceptance gate.

### Phase 2a update — CUDA mixed-IQ primitives

- Exposed the already-vendored llama.cpp MIT MMVQ for IQ2_XS, IQ2_S,
  IQ3_XXS, IQ3_S, IQ4_XS and IQ1_M; no duplicate decoder and no persistent
  dequantized weight cache.
- Added exact IQ2_S GPU get-row for the actual token embedding.
- RTX 5070 Ti / sm_120a test passes constant and varied activation cases for
  all nine quant types at K=5120; host oracle is ggml CPU dequantization.
- Completed the pre-existing CUDA 12.8 rsqrt wrapper conversion in GLM/KDA and
  GLM-vision code because the current base otherwise failed to compile with
  `identifier rsqrtf is undefined in device code`.
- `make -B ds4 CUDA_ARCH=sm_120` succeeds. End-to-end Qwen CUDA graph remains
  unimplemented at this checkpoint.

## Long context: the -ctk / -ctv KV cache type (CUDA)

The attention KV cache holds 2048 bytes per position per layer in each of K and
V, which is 64 KiB per token in f16, so a 16 GiB card runs out of context at
32768 tokens. `-ctk q8_0 -ctv q8_0` stores 32 values per block behind an fp16
scale - 1088 bytes per position, 34 KiB per token - and roughly doubles the
context that fits:

| KV | bytes/token | context that fits | MEAN_NLL | recall at 19.5k |
|---|---|---|---|---|
| `f16` (default) | 64 KiB | 32768 measured | **1.80954673** | 4/4 |
| `q8_0` | 34 KiB | ~66k (65536 loads) | **1.80900178** | 4/4, identical |
| `q4_0` | 18 KiB | **131072 loads** | **1.82242633** (+0.7%) | **4/4, identical** |

The NLL figures are with the split-KV decode, which re-associates the softmax and
is why they differ from the 1.81334038 this document used to quote.  Over 16
tokens f16 and q8_0 differ by less than that test can resolve; **q4_0 is the first
whose difference is outside the noise**, and the instrument that decides between
them is the recall test, where all three answer four for four with identical text.

Writing neither flag is exactly the previous behaviour: f16 stays the release
path and its gate is unchanged. The names are llama.cpp's, so the values are the
ones that command line already has.

```sh
# long context, quantized KV
./ds4-server -m <trunk> -ctk q8_0 -ctv q8_0 --ctx 65536 ...

# unchanged default
./ds4-server -m <trunk> --ctx 32768 ...
```

Recall was measured rather than assumed: four access codes placed at 2%, 25%,
50% and 75% of a 19530-token prompt, asked for at the end. Four for four in both
formats, with identical answers. That rules out a broken quantization, not
subtle degradation - the NLL figure is the measure for that.

Three things to know before choosing it:

- **Prefill is about 1.8x slower**: 282 tok/s against 514 on the same prompt.
  The attention is bound by the per-key latency chain, and dequantizing a q8_0
  value costs a byte load, a scale load, a conversion and a multiply inside that
  chain. A shared-memory tile was tried for this and measured nothing.
- **The disk KV cache refuses to write or read in a quantized session.** The
  payload header does not record the format yet, and a q8_0 cache reloaded as
  f16 would decode garbage. The server reports the checkpoint as failed and
  keeps serving; the cost is that every new request pays its own prefill.
- **K and V must name the same type.** They share one layout and one pair of
  kernels; a mixed pair is refused at startup rather than stored as one.

`q4_0` reaches **131072**: 576 bytes per position is 18.4 KiB per token, and a
server at that context measured 30.3 tok/s of decode at 19.5k with the recall
intact.  The engine prints the format it chose at startup and records it in the
disk KV payload, so a cache written by one format is never read by another.

The kernels are duplicated rather than unified on purpose: the build uses
`--use_fast_math`, so a refactor that is only "semantically" equivalent can move
the release path's numbers, and one did. `DS4_KV_Q8=1` selects the same format
for the test binaries, which take no engine options.

### Deep decode, and what is not a supported mode yet

Decode slows down with depth: 12.95 tok/s at 7.1k of context, 7.51 at 14.3k, 5.25
at 21.5k. The cost is linear in the context and it is the attention - a decode
token at 24k spends 195-202 ms of its 217 in it, one block per head walking the
whole key range. A split-KV variant (the key range cut into parts, combined with a
second kernel) measures **5.8x** at 21.5k, saturating at 30.4 tok/s, with recall
unchanged.

It is **the decode path now** (64 parts), and the remaining cost is understood:
of the 28 ms a token takes at 19.5k, 16.3 are the q/k/v projections - that is the
10.95 GiB of weights streamed once per token, about 640 GB/s, which a single token
cannot avoid - 8.5 are the attention, and 2.4 everything else. The trunk NLL moved
with it, from 1.81334038 to 1.80954673, because combining partial softmax states
re-associates the sum.

Also worth knowing: **MTP drafting only pays at short contexts.** At 24k it buys
about 3% (4.56 against 4.43 tok/s) because each draft walks the same long key
range, while at short contexts it is worth +30% (64.9 against 42.8 tok/s). So use
`--mtp` for chat and leave it out for long documents; the flags give both profiles
already.
