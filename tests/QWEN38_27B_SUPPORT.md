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

The f16 prefill runs its attention on tensor cores (flash-attention-2 shape): the
query and the attention weights are rounded to f16, which moves the first-token
logit by at most 0.17 where `-ctk q4_0` already accepts 0.25, and buys 1.6x on the
attention and about 3% of prefill.  The recall at 19583 tokens is unaffected: four
for four.  The decode path is a different kernel and is untouched.

The NLL gate is a decode gate - the session test evaluates the prompt one token at
a time and never runs the chunk kernel a real request's prefill uses.  The
regression now checks the prefill too, by running the same prompt in all three
formats and comparing the first-token logit to f16 instead of to a fixed number,
with tolerances calibrated on both sides.  It exists because a mismatch between
the f16 and quantized chunk kernels' row indexing left half the rows stale and no
gate above noticed.

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

- **Prefill is the same as f16**: 1415 tok/s (`q8_0`) and 1398 (`q4_0`) against
  1423 on the same prompt, where it used to be 941 and 893.  A quantized chunk
  widens its key range to f16 once per layer and then runs the f16 attention kernel,
  so the format costs nothing in the prefill; the widening rounds the dequantized
  values to f16, an extra ~1e-3 on them, which is visible in the prefill's logits
  and not in the recall (four for four at 19.9k with `q4_0`).  The
  quantized prefill twin now has the same shape as the f16 one - a shared tile,
  two query rows per warp, the softmax off lane 0 - and, in the step that
  mattered, its tile holds **dequantized floats** instead of the cache's bytes:
  the dequantization is then paid once per value by the staging instead of once
  per value *and row* inside the key loop, which is what had made a tile look
  useless the first time it was tried.  Its attention core went from 252 ms to 74
  (`q8_0`) and 223 to 73 (`q4_0`) on the deepest chunk, against f16's 48.
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

Also worth knowing: **MTP drafting is close to free at short contexts and nearly
absurd at depth.** Measured after the split-KV decode, so the attention is no
longer what limits the verify pass:

| context | decode without | decode with | prefill without | prefill with |
|---|---|---|---|---|
| ~2k | 42.8 tok/s | **64.9 tok/s** (+52%) | - | - |
| ~16k | 44.3 tok/s | **46.2 tok/s** (+4.5%) | 1033 tok/s | **802 tok/s** (-22%) |
| ~20k (ctx 24576) | 41.8 tok/s | 37.2 tok/s (-11%) | 854 tok/s | 645 tok/s |

The last row is not a drafting result, it is a residency one: at ctx 24576 the
draft head does not fit next to the KV any more, the engine says so ("sidecar
0.79 GiB + 0.17 GiB state … exceeds the resident budget … 16k measured safe"),
and everything slows down, prefill included - the prefill never touches the draft
head, so what is being paid there is the weight cache thinning out around it.

The reason depth costs drafting so much is the verify pass: one trunk pass over
four drafted tokens pays the 10.95 GiB of weights once - which is the whole point
- but pays the attention four times, and at 16k the attention is a third of a
token. So `--mtp` is for chat, which is what the default `run-qwen-server.sh`
profile is for (ctx 16384 with the draft head, the largest context where it
fits); use the `long` profile for documents.

## The server profiles

`run-qwen-server.sh` is the shortcut for a server that is ready to use.  It has
four profiles and can be adjusted without editing it:

| profile | model | context | KV |
|---|---|---|---|
| `merged` (default) | trunk with the draft head inside it | 16384 | f16 |
| `sidecar` | trunk + NVFP4 draft sidecar | 16384 | f16 |
| `long` | trunk | 32768 | f16 |
| `q4` | trunk | 131072 | q4_0 |
| `xxs` | IQ3_XXS trunk with the draft head inside it | 32768 | f16 |

`./run-qwen-server.sh --help` prints the same list.  Any profile accepts:

    DS4_CTX         context tokens            (default: the profile's own)
    DS4_CTK/DS4_CTV KV type, f16 (default), q8_0 or q4_0; K and V must match
    DS4_HOST        bind address              (default 127.0.0.1; 0.0.0.0 for the LAN)
    DS4_PORT        port                      (default 8080)
    DS4_POWER       1..100 duty cycle         (default: the engine's 100)
    DS4_QWEN_MODEL_DIR, DS4_KV_DIR

so `DS4_CTK=q8_0 DS4_CTV=q8_0 DS4_CTX=65536 ./run-qwen-server.sh long` is the f16
profile made twice as wide without touching the file.  Mismatched `DS4_CTK` and
`DS4_CTV` are refused with a message instead of starting.  All profiles share one
disk KV directory and always pass `--kv-cache-reject-different-quant`, so a
checkpoint written with one KV type is never resumed by another.

Why `q4` exists: attention K/V is 64 KiB per token as f16, and at 131072 tokens
the f16 K alone asks the allocator for 4096 MiB, which this 16 GiB card refuses.
q4_0 stores it at about a quarter, and the profile starts and serves at 131072
with 15411 MiB of VRAM in use.  Measured on the card: at the same context q4_0
costs a few percent of decode at 2048 (47.1-47.6 against f16's 49.1-49.2 tok/s) and
much more deeper down - 23.2 against 36.1 at 32768. The two paths differ there: the
prefill widens the chunk once into an f16 mirror and then runs the f16 kernel, which
is why prefill is unaffected (531 against 528 tok/s at 32768), while the decode's
split-KV kernel reads the quantized cache directly and dequantizes inside the
attention loop, where at depth the attention is most of the token. At 131072 there is
no f16 comparison to make: that cache does not allocate at all.

### The second trunk format

`Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf` is a differently mixed trunk: the name says
IQ3_XXS, but the file carries nine quantization types and six of them were ones the
engine did not execute - IQ3_XXS, IQ2_S, IQ2_XS, IQ4_XS, IQ1_M and IQ1_S, 59% of
its tensors. The engine refused the file at load with a message naming the type.
It runs now, and the whole difference is one missing type plus three dispatch lists:
the vendored MMQ and MMVQ kernels already had IQ1_S, and the loader's own type table
had its block size as 110 bytes, the size of IQ3_S, against the real 50 - wrong
arithmetic that would have been read as corruption.

What it buys is 10.44 GiB instead of 11.77, which is what lets the draft head run at
32768 where the release trunk stops at 16384. Measured on this card, same prompt:
58.55 tok/s of decode at 32768 and 996.8 tok/s of prefill, against the release
trunk's 55.75 and 1095.3 at 16384 - so decode improves, prefill gives up about 9%,
and the engine's resident-budget warning is printed even though its numbers are for
the separate NVFP4 sidecar and the measurement does not collapse.

The price is quality, and it is measured rather than assumed: the 16-token sentence
the regression uses reads NLL 1.87606303 where the release trunk reads 1.80954673,
and the CPU reference agrees with the CUDA path to 0.0027 nats on that file. Both
values are pinned by the regression, so neither can drift unnoticed.

The release trunks are unaffected, and that is checked rather than asserted. They
contain no IQ1_S at all, so the only loader table line that changed cannot reach
them; the five types that made the new file look exotic - IQ3_XXS, IQ4_XS, IQ2_S,
IQ2_XS, IQ2_XXS - are in the release trunk too and were already executable, so no
kernel or case label of theirs was touched. The one shared condition that did change
is the eight-row chunking guard, which now covers IQ1_S as well: for IQ1_M, the type
the release trunk uses there (one tensor, blk.13.ffn_gate), the condition is
identical before and after. The pinned NLL 1.80954673 and the whole regression,
measured on the committed tree, are the same values as before the change.

`ds4-bench` is not the tool to compare this with.  It fails to create a session at
131072 with q4_0 even with 13947 MiB free ("failed to allocate Qwen CUDA tensor
hidden", 10 MiB), so it does not allocate the way the server does; and its
`--ctx-max` is divided by the KV ratio for quantized types - 8192 asked measures at
2048 - so its `ctx_tokens` column is the context actually run, not the one
requested.
