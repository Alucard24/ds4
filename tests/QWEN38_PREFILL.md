# Qwen3.8 prefill: the gap was lazy weight loading, not the kernels

Status: **RESOLVED.** The 8x prefill gap was a missing weight preload on the
Qwen engine path, not the MMQ kernels. Fixed; the first prompt runs at ~1030
tok/s instead of ~200 tok/s, and the same prompt through llama.cpp is 1526-1632.

## The measurement error that hid it

The phase report printed the three fine-grained matmul counters (FFN norm, gate,
up) as a running total instead of per chunk, so a warm chunk looked as slow as a
cold one and the residency hypothesis was discarded too early. Fixed: the
counters are now deltas.

With the fix, two different 350-token prompts in one process:

| | FFN gate, 64 layers | chunk |
|---|---|---|
| first prompt (cold) | 364.8 ms | 1207 ms |
| second prompt (resident) | **41 ms** | **422 ms** |

41 ms for 64 gate matmuls is **0.64 ms per matmul = 48 TMAC/s**, the same rate
llama.cpp reaches. 364.8 ms is the same kernel with the weights still in host
memory: each weight range was faulted in on first use at about 23 GB/s
(114 MB per matmul / 5 ms).

## The fix

Every model family preloads its tensor spans at engine open except Qwen3.8,
whose branch returns before the generic preload. It now calls
`accelerator_cache_model_tensors()` like the others. The cost moves from the
first reply into startup (the process still finishes a 350-token prompt, load
included, in 2.59-2.68 s).

## Numbers after the fix

| | ds4 | llama.cpp |
|---|---|---|
| 350-token prefill, first prompt after start | ~1030 tok/s | 1526-1632 tok/s (pp512) |
| decode | 42.5 tok/s, 53.9 with MTP speculation (1.27x) | 54.3 tok/s (tg128) |
| NLL gate | 1.81334038 unchanged | - |

The residual factor of ~1.6 is in the non-GEMM part of the chunk, where the GDN
recurrence and the GA attention now dominate: the recurrence is a sequential
scan over tokens with 48 blocks of 128 threads, which is the algorithmic item
left if more prefill speed is wanted.

## What the investigation ruled out (all by measurement)

- kernel revision: the ported later upstream revision runs at the same speed
  (gate 350.9 vs 343.4 ms per 64 layers), and `mma.cuh` is byte-identical;
- tile width: `DS4_CUDA_MMQ_X_MAX` sweep, 128 default vs 64 is -6%, rest worse;
- occupancy: the kernel uses 254-255 registers in both builds, giving one
  resident block per SM (8 of 64 warps) in llama.cpp as well - verified with
  cuobjdump -res-usage on both binaries;
- `-DNDEBUG`: added to the MMQ objects, registers unchanged at 255;
- cuBLAS for large batches: `ggml_cuda_should_use_mmq()` returns true whenever
  Turing MMA is available, so llama.cpp runs the same MMQ kernels;
- per-call pool allocation, duplicated matmuls: no measurable effect;
- code generation: the SASS differs by 11% instructions, which cannot explain 8x.

## Superseded text

Kept because it documents the porting surface and the dead ends.

Result of running the ported revision against the vendored one, same model, same
chunk (350 tokens), phase timing per matmul, 64 layers each:

| | FFN gate | FFN up |
|---|---|---|
| vendored revision | 343.4 ms | 314.2 ms |
| ported later revision (`DS4_MMQ_NEW=1`) | 350.9 ms | 319.7 ms |

So the later revision is *not* faster here: the ~5.7-6.2 TMAC/s is what this
kernel family does on this machine, and the prefill gap is not a stale kernel.
The wiring was therefore removed from the release path (no measured win), while
the ported tree stays as a compilable, validated second implementation:

```sh
# compile check (no build wiring)
/opt/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a   -Icuda/mmq -Icuda/mmq/new -I. -c -o /tmp/x.o cuda/mmq/ds4_mmq_new_probe.cu
```

Other hypotheses tested and refuted since, all on the same shape:

- **cold weight upload**: chunks 2 and 3 of a 1200-token prompt are *slower*
  than chunk 1 (427 and 450 ms vs 369 ms for the gate matmul), so residency is
  not the limiter and the per-matmul time is steady state;
- **llama.cpp using cuBLAS for large batches**: `ggml_cuda_should_use_mmq()`
  returns true whenever Turing MMA is available, which is the case on sm_120,
  so llama.cpp runs the same MMQ kernels;
- the tile sweep, the D2R path, pool caching and matmul duplication, as
  recorded above.

What is left, and why it needs a profiler: the kernel is neither
compute-bound (2-3 TMAC/s against ~88 TMAC/s of int8 tensor peak) nor
bandwidth-bound (20 GB/s of weight traffic against 460 available), which points
at occupancy or launch configuration of the *adapted* build rather than at the
kernel source. `nsys`/`ncu` are not installed and `nvprof` refuses compute
capability 8.0+, so the next step is either installing one of them or
instrumenting the launch (grid, block, shared bytes, achieved occupancy via
cudaOccupancyMaxActiveBlocksPerMultiprocessor) and comparing against
llama.cpp's own numbers.

Superseded text follows, kept because it documents the porting surface.

Compile probe result (this is the question that gated the port): the current
upstream kernels compile against ds4's existing adapter. `cuda/mmq/new/` holds
the later revision (13 files, symbols prefixed `ds4n_`, self-contained include
graph), and `cuda/mmq/ds4_mmq_new_probe.cu` instantiates the dense IQ3_S case
with `nvcc -gencode arch=compute_120a,code=sm_120a`: 0 errors, 2.5 MB object.
Only two additions were needed outside that directory:

- `cuda/mmq/ggml-common.h`: the `Q2_0` quant type (upstream added it after
  these headers were vendored);
- `cuda/mmq/ds4_ggml_stubs.h`: the matching `GGML_TYPE_Q2_0 = 42` enum value.

What remains, in order:

1. `cuda/mmq/ds4_mmq_new.cu`: an `extern "C"` dense entry mirroring
   `ds4_mmq_quant_dense_impl` but building `ds4n_mmq_args` from plain pointers
   and calling `ds4n_mul_mat_q_case<type>`; check which activation quantizer the
   later revision expects (`quantize_mmq_q8_1_cuda` signature changed) and copy
   its `quantize.cuh`/`quantize.cu` into `new/` if needed.
2. Makefile: add the new object to `MMQ_OBJS`.
3. `ds4_cuda.cu`: route `ds4_gpu_matmul_quant_tensor`'s dense path to the new
   entry behind `DS4_MMQ_NEW=1`, leaving the old path default until proven.
4. Measure the single matmul (5120x17408, M=350, IQ3_S): 5.9 ms today, target
   < 1 ms; then the gates (`test-qwen38-cuda` oracle, NLL 1.81334038, MTP 39/39).
5. Only then extend to the remaining types and make it the default.
 This document is the work list, the evidence
behind it, and the gates that must stay green. It exists so the porting does not
have to repeat the diagnosis.

## Evidence (all measured on RTX 5070 Ti 16 GB, IQ3_S trunk, CUDA sm_120a)

Per 350-token chunk, with `DS4_QWEN38_PHASE_TIMING=1` (in-stream CUDA events, no
pipeline draining; see `ds4_gpu_phase_mark`):

| segment | time |
|---|---|
| post-attention norm, 64 layers | 1.8 ms (pure memory, full bandwidth) |
| FFN gate matmul, 64 layers (5120x17408, M=350) | 345.2 ms |
| FFN up matmul, 64 layers (same shape) | 320.6 ms |
| input projections | 324.0 ms |
| GDN recurrence + GA attention | 244.8 ms |
| output projection + residual | 134.8 ms |
| FFN down + residual + LM head | 439.7 ms |

- One quantized matmul at that shape costs **5.9 ms = 5.7-6.2 TMAC/s**.
- llama.cpp's `pp512` on the same model and card (build `00ed45216`) is
  **1702 tok/s**, which implies ~47 TMAC/s.
- The engine moves 11.76 GiB of weights in ~1.75 s, i.e. ~7 GB/s effective: far
  from the 460 GB/s available, so this is not bandwidth, it is the kernel.
- Whole-prompt numbers: 350 tokens in ~1.75 s (**~200 tok/s**). Note that the
  663-678 tok/s once reported here was a REPL prefix-cache hit, not a prefill.

Negative A/Bs (do not repeat these):

| experiment | result |
|---|---|
| `DS4_CUDA_MMQ_X_MAX` 128 (default) / 64 / 32 / 16 / 8 | 1836 / 1724 / 1828 / 1910 / 2241 ms -> tile width is not the cause |
| `DS4_MMQ_D2R` path | threshold is 1024 columns, not reached at M=350 |
| caching the vendored pool's `cudaMallocAsync`/`cudaFreeAsync` blocks | no change; change reverted |
| removing 2 of 7 matmuls per layer (a duplicated FFN pair that was found and fixed) | within noise -> not a count problem |

## Cause

The vendored kernels are pinned to upstream `5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a`
(2026-05-14, see `cuda/mmq/VENDOR.md`). Upstream has since restructured MMQ from
a runtime tile choice into a **per-architecture config table with compile-time
`J`/`fallback` instantiation**:

| | vendored (ds4) | current upstream |
|---|---|---|
| `mma.cuh` (MMA primitives) | 1456 lines | 1456 lines, **identical** |
| `mmq.cuh` | 4500 lines monolithic | 1598 lines restructured |
| `mmq.cu` | - | 386 lines |
| `mmq-config-*.cuh` | does not exist | 10 files, 2278 lines |
| config struct | - | `{type, nthreads, occupancy, I, J, sram_layout, K_vram, stream_k, fallback}` |

The new knobs are exactly the ones that move a small-M GEMM: `stream_k`
decomposition (removes tail/wave quantization), `K_vram` tile, per-type SRAM
layouts (e.g. `MMQ_SRAM_LAYOUT_Q3_K` for IQ2_S), and an occupancy target. The
MMA primitives being identical means **no math has to be rewritten**: this is
tile/layout/dispatch plumbing.

## Work list

1. Bring from upstream: `mmq.cuh`, `mmq.cu` (dense half: the type switch and the
   `!ids` branch of `ggml_cuda_mul_mat_q`), the `10` `mmq-config-*.cuh` files,
   the new `quantize_mmq_q8_1_cuda` / `quantize_mmq_fp4_cuda` activation
   quantizers, and the ~311-line `common.cuh` diff (helpers:
   `ggml_cuda_pdl_lc`/`pdl_sync`/`syncwarp`, `ggml_cuda_kernel_launch`,
   `ggml_cuda_type_traits`, `ggml_cuda_is_aligned`, the pdl config).
2. Adapter surface to add in `cuda/mmq/ds4_ggml_stubs.{h,cu}` /
   `cuda/mmq/ds4_mmq.{h,cu}`: `ggml_cuda_info()` with per-device
   `{cc, nsm, warp_size}`, `ggml_cuda_mmq_get_J_max`, `QK8_1_MMQ`,
   `MATRIX_ROW_PADDING`, `GGML_PAD`, `blackwell_mma_available` (already present).
3. Rewrite `ds4_mmq_quant_dense_impl` to build `mmq_args` from plain pointers
   (it already constructs the old equivalent) and call the new
   `mul_mat_q_case<type>`; keep the old path behind the existing entry points
   until the new one passes the gates.
4. Instantiate only the types ds4 executes: Q8_0, Q2_K, Q4_K, IQ2_XXS, IQ2_XS,
   IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ1_M, MXFP4. Compile time is the practical
   limit here, not correctness.

## First measurable step (dense-only slice)

Micro-benchmark one matmul, shape 5120x17408, M=350, IQ3_S:

- today: **5.9 ms** per call;
- target: **< 1 ms** (the shape is ~0.7 ms of weight traffic at full bandwidth);
- correctness gate: `make test-qwen38-cuda DS4_TEST_GGML_CPU=.../libggml-cpu.so`,
  which compares every mixed-IQ type against the ggml CPU dequantization, so the
  port must not change a single decoded value.

If that single number moves, extend to the rest of the types and shapes.

## Gates that must stay green after any step

- `tests/test_qwen38_cuda` - mixed-IQ numerical oracle per type vs ggml CPU.
- `MEAN_NLL 1.81334038` - `tests/test_qwen38_session_cuda`.
- MTP 39/39 identical to one-token greedy at ~1.26x - `tests/test_qwen38_mtp`.
- `tests/test_qwen3vl_*`, container gate, `./ds4_test --server`.
- `ds4-bench` prefill/gen before and after, same machine.
- No change to any emitted number: this is a kernel port, not a numerics change.
