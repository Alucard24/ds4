# CUDA Xid incident, 2026-09-15

**Status: open. Do not run GPU workloads of this project on this machine until the
fault is found and fixed.** This is a field report, not a post-mortem: the cause is
narrowed, not identified.

## What happened

Two hard shutdowns of the machine on 2026-09-15, both during GPU runs of this
project. The GPU stopped responding to the driver ("no devices found"), the display
subsystem went with it, and the second boot log records the driver asking for an OS
reboot. The escalation is visible in the kernel log, and every step is attributed to
this project's binary:

```
set 12 18:38:24  Xid 13  Graphics SM Warp Exception on (GPC 0, TPC 0, SM 0): Out Of Range Address
set 12 18:38:24  Xid 13  Graphics SM Global Exception ... Multiple Warp Errors
set 12 18:38:24  Xid 13  Graphics Exception: ESR 0x505730=0xc02000e ...
set 12 18:38:24  Xid 43  pid=1628348, name=ds4, channel 0x00000003      (and 4 more ds4 pids to 18:41)
...
set 15 06:26:15  Xid 79  GPU has fallen off the bus.
set 15 06:26:15  Xid 154 GPU recovery action changed from 0x0 (None) to 0x2 (OS Reboot)
```

- **Xid 13 with `Out Of Range Address` is an application fault**: a kernel read or
  wrote outside its allocation. 847 lines in the 12 September burst, every one of
  them from `name=ds4`. No Xid in the recorded history comes from any other program.
- **Xid 43** is the driver resetting the channel because the GPU stopped processing:
  the faulting process took its context down, five separate `ds4` pids inside three
  minutes.
- **Xid 79 / 154** is the device dropping off the bus and the driver escalating to a
  system reboot. That is what forced the shutdowns.

So the chain that took the machine down starts inside a kernel of this project. The
driver-level assertions seen first (`NVKMS memory for GEM object`, `scratch jobs
timed out`) are downstream of it.

## What is known about where

- The 12 September burst sits between two commits of the quantized-prefill work
  (`afb4681` at 16:24, `b99cc46` at 19:51); no commit exists in that window, so the
  binary running then was an uncommitted state of that work. It is a strong
  candidate for the burst, not an established cause.
- The two events on 15 September came from the **committed** build, on the plain f16
  path: the runs were agent turns on the IQ3_S trunk (f16 KV, no steering file, no
  MTP). So a faulting kernel exists in shipped code, not only in a discarded
  experiment.
- The kernels active in that flow: the FA-2 chunk prefill, the split-KV decode, the
  GDN recurrence/output kernels, the quantized matmul/matvec and the elementwise
  kernels. The widening path for quantized KV was read line by line during this
  investigation and its bounds hold (grid-stride loop, two-half scratch, rows
  bounded by `ctx_size`).

Candidate shapes worth checking first, in order:

1. **Tail fragments at the end of an allocation.** A 16-wide fragment load whose
   remainder is not masked reads past the last row when the chunk fills the cache
   exactly (`start_pos + n_tokens == ctx_size`). Inside the allocation this is
   harmless and invisible; the last row is where it escapes.
2. **A stride written by one format and read by another.** The KV cache is
   per-position block-sized for q8_0/q4_0 (1088/576 bytes per position per layer)
   and 2048 for f16. A reader that assumes f16 bytes over a quantized tensor reads
   about twice the allocation. The payload path already had exactly this bug, fixed
   for the checkpoint case.
3. **Unsigned underflow on a count that can be zero** (a `rows - 1` style offset, a
   `positions = start + n` with `n == 0`).

## How to find it, and the rule

`compute-sanitizer` is the right tool. It was not installed with the Arch `cuda`
12.8.1-3 package, and installing the repository's `cuda` 13.3.1 would have replaced
the toolkit this project builds with. Instead the 13.3.1 package was downloaded
without installing it, only `opt/cuda/bin/compute-sanitizer` and
`opt/cuda/compute-sanitizer/` were extracted, and they live under the owner's home
at `~/opt/cuda-cs/`; the toolkit is still 12.8. Version reported:
`2026.2.1.0`. Reproducing is still a deliberate, supervised act on hardware the
owner chooses, not something an agent launches on its own.

First sanitizer result, same day: the FA-2 fragment self-test
(`tests/qwen38_mma_frag_test.cu`, tiny allocations, the MMA layout the attention
depends on) ran under `--tool memcheck` with **0 errors** and all 128 values correct
for both m16n8k16 and m16n8k8.  The fragment code is clean, so the fault is
elsewhere: the next targets are the other spike
(`tests/qwen38_mma_qk_spike.cu`), then a small synthetic harness that drives the
real prefill and widening kernels through the public GPU entry points with boundary
shapes (`start_pos + n_tokens == ctx_size`, tails of 1..7 positions, both KV
formats), which is what covers launch-time indexing rather than the fragment
layout.  A full-model run under memcheck is unlikely to fit a 16 GiB card: memcheck
keeps shadow state for every tracked allocation next to the 10.95 GiB of weights.

Rules that follow from this incident, and that apply to any agent working in this
repository:

- **No GPU or model runs on the owner's machine without the owner's explicit
  approval for that specific run**, and never a sequence of them.
- **The first error, warning or unexpected device message stops the work.** It is
  reported, not interpreted as noise. The second shutdown of this day happened
  because a `CUDA init set device failed` was read as model noise and the runs
  continued.
- A single run, watched, is the most that is ever proposed without being asked.

## Host-side guards added with this report

So that a wrong size or position becomes an error instead of a fault:

- the quantized prefill widening checks that the widener's row count fits the cache
  tensor it reads, and refuses the layer with a message rather than launching;
- the activation dump refuses `rows == 0` instead of computing a wrapping offset
  from `rows - 1`.

These are guards, not a fix. The fault itself is still open.

## First supervised run after the guards (2026-09-15)

One run, watched, single process, on the f16 prefill path that faulted earlier:
`ds4 -c 4096 --raw --temp 0 --dump-logprobs /tmp/xid-check.json --logprobs-top-k 1`,
17 input tokens, one chunk.

- first-token logit **16.5558395**, the value the regression pins for this prompt:
  the spare row added to the KV stride did not change the addressing;
- **0 Xid** in the kernel log afterwards, GPU alive at 44 C.

What this establishes: this shape is clean and the padding is functionally neutral.
What it does not establish: that the fault is gone.  One run, one shape.  The two
events of 15 September came from agent runs (many short turns, a KV payload save per
turn, possible rewind and re-prefill) and the 12 September burst came from the
quantized-prefill work; neither shape was exercised here.  The offending kernel is
still unidentified, and the guards of Phase A are still the agreed route to find it
without risking the card again.

## Boundary harness result (same day)

`tests/qwen38_kv_bounds.cu` drives the real prefill and decode entry points with
tiny tensors (ctx 256 and 512) and boundary shapes: tails of 1, 4, 7, 15, 17, 507
and 512 positions, `start_pos + n_tokens` landing exactly on the last cache
position, and both KV formats (f16 and q8_0, so the widening path runs too), under
`--tool memcheck` with the layer's real norm-weight offsets.

Result: every shape ran (`prepare`, `chunk`, `decode`, `sync` all succeeded) and
**no invalid access was reported**.  The only two entries in the sanitizer log are
`cudaErrorNotSupported` from `cudaHostRegister` - the host registration this device
does not support, which the engine already prints and skips - not memory faults.

So candidate 1 (launch-time indexing of the prefill path) and the fragment layout
(candidate 1's inner form, clean in its own test) are both ruled out for these
shapes. The same drive was later extended to **q4_0** (11 shapes x 3 formats = 33 launches, widening included): every shape ran, memcheck reported nothing but the two known API notices, verified in the log text.  What the harness does not cover, and what the next pass has to: the GDN
half of the layers (`ds4_gpu_qwen38_gdn_chunk` / `gdn_decode`), the MTP draft path,
and the agent-shaped flow (many short turns, a KV payload save per turn, rewind and
re-prefill) that produced both 15 September events.

## GDN half driven too (same day)

The harness now also drives `ds4_gpu_qwen38_gdn_chunk` and `_gdn_decode` with token
counts 1, 7, 16, 17 and 512 and the real weight offsets of block 0 (conv1d, a, dt,
norm).  Every shape ran and memcheck reported **no invalid access**; the only log
entries remain the `cudaHostRegister` API notices.

Covered so far, all clean: the attention/cache path at 22 boundary shapes in both KV
formats, the fragment layout in its own test, and now the GDN recurrence and output.
Not covered yet: the MTP draft path (it reaches the same GA entry points with its
own tensors, so it is partly exercised already) and the **agent-shaped flow** - many
short turns, a KV payload save per turn, rewind and re-prefill - which is where both
15 September reports came from.  The next single attempt is that flow under
memcheck: if the sanitizer's shadow memory does not fit next to the 10.95 GiB of
weights it fails cleanly with an allocation error, which is itself the result.

## The agent-shaped state flow under memcheck (same day)

`tests/test_qwen38_session_cuda` under `--tool memcheck`: prefill, a rewrite to a
different first token and back, a shorten-and-extend replay, a payload save, a
continue, a restore and a truncated-payload rejection - the state operations an
agent turn performs, minus the tools.

- the test passed with the pinned trunk NLL **1.80954673** and
  `Qwen CUDA session PASS`, so the flow is functionally intact under the sanitizer;
- the sanitizer reported **4 errors, all four `cudaErrorNotSupported`** from
  `cudaHostRegister` / `cudaGetLastError` - the API notices this device already
  produces - and **no invalid access, no kernel named**.  Shadow memory fit next to
  the weights, so the flow really did execute under instrumentation.

Eliminated so far, all measured, not assumed: the fragment layout, the launch-time
shapes of the attention and widening path in both KV formats, the GDN recurrence and
output, and now the payload/rewrite/replay state path.  Still unexplained: the two
Xid 13 reports themselves.  Kernel surface not yet driven: the quantized matmul
family (partly covered by `tests/test_qwen38_cuda`, which compares against ggml and
can be run under the sanitizer the same way) and the elementwise norm/rope/SwiGLU
family.

## Quantized matmul family under memcheck (same day)

`tests/test_qwen38_cuda` (q2_K, q4_K, iq2_xxs, iq2_xs, iq3_xxs, iq3_s, iq2_s, iq4_xs
at batch 16, each compared against a ggml dot) under `--tool memcheck`: every case
PASS with max_abs between 0.012 and 0.43, and **ERROR SUMMARY: 0 errors** - not even
the cudaHostRegister notices appear in this one.

Measured and eliminated, in order: the fragment layout, the attention and widening
launch shapes in both KV formats, the GDN recurrence and output, the payload /
rewrite / replay state path, and now the quantized matmul family.  The two Xid 13
reports remain unexplained, and the kernel surface not yet driven is down to the
elementwise family (rms_norm, rope, SwiGLU, add), the f16/BF16 projection paths, the
vision tower and the MTP draft block.  The elementwise harness is the next single
attempt; if it is clean too, the honest conclusion shifts from "an index is wrong in
a kernel" to "a state none of these harnesses reproduces", which changes what should
be probed next rather than closing the incident.

## Elementwise family driven too (same day)

The harness now also drives `ds4_gpu_add_tensor`, `ds4_gpu_swiglu_tensor` and
`ds4_gpu_rms_norm_weight_rows_tensor` at 5120-wide rows of 1, 7, 16, 17 and 512
under memcheck.  Every shape ran, no invalid access, and the two log entries are the
same `cudaHostRegister` / `cudaGetLastError` API notices seen in the other runs.

Seven kernel families measured clean: fragments, the QK spike, attention and
widening at boundary shapes in both KV formats, GDN recurrence and output, the
payload/rewrite/replay state path, the quantized matmul family, and now the
elementwise family.  What is left un-driven is narrow: the f16/BF16 projection
paths, the vision tower and the MTP draft block.  If those are clean as well, the
honest reading is no longer "an index is wrong somewhere" but "a state or an
ordering none of these harnesses reproduces", and the next step becomes a
reproduction strategy rather than another sweep.

## The agent flow itself, under memcheck (same day)

The shape that produced both 15 September reports, executed with instrumentation: one
`ds4-agent` turn at `-c 2048`, the model loaded, the KV payload saved for the session,
a tool call issued and executed (`report.txt` read, its contents returned correctly).

Sanitizer result: **4 errors, and all four are the `cudaHostRegister` /
`cudaGetLastError` API notices** this device produces - no `Invalid`, no
`out-of-bounds`, no kernel named.  The memory shadow fit next to the weights, so the
flow really executed under instrumentation rather than failing cleanly.

That is the eighth surface driven clean, and the only one that reproduces the shape of
the two crashes.  The honest reading now: no straightforward out-of-range index has
been found in any kernel family that has been driven, including the flow that failed.
What remains possible, and has to be probed differently rather than swept again: a
state or ordering the harnesses do not reproduce (a long context crossing chunk
boundaries, a payload written and restored across a format or configuration change, a
sequence of many turns), the f16/BF16 projection paths, the vision tower, the MTP
draft block, or a fault whose trigger is not in this project at all - which the Xid 79
reports would allow but the Xid 13 attribution to `ds4` argues against.

## Deep multi-chunk prefill via one agent turn, under memcheck (same day)

The two sessions that faulted both accumulated several turns before the Xid;
no harness so far had driven a deep context in a single process. This run did:
one `ds4-agent` turn at `-c 4096` with a ~1400-token prompt (three prefill chunks),
a native Qwen tool call executed, KV payload saved, answer returned correctly.

Sanitizer: **4 errors, all four the `cudaHostRegister` / `cudaGetLastError` API
notices** - no `Invalid`, no `out-of-bounds`, no kernel named. The turn is
functionally intact (VERDE was read from the tool result and printed back).

## Many tool calls in one turn, under memcheck (same day)

The missing shape: several tool calls in a single agent turn (read, two writes,
two bash runs, reply), each appending KV to the live session. It ran to
completion and produced every artifact. Sanitizer: **4 errors, all four the
`cudaHostRegister` / `cudaGetLastError` API notices** - no `Invalid`, no
`out-of-bounds`, no kernel named.

## Remaining surfaces driven: MTP and vision (same day)

- `tests/test_qwen38_mtp` (NVFP4 sidecar draft + trunk, 48 tokens, trunk path
  bit-identical): **PASS**, sanitizer reported 10 errors, all ten the known
  `cudaHostRegister` / `cudaGetLastError` API notices, verified in the log text.
- `tests/test_qwen3vl_vision` (BF16 tower + mmproj, embedded fixture):
  **PASS** (`temporal frame merge`), **ERROR SUMMARY: 0 errors**.

Nothing this project can execute has produced an invalid access under
compute-sanitizer: fragments, spikes, attention and widening in three KV formats,
GDN, elementwise, the quantized matmul family, the state flow, the agent flow, deep
prefill, the MTP draft path and the vision tower. The two Xid 13 reports of
15 September are therefore not reproducible by any harness built so far, and the
19xx-line incident stands as follows: either a state or an ordering no harness
reproduces (long context crossing chunk boundaries in a dirty cache, a payload
written and restored across a configuration change, a driver already degrading), or
the September 12 burst and the two later events have different causes, the first
belonging to code that was committed, measured wrong and then replaced.

## BF16 projections driven standalone (same day)

The only dense path never driven on its own: the model file holds **zero F16
tensors**, so `ds4_gpu_matmul_f16_tensor` is dead code for this model, and 96 BF16
tensors (`ssm_alpha`/`ssm_beta`, 5120x48). A minimal probe called the BF16 matmul
on the real `blk.0.ssm_alpha.weight` with 1, 7, 16, 17 and 64 rows under memcheck:
all ran, and the two log entries are again the `cudaHostRegister` /
`cudaGetLastError` notices. (The same kernel also runs inside every model execution
already driven clean; this closes the standalone gap, including row tails.)
