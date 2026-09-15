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
