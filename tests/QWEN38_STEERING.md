# Qwen directional steering validation

Runtime usage and cache policy: [../dir-steering/README.md](../dir-steering/README.md).
IQ3_S remains the release/default model; IQ3_XXS is optional.

## Reproducible gates

```sh
make tests/test_qwen38_steering CUDA_ARCH=sm_120
./tests/test_qwen38_steering "$MODEL"
DS4_KV_Q8=1 ./tests/test_qwen38_steering "$MODEL"
DS4_KV_Q4=1 ./tests/test_qwen38_steering "$MODEL"

# Linux CPU reference executable; does not replace the CUDA CLI binaries.
make tests/test_qwen38_steering_cpu
DS4_TEST_STEERING_CPU=1 ./tests/test_qwen38_steering_cpu "$MODEL"

python3 tests/test_qwen_server_launcher.py
python3 tests/test_qwen38_capture.py "$MODEL" /tmp/qwen-style.json
make tests/test_qwen38_mtp CUDA_ARCH=sm_120
DS4_TEST_STEERING_FILE=/tmp/qwen-style.f32 \
  ./tests/test_qwen38_mtp "$MODEL" "$MTP" \
  'The capital of France is Paris. The largest ocean on Earth is the Pacific Ocean.'

./tests/run_qwen38_regression.sh
```

The C gate tests both attention and FFN liveness, positive/negative scales,
zero-file and zero-scale full-logit bit comparisons, session-local live changes
and invalidation, exact disk continuation, mismatched directions/scales/model
identity/mode, truncation, and malformed direction files. It compares the CUDA
projection at width 5120, 17 rows, against a double-precision CPU oracle.
The test initially failed on live scale activation: Qwen's early engine-open
return bypassed the generic direction loader. Moving the call into the Qwen
branch fixed that failure; mere successful compilation had not caught it.

Optional `DS4_TEST_STEERING_EXPORT=/tmp/cpu.payload` and
`DS4_TEST_STEERING_IMPORT=/tmp/gpu.payload` let the same test save/restore a
steered payload across CPU and CUDA. Use f16 GPU KV for this conversion check.
This checks acceptance and finite continuation, not CPU/GPU bit equality.

## Results on RTX 5070 Ti 16GB / Linux

- Re-run at the `stable-34` state (native Qwen tool path in `ds4-agent`, drafting
  profiles defaulting to 32768): exit 0, 0 failing assertions, 8 minutes wall clock
  including the `make clean` rebuilds - the earlier "about an hour" was my own wrong
  estimate. Same six pins, `recall at ~19.5k: 4/4`, prefill first-token logits f16
  16.5558395 / q8_0 16.5136585 / q4_0 16.2306328, bench prefill 1452 t/s, MTP 48/48
  on both runs, `agent KV cache: saved (quant_bits=0) and reloaded`, four steering
  gates, q4_0 and q8_0 disk cache hits, `server: OK`. One step does not run here:
  `make test-tokenizer-vectors` has no such target in this Makefile, so the script
  prints that it is covered by `run_qwen38_cpu.sh`.

- Full `run_qwen38_regression.sh`: exit 0, including six weight/KV session cells,
  mixed-IQ kernels, prefill/recall, image/video, MTP, frontend smoke tests and
  server protocol/cache checks. All six pre-existing NLL pins unchanged:

  | Weights | f16 | q8_0 | q4_0 |
  |---|---|---|---|
  | IQ3_S | 1.80954673 | 1.80900178 | 1.82242633 |
  | IQ3_XXS | 1.87606303 | 1.87684186 | 1.87755574 |

- New steering C gate passed on IQ3_S CPU and CUDA f16/q8_0/q4_0, and IQ3_XXS
  CUDA f16. CUDA projection max error versus the double oracle: <6e-8 for both
  tested signs. Zero directions and zero scales: full logits bit-identical on
  the gate's prompt and next-token evaluation, not a universal equivalence claim.
- CPU -> CUDA and CUDA -> CPU steered payload imports and finite continuations
  passed. Source element width, not destination arena width, sizes the payload.
- Final-row capture: paired prompts with an identical first chunk but differing
  suffixes produced identical first-chunk dumps and different final-row dumps
  in all 64 layers, for both `ffn_out` and `attn_out`. The builder produced
  1,310,720 bytes, all rows finite and unit-normalized, from the benign pair.
- Steered MTP originally diverged at token 14 with batch verification. Serial
  verification fixed the tested case. The consolidated gate reproduced all
  48 plain-greedy tokens; drafting left final trunk logits bit-identical. Its
  measured steered rates were 40.9 t/s greedy vs 39.4 t/s speculative. This is
  a correctness fallback, not a steered MTP speedup.
- Independent CLI comparison against the pre-change executable: identical
  top-20 logprob JSON over eight greedy tokens without steering.
- Manual server checks (private cache directories, trace enabled): normal ->
  steered rejection, matching steered restart hit (1256 tokens, 51 ms load),
  changed-scale rejection, steered -> normal rejection, and a normalized
  tool-call/result multi-turn Chat request. All requests completed successfully;
  rejects fell back to rebuilding. A shared text-key directory can prevent
  saving a new steered entry over an existing normal one, so the launcher uses
  separate namespaces.
- `make cpu` initially failed linking the server's pre-existing prompt-prefix
  dependency. Adding `ds4_prompt_prefix.o` to both CPU link recipes fixed it.
  All five CPU binaries built, followed by clean CUDA rebuilds. No Mac execution.
  Both builds still emit a snprintf truncation warning in the untouched
  `ds4_agent.c`; this work does not address that warning.

## No-steering performance check

Before binary compiled from commit `67c302f`'s `ds4.c`; same bench source,
CUDA objects, machine, IQ3_S file, prompt and options. Two alternating runs per
state, `--ctx-max 4096 --gen-tokens 64`; means below (not an optimization claim).

| Frontier | Before prefill t/s | After prefill t/s | Before gen t/s | After gen t/s |
|---|---:|---:|---:|---:|
| 2048 | 1380.5 | 1365.4 | 49.03 | 48.83 |
| 4096 | 1429.7 | 1420.6 | 47.65 | 47.69 |

Small measured changes (~-1.1% to +0.1%) do not establish performance at other
depths or under every workload. Active projections add work and serial steered
MTP verification forfeits batched verification throughput.

## Boundaries

No held-out style/refusal efficacy or general capability-preservation claim;
no long-running soak, exhaustive contexts or all API endpoint combinations.
Steered vision quality, XXS quantized steering integration and embedded-MTP
steering are not covered by the consolidated gates above. Metal/ROCm and huge
DeepSeek model suites were not run: unavailable hardware/model capacity here.
Local stat-based steered cache identity is deliberately conservative, not a
portable content hash. Unsteered same-family weight identity remains legacy
behavior; separate cache directories are still advised across different GGUFs.
