# Directional Steering

Directional steering is a runtime activation edit for DS4. A steering file is a
flat `f32` matrix with one normalized hidden-width direction per normal
transformer layer. During inference, ds4 can apply the edit after attention
outputs, FFN outputs, or both:

```text
y = y - scale * direction[layer] * dot(direction[layer], y)
```

Positive scale removes the represented direction. Negative scale amplifies it.
With no steering file or zero scales, ds4 follows the normal inference path.

The file shape depends on the model:

- DeepSeek V4 Flash: `43 x 4096`.
- GLM 5.3 Flash: `45 x 4096`. The separate MTP predictor layer is omitted.
- Qwen3.8 Flash Next: `48 x 2560`. FFN steering is applied to each
  hyper-connection branch of the residual; dumps average those branches
  at the last prompt token.
- Qwen3.8-27B: `64 x 5120` (1,310,720 bytes). CPU reference and CUDA runtime;
  the separate/embedded MTP block 64 is omitted.

GLM 5.2 steering is not implemented.

## Runtime Options

```text
--dir-steering-file FILE   load one f32 direction per normal model layer
--dir-steering-ffn F       apply steering after FFN outputs; default is 1 when a file is provided
--dir-steering-attn F      apply steering after attention outputs; default is 0
```

The FFN output is usually the best first target because it is late enough in
each layer to represent behavior, style, and topic signals. Attention steering
is available for experiments, but it can be more fragile.

## Qwen3.8-27B (CUDA)

IQ3_S remains the default. Steering is optional, does not edit the GGUF, and is
also executable on IQ3_XXS. Build a direction using the exact target quantization
and evaluate its effect rather than assuming a direction transfers unchanged.

```sh
MODEL=/path/to/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
python3 dir-steering/tools/build_direction.py \
  --profile qwen3.8-27b --ds4 ./ds4 --model "$MODEL" \
  --good-file dir-steering/examples/succinct.txt \
  --bad-file dir-steering/examples/verbose.txt \
  --component ffn_out --ctx 512 --out dir-steering/out/qwen-style.json

./ds4 -m "$MODEL" --ctx 2048 --nothink --temp 0 -n 160 \
  --dir-steering-file dir-steering/out/qwen-style.f32 \
  --dir-steering-ffn -0.5 -p "Explain why databases use indexes."

# Server: same IQ3_S default profile, with an isolated steering cache namespace.
DS4_STEERING_FILE="$PWD/dir-steering/out/qwen-style.f32" \
DS4_STEERING_FFN=-0.5 ./run-qwen-server.sh
# Normal mode: leave the DS4_STEERING_* variables unset.
```

Both `ffn_out` and `attn_out` capture/edit the sublayer output **before** residual
addition, across decode and chunk prefill. Attention includes GDN output
projection as well as full-attention output projection. The tool captures the
**last prompt row of the final chunk**. Qwen dump filenames use the chunk-start
position; each file holds its last row. The initial `stable-29` dump captured
only row zero of the first chunk: its file sizes/norms did not establish a valid
prompt-conditioned direction. Rebuild directions made with that version.

Directions must have the exact shape and finite float32 values; scales must be
finite and within [-100, 100]. Use normalized rows (the builder normalizes them).
Zero rows are allowed for no-op diagnostics. A file with both scales zero is
still validated/loaded, but no projection is launched. The existing live FFN
setter is session-local for Qwen and invalidates KV/GDN and MTP history on a
scale change: callers must sync the full prompt again before decoding.

**MTP correctness:** the draft head is unedited; the trunk is steered. A style
fixture exposed a batch-verify/ordinary-decode greedy divergence. Active steering
therefore verifies drafts on the ordinary token path. This preserved the tested
greedy stream, but **does not retain the unsteered batched MTP speedup**. No-file
and zero-scale timelines retain the existing MTP path. Sampled Qwen decoding
continues to use its existing one-token verification fallback.

**Cache compatibility:** unsteered/zero-scale payloads retain their previous
layout. Steered payloads carry a versioned extension with the exact direction
bytes, both scales, and a conservative local GGUF identity (device, inode, size,
mtime and ctime). Mismatches are rejected before live state is changed; copied
or touched GGUFs require rebuilding steered caches. This is local compatibility
checking, not a cryptographic content identity for portable model files. Legacy
unsteered caches still lack a same-family weight-file identity; use different
cache directories for different GGUFs when bypassing the launcher.

`run-qwen-server.sh` places steered caches beneath
`$DS4_KV_DIR/steered/<hash>` (default root `~/.ds4/server-kv`), separated by
model path, direction content, scales and KV format. A shared directory passed
directly to the server remains safe against steering mismatch, but an existing
incompatible text-key entry can prevent useful new saves: use separate dirs.
Trace/cache text remains plaintext; steering does not change privacy behavior.

These checks establish runtime wiring and bounded regression behavior, not
preservation of every capability or a guaranteed style/refusal effect. Evaluate
held-out prompts and quality before relying on a direction.

## GLM 5.3 Example

Build a GLM 5.3 direction from paired target and control prompt lists:

```sh
python3 dir-steering/tools/build_direction.py \
  --profile glm-5.3-flash \
  --ds4 ./ds4 \
  --model gguf/GLM-5.3-Flash-Q2.gguf \
  --good-file /path/to/target-prompts.txt \
  --bad-file /path/to/control-prompts.txt \
  --out dir-steering/out/glm53-direction.json \
  --component ffn_out \
  --ctx 512
```

Generated `.f32` vectors are local artifacts and are not stored in the
repository. GLM 5.3 steering works with `--mtp`, `ds4-server`, native session
batching, and two-Mac tensor parallelism. For tensor parallelism, pass the same
steering file and scales to both the worker and coordinator.

## Verbosity Example

The bundled example builds a style direction from 100 paired prompts. Each pair
asks for the same information in two ways:

- `examples/succinct.txt`: terse target prompts.
- `examples/verbose.txt`: detailed contrast prompts.

Because the extracted direction is `succinct - verbose`, negative FFN scales
make answers shorter, while positive FFN scales tend to make answers longer and
more explanatory.

Build the vector:

```sh
python3 dir-steering/tools/build_direction.py \
  --profile deepseek-v4-flash \
  --ds4 ./ds4 \
  --model ds4flash.gguf \
  --good-file dir-steering/examples/succinct.txt \
  --bad-file dir-steering/examples/verbose.txt \
  --out dir-steering/out/verbosity.json \
  --component ffn_out \
  --ctx 512
```

This writes:

```text
dir-steering/out/verbosity.json
dir-steering/out/verbosity.f32
```

Try a terse run:

```sh
./ds4 -m ds4flash.gguf --nothink --temp 0 -n 160 \
  --dir-steering-file dir-steering/out/verbosity.f32 \
  --dir-steering-ffn -1 \
  -p "Explain why databases use indexes."
```

Try a verbose run:

```sh
./ds4 -m ds4flash.gguf --nothink --temp 0 -n 220 \
  --dir-steering-file dir-steering/out/verbosity.f32 \
  --dir-steering-ffn 2 \
  -p "Explain why databases use indexes."
```

The same vector can be used in either direction. The sign is the important part:

- negative scale amplifies the succinct target direction;
- positive scale suppresses that direction and usually gives the model more room
  to elaborate.

## Evaluating Scales

Use the sweep helper to test several strengths on a fixed prompt set:

```sh
python3 dir-steering/tools/run_sweep.py \
  --ds4 ./ds4 \
  --model ds4flash.gguf \
  --direction dir-steering/out/verbosity.f32 \
  --prompts dir-steering/examples/eval_prompts.txt \
  --scales "-1,-0.5,0,0.5,1,2" \
  --tokens 180 \
  --nothink
```

Start with FFN scales between `-1` and `2`. If the model becomes repetitive,
ignores the prompt, or starts losing factual content, the scale is too strong.
For this example, `-1` is a good first terse setting and `2` is a good first
verbose setting. Strong negative scales such as `-2` or `-3` can over-amplify
the terse direction and collapse into repetition on some prompts.

## Observed Effect

With the 100-pair vector built from the commands above, local greedy checks
showed the expected behavior:

- Prompt: `Explain why databases use indexes.`
- `--dir-steering-ffn -1`: 67 words, one compact paragraph.
- `--dir-steering-ffn 0`: 136 words, structured explanation.
- `--dir-steering-ffn 1`: 140 words, structured explanation with more detail.

On a prompt that the unsteered model already answered briefly, positive steering
made the expansion more visible:

- Prompt: `What does DNS do?`
- `--dir-steering-ffn 0`: 44 words.
- `--dir-steering-ffn 2`: 171 words, with sections and step-by-step detail.

## Building Other Directions

The extractor compares two prompt sets:

- `good-file`: target prompts for the direction you want to represent.
- `bad-file`: contrast prompts that should be separated from the target.

It captures DS4 activations from the same local GPU graph used for inference,
averages target minus contrast, normalizes one vector per layer, and writes both
metadata JSON and the runtime `.f32` file.

Concept removal:

1. Put concept-heavy prompts in `good-file`.
2. Put neutral prompts in `bad-file`.
3. Run with a positive FFN scale.

Concept amplification:

1. Put desired concept prompts in `good-file`.
2. Put neutral prompts in `bad-file`.
3. Run with a negative FFN scale.

Style control:

1. Put prompts for the target style in `good-file`.
2. Put contrasting style prompts in `bad-file`.
3. Use negative scale to amplify the target style, positive scale to reduce it.

The method is not a fine-tune. It is a low-rank runtime edit, so it works best
for coarse behavior, topic, or style directions that are consistently present in
the activation captures.

## Qwen3.8 Flash Next

Capture uses `--think` / `--nothink`
(not `--think-high`). Dumps track the prompt phase explicitly, including
one-token tails, and retain the last prompt token during ordinary and MTP
decode. `attn_out` captures the output projection of both GDN and full-attention
layers, giving one row for each of the 48 trunk layers:

```sh
python3 dir-steering/tools/build_direction.py \
  --profile qwen3.8-flash-next \
  --ds4 ./ds4 \
  --model gguf/Qwen3.8-Flash-Next-Q4.gguf \
  --good-file /path/to/target-prompts.txt \
  --bad-file /path/to/control-prompts.txt \
  --out dir-steering/out/qwen38-direction.json \
  --component ffn_out \
  --ctx 512
```

Qwen steering is Metal-only. `--mtp-model`, SSD streaming, and `--power`
remain unsupported for this graph. The bank contains only the 48 trunk layers;
the embedded MTP predictor remains unsteered. Its drafts are verified by the
steered target trunk, so `--mtp` remains supported.
