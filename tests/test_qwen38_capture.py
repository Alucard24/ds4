#!/usr/bin/env python3
"""Live CUDA regression: steering capture must use the final prompt chunk.

Uses paired benign prompts with an identical long prefix. The first chunk is
identical, but the final rows must differ. Tests both advertised components and
builds a direction artifact for the MTP gate. No external Python dependencies.
"""
import argparse
import array
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_direction", ROOT / "dir-steering/tools/build_direction.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("out", type=Path, help="output metadata JSON; f32 written beside it")
    args = ap.parse_args()
    # Force several chunks on a small prompt to catch first-row/first-chunk bugs.
    os.environ["DS4_QWEN38_PREFILL_CHUNK"] = "16"
    model = args.model.resolve()
    common = "We are preparing an introduction to basic computer networking. " * 5
    prompts = [common + "Explain DNS in one short sentence.",
               common + "Explain DNS in a detailed essay with examples."]
    with tempfile.TemporaryDirectory(prefix="qwen-capture-test-") as td:
        root = Path(td)
        for component in ("ffn_out", "attn_out"):
            rows = []
            paths = []
            for i, prompt in enumerate(prompts):
                work = root / f"{component}-{i}"
                work.mkdir()
                rows.append(builder.run_capture(ROOT / "ds4", model, prompt,
                            "You are a helpful assistant.", False, 512,
                            component, 64, 5120, work, qwen=True))
                dumps = list(work.glob(f"dump_{component}-0_pos*.bin"))
                assert len(dumps) > 1, "test did not exercise multiple chunks"
                last = max(dumps, key=lambda p: int(p.stem.rsplit("_pos", 1)[1]))
                expected = array.array("f")
                expected.frombytes(last.read_bytes())
                assert rows[-1][0] == list(expected), "capture did not select final chunk"
                paths.append(work)
            first0 = (paths[0] / f"dump_{component}-0_pos0.bin").read_bytes()
            first1 = (paths[1] / f"dump_{component}-0_pos0.bin").read_bytes()
            assert first0 == first1, "common-prefix first chunks differ"
            changed = sum(a != b for a, b in zip(rows[0], rows[1]))
            assert changed == 64, f"prompt suffix changed only {changed}/64 layers"
            print(f"{component}: identical first chunk; final prompt rows differ in 64/64 layers")
        good = root / "good.txt"
        bad = root / "bad.txt"
        good.write_text(prompts[0] + "\n")
        bad.write_text(prompts[1] + "\n")
        subprocess.run(["python3", str(ROOT / "dir-steering/tools/build_direction.py"),
                        "--ds4", str(ROOT / "ds4"), "--model", str(model),
                        "--profile", "qwen3.8-27b", "--good-file", str(good),
                        "--bad-file", str(bad), "--ctx", "512", "--out", str(args.out)],
                       check=True)
    data = array.array("f")
    data.frombytes(args.out.with_suffix(".f32").read_bytes())
    assert len(data) == 64 * 5120
    for layer in range(64):
        row = data[layer * 5120:(layer + 1) * 5120]
        assert all(math.isfinite(x) for x in row)
        assert abs(sum(x*x for x in row) - 1) < 1e-6
    print("Qwen capture/build PASS")


if __name__ == "__main__":
    main()
