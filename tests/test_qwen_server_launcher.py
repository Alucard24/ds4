#!/usr/bin/env python3
"""Model-free launcher argv/profile/cache-namespace regression."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="qwen-launcher-") as td:
    root = Path(td)
    shutil.copyfile(ROOT / "run-qwen-server.sh", root / "run-qwen-server.sh")
    server = root / "ds4-server"
    server.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
    server.chmod(0o700)
    model_dir = root / "models with spaces"
    model_dir.mkdir()
    for name in ("IQ3_S", "IQ3_S-mtp", "IQ3_XXS-mtp"):
        (model_dir / f"Qwen3.8-27B-GSQ-RCO-{name}.gguf").touch()
    env = {k: v for k, v in os.environ.items() if not k.startswith("DS4_")}
    env.update(DS4_QWEN_MODEL_DIR=str(model_dir), DS4_KV_DIR=str(root / "cache"))

    def run(profile=None, **extra):
        cmd = ["sh", str(root / "run-qwen-server.sh")]
        if profile:
            cmd.append(profile)
        p = subprocess.run(cmd, env=env | extra, capture_output=True, text=True)
        assert p.returncode == 0, p.stderr
        return json.loads(p.stdout)

    def value(argv, flag):
        return argv[argv.index(flag) + 1]

    for profile, ctx in ((None, 16384), ("sidecar", 16384), ("long", 32768),
                         ("q4", 131072), ("xxs", 49152)):
        argv = run(profile)
        assert value(argv, "--ctx") == str(ctx)
        assert ("IQ3_XXS" in value(argv, "-m")) == (profile == "xxs")
        assert value(argv, "-ctk") == ("q4_0" if profile == "q4" else "f16")
        assert "--dir-steering-file" not in argv
        assert value(argv, "--kv-disk-dir") == env["DS4_KV_DIR"]

    prefix = str(root / "prefix [one] *.txt")
    argv = run(DS4_PREFIX=prefix, DS4_SYSTEM="one two * three", DS4_POWER="0")
    assert value(argv, "--prefix-file") == prefix
    assert value(argv, "--system") == "one two * three"
    assert value(argv, "--power") == "0"
    assert value(run("q4", DS4_CTK="f16"), "-ctk") == "f16"
    assert value(run(DS4_CTK="q8_0"), "-ctv") == "q8_0"
    direction = root / "direction [one] *.f32"
    direction.write_bytes(b"test")
    argv = run(DS4_STEERING_FILE=str(direction))
    assert value(argv, "--dir-steering-file") == str(direction)
    assert value(argv, "--dir-steering-ffn") == "1"
    kv = value(argv, "--kv-disk-dir")
    assert kv.startswith(env["DS4_KV_DIR"] + "/steered/")
    assert value(run(DS4_STEERING_FILE=str(direction), DS4_STEERING_FFN="0.5"), "--kv-disk-dir") != kv
    direction.write_bytes(b"other direction")
    assert value(run(DS4_STEERING_FILE=str(direction)), "--kv-disk-dir") != kv
    for bad in ({"DS4_CTK": "q8_0", "DS4_CTV": "f16"}, {"DS4_STEERING_FFN": "1"}):
        p = subprocess.run(["sh", str(root / "run-qwen-server.sh")],
                           env=env | bad, capture_output=True)
        assert p.returncode != 0
print("Qwen server launcher PASS")
