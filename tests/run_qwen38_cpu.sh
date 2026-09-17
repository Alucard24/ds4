#!/usr/bin/env bash
# Private-kernel tests, no model required. Optional arguments compare quant rows
# against a local ggml shared library and the actual mixed-IQ model tensors.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $(uname -s) != Linux ]]; then
    echo 'This reference test launcher currently supports Linux only.' >&2
    exit 1
fi
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT
flags=(-O1 -g -D_GNU_SOURCE -DDS4_NO_GPU -std=c99 -ffunction-sections -fdata-sections -I.)
if [[ ${SANITIZE:-0} == 1 ]]; then
    flags+=(-fsanitize=address,undefined -fno-omit-frame-pointer)
fi
"${CC:-cc}" "${flags[@]}" tests/test_qwen38_cpu.c -Wl,--gc-sections \
    -lm -pthread -ldl -o "$build/test_qwen38_cpu"
"$build/test_qwen38_cpu" "$@"
