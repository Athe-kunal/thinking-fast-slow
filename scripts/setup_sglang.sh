#!/usr/bin/env bash
# Builds the SGLang environment for Nemotron-Labs-Diffusion.
#
# Uses NVIDIA's DLLM fork (hutm/sglang), pinned to the commit that adds
# LinearSpec + LoRA + AR (branch upstream/2-dllm-linearspec). The upstream PR
# (sgl-project/sglang#25803) is not merged. Installs into its own venv so the
# main project's torch is untouched.
#
#   scripts/setup_sglang.sh
#
# Env overrides: SGLANG_VENV (default .venv-sglang), SGLANG_SRC
# (default third_party/sglang).
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
venv="${SGLANG_VENV:-${root}/.venv-sglang}"
src="${SGLANG_SRC:-${root}/third_party/sglang}"
commit=23472a0ebd85
kernel_url="https://github.com/sgl-project/whl/releases/download/v0.4.5/sglang_kernel-0.4.5+cu129-cp310-abi3-manylinux2014_x86_64.whl"

if [[ ! -d "${src}/.git" ]]; then
  git clone https://github.com/hutm/sglang "${src}"
fi
git -C "${src}" fetch -q origin "${commit}" 2>/dev/null || true
git -C "${src}" checkout -q "${commit}"

uv venv -q -p 3.12 "${venv}"
VIRTUAL_ENV="${venv}" uv pip install -q -e "${src}/python" \
  --override "${root}/configs/sglang/overrides.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match
# The PyPI sglang-kernel wheel links CUDA 13; use the CUDA 12.9 build.
VIRTUAL_ENV="${venv}" uv pip install -q --reinstall --no-deps \
  "sglang-kernel @ ${kernel_url}"

"${venv}/bin/python" - <<'PY'
import torch, sgl_kernel
from sglang.srt.dllm.algorithm import algo_name_to_cls
print("torch", torch.__version__, "cuda ok:", torch.cuda.is_available())
print("dllm algorithms:", sorted(algo_name_to_cls))
PY
