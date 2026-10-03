#!/usr/bin/env bash
# Serves Nemotron-Labs-Diffusion with the SGLang DLLM fork (third_party/sglang).
#
#   scripts/launch_sglang.sh dlm          # FastDiffuser (block diffusion)
#   scripts/launch_sglang.sh linear_spec  # diffusion draft + AR verify
#   scripts/launch_sglang.sh ar           # pure autoregressive
#
# One mode per server: SGLang picks the decoding algorithm at launch.
# Env overrides: MODEL, PORT, MEM_FRAC, MAX_REQS, CTX_LEN, ATTN_BACKEND.
# Set up the environment first with scripts/setup_sglang.sh.
set -euo pipefail

mode="${1:?usage: launch_sglang.sh <dlm|linear_spec|ar>}"
root="$(cd "$(dirname "$0")/.." && pwd)"

MODEL="${MODEL:-nvidia/Nemotron-Labs-Diffusion-3B}"
PORT="${PORT:-30000}"
MEM_FRAC="${MEM_FRAC:-0.5}"
MAX_REQS="${MAX_REQS:-4}"
CTX_LEN="${CTX_LEN:-4096}"
ATTN_BACKEND="${ATTN_BACKEND:-flashinfer}"

override=()
case "${mode}" in
  dlm)
    algo=FastDiffuser
    cfg="${root}/configs/sglang/fastdiffuser.yaml"
    ;;
  linear_spec)
    algo=LinearSpec
    cfg="${root}/configs/sglang/linearspec.yaml"
    ;;
  ar)
    algo=FastDiffuser
    cfg="${root}/configs/sglang/ar.yaml"
    override=(--json-model-override-args '{"ar_mode": true}')
    ;;
  *)
    echo "unknown mode: ${mode}" >&2
    exit 2
    ;;
esac

exec "${root}/.venv-sglang/bin/python" -m sglang.launch_server \
  --model-path "${MODEL}" \
  --trust-remote-code \
  --tp-size 1 \
  --mem-fraction-static "${MEM_FRAC}" \
  --max-running-requests "${MAX_REQS}" \
  --attention-backend "${ATTN_BACKEND}" \
  "${override[@]}" \
  --dllm-algorithm "${algo}" \
  --dllm-algorithm-config "${cfg}" \
  --cuda-graph-bs 1 2 3 4 \
  --context-length "${CTX_LEN}" \
  --host 127.0.0.1 \
  --port "${PORT}"
