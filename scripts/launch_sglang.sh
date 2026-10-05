#!/usr/bin/env bash
# Serves Nemotron-Labs-Diffusion with the SGLang DLLM fork (third_party/sglang).
#
#   scripts/launch_sglang.sh dlm          # FastDiffuser (block diffusion)
#   scripts/launch_sglang.sh linear_spec  # diffusion draft + AR verify
#   scripts/launch_sglang.sh ar           # pure autoregressive
#   scripts/launch_sglang.sh routed       # learned AR/diffusion router
#   MODEL=<merged export> scripts/launch_sglang.sh span   # model-emitted <diff> spans
#
# One mode per server: SGLang picks the decoding algorithm at launch.
# Env overrides: MODEL, PORT, MEM_FRAC, MAX_REQS, CTX_LEN, ATTN_BACKEND.
# Routed mode: POLICY_MODE (sample | greedy | fixed_ar | fixed_dlm |
# random:<p>), AR_CHUNK, FORCED_AR_TOKENS, BLOCK_SIZE, THRESHOLD, TRACE_DIR,
# ROUTER_CKPT.
# It runs eagerly (no CUDA graphs) because the algorithm reads each forward's
# hidden states.
# Set up the environment first with scripts/setup_sglang.sh.
set -euo pipefail

mode="${1:?usage: launch_sglang.sh <dlm|linear_spec|ar|routed|span>}"
root="$(cd "$(dirname "$0")/.." && pwd)"

MODEL="${MODEL:-nvidia/Nemotron-Labs-Diffusion-3B}"
PORT="${PORT:-30000}"
MEM_FRAC="${MEM_FRAC:-0.5}"
MAX_REQS="${MAX_REQS:-4}"
CTX_LEN="${CTX_LEN:-4096}"
ATTN_BACKEND="${ATTN_BACKEND:-flashinfer}"

override=()
graph_args=(--cuda-graph-bs 1 2 3 4)
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
  routed)
    algo=RoutedDecoding
    cfg="$(mktemp --suffix=.yaml)"
    {
      echo "algorithm: RoutedDecoding"
      echo "causal_context: true"
      echo "first_done_first_out_mode: false"
      echo "threshold: ${THRESHOLD:-0.9}"
      echo "ar_chunk: ${AR_CHUNK:-8}"
      echo "forced_ar_tokens: ${FORCED_AR_TOKENS:-0}"
      echo "block_size: ${BLOCK_SIZE:-32}"
      echo "policy_mode: \"${POLICY_MODE:-sample}\""
      if [[ -n "${TRACE_DIR:-}" ]]; then echo "trace_dir: \"${TRACE_DIR}\""; fi
      if [[ -n "${ROUTER_CKPT:-}" ]]; then
        echo "router_checkpoint: \"${ROUTER_CKPT}\""
      fi
    } > "${cfg}"
    graph_args=(--disable-cuda-graph)
    ;;
  span)
    # Model-switched decoding: AR until <diff>, block diffusion until </diff>.
    # MODEL must be a merged span-SFT export (scripts/export_merged.py).
    algo=SpanDecoding
    cfg="$(mktemp --suffix=.yaml)"
    {
      echo "algorithm: SpanDecoding"
      echo "causal_context: true"
      echo "first_done_first_out_mode: false"
      echo "threshold: ${THRESHOLD:-0.9}"
      echo "block_size: ${BLOCK_SIZE:-8}"
      if [[ -n "${TRACE_DIR:-}" ]]; then echo "trace_dir: \"${TRACE_DIR}\""; fi
    } > "${cfg}"
    graph_args=(--disable-cuda-graph)
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
  "${graph_args[@]}" ${EXTRA_ARGS:-} \
  --context-length "${CTX_LEN}" \
  --host 127.0.0.1 \
  --port "${PORT}"
