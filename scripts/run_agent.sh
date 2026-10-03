#!/usr/bin/env bash
# Runs mini-swe-agent against the local Nemotron server in AR or diffusion mode.
#
#   scripts/run_agent.sh ar  "Create hello.py that prints hello"
#   scripts/run_agent.sh dlm "Create hello.py that prints hello"
#
# Start the server first:  uv run python -m src.server --port 8000
set -euo pipefail

mode="${1:?usage: run_agent.sh <ar|dlm> <task>}"
task="${2:?usage: run_agent.sh <ar|dlm> <task>}"

exec uv run mini \
  -c mini_textbased.yaml \
  -c configs/mini_nemotron.yaml \
  -c "model.model_name=openai/nemotron-${mode}" \
  -t "${task}" "${@:3}"
