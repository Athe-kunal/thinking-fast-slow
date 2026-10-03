
## Nemotron-Labs-Diffusion: AR + diffusion inference

One model load, two decoding modes (`ar`, `dlm`, `mix`).

### Setup

The NVIDIA driver here is 550 (CUDA 12.4), so torch is pinned to the cu124
build (`torch 2.6`) via `[tool.uv.sources]` in `pyproject.toml`. The model
needs `transformers>=5`.

```bash
uv sync
```

### Direct inference

```bash
uv run python -m src.main                                  # chat; :ar :dlm :mix :compare
uv run python -m src.main --prompt "Hello" --compare       # both modes
```

### OpenAI-compatible server + mini-swe-agent

The mode is picked per request through the model name
(`nemotron-ar`, `nemotron-dlm`, `nemotron-mix`).

```bash
uv run python -m src.server --port 8011
scripts/run_agent.sh ar  "Create hello.py that prints hello"
scripts/run_agent.sh dlm "Create hello.py that prints hello"
```

The agent uses text-based actions (`configs/mini_nemotron.yaml`) because the
server has no native tool calling.

### vLLM

Not included: `vllm` 0.30 requires `torch 2.13`, which needs a newer driver
than 550, and it has no support for this model's block-diffusion decoding (the
reference repo serves it with SGLang instead).

### Lint / format / types

```bash
uv run pyink src && uv run ruff check src && uv run ty check src
```

### Harbor-format tasks

```bash
uv run python -m src.harbor_runner tasks/hello-world tasks/sum-numbers --mode ar dlm
```

Each task is run with mini-swe-agent (on the host) against the server. Then the
task's `tests/test.sh` runs in the same container and its reward is read.
