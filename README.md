
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

## SGLang serving (NVIDIA DLLM fork)

NVIDIA's fast inference path is an SGLang fork (`hutm/sglang`); the upstream
PR, sgl-project/sglang#25803, is not merged. `scripts/setup_sglang.sh` clones
it at a pinned commit into `third_party/sglang` and builds a separate
`.venv-sglang`, swapping the fork's CUDA 13 pins for CUDA 12.x builds that run
on driver 550 (`configs/sglang/overrides.txt`).

```bash
scripts/setup_sglang.sh
scripts/launch_sglang.sh linear_spec        # or: dlm | ar   (PORT=30000)
uv run python -m scripts.compare_sglang --mode linear_spec   # vs HF reference
```

One decoding mode per server (chosen at launch). Measured on one A100
(3B, 30 GSM8K questions, greedy, concurrency 1):

| mode | SGLang acc. | HF acc. | identical outputs | SGLang tok/s | HF tok/s |
|---|---|---|---|---|---|
| linear_spec | 86.7% | 86.7% | 19/30 | 450 | 187 |
| dlm (FastDiffuser, threshold 0.9) | 83.3% | 86.7% | 8/30 | 291 | 116 |
| ar (`ar_mode` + FastDiffuser) | **36.7%** | 86.7% | 0/30 | 109 | 51 |

SGLang's `ar` mode is not the model's real AR path and loses accuracy; use
`linear_spec` (lossless AR-verified) or the HF engine for AR.

`scripts/check_switch.py` verifies that AR <-> diffusion switching can be
driven by the causal hidden state (see that file's docstring).

## Running experiments (config-driven)

Experiments are YAML files in `configs/experiments/` (see its README for the
schema): decoding granularity, RL hyperparameters, data mix, evaluated routers
and sweeps. No code changes are needed for a new experiment.

```bash
uv run python -m scripts.run_experiments configs/experiments/e15_blk4_ar1.yaml --gpus 0 1 2 3
uv run python -m scripts.summarize_experiments --reference baselines:random:0.0
```
