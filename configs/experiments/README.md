# Experiment configs

Each YAML file describes one experiment for `scripts/run_experiments.py`. It
overrides only what differs from `DEFAULTS` at the top of that script (unknown
keys are rejected) and may sweep any keys:

```yaml
name: my_experiment            # default: file name
gpus_per_run: 2                # GPUs per run (= SGLang servers when training)
decoding:                      # used for RL rollouts AND evaluation
  block_size: 4                # diffusion block length
  ar_chunk: 1                  # AR tokens per router decision
  forced_ar_tokens: 0          # always-AR start (router not consulted)
  threshold: 0.9               # diffusion unmasking confidence
train:                         # RL hyperparameters; `train: null` = eval only
  entropy_coef: 0.01
  adv_norm: none               # none | std
  cost_weight: 0.1
  weights: {gsm8k: 0.35, mbpp: 0.15, kodcode: 0.5}
eval:
  benchmarks: [gsm8k, humaneval]
  routers: [learned, learned:iter50, random:1.0, entropy:0.5]
sweep:                         # cartesian product -> one run per point
  train.entropy_coef: [0.005, 0.01, 0.02]
  decoding.block_size: [4, 8]
```

Run, preview, summarize:

```bash
uv run python -m scripts.run_experiments configs/experiments/e15_blk4_ar1.yaml --gpus 0 1 2 3
uv run python -m scripts.run_experiments configs/experiments/*.yaml --gpus 0 1 --dry-run
uv run python -m scripts.summarize_experiments --reference baselines:random:0.0
```

Outputs: `runs/experiments/<run>/{config.yaml, train/, eval/, results.json}`.

The configs `e13_*`, `e14_*`, `e15_*` record the experiments started by hand on
2026-10-04 (before this runner existed); their outputs are under
`runs/router_rl/`, not `runs/experiments/`.
