"""Runs router experiments described by YAML files.

An experiment file overrides `DEFAULTS` below (unknown keys are errors) and
may define a `sweep` (cartesian product over dotted keys), e.g.

    name: entropy_grid
    decoding: {block_size: 4, ar_chunk: 1}
    train: {entropy_coef: 0.01}
    eval: {routers: [learned, random:1.0]}
    sweep:
      train.entropy_coef: [0.005, 0.01, 0.02]

Each run (one sweep point) trains the router (if `train` is not null) on
SGLang RoutedDecoding servers, then evaluates the routers listed in
`eval.routers` with the HF reference decoder. `learned` means the run's final
checkpoint, `learned:iter50` an intermediate one; other entries are
`scripts.eval_router_benchmarks` router specs (`random:<p>`, `entropy:<nats>`,
`learned:<path>`). Decoding settings (`decoding.*`) apply to both training
rollouts and evaluation.

Runs are scheduled on a GPU pool, `gpus_per_run` GPUs each; several runs
execute in parallel when the pool allows. Everything for a run lives in
`runs/experiments/<run name>/`: `config.yaml` (resolved), `train/`, `eval/`,
`results.json`. Re-running skips finished stages (training resumes, evaluation
skips finished items).

    uv run python -m scripts.run_experiments configs/experiments/e15.yaml \
        --gpus 0 1 2 3
    uv run python -m scripts.run_experiments configs/experiments/*.yaml \
        --gpus 0 1 --dry-run
"""

import argparse
import copy
import itertools
import json
import pathlib
import queue
import subprocess
import sys
import threading
import time

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "experiments"
SGLANG_PYTHON = ROOT / ".venv-sglang" / "bin" / "python"

DEFAULTS: dict = {
    "name": None,
    "gpus_per_run": 2,
    # Shared by training rollouts (SGLang) and evaluation (HF).
    "decoding": {
        "block_size": 32,  # diffusion block length
        "ar_chunk": 8,  # AR tokens per router decision
        "forced_ar_tokens": 0,  # always-AR start, router not consulted
        "threshold": 0.9,  # diffusion unmasking confidence
    },
    # null -> no training (evaluate given routers only).
    "train": {
        "iterations": 100,
        "prompts": 32,
        "group_size": 8,
        "max_tokens": 512,
        "cost_weight": 0.1,
        "lr": 3.0e-4,
        "epochs": 2,
        "clip": 0.2,
        "entropy_coef": 0.01,
        "adv_norm": "none",
        "weights": {"gsm8k": 0.35, "mbpp": 0.15, "kodcode": 0.5},
        "kodcode_limit": 3000,
        "max_reqs": 64,
        "mem_frac": 0.75,
        "save_every": 10,
        "seed": 0,
    },
    "eval": {
        "benchmarks": ["gsm8k", "humaneval"],
        "routers": ["learned"],
        "max_tokens": 1024,
        "workers_per_gpu": 6,
        "limit": None,  # cap items per benchmark (smoke tests)
    },
    "sweep": {},
}


# ----------------------------------------------------------------- config
def merge(base: dict, override: dict, path: str = "") -> dict:
    """Deep-merges `override` into a copy of `base`; unknown keys raise."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        where = f"{path}{key}"
        if key not in base:
            raise KeyError(f"unknown config key {where!r}")
        if isinstance(base[key], dict) and key not in ("weights", "sweep"):
            if value is None:
                out[key] = None
            elif not isinstance(value, dict):
                raise TypeError(f"{where!r} must be a mapping")
            else:
                out[key] = merge(base[key], value, where + ".")
        else:
            out[key] = value
    return out


def set_dotted(cfg: dict, dotted: str, value: object) -> None:
    """Sets cfg["a"]["b"] for dotted key "a.b" (key must exist)."""
    *parents, leaf = dotted.split(".")
    node = cfg
    for p in parents:
        node = node[p]
    if leaf not in node:
        raise KeyError(f"unknown sweep key {dotted!r}")
    node[leaf] = value


def expand(path: pathlib.Path) -> list[dict]:
    """Loads one experiment file and expands its sweep into runs."""
    raw = yaml.safe_load(path.read_text()) or {}
    base = merge(DEFAULTS, raw)
    base["name"] = base["name"] or path.stem
    sweep = base.pop("sweep") or {}
    if not sweep:
        return [base]
    keys = list(sweep)
    runs = []
    for values in itertools.product(*(sweep[k] for k in keys)):
        run = copy.deepcopy(base)
        for k, v in zip(keys, values, strict=True):
            set_dotted(run, k, v)
        suffix = "_".join(
            f"{k.split('.')[-1]}{v}" for k, v in zip(keys, values, strict=True)
        )
        run["name"] = f"{base['name']}_{suffix}"
        runs.append(run)
    return runs


# ----------------------------------------------------------------- commands
def decoding_suffix(dec: dict) -> str:
    """Eval setting suffix encoding the decoding granularity."""
    suffix = f"@ar{dec['ar_chunk']}@blk{dec['block_size']}"
    if dec["forced_ar_tokens"]:
        suffix += f"@first{dec['forced_ar_tokens']}"
    return suffix


def train_command(run: dict, out: pathlib.Path, gpus: list, port: int) -> list:
    """Command line for scripts.train_router_rl."""
    t, d = run["train"], run["decoding"]
    weights = ",".join(f"{k}:{v}" for k, v in t["weights"].items())
    return [
        str(SGLANG_PYTHON), "-m", "scripts.train_router_rl",
        "--out", str(out), "--gpus", *map(str, gpus), "--port-base", str(port),
        "--iterations", str(t["iterations"]), "--prompts", str(t["prompts"]),
        "--group-size", str(t["group_size"]),
        "--max-tokens", str(t["max_tokens"]),
        "--cost-weight", str(t["cost_weight"]), "--lr", str(t["lr"]),
        "--epochs", str(t["epochs"]), "--clip", str(t["clip"]),
        "--entropy-coef", str(t["entropy_coef"]), "--adv-norm", t["adv_norm"],
        "--weights", weights, "--kodcode-limit", str(t["kodcode_limit"]),
        "--max-reqs", str(t["max_reqs"]), "--mem-frac", str(t["mem_frac"]),
        "--save-every", str(t["save_every"]), "--seed", str(t["seed"]),
        "--block-size", str(d["block_size"]), "--ar-chunk", str(d["ar_chunk"]),
        "--forced-ar-tokens", str(d["forced_ar_tokens"]),
        "--threshold", str(d["threshold"]),
    ]  # fmt: skip


def eval_settings(run: dict, train_dir: pathlib.Path) -> list[str]:
    """Expands eval.routers into eval_router_benchmarks settings."""
    suffix = decoding_suffix(run["decoding"])
    settings = []
    for spec in run["eval"]["routers"]:
        if spec == "learned" or spec.startswith("learned:iter"):
            if run["train"] is None:
                raise ValueError(f"{spec!r} needs a train section")
            it = (
                run["train"]["iterations"]
                if spec == "learned"
                else int(spec.removeprefix("learned:iter"))
            )
            spec = f"learned:{train_dir / f'router_iter{it:04d}.pt'}"
        settings.append(spec + suffix)
    return settings


def eval_commands(
    run: dict, out: pathlib.Path, gpus: list, settings: list
) -> list[list]:
    """Generate + score command lines for scripts.eval_router_benchmarks."""
    e, d = run["eval"], run["decoding"]
    common = [
        "--out", str(out), "--benchmarks", *e["benchmarks"],
        "--max-tokens", str(e["max_tokens"]),
        "--threshold", str(d["threshold"]),
    ]  # fmt: skip
    gen = [
        "uv", "run", "python", "-m", "scripts.eval_router_benchmarks",
        "generate", *common, "--gpus", *map(str, gpus),
        "--workers", str(e["workers_per_gpu"] * len(gpus)),
        "--settings", *settings,
    ]  # fmt: skip
    if e["limit"] is not None:
        gen += ["--limit", str(e["limit"])]
    score = [
        "uv", "run", "python", "-m", "scripts.eval_router_benchmarks",
        "score", *common,
    ]  # fmt: skip
    return [gen, score]


# ----------------------------------------------------------------- running
def log(msg: str) -> None:
    """Prints a timestamped line (shared by all runner threads)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def execute(cmd: list, log_path: pathlib.Path, dry_run: bool) -> int:
    """Runs a command with output appended to `log_path`."""
    if dry_run:
        print("    $ " + " ".join(cmd))
        return 0
    with log_path.open("a") as f:
        return subprocess.run(
            cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT
        ).returncode


def run_one(run: dict, gpus: list, port: int, dry_run: bool) -> None:
    """Trains and evaluates one run on the given GPUs."""
    out = RUNS / run["name"]
    train_dir, eval_dir = out / "train", out / "eval"
    if not dry_run:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.yaml").write_text(yaml.safe_dump(run, sort_keys=False))
    log(f"{run['name']}: start on GPUs {gpus}")
    if run["train"] is not None:
        final = train_dir / f"router_iter{run['train']['iterations']:04d}.pt"
        if final.exists():
            log(f"{run['name']}: training already finished")
        else:
            rc = execute(
                train_command(run, train_dir, gpus, port),
                out / "train.log",
                dry_run,
            )
            if rc != 0:
                log(
                    f"{run['name']}: training FAILED (exit {rc}), see train.log"
                )
                return
    settings = eval_settings(run, train_dir)
    for cmd in eval_commands(run, eval_dir, gpus, settings):
        rc = execute(cmd, out / "eval.log", dry_run)
        if rc != 0:
            log(f"{run['name']}: evaluation FAILED (exit {rc}), see eval.log")
            return
    if not dry_run:
        rows = json.loads((eval_dir / "summary.json").read_text())
        (out / "results.json").write_text(
            json.dumps(
                {"name": run["name"], "settings": settings, "rows": rows},
                indent=1,
            )
        )
    log(f"{run['name']}: done")


def main() -> None:
    """Expands all experiment files and schedules their runs on the pool."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("configs", nargs="+", type=pathlib.Path)
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--port-base", type=int, default=30100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    runs = [r for path in args.configs for r in expand(path)]
    names = [r["name"] for r in runs]
    if len(set(names)) != len(names):
        sys.exit(f"duplicate run names: {names}")
    width = {r["gpus_per_run"] for r in runs}
    if len(width) != 1:
        sys.exit("all runs in one invocation must share gpus_per_run")
    per_run = width.pop()
    slots = [
        args.gpus[i : i + per_run]
        for i in range(0, len(args.gpus) - per_run + 1, per_run)
    ]
    if not slots:
        sys.exit(f"need at least {per_run} GPUs")
    log(f"{len(runs)} runs on {len(slots)} slot(s): {slots}")

    todo: queue.Queue = queue.Queue()
    for r in runs:
        todo.put(r)

    def worker(slot: int, gpus: list) -> None:
        while True:
            try:
                run = todo.get_nowait()
            except queue.Empty:
                return
            run_one(run, gpus, args.port_base + 20 * slot, args.dry_run)

    threads = [
        threading.Thread(target=worker, args=(i, g))
        for i, g in enumerate(slots)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log("all runs finished")


if __name__ == "__main__":
    main()
