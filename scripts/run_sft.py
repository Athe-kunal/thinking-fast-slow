"""Runs span-SFT experiments described by YAML files (`configs/sft/`).

Same conventions as `scripts.run_experiments`: a file overrides `DEFAULTS`
(unknown keys are errors) and may `sweep` dotted keys. Each run trains (unless
`train` is null) on one GPU, then evaluates every entry of `eval.settings` on
the held-out tool-calling turns:

    spans   the run's adapter, diffusion inside <diff> spans
    ar      the run's adapter, every token AR
    base    the base model (no adapter), AR, prompts without markers

Outputs: `runs/sft/<run>/{config.yaml,train.log,train/,eval/,results.json}`.

    uv run python -m scripts.run_sft configs/sft/e17_span_sft.yaml --gpus 2 3
"""

import argparse
import json
import pathlib
import queue
import subprocess
import sys
import threading

import yaml

from scripts.run_experiments import expand, log

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "sft"

DEFAULTS: dict = {
    "name": None,
    "data": {
        "shards": [0],  # tool-calling parquet shards (of 13)
        "max_len": 4096,  # drop longer conversations
        "heldout_percent": 5,
        "train_limit": 8000,  # training conversations
        "diff_spans": True,  # wrap tool calls in <diff> ... </diff>
    },
    "loss": {
        "block_size": 8,  # diffusion block inside spans
        "ar_in_spans": True,  # AR loss on span tokens too
        "ar_weight": 1.0,
        "dlm_weight": 1.0,
    },
    "lora": {"rank": 32, "alpha": 64},
    # null -> no training (evaluate the base model only).
    "train": {
        "epochs": 1,
        "batch": 16,  # conversations per optimizer step
        "lr": 1.0e-4,
        "rows_lr": 1.0e-3,  # span-token embedding / head rows
        "warmup": 20,
        "grad_clip": 1.0,
        "save_every": 100,
        "seed": 0,
    },
    "eval": {
        "settings": ["spans", "ar"],
        "limit": None,  # held-out turns (730 available)
        "max_new_tokens": 2048,
        "threshold": 0.9,
        "workers_per_gpu": 3,
    },
    "sweep": {},
}


def execute(
    cmd: list, log_path: pathlib.Path, env_gpu: str | None, dry: bool
) -> int:
    """Runs a command, output appended to `log_path`."""
    if dry:
        print("    $ " + " ".join(cmd))
        return 0
    import os

    env = dict(os.environ)
    if env_gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = env_gpu
    with log_path.open("a") as f:
        return subprocess.run(
            cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env
        ).returncode


def run_one(run: dict, gpus: list, dry: bool) -> None:
    """Trains and evaluates one run."""
    out = RUNS / run["name"]
    if not dry:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.yaml").write_text(yaml.safe_dump(run, sort_keys=False))
    log(f"{run['name']}: start on GPUs {gpus}")
    adapter = out / "train" / "adapter_final.pt"
    if run["train"] is not None:
        cmd = ["uv", "run", "python", "-m", "scripts.train_sft",
               "--config", str(out / "config.yaml"), "--out", str(out / "train")]  # fmt: skip
        if execute(cmd, out / "train.log", gpus[0], dry):
            log(f"{run['name']}: training FAILED, see train.log")
            return
    e = run["eval"]
    for setting in e["settings"]:
        if setting in ("spans", "ar") and run["train"] is None:
            raise ValueError(f"{setting!r} needs a train section")
        cmd = ["uv", "run", "python", "-m", "scripts.eval_toolcall",
               "--out", str(out / "eval"), "--setting", setting,
               "--mode", "ar" if setting in ("ar", "base") else "spans",
               "--gpus", *gpus, "--workers-per-gpu", str(e["workers_per_gpu"]),
               "--max-new-tokens", str(e["max_new_tokens"]),
               "--block-size", str(run["loss"]["block_size"]),
               "--threshold", str(e["threshold"])]  # fmt: skip
        if setting != "base":
            cmd += ["--adapter", str(adapter)]
            if run["data"]["diff_spans"]:
                cmd.append("--diff-spans")
        if e["limit"] is not None:
            cmd += ["--limit", str(e["limit"])]
        if execute(cmd, out / "eval.log", None, dry):
            log(f"{run['name']}: evaluation {setting} FAILED, see eval.log")
            return
    if not dry:
        summary = json.loads((out / "eval" / "summary.json").read_text())
        (out / "results.json").write_text(
            json.dumps({"name": run["name"], "summary": summary}, indent=1)
        )
    log(f"{run['name']}: done")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("configs", nargs="+", type=pathlib.Path)
    parser.add_argument(
        "--gpus", nargs="+", required=True, help="one run per GPU"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    runs = [r for p in args.configs for r in expand(p, DEFAULTS)]
    names = [r["name"] for r in runs]
    if len(set(names)) != len(names):
        sys.exit(f"duplicate run names: {names}")
    log(f"{len(runs)} runs on GPUs {args.gpus}")
    todo: queue.Queue = queue.Queue()
    for r in runs:
        todo.put(r)

    def worker(gpu: str) -> None:
        while True:
            try:
                run = todo.get_nowait()
            except queue.Empty:
                return
            run_one(run, [gpu], args.dry_run)

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log("all runs finished")


if __name__ == "__main__":
    main()
