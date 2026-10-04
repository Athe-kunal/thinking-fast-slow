"""Evaluates AR/diffusion routing policies on GSM8K and HumanEval.

A setting names a router: `random:<p>` routes each 32-token segment to
diffusion with probability p (AR segments are also 32 tokens, so p is the
expected diffusion token share; p=0 is pure AR, p=1 pure diffusion), and
`entropy:<nats>` uses diffusion when the next-token entropy at the decision
point is below the threshold. A suffix `@ar<n>` (e.g. `entropy:0.5@ar1`)
sets the AR segment length, i.e. how many AR tokens are decoded before the
router decides again (default `--ar-chunk`). Generation runs several worker
processes per GPU; each loads the model once and takes every n-th item from a
priority-ordered list (settings in the order given), appending results to its
own JSONL file, so a rerun resumes where it stopped.

    # generate (GPU), then score (GSM8K inline, HumanEval in a sandbox)
    uv run python -m scripts.eval_router_benchmarks generate \
        --out runs/router_bench --gpus 2 3 --workers 12 \
        --settings random:0.0 random:0.2 random:0.8 random:1.0 entropy:0.5
    uv run python -m scripts.eval_router_benchmarks score \
        --out runs/router_bench

HumanEval completions are executed inside a network-less Docker container,
never on the host.
"""

import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import tempfile

GSM8K_INSTRUCTION = (
    "Solve the following math problem. Put the final numerical answer "
    "inside \\boxed{} at the very end.\n\n"
)
HUMANEVAL_INSTRUCTION = (
    "Complete the following Python function. Respond with the complete "
    "function, including the signature and any imports it needs, in a single "
    "```python code block.\n\n```python\n{prompt}```"
)
SANDBOX_IMAGE = "python:3.12-slim"
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def load_benchmark(name: str) -> list[dict]:
    """Returns benchmark items as dicts with id, prompt and reference data."""
    from datasets import load_dataset  # noqa: PLC0415

    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        return [
            {
                "id": i,
                "prompt": GSM8K_INSTRUCTION + r["question"],
                "gold": r["answer"].split("####")[-1].strip().replace(",", ""),
            }
            for i, r in enumerate(ds)
        ]
    if name == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test")
        return [
            {
                "id": i,
                "prompt": HUMANEVAL_INSTRUCTION.format(prompt=r["prompt"]),
                "task_prompt": r["prompt"],
                "test": r["test"],
                "entry_point": r["entry_point"],
            }
            for i, r in enumerate(ds)
        ]
    raise ValueError(f"unknown benchmark {name!r}")


def setting_of(record: dict) -> str:
    """Returns a record's setting; early records only stored `p_dlm`."""
    return record.get("setting") or f"random:{record['p_dlm']}"


def parse_setting(setting: str, default_ar_chunk: int) -> tuple[str, int]:
    """Splits "entropy:0.5@ar1" into ("entropy:0.5", 1).

    The optional "@ar<n>" suffix sets the AR segment length (tokens decoded
    before the router decides again); without it `default_ar_chunk` is used.
    """
    router, _, suffix = setting.partition("@")
    if not suffix:
        return router, default_ar_chunk
    if not suffix.startswith("ar"):
        raise ValueError(f"unknown setting suffix {suffix!r}")
    return router, int(suffix.removeprefix("ar"))


def make_router(setting: str, bench: str, idx: int):  # noqa: ANN201
    """Builds the router for one item of one setting."""
    from src import router as router_lib  # noqa: PLC0415

    kind, value = setting.partition("@")[0].split(":")
    if kind == "random":
        p = float(value)
        return router_lib.RandomRouter(p, item_seed(p, bench, idx))
    if kind == "entropy":
        return router_lib.EntropyRouter(float(value))
    raise ValueError(f"unknown setting {setting!r}")


def item_seed(p_dlm: float, bench: str, idx: int) -> int:
    """Stable per-item router seed, independent of which worker runs it."""
    digest = hashlib.sha256(f"{p_dlm}/{bench}/{idx}".encode()).hexdigest()
    return int(digest[:8], 16)


def run_worker(args: argparse.Namespace) -> None:
    """Generates this worker's share of all (setting, benchmark, item)s."""
    import torch  # noqa: PLC0415

    from src import engine as engine_lib  # noqa: PLC0415

    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    out_path = args.out / f"worker{args.worker}.jsonl"
    # Skip items finished by any worker, so a restart with a different
    # number of workers neither redoes nor drops work.
    done = {
        (setting_of(r), r["bench"], r["id"]) for r in load_records(args.out)
    }

    data = {b: load_benchmark(b) for b in args.benchmarks}
    work = [
        (s, b, item)
        for s in args.settings
        for b in args.benchmarks
        for item in data[b][: args.limit]
    ]
    mine = [w for k, w in enumerate(work) if k % args.workers == args.worker]
    todo = [(s, b, it) for s, b, it in mine if (s, b, it["id"]) not in done]
    print(
        f"worker {args.worker}: {len(todo)}/{len(mine)} items to do", flush=True
    )

    engine = engine_lib.NemotronEngine(device="cuda", ar_chunk=args.ar_chunk)
    with out_path.open("a") as f:
        for setting, bench, item in todo:
            engine.router = make_router(setting, bench, item["id"])
            engine.ar_chunk = parse_setting(setting, args.ar_chunk)[1]
            result = engine.generate(
                item["prompt"],
                mode="mix",
                max_new_tokens=args.max_tokens,
                block_length=args.block_length,
                threshold=args.threshold,
            )
            record = {
                "setting": setting,
                "bench": bench,
                "id": item["id"],
                "text": result.text,
                "num_tokens": result.num_tokens,
                "nfe": result.nfe,
                "seconds": result.seconds,
                "segments": result.segments,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()


def load_records(out: pathlib.Path) -> list[dict]:
    """Returns all generated records, one per (setting, benchmark, item)."""
    records: dict[tuple, dict] = {}
    for path in sorted(out.glob("worker*.jsonl")):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            records.setdefault((setting_of(r), r["bench"], r["id"]), r)
    return list(records.values())


def generate(args: argparse.Namespace) -> None:
    """Launches the worker processes and waits for them."""
    gpus = args.gpus or os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if not gpus or gpus == [""]:
        sys.exit("Pass --gpus or set CUDA_VISIBLE_DEVICES.")
    args.out.mkdir(parents=True, exist_ok=True)
    procs = []
    for w in range(args.workers):
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": gpus[w % len(gpus)],
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
        cmd = [
            sys.executable, "-m", "scripts.eval_router_benchmarks", "worker",
            "--out", str(args.out),
            "--worker", str(w),
            "--workers", str(args.workers),
            "--settings", *args.settings,
            "--benchmarks", *args.benchmarks,
            "--max-tokens", str(args.max_tokens),
            "--block-length", str(args.block_length),
            "--ar-chunk", str(args.ar_chunk),
            "--threshold", str(args.threshold),
            "--gpu-memory-fraction", str(args.gpu_memory_fraction),
        ]  # fmt: skip
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        log = (args.out / f"worker{w}.log").open("a")
        procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=log))
    codes = [p.wait() for p in procs]
    print("worker exit codes:", codes)


def score_gsm8k(text: str, gold: str) -> bool:
    r"""NVIDIA evaluate.py scoring: last \boxed{}, else last number."""
    boxed = BOXED.findall(text)
    pred = boxed[-1].strip() if boxed else None
    if pred is None:
        nums = NUMBER.findall(text.replace(",", ""))
        pred = nums[-1] if nums else None
    if pred is None:
        return False
    try:
        return abs(float(pred.replace(",", "")) - float(gold)) < 1e-6
    except ValueError:
        return pred.strip() == gold.strip()


def humaneval_program(text: str, item: dict) -> str:
    """Builds the test program for one HumanEval completion."""
    blocks = CODE_BLOCK.findall(text)
    code = blocks[0] if blocks else text
    if f"def {item['entry_point']}" not in code:
        code = item["task_prompt"] + code
    return f"{code}\n\n{item['test']}\n\ncheck({item['entry_point']})\n"


RUNNER = """
import json, pathlib, subprocess, sys
results = {}
for f in sorted(pathlib.Path("/work").glob("*.py")):
    try:
        r = subprocess.run([sys.executable, str(f)], capture_output=True,
                           timeout=10, cwd="/tmp")
        results[f.stem] = r.returncode == 0
    except subprocess.TimeoutExpired:
        results[f.stem] = False
print(json.dumps(results))
"""


def run_sandboxed(programs: dict[str, str]) -> dict[str, bool]:
    """Executes programs in a network-less container; returns pass/fail."""
    with tempfile.TemporaryDirectory() as tmp:
        for name, src in programs.items():
            (pathlib.Path(tmp) / f"{name}.py").write_text(src)
        (pathlib.Path(tmp) / "runner.py.txt").write_text(RUNNER)
        out = subprocess.run(
            [
                "docker", "run", "--rm", "--network", "none",
                "--memory", "4g", "--cpus", "4", "--pids-limit", "256",
                "-v", f"{tmp}:/work:ro", SANDBOX_IMAGE,
                "python", "/work/runner.py.txt",
            ],  # fmt: skip
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    return json.loads(out.strip().splitlines()[-1])


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    z = 1.96
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = (
        z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    )
    return centre - half, centre + half


def program_name(record: dict) -> str:
    """Sandbox file stem for one HumanEval record."""
    tag = re.sub(r"[^0-9a-zA-Z]+", "_", setting_of(record))
    return f"{tag}__{record['id']}"


def mcnemar(a: dict, b: dict) -> tuple[int, int, float]:
    """Exact two-sided McNemar test on paired per-item correctness.

    Returns:
        (items only `a` solved, items only `b` solved, p-value).
    """
    ids = a.keys() & b.keys()
    x = sum(a[i] and not b[i] for i in ids)
    y = sum(b[i] and not a[i] for i in ids)
    m = x + y
    if m == 0:
        return x, y, 1.0
    tail = sum(math.comb(m, j) for j in range(min(x, y) + 1)) / 2**m
    return x, y, min(1.0, 2 * tail)


def score(args: argparse.Namespace) -> None:
    """Scores all generated records and prints the summary table."""
    records = [
        r for r in load_records(args.out) if r["bench"] in args.benchmarks
    ]
    data = {
        b: {it["id"]: it for it in load_benchmark(b)} for b in args.benchmarks
    }

    he = [r for r in records if r["bench"] == "humaneval"]
    if he:
        subprocess.run(
            ["docker", "pull", "-q", SANDBOX_IMAGE],
            capture_output=True,
            check=False,
        )
        programs = {
            program_name(r): humaneval_program(
                r["text"], data["humaneval"][r["id"]]
            )
            for r in he
        }
        passed = run_sandboxed(programs)
    for r in records:
        if r["bench"] == "gsm8k":
            r["correct"] = score_gsm8k(
                r["text"], data["gsm8k"][r["id"]]["gold"]
            )
        else:
            r["correct"] = passed[program_name(r)]

    rows = []
    settings = sorted({setting_of(r) for r in records})
    for bench in args.benchmarks:
        for setting in settings:
            rs = [
                r
                for r in records
                if r["bench"] == bench and setting_of(r) == setting
            ]
            if not rs:
                continue
            k, n = sum(r["correct"] for r in rs), len(rs)
            seg = " ".join(r["segments"] for r in rs)
            dlm = sum(int(c) for c in re.findall(r"dlm(\d+)", seg))
            ar = sum(int(c) for c in re.findall(r"ar(\d+)", seg))
            tokens = sum(r["num_tokens"] for r in rs)
            lo, hi = wilson(k, n)
            rows.append(
                {
                    "bench": bench,
                    "setting": setting,
                    "n": n,
                    "accuracy": k / n,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "dlm_token_share": dlm / max(1, dlm + ar),
                    "tokens_per_forward": tokens / sum(r["nfe"] for r in rs),
                    "tokens_per_second": tokens / sum(r["seconds"] for r in rs),
                    "mean_tokens": tokens / n,
                    "truncated": sum(
                        r["num_tokens"] >= args.max_tokens for r in rs
                    ),
                }
            )
    print(
        f"{'bench':10} {'setting':12} {'n':>5} {'acc':>7} {'95% CI':>15} "
        f"{'dlm-tok':>8} {'tok/fwd':>7} {'tok/s':>6} {'len':>5} {'trunc':>5}"
    )
    for r in rows:
        print(
            f"{r['bench']:10} {r['setting']:12} {r['n']:5d} "
            f"{r['accuracy']:7.1%} [{r['ci95_lo']:5.1%}, {r['ci95_hi']:5.1%}] "
            f"{r['dlm_token_share']:8.1%} {r['tokens_per_forward']:7.2f} "
            f"{r['tokens_per_second']:6.1f} {r['mean_tokens']:5.0f} "
            f"{r['truncated']:5d}"
        )
    correct = {(setting_of(r), r["bench"]): {} for r in records}
    for r in records:
        correct[(setting_of(r), r["bench"])][r["id"]] = r["correct"]
    print(f"\nPaired exact McNemar vs {args.reference}:")
    for bench in args.benchmarks:
        ref = correct.get((args.reference, bench))
        if ref is None:
            continue
        for setting in settings:
            if setting == args.reference or (setting, bench) not in correct:
                continue
            x, y, pval = mcnemar(ref, correct[(setting, bench)])
            print(
                f"  {bench:10} {setting:12} only-ref {x:4d}  "
                f"only-setting {y:4d}  p={pval:.3f}"
            )
    (args.out / "summary.json").write_text(json.dumps(rows, indent=1))
    (args.out / "scored.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )


def main() -> None:
    """Parses the subcommand and runs it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "worker", "score"))
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument(
        "--settings",
        nargs="+",
        default=["random:0.2", "random:0.8", "random:0.0", "random:1.0"],
        help="Routers to evaluate: random:<p> or entropy:<nats>.",
    )
    parser.add_argument(
        "--reference",
        default="random:0.0",
        help="Setting the paired tests compare against (default: pure AR).",
    )
    parser.add_argument(
        "--benchmarks", nargs="+", default=["gsm8k", "humaneval"]
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=None,
        help="GPU ids; workers are assigned round-robin (default: env).",
    )
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument(
        "--ar-chunk",
        type=int,
        default=32,
        help="AR segment length; equal to --block-length so p = token share.",
    )
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.15)
    args = parser.parse_args()
    {"generate": generate, "worker": run_worker, "score": score}[args.command](
        args
    )


if __name__ == "__main__":
    main()
