r"""Training prompts and verifiable rewards for router RL.

Sources (training splits only; evaluation stays on GSM8K test / HumanEval):
    gsm8k     openai/gsm8k main/train (7,473), answer checked like NVIDIA's
              evaluate.py (last \\boxed{}, else last number).
    mbpp      google-research-datasets/mbpp full train + validation (464),
              checked by running its assert tests.
    kodcode   KodCode/KodCode-V1 (CC-BY-NC-4.0), a filtered sample of
              function-level tasks, checked by running its pytest file.
              Only items whose reference solution passes in our sandbox
              are kept (validated once, cached in runs/rl_cache/).

Model-written code runs only inside a network-less Docker container built
from docker/sandbox.Dockerfile, several programs in parallel.
"""

import json
import pathlib
import random
import re
import subprocess
import tempfile

from scripts.eval_router_benchmarks import GSM8K_INSTRUCTION, score_gsm8k

SANDBOX_IMAGE = "thinking-fast-slow-sandbox"
CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)
KODCODE_SHARDS = (0, 5, 10)
CACHE_DIR = pathlib.Path(__file__).resolve().parent.parent / "runs" / "rl_cache"

MBPP_TEMPLATE = (
    "Write a Python function for the task below. Your code should pass these "
    "tests:\n\n{tests}\n\nRespond with the complete function, including any "
    "imports it needs, in a single ```python code block.\n\nTask: {text}"
)
KODCODE_TEMPLATE = (
    "{question}\n\nImplement `{signature}`. Respond with the complete code, "
    "including any imports it needs, in a single ```python code block."
)


def _gsm8k() -> list[dict]:
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset("openai/gsm8k", "main", split="train")
    return [
        {
            "source": "gsm8k",
            "id": f"gsm8k-{i}",
            "prompt": GSM8K_INSTRUCTION + r["question"],
            "gold": r["answer"].split("####")[-1].strip().replace(",", ""),
        }
        for i, r in enumerate(ds)
    ]


def _mbpp() -> list[dict]:
    from datasets import load_dataset  # noqa: PLC0415

    items = []
    for split in ("train", "validation"):
        for r in load_dataset(
            "google-research-datasets/mbpp", "full", split=split
        ):
            items.append(
                {
                    "source": "mbpp",
                    "id": f"mbpp-{r['task_id']}",
                    "prompt": MBPP_TEMPLATE.format(
                        tests="\n".join(r["test_list"]), text=r["text"]
                    ),
                    "setup": r["test_setup_code"] or "",
                    "tests": r["test_list"],
                }
            )
    return items


def _kodcode(limit: int, seed: int) -> list[dict]:
    """Filtered KodCode sample whose reference solutions pass our sandbox.

    Filters: instruct style, one function, easy/medium, GPT pass rate >= 0.3,
    benchmark similarity <= 0.85.
    """
    import pandas as pd  # noqa: PLC0415
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    frames = [
        pd.read_parquet(
            hf_hub_download(
                "KodCode/KodCode-V1",
                f"data/train-{s:05d}-of-00015.parquet",
                repo_type="dataset",
            ),
            columns=[
                "question_id", "question", "test", "test_info", "solution",
                "style",
                "gpt_difficulty", "gpt_pass_percentage", "benchmark_similarity",
            ],
        )
        for s in KODCODE_SHARDS
    ]  # fmt: skip
    df = pd.concat(frames)
    df = df[
        (df["style"] == "instruct")
        & df["gpt_difficulty"].isin(["easy", "medium"])
        & (df["gpt_pass_percentage"] >= 0.3)
        & (df["benchmark_similarity"] <= 0.85)
        & (df["test_info"].map(len) == 1)
    ]
    rows = df.sample(n=min(limit, len(df)), random_state=seed).to_dict(
        "records"
    )
    items = [
        {
            "source": "kodcode",
            "id": f"kodcode-{r['question_id']}",
            "prompt": KODCODE_TEMPLATE.format(
                question=r["question"],
                signature=r["test_info"][0]["function_declaration"],
            ),
            "test": r["test"],
            "solution": r["solution"],
        }
        for r in rows
    ]
    return _validated(items, CACHE_DIR / f"kodcode_valid_{limit}_{seed}.json")


def _validated(items: list[dict], cache: pathlib.Path) -> list[dict]:
    """Keeps items whose reference solution passes (result cached)."""
    if cache.exists():
        valid = set(json.loads(cache.read_text()))
    else:
        refs = [f"```python\n{it['solution']}\n```" for it in items]
        ok = score_batch(items, refs)
        valid = {
            it["id"] for it, passed in zip(items, ok, strict=True) if passed
        }
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(sorted(valid)))
    return [it for it in items if it["id"] in valid]


def load_items(kodcode_limit: int = 3000, seed: int = 0) -> dict[str, list]:
    """Returns training items per source."""
    return {
        "gsm8k": _gsm8k(),
        "mbpp": _mbpp(),
        "kodcode": _kodcode(kodcode_limit, seed),
    }


def sample_batch(
    items: dict[str, list],
    weights: dict[str, float],
    n: int,
    rng: random.Random,
) -> list[dict]:
    """Draws n prompts, choosing the source by `weights` for each draw."""
    sources = list(weights)
    picks = rng.choices(sources, weights=[weights[s] for s in sources], k=n)
    return [rng.choice(items[s]) for s in picks]


def extract_code(text: str) -> str:
    """First fenced code block, else the whole text."""
    blocks = CODE_BLOCK.findall(text)
    return blocks[0] if blocks else text


def _code_job(item: dict, text: str) -> dict:
    """Files and command that pass iff the completion is correct."""
    code = extract_code(text)
    if item["source"] == "mbpp":
        program = "\n".join([code, item["setup"], *item["tests"]])
        return {"files": {"main.py": program}, "cmd": ["python", "main.py"]}
    return {
        "files": {"solution.py": code, "test_solution.py": item["test"]},
        "cmd": [
            "python", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
            "test_solution.py",
        ],
    }  # fmt: skip


RUNNER = r"""
import concurrent.futures, json, pathlib, subprocess, sys
jobs = json.loads(pathlib.Path("/work/jobs.json").read_text())
def run(name):
    job = jobs[name]
    try:
        r = subprocess.run(job["cmd"], cwd=f"/tmp/{name}", capture_output=True,
                           timeout=10)
        return name, r.returncode == 0
    except subprocess.TimeoutExpired:
        return name, False
for name, job in jobs.items():
    d = pathlib.Path(f"/tmp/{name}"); d.mkdir()
    for fname, src in job["files"].items():
        (d / fname).write_text(src)
with concurrent.futures.ThreadPoolExecutor(8) as pool:
    print(json.dumps(dict(pool.map(run, jobs))))
"""


def run_code_jobs(jobs: dict[str, dict]) -> dict[str, bool]:
    """Runs all jobs in one network-less container; returns pass/fail."""
    if not jobs:
        return {}
    with tempfile.TemporaryDirectory() as tmp:
        (pathlib.Path(tmp) / "jobs.json").write_text(json.dumps(jobs))
        (pathlib.Path(tmp) / "runner.py").write_text(RUNNER)
        out = subprocess.run(
            [
                "docker", "run", "--rm", "--network", "none",
                "--memory", "8g", "--cpus", "8", "--pids-limit", "1024",
                "-v", f"{tmp}:/work:ro", SANDBOX_IMAGE,
                "python", "/work/runner.py",
            ],  # fmt: skip
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    return json.loads(out.strip().splitlines()[-1])


def score_batch(items: list[dict], texts: list[str]) -> list[bool]:
    """Correctness of each completion for its item."""
    results: list[bool | None] = [None] * len(items)
    jobs = {}
    for i, (item, text) in enumerate(zip(items, texts, strict=True)):
        if item["source"] == "gsm8k":
            results[i] = score_gsm8k(text, item["gold"])
        else:
            jobs[f"job{i}"] = _code_job(item, text)
    for name, ok in run_code_jobs(jobs).items():
        results[int(name.removeprefix("job"))] = ok
    return [bool(r) for r in results]
