"""Runs Harbor-format tasks with mini-swe-agent against the local server.

A Harbor task is a directory with `instruction.md`, `task.toml`,
`environment/Dockerfile` and `tests/test.sh` (which writes the reward to
`/logs/verifier/reward.txt`). The agent runs on the host and acts inside the
task container; the verifier runs in that same container afterwards.

    uv run python -m src.harbor_runner tasks/hello-world --mode dlm
"""

import argparse
import json
import pathlib
import subprocess
import time
import tomllib
import typing

from minisweagent.agents import get_agent
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.models import get_model
from minisweagent.utils.serialize import recursive_merge

CONFIGS = ("mini_textbased.yaml", "configs/mini_nemotron.yaml")


def run_task(
    task_dir: pathlib.Path, mode: str, step_limit: int, out_dir: pathlib.Path
) -> dict:
    """Runs one Harbor task and returns its result.

    Args:
        task_dir: Path to the Harbor task directory.
        mode: Decoding mode, "ar" or "dlm".
        step_limit: Maximum number of agent steps.
        out_dir: Directory for the trajectory file.

    Returns:
        Dict with task, mode, reward, exit status, steps and seconds.
    """
    name = task_dir.name
    task_cfg = tomllib.loads((task_dir / "task.toml").read_text())
    instruction = (task_dir / "instruction.md").read_text().strip()
    image = f"harbor-local-{name}"
    subprocess.run(
        ["docker", "build", "-q", "-t", image, str(task_dir / "environment")],
        check=True,
    )

    config = recursive_merge(
        *(get_config_from_spec(spec) for spec in CONFIGS),
        {
            "model": {"model_name": f"openai/nemotron-{mode}"},
            "environment": {
                "environment_class": "docker",
                "image": image,
                "cwd": "/app",
                "timeout": int(task_cfg["agent"]["timeout_sec"]),
            },
            "agent": {
                "step_limit": step_limit,
                "output_path": str(out_dir / f"{name}.{mode}.traj.json"),
            },
        },
    )
    model = get_model(config=config["model"])
    env = typing.cast(
        DockerEnvironment,
        get_environment(config["environment"], default_type="docker"),
    )
    agent = typing.cast(
        DefaultAgent,
        get_agent(model, env, config["agent"], default_type="default"),
    )

    start = time.perf_counter()
    try:
        info = agent.run(instruction)
        exit_status = info.get("exit_status", "unknown")
    except Exception as e:  # Keep going so the verifier still runs.
        exit_status = f"{type(e).__name__}: {e}"
    seconds = time.perf_counter() - start

    # Verifier: copy tests in, run, read reward.
    cid = str(env.container_id)
    subprocess.run(
        ["docker", "exec", cid, "mkdir", "-p", "/logs/verifier"], check=True
    )
    subprocess.run(
        ["docker", "cp", str(task_dir / "tests") + "/.", f"{cid}:/tests"],
        check=True,
    )
    subprocess.run(
        ["docker", "exec", cid, "bash", "/tests/test.sh"],
        capture_output=True,
        timeout=int(task_cfg["verifier"]["timeout_sec"]),
    )
    reward = subprocess.run(
        ["docker", "exec", cid, "cat", "/logs/verifier/reward.txt"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    env.cleanup()
    return {
        "task": name,
        "mode": mode,
        "reward": reward or "none",
        "exit_status": exit_status,
        "steps": agent.n_calls,
        "seconds": round(seconds, 1),
    }


def main() -> None:
    """Runs the given tasks in the requested modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", nargs="+", type=pathlib.Path)
    parser.add_argument("--mode", nargs="+", default=["ar", "dlm"])
    parser.add_argument("--step-limit", type=int, default=15)
    parser.add_argument("--out", type=pathlib.Path, default="runs")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for task_dir in args.tasks:
        for mode in args.mode:
            result = run_task(task_dir, mode, args.step_limit, args.out)
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
