"""Runs mini-swe-agent on SWE-bench Verified with randomly routed decoding.

For each router setting (probability of choosing diffusion per segment), the
3B model is served in "mix" mode with a `RandomRouter`, and mini-swe-agent
solves the instances listed in `instances.json`. Instances are processed in
waves so Docker images can be deleted as soon as a wave is scored:

    pull the wave's images -> run every setting on them (parallel lanes, one
    model server per lane, all on one GPU) -> score with the SWE-bench
    harness -> delete the images this run pulled -> next wave.

Re-running resumes: instances already in a lane's preds.json are skipped by
mini-swe-agent, and scored waves are skipped here.

    CUDA_VISIBLE_DEVICES=3 uv run python -m scripts.swebench_router_experiment \
        --out runs/swebench_random_router --p-dlm 0.2 0.8
"""

import argparse
import concurrent.futures
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv/bin/python"
MINI = ROOT / ".venv/bin/mini-extra"
SWEBENCH_PYTHON = ROOT / ".venv-swebench/bin/python"
DATASET = "princeton-nlp/SWE-bench_Verified"
MIN_FREE_GB = 25


def log(msg: str) -> None:
    """Prints a timestamped progress line."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def image_name(instance_id: str) -> str:
    """Returns the prebuilt SWE-bench image used by mini and the harness."""
    key = instance_id.replace("__", "_1776_").lower()
    return f"swebench/sweb.eval.x86_64.{key}:latest"


def local_images() -> set[str]:
    """Returns the repo:tag names of all local Docker images."""
    out = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {line.removeprefix("docker.io/") for line in out.split()}


def free_gb(path: str = "/mnt/containers") -> float:
    """Returns free disk space in GB on the Docker volume."""
    return shutil.disk_usage(path).free / 1e9


def pull_images(ids: list[str]) -> None:
    """Pulls the wave's images, four at a time."""

    def pull(iid: str) -> None:
        subprocess.run(
            ["docker", "pull", "-q", image_name(iid)],
            capture_output=True,
            check=False,
        )

    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        list(pool.map(pull, ids))


class Lane:
    """One model server plus the mini-swe-agent process that uses it."""

    def __init__(
        self,
        setting: str,
        p_dlm: float,
        index: int,
        port: int,
        out: pathlib.Path,
    ) -> None:
        """Initializes the lane.

        Args:
            setting: Name of the router setting, e.g. "p0.2".
            p_dlm: Probability of routing a segment to diffusion.
            index: Lane index within the setting (also the router seed).
            port: Port of this lane's model server.
            out: Output directory of the whole experiment.
        """
        self.setting, self.p_dlm, self.index, self.port = (
            setting,
            p_dlm,
            index,
            port,
        )
        self.dir = out / setting / f"lane{index}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.server: subprocess.Popen | None = None

    def start_server(self, args: argparse.Namespace) -> None:
        """Starts the model server on the lane's port."""
        cmd = [
            str(PYTHON), "-m", "src.server",
            "--mode", "mix",
            "--router", "random",
            "--p-dlm", str(self.p_dlm),
            "--seed", str(self.index),
            "--ar-chunk", str(args.ar_chunk),
            "--max-new-tokens", str(args.max_tokens),
            "--log-file", str(self.dir / "requests.jsonl"),
            "--port", str(self.port),
            "--gpu-memory-fraction", str(args.gpu_memory_fraction),
        ]  # fmt: skip
        env = {
            **os.environ,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
        self.server = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=(self.dir / "server.log").open("a"),
            stderr=subprocess.STDOUT,
        )

    def wait_ready(self, timeout: float = 900) -> None:
        """Blocks until the server answers, or raises."""
        url = f"http://127.0.0.1:{self.port}/v1/models"
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.server is not None and self.server.poll() is not None:
                raise RuntimeError(f"server on port {self.port} exited")
            try:
                urllib.request.urlopen(url, timeout=5)
                return
            except OSError:
                time.sleep(5)
        raise TimeoutError(f"server on port {self.port} not ready")

    def run_agent(self, ids: list[str], args: argparse.Namespace) -> int:
        """Runs mini-swe-agent on `ids`; returns its exit code."""
        if not ids:
            return 0
        cmd = [
            str(MINI), "swebench",
            "--subset", "verified",
            "--split", "test",
            "--filter", "^(" + "|".join(ids) + ")$",
            "--output", str(self.dir),
            "--workers", "1",
            "-c", "swebench_backticks.yaml",
            "-c", str(ROOT / "configs/mini_nemotron.yaml"),
            "-c", "model.model_name=openai/nemotron-mix",
            "-c", f"model.model_kwargs.api_base=http://127.0.0.1:{self.port}/v1",
            "-c", f"model.model_kwargs.max_tokens={args.max_tokens}",
            "-c", f"agent.step_limit={args.step_limit}",
            "-c", "environment.pull_timeout=1800",
        ]  # fmt: skip
        with (self.dir / "agent.log").open("a") as f:
            return subprocess.run(
                cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT
            ).returncode

    def preds(self) -> dict:
        """Returns this lane's predictions keyed by instance id."""
        path = self.dir / "preds.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def stop(self) -> None:
        """Stops the model server."""
        if self.server is not None and self.server.poll() is None:
            self.server.terminate()
            self.server.wait(timeout=60)


def score(
    setting_dir: pathlib.Path, wave: int, preds: dict, ids: list[str]
) -> dict[str, bool]:
    """Scores one wave of one setting with the SWE-bench harness.

    Returns:
        Map from instance id to resolved, for instances with a prediction.
    """
    eval_dir = (setting_dir / "eval" / f"wave{wave}").resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)
    wave_preds = {i: preds[i] for i in ids if i in preds}
    nonempty = [i for i, p in wave_preds.items() if p.get("model_patch")]
    results = {i: False for i in wave_preds}
    if not nonempty:
        return results
    preds_path = eval_dir / "preds.json"
    preds_path.write_text(json.dumps(wave_preds))
    run_id = f"{setting_dir.name}_wave{wave}"
    cmd = [
        str(SWEBENCH_PYTHON), "-m", "swebench.harness.run_evaluation",
        "-d", DATASET, "-s", "test",
        "-p", str(preds_path),
        "-i", *nonempty,
        "-id", run_id,
        "--namespace", "swebench",
        "--max_workers", "4",
        "--cache_level", "instance",
    ]  # fmt: skip
    with (eval_dir / "harness.log").open("a") as f:
        subprocess.run(cmd, cwd=eval_dir, stdout=f, stderr=subprocess.STDOUT)
    # Per-instance reports; the harness can crash during cleanup before it
    # writes its run-level summary, so do not rely on that file.
    for report in eval_dir.glob(
        f"logs/run_evaluation/{run_id}/*/*/report.json"
    ):
        for iid, entry in json.loads(report.read_text()).items():
            if iid in results:
                results[iid] = bool(entry.get("resolved"))
    return results


def main() -> None:
    """Runs all waves for all router settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--p-dlm", type=float, nargs="+", default=[0.2, 0.8])
    parser.add_argument("--lanes", type=int, default=2, help="per setting")
    parser.add_argument("--wave-size", type=int, default=10)
    parser.add_argument("--step-limit", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--ar-chunk", type=int, default=32)
    parser.add_argument("--port-base", type=int, default=8040)
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=0.30,
        help="Per-server GPU memory cap (all servers share one GPU).",
    )
    parser.add_argument("--max-instances", type=int, default=None)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        sys.exit("Set CUDA_VISIBLE_DEVICES to the GPU to use.")
    out = args.out.resolve()
    ids = json.loads((out / "instances.json").read_text())[: args.max_instances]
    waves = [
        ids[i : i + args.wave_size] for i in range(0, len(ids), args.wave_size)
    ]
    preexisting = local_images()
    results_path = out / "results.json"
    results = (
        json.loads(results_path.read_text()) if results_path.exists() else {}
    )

    lanes = [
        Lane(f"p{p}", p, k, args.port_base + 10 * s + k, out)
        for s, p in enumerate(args.p_dlm)
        for k in range(args.lanes)
    ]
    for lane in lanes:
        lane.start_server(args)
    try:
        for lane in lanes:
            lane.wait_ready()
        log(
            f"{len(lanes)} servers ready; "
            f"{len(waves)} waves of {args.wave_size}"
        )

        for w, wave in enumerate(waves):
            settings = {lane.setting for lane in lanes}
            if all(f"wave{w}" in results.get(s, {}) for s in settings):
                log(f"wave {w}: already scored, skipping")
                continue
            while free_gb() < MIN_FREE_GB:
                log(f"only {free_gb():.0f} GB free; waiting")
                time.sleep(300)
            log(f"wave {w}: pulling {len(wave)} images")
            pull_images(wave)

            log(f"wave {w}: running agents")
            with concurrent.futures.ThreadPoolExecutor(len(lanes)) as pool:
                futures = []
                for lane in lanes:
                    chunk = wave[lane.index :: args.lanes]
                    futures.append(pool.submit(lane.run_agent, chunk, args))
                for f in futures:
                    f.result()

            for setting in sorted(settings):
                preds = {}
                for lane in lanes:
                    if lane.setting == setting:
                        preds.update(lane.preds())
                resolved = score(out / setting, w, preds, wave)
                results.setdefault(setting, {})[f"wave{w}"] = resolved
                n = sum(resolved.values())
                log(f"wave {w}: {setting} resolved {n}/{len(wave)}")
            results_path.write_text(json.dumps(results, indent=1))

            to_remove = [
                image_name(i) for i in wave if image_name(i) not in preexisting
            ]
            subprocess.run(
                ["docker", "rmi", "-f", *to_remove],
                capture_output=True,
                check=False,
            )
            log(
                f"wave {w}: removed {len(to_remove)} images, "
                f"{free_gb():.0f} GB free"
            )
    finally:
        for lane in lanes:
            lane.stop()
    log("done")


if __name__ == "__main__":
    main()
