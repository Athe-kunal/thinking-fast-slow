"""Trains the AR/diffusion router with GRPO on SGLang rollouts.

Only the small `RouterPolicy` is trained; the 3B backbone stays frozen and is
served by SGLang `RoutedDecoding` servers (one per GPU) with the router in
`sample` mode. Each iteration:

1. Sample prompts (GSM8K train / MBPP train+val / KodCode subset) and run G
   rollouts per prompt (backbone greedy, router sampled).
2. Read each rollout's trace: router decisions (features, action, logp) and
   per-call costs (tokens, forward passes).
3. Reward = correct - cost_weight * forward_passes / tokens.
4. Group-relative advantages (per prompt): r - group mean (`--adv-norm none`,
   default) or standardized; zero-variance groups are dropped.
5. PPO-clipped policy-gradient update of the router on the traced features
   (no backbone forward), plus an entropy bonus.
6. Push only `router.*` tensors to every server (update_weights_from_tensor).

Checkpoints are router-only (`router_iter*.pt`, ~3 MB). Run with the SGLang
environment, which provides the weight-update serializer:

    .venv-sglang/bin/python -m scripts.train_router_rl \
        --out runs/router_rl/run1 --gpus 2 3 --iterations 100
"""

import argparse
import base64
import concurrent.futures
import json
import os
import pathlib
import random
import secrets
import signal
import subprocess
import time
import typing
import urllib.request

import numpy as np
import torch

from scripts import rl_data
from src import router_policy

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL = "nvidia/Nemotron-Labs-Diffusion-3B"


def log(msg: str) -> None:
    """Prints a timestamped progress line."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def http_post(port: int, path: str, payload: dict, timeout: float) -> dict:
    """POSTs JSON to a local server and returns the JSON response."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


# ----------------------------------------------------------------- servers
class Server:
    """One SGLang RoutedDecoding server on one GPU."""

    def __init__(self, gpu: str, port: int, out: pathlib.Path) -> None:
        """Initializes the server handle (not started yet)."""
        self.gpu, self.port = gpu, port
        self.trace_dir = out / "trace" / f"gpu{gpu}"
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = out / f"server_gpu{gpu}.log"
        self.proc: subprocess.Popen | None = None

    def start(self, args: argparse.Namespace) -> None:
        """Launches the server in sample mode."""
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": self.gpu,
            "PORT": str(self.port),
            "POLICY_MODE": "sample",
            "AR_CHUNK": str(args.ar_chunk),
            "TRACE_DIR": str(self.trace_dir),
            "MAX_REQS": str(args.max_reqs),
            "MEM_FRAC": str(args.mem_frac),
        }
        # Own process group, so stop() also reaches SGLang's child processes
        # (scheduler, detokenizer).
        self.proc = subprocess.Popen(
            [str(ROOT / "scripts/launch_sglang.sh"), "routed"],
            env=env,
            stdout=self.log_path.open("a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def wait_ready(self, timeout: float = 900) -> None:
        """Blocks until /health_generate answers, or raises."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(f"server on GPU {self.gpu} exited")
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/health_generate", timeout=10
                )
                return
            except OSError:
                time.sleep(5)
        raise TimeoutError(f"server on GPU {self.gpu} not ready")

    def stop(self) -> None:
        """Stops the whole server process group; never raises.

        SIGTERM first, SIGKILL if the group is still alive after 60 s.
        """
        if self.proc is None:
            return
        try:
            pgid = os.getpgid(self.proc.pid)
        except ProcessLookupError:
            return
        for sig, wait in ((signal.SIGTERM, 60), (signal.SIGKILL, 10)):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                return
            try:
                self.proc.wait(timeout=wait)
                return
            except subprocess.TimeoutExpired:
                log(f"server on GPU {self.gpu} still alive after {sig.name}")


def push_router(
    policy: router_policy.RouterPolicy, servers: list[Server]
) -> None:
    """Sends only the router's tensors to every server."""
    # Only in .venv-sglang, which runs this script.
    from sglang.srt.utils import (  # noqa: PLC0415  # ty: ignore[unresolved-import]
        MultiprocessingSerializer,
    )

    for s in servers:
        # Fresh shared-memory tensors per server: a receiver consumes the
        # shared-memory files it opens, so one payload cannot serve two.
        named = [
            (f"router.{k}", v.detach().cpu().clone().contiguous())
            for k, v in policy.state_dict().items()
        ]
        payload = {
            "serialized_named_tensors": [
                MultiprocessingSerializer.serialize(named, output_str=True)
            ],
            "flush_cache": False,
        }
        out = http_post(s.port, "update_weights_from_tensor", payload, 120)
        del named
        if not out.get("success"):
            raise RuntimeError(f"router push to GPU {s.gpu} failed: {out}")


# ----------------------------------------------------------------- rollouts
def read_trace(path: pathlib.Path, completion_tokens: int) -> dict:
    """Parses and deletes one rollout's trace file.

    Args:
        path: The rollout's trace file.
        completion_tokens: Tokens the server actually returned. The trace
            counts whole committed blocks, but the scheduler truncates the
            last block at max_new_tokens, so the server count is the true
            denominator for the cost; forward passes are counted as spent.
    """
    decisions, traced_tokens, nfe, dlm_tokens = [], 0, 1, 0  # nfe: prefill
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["type"] == "decision":
            feats = np.frombuffer(base64.b64decode(r["features"]), np.float32)
            decisions.append((feats.copy(), r["action"], r["logp"]))
        else:
            traced_tokens += r["tokens"]
            nfe += r["nfe"]
            dlm_tokens += r["tokens"] if r["action"] == 1 else 0
    path.unlink()
    tokens = min(traced_tokens, completion_tokens)
    return {
        "decisions": decisions,
        "tokens": tokens,
        "nfe": nfe,
        "dlm_tokens": min(dlm_tokens, tokens),
    }


def run_rollout(server: Server, rid: str, input_ids: list, max_tokens: int):
    """One generation; returns (text, trace) or None on failure."""
    payload = {
        "input_ids": input_ids,
        "rid": rid,
        "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0},
    }
    trace_path = server.trace_dir / f"{rid}.jsonl"
    trace_path.unlink(missing_ok=True)  # never append to a stale trace
    try:
        out = http_post(server.port, "generate", payload, 1800)
        completion = int(out["meta_info"]["completion_tokens"])
        return out["text"], read_trace(trace_path, completion)
    except (OSError, KeyError, json.JSONDecodeError) as e:
        log(f"rollout {rid} failed: {e}")
        return None


# ----------------------------------------------------------------- training
def group_advantages(
    rewards: list[float], groups: list[int], norm: str
) -> np.ndarray:
    """Group-relative advantages; zero-variance groups get 0.

    Args:
        rewards: Reward per rollout.
        groups: Prompt-group id per rollout.
        norm: "none" -> r - group mean (keeps reward scale, so small cost
            differences stay small); "std" -> (r - mean) / std (standard
            GRPO; collapsed to always-diffusion in run1, see RESEARCH E11).
    """
    rewards_np = np.asarray(rewards, dtype=np.float64)
    adv = np.zeros_like(rewards_np)
    for g in set(groups):
        idx = [i for i, x in enumerate(groups) if x == g]
        r = rewards_np[idx]
        if len(idx) > 1 and r.std() > 1e-6:
            adv[idx] = r - r.mean()
            if norm == "std":
                adv[idx] /= r.std() + 1e-6
    return adv


def group_stats(
    correct: list[bool], rewards: list[float], groups: list[int]
) -> dict:
    """Diagnostics: how often routing changes correctness within a group."""
    by_group: dict[int, list[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    mixed = [len({correct[i] for i in idx}) > 1 for idx in by_group.values()]
    spread = [
        float(np.std([rewards[i] for i in idx])) for idx in by_group.values()
    ]
    return {
        "mixed_correct_group_frac": float(np.mean(mixed)) if mixed else 0.0,
        "group_reward_std": float(np.mean(spread)) if spread else 0.0,
    }


def ppo_update(
    policy: router_policy.RouterPolicy,
    optim: torch.optim.Optimizer,
    feats: torch.Tensor,
    actions: torch.Tensor,
    logp_old: torch.Tensor,
    adv: torch.Tensor,
    args: argparse.Namespace,
) -> dict:
    """PPO-clipped policy-gradient steps on all decisions of the batch."""
    stats = {}
    for epoch in range(args.epochs):
        logits = policy(feats)
        logp = router_policy.action_log_prob(logits, actions)
        ratio = torch.exp(logp - logp_old)
        clipped = torch.clamp(ratio, 1 - args.clip, 1 + args.clip)
        pg = -torch.min(ratio * adv, clipped * adv).mean()
        entropy = router_policy.bernoulli_entropy(logits).mean()
        loss = pg - args.entropy_coef * entropy
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optim.step()
        if epoch == 0:
            stats["logp_mismatch"] = float(
                (logp - logp_old).abs().max().detach()
            )
            stats["policy_entropy"] = float(entropy.detach())
            stats["p_dlm_mean"] = float(torch.sigmoid(logits).mean().detach())
        stats["loss"] = float(loss.detach())
        stats["clip_frac"] = float(
            ((ratio - 1).abs() > args.clip).float().mean()
        )
    return stats


def iteration(
    it: int,
    args: argparse.Namespace,
    items: dict,
    prompt_ids: dict,
    servers: list[Server],
    policy: router_policy.RouterPolicy,
    optim: torch.optim.Optimizer,
    rng: random.Random,
) -> dict:
    """Runs one rollout + update iteration; returns metrics."""
    t0 = time.time()
    batch = rl_data.sample_batch(items, args.weights, args.prompts, rng)
    jobs = [
        (g, item, f"{args.run_tag}-it{it}-p{g}-r{k}")
        for g, item in enumerate(batch)
        for k in range(args.group_size)
    ]
    with concurrent.futures.ThreadPoolExecutor(
        args.max_reqs * len(servers)
    ) as pool:
        futures = [
            pool.submit(
                run_rollout,
                servers[j % len(servers)],
                rid,
                prompt_ids[item["id"]],
                args.max_tokens,
            )
            for j, (_, item, rid) in enumerate(jobs)
        ]
        results = [f.result() for f in futures]
    t_rollout = time.time() - t0

    kept = [
        (j, r) for (j, r) in zip(jobs, results, strict=True) if r is not None
    ]
    correct = rl_data.score_batch(
        [j[1] for j, _ in kept], [r[0] for _, r in kept]
    )
    costs = [r[1]["nfe"] / max(1, r[1]["tokens"]) for _, r in kept]
    rewards = [
        float(c) - args.cost_weight * cost
        for c, cost in zip(correct, costs, strict=True)
    ]
    groups = [j[0] for j, _ in kept]
    adv = group_advantages(rewards, groups, args.adv_norm)

    rows, acts, logps, advs = [], [], [], []
    for (_, r), a in zip(kept, adv, strict=True):
        for feats, action, logp in r[1]["decisions"]:
            rows.append(feats)
            acts.append(action)
            logps.append(logp)
            advs.append(a)
    stats = {}
    if rows and np.abs(advs).sum() > 0:
        stats = ppo_update(
            policy,
            optim,
            torch.from_numpy(np.stack(rows)),
            torch.tensor(acts),
            torch.tensor(logps, dtype=torch.float32),
            torch.tensor(advs, dtype=torch.float32),
            args,
        )
        push_router(policy, servers)

    tokens = sum(r[1]["tokens"] for _, r in kept)
    by_source = {}
    for src in args.weights:
        idx = [i for i, (j, _) in enumerate(kept) if j[1]["source"] == src]
        if idx:
            by_source[f"acc_{src}"] = float(np.mean([correct[i] for i in idx]))
    return {
        "iteration": it,
        "rollouts": len(kept),
        "failed": len(jobs) - len(kept),
        "accuracy": float(np.mean(correct)) if correct else 0.0,
        **by_source,
        "reward": float(np.mean(rewards)) if rewards else 0.0,
        "cost": float(np.mean(costs)) if costs else 0.0,
        "tokens_per_forward": tokens
        / max(1, sum(r[1]["nfe"] for _, r in kept)),
        "dlm_token_share": sum(r[1]["dlm_tokens"] for _, r in kept)
        / max(1, tokens),
        "decisions": len(rows),
        "adv_nonzero_frac": float(np.mean(np.abs(adv) > 0))
        if len(adv)
        else 0.0,
        **group_stats(correct, rewards, groups),
        "seconds_rollout": t_rollout,
        "seconds_total": time.time() - t0,
        **stats,
    }


# ----------------------------------------------------------------- main
def parse_weights(spec: str) -> dict[str, float]:
    """Parses "gsm8k:0.35,mbpp:0.15,kodcode:0.5"."""
    return {k: float(v) for k, v in (p.split(":") for p in spec.split(","))}


def main() -> None:
    """Launches servers, then alternates rollouts and router updates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--gpus", nargs="+", default=["2", "3"])
    parser.add_argument("--port-base", type=int, default=30031)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--prompts", type=int, default=32, help="per iteration")
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--ar-chunk", type=int, default=8)
    parser.add_argument("--cost-weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.05)
    parser.add_argument("--adv-norm", choices=("none", "std"), default="none")
    parser.add_argument(
        "--weights",
        type=parse_weights,
        default="gsm8k:0.35,mbpp:0.15,kodcode:0.5",
    )
    parser.add_argument("--kodcode-limit", type=int, default=3000)
    parser.add_argument("--max-reqs", type=int, default=64, help="per server")
    parser.add_argument("--mem-frac", type=float, default=0.75)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy("file_system")
    # Per-process tag in every request id: a resumed run replays iteration
    # numbers, and must never read trace files left by the crashed attempt.
    args.run_tag = secrets.token_hex(4)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps(vars(args), default=str))

    from transformers import (  # noqa: PLC0415
        AutoTokenizer,
        PreTrainedTokenizerBase,
    )

    items = rl_data.load_items(args.kodcode_limit, args.seed)
    tok = typing.cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True),
    )
    prompt_ids = {
        it["id"]: tok(
            tok.apply_chat_template(
                [{"role": "user", "content": it["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
        ).input_ids
        for src in items.values()
        for it in src
    }
    log(f"items: { {k: len(v) for k, v in items.items()} }")

    policy = router_policy.RouterPolicy(3072)
    optim = torch.optim.Adam(policy.parameters(), lr=args.lr)
    rng = random.Random(args.seed)
    start = 0
    state_path = args.out / "trainer_state.pt"
    if state_path.exists():
        state = torch.load(state_path, weights_only=False)
        policy.load_state_dict(state["policy"])
        optim.load_state_dict(state["optim"])
        rng.setstate(state["rng"])
        start = state["iteration"] + 1
        log(f"resuming at iteration {start}")

    servers = [
        Server(g, args.port_base + i, args.out) for i, g in enumerate(args.gpus)
    ]
    for s in servers:
        s.start(args)
    try:
        for s in servers:
            s.wait_ready()
        push_router(policy, servers)
        log(f"{len(servers)} servers ready")
        metrics_path = args.out / "metrics.jsonl"
        for it in range(start, args.iterations):
            m = iteration(
                it, args, items, prompt_ids, servers, policy, optim, rng
            )
            with metrics_path.open("a") as f:
                f.write(json.dumps(m) + "\n")
            log(
                f"it {it}: acc {m['accuracy']:.3f} reward {m['reward']:.3f} "
                f"dlm {m['dlm_token_share']:.2f} tok/fwd "
                f"{m['tokens_per_forward']:.2f} p_dlm "
                f"{m.get('p_dlm_mean', float('nan')):.2f} "
                f"({m['seconds_total']:.0f}s)"
            )
            if (it + 1) % args.save_every == 0 or it + 1 == args.iterations:
                policy.save(str(args.out / f"router_iter{it + 1:04d}.pt"))
                torch.save(
                    {
                        "policy": policy.state_dict(),
                        "optim": optim.state_dict(),
                        "rng": rng.getstate(),
                        "iteration": it,
                    },
                    state_path,
                )
    finally:
        for s in servers:
            s.stop()


if __name__ == "__main__":
    main()
