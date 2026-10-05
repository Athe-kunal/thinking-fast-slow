"""Tool-calling SFT data from nvidia/Nemotron-Post-Training-Dataset-v1.

Reads the `tool_calling` split (reasoning-on conversations: <think> text,
tool calls, tool results, final answer), keeps conversations with at least one
tool call that render to at most `max_len` tokens, and splits them by a hash of
the uuid into train / held-out. Records are cached as JSONL (chat messages and
tools; tokenization happens at training time so span marking stays a config
choice).

    uv run python -m scripts.sft_data --shards 0 --max-len 4096
"""

import argparse
import hashlib
import json
import pathlib

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from src import spans

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / "runs" / "sft_cache"
DATASET = "nvidia/Nemotron-Post-Training-Dataset-v1"
REPO = "nvidia/Nemotron-Labs-Diffusion-3B"


def heldout(uuid: str, percent: int) -> bool:
    """Stable split: True for ~`percent`% of uuids."""
    return int(hashlib.sha1(uuid.encode()).hexdigest(), 16) % 100 < percent


def cache_path(shards: list[int], max_len: int) -> pathlib.Path:
    """Cache file for a shard selection."""
    tag = "-".join(map(str, shards))
    return CACHE / f"tool_s{tag}_len{max_len}.jsonl"


def build(
    shards: list[int], max_len: int, heldout_percent: int
) -> pathlib.Path:
    """Downloads, filters and caches the data; returns the cache path."""
    out = cache_path(shards, max_len)
    if out.exists():
        return out
    tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    with out.with_suffix(".tmp").open("w") as f:
        for shard in shards:
            path = hf_hub_download(
                DATASET,
                f"data/tool-{shard:05d}-of-00013.parquet",
                repo_type="dataset",
            )
            for rec in pq.read_table(path).to_pylist():
                try:
                    messages, tools = spans.to_chat(rec)
                    if not any(m.get("tool_calls") for m in messages):
                        raise ValueError("no tool call")
                    ex = spans.render(tok, messages, tools, diff_spans=True)
                    if len(ex.ids) > max_len or not any(ex.span):
                        raise ValueError("too long or no span")
                except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                    dropped += 1
                    continue
                split = (
                    "heldout"
                    if heldout(rec["uuid"], heldout_percent)
                    else "train"
                )
                f.write(json.dumps({"uuid": rec["uuid"], "split": split,
                                    "messages": messages, "tools": tools}) + "\n")  # fmt: skip
                kept += 1
    out.with_suffix(".tmp").rename(out)
    print(f"kept {kept}, dropped {dropped} -> {out}")
    return out


def load(path: pathlib.Path, split: str, limit: int | None, seed: int) -> list:
    """Records of one split, shuffled with `seed`, at most `limit`."""
    import random

    recs = [r for r in map(json.loads, path.open()) if r["split"] == split]
    random.Random(seed).shuffle(recs)
    return recs[:limit] if limit else recs


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, nargs="+", default=[0])
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--heldout-percent", type=int, default=5)
    args = parser.parse_args()
    build(args.shards, args.max_len, args.heldout_percent)


if __name__ == "__main__":
    main()
