"""Checks SGLang RoutedDecoding against the HF reference decoders.

Sends prompts to a running routed server (`scripts/launch_sglang.sh routed`)
and compares token-for-token with HF: POLICY_MODE=fixed_ar vs `ar_generate`,
POLICY_MODE=fixed_dlm vs `generate(block 32, threshold 0.9)`.

    CUDA_VISIBLE_DEVICES=3 uv run python -m scripts.check_routed_parity \
        --mode fixed_ar --port 30021 -n 10
"""

import argparse
import json
import urllib.request

from scripts.eval_router_benchmarks import load_benchmark
from src import engine as engine_lib


def sglang_generate(port: int, input_ids: list[int], rid: str, n: int) -> list:
    """Greedy generation on the SGLang server; returns output token ids."""
    body = json.dumps(
        {
            "input_ids": input_ids,
            "rid": rid,
            "sampling_params": {"max_new_tokens": n, "temperature": 0},
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)["output_ids"]


def main() -> None:
    """Runs the comparison and prints per-prompt agreement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("fixed_ar", "fixed_dlm"), required=True
    )
    parser.add_argument("--port", type=int, default=30021)
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    engine = engine_lib.NemotronEngine()
    model, eos = engine.model, engine.tokenizer.eos_token_id
    items = (
        load_benchmark("gsm8k")[: args.n // 2]
        + load_benchmark("humaneval")[: args.n - args.n // 2]
    )
    identical = 0
    for i, item in enumerate(items):
        ids = engine.encode_chat(item["prompt"])
        if args.mode == "fixed_ar":
            out, _ = model.ar_generate(
                ids, max_new_tokens=args.max_tokens, eos_token_id=eos
            )
        else:
            out, _ = model.generate(
                ids,
                max_new_tokens=args.max_tokens,
                block_length=32,
                threshold=0.9,
                eos_token_id=eos,
            )
        ref = out[0, ids.shape[1] :].tolist()
        if eos in ref:
            ref = ref[: ref.index(eos) + 1]
        got = sglang_generate(
            args.port,
            ids[0].tolist(),
            f"parity-{args.mode}-{i}",
            args.max_tokens,
        )
        common = next(
            (
                k
                for k, (a, b) in enumerate(zip(ref, got, strict=False))
                if a != b
            ),
            min(len(ref), len(got)),
        )
        same = ref == got
        identical += same
        print(
            f"[{i}] identical={same} common prefix {common} "
            f"(hf {len(ref)} tok, sglang {len(got)} tok)",
            flush=True,
        )
    print(f"{args.mode}: identical {identical}/{len(items)}")


if __name__ == "__main__":
    main()
